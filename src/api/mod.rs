use axum::extract::State;
use isolang::Language;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use axum_web::object::PackObject;

use crate::db::{self, qdrant};
use crate::lang::LanguageDetector;
use crate::openai;

pub mod embedding;
pub mod summarizing;
pub mod translating;

pub const APP_NAME: &str = env!("CARGO_PKG_NAME");
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

// dashes (------) is a horizontal rule, work as a top section separator
static SECTION_SEPARATOR: &str = "------";

// gpt-35-16k
static TRANSLATE_SECTION_TOKENS: usize = 6000;
static TRANSLATE_HIGH_TOKENS: usize = 8000;

// gpt-35-16k
static SUMMARIZE_SECTION_TOKENS: usize = 14000;
static SUMMARIZE_HIGH_TOKENS: usize = 15000;

// https://community.openai.com/t/embedding-text-length-vs-accuracy/96564
static EMBEDDING_SECTION_TOKENS: usize = 600;
static EMBEDDING_HIGH_TOKENS: usize = 800;
// https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/switching-endpoints#azure-openai-embeddings-multiple-input-support
static EMBEDDING_MAX_ARRAY: usize = 16;
static EMBEDDING_MAX_TOKENS: usize = 7600;

#[derive(Clone)]
pub struct AppState {
    pub ld: Arc<LanguageDetector>,
    pub ai: Arc<openai::OpenAI>,
    pub scylla: Arc<db::scylladb::ScyllaDB>,
    pub qdrant: Arc<qdrant::Qdrant>,
    pub translating: Arc<String>, // keep the number of concurrent translating tasks
    pub embedding: Arc<String>,   // keep the number of concurrent embedding tasks
}

#[derive(Serialize, Deserialize)]
pub struct AppVersion {
    pub name: String,
    pub version: String,
}

#[derive(Serialize, Deserialize)]
pub struct AppInfo {
    pub tokio_translating_tasks: i64, // the number of concurrent translating tasks
    pub tokio_embedding_tasks: i64,   // the number of concurrent embedding tasks

    // https://docs.rs/scylla/latest/scylla/struct.Metrics.html
    pub scylla_latency_avg_ms: u64,
    pub scylla_latency_p99_ms: u64,
    pub scylla_latency_p90_ms: u64,
    pub scylla_errors_num: u64,
    pub scylla_queries_num: u64,
    pub scylla_errors_iter_num: u64,
    pub scylla_queries_iter_num: u64,
    pub scylla_retries_num: u64,
}

pub async fn version(to: PackObject<()>, State(_): State<Arc<AppState>>) -> PackObject<AppVersion> {
    to.with(AppVersion {
        name: APP_NAME.to_string(),
        version: APP_VERSION.to_string(),
    })
}

pub async fn healthz(to: PackObject<()>, State(app): State<Arc<AppState>>) -> PackObject<AppInfo> {
    let m = app.scylla.metrics();
    to.with(AppInfo {
        tokio_translating_tasks: Arc::strong_count(&app.translating) as i64 - 1,
        tokio_embedding_tasks: Arc::strong_count(&app.embedding) as i64 - 1,
        scylla_latency_avg_ms: m.get_latency_avg_ms().unwrap_or(0),
        scylla_latency_p99_ms: m.get_latency_percentile_ms(99.0f64).unwrap_or(0),
        scylla_latency_p90_ms: m.get_latency_percentile_ms(90.0f64).unwrap_or(0),
        scylla_errors_num: m.get_errors_num(),
        scylla_queries_num: m.get_queries_num(),
        scylla_errors_iter_num: m.get_errors_iter_num(),
        scylla_queries_iter_num: m.get_queries_iter_num(),
        scylla_retries_num: m.get_retries_num(),
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TEOutput {
    pub cid: PackObject<xid::Id>,                // document id
    pub detected_language: PackObject<Language>, // the origin language detected.
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TEContent {
    pub id: String, // node id in the document
    pub texts: Vec<String>,
}

pub type TEContentList = Vec<TEContent>;

impl TEContent {
    pub fn to_translating_string(&self) -> String {
        serde_json::to_string(&self.texts).expect("TEContent::to_translating_string error")
    }

    pub fn to_string(&self, sep: char) -> String {
        let mut tes = String::new();
        for t in &self.texts {
            for s in t.split_whitespace() {
                if !s.is_empty() {
                    if !tes.is_empty() {
                        tes.push(sep);
                    }
                    tes.push_str(s);
                }
            }
        }
        tes
    }
}

#[derive(Serialize)]
pub struct TEUnit {
    pub tokens: usize,
    pub content: TEContentList,
}

impl TEUnit {
    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::with_capacity(self.content.len());
        for c in &self.content {
            ids.push(c.id.clone());
        }
        ids
    }

    pub fn to_translating_list(&self) -> Vec<&Vec<String>> {
        let mut res: Vec<&Vec<String>> = Vec::with_capacity(self.content.len());
        for c in &self.content {
            res.push(&c.texts);
        }
        res
    }

    pub fn to_embedding_string(&self) -> String {
        let mut tes: String = String::new();
        for c in &self.content {
            tes.push_str(c.to_string(' ').trim());
            if tes.ends_with('.') {
                tes.push(' ');
            } else {
                tes.push_str(". ");
            }
        }
        tes.trim().to_string()
    }

    pub fn replace_texts(&self, input: &Vec<Vec<String>>) -> TEContentList {
        let mut res: TEContentList = Vec::with_capacity(input.len());
        if input.len() != self.content.len() {
            return res;
        }

        for (i, v) in input.iter().enumerate() {
            res.push(TEContent {
                id: self.content[i].id.clone(),
                texts: v.to_owned(),
            });
        }
        res
    }
}

pub trait TESegmenter {
    fn detect_lang_string(&self) -> String;
    fn segment(&self, tokens_len: fn(&str) -> usize) -> Vec<TEUnit>;
    fn segment_for_summarizing(&self, tokens_len: fn(&str) -> usize) -> Vec<String>;
    fn segment_for_embedding(&self, tokens_len: fn(&str) -> usize) -> Vec<Vec<TEUnit>>;
}

impl TESegmenter for TEContentList {
    fn detect_lang_string(&self) -> String {
        let mut detect_lang = String::new();

        for c in self {
            if detect_lang.len() < 1024 {
                detect_lang.push_str(c.to_string('\n').as_str());
                detect_lang.push('\n');
            }
        }
        detect_lang
    }

    fn segment(&self, tokens_len: fn(&str) -> usize) -> Vec<TEUnit> {
        let mut list: Vec<TEUnit> = Vec::new();
        let mut unit: TEUnit = TEUnit {
            tokens: 0,
            content: Vec::new(),
        };

        for c in self {
            if c.texts.is_empty() {
                if c.id == SECTION_SEPARATOR {
                    // segment embedding content by section separator
                    if unit.tokens >= TRANSLATE_SECTION_TOKENS {
                        list.push(unit);
                        unit = TEUnit {
                            tokens: 0,
                            content: Vec::new(),
                        };
                    }
                }

                continue;
            }

            let ctl = tokens_len(&c.to_translating_string());

            if unit.tokens + ctl >= TRANSLATE_HIGH_TOKENS {
                unit.tokens += ctl;
                unit.content.push(c.clone());
                list.push(unit);
                unit = TEUnit {
                    tokens: 0,
                    content: Vec::new(),
                };
            } else {
                unit.tokens += ctl;
                unit.content.push(c.clone());
            }
        }

        if unit.tokens > 0 {
            list.push(unit);
        }

        list
    }

    fn segment_for_summarizing(&self, tokens_len: fn(&str) -> usize) -> Vec<String> {
        let mut list: Vec<String> = Vec::new();
        let mut unit: Vec<String> = Vec::new();
        let mut tokens = 0usize;

        for c in self {
            if c.texts.is_empty() {
                if c.id == SECTION_SEPARATOR && tokens >= SUMMARIZE_SECTION_TOKENS {
                    list.push(unit.join("\n"));
                    tokens = 0;
                    unit.truncate(0);
                }

                continue;
            }

            let strs = c.to_string(' ');
            let ctl = tokens_len(&strs);

            if tokens + ctl >= SUMMARIZE_HIGH_TOKENS {
                tokens += ctl;
                unit.push(strs);
                list.push(unit.join("\n"));
                tokens = 0;
                unit.truncate(0);
            } else {
                tokens += ctl;
                unit.push(strs);
            }
        }

        if tokens > 0 {
            list.push(unit.join("\n"));
        }

        list
    }

    fn segment_for_embedding(&self, tokens_len: fn(&str) -> usize) -> Vec<Vec<TEUnit>> {
        let mut list: Vec<Vec<TEUnit>> = Vec::new();
        let mut group: Vec<TEUnit> = Vec::new();
        let mut group_tokens: usize = 0;
        let mut unit: TEUnit = TEUnit {
            tokens: 0,
            content: Vec::new(),
        };

        for c in self {
            if c.texts.is_empty() {
                if c.id == SECTION_SEPARATOR {
                    // segment embedding content by section separator
                    if unit.tokens >= EMBEDDING_SECTION_TOKENS {
                        group_tokens += unit.tokens;
                        group.push(unit);
                        unit = TEUnit {
                            tokens: 0,
                            content: Vec::new(),
                        };
                    }

                    if group_tokens >= EMBEDDING_MAX_TOKENS || group.len() >= EMBEDDING_MAX_ARRAY {
                        list.push(group);
                        group_tokens = 0;
                        group = Vec::new();
                    }
                }

                continue;
            }

            let ctl = tokens_len(&c.to_string(' '));

            if unit.tokens + ctl >= EMBEDDING_HIGH_TOKENS {
                unit.tokens += ctl;
                unit.content.push(c.clone());
                group_tokens += unit.tokens;
                group.push(unit);
                unit = TEUnit {
                    tokens: 0,
                    content: Vec::new(),
                };

                if group_tokens >= EMBEDDING_MAX_TOKENS || group.len() >= EMBEDDING_MAX_ARRAY {
                    list.push(group);
                    group_tokens = 0;
                    group = Vec::new();
                }
            } else {
                unit.tokens += ctl;
                unit.content.push(c.clone());
            }
        }

        if unit.tokens > 0 {
            group_tokens += unit.tokens;
            group.push(unit);
        }

        if group_tokens > 0 {
            list.push(group)
        }

        list
    }
}
