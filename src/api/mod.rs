use axum::extract::State;
use axum_web::object::PackObject;
use isolang::Language;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

// gpt-35-turbo, 4096
static SUMMARIZE_SECTION_TOKENS: usize = 2400;
pub(crate) static SUMMARIZE_HIGH_TOKENS: usize = 3000;

// text-embedding-ada-002, 8191
// https://community.openai.com/t/embedding-text-length-vs-accuracy/96564
static EMBEDDING_SECTION_TOKENS: usize = 600;
static EMBEDDING_HIGH_TOKENS: usize = 800;
// https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/switching-endpoints#azure-openai-embeddings-multiple-input-support
static EMBEDDING_MAX_ARRAY: usize = 16;
static EMBEDDING_MAX_TOKENS: usize = 7000;

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

pub(crate) struct TEParams {
    pub gid: xid::Id,
    pub cid: xid::Id,
    pub language: Language,
    pub version: i16,
    pub content: TEContentList,
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

    pub fn to_translating_list(&self) -> Vec<Vec<String>> {
        let mut res: Vec<Vec<String>> = Vec::with_capacity(self.content.len());
        let mut i = 0u32;
        for c in &self.content {
            i += 1;
            let mut l: Vec<String> = Vec::with_capacity(c.texts.len() + 1);
            l.push(format!("{}:", i));
            l.extend_from_slice(&c.texts);
            res.push(l);
        }
        res
    }

    pub fn replace_texts(&self, input: &[Vec<String>]) -> TEContentList {
        let len = self.content.len();
        let mut res: TEContentList = Vec::with_capacity(len);
        let mut iter = input.iter();
        let (mut o, mut v) = Self::extract_order(iter.next());
        for i in 0..len {
            let mut te = TEContent {
                id: self.content[i].id.clone(),
                texts: Vec::new(),
            };

            if o <= i + 1 {
                te.texts.extend_from_slice(v);
                (o, v) = Self::extract_order(iter.next());
            }
            res.push(te);
        }

        res
    }

    // ["1:", "text1", ...] => (1, ["text1", ...])
    // ["text1", ...] => (0, ["text1", ...])
    // [] => (0, [])
    fn extract_order(v: Option<&Vec<String>>) -> (usize, &[String]) {
        match v {
            Some(v) => {
                if v.is_empty() {
                    return (0, v);
                }
                // the ':' maybe translated by AI
                let o = if v[0].ends_with(&COLONS) {
                    let mut s = v[0].clone();
                    s.pop();
                    s.parse::<usize>().unwrap_or(0)
                } else {
                    0
                };

                if o > 0 {
                    (o, &v[1..])
                } else {
                    (0, v)
                }
            }
            None => (0, &[]),
        }
    }
}

// https://en.wikipedia.org/wiki/Colon_(punctuation)
const COLONS: [char; 8] = [
    '\u{003A}', '\u{02F8}', '\u{05C3}', '\u{2236}', '\u{A789}', '\u{FE13}', '\u{FF1A}', '\u{FE55}',
];

pub trait TESegmenter {
    fn detect_lang_string(&self) -> String;
    fn segment(&self, model: &openai::AIModel, tokens_len: fn(&str) -> usize) -> Vec<TEUnit>;
    fn segment_for_summarizing(&self, tokens_len: fn(&str) -> usize) -> Vec<String>;
    fn segment_for_embedding(&self, tokens_len: fn(&str) -> usize) -> Vec<Vec<TEUnit>>;
}

impl TESegmenter for TEContentList {
    fn detect_lang_string(&self) -> String {
        let mut detect_language = String::with_capacity(5000);

        for c in self {
            if detect_language.len() > 4096 {
                break;
            }
            detect_language.push_str(c.to_string('\n').as_str());
            detect_language.push('\n');
        }

        detect_language
    }

    fn segment(&self, model: &openai::AIModel, tokens_len: fn(&str) -> usize) -> Vec<TEUnit> {
        let mut list: Vec<TEUnit> = Vec::new();
        let mut unit: TEUnit = TEUnit {
            tokens: 0,
            content: Vec::new(),
        };
        let (st, ht) = model.translating_segment_tokens();

        for c in self {
            if c.texts.is_empty() {
                if c.id == SECTION_SEPARATOR {
                    // segment embedding content by section separator
                    if unit.tokens >= st {
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

            if unit.tokens + ctl > ht {
                if !unit.content.is_empty() {
                    list.push(unit);
                }
                unit = TEUnit {
                    tokens: ctl,
                    content: vec![c.clone()],
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

            if tokens + ctl > SUMMARIZE_HIGH_TOKENS {
                if !unit.is_empty() {
                    list.push(unit.join("\n"));
                }

                tokens = ctl;
                unit.truncate(0);
                unit.push(strs);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teunit_to_translating() {
        let unit = TEUnit {
            tokens: 0,
            content: vec![
                TEContent {
                    id: "abc".to_string(),
                    texts: vec!["text1".to_string(), "text2".to_string()],
                },
                TEContent {
                    id: "efg".to_string(),
                    texts: vec!["text3".to_string(), "text4".to_string()],
                },
            ],
        };

        let rt = unit.to_translating_list();
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            vec!["1:".to_string(), "text1".to_string(), "text2".to_string()]
        );
        assert_eq!(
            rt[1],
            vec!["2:".to_string(), "text3".to_string(), "text4".to_string()]
        );

        let rt = unit.replace_texts(&[
            vec!["1:".to_string(), "text_1".to_string(), "text_2".to_string()],
            vec!["2:".to_string(), "text_3".to_string(), "text_4".to_string()],
        ]);
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            TEContent {
                id: "abc".to_string(),
                texts: vec!["text_1".to_string(), "text_2".to_string()],
            },
        );
        assert_eq!(
            rt[1],
            TEContent {
                id: "efg".to_string(),
                texts: vec!["text_3".to_string(), "text_4".to_string()],
            },
        );

        let rt = unit.replace_texts(&[
            vec!["text_1".to_string(), "text_2".to_string()],
            vec!["2:".to_string(), "text_3".to_string(), "text_4".to_string()],
        ]);
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            TEContent {
                id: "abc".to_string(),
                texts: vec!["text_1".to_string(), "text_2".to_string()],
            },
        );
        assert_eq!(
            rt[1],
            TEContent {
                id: "efg".to_string(),
                texts: vec!["text_3".to_string(), "text_4".to_string()],
            },
        );

        let rt = unit.replace_texts(&[
            vec!["1:".to_string(), "text_1".to_string(), "text_2".to_string()],
            vec!["text_3".to_string(), "text_4".to_string()],
        ]);
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            TEContent {
                id: "abc".to_string(),
                texts: vec!["text_1".to_string(), "text_2".to_string()],
            },
        );
        assert_eq!(
            rt[1],
            TEContent {
                id: "efg".to_string(),
                texts: vec!["text_3".to_string(), "text_4".to_string()],
            },
        );

        let rt = unit.replace_texts(&[vec![
            "1:".to_string(),
            "text_1".to_string(),
            "text_2".to_string(),
        ]]);
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            TEContent {
                id: "abc".to_string(),
                texts: vec!["text_1".to_string(), "text_2".to_string()],
            },
        );
        assert_eq!(
            rt[1],
            TEContent {
                id: "efg".to_string(),
                texts: vec![],
            },
        );

        let rt = unit.replace_texts(&[vec![
            "2:".to_string(),
            "text_1".to_string(),
            "text_2".to_string(),
        ]]);
        assert_eq!(rt.len(), 2);
        assert_eq!(
            rt[0],
            TEContent {
                id: "abc".to_string(),
                texts: vec![],
            },
        );
        assert_eq!(
            rt[1],
            TEContent {
                id: "efg".to_string(),
                texts: vec!["text_1".to_string(), "text_2".to_string()],
            },
        );
    }
}
