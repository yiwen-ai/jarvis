use serde::{Deserialize, Serialize};

// dashes (------) is a horizontal rule, work as a top section separator
static SECTION_SEPARATOR: &str = "------";

static TRANSLATE_SECTION_TOKENS: usize = 1800;
static TRANSLATE_HIGH_TOKENS: usize = 2000;

// https://community.openai.com/t/embedding-text-length-vs-accuracy/96564
static EMBEDDING_SECTION_TOKENS: usize = 400;
static EMBEDDING_HIGH_TOKENS: usize = 600;

pub trait Validator {
    fn validate(&self) -> Option<String>;
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

#[derive(Serialize, Deserialize)]
pub struct TEInput {
    pub did: String,  // document id
    pub lang: String, // the target language translate to
    pub version: u16, // should <= i16::MAX

    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub content: TEContentList,
}

impl Validator for TEInput {
    fn validate(&self) -> Option<String> {
        if self.did.is_empty() {
            return Some("invalid document id".to_string());
        }

        if self.lang.is_empty() {
            return Some("invalid target language".to_string());
        }

        if self.version == 0 || self.version > i16::MAX as u16 {
            return Some(format!("invalid version {}", self.version));
        }

        None
    }
}

#[derive(Serialize, Deserialize)]
pub struct TEOutput {
    pub did: String,  // document id
    pub lang: String, // the origin language detected.
    pub used_tokens: usize,
    pub content: TEContentList,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TEContent {
    pub id: String, // node id in the document
    pub texts: Vec<String>,
}

pub type TEContentList = Vec<TEContent>;

impl TEContent {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("TEContent::to_json_string error")
    }

    pub fn to_embedding_string(&self) -> String {
        let mut tes = String::new();
        for t in &self.texts {
            for s in t.split_whitespace() {
                if !s.is_empty() {
                    if !tes.is_empty() {
                        tes.push(' ');
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
    index: usize,
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

    pub fn content_to_json_string(&self) -> String {
        serde_json::to_string(&self.content).expect("TEUnit::content_to_json_string error")
    }

    pub fn content_to_embedding_string(&self) -> String {
        let mut tes: Vec<String> = Vec::new();
        for c in &self.content {
            tes.push(c.to_embedding_string());
        }
        tes.join("; ")
    }
}

pub trait TESegmenter {
    fn detect_lang_string(&self) -> String;
    fn segment(&self, tokens_len: fn(&str) -> usize, for_embedding: bool) -> Vec<TEUnit>;
}

impl TESegmenter for TEContentList {
    fn detect_lang_string(&self) -> String {
        let mut detect_lang = String::new();

        for c in self {
            if detect_lang.len() < 1024 {
                detect_lang.push_str(c.to_embedding_string().as_str());
                detect_lang.push('\n');
            }
        }
        detect_lang
    }

    fn segment(&self, tokens_len: fn(&str) -> usize, for_embedding: bool) -> Vec<TEUnit> {
        let mut list: Vec<TEUnit> = Vec::new();
        let mut unit: TEUnit = TEUnit {
            index: 0,
            tokens: 0,
            content: Vec::new(),
        };

        let section_tokens = if for_embedding {
            EMBEDDING_SECTION_TOKENS
        } else {
            TRANSLATE_SECTION_TOKENS
        };

        let high_tokens = if for_embedding {
            EMBEDDING_HIGH_TOKENS
        } else {
            TRANSLATE_HIGH_TOKENS
        };

        for c in self {
            if c.texts.is_empty() {
                if c.id == SECTION_SEPARATOR {
                    // segment embedding content by section separator
                    if unit.tokens > section_tokens && unit.index >= list.len() {
                        list.push(unit);
                        unit = TEUnit {
                            index: list.len(),
                            tokens: 0,
                            content: Vec::new(),
                        };
                    }
                }

                continue;
            }

            let ctl = if for_embedding {
                tokens_len(&c.to_embedding_string())
            } else {
                tokens_len(&c.to_json_string())
            };

            if unit.tokens + ctl > high_tokens {
                unit.tokens += ctl;
                unit.content.push(c.clone());
                list.push(unit);
                unit = TEUnit {
                    index: list.len(),
                    tokens: 0,
                    content: Vec::new(),
                };
            } else {
                unit.tokens += ctl;
                unit.content.push(c.clone());
            }
        }

        if unit.index >= list.len() {
            list.push(unit);
        }

        list
    }
}
