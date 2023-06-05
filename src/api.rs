use axum::{extract::State, Extension, Json};
use std::{str::FromStr, sync::Arc};

use crate::context::ReqContext;
use crate::db;
use crate::erring::{HTTPError, SuccessResponse};
use crate::json_util::RawJSON;
use crate::lang::{Language, LanguageDetector};
use crate::model::{self, TESegmenter};
use crate::openai;
use crate::tokenizer;

pub struct AppState {
    pub ld: LanguageDetector,
    pub ai: openai::OpenAI,
    pub scylla: db::scylladb::ScyllaDB,
    pub translating: Arc<String>, // keep the number of concurrent translating tasks
    pub embedding: Arc<String>,   // keep the number of concurrent embedding tasks
}

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn version() -> Json<model::AppVersion> {
    Json(model::AppVersion {
        name: APP_NAME.to_string(),
        version: APP_VERSION.to_string(),
    })
}

pub async fn healthz(State(app): State<Arc<AppState>>) -> Json<model::AppInfo> {
    Json(model::AppInfo {
        translating: Arc::strong_count(&app.translating) - 1,
        embedding: Arc::strong_count(&app.embedding) - 1,
    })
}

pub async fn translate_and_embedding(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    Json(input): Json<model::TEInput>,
) -> Result<Json<SuccessResponse<model::TEOutput>>, HTTPError> {
    let counter = app.translating.clone();

    let target_lang = input.lang.to_lowercase();
    if Language::from_str(&target_lang).is_err() {
        return Err(HTTPError {
            code: 400,
            message: format!("Unsupported language '{}' to translate", &target_lang),
            data: None,
        });
    }

    let origin_lang = app.ld.detect_lang(&input.content.detect_lang_string());
    if origin_lang == target_lang {
        return Err(HTTPError {
            code: 400,
            message: format!(
                "No need to translate from '{}' to '{}'",
                origin_lang, target_lang
            ),
            data: None,
        });
    }

    let translate_input = input.content.segment(tokenizer::tokens_len, false);

    let res = translate(
        app.clone(),
        &ctx.xid,
        &ctx.user,
        &input.did,
        &origin_lang,
        &target_lang,
        &translate_input,
    )
    .await;

    if res.is_err() {
        return Err(res.err().unwrap());
    }

    let res = res.ok().unwrap();

    // start embedding in the background immediately.
    let origin_embedding_input = input.content.segment(tokenizer::tokens_len, true);
    tokio::spawn(embedding(
        app.clone(),
        ctx.xid.clone(),
        ctx.user.clone(),
        input.did.clone(),
        origin_lang.clone(),
        origin_embedding_input,
    ));

    let target_embedding_input = res.content.segment(tokenizer::tokens_len, true);
    tokio::spawn(embedding(
        app.clone(),
        ctx.xid.clone(),
        ctx.user.clone(),
        input.did.clone(),
        target_lang.clone(),
        target_embedding_input,
    ));

    let _ = counter.as_str(); // avoid unused warning
    Ok(Json(SuccessResponse { result: res }))
}

async fn translate(
    app: Arc<AppState>,
    xid: &str,
    user: &str,
    did: &str,
    origin_lang: &str,
    target_lang: &str,
    input: &Vec<model::TEUnit>,
) -> Result<model::TEOutput, HTTPError> {
    let mut rt = model::TEOutput {
        did: did.to_string(),
        lang: origin_lang.to_string(),
        used_tokens: 0,
        content: Vec::new(),
    };

    for unit in input {
        let res = app
            .ai
            .translate(
                xid,
                user,
                origin_lang,
                target_lang,
                &unit.content_to_json_string(),
            )
            .await;
        if res.is_err() {
            return Err(HTTPError::from(res.err().unwrap()));
        }

        let (total_tokens, content) = res.unwrap();
        rt.used_tokens += total_tokens as usize;

        let mut list = serde_json::from_str::<Vec<model::TEContent>>(&content);
        if list.is_err() {
            match RawJSON::new(&content).fix_me() {
                Ok(fixed) => {
                    list = serde_json::from_str::<Vec<model::TEContent>>(&fixed);
                }
                Err(er) => {
                    log::error!(target: "json_parse_error",
                        xid = xid,
                        origin = &content;
                        "{}", &er,
                    );
                }
            }
        }

        if list.is_err() {
            return Err(HTTPError {
                code: 422,
                message: list.err().unwrap().to_string(),
                data: None,
            });
        };

        rt.content.extend(list.unwrap());
    }

    Ok(rt)
}

async fn embedding(
    app: Arc<AppState>,
    xid: String,
    user: String,
    _did: String,
    _lang: String,
    input: Vec<model::TEUnit>,
) {
    let counter = app.embedding.clone();
    for unit in input {
        let res = app
            .ai
            .embedding(&xid, &user, &unit.content_to_embedding_string())
            .await;
        if res.is_err() {
            return;
        }

        let (total_tokens, embeddings) = res.unwrap();
        // save to db
        println!("EMBD: {}, {:?}", total_tokens, embeddings)
    }

    let _ = counter.as_str(); // avoid unused warning
}
