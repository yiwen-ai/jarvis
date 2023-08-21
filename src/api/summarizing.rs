use axum::{extract::State, Extension};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, PackObject};
use scylla_orm::ColumnsMap;

use crate::db;
use crate::lang::Language;
use crate::tokenizer;

use crate::api::{AppState, TEContentList, TEOutput, TESegmenter};

#[derive(Debug, Deserialize, Validate)]
pub struct SummarizingInput {
    pub gid: PackObject<xid::Id>,       // group id, content belong to
    pub cid: PackObject<xid::Id>,       // creation id
    pub language: PackObject<Language>, // the target language translate to
    #[validate(range(min = 1, max = 10000))]
    pub version: u16,

    pub model: Option<String>,
    pub content: Option<PackObject<Vec<u8>>>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct SummarizingOutput {
    pub gid: PackObject<xid::Id>,
    pub cid: PackObject<xid::Id>,       // document id
    pub language: PackObject<Language>, // the origin language detected.
    pub version: u16,
    pub model: String,
    pub tokens: u32,
    pub summary: String,
    pub error: String,
}

pub async fn get(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<SummarizingInput>,
) -> Result<PackObject<SuccessResponse<SummarizingOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid.to_owned();
    let cid = *input.cid.to_owned();
    let language = *input.language.to_owned();

    ctx.set_kvs(vec![
        ("action", "get_summarizing".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    let mut doc = db::Summarizing::with_pk(gid, cid, language, input.version as i16);
    doc.get_one(&app.scylla, vec![]).await?;

    Ok(to.with(SuccessResponse::new(SummarizingOutput {
        gid: to.with(doc.gid),
        cid: to.with(doc.cid),
        language: to.with(doc.language),
        version: doc.version as u16,
        model: doc.model,
        tokens: doc.tokens as u32,
        summary: doc.summary,
        error: doc.error,
    })))
}

pub async fn create(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<SummarizingInput>,
) -> Result<PackObject<SuccessResponse<TEOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid;
    let cid = *input.cid;
    let language = *input.language;

    ctx.set_kvs(vec![
        ("action", "create_translating".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    if language == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    let content: TEContentList =
        cbor_from_slice(&input.content.unwrap_or_default()).map_err(|e| HTTPError {
            code: 400,
            message: format!("Invalid content: {}", e),
            data: None,
        })?;

    let content = content.segment_for_summarizing(tokenizer::tokens_len);
    if content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to summarize".to_string(),
        ));
    }

    tokio::spawn(summarize(
        app,
        ctx.rid.clone(),
        ctx.user.to_string(),
        gid,
        cid,
        input.version as i16,
        language,
        content,
    ));

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(cid),
        detected_language: to.with(language),
    })))
}

async fn summarize(
    app: Arc<AppState>,
    rid: String,
    user: String,
    gid: xid::Id,
    cid: xid::Id,
    version: i16,
    language: Language,
    input: Vec<String>,
) {
    let pieces = input.len();
    let start = Instant::now();
    let tokio_translating = app.translating.clone();
    let mut used_tokens: usize = 0;

    let mut output = String::new();
    for c in input {
        let unit_elapsed = start.elapsed().as_millis() as u64;
        let text = if output.is_empty() {
            c.to_owned()
        } else {
            output.clone() + "\n" + &c
        };

        let res = app
            .ai
            .summarize(&rid, &user, language.to_name(), &text)
            .await;
        let ai_elapsed = start.elapsed().as_millis() as u64 - unit_elapsed;
        match res {
            Err(err) => {
                let mut doc = db::Summarizing::with_pk(gid, cid, language, version);
                let mut cols = ColumnsMap::with_capacity(1);
                cols.set_as("error", &err.to_string());
                let _ = doc.upsert_fields(&app.scylla, cols).await;

                log::error!(target: "summarizing",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_639_3().to_string(),
                    version = version,
                    elapsed = ai_elapsed;
                    "{}", err.to_string(),
                );
                return;
            }
            Ok(_) => {
                log::info!(target: "summarizing",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_639_3().to_string(),
                    version = version,
                    elapsed = ai_elapsed;
                    "success",
                );
            }
        }
        let res = res.unwrap();
        used_tokens += res.0 as usize;
        output = res.1
    }

    // save target lang doc to db
    let mut doc = db::Summarizing::with_pk(gid, cid, language, version);
    let mut cols = ColumnsMap::with_capacity(4);
    cols.set_as("model", &"gpt3.5".to_string());
    cols.set_as("tokens", &(used_tokens as i32));
    cols.set_as("summary", &output);
    cols.set_as("error", &"".to_string());

    match doc.upsert_fields(&app.scylla, cols).await {
        Err(err) => {
            log::error!(target: "summarizing",
                action = "to_scylla",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                language = language.to_639_3().to_string(),
                version = version,
                elapsed = start.elapsed().as_millis() as u64,
                summary = doc.summary.len(),
                pieces = pieces;
                "{}", err,
            );
        }
        Ok(_) => {
            log::info!(target: "summarizing",
                action = "finish",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                language = language.to_639_3().to_string(),
                version = version,
                elapsed = start.elapsed().as_millis() as u64,
                summary = doc.summary.len(),
                pieces = pieces;
                "success",
            );
        }
    };

    let _ = tokio_translating.as_str(); // avoid unused warning
}
