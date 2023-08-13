use axum::{extract::State, Extension};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, cbor_to_vec, PackObject};
use scylla_orm::ColumnsMap;

use crate::db;
use crate::lang::Language;
use crate::tokenizer;

use crate::api::{AppState, TEContentList, TEOutput, TESegmenter, TEUnit};

#[derive(Debug, Deserialize, Validate)]
pub struct TranslatingInput {
    pub gid: PackObject<xid::Id>,       // group id, content belong to
    pub cid: PackObject<xid::Id>,       // creation id
    pub language: PackObject<Language>, // the target language translate to
    #[validate(range(min = 1, max = 10000))]
    pub version: u16,

    pub model: Option<String>,
    pub content: Option<PackObject<Vec<u8>>>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct TranslatingOutput {
    pub gid: PackObject<xid::Id>,
    pub cid: PackObject<xid::Id>,       // document id
    pub language: PackObject<Language>, // the origin language detected.
    pub version: u16,
    pub model: String,
    pub tokens: u32,
    pub error: String,
    pub content: PackObject<Vec<u8>>,
}

pub async fn get(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<TranslatingInput>,
) -> Result<PackObject<SuccessResponse<TranslatingOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid.to_owned();
    let cid = *input.cid.to_owned();
    let language = *input.language.to_owned();

    ctx.set_kvs(vec![
        ("action", "get_translating".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    let mut doc = db::Translating::with_pk(gid, cid, language, input.version as i16);
    doc.get_one(&app.scylla, vec![]).await?;

    Ok(to.with(SuccessResponse::new(TranslatingOutput {
        gid: to.with(doc.gid),
        cid: to.with(doc.cid),
        language: to.with(doc.language),
        version: doc.version as u16,
        model: doc.model,
        tokens: doc.tokens as u32,
        content: to.with(doc.content),
        error: doc.error,
    })))
}

pub async fn list_languages(
    to: PackObject<()>,
    State(_): State<Arc<AppState>>,
) -> Result<PackObject<SuccessResponse<Vec<(String, String, String)>>>, HTTPError> {
    let languages = isolang::languages();
    let mut list: Vec<(String, String, String)> = Vec::new();
    for lg in languages {
        if lg.to_639_1().is_none() || lg.to_autonym().is_none() {
            continue;
        }

        list.push((
            lg.to_639_3().to_string(),
            lg.to_name().to_string(),
            lg.to_autonym().unwrap().to_string(),
        ));
    }
    Ok(to.with(SuccessResponse {
        total_size: Some(list.len() as u64),
        next_page_token: None,
        result: list,
    }))
}

#[derive(Debug, Deserialize, Validate)]
pub struct DetectLangInput {
    pub gid: PackObject<xid::Id>,       // group id, content belong to
    pub language: PackObject<Language>, // the fallback language if detect failed
    pub content: PackObject<Vec<u8>>,
}

pub async fn detect_lang(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<DetectLangInput>,
) -> Result<PackObject<SuccessResponse<TEOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid;
    let fallback_language = *input.language;

    ctx.set_kvs(vec![
        ("action", "detect_lang".into()),
        ("gid", gid.to_string().into()),
    ])
    .await;

    let content: TEContentList = cbor_from_slice(&input.content).map_err(|e| HTTPError {
        code: 400,
        message: format!("Invalid content: {}", e),
        data: None,
    })?;

    if content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to translate".to_string(),
        ));
    }

    let string = content.detect_lang_string();
    ctx.set("input_size", string.len().into()).await;
    let mut detected_language = app.ld.detect_lang(&string);
    if detected_language == Language::Und {
        ctx.set("result", "failed".into()).await;
        detected_language = fallback_language;
    }

    ctx.set("language", detected_language.to_639_3().to_string().into())
        .await;

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(xid::Id::default()),
        detected_language: to.with(detected_language),
    })))
}

pub async fn create(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<TranslatingInput>,
) -> Result<PackObject<SuccessResponse<TEOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid;
    let cid = *input.cid;
    let target_language = *input.language;

    ctx.set_kvs(vec![
        ("action", "create_translating".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", target_language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    if target_language == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    let content: TEContentList =
        cbor_from_slice(&input.content.unwrap_or_default()).map_err(|e| HTTPError {
            code: 400,
            message: format!("Invalid content: {}", e),
            data: None,
        })?;
    if content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to translate".to_string(),
        ));
    }

    let detected_language = app.ld.detect_lang(&content.detect_lang_string());
    if detected_language == target_language {
        return Err(HTTPError::new(
            400,
            format!(
                "No need to translate from '{}' to '{}'",
                detected_language, target_language
            ),
        ));
    }

    tokio::spawn(translate(
        app,
        ctx.rid.clone(),
        ctx.user.to_string(),
        gid,
        cid,
        input.version as i16,
        detected_language,
        target_language,
        content.segment(tokenizer::tokens_len),
    ));

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(cid),
        detected_language: to.with(detected_language),
    })))
}

async fn translate(
    app: Arc<AppState>,
    rid: String,
    user: String,
    gid: xid::Id,
    cid: xid::Id,
    version: i16,
    detected_language: Language,
    target_lang: Language,
    input: Vec<TEUnit>,
) {
    let pieces = input.len();
    let start = Instant::now();
    let tokio_translating = app.translating.clone();
    let mut content_list: TEContentList = Vec::new();
    let mut used_tokens: usize = 0;
    let origin_language = detected_language.to_name();
    let language = target_lang.to_name();

    for unit in input {
        let unit_elapsed = start.elapsed().as_millis() as u64;
        let res = app
            .ai
            .translate(
                &rid,
                &user,
                origin_language,
                language,
                &unit.to_translating_list(),
            )
            .await;
        let ai_elapsed = start.elapsed().as_millis() as u64 - unit_elapsed;
        match res {
            Err(err) => {
                let mut doc = db::Translating::with_pk(gid, cid, target_lang, version);
                let mut cols = ColumnsMap::with_capacity(1);
                cols.set_as("error", &err.to_string());
                let _ = doc.upsert_fields(&app.scylla, cols).await;

                log::error!(target: "translating",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_string(),
                    version = version,
                    elapsed = ai_elapsed;
                    "{}", err.to_string(),
                );
                return;
            }
            Ok(_) => {
                log::info!(target: "translating",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_string(),
                    version = version,
                    elapsed = ai_elapsed;
                    "success",
                );
            }
        }

        let (total_tokens, content) = res.unwrap();
        used_tokens += total_tokens as usize;
        content_list.extend(unit.replace_texts(&content));
    }

    // save target lang doc to db
    let mut doc = db::Translating::with_pk(gid, cid, target_lang, version);
    let content = cbor_to_vec(&content_list);
    if content.is_err() {
        log::warn!(target: "translating",
            action = "to_cbor",
            rid = &rid,
            gid = gid.to_string(),
            cid = cid.to_string(),
            language = language.to_string(),
            version = version,
            elapsed = start.elapsed().as_millis() as u64,
            pieces = pieces;
            "{}", content.unwrap_err().to_string(),
        );
        return;
    }
    let mut cols = ColumnsMap::with_capacity(4);
    cols.set_as("model", &"gpt3.5".to_string());
    cols.set_as("tokens", &(used_tokens as i32));
    cols.set_as("content", &content.unwrap());
    cols.set_as("error", &"".to_string());

    match doc.upsert_fields(&app.scylla, cols).await {
        Err(err) => {
            log::error!(target: "translating",
                action = "to_scylla",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                language = language.to_string(),
                version = version,
                elapsed = start.elapsed().as_millis() as u64,
                pieces = pieces;
                "{}", err,
            );
        }
        Ok(_) => {
            log::info!(target: "translating",
                action = "finish",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                language = language.to_string(),
                version = version,
                elapsed = start.elapsed().as_millis() as u64,
                pieces = pieces;
                "success",
            );
        }
    };

    let _ = tokio_translating.as_str(); // avoid unused warning
}
