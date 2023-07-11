use axum::{extract::State, Extension};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::PackObject;

use crate::db::{self};
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
    pub content: Option<TEContentList>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct TranslatingOutput {
    pub gid: PackObject<xid::Id>,
    pub cid: PackObject<xid::Id>,       // document id
    pub language: PackObject<Language>, // the origin language detected.
    pub version: u16,
    pub model: String,
    pub tokens: u32,
    pub content: TEContentList,
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
    let content: TEContentList = ciborium::from_reader(&doc.content[..])
        .map_err(|err| HTTPError::new(500, err.to_string()))?;

    Ok(to.with(SuccessResponse::new(TranslatingOutput {
        gid: to.with(doc.gid),
        cid: to.with(doc.cid),
        language: to.with(doc.language),
        version: doc.version as u16,
        model: doc.model,
        tokens: doc.tokens as u32,
        content,
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
    let target_lang = *input.language;

    ctx.set_kvs(vec![
        ("action", "create_translating".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", target_lang.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    if target_lang == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    let content = input.content.unwrap_or_default();
    if content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to translate".to_string(),
        ));
    }

    let detected_language = app.ld.detect_lang(&content.detect_lang_string());
    if detected_language == target_lang {
        return Err(HTTPError::new(
            400,
            format!(
                "No need to translate from '{}' to '{}'",
                detected_language, target_lang
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
        target_lang,
        content.segment(tokenizer::tokens_len, false),
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
    let origin_lang = detected_language.to_name();
    let lang = target_lang.to_name();

    for unit in input {
        let unit_elapsed = start.elapsed().as_millis() as u64;
        let res = app
            .ai
            .translate(&rid, &user, origin_lang, lang, &unit.to_translating_list())
            .await;
        let ai_elapsed = start.elapsed().as_millis() as u64 - unit_elapsed;
        match res {
            Err(err) => {
                log::error!(target: "translating",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    lang = lang.to_string(),
                    ver = version,
                    elapsed = ai_elapsed;
                    "{}", err.to_string(),
                );
                return;
            }
            Ok(_) => {
                log::info!(target: "embedding",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    lang = lang.to_string(),
                    ver = version,
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
    doc.model = "gpt3.5".to_string();
    doc.tokens = used_tokens as i32;

    if let Err(err) = ciborium::into_writer(&content_list, &mut doc.content) {
        log::warn!(target: "translating",
            action = "to_cbor",
            rid = &rid,
            gid = gid.to_string(),
            cid = cid.to_string(),
            lang = lang.to_string(),
            ver = version,
            elapsed = start.elapsed().as_millis() as u64,
            pieces = pieces;
            "{}", err,
        );
        return;
    }

    match doc.save(&app.scylla).await {
        Err(err) => {
            log::error!(target: "translating",
                action = "to_scylla",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                lang = lang.to_string(),
                ver = version,
                elapsed = start.elapsed().as_millis() as u64,
                pieces = pieces;
                "{}", err,
            );
        }
        Ok(false) => {
            log::warn!(target: "translating",
                action = "to_scylla",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                lang = lang.to_string(),
                ver = version,
                elapsed = start.elapsed().as_millis() as u64,
                pieces = pieces;
                "exists",
            );
        }
        Ok(true) => {
            log::info!(target: "translating",
                action = "finish",
                rid = &rid,
                gid = gid.to_string(),
                cid = cid.to_string(),
                lang = lang.to_string(),
                ver = version,
                elapsed = start.elapsed().as_millis() as u64,
                pieces = pieces;
                "success",
            );
        }
    };

    let _ = tokio_translating.as_str(); // avoid unused warning
}