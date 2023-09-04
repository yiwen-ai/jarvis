use axum::{extract::State, Extension};
use serde::{Deserialize, Serialize};
use std::{str::FromStr, sync::Arc, time::Instant};
use tokio::sync::{mpsc, Semaphore};
use validator::Validate;

use axum_web::context::{unix_ms, ReqContext};
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, cbor_to_vec, PackObject};
use scylla_orm::ColumnsMap;

use crate::api::{AppState, TEContentList, TEOutput, TEParams, TESegmenter, PARALLEL_WORKS};
use crate::db;
use crate::lang::Language;
use crate::openai;
use crate::tokenizer;

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
    pub progress: i8,
    pub updated_at: i64,
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
        progress: doc.progress,
        updated_at: doc.updated_at,
        tokens: doc.tokens as u32,
        content: to.with(doc.content),
        error: doc.error,
    })))
}

const IGNORE_LANGGUAGES: [&str; 6] = ["abk", "ava", "bak", "lim", "nya", "iii"];

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

        if !IGNORE_LANGGUAGES.contains(&lg.to_639_3()) {
            list.push((
                lg.to_639_3().to_string(),
                lg.to_name().to_string(),
                lg.to_autonym().unwrap().to_string(),
            ));
        }
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
    let model = match input.model {
        Some(model) => openai::AIModel::from_str(&model.to_lowercase())?,
        None => openai::AIModel::GPT3_5,
    };

    ctx.set_kvs(vec![
        ("action", "create_translating".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", target_language.to_639_3().to_string().into()),
        ("version", input.version.into()),
        ("model", model.to_string().into()),
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

    let now = unix_ms() as i64;
    let mut doc = db::Translating::with_pk(gid, cid, target_language, input.version as i16);
    if doc
        .get_one(
            &app.scylla,
            vec![
                "model".to_string(),
                "updated_at".to_string(),
                "error".to_string(),
            ],
        )
        .await
        .is_ok()
        && doc.model == model.to_string()
        && doc.error.is_empty()
        && now - doc.updated_at < 3600 * 1000
    {
        ctx.set("exists", true.into()).await;
        return Ok(to.with(SuccessResponse::new(TEOutput {
            cid: to.with(cid),
            detected_language: to.with(detected_language),
        })));
    }

    let mut cols = ColumnsMap::with_capacity(6);
    cols.set_as("model", &model.to_string());
    cols.set_as("updated_at", &now);
    cols.set_as("progress", &0i8);
    cols.set_as("tokens", &0i32);
    cols.set_as("content", &Vec::<u8>::new());
    cols.set_as("error", &"".to_string());
    doc.upsert_fields(&app.scylla, cols).await?;

    tokio::spawn(translate(
        app,
        ctx.rid.clone(),
        ctx.user,
        TEParams {
            gid,
            cid,
            version: input.version as i16,
            language: target_language,
            content,
        },
        detected_language,
        model,
    ));

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(cid),
        detected_language: to.with(detected_language),
    })))
}

async fn translate(
    app: Arc<AppState>,
    rid: String,
    user: xid::Id,
    te: TEParams,
    origin_language: Language,
    model: openai::AIModel,
) {
    let tokio_translating = app.translating.clone();

    let content = te.content.segment(&model, tokenizer::tokens_len);
    let pieces = content.len();
    let start = Instant::now();

    log::info!(target: "translating",
        action = "start_job",
        rid = rid,
        user = user.to_string(),
        gid = te.gid.to_string(),
        cid = te.cid.to_string(),
        language = te.language.to_639_3().to_string(),
        pieces = pieces;
        "",
    );

    let semaphore = Arc::new(Semaphore::new(PARALLEL_WORKS));
    let (tx, mut rx) =
        mpsc::channel::<(usize, ReqContext, Result<(u32, TEContentList), HTTPError>)>(pieces);
    for (i, unit) in content.into_iter().enumerate() {
        let rid = rid.clone();
        let user = user;
        let app = app.clone();
        let origin = origin_language.to_name();
        let lang = te.language.to_name();
        let model = model.clone();
        let tx = tx.clone();
        let sem = semaphore.clone();
        tokio::spawn(async move {
            if let Ok(permit) = sem.acquire().await {
                let ctx = ReqContext::new(rid, user, 0);
                match app
                    .ai
                    .translate(&ctx, &model, origin, lang, &unit.to_translating_list())
                    .await
                {
                    Ok((used_tokens, content)) => {
                        drop(permit);
                        let _ = tx
                            .send((i, ctx, Ok((used_tokens, unit.replace_texts(&content)))))
                            .await;
                    }
                    Err(err) => {
                        sem.close();
                        let _ = tx.send((i, ctx, Err(err))).await;
                    }
                };
            }
        });
    }
    drop(tx);

    let mut total_tokens: usize = 0;
    let mut progress = 0usize;
    let mut doc = db::Translating::with_pk(te.gid, te.cid, te.language, te.version);
    let mut res_list: Vec<TEContentList> = Vec::with_capacity(pieces);
    res_list.resize(pieces, vec![]);

    while let Some((i, ctx, res)) = rx.recv().await {
        let ai_elapsed = ctx.start.elapsed().as_millis() as u64;
        let kv = ctx.get_kv().await;
        if let Err(err) = res {
            let mut cols = ColumnsMap::with_capacity(2);
            cols.set_as("updated_at", &(unix_ms() as i64));
            cols.set_as("error", &err.to_string());
            let _ = doc.upsert_fields(&app.scylla, cols).await;

            log::error!(target: "translating",
                action = "call_openai",
                rid = ctx.rid,
                cid = te.cid.to_string(),
                language = te.language.to_639_3().to_string(),
                start = ctx.unix_ms,
                elapsed = ai_elapsed,
                piece_at = i,
                kv = log::as_serde!(kv);
                "{}", err.to_string(),
            );
            return;
        }

        let (used_tokens, content) = res.unwrap();
        total_tokens += used_tokens as usize;
        progress += 1;
        res_list[i] = content;

        let mut cols = ColumnsMap::with_capacity(3);
        cols.set_as("updated_at", &(unix_ms() as i64));
        cols.set_as("progress", &((progress * 100 / pieces) as i8));
        cols.set_as("tokens", &(total_tokens as i32));
        let _ = doc.upsert_fields(&app.scylla, cols).await;

        log::info!(target: "translating",
            action = "call_openai",
            rid = ctx.rid,
            cid = te.cid.to_string(),
            start = ctx.unix_ms,
            elapsed = ai_elapsed,
            tokens = used_tokens,
            total_elapsed = start.elapsed().as_millis(),
            total_tokens = total_tokens,
            piece_at = i,
            kv = log::as_serde!(kv);
            "{}/{}", progress, pieces,
        );
    }

    let mut content_list: TEContentList =
        Vec::with_capacity(res_list.iter().map(|x| x.len()).sum());
    for content in res_list {
        content_list.extend(content);
    }

    // save target lang doc to db
    let content = cbor_to_vec(&content_list);
    if content.is_err() {
        let err = content.unwrap_err().to_string();
        let mut cols = ColumnsMap::with_capacity(2);
        cols.set_as("updated_at", &(unix_ms() as i64));
        cols.set_as("error", &err);
        let _ = doc.upsert_fields(&app.scylla, cols).await;

        log::warn!(target: "translating",
            action = "to_cbor",
            rid = &rid,
            cid = te.cid.to_string();
            "{}", err,
        );
        return;
    }

    let mut cols = ColumnsMap::with_capacity(5);
    let content = content.unwrap();
    cols.set_as("updated_at", &(unix_ms() as i64));
    cols.set_as("progress", &100i8);
    cols.set_as("tokens", &(total_tokens as i32));
    cols.set_as("content", &content);
    cols.set_as("error", &"".to_string());

    let elapsed = start.elapsed().as_millis() as u64;
    match doc.upsert_fields(&app.scylla, cols).await {
        Err(err) => {
            log::error!(target: "translating",
                action = "to_scylla",
                rid = &rid,
                cid = te.cid.to_string(),
                elapsed = start.elapsed().as_millis() as u64 - elapsed,
                content_length = content.len();
                "{}", err,
            );
        }
        Ok(_) => {
            log::info!(target: "translating",
                action = "to_scylla",
                rid = &rid,
                cid = te.cid.to_string(),
                elapsed = start.elapsed().as_millis() as u64 - elapsed,
                content_length = content.len();
                "success",
            );
        }
    };

    log::info!(target: "translating",
        action = "finish_job",
        rid = rid,
        cid = te.cid.to_string(),
        elapsed = start.elapsed().as_millis() as u64,
        pieces = pieces,
        total_tokens = total_tokens;
        "",
    );

    let _ = tokio_translating.as_str(); // avoid unused warning
}
