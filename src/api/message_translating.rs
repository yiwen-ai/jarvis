use axum::{extract::State, Extension};
use serde::{Deserialize, Serialize};
use std::{str::FromStr, sync::Arc, time::Instant};
use tokio::sync::{mpsc, Semaphore};
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, cbor_to_vec, PackObject};

use crate::api::{AppState, TEContentList, TESegmenter, PARALLEL_WORKS};

use crate::lang::Language;
use crate::openai;
use crate::tokenizer;

#[derive(Debug, Deserialize, Validate)]
pub struct MessageTranslatingInput {
    pub id: PackObject<xid::Id>,        // message id
    pub language: PackObject<Language>, // the target language translate to
    #[validate(range(min = 1, max = 32767))]
    pub version: u16,

    pub from_language: Option<PackObject<Language>>,
    pub model: Option<String>,
    pub context: Option<String>,
    pub content: Option<PackObject<Vec<u8>>>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct MessageTranslatingOutput {
    pub model: String,
    pub progress: i8,
    pub tokens: u32,
    pub error: String,
    pub content: PackObject<Vec<u8>>,
}

fn mt_key(id: &xid::Id, lang: &Language, ver: u16) -> String {
    format!("MT:{}:{}:{}", id, lang.to_639_3(), ver)
}

pub async fn get(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<MessageTranslatingInput>,
) -> Result<PackObject<SuccessResponse<MessageTranslatingOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let id = *input.id.to_owned();
    let language = *input.language.to_owned();

    ctx.set_kvs(vec![
        ("action", "get_message_translating".into()),
        ("id", id.to_string().into()),
        ("language", language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    let key = mt_key(&id, &language, input.version);
    let data = app
        .redis
        .get_data(&key)
        .await
        .map_err(|e| HTTPError::new(404, e.to_string()))?;

    let output: MessageTranslatingOutput = cbor_from_slice(&data).map_err(|e| HTTPError {
        code: 500,
        message: format!("Invalid content: {}", e),
        data: None,
    })?;

    Ok(to.with(SuccessResponse::new(output)))
}

pub async fn create(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<MessageTranslatingInput>,
) -> Result<PackObject<SuccessResponse<MessageTranslatingOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let id = *input.id;
    let target_language = *input.language;
    let model = match input.model {
        Some(model) => openai::AIModel::from_str(&model.to_lowercase())?,
        None => openai::AIModel::GPT3_5,
    };
    let from_language = *input.from_language.unwrap_or_default();
    if from_language == target_language
        || from_language == Language::Und
        || target_language == Language::Und
    {
        return Err(HTTPError::new(
            400,
            format!(
                "can not translate from '{}' to '{}'",
                from_language, target_language
            ),
        ));
    }

    ctx.set_kvs(vec![
        ("action", "create_message_translating".into()),
        ("id", id.to_string().into()),
        ("language", target_language.to_639_3().to_string().into()),
        ("version", input.version.into()),
        ("from_language", from_language.to_639_3().to_string().into()),
        ("model", model.to_string().into()),
    ])
    .await;

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

    let key = mt_key(&id, &target_language, input.version);
    if let Ok(data) = app.redis.get_data(&key).await {
        ctx.set("exists", true.into()).await;
        let doc: MessageTranslatingOutput = cbor_from_slice(&data).map_err(|e| HTTPError {
            code: 500,
            message: format!("Invalid content: {}", e),
            data: None,
        })?;
        return Ok(to.with(SuccessResponse::new(doc)));
    }

    let doc = MessageTranslatingOutput {
        model: model.to_string(),
        ..Default::default()
    };
    let data = cbor_to_vec(&doc).map_err(|e| HTTPError {
        code: 500,
        message: format!("Invalid content: {}", e),
        data: None,
    })?;

    match app.redis.new_data(&key, data, 600 * 1000).await {
        Err(err) => Err(HTTPError::new(500, err.to_string())),
        Ok(false) => Ok(to.with(SuccessResponse::new(doc))),
        Ok(true) => {
            tokio::spawn(translate(
                app,
                ctx.rid.clone(),
                ctx.user,
                TParams {
                    id,
                    version: input.version as i16,
                    language: target_language,
                    content,
                },
                input.context.unwrap_or_default(),
                from_language,
                model,
            ));
            Ok(to.with(SuccessResponse::new(doc)))
        }
    }
}

pub(crate) struct TParams {
    pub id: xid::Id,
    pub language: Language,
    pub version: i16,
    pub content: TEContentList,
}

async fn translate(
    app: Arc<AppState>,
    rid: String,
    user: xid::Id,
    te: TParams,
    context: String,
    origin_language: Language,
    model: openai::AIModel,
) {
    let tokio_translating = app.translating.clone();

    let content = te.content.segment(&model, tokenizer::tokens_len);
    let pieces = content.len();
    let start = Instant::now();

    log::info!(target: "message_translating",
        action = "start_job",
        rid = rid,
        user = user.to_string(),
        id = te.id.to_string(),
        language = te.language.to_639_3().to_string(),
        pieces = pieces;
        "",
    );

    let semaphore = Arc::new(Semaphore::new(PARALLEL_WORKS));
    let (tx, mut rx) =
        mpsc::channel::<(usize, ReqContext, Result<(u32, TEContentList), HTTPError>)>(pieces);
    for (i, unit) in content.into_iter().enumerate() {
        let rid = rid.clone();
        let app = app.clone();
        let origin = origin_language.to_name();
        let lang = te.language.to_name();
        let model = model.clone();
        let tx = tx.clone();
        let sem = semaphore.clone();
        let context = context.clone();
        tokio::spawn(async move {
            if let Ok(permit) = sem.acquire().await {
                let ctx = ReqContext::new(rid, user, 0);
                match app
                    .ai
                    .translate(
                        &ctx,
                        &model,
                        &context,
                        origin,
                        lang,
                        &unit.to_translating_list(),
                    )
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
    let key = mt_key(&te.id, &te.language, te.version as u16);
    let mut doc = MessageTranslatingOutput {
        model: model.to_string(),
        ..Default::default()
    };
    let mut res_list: Vec<TEContentList> = Vec::with_capacity(pieces);
    res_list.resize(pieces, vec![]);

    while let Some((i, ctx, res)) = rx.recv().await {
        let ai_elapsed = ctx.start.elapsed().as_millis() as u64;
        let kv = ctx.get_kv().await;
        if let Err(err) = res {
            doc.error = err.to_string();
            doc.progress = 0;
            if let Ok(data) = cbor_to_vec(&doc) {
                let _ = app.redis.update_data(&key, data).await;
            }

            log::error!(target: "message_translating",
                action = "call_openai",
                rid = ctx.rid,
                id = te.id.to_string(),
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

        doc.progress = (progress * 100 / pieces) as i8;
        doc.tokens = total_tokens as u32;
        if let Ok(data) = cbor_to_vec(&doc) {
            let _ = app.redis.update_data(&key, data).await;
        }

        log::info!(target: "message_translating",
            action = "call_openai",
            rid = ctx.rid,
            cid = te.id.to_string(),
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
    if let Err(err) = content {
        doc.error = err.to_string();
        doc.progress = 0;
        if let Ok(data) = cbor_to_vec(&doc) {
            let _ = app.redis.update_data(&key, data).await;
        }

        log::warn!(target: "message_translating",
            action = "to_cbor",
            rid = &rid,
            id = te.id.to_string();
            "{}", err,
        );
        return;
    }

    let content = content.unwrap();
    let content_length = content.len();
    doc.content = PackObject::Cbor(content);
    doc.progress = 100i8;
    doc.tokens = total_tokens as u32;
    doc.error = "".to_string();

    let elapsed = start.elapsed().as_millis() as u64;
    match cbor_to_vec(&doc) {
        Err(err) => {
            log::error!(target: "message_translating",
                action = "cbor_to_vec",
                rid = &rid,
                cid = te.id.to_string(),
                elapsed = start.elapsed().as_millis() as u64 - elapsed,
                content_length = content_length;
                "{}", err,
            );
        }
        Ok(data) => {
            match app.redis.update_data(&key, data).await {
                Err(err) => {
                    log::error!(target: "message_translating",
                        action = "to_redis",
                        rid = &rid,
                        cid = te.id.to_string(),
                        elapsed = start.elapsed().as_millis() as u64 - elapsed,
                        content_length = content_length;
                        "{}", err,
                    );
                }
                Ok(_) => {
                    log::info!(target: "message_translating",
                        action = "finish_job",
                        rid = rid,
                        cid = te.id.to_string(),
                        elapsed = start.elapsed().as_millis() as u64,
                        pieces = pieces,
                        total_tokens = total_tokens;
                        "",
                    );
                }
            };
        }
    }

    let _ = tokio_translating.as_str(); // avoid unused warning
}
