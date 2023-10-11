use axum::{extract::State, Extension};
use finl_unicode::categories::CharacterCategories;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use tokio::sync::{mpsc, Semaphore};
use validator::Validate;

use axum_web::context::{unix_ms, ReqContext};
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, PackObject};
use scylla_orm::ColumnsMap;

use crate::api::{
    extract_summary_keywords, AppState, TEContentList, TEOutput, TEParams, TESegmenter,
    PARALLEL_WORKS, SUMMARIZE_HIGH_TOKENS,
};
use crate::db;
use crate::lang::Language;
use crate::openai;
use crate::tokenizer;

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
    pub progress: i8,
    pub updated_at: i64,
    pub tokens: u32,
    pub summary: String,
    pub keywords: Vec<String>,
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

    let (summary, keywords) = extract_summary_keywords(&doc.summary);
    Ok(to.with(SuccessResponse::new(SummarizingOutput {
        gid: to.with(doc.gid),
        cid: to.with(doc.cid),
        language: to.with(doc.language),
        version: doc.version as u16,
        model: doc.model,
        progress: doc.progress,
        updated_at: doc.updated_at,
        tokens: doc.tokens as u32,
        summary,
        keywords,
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
        ("action", "create_summarizing".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().to_string().into()),
        ("version", input.version.into()),
    ])
    .await;

    if language == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    let now = unix_ms() as i64;
    let mut doc = db::Summarizing::with_pk(gid, cid, language, input.version as i16);
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
        && doc.error.is_empty()
        && now - doc.updated_at < 3600 * 1000
    {
        ctx.set("exists", true.into()).await;

        return Ok(to.with(SuccessResponse::new(TEOutput {
            cid: to.with(cid),
            detected_language: to.with(language),
        })));
    }

    let mut cols = ColumnsMap::with_capacity(6);
    cols.set_as("model", &openai::AIModel::GPT3_5.to_string());
    cols.set_as("updated_at", &now);
    cols.set_as("progress", &0i8);
    cols.set_as("tokens", &0i32);
    cols.set_as("summary", &"".to_string());
    cols.set_as("error", &"".to_string());
    doc.upsert_fields(&app.scylla, cols).await?;

    let content: TEContentList =
        cbor_from_slice(&input.content.unwrap_or_default()).map_err(|e| HTTPError {
            code: 400,
            message: format!("Invalid content: {}", e),
            data: None,
        })?;

    tokio::spawn(summarize(
        app,
        ctx.rid.clone(),
        ctx.user,
        TEParams {
            gid,
            cid,
            version: input.version as i16,
            language,
            content,
        },
    ));

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(cid),
        detected_language: to.with(language),
    })))
}

async fn summarize(app: Arc<AppState>, rid: String, user: xid::Id, te: TEParams) {
    let content = te.content.segment_for_summarizing(tokenizer::tokens_len);
    if content.is_empty() {
        return;
    }

    let tokio_translating = app.translating.clone();
    let pieces = content.len();
    let start = Instant::now();

    log::info!(target: "summarizing",
        action = "start_job",
        rid = rid.clone(),
        user = user.to_string(),
        gid = te.gid.to_string(),
        cid = te.cid.to_string(),
        language = te.language.to_639_3().to_string(),
        version = te.version,
        pieces = pieces;
        "",
    );

    let mut progress = 0usize;
    let mut total_tokens = 00usize;
    let mut doc = db::Summarizing::with_pk(te.gid, te.cid, te.language, te.version);
    let mut keywords_input = content[0].clone();

    let mut output = if pieces == 1 && tokenizer::tokens_len(&content[0]) <= 100 {
        content[0].replace('\n', ". ")
    } else {
        let semaphore = Arc::new(Semaphore::new(PARALLEL_WORKS));
        let (tx, mut rx) =
            mpsc::channel::<(usize, ReqContext, Result<(u32, String), HTTPError>)>(pieces);

        for (i, text) in content.into_iter().enumerate() {
            let rid = rid.clone();
            let app = app.clone();
            let lang = te.language.to_name();
            let tx = tx.clone();
            let sem = semaphore.clone();
            tokio::spawn(async move {
                if let Ok(permit) = sem.acquire().await {
                    let ctx = ReqContext::new(rid, user, 0);
                    let res = if tokenizer::tokens_len(&text) > 100 {
                        app.ai.summarize(&ctx, lang, &text).await
                    } else {
                        // do not need summarizing if too short
                        Ok((0, text.clone()))
                    };

                    if res.is_ok() {
                        drop(permit)
                    } else {
                        sem.close();
                    }
                    let _ = tx.send((i, ctx, res)).await;
                }
            });
        }
        drop(tx);

        let mut res_list: Vec<String> = Vec::with_capacity(pieces);
        res_list.resize(pieces, "".to_string());

        while let Some((i, ctx, res)) = rx.recv().await {
            let ai_elapsed = ctx.start.elapsed().as_millis() as u64;
            let kv = ctx.get_kv().await;
            if let Err(err) = res {
                let mut cols = ColumnsMap::with_capacity(2);
                cols.set_as("updated_at", &(unix_ms() as i64));
                cols.set_as("error", &err.to_string());
                let _ = doc.upsert_fields(&app.scylla, cols).await;

                log::error!(target: "summarizing",
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

            let res = res.unwrap();
            let used_tokens = res.0 as usize;
            total_tokens += used_tokens;
            progress += 1;
            res_list[i] = res.1;

            let mut cols = ColumnsMap::with_capacity(3);
            cols.set_as("updated_at", &(unix_ms() as i64));
            cols.set_as("progress", &((progress * 100 / pieces + 1) as i8));
            cols.set_as("tokens", &(total_tokens as i32));
            let _ = doc.upsert_fields(&app.scylla, cols).await;

            log::info!(target: "summarizing",
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
                "{}/{}", progress, pieces+1,
            );
        }

        if res_list.len() == 1 {
            res_list[0].to_owned()
        } else {
            // extract summary from all pieces and summarize again.
            let mut res_list: Vec<String> = res_list;
            let mut tokens_list: Vec<usize> =
                res_list.iter().map(|s| tokenizer::tokens_len(s)).collect();
            while tokens_list.len() > 2 && tokens_list.iter().sum::<usize>() > SUMMARIZE_HIGH_TOKENS
            {
                let i = tokens_list.len() / 2 + 1;
                // ignore pieces in middle.
                res_list.remove(i);
                tokens_list.remove(i);
            }

            let ctx = ReqContext::new(rid.clone(), user, 0);
            let res = app
                .ai
                .summarize(&ctx, te.language.to_name(), &res_list.join("\n"))
                .await;
            let ai_elapsed = ctx.start.elapsed().as_millis() as u64;
            let kv = ctx.get_kv().await;
            if let Err(err) = res {
                let mut cols = ColumnsMap::with_capacity(2);
                cols.set_as("updated_at", &(unix_ms() as i64));
                cols.set_as("error", &err.to_string());
                let _ = doc.upsert_fields(&app.scylla, cols).await;

                log::error!(target: "summarizing",
                    action = "call_openai",
                    rid = ctx.rid,
                    cid = te.cid.to_string(),
                    language = te.language.to_639_3().to_string(),
                    elapsed = ai_elapsed,
                    piece_at = pieces,
                    kv = log::as_serde!(kv);
                    "{}", err.to_string(),
                );
                return;
            }

            let res = res.unwrap();
            let used_tokens = res.0 as usize;
            total_tokens += used_tokens;
            progress += 1;

            let mut cols = ColumnsMap::with_capacity(3);
            cols.set_as("updated_at", &(unix_ms() as i64));
            cols.set_as("progress", &100i8);
            cols.set_as("tokens", &(total_tokens as i32));
            let _ = doc.upsert_fields(&app.scylla, cols).await;

            log::info!(target: "summarizing",
                action = "call_openai",
                rid = ctx.rid,
                cid = te.cid.to_string(),
                elapsed = ai_elapsed,
                tokens = used_tokens,
                total_elapsed = start.elapsed().as_millis(),
                total_tokens = total_tokens,
                piece_at = pieces,
                kv = log::as_serde!(kv);
                "{}/{}", progress, pieces+1,
            );

            res.1
        }
    };

    // get keywords
    {
        if pieces > 1 {
            keywords_input = output.clone();
        }
        let ctx = ReqContext::new(rid.clone(), user, 0);
        let res = app
            .ai
            .keywords(&ctx, te.language.to_name(), &keywords_input)
            .await;
        let ai_elapsed = ctx.start.elapsed().as_millis() as u64;
        let kv = ctx.get_kv().await;

        match res {
            Err(err) => {
                log::error!(target: "keywords",
                    action = "call_openai",
                    rid = ctx.rid,
                    cid = te.cid.to_string(),
                    language = te.language.to_639_3().to_string(),
                    elapsed = ai_elapsed,
                    piece_at = pieces,
                    kv = log::as_serde!(kv);
                    "{}", err.to_string(),
                );
            }
            Ok(res) => {
                total_tokens += res.0 as usize;
                let keywords: Vec<&str> = res
                    .1
                    .trim()
                    .split(char::is_punctuation)
                    .filter_map(|s| match s.trim_matches(|c: char| !c.is_letter()) {
                        "" => None,
                        v => Some(v),
                    })
                    .collect();
                output = keywords.join(", ") + "\n" + &output;
            }
        }
    }

    // save target lang doc to db
    let mut cols = ColumnsMap::with_capacity(5);
    cols.set_as("updated_at", &(unix_ms() as i64));
    cols.set_as("progress", &100i8);
    cols.set_as("tokens", &(total_tokens as i32));
    cols.set_as("summary", &output);
    cols.set_as("error", &"".to_string());

    let elapsed = start.elapsed().as_millis() as u64;
    match doc.upsert_fields(&app.scylla, cols).await {
        Err(err) => {
            log::error!(target: "summarizing",
                action = "to_scylla",
                rid = rid.clone(),
                cid = te.cid.to_string(),
                elapsed = start.elapsed().as_millis() as u64 - elapsed,
                summary_length = output.len();
                "{}", err,
            );
        }
        Ok(_) => {
            log::info!(target: "summarizing",
                action = "to_scylla",
                rid = rid.clone(),
                cid = te.cid.to_string(),
                elapsed = start.elapsed().as_millis() as u64 - elapsed,
                summary_length = output.len();
                "",
            );
        }
    };

    log::info!(target: "summarizing",
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
