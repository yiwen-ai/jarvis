use axum::{extract::State, Extension, Json};
use qdrant_client::qdrant::point_id::PointIdOptions;
use std::{str::FromStr, sync::Arc};

use crate::context::ReqContext;
use crate::db::{self, qdrant};
use crate::erring::{HTTPError, SuccessResponse};
use crate::lang::{normalize_lang, Language, LanguageDetector};
use crate::model::{self, TESegmenter, Validator};
use crate::openai;
use crate::tokenizer;

pub struct AppState {
    pub ld: LanguageDetector,
    pub ai: openai::OpenAI,
    pub scylla: db::scylladb::ScyllaDB,
    pub qdrant: qdrant::Qdrant,
    pub translating: Arc<String>, // keep the number of concurrent translating tasks
    pub embedding: Arc<String>,   // keep the number of concurrent embedding tasks
}

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

static JARVIS: &str = "jarvis00000000000000";

pub async fn version() -> Json<model::AppVersion> {
    Json(model::AppVersion {
        name: APP_NAME.to_string(),
        version: APP_VERSION.to_string(),
    })
}

pub async fn healthz(State(app): State<Arc<AppState>>) -> Json<model::AppInfo> {
    let m = app.scylla.metrics();
    Json(model::AppInfo {
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

pub async fn get_translating(
    State(app): State<Arc<AppState>>,
    Json(input): Json<model::TEInput>,
) -> Result<Json<SuccessResponse<model::TEOutput>>, HTTPError> {
    if let Some(err) = input.validate() {
        return Err(HTTPError {
            code: 400,
            message: err,
            data: None,
        });
    }

    let did = xid_from_str(&input.did)?;
    let lang = normalize_lang(&input.lang);
    if Language::from_str(&lang).is_err() {
        return Err(HTTPError {
            code: 400,
            message: format!("unsupported language '{}'", &lang),
            data: None,
        });
    }

    let mut doc = db::Translating::new(did, input.version as i16, lang.clone());
    doc.fill(&app.scylla, vec![])
        .await
        .map_err(HTTPError::from)?;

    let content: model::TEContentList = doc
        .columns
        .get_from_cbor("content")
        .map_err(HTTPError::from)?;
    let res = model::TEOutput {
        did: did.to_string(),
        lang: lang.clone(),
        used_tokens: 0,
        content,
    };

    Ok(Json(SuccessResponse { result: res }))
}

pub async fn search_content(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    Json(input): Json<model::SearchInput>,
) -> Result<Json<SuccessResponse<Vec<model::TEOutput>>>, HTTPError> {
    if input.input.is_empty() {
        return Err(HTTPError {
            code: 400,
            message: "input is empty".to_string(),
            data: None,
        });
    }

    let embedding_res = app
        .ai
        .embedding(&ctx.rid, &ctx.user, &input.input)
        .await
        .map_err(HTTPError::from)?;

    let mut f = qdrant::Filter {
        should: Vec::new(),
        must: Vec::new(),
        must_not: Vec::new(),
    };
    if !input.did.is_empty() {
        let mut fc = qdrant::FieldCondition::default();
        fc.key = "did".to_string();
        fc.r#match = Some(qdrant::Match {
            match_value: Some(qdrant::MatchValue::Text(input.did)),
        });
        f.must.push(qdrant::Condition::from(fc))
    }
    if !input.lang.is_empty() {
        let mut fc = qdrant::FieldCondition::default();
        fc.key = "lang".to_string();
        fc.r#match = Some(qdrant::Match {
            match_value: Some(qdrant::MatchValue::Text(input.lang)),
        });
        f.must.push(qdrant::Condition::from(fc))
    }
    if !input.user.is_empty() {
        let mut fc = qdrant::FieldCondition::default();
        fc.key = "user".to_string();
        fc.r#match = Some(qdrant::Match {
            match_value: Some(qdrant::MatchValue::Text(input.user)),
        });
        f.must.push(qdrant::Condition::from(fc))
    }

    let f = if !f.must.is_empty() { Some(f) } else { None };
    let qd_res = app
        .qdrant
        .search_points(embedding_res.1, f)
        .await
        .map_err(HTTPError::from)?;

    let mut res: Vec<model::TEOutput> = Vec::with_capacity(qd_res.result.len());
    for q in qd_res.result {
        let id = match q.id {
            None => {
                return Err(HTTPError {
                    code: 500,
                    message: "invalid ScoredPoint id from result".to_string(),
                    data: Some(serde_json::Value::String(format!("{:?}", q.id))),
                });
            }
            Some(id) => match id.point_id_options {
                Some(PointIdOptions::Uuid(x)) => x,
                _ => {
                    return Err(HTTPError {
                        code: 500,
                        message: "invalid ScoredPoint id from result".to_string(),
                        data: Some(serde_json::Value::String(format!("{:?}", id))),
                    });
                }
            },
        };

        let id = uuid::Uuid::from_str(&id).map_err(|e| HTTPError {
            code: 500,
            message: format!("extract uuid error: {}", e),
            data: None,
        })?;

        let mut doc = db::Embedding::from_uuid(id);
        doc.fill(&app.scylla, vec![])
            .await
            .map_err(HTTPError::from)?;

        res.push(model::TEOutput {
            did: doc.did.to_string(),
            lang: doc
                .columns
                .get_as::<String>("lang")
                .unwrap_or("".to_string()),
            used_tokens: 0,
            content: doc
                .columns
                .get_from_cbor("content")
                .map_err(HTTPError::from)?,
        });
    }

    Ok(Json(SuccessResponse { result: res }))
}

pub async fn translate_and_embedding(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    Json(input): Json<model::TEInput>,
) -> Result<Json<SuccessResponse<model::TEOutput>>, HTTPError> {
    if let Some(err) = input.validate() {
        return Err(HTTPError {
            code: 400,
            message: err,
            data: None,
        });
    }

    let did = xid_from_str(&input.did)?;
    let uid = if ctx.user.is_empty() {
        xid_from_str(JARVIS)?
    } else {
        xid_from_str(&ctx.user)?
    };

    let target_lang = normalize_lang(&input.lang);
    if Language::from_str(&target_lang).is_err() {
        return Err(HTTPError {
            code: 400,
            message: format!("unsupported language '{}' to translate", &target_lang),
            data: None,
        });
    }

    if input.content.is_empty() {
        return Err(HTTPError {
            code: 400,
            message: "empty content to translate".to_string(),
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

    let tokio_translating = app.translating.clone();
    let translate_input = input.content.segment(tokenizer::tokens_len, false);

    let res = translate(
        app.clone(),
        &ctx.rid,
        &ctx.user,
        &input.did,
        &origin_lang,
        &target_lang,
        &translate_input,
    )
    .await?;

    let mut user_counter = db::Counter::new(uid);
    let _ = user_counter
        .incr_translating(&app.scylla, res.used_tokens as i64)
        .await;

    // save origin lang doc to db
    let mut origin_doc = db::Translating::new(did, input.version as i16, origin_lang.clone());
    origin_doc.columns.set_ascii("user", &uid.to_string());
    origin_doc
        .columns
        .set_in_cbor("content", &input.content)
        .map_err(HTTPError::from)?;

    origin_doc
        .save(&app.scylla)
        .await
        .map_err(HTTPError::from)?;

    // save target lang doc to db
    let mut target_doc = db::Translating::new(did, input.version as i16, target_lang.clone());
    target_doc.columns.set_ascii("user", &uid.to_string());
    target_doc
        .columns
        .append_map_i32("tokens", "gpt3.5", res.used_tokens as i32);
    target_doc
        .columns
        .set_in_cbor("content", &res.content)
        .map_err(HTTPError::from)?;

    target_doc
        .save(&app.scylla)
        .await
        .map_err(HTTPError::from)?;

    // start embedding in the background immediately.
    let origin_embedding_input = input.content.segment(tokenizer::tokens_len, true);
    tokio::spawn(embedding(
        app.clone(),
        ctx.rid.clone(),
        ctx.user.clone(),
        did,
        origin_lang.clone(),
        origin_embedding_input,
    ));

    let target_embedding_input = res.content.segment(tokenizer::tokens_len, true);
    tokio::spawn(embedding(
        app.clone(),
        ctx.rid.clone(),
        ctx.user.clone(),
        did,
        target_lang.clone(),
        target_embedding_input,
    ));

    let _ = tokio_translating.as_str(); // avoid unused warning
    Ok(Json(SuccessResponse { result: res }))
}

async fn translate(
    app: Arc<AppState>,
    rid: &str,
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
                rid,
                user,
                origin_lang,
                target_lang,
                &unit.to_translating_list(),
            )
            .await;
        if res.is_err() {
            // here we can retry for 5xx error
            return Err(HTTPError::from(res.err().unwrap()));
        }

        let (total_tokens, content) = res.unwrap();
        rt.used_tokens += total_tokens as usize;
        rt.content.extend(unit.replace_texts(&content));
    }

    Ok(rt)
}

async fn embedding(
    app: Arc<AppState>,
    rid: String,
    user: String,
    did: xid::Id,
    lang: String,
    input: Vec<model::TEUnit>,
) {
    let _ = app.embedding.clone();
    let mut total_tokens: i64 = 0;
    let mut points: Vec<qdrant::PointStruct> = Vec::with_capacity(input.len());
    for unit in input {
        let res = app
            .ai
            .embedding(&rid, &user, &unit.to_embedding_string())
            .await;
        if res.is_err() {
            log::warn!(target: "embedding",
                action = "call_openai",
                rid = &rid,
                user = &user,
                did = did.to_string(),
                lang = &lang;
                "{}", res.err().unwrap(),
            );
            return;
        }

        let (used_tokens, embeddings) = res.unwrap();
        total_tokens += used_tokens as i64;
        // save target lang content embedding to db
        let mut content_embedding = db::Embedding::from(did, &lang, &unit.ids().join(","));
        content_embedding.columns.set_ascii("user", &user);
        content_embedding
            .columns
            .append_map_i32("tokens", "ada2", used_tokens as i32);

        if let Err(err) = content_embedding
            .columns
            .set_in_cbor("content", &unit.content)
        {
            log::warn!(target: "embedding",
                action = "set_cbor",
                rid = &rid,
                user = &user,
                did = did.to_string(),
                lang = &lang;
                "{}", err,
            );
            return;
        }

        content_embedding.columns.set_list_f32("ada2", &embeddings);
        if let Err(err) = content_embedding.save(&app.scylla).await {
            log::warn!(target: "embedding",
                action = "save_db",
                rid = &rid,
                user = &user,
                did = did.to_string(),
                lang = &lang;
                "{}", err,
            );
            return;
        }

        points.push(content_embedding.qdrant_point());
        log::info!(target: "embedding",
            action = "finish_embedding",
            rid = &rid,
            user = &user,
            did = did.to_string(),
            lang = &lang,
            used_tokens = used_tokens,
            ids = log::as_serde!(unit.ids());
            "",
        );
    }

    let uid = xid::Id::from_str(&user).unwrap_or(xid::Id::from_str(JARVIS).unwrap());
    let mut user_counter = db::Counter::new(uid);
    let _ = user_counter.incr_embedding(&app.scylla, total_tokens).await;

    let points_len = points.len();
    if let Err(err) = app.qdrant.add_points(points).await {
        log::warn!(target: "qdrant",
            action = "add_points",
            rid = &rid,
            user = &user,
            did = did.to_string(),
            lang = &lang,
            points = points_len;
            "{}", err,
        );
    } else {
        log::info!(target: "qdrant",
            action = "add_points",
            rid = &rid,
            user = &user,
            did = did.to_string(),
            lang = &lang,
            points = points_len;
            "",
        );
    }

    // let _ = tokio_embedding.as_str(); // avoid unused warning
}

fn xid_from_str(s: &str) -> Result<xid::Id, HTTPError> {
    let id = xid::Id::from_str(s).map_err(|e| HTTPError {
        code: 400,
        message: format!("parse document id error: {}", e),
        data: None,
    })?;
    if id.to_string().as_str() != s {
        return Err(HTTPError {
            code: 400,
            message: format!("xid({}) check failed", s),
            data: None,
        });
    }
    Ok(id)
}
