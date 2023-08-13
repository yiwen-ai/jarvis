use axum::{extract::State, Extension};
use qdrant_client::qdrant::point_id::PointIdOptions;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::{str::FromStr, sync::Arc};
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, PackObject};

use crate::db::{self, qdrant};
use crate::lang::Language;
use crate::tokenizer;

use crate::api::{AppState, TEContentList, TEOutput, TESegmenter, TEUnit};

#[derive(Debug, Deserialize, Validate)]
pub struct SearchInput {
    pub input: String,                          // the input text
    pub public: Option<bool>,                   // search public content
    pub gid: Option<PackObject<xid::Id>>,       // group id, content belong to
    pub language: Option<PackObject<Language>>, // the target language
    pub cid: Option<PackObject<xid::Id>>,       // creation id
}

#[derive(Debug, Default, Serialize, Validate)]
pub struct SearchOutput {
    pub gid: PackObject<xid::Id>,       // group id, content belong to
    pub cid: PackObject<xid::Id>,       // creation id
    pub language: PackObject<Language>, // the target language
    pub version: u16,
    pub ids: String,
    pub content: PackObject<Vec<u8>>,
}

pub async fn search(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<SearchInput>,
) -> Result<PackObject<SuccessResponse<Vec<SearchOutput>>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    if input.input.is_empty() {
        return Err(HTTPError::new(400, "Input is empty".to_string()));
    }

    let q: Vec<&str> = input.input.split_whitespace().collect();
    let q = q.join(" ");
    let tokens = tokenizer::tokens_len(&q);

    ctx.set_kvs(vec![("action", "search".into()), ("tokens", tokens.into())])
        .await;

    if tokens < 5 {
        return Ok(to.with(SuccessResponse::new(vec![])));
    }

    let embedding_res = app
        .ai
        .embedding(&ctx.rid, &ctx.user.to_string(), &vec![q.clone()])
        .await
        .map_err(HTTPError::from)?;

    let mut f = qdrant::Filter {
        should: Vec::new(),
        must: Vec::new(),
        must_not: Vec::new(),
    };

    let mut public = input.public.unwrap_or(false);
    if input.gid.is_none() {
        public = true;
    }

    if let Some(gid) = input.gid.clone().map(|v| v.unwrap()) {
        ctx.set("gid", gid.to_string().into()).await;
        let fc = qdrant::FieldCondition {
            key: "gid".to_string(),
            r#match: Some(qdrant::Match {
                match_value: Some(qdrant::MatchValue::Text(gid.to_string())),
            }),
            ..qdrant::FieldCondition::default()
        };
        f.must.push(qdrant::Condition::from(fc))
    }

    if let Some(language) = input.language.map(|v| v.unwrap()) {
        ctx.set("language", language.to_639_3().into()).await;
        let fc = qdrant::FieldCondition {
            key: "language".to_string(),
            r#match: Some(qdrant::Match {
                match_value: Some(qdrant::MatchValue::Text(language.to_639_3().to_string())),
            }),
            ..qdrant::FieldCondition::default()
        };
        f.must.push(qdrant::Condition::from(fc))
    }

    if let Some(cid) = input.cid.map(|v| v.unwrap()) {
        ctx.set("cid", cid.to_string().into()).await;
        let fc = qdrant::FieldCondition {
            key: "cid".to_string(),
            r#match: Some(qdrant::Match {
                match_value: Some(qdrant::MatchValue::Text(cid.to_string())),
            }),
            ..qdrant::FieldCondition::default()
        };
        f.must.push(qdrant::Condition::from(fc))
    }

    let f = if !f.must.is_empty() { Some(f) } else { None };
    let embedding = embedding_res.1[0].to_owned();
    let qd_res = if public {
        app.qdrant
            .search_public_points(embedding, f)
            .await
            .map_err(HTTPError::from)?
    } else {
        app.qdrant
            .search_points(embedding, f)
            .await
            .map_err(HTTPError::from)?
    };

    ctx.set("qd_results", qd_res.result.len().into()).await;
    let mut res: Vec<SearchOutput> = Vec::with_capacity(qd_res.result.len());
    for q in qd_res.result {
        let id = match q.id {
            None => {
                return Err(HTTPError {
                    code: 500,
                    message: "Invalid ScoredPoint id from result".to_string(),
                    data: Some(serde_json::Value::String(format!("{:?}", q.id))),
                });
            }
            Some(id) => match id.point_id_options {
                Some(PointIdOptions::Uuid(x)) => x,
                _ => {
                    return Err(HTTPError {
                        code: 500,
                        message: "Invalid ScoredPoint id from result".to_string(),
                        data: Some(serde_json::Value::String(format!("{:?}", id))),
                    });
                }
            },
        };

        let id = uuid::Uuid::from_str(&id).map_err(|e| HTTPError {
            code: 500,
            message: format!("Extract uuid error: {}", e),
            data: None,
        })?;

        let mut doc = db::Embedding::with_pk(id);

        doc.get_one(
            &app.scylla,
            vec![
                "gid".to_string(),
                "cid".to_string(),
                "language".to_string(),
                "version".to_string(),
            ],
        )
        .await
        .map_err(HTTPError::from)?;

        let to_cid = to.with(doc.cid);
        if res.iter().any(|v| v.cid == to_cid) {
            continue;
        }

        res.push(SearchOutput {
            gid: to.with(doc.gid),
            cid: to_cid,
            language: to.with(doc.language),
            version: doc.version as u16,
            ..Default::default()
        });
    }

    ctx.set("results", res.len().into()).await;
    Ok(to.with(SuccessResponse::new(res)))
}

#[derive(Debug, Deserialize, Validate)]
pub struct EmbeddingInput {
    pub gid: PackObject<xid::Id>, // group id, content belong to
    pub cid: PackObject<xid::Id>, // creation id
    pub language: PackObject<Language>,
    #[validate(range(min = 1, max = 10000))]
    pub version: u16,
    pub content: PackObject<Vec<u8>>,
}

pub async fn create(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<EmbeddingInput>,
) -> Result<PackObject<SuccessResponse<TEOutput>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid;
    let cid = *input.cid;
    let language = *input.language;

    if language == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    ctx.set_kvs(vec![
        ("action", "create_embedding".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().into()),
        ("version", input.version.into()),
    ])
    .await;

    if input.content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to translate".to_string(),
        ));
    }

    let content: TEContentList = cbor_from_slice(&input.content).map_err(|e| HTTPError {
        code: 400,
        message: format!("Invalid content: {}", e),
        data: None,
    })?;

    // start embedding in the background immediately.
    let embedding_input = content.segment_for_embedding(tokenizer::tokens_len);
    tokio::spawn(embedding(
        app,
        ctx.rid.clone(),
        ctx.user.to_string(),
        gid,
        cid,
        input.version as i16,
        language,
        embedding_input,
    ));

    Ok(to.with(SuccessResponse::new(TEOutput {
        cid: to.with(cid),
        detected_language: to.with(language),
    })))
}

async fn embedding(
    app: Arc<AppState>,
    rid: String,
    user: String,
    gid: xid::Id,
    cid: xid::Id,
    version: i16,
    lang: Language,
    input: Vec<Vec<TEUnit>>,
) {
    let pieces = input.len();
    let start = Instant::now();
    let tokio_embedding = app.embedding.clone();
    let mut total_tokens: i32 = 0;

    for unit_group in input {
        let unit_elapsed = start.elapsed().as_millis() as u64;
        let embedding_input: Vec<String> = unit_group
            .iter()
            .map(|unit| unit.to_embedding_string())
            .collect();
        let res = app.ai.embedding(&rid, &user, &embedding_input).await;

        let ai_elapsed = start.elapsed().as_millis() as u64 - unit_elapsed;
        match res {
            Err(err) => {
                log::error!(target: "embedding",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = lang.to_string(),
                    elapsed = ai_elapsed;
                    "{}", err.to_string(),
                );
                continue;
            }
            Ok(_) => {
                log::info!(target: "embedding",
                    action = "call_openai",
                    rid = &rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = lang.to_string(),
                    elapsed = ai_elapsed;
                    "success",
                );
            }
        }

        let (used_tokens, embeddings) = res.unwrap();
        total_tokens += used_tokens as i32;
        for (i, unit) in unit_group.iter().enumerate() {
            let mut doc = db::Embedding::from(cid, lang, unit.ids().join(","));
            doc.gid = gid;
            doc.version = version;

            if let Err(err) = ciborium::into_writer(&unit.content, &mut doc.content) {
                log::warn!(target: "embedding",
                    action = "to_cbor",
                    rid = rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = lang.to_string(),
                    elapsed = ai_elapsed;
                    "{}", err,
                );
                continue;
            }

            let res = doc.save(&app.scylla).await;
            let scylla_elapsed = start.elapsed().as_millis() as u64 - ai_elapsed - unit_elapsed;
            match res {
                Err(err) => {
                    log::error!(target: "embedding",
                        action = "to_scylla",
                        rid = &rid,
                        gid = gid.to_string(),
                        cid = cid.to_string(),
                        language = lang.to_string(),
                        used_tokens = used_tokens,
                        ids = log::as_serde!(unit.ids()),
                        ai_elapsed = ai_elapsed,
                        elapsed = scylla_elapsed;
                        "{}", err,
                    );
                }
                Ok(_) => {
                    log::info!(target: "embedding",
                        action = "to_scylla",
                        rid = &rid,
                        gid = gid.to_string(),
                        cid = cid.to_string(),
                        language = lang.to_string(),
                        used_tokens = used_tokens,
                        ids = log::as_serde!(unit.ids()),
                        ai_elapsed = ai_elapsed,
                        elapsed = scylla_elapsed;
                        "success",
                    );

                    let vectors = embeddings[i].to_vec();
                    match app.qdrant.add_points(vec![doc.qdrant_point(vectors)]).await {
                        Ok(()) => {
                            log::info!(target: "qdrant",
                                action = "to_qdrant",
                                rid = rid,
                                gid = gid.to_string(),
                                cid = cid.to_string(),
                                language = lang.to_string(),
                                ai_elapsed = ai_elapsed,
                                scylla_elapsed = scylla_elapsed,
                                elapsed = start.elapsed().as_millis() as u64 - scylla_elapsed - ai_elapsed - unit_elapsed;
                                "",
                            )
                        }
                        Err(err) => {
                            log::error!(target: "qdrant",
                                action = "to_qdrant",
                                rid = rid,
                                gid = gid.to_string(),
                                cid = cid.to_string(),
                                language = lang.to_string(),
                                ai_elapsed = ai_elapsed,
                                scylla_elapsed = scylla_elapsed,
                                elapsed = start.elapsed().as_millis() as u64- scylla_elapsed- ai_elapsed - unit_elapsed;
                                "{}", err,
                            )
                        }
                    }
                }
            };
        }
    }

    log::info!(target: "embedding",
        action = "finish",
        rid = &rid,
        gid = gid.to_string(),
        cid = cid.to_string(),
        language = lang.to_string(),
        elapsed = start.elapsed().as_millis() as u64,
        pieces = pieces,
        used_tokens = total_tokens;
        "success",
    );

    let _ = tokio_embedding.as_str(); // avoid unused warning
}

#[derive(Debug, Deserialize, Validate)]
pub struct EmbeddingPublicInput {
    pub gid: PackObject<xid::Id>,       // group id, content belong to
    pub cid: PackObject<xid::Id>,       // creation id
    pub language: PackObject<Language>, // the target language translate to
    #[validate(range(min = 1, max = 10000))]
    pub version: u16,
}

pub async fn public(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<EmbeddingPublicInput>,
) -> Result<PackObject<SuccessResponse<()>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let gid = *input.gid;
    let cid = *input.cid;
    let language = *input.language;

    ctx.set_kvs(vec![
        ("action", "make_public".into()),
        ("gid", gid.to_string().into()),
        ("cid", cid.to_string().into()),
        ("language", language.to_639_3().into()),
        ("version", input.version.into()),
    ])
    .await;
    let docs = db::Embedding::list_by_cid(
        &app.scylla,
        cid,
        gid,
        language,
        input.version as i16,
        vec!["cid".to_string()],
    )
    .await?;
    ctx.set("pieces", docs.len().into()).await;

    let rid = ctx.rid.clone();
    let points = docs.into_iter().map(|doc| doc.uuid).collect();
    let qdrant = app.qdrant.clone();
    tokio::spawn(async move {
        let start = Instant::now();
        let tokio_embedding = app.embedding.clone();
        match qdrant.copy_to_public(points).await {
            Ok(()) => {
                log::info!(target: "qdrant",
                    action = "to_public",
                    rid = rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_string(),
                    elapsed = start.elapsed().as_millis() as u64;
                    "success",
                )
            }
            Err(err) => {
                log::error!(target: "qdrant",
                    action = "to_public",
                    rid = rid,
                    gid = gid.to_string(),
                    cid = cid.to_string(),
                    language = language.to_string(),
                    elapsed = start.elapsed().as_millis() as u64;
                    "{}", err,
                )
            }
        }
        let _ = tokio_embedding.as_str(); // avoid unused warning
    });

    Ok(to.with(SuccessResponse::new(())))
}
