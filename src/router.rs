use axum::{middleware, routing, Router};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    catch_panic::CatchPanicLayer,
    compression::{predicate::SizeAbove, CompressionLayer},
};

use axum_web::context;
use axum_web::encoding;

use crate::api;
use crate::conf;
use crate::db;
use crate::lang;
use crate::openai;

pub async fn new(cfg: conf::Conf) -> anyhow::Result<(Arc<api::AppState>, Router)> {
    let app_state = Arc::new(new_app_state(cfg).await?);

    let mds = ServiceBuilder::new()
        .layer(CatchPanicLayer::new())
        .layer(middleware::from_fn(context::middleware))
        .layer(CompressionLayer::new().compress_when(SizeAbove::new(encoding::MIN_ENCODING_SIZE)));

    let app = Router::new()
        .route("/", routing::get(api::version))
        .route("/healthz", routing::get(api::healthz))
        .nest(
            "/v1/translating",
            Router::new()
                .route("/", routing::post(api::translating::create))
                .route("/get", routing::post(api::translating::get))
                .route(
                    "/list_languages",
                    routing::get(api::translating::list_languages),
                )
                .route(
                    "/detect_language",
                    routing::post(api::translating::detect_lang),
                ),
        )
        .nest(
            "/v1/summarizing",
            Router::new()
                .route("/", routing::post(api::summarizing::create))
                .route("/get", routing::post(api::summarizing::get)),
        )
        .nest(
            "/v1/embedding",
            Router::new()
                .route("/", routing::post(api::embedding::create))
                .route("/search", routing::post(api::embedding::search))
                .route("/public", routing::post(api::embedding::public)),
        )
        .route_layer(mds)
        .with_state(app_state.clone());

    Ok((app_state, app))
}

async fn new_app_state(cfg: conf::Conf) -> anyhow::Result<api::AppState> {
    let ld = lang::LanguageDetector::new();
    let ai = openai::OpenAI::new(cfg.ai);

    let keyspace = if cfg.env == "test" {
        "jarvis_test"
    } else {
        "jarvis"
    };
    let scylla = db::scylladb::ScyllaDB::new(cfg.scylla, keyspace).await?;
    let qdrant = db::qdrant::Qdrant::new(cfg.qdrant, keyspace).await?;
    Ok(api::AppState {
        ld: Arc::new(ld),
        ai: Arc::new(ai),
        scylla: Arc::new(scylla),
        qdrant: Arc::new(qdrant),
        translating: Arc::new("translating".to_string()),
        embedding: Arc::new("embedding".to_string()),
    })
}
