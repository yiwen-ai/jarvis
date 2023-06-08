use std::{net::SocketAddr, sync::Arc};

use axum::{http::header::HeaderName, middleware, routing::get, routing::post, Router};
use structured_logger::{async_json::new_writer, Builder};
use tokio::{
    io, signal,
    time::{sleep, Duration},
};
use tower::ServiceBuilder;
use tower_http::{
    catch_panic::CatchPanicLayer, compression::CompressionLayer,
    propagate_header::PropagateHeaderLayer,
};

mod api;
mod conf;
mod context;
mod db;
mod erring;
mod json_util;
mod lang;
mod model;
mod object;
mod openai;
mod tokenizer;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));

    Builder::with_level(cfg.log.level.as_str())
        .with_target_writer("*", new_writer(io::stdout()))
        .init();

    log::debug!("{:?}", cfg);

    let ld = if cfg.env == "prod" {
        lang::LanguageDetector::new()
    } else {
        lang::LanguageDetector::new_dev()
    };

    let ai = openai::OpenAI::new(cfg.azureai);

    let keyspace = if cfg.env == "test" {
        "jarvis_test"
    } else {
        "jarvis"
    };
    let scylla = db::scylladb::ScyllaDB::new(cfg.scylla, keyspace).await?;

    let qdrant = db::qdrant::Qdrant::new(cfg.qdrant, keyspace).await?;

    let app_state = Arc::new(api::AppState {
        ld,
        ai,
        scylla,
        qdrant,
        translating: Arc::new("translating".to_string()),
        embedding: Arc::new("embedding".to_string()),
    });

    let mds = ServiceBuilder::new()
        .layer(middleware::from_fn(context::middleware))
        .layer(CatchPanicLayer::new())
        .layer(CompressionLayer::new())
        .layer(PropagateHeaderLayer::new(HeaderName::from_static(
            "x-request-id",
        )));

    let app = Router::new()
        .route("/", get(api::version))
        .route("/healthz", get(api::healthz))
        .route("/te", post(api::translate_and_embedding))
        .route("/translating:get", post(api::get_translating))
        .route("/search", post(api::search_content))
        .route_layer(mds)
        .with_state(app_state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], cfg.server.port));
    log::info!("Javis start {} at {}", cfg.env, &addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal(
            app_state.clone(),
            cfg.server.graceful_shutdown,
        ))
        .await?;

    Ok(())
}

async fn shutdown_signal(app: Arc<api::AppState>, wait_secs: usize) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    log::info!("signal received, starting graceful shutdown");

    let mut secs = wait_secs;
    loop {
        let translatings = Arc::strong_count(&app.translating);
        let embeddings = Arc::strong_count(&app.embedding);
        if secs == 0 || (translatings <= 1 && embeddings <= 1) {
            log::info!("Goodbye!"); // Say goodbye and then be terminated...
            return;
        }

        log::info!(
            "signal received, waiting for {} translatings and {} embeddings to finish, or countdown: {} seconds",
            translatings,
            embeddings,
            secs
        );
        secs -= 1;
        sleep(Duration::from_secs(1)).await;
    }
}
