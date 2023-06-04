use axum::{http::Request, middleware::Next, response::Response};
use hyper::HeaderMap;
use serde_json::Value;
use std::sync::Arc;
use std::{collections::BTreeMap, time::Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

pub use structured_logger::unix_ms;

#[derive(Clone)]
pub struct ContextLayer;

pub struct ReqContext {
    pub xid: String,
    pub user: String,
    pub unix_ms: u64,
    pub start: Instant,
    pub kv: RwLock<BTreeMap<String, Value>>,
}

impl ReqContext {
    pub fn new(xid: &str, user: &str) -> Self {
        Self {
            xid: xid.to_string(),
            user: user.to_string(),
            unix_ms: unix_ms(),
            start: Instant::now(),
            kv: RwLock::new(BTreeMap::new()),
        }
    }
}

pub async fn middleware<B>(mut req: Request<B>, next: Next<B>) -> Response {
    let method = req.method().to_string();
    let uri = req.uri().to_string();
    let xid = extract_header(req.headers(), "x-request-id", || Uuid::new_v4().to_string());
    let user = extract_header(req.headers(), "x-user-id", || "".to_string());

    let ctx = Arc::new(ReqContext::new(&xid, &user));
    req.extensions_mut().insert(ctx.clone());

    let res = next.run(req).await;
    let kv = ctx.kv.read().await;
    let status = res.status().as_u16();
    log::info!(target: "api",
        method = method,
        uri = uri,
        xid = xid,
        user = user,
        status = status,
        start = ctx.unix_ms,
        elapsed = ctx.start.elapsed().as_millis() as u64,
        kv = log::as_serde!(*kv);
        "",
    );

    res
}

pub fn extract_header(hm: &HeaderMap, key: &str, or: impl FnOnce() -> String) -> String {
    match hm.get(key) {
        None => or(),
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => or(),
        },
    }
}
