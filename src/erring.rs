use std::{error::Error, fmt, fmt::Debug};

use axum::http;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone)]
pub struct HTTPError {
    pub code: u16,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: HTTPError,
}

#[derive(Serialize, Deserialize)]
pub struct SuccessResponse<S: Serialize> {
    pub result: S,
}

impl fmt::Display for HTTPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).unwrap_or(self.message.clone())
        )
    }
}

impl Error for HTTPError {}

impl IntoResponse for HTTPError {
    fn into_response(self) -> Response {
        let status = if self.code < 400 {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::from_u16(self.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        };

        let body = Json(ErrorResponse { error: self });
        (status, body).into_response()
    }
}

impl HTTPError {
    pub fn from(err: anyhow::Error) -> Self {
        match err.downcast::<HTTPError>() {
            Ok(err) => err,
            Err(err) => Self {
                code: 500,
                message: err.to_string(),
                data: None,
            },
        }
    }
}

pub fn headers_to_json(headers: &http::HeaderMap) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, value) in headers {
        map.insert(
            key.as_str().to_string(),
            serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
        );
    }
    serde_json::Value::Object(map)
}
