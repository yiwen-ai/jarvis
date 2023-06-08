use async_trait::async_trait;
use axum::{
    body::HttpBody,
    extract::{FromRequest, FromRequestParts},
    http::{
        header::{self, HeaderMap, HeaderValue},
        request::{Parts, Request},
        StatusCode,
    },
    response::{IntoResponse, Response},
    BoxError,
};
use bytes::{BufMut, Bytes, BytesMut};
use serde::{de::DeserializeOwned, Serialize};
use std::{convert::Infallible, error::Error};

use crate::erring::HTTPError;

#[derive(Debug, Clone, Default)]
pub struct Object<T>(pub ObjectType, pub T);

#[derive(Debug, Clone, Default)]
pub enum ObjectType {
    #[default]
    Json,
    Cbor,
    Other(String),
}

impl ObjectType {
    pub fn or_default(&self) -> Self {
        match self {
            ObjectType::Json => ObjectType::Json,
            ObjectType::Cbor => ObjectType::Cbor,
            ObjectType::Other(_) => ObjectType::Json,
        }
    }
}

#[async_trait]
impl<S> FromRequestParts<S> for ObjectType
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        match get_content_type(&parts.headers) {
            ObjectType::Json => Ok(ObjectType::Json),
            ObjectType::Cbor => Ok(ObjectType::Cbor),
            ObjectType::Other(v) => {
                if let Some(accept) = parts.headers.get(header::ACCEPT) {
                    if let Ok(accept) = accept.to_str() {
                        if accept.contains("application/cbor") {
                            return Ok(ObjectType::Cbor);
                        }
                        if accept.contains("application/json") {
                            return Ok(ObjectType::Json);
                        }
                        if let Some(accept) = accept.split(',').collect::<Vec<&str>>().first() {
                            return Ok(ObjectType::Other(accept.to_string()));
                        }
                    }
                }
                Ok(ObjectType::Other(v))
            }
        }
    }
}

#[async_trait]
impl<T, S, B> FromRequest<S, B> for Object<T>
where
    T: DeserializeOwned,
    B: HttpBody + Send + 'static,
    B::Data: Send,
    B::Error: Into<BoxError>,
    S: Send + Sync,
{
    type Rejection = HTTPError;

    async fn from_request(req: Request<B>, state: &S) -> Result<Self, Self::Rejection> {
        let ct = get_content_type(req.headers());
        if let ObjectType::Other(val) = ct {
            return Err(HTTPError {
                code: StatusCode::UNSUPPORTED_MEDIA_TYPE.as_u16(),
                message: format!("Unsupported media type, {}", val),
                data: None,
            });
        }

        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(|err| HTTPError {
                code: StatusCode::BAD_REQUEST.as_u16(),
                message: format!("Invalid body, {}", err),
                data: None,
            })?;

        match ct {
            ObjectType::Json => {
                let value: T = serde_json::from_slice(&bytes).map_err(|err| HTTPError {
                    code: StatusCode::BAD_REQUEST.as_u16(),
                    message: format!("Invalid JSON body, {}", err),
                    data: None,
                })?;
                Ok(Object(ct, value))
            }
            ObjectType::Cbor => {
                let value: T = ciborium::from_reader(&bytes[..]).map_err(|err| HTTPError {
                    code: StatusCode::BAD_REQUEST.as_u16(),
                    message: format!("Invalid CBOR body, {}", err),
                    data: None,
                })?;
                Ok(Object(ct, value))
            }
            _ => unreachable!(),
        }
    }
}

fn get_content_type(headers: &HeaderMap) -> ObjectType {
    let content_type = if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
        content_type
    } else {
        return ObjectType::Other("".to_string());
    };

    let content_type = if let Ok(content_type) = content_type.to_str() {
        content_type
    } else {
        return ObjectType::Other("".to_string());
    };

    let mime = if let Ok(mime) = content_type.parse::<mime::Mime>() {
        mime
    } else {
        return ObjectType::Other(content_type.to_string());
    };

    if mime.type_() == "application" {
        if mime.subtype() == "cbor" || mime.suffix().map_or(false, |name| name == "cbor") {
            return ObjectType::Cbor;
        } else if mime.subtype() == "json" || mime.suffix().map_or(false, |name| name == "json") {
            return ObjectType::Json;
        }
    }

    ObjectType::Other(content_type.to_string())
}

impl<T> IntoResponse for Object<T>
where
    T: Serialize,
{
    fn into_response(self) -> Response {
        // Use a small initial capacity of 128 bytes like serde_json::to_vec
        // https://docs.rs/serde_json/1.0.82/src/serde_json/ser.rs.html#2189
        let mut buf = BytesMut::with_capacity(128).writer();
        let res: Result<Response, Box<dyn Error>> = match self.0 {
            ObjectType::Json => match serde_json::to_writer(&mut buf, &self.1) {
                Ok(()) => Ok((
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static(mime::APPLICATION_JSON.as_ref()),
                    )],
                    buf.into_inner().freeze(),
                )
                    .into_response()),
                Err(err) => Err(Box::new(err)),
            },
            ObjectType::Cbor => match ciborium::into_writer(&self.1, &mut buf) {
                Ok(()) => Ok((
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/cbor"),
                    )],
                    buf.into_inner().freeze(),
                )
                    .into_response()),
                Err(err) => Err(Box::new(err)),
            },
            ObjectType::Other(val) => Ok((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static(mime::TEXT_PLAIN_UTF_8.as_ref()),
                )],
                val,
            )
                .into_response()),
        };

        match res {
            Ok(res) => res,
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static(mime::TEXT_PLAIN_UTF_8.as_ref()),
                )],
                err.to_string(),
            )
                .into_response(),
        }
    }
}
