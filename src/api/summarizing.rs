use axum::{extract::State, Extension};
use serde::Deserialize;
use std::sync::Arc;
use validator::Validate;

use axum_web::context::ReqContext;
use axum_web::erring::{HTTPError, SuccessResponse};
use axum_web::object::{cbor_from_slice, PackObject};

use crate::lang::Language;
use crate::tokenizer;

use crate::api::{AppState, TEContentList, TESegmenter};

#[derive(Debug, Deserialize, Validate)]
pub struct SummarizingInput {
    pub language: PackObject<Language>, // the content language
    pub content: PackObject<Vec<u8>>,
}

pub async fn create(
    State(app): State<Arc<AppState>>,
    Extension(ctx): Extension<Arc<ReqContext>>,
    to: PackObject<SummarizingInput>,
) -> Result<PackObject<SuccessResponse<String>>, HTTPError> {
    let (to, input) = to.unpack();
    input.validate()?;

    let language = *input.language;

    ctx.set_kvs(vec![
        ("action", "create_summarizing".into()),
        ("language", language.to_639_3().to_string().into()),
    ])
    .await;

    if language == Language::Und {
        return Err(HTTPError::new(400, "Invalid language".to_string()));
    }

    let content: TEContentList = cbor_from_slice(&input.content).map_err(|e| HTTPError {
        code: 400,
        message: format!("Invalid content: {}", e),
        data: None,
    })?;

    let content = content.segment_for_summarizing(tokenizer::tokens_len);
    if content.is_empty() {
        return Err(HTTPError::new(
            400,
            "Empty content to summarize".to_string(),
        ));
    }

    let mut tokens = 0usize;
    let mut output: Vec<String> = Vec::with_capacity(content.len());
    for c in content {
        let res = app
            .ai
            .summarize(&ctx.rid, &ctx.user.to_string(), language.to_name(), &c)
            .await?;
        tokens += res.0 as usize;
        output.push(res.1)
    }

    if output.len() > 1 {
        let res = app
            .ai
            .summarize(
                &ctx.rid,
                &ctx.user.to_string(),
                language.to_name(),
                &output.join("\n"),
            )
            .await?;
        tokens += res.0 as usize;
        output.truncate(0);
        output.push(res.1)
    }

    ctx.set_kvs(vec![
        ("total_tokens", tokens.into()),
        ("elapsed", (ctx.start.elapsed().as_millis() as u64).into()),
    ])
    .await;

    Ok(to.with(SuccessResponse::new(output[0].to_owned())))
}
