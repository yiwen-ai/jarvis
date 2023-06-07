use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use tokio::time::Duration;

pub use qdrant_client::qdrant::{
    r#match::MatchValue, Condition, FieldCondition, Filter, Match, PointId, PointStruct,
    SearchPoints, SearchResponse, Value, Vectors, WithPayloadSelector,
};

use crate::conf;

pub struct Qdrant {
    client: QdrantClient,
    collection_name: String,
}

impl Qdrant {
    pub async fn new(cfg: conf::Qdrant, collection_name: &str) -> anyhow::Result<Self> {
        let config = QdrantClientConfig {
            uri: cfg.url,
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(3),
            keep_alive_while_idle: true,
            api_key: None,
        };
        let client = QdrantClient::new(Some(config))?;

        let _ = client.collection_info(collection_name).await?;
        Ok(Qdrant {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    pub async fn add_points(&self, points: Vec<PointStruct>) -> anyhow::Result<()> {
        self.client
            .upsert_points(&self.collection_name, points, None)
            .await
            .map(|_| ())
    }

    pub async fn search_points(
        &self,
        vector: Vec<f32>,
        f: Option<Filter>,
    ) -> anyhow::Result<SearchResponse> {
        let search_result = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.to_string(),
                vector,
                filter: f,
                limit: 6,
                with_vectors: None,
                with_payload: Some(WithPayloadSelector::from(true)),
                params: None,
                score_threshold: None,
                offset: None,
                ..Default::default()
            })
            .await?;
        Ok(search_result)
    }
}
