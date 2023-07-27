use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use tokio::time::Duration;

pub use qdrant_client::qdrant::{
    r#match::MatchValue, Condition, FieldCondition, Filter, Match, PointId, PointStruct,
    ReadConsistency, SearchPoints, SearchResponse, Value, Vectors, WithPayloadSelector,
    WithVectorsSelector,
};

use crate::conf;

pub struct Qdrant {
    client: QdrantClient,
    client_public: QdrantClient,
    collection_name: String,
    collection_pub: String,
}

impl Qdrant {
    pub async fn new(cfg: conf::Qdrant, collection_name: &str) -> anyhow::Result<Self> {
        let client = QdrantClient::new(Some(QdrantClientConfig {
            uri: cfg.url.clone(),
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(3),
            keep_alive_while_idle: true,
            api_key: None,
        }))?;
        let _ = client.collection_info(collection_name).await?;

        let client_public = QdrantClient::new(Some(QdrantClientConfig {
            uri: cfg.url,
            timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(3),
            keep_alive_while_idle: true,
            api_key: None,
        }))?;
        let _ = client_public
            .collection_info(collection_name.to_string() + "_pub")
            .await?;
        Ok(Qdrant {
            client,
            client_public,
            collection_name: collection_name.to_string(),
            collection_pub: collection_name.to_string() + "_pub",
        })
    }

    pub async fn add_points(&self, points: Vec<PointStruct>) -> anyhow::Result<()> {
        self.client
            .upsert_points(&self.collection_name, points, None)
            .await
            .map(|_| ())
    }

    pub async fn copy_to_public(&self, points: Vec<uuid::Uuid>) -> anyhow::Result<()> {
        let ids: Vec<PointId> = points
            .iter()
            .map(|p| PointId::from(p.to_string()))
            .collect();
        let res = self
            .client
            .get_points(
                &self.collection_name,
                &ids,
                Some(WithVectorsSelector::from(true)),
                Some(WithPayloadSelector::from(true)),
                Some(ReadConsistency::default()),
            )
            .await?;

        let points: Vec<PointStruct> = res
            .result
            .into_iter()
            .map(|p| PointStruct {
                id: p.id,
                payload: p.payload,
                vectors: p.vectors,
            })
            .collect();
        self.client_public
            .upsert_points(&self.collection_pub, points, None)
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
                limit: 3,
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

    pub async fn search_public_points(
        &self,
        vector: Vec<f32>,
        f: Option<Filter>,
    ) -> anyhow::Result<SearchResponse> {
        let search_result = self
            .client_public
            .search_points(&SearchPoints {
                collection_name: self.collection_name.to_string(),
                vector,
                filter: f,
                limit: 3,
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
