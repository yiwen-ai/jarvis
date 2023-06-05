use anyhow::Result;
use scylla::{
    statement::{Consistency, SerialConsistency},
    transport::{Compression, ExecutionProfile},
    Metrics, Session, SessionBuilder,
};
use std::{sync::Arc, time::Duration};

pub use scylla::query::Query;

use crate::conf;

pub struct ScyllaDB {
    session: Session,
}

impl ScyllaDB {
    pub async fn new(cfg: conf::ScyllaDB) -> Result<Self> {
        // use tls https://github.com/scylladb/scylla-rust-driver/blob/main/examples/tls.rs

        let handle = ExecutionProfile::builder()
            .consistency(Consistency::LocalQuorum)
            .serial_consistency(Some(SerialConsistency::LocalSerial))
            .request_timeout(Some(Duration::from_secs(5)))
            .build()
            .into_handle();

        let session: Session = SessionBuilder::new()
            .known_nodes(&cfg.nodes)
            .user(cfg.username, cfg.password)
            .compression(Some(Compression::Lz4))
            .default_execution_profile_handle(handle)
            .build()
            .await?;

        session.use_keyspace("jarvis", false).await?;

        Ok(Self { session })
    }

    pub fn metrics(&self) -> Arc<Metrics> {
        self.session.get_metrics()
    }

    pub async fn execute(&self, query: Query) -> Result<()> {
        self.session.query(query, &[]).await?;
        Ok(())
    }
}
