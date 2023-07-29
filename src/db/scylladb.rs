use futures::{stream::StreamExt, Stream};
use scylla::{
    frame::value::{BatchValues, ValueList},
    statement::{prepared_statement::PreparedStatement, Consistency, SerialConsistency},
    transport::{query_result::QueryResult, Compression, ExecutionProfile},
    Metrics, Session, SessionBuilder,
};
use std::{sync::Arc, time::Duration};

pub use scylla::{
    batch::Batch,
    frame::response::result::{ColumnType, Row},
    query::Query,
    Bytes,
};

use crate::conf;

pub struct ScyllaDB {
    session: Session,
}

impl ScyllaDB {
    pub async fn new(cfg: conf::ScyllaDB, keyspace: &str) -> anyhow::Result<Self> {
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

        if !keyspace.is_empty() {
            session.use_keyspace(keyspace, false).await?;
        }

        Ok(Self { session })
    }

    pub fn metrics(&self) -> Arc<Metrics> {
        self.session.get_metrics()
    }

    #[cfg(test)]
    pub async fn init_tables_for_test(&self) -> anyhow::Result<()> {
        let schema = std::include_str!("../../cql/schema_keyspace_test.cql");
        exec_cqls(self, schema).await?;

        let schema = std::include_str!("../../cql/schema_table.cql");
        exec_cqls(self, schema).await?;
        Ok(())
    }

    pub async fn query(
        &self,
        query: impl Into<Query>,
        params: impl ValueList,
    ) -> anyhow::Result<QueryResult> {
        let res = self.session.query(query, params).await?;
        Ok(res)
    }

    pub async fn execute(
        &self,
        query: impl Into<Query>,
        params: impl ValueList,
    ) -> anyhow::Result<QueryResult> {
        let mut prepared: PreparedStatement = self.session.prepare(query).await?;

        prepared.set_consistency(Consistency::Quorum);
        let res = self.session.execute(&prepared, params).await?;
        Ok(res)
    }

    pub async fn execute_iter(
        &self,
        query: impl Into<Query>,
        params: impl ValueList,
    ) -> anyhow::Result<Vec<Row>> {
        let mut prepared: PreparedStatement = self.session.prepare(query).await?;

        prepared.set_consistency(Consistency::Quorum);
        let mut rows_stream = self.session.execute_iter(prepared, params).await?;

        let (capacity, _) = rows_stream.size_hint();
        let mut rows: Vec<Row> = Vec::with_capacity(capacity);
        while let Some(next_row) = rows_stream.next().await {
            rows.push(next_row?);
        }
        Ok(rows)
    }

    pub async fn execute_paged(
        &self,
        query: impl Into<Query>,
        params: impl ValueList,
        paging_state: Option<Bytes>,
    ) -> anyhow::Result<Vec<Row>> {
        let mut prepared: PreparedStatement = self.session.prepare(query).await?;

        prepared.set_consistency(Consistency::Quorum);
        let res = self
            .session
            .execute_paged(&prepared, params, paging_state)
            .await?;

        Ok(res.rows.unwrap_or_default())
    }

    pub async fn batch(
        &self,
        statements: Vec<&str>,
        values: impl BatchValues,
    ) -> anyhow::Result<QueryResult> {
        let mut batch: Batch = Default::default();
        for statement in statements {
            batch.append_statement(statement);
        }
        let mut prepared_batch: Batch = self.session.prepare_batch(&batch).await?;
        prepared_batch.set_consistency(Consistency::Quorum);
        let res = self.session.batch(&prepared_batch, values).await?;
        Ok(res)
    }
}

pub fn extract_applied(res: QueryResult) -> bool {
    let res = res
        .single_row()
        .map(|r| r.columns[0].as_ref().and_then(|r| r.as_boolean()))
        .unwrap_or(Some(false));
    res == Some(true)
}

pub async fn exec_cqls(db: &ScyllaDB, cqls: &str) -> anyhow::Result<()> {
    let lines = cqls.lines();
    let mut cql = String::new();
    let mut cqls: Vec<String> = Vec::new();
    for line in lines {
        let line = line.split("--").collect::<Vec<&str>>()[0].trim();
        if line.is_empty() {
            continue;
        }
        cql.push(' ');
        cql.push_str(line);
        if cql.ends_with(';') {
            cqls.push(cql.trim().trim_end_matches(';').to_string());
            cql.clear();
        }
    }

    for cql in cqls {
        let res = db
            .query(cql.clone(), &[])
            .await
            .map_err(|err| anyhow::anyhow!("\ncql: {}\nerror: {}", &cql, &err));
        if res.is_err() {
            let res = res.unwrap_err();
            if res.to_string().contains("Index already exists") {
                println!("WARN: {}", res);
            } else {
                return Err(res);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::conf;
    use crate::db;
    use tokio::sync::OnceCell;

    static DB: OnceCell<db::scylladb::ScyllaDB> = OnceCell::const_new();

    async fn get_db() -> &'static db::scylladb::ScyllaDB {
        DB.get_or_init(|| async {
            let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
            let res = db::scylladb::ScyllaDB::new(cfg.scylla, "").await;
            res.unwrap()
        })
        .await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exec_cqls_works() {
        let db = get_db().await;
        db.init_tables_for_test().await.unwrap();
    }
}
