use std::collections::BTreeMap;

use super::{scylladb, ToAnyhowError};

// TABLE: jarvis.counter
pub struct Counter {
    pub uid: xid::Id,

    columns: scylladb::ColumnsMap,
}

impl Counter {
    pub fn new(uid: xid::Id) -> Self {
        Self {
            uid,
            columns: scylladb::ColumnsMap::new(),
        }
    }

    pub fn counters(&self) -> BTreeMap<String, i64> {
        let mut counters: BTreeMap<String, i64> = BTreeMap::new();
        for (k, v) in self.columns.iter() {
            counters.insert(k.to_string(), v.as_counter().map_or(0, |c| c.0));
        }
        counters
    }

    pub async fn fill(
        &mut self,
        db: &scylladb::ScyllaDB,
        select_fields: Vec<&str>,
    ) -> anyhow::Result<()> {
        let fields = if select_fields.is_empty() {
            Self::get_fields()
        } else {
            select_fields
        };

        let query = format!(
            "SELECT {} FROM counter WHERE uid=? LIMIT 1",
            fields.join(",")
        );
        let params = (self.uid.as_bytes(),);
        let res = db.execute(query, params).await?.single_row();

        if let Err(err) = res {
            return Err(err.to_anyhow_error());
        }
        self.columns.fill(res.unwrap(), fields)?;

        Ok(())
    }

    pub async fn incr_embedding(
        &mut self,
        db: &scylladb::ScyllaDB,
        tokens: i64,
    ) -> anyhow::Result<()> {
        let query = "UPDATE counter SET embedding=embedding+1, embedding_tokens=embedding_tokens+?  WHERE uid=?";
        let params = (tokens, self.uid.as_bytes());
        db.execute(query, params).await?;
        Ok(())
    }

    pub async fn incr_translating(
        &mut self,
        db: &scylladb::ScyllaDB,
        tokens: i64,
    ) -> anyhow::Result<()> {
        let query = "UPDATE counter SET translating=translating+1, translating_tokens=translating_tokens+? WHERE uid=?";
        let params = (tokens, self.uid.as_bytes());
        db.execute(query, params).await?;
        Ok(())
    }

    pub fn get_fields() -> Vec<&'static str> {
        vec![
            "embedding",
            "embedding_tokens",
            "translating",
            "translating_tokens",
        ]
    }
}

#[cfg(test)]
mod tests {

    use tokio::sync::OnceCell;

    use crate::{conf, erring};

    use super::*;

    static DB: OnceCell<scylladb::ScyllaDB> = OnceCell::const_new();

    async fn get_db() -> scylladb::ScyllaDB {
        let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
        let res = scylladb::ScyllaDB::new(cfg.scylla, "jarvis_test").await;
        res.unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn counter_model_works() {
        let db = DB.get_or_init(get_db).await;
        let uid = xid::new();
        let mut doc = Counter::new(uid);

        let res = doc.fill(db, vec![]).await;
        assert!(res.is_err());
        assert_eq!(erring::HTTPError::from(res.unwrap_err()).code, 404);

        doc.incr_embedding(db, 99).await.unwrap();

        let mut doc2 = Counter::new(uid);
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.counters(),
            BTreeMap::from([
                ("embedding".to_string(), 1i64),
                ("embedding_tokens".to_string(), 99i64),
            ])
        );

        doc.incr_translating(db, 199).await.unwrap();
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.counters(),
            BTreeMap::from([
                ("embedding".to_string(), 1i64),
                ("embedding_tokens".to_string(), 99i64),
                ("translating".to_string(), 1i64),
                ("translating_tokens".to_string(), 199i64),
            ])
        );

        doc2.incr_embedding(db, 100).await.unwrap();
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.counters(),
            BTreeMap::from([
                ("embedding".to_string(), 2i64),
                ("embedding_tokens".to_string(), 199i64),
                ("translating".to_string(), 1i64),
                ("translating_tokens".to_string(), 199i64),
            ])
        );
    }
}
