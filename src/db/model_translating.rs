use isolang::Language;

use axum_web::erring::HTTPError;
use scylla_orm::{ColumnsMap, CqlValue, ToCqlVal};
use scylla_orm_macros::CqlOrm;

use crate::db::scylladb;

#[derive(Debug, Default, Clone, CqlOrm)]
pub struct Translating {
    pub gid: xid::Id,
    pub cid: xid::Id,
    pub language: Language,
    pub version: i16,
    pub model: String,
    pub progress: i8,
    pub updated_at: i64,
    pub tokens: i32,
    pub content: Vec<u8>,
    pub error: String,

    pub _fields: Vec<String>, // selected fields，`_` 前缀字段会被 CqlOrm 忽略
}

impl Translating {
    pub fn with_pk(gid: xid::Id, cid: xid::Id, language: Language, version: i16) -> Self {
        Self {
            gid,
            cid,
            language,
            version,
            ..Default::default()
        }
    }

    pub fn select_fields(select_fields: Vec<String>, with_pk: bool) -> anyhow::Result<Vec<String>> {
        if select_fields.is_empty() {
            return Ok(Self::fields());
        }

        let fields = Self::fields();
        for field in &select_fields {
            if !fields.contains(field) {
                return Err(HTTPError::new(400, format!("Invalid field: {}", field)).into());
            }
        }

        let mut select_fields = select_fields;
        if with_pk {
            let field = "gid".to_string();
            if !select_fields.contains(&field) {
                select_fields.push(field);
            }
            let field = "cid".to_string();
            if !select_fields.contains(&field) {
                select_fields.push(field);
            }
            let field = "language".to_string();
            if !select_fields.contains(&field) {
                select_fields.push(field);
            }
            let field = "version".to_string();
            if !select_fields.contains(&field) {
                select_fields.push(field);
            }
        }

        Ok(select_fields)
    }

    pub async fn get_one(
        &mut self,
        db: &scylladb::ScyllaDB,
        select_fields: Vec<String>,
    ) -> anyhow::Result<()> {
        let fields = Self::select_fields(select_fields, false)?;
        self._fields = fields.clone();

        let query = format!(
            "SELECT {} FROM translating WHERE gid=? AND cid=? AND language=? AND version=? LIMIT 1",
            fields.join(",")
        );
        let params = (
            self.gid.to_cql(),
            self.cid.to_cql(),
            self.language.to_cql(),
            self.version,
        );
        let res = db.execute(query, params).await?.single_row()?;

        let mut cols = ColumnsMap::with_capacity(fields.len());
        cols.fill(res, &fields)?;
        self.fill(&cols);

        Ok(())
    }

    pub async fn upsert_fields(
        &mut self,
        db: &scylladb::ScyllaDB,
        cols: ColumnsMap,
    ) -> anyhow::Result<bool> {
        let valid_fields = [
            "model",
            "progress",
            "updated_at",
            "tokens",
            "content",
            "error",
        ];

        let mut set_fields: Vec<String> = Vec::with_capacity(cols.len());
        let mut params: Vec<CqlValue> = Vec::with_capacity(cols.len() + 4);
        for (k, v) in cols.iter() {
            if !valid_fields.contains(&k.as_str()) {
                return Err(HTTPError::new(400, format!("Invalid field: {}", k)).into());
            }
            set_fields.push(format!("{}=?", k));
            params.push(v.to_owned());
        }

        let query = format!(
            "UPDATE translating SET {} WHERE gid=? AND cid=? AND language=? AND version=?",
            set_fields.join(",")
        );
        params.push(self.gid.to_cql());
        params.push(self.cid.to_cql());
        params.push(self.language.to_cql());
        params.push(self.version.to_cql());

        let _ = db.execute(query, params).await?;
        Ok(true)
    }

    pub async fn delete(&mut self, db: &scylladb::ScyllaDB) -> anyhow::Result<bool> {
        let query = "DELETE FROM translating WHERE gid=? AND cid=? AND language=? AND version=?";
        let params = (
            self.gid.to_cql(),
            self.cid.to_cql(),
            self.language.to_cql(),
            self.version.to_cql(),
        );
        let _ = db.execute(query, params).await?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::{str::FromStr, vec};
    use tokio::sync::OnceCell;

    use crate::conf;
    use crate::db::USER_JARVIS;
    use crate::openai;

    use super::*;

    static DB: OnceCell<scylladb::ScyllaDB> = OnceCell::const_new();

    async fn get_db() -> scylladb::ScyllaDB {
        let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
        let res = scylladb::ScyllaDB::new(cfg.scylla, "jarvis_test").await;
        res.unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn translating_model_works() {
        let db = DB.get_or_init(get_db).await;
        let cid = xid::new();
        let gid = xid::Id::from_str(USER_JARVIS).unwrap();
        let mut doc = Translating::with_pk(gid, cid, Language::Eng, 1);

        let res = doc.get_one(db, vec![]).await;
        assert!(res.is_err());
        let err: HTTPError = res.unwrap_err().into();
        assert_eq!(err.code, 404);

        let content: Vec<u8> = vec![0x80];

        let mut cols = ColumnsMap::with_capacity(4);
        cols.set_as("model", &openai::AIModel::GPT3_5.to_string());
        cols.set_as("tokens", &(1000i32));
        cols.set_as("content", &content);

        doc.upsert_fields(db, cols).await.unwrap();

        let mut doc2 = Translating::with_pk(gid, cid, Language::Eng, 1);
        doc2.get_one(db, vec![]).await.unwrap();

        assert_eq!(doc2.tokens, 1000i32);
        assert_eq!(doc2.content, content);
        assert_eq!(doc2.error, "".to_string());

        let mut doc3 = Translating::with_pk(gid, cid, Language::Eng, 1);
        doc3.get_one(db, vec!["error".to_string()]).await.unwrap();
        assert_eq!(doc3.tokens, 0i32);
        assert_eq!(doc3.content.len(), 0);
        assert_eq!(doc3.error, "".to_string());

        let mut cols = ColumnsMap::with_capacity(1);
        cols.set_as("error", &"some error".to_string());
        doc.upsert_fields(db, cols).await.unwrap();

        let mut doc3 = Translating::with_pk(gid, cid, Language::Eng, 1);
        doc3.get_one(db, vec![]).await.unwrap();
        assert_eq!(doc3.tokens, 1000i32);
        assert_eq!(doc3.content, content);
        assert_eq!(doc3.error, "some error".to_string());

        let mut doc = Translating::with_pk(gid, cid, Language::Eng, 2);
        let mut cols = ColumnsMap::with_capacity(1);
        cols.set_as("error", &"some error".to_string());
        doc.upsert_fields(db, cols).await.unwrap();
        doc.get_one(db, vec![]).await.unwrap();
        assert_eq!(doc.tokens, 0i32);
        assert_eq!(doc.content.len(), 0);
        assert_eq!(doc.error, "some error".to_string());
    }
}
