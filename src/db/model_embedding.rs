use isolang::Language;
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

use axum_web::erring::HTTPError;
use scylla_orm::{ColumnsMap, CqlValue, ToCqlVal};
use scylla_orm_macros::CqlOrm;

use crate::db::{qdrant, scylladb};

#[derive(Debug, Default, Clone, CqlOrm)]
pub struct Embedding {
    pub uuid: uuid::Uuid,
    pub cid: xid::Id,
    pub language: Language,
    pub version: i16,
    pub ids: String,
    pub gid: xid::Id,
    pub content: Vec<u8>,

    pub _fields: Vec<String>, // selected fields，`_` 前缀字段会被 CqlOrm 忽略
}

impl Embedding {
    pub fn with_pk(uuid: uuid::Uuid) -> Self {
        Self {
            uuid,
            ..Default::default()
        }
    }

    pub fn from(cid: xid::Id, lang: Language, ids: String) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(cid.as_bytes());
        hasher.update(lang.to_639_3().as_bytes());
        hasher.update(ids.as_bytes());
        let digest = hasher.finalize();
        let mut code = [0u8; 16];
        code.copy_from_slice(&digest[..16]);
        let mut doc = Self::with_pk(uuid::Uuid::from_bytes(code));
        doc.cid = cid;
        doc.language = lang;
        doc.ids = ids;
        doc
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
            let field = "uuid".to_string();
            if !select_fields.contains(&field) {
                select_fields.push(field);
            }
        }

        Ok(select_fields)
    }

    pub fn qdrant_point(&self, vectors: Vec<f32>) -> qdrant::PointStruct {
        let mut point = qdrant::PointStruct {
            id: Some(qdrant::PointId::from(self.uuid.to_string())),
            vectors: Some(qdrant::Vectors::from(vectors)),
            payload: HashMap::new(),
        };

        point
            .payload
            .insert("cid".to_string(), qdrant::Value::from(self.cid.to_string()));
        point.payload.insert(
            "language".to_string(),
            qdrant::Value::from(self.language.to_639_3()),
        );
        point
            .payload
            .insert("gid".to_string(), qdrant::Value::from(self.gid.to_string()));
        point
    }

    pub async fn get_one(
        &mut self,
        db: &scylladb::ScyllaDB,
        select_fields: Vec<String>,
    ) -> anyhow::Result<()> {
        let fields = Self::select_fields(select_fields, false)?;
        self._fields = fields.clone();

        let query = format!(
            "SELECT {} FROM embedding WHERE uuid=? LIMIT 1",
            fields.join(",")
        );
        let params = (self.uuid.to_cql(),);
        let res = db.execute(query, params).await?.single_row()?;

        let mut cols = ColumnsMap::with_capacity(fields.len());
        cols.fill(res, &fields)?;
        self.fill(&cols);

        Ok(())
    }

    pub async fn save(&mut self, db: &scylladb::ScyllaDB) -> anyhow::Result<bool> {
        let fields = Self::fields();
        self._fields = fields.clone();

        let mut cols_name: Vec<&str> = Vec::with_capacity(fields.len());
        let mut vals_name: Vec<&str> = Vec::with_capacity(fields.len());
        let mut params: Vec<&CqlValue> = Vec::with_capacity(fields.len());
        let cols = self.to();

        for field in &fields {
            cols_name.push(field);
            vals_name.push("?");
            params.push(cols.get(field).unwrap());
        }

        // overwrite with new values
        let query = format!(
            "INSERT INTO embedding ({}) VALUES ({})",
            cols_name.join(","),
            vals_name.join(",")
        );

        let _ = db.execute(query, params).await?;
        Ok(true)
    }

    pub async fn list_by_cid(
        db: &scylladb::ScyllaDB,
        cid: xid::Id,
        gid: xid::Id,
        lang: Language,
        version: i16,
        select_fields: Vec<String>,
    ) -> anyhow::Result<Vec<Embedding>> {
        let fields = Self::select_fields(select_fields, true)?;

        let query = format!(
            "SELECT {} FROM embedding WHERE cid=? AND language=? AND version=? AND gid=? LIMIT 1000 ALLOW FILTERING BYPASS CACHE USING TIMEOUT 10s",
            fields.clone().join(",")
        );
        let params = (cid.to_cql(), lang.to_cql(), version, gid.to_cql());
        let rows = db.execute_iter(query, params).await?;

        let mut res: Vec<Embedding> = Vec::with_capacity(rows.len());
        for row in rows {
            let mut doc = Embedding::default();
            let mut cols = ColumnsMap::with_capacity(fields.len());
            cols.fill(row, &fields)?;
            doc.fill(&cols);
            doc._fields = fields.clone();
            res.push(doc);
        }

        Ok(res)
    }
}
