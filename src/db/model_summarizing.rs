use isolang::Language;

use axum_web::erring::HTTPError;
use scylla_orm::{ColumnsMap, CqlValue, ToCqlVal};
use scylla_orm_macros::CqlOrm;

use crate::db::scylladb;

#[derive(Debug, Default, Clone, CqlOrm)]
pub struct Summarizing {
    pub gid: xid::Id,
    pub cid: xid::Id,
    pub language: Language,
    pub version: i16,
    pub model: String,
    pub tokens: i32,
    pub summary: String,
    pub error: String,

    pub _fields: Vec<String>, // selected fields，`_` 前缀字段会被 CqlOrm 忽略
}

impl Summarizing {
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
            "SELECT {} FROM summarizing WHERE gid=? AND cid=? AND language=? AND version=? LIMIT 1",
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

    // pub async fn save(&mut self, db: &scylladb::ScyllaDB) -> anyhow::Result<bool> {
    //     let fields = Self::fields();
    //     self._fields = fields.clone();

    //     let mut cols_name: Vec<&str> = Vec::with_capacity(fields.len());
    //     let mut vals_name: Vec<&str> = Vec::with_capacity(fields.len());
    //     let mut params: Vec<&CqlValue> = Vec::with_capacity(fields.len());
    //     let cols = self.to();

    //     for field in &fields {
    //         cols_name.push(field);
    //         vals_name.push("?");
    //         params.push(cols.get(field).unwrap());
    //     }

    //     let query = format!(
    //         "INSERT INTO summarizing ({}) VALUES ({})",
    //         cols_name.join(","),
    //         vals_name.join(",")
    //     );

    //     let _ = db.execute(query, params).await?;
    //     Ok(true)
    // }

    pub async fn upsert_fields(
        &mut self,
        db: &scylladb::ScyllaDB,
        cols: ColumnsMap,
    ) -> anyhow::Result<bool> {
        let valid_fields = vec!["model", "tokens", "summary", "error"];

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
            "UPDATE summarizing SET {} WHERE gid=? AND cid=? AND language=? AND version=?",
            set_fields.join(",")
        );
        params.push(self.gid.to_cql());
        params.push(self.cid.to_cql());
        params.push(self.language.to_cql());
        params.push(self.version.to_cql());

        let _ = db.execute(query, params).await?;
        Ok(true)
    }
}
