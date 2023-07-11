use isolang::Language;


use axum_web::erring::HTTPError;
use scylla_orm::{ColumnsMap, CqlValue, ToCqlVal};
use scylla_orm_macros::CqlOrm;

use crate::db::{scylladb, scylladb::extract_applied};

#[derive(Debug, Default, Clone, CqlOrm)]
pub struct Translating {
    pub gid: xid::Id,
    pub cid: xid::Id,
    pub language: Language,
    pub version: i16,
    pub model: String,
    pub tokens: i32,
    pub content: Vec<u8>,

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

        let query = format!(
            "INSERT INTO translating ({}) VALUES ({}) IF NOT EXISTS",
            cols_name.join(","),
            vals_name.join(",")
        );

        let res = db.execute(query, params).await?;
        if !extract_applied(res) {
            return Err(HTTPError::new(
                409,
                format!(
                    "{}, {}, {}, {} already exists",
                    self.gid, self.cid, self.language, self.version
                ),
            )
            .into());
        }

        Ok(true)
    }
}

// #[cfg(test)]
// mod tests {
//     use std::str::FromStr;
//     use tokio::sync::OnceCell;

//     use crate::{conf, erring, model};

//     use super::*;

//     static DB: OnceCell<scylladb::ScyllaDB> = OnceCell::const_new();

//     async fn get_db() -> scylladb::ScyllaDB {
//         let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
//         let res = scylladb::ScyllaDB::new(cfg.scylla, "jarvis_test").await;
//         res.unwrap()
//     }

//     #[tokio::test(flavor = "current_thread")]
//     #[ignore]
//     async fn translating_model_works() {
//         let db = DB.get_or_init(get_db).await;
//         let cid = xid::new();
//         let uid = xid::Id::from_str("jarvis00000000000000").unwrap();
//         let mut doc = Translating::new(cid, 1, "English".to_string());

//         let res = doc.fill(db, vec![]).await;
//         assert!(res.is_err());
//         assert_eq!(erring::HTTPError::from(res.unwrap_err()).code, 404);

//         let content: model::TEContentList =
//             serde_json::from_str(r#"[{"id":"abcdef","texts":["hello world","你好，世界"]}]"#)
//                 .unwrap();
//         assert_eq!(content[0].texts[1], "你好，世界");

//         doc.columns.set_ascii("gid", &uid.to_string());
//         doc.columns
//             .set_in_cbor("content", &content)
//             .map_err(erring::HTTPError::from)
//             .unwrap();

//         doc.save(db).await.unwrap();

//         let mut doc2 = Translating::new(cid, 1, "English".to_string());
//         doc2.fill(db, vec![]).await.unwrap();

//         assert_eq!(
//             doc2.columns.get_as::<String>("gid"),
//             Ok("jarvis00000000000000".to_string())
//         );
//         let content2: model::TEContentList = doc2.columns.get_from_cbor("content").unwrap();

//         assert_eq!(content2, content);

//         let mut doc3 = Translating::new(cid, 1, "English".to_string());
//         doc3.fill(db, vec!["content"]).await.unwrap();
//         assert!(!doc3.columns.has("gid"));
//         assert!(!doc3.columns.has("tokens"));
//         assert!(!doc3.columns.has("gpt4"));
//         assert!(doc3.columns.has("content"));
//         assert_eq!(
//             doc3.columns.get_as::<Vec<u8>>("content").unwrap(),
//             doc2.columns.get_as::<Vec<u8>>("content").unwrap()
//         )
//         // println!("doc: {:#?}", doc2);
//     }
// }
