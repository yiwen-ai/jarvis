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
            "lang".to_string(),
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

        let query = scylladb::Query::new(format!(
            "SELECT {} FROM embedding WHERE cid=? AND language=? AND version=? AND gid=? LIMIT 1000 ALLOW FILTERING BYPASS CACHE USING TIMEOUT 10s",
            fields.clone().join(",")
        ))
        .with_page_size(1000i32);
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

// #[cfg(test)]
// mod tests {

//     use std::{collections::BTreeMap, str::FromStr};
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
//     async fn embedding_model_works() {
//         let db = DB.get_or_init(get_db).await;
//         let cid = xid::new();
//         let lang = "English";
//         let ids = "abcdef";
//         let uid = xid::Id::from_str("jarvis00000000000000").unwrap();
//         let mut doc = Embedding::from(cid, lang, ids);

//         let t_uuid = doc.uuid();
//         println!("doc.uuid: {}", t_uuid);
//         let t_doc = Embedding::from_uuid(t_uuid);
//         assert_eq!(t_doc.cid, doc.cid);
//         assert_eq!(t_doc.crc, doc.crc);

//         let res = doc.fill(db, vec![]).await;
//         assert!(res.is_err());
//         assert_eq!(erring::HTTPError::from(res.unwrap_err()).code, 404);

//         let content: model::TEContentList =
//             serde_json::from_str(r#"[{"id":"abcdef","texts":["hello world","你好，世界"]}]"#)
//                 .unwrap();
//         assert_eq!(content[0].texts[1], "你好，世界");

//         doc.columns.set_ascii("gid", &uid.to_string());
//         doc.columns.append_map_i32("tokens", "ada2", 998);
//         doc.columns
//             .set_in_cbor("content", &content)
//             .map_err(erring::HTTPError::from)
//             .unwrap();
//         doc.columns.set_list_f32("ada2", &vec![1.01f32, 1.02f32]);

//         doc.save(db).await.unwrap();

//         let mut doc2 = Embedding::new(cid, doc.crc);
//         doc2.fill(db, vec![]).await.unwrap();

//         assert_eq!(
//             doc2.columns.get_as::<String>("gid"),
//             Ok("jarvis00000000000000".to_string())
//         );
//         assert_eq!(doc2.columns.get_as::<String>("lang"), Ok(lang.to_string()));
//         assert_eq!(
//             doc2.columns.get_as::<BTreeMap<String, i32>>("tokens"),
//             Ok(BTreeMap::from([("ada2".to_string(), 998i32)]))
//         );
//         assert_eq!(
//             doc2.columns.get_as::<String>("ids"),
//             Ok("abcdef".to_string())
//         );
//         let data = doc2.columns.get_as::<Vec<u8>>("content").unwrap();
//         let content2: model::TEContentList = ciborium::from_reader(&data[..]).unwrap();

//         assert_eq!(content2, content);

//         let mut doc3 = Embedding::new(cid, doc.crc);
//         doc3.fill(db, vec!["ids", "ada2"]).await.unwrap();
//         assert!(!doc3.columns.has("gid"));
//         assert!(!doc3.columns.has("tokens"));
//         assert!(!doc3.columns.has("gpt4"));
//         assert!(!doc3.columns.has("content"));
//         assert!(doc3.columns.has("ids"));
//         assert_eq!(
//             doc3.columns.get_as::<String>("ids"),
//             Ok("abcdef".to_string())
//         );
//         assert_eq!(
//             doc3.columns.get_as::<Vec<f32>>("ada2").unwrap(),
//             vec![1.01f32, 1.02f32]
//         );
//         // println!("doc: {:#?}", doc2);
//     }
// }
