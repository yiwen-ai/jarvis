use std::collections::HashMap;

use super::{qdrant, scylladb, scylladb::CqlValue, ToAnyhowError};

// TABLE: jarvis.embedding
pub struct Embedding {
    pub did: xid::Id,
    pub crc: [u8; 4],

    pub columns: scylladb::ColumnsMap,
}

impl Embedding {
    pub fn new(did: xid::Id, crc: [u8; 4]) -> Self {
        Self {
            did,
            crc,
            columns: scylladb::ColumnsMap::new(),
        }
    }

    pub fn from(did: xid::Id, lang: &str, ids: &str) -> Self {
        let crc: [u8; 4] = crc32fast::hash(format!("{}:{}", lang, ids).as_bytes()).to_be_bytes();
        let mut doc = Self::new(did, crc);
        doc.columns.set_ascii("lang", lang);
        doc.columns.set_ascii("ids", ids);
        doc
    }

    pub fn from_uuid(id: uuid::Uuid) -> Self {
        let data = id.as_bytes();
        let mut did = [0_u8; 12];
        let mut crc = [0_u8; 4];
        did.copy_from_slice(&data[..12]);
        crc.copy_from_slice(&data[12..]);
        Self::new(xid::Id(did), crc)
    }

    // uuid v8
    pub fn uuid(&self) -> uuid::Uuid {
        let mut buf: uuid::Bytes = [0_u8; 16];
        buf[..12].copy_from_slice(self.did.as_bytes());
        buf[12..].copy_from_slice(&self.crc);
        uuid::Uuid::from_bytes(buf)
    }

    pub fn qdrant_point(&self) -> qdrant::PointStruct {
        let mut point = qdrant::PointStruct {
            id: Some(qdrant::PointId::from(self.uuid().to_string())),
            vectors: None,
            payload: HashMap::new(),
        };

        point
            .payload
            .insert("did".to_string(), qdrant::Value::from(self.did.to_string()));
        if let Ok(user) = self.columns.get_as::<String>("user") {
            point
                .payload
                .insert("user".to_string(), qdrant::Value::from(user));
        }
        if let Ok(lang) = self.columns.get_as::<String>("lang") {
            point
                .payload
                .insert("lang".to_string(), qdrant::Value::from(lang));
        }
        if let Ok(ids) = self.columns.get_as::<String>("ids") {
            point
                .payload
                .insert("ids".to_string(), qdrant::Value::from(ids));
        }
        if let Ok(vectors) = self.columns.get_as::<Vec<f32>>("ada2") {
            point.vectors = Some(qdrant::Vectors::from(vectors))
        }
        point
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
            "SELECT {} FROM embedding WHERE did=? AND crc=? LIMIT 1",
            fields.join(",")
        );
        let params = (self.did.as_bytes(), &self.crc);
        let res = db.execute(query, params).await?.single_row();

        if let Err(err) = res {
            return Err(err.to_anyhow_error());
        }
        self.columns.fill(res.unwrap(), fields)?;

        Ok(())
    }

    pub async fn save(&self, db: &scylladb::ScyllaDB) -> anyhow::Result<()> {
        let tokens = CqlValue::Map(Vec::new());
        let query = "INSERT INTO embedding (did,crc,lang,user,tokens,ids,content,ada2) VALUES (?,?,?,?,?,?,?,?) USING TTL 0";
        let params = (
            self.did.as_bytes(),
            self.crc,
            self.columns
                .get("lang")
                .ok_or(anyhow::anyhow!("lang not found"))?,
            self.columns
                .get("user")
                .ok_or(anyhow::anyhow!("user not found"))?,
            self.columns.get("tokens").unwrap_or(&tokens),
            self.columns
                .get("ids")
                .ok_or(anyhow::anyhow!("ids not found"))?,
            self.columns
                .get("content")
                .ok_or(anyhow::anyhow!("content not found"))?,
            self.columns
                .get("ada2")
                .ok_or(anyhow::anyhow!("ada2 embedding not found"))?,
        );
        let _ = db.execute(query, params).await?;

        Ok(())
    }

    pub fn get_fields() -> Vec<&'static str> {
        vec!["lang", "user", "tokens", "ids", "content", "ada2"]
    }
}

mod tests {

    use std::{collections::BTreeMap, str::FromStr};
    use tokio::sync::OnceCell;

    use super::*;
    use crate::{conf, erring::HTTPError, model::TEContentList};

    static DB: OnceCell<scylladb::ScyllaDB> = OnceCell::const_new();

    async fn get_db() -> scylladb::ScyllaDB {
        let cfg = conf::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
        let res = scylladb::ScyllaDB::new(cfg.scylla, "jarvis_test").await;
        res.unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn embedding_model_works() {
        let db = DB.get_or_init(get_db).await;
        let did = xid::new();
        let lang = "English";
        let ids = "abcdef";
        let uid = xid::Id::from_str("jarvis00000000000000").unwrap();
        let mut doc = Embedding::from(did, lang, ids);

        let t_uuid = doc.uuid();
        println!("doc.uuid: {}", t_uuid);
        let t_doc = Embedding::from_uuid(t_uuid);
        assert_eq!(t_doc.did, doc.did);
        assert_eq!(t_doc.crc, doc.crc);

        let res = doc.fill(db, vec![]).await;
        assert!(res.is_err());
        assert_eq!(HTTPError::from(res.unwrap_err()).code, 404);

        let content: TEContentList =
            serde_json::from_str(r#"[{"id":"abcdef","texts":["hello world","你好，世界"]}]"#)
                .unwrap();
        assert_eq!(content[0].texts[1], "你好，世界");

        doc.columns.set_ascii("user", &uid.to_string());
        doc.columns.append_map_i32("tokens", "ada2", 998);
        doc.columns
            .set_in_cbor("content", &content)
            .map_err(HTTPError::from)
            .unwrap();
        doc.columns.set_list_f32("ada2", &vec![1.01f32, 1.02f32]);

        doc.save(db).await.unwrap();

        let mut doc2 = Embedding::new(did, doc.crc);
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.columns.get_as::<String>("user"),
            Ok("jarvis00000000000000".to_string())
        );
        assert_eq!(doc2.columns.get_as::<String>("lang"), Ok(lang.to_string()));
        assert_eq!(
            doc2.columns.get_as::<BTreeMap<String, i32>>("tokens"),
            Ok(BTreeMap::from([("ada2".to_string(), 998i32)]))
        );
        assert_eq!(
            doc2.columns.get_as::<String>("ids"),
            Ok("abcdef".to_string())
        );
        let data = doc2.columns.get_as::<Vec<u8>>("content").unwrap();
        let content2: TEContentList = ciborium::from_reader(&data[..]).unwrap();

        assert_eq!(content2, content);

        let mut doc3 = Embedding::new(did, doc.crc);
        doc3.fill(db, vec!["ids", "ada2"]).await.unwrap();
        assert!(!doc3.columns.has("user"));
        assert!(!doc3.columns.has("tokens"));
        assert!(!doc3.columns.has("gpt4"));
        assert!(!doc3.columns.has("content"));
        assert!(doc3.columns.has("ids"));
        assert_eq!(
            doc3.columns.get_as::<String>("ids"),
            Ok("abcdef".to_string())
        );
        assert_eq!(
            doc3.columns.get_as::<Vec<f32>>("ada2").unwrap(),
            vec![1.01f32, 1.02f32]
        );
        // println!("doc: {:#?}", doc2);
    }
}
