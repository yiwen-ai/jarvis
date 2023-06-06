use super::{scylladb, scylladb::CqlValue, ToAnyhowError};

// did     BLOB,                # document id, 12 bytes, https://docs.rs/xid/latest/xid/
// lang    ASCII,               # document language
// ids     FROZEN<LIST<ASCII>>, # content's nodes ids list
// user    ASCII,
// tokens  MAP<ASCII, INT>,     # tokens uåsed, example: {"ada2": 299}, ada2 is text-embedding-ada-002
// content BLOB,                # a well processed and segmented content list for embedding in CBOR format
// ada2    LIST<FLOAT>          # embedding by text-embedding-ada-002, 1536 dimensions
pub struct Embedding {
    pub did: xid::Id,
    pub lang: String,
    pub ids: Vec<String>,

    pub columns: scylladb::ColumnsMap,
}

impl Embedding {
    pub fn new(did: xid::Id, lang: String, ids: Vec<String>) -> Self {
        Self {
            did,
            lang,
            ids,
            columns: scylladb::ColumnsMap::new(),
        }
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
            "SELECT {} FROM embedding WHERE did=? AND lang=? AND ids=? LIMIT 1",
            fields.join(",")
        );
        let params = (self.did.as_bytes().to_vec(), &self.lang, &self.ids);
        let res = db.execute(query, params).await?.single_row();

        if let Err(err) = res {
            return Err(err.to_anyhow_error());
        }
        self.columns.fill(res.unwrap(), fields)?;

        Ok(())
    }

    pub async fn save(&self, db: &scylladb::ScyllaDB) -> anyhow::Result<()> {
        let tokens = CqlValue::Map(Vec::new());
        let query = "INSERT INTO embedding (did,lang,ids,user,tokens,content,ada2) VALUES (?,?,?,?,?,?,?) USING TTL 0";
        let params = (
            self.did.as_bytes().to_vec(),
            &self.lang,
            &self.ids,
            self.columns
                .get("user")
                .ok_or(anyhow::anyhow!("user not found"))?,
            self.columns.get("tokens").unwrap_or(&tokens),
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
        vec!["user", "tokens", "content", "ada2"]
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
        let uid = xid::Id::from_str("jarvis00000000000000").unwrap();
        let mut doc = Embedding::new(did, "English".to_string(), vec!["abcdef".to_string()]);

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

        let mut doc2 = Embedding::new(did, "English".to_string(), vec!["abcdef".to_string()]);
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.columns.get_as::<String>("user"),
            Ok("jarvis00000000000000".to_string())
        );
        assert_eq!(
            doc2.columns.get_as::<BTreeMap<String, i32>>("tokens"),
            Ok(BTreeMap::from([("ada2".to_string(), 998i32)]))
        );
        let data = doc2.columns.get_as::<Vec<u8>>("content").unwrap();
        let content2: TEContentList = ciborium::from_reader(&data[..]).unwrap();

        assert_eq!(content2, content);

        let mut doc3 = Embedding::new(did, "English".to_string(), vec!["abcdef".to_string()]);
        doc3.fill(db, vec!["content", "ada2"]).await.unwrap();
        assert!(!doc3.columns.has("user"));
        assert!(!doc3.columns.has("tokens"));
        assert!(!doc3.columns.has("gpt4"));
        assert!(doc3.columns.has("content"));
        assert_eq!(doc3.columns.get_as::<Vec<u8>>("content").unwrap(), data);
        assert_eq!(
            doc3.columns.get_as::<Vec<f32>>("ada2").unwrap(),
            vec![1.01f32, 1.02f32]
        );
        // println!("doc: {:#?}", doc2);
    }
}
