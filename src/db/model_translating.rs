use super::{scylladb, scylladb::CqlValue, ToAnyhowError};

// did     BLOB,             # document id, 12 bytes, https://docs.rs/xid/latest/xid/
// lang    ASCII,            # document language
// ver     SMALLINT,         # document version by language
// user    ASCII,
// tokens  MAP<ASCII, INT>,  # tokens uåsed, example: {"gpt3": 1299}, gpt3 is gpt-3.5-turbo
// content BLOB,             # a well processed content list, from client input or default GPT model, gpt3.
// gpt4    BLOB              # optional content list, from GPT-4 model, gpt4.
#[derive(Debug, PartialEq)]
pub struct Translating {
    pub did: xid::Id,
    pub lang: String,
    pub ver: i16,

    pub columns: scylladb::ColumnsMap,
}

impl Translating {
    pub fn new(did: xid::Id, lang: String, ver: i16) -> Self {
        Self {
            did,
            lang,
            ver,
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
            "SELECT {} FROM translating WHERE did=? AND lang=? AND ver=? LIMIT 1",
            fields.join(",")
        );
        let params = (self.did.as_bytes().to_vec(), &self.lang, self.ver);
        let res = db.execute(query, params).await?.single_row();

        if let Err(err) = res {
            return Err(err.to_anyhow_error());
        }
        self.columns.fill(res.unwrap(), fields)?;

        Ok(())
    }

    pub async fn save(&self, db: &scylladb::ScyllaDB) -> anyhow::Result<()> {
        let tokens = CqlValue::Map(Vec::new());
        let query = "INSERT INTO translating (did,lang,ver,user,tokens,content) VALUES (?,?,?,?,?,?) USING TTL 0";
        let params = (
            self.did.as_bytes().to_vec(),
            &self.lang,
            self.ver,
            self.columns
                .get("user")
                .ok_or(anyhow::anyhow!("user not found"))?,
            self.columns.get("tokens").unwrap_or(&tokens),
            self.columns
                .get("content")
                .ok_or(anyhow::anyhow!("content not found"))?,
        );
        let _ = db.execute(query, params).await?;

        Ok(())
    }

    pub fn get_fields() -> Vec<&'static str> {
        vec!["user", "tokens", "content", "gpt4"]
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
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
    async fn translating_model_works() {
        let db = DB.get_or_init(get_db).await;
        let did = xid::new();
        let uid = xid::Id::from_str("jarvis00000000000000").unwrap();
        let mut doc = Translating::new(did, "English".to_string(), 1);

        let res = doc.fill(db, vec![]).await;
        assert!(res.is_err());
        assert_eq!(HTTPError::from(res.unwrap_err()).code, 404);

        let content: TEContentList =
            serde_json::from_str(r#"[{"id":"abcdef","texts":["hello world","你好，世界"]}]"#)
                .unwrap();
        assert_eq!(content[0].texts[1], "你好，世界");

        doc.columns.set_ascii("user", &uid.to_string());
        doc.columns
            .set_in_cbor("content", &content)
            .map_err(HTTPError::from)
            .unwrap();

        doc.save(db).await.unwrap();

        let mut doc2 = Translating::new(did, "English".to_string(), 1);
        doc2.fill(db, vec![]).await.unwrap();

        assert_eq!(
            doc2.columns.get_as::<String>("user"),
            Ok("jarvis00000000000000".to_string())
        );
        let content2: TEContentList = doc2.columns.get_from_cbor("content").unwrap();

        assert_eq!(content2, content);

        let mut doc3 = Translating::new(did, "English".to_string(), 1);
        doc3.fill(db, vec!["content"]).await.unwrap();
        assert!(!doc3.columns.has("user"));
        assert!(!doc3.columns.has("tokens"));
        assert!(!doc3.columns.has("gpt4"));
        assert!(doc3.columns.has("content"));
        assert_eq!(
            doc3.columns.get_as::<Vec<u8>>("content").unwrap(),
            doc2.columns.get_as::<Vec<u8>>("content").unwrap()
        )
        // println!("doc: {:#?}", doc2);
    }
}
