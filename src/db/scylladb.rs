use scylla::{
    cql_to_rust::{FromCqlVal, FromCqlValError},
    frame::response::result::Row,
    frame::value::ValueList,
    statement::{prepared_statement::PreparedStatement, Consistency, SerialConsistency},
    transport::{
        errors::QueryError, query_result::QueryResult, query_result::SingleRowError, Compression,
        ExecutionProfile,
    },
    Metrics, Session, SessionBuilder,
};
use serde::{de::DeserializeOwned, Serialize};

use std::{
    collections::{btree_map::Iter, BTreeMap},
    sync::Arc,
    time::Duration,
};

pub use scylla::{frame::response::result::CqlValue, query::Query};

use crate::conf;
use crate::erring::HTTPError;

use super::ToAnyhowError;

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

        session.use_keyspace(keyspace, false).await?;

        Ok(Self { session })
    }

    pub fn metrics(&self) -> Arc<Metrics> {
        self.session.get_metrics()
    }

    pub async fn execute(
        &self,
        query: impl Into<Query>,
        params: impl ValueList,
    ) -> anyhow::Result<QueryResult> {
        let mut prepared: PreparedStatement = self.session.prepare(query).await?;

        prepared.set_consistency(Consistency::One);
        match self.session.execute(&prepared, params).await {
            Ok(result) => Ok(result),
            Err(err) => Err(err.to_anyhow_error()),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct ColumnsMap(BTreeMap<String, CqlValue>);

impl ColumnsMap {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&CqlValue> {
        match self.0.get(key) {
            Some(v) => Some(v),
            None => None,
        }
    }

    pub fn get_as<T: FromCqlVal<CqlValue>>(&self, key: &str) -> anyhow::Result<T, FromCqlValError> {
        match self.0.get(key) {
            Some(v) => T::from_cql(v.clone()),
            None => Err(FromCqlValError::ValIsNull),
        }
    }

    pub fn get_from_cbor<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<T> {
        let data = self.get_as::<Vec<u8>>(key)?;
        let val: T = ciborium::from_reader(&data[..])?;
        Ok(val)
    }

    pub fn iter(&self) -> Iter<'_, String, CqlValue> {
        self.0.iter()
    }

    pub fn set_ascii(&mut self, key: &str, val: &str) {
        self.0
            .insert(key.to_string(), CqlValue::Ascii(val.to_owned()));
    }

    pub fn set_list_f32(&mut self, key: &str, val: &Vec<f32>) {
        let mut list: Vec<CqlValue> = Vec::with_capacity(val.len());
        for v in val {
            list.push(CqlValue::Float(*v));
        }

        self.0.insert(key.to_string(), CqlValue::List(list));
    }

    pub fn append_map_i32(&mut self, map_name: &str, key: &str, val: i32) {
        let mut map: Vec<(CqlValue, CqlValue)> = Vec::new();

        map.push((CqlValue::Ascii(key.to_string()), CqlValue::Int(val)));

        if let Some(old) = self.0.get(map_name) {
            for val in old.as_map().unwrap() {
                if val.0.as_ascii().unwrap() != key {
                    map.push((val.0.clone(), val.1.clone()));
                }
            }
        }
        self.0.insert(map_name.to_string(), CqlValue::Map(map));
    }

    pub fn set_in_cbor<T: ?Sized + Serialize>(&mut self, key: &str, val: &T) -> anyhow::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        ciborium::into_writer(val, &mut buf)?;
        self.0.insert(key.to_string(), CqlValue::Blob(buf));
        Ok(())
    }

    pub fn fill(&mut self, row: Row, fields: Vec<&str>) -> anyhow::Result<()> {
        if row.columns.len() != fields.len() {
            return Err(anyhow::Error::new(HTTPError {
                code: 500,
                message: format!(
                    "ColumnsMap::fill: row.columns.len({}) != fields.len({})",
                    row.columns.len(),
                    fields.len()
                ),
                data: None,
            }));
        }
        for (i, val) in row.columns.iter().enumerate() {
            if let Some(v) = val {
                self.0.insert(fields[i].to_owned(), v.to_owned());
            }
        }
        Ok(())
    }
}

// TODO https://docs.rs/scylla/latest/scylla/transport/errors/enum.QueryError.html
impl ToAnyhowError for QueryError {
    fn to_anyhow_error(self) -> anyhow::Error {
        match self {
            QueryError::DbError(dberr, msg) => anyhow::Error::new(HTTPError {
                code: 500,
                message: msg,
                data: Some(serde_json::Value::String(dberr.to_string())),
            }),
            _ => anyhow::Error::new(HTTPError {
                code: 500,
                message: self.to_string(),
                data: None,
            }),
        }
    }
}

impl ToAnyhowError for SingleRowError {
    fn to_anyhow_error(self) -> anyhow::Error {
        anyhow::Error::new(HTTPError {
            code: 404,
            message: self.to_string(),
            data: None,
        })
    }
}

impl ToAnyhowError for FromCqlValError {
    fn to_anyhow_error(self) -> anyhow::Error {
        anyhow::Error::new(HTTPError {
            code: 422,
            message: self.to_string(),
            data: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_map_works() {
        let mut map = ColumnsMap::new();

        assert_eq!(map.len(), 0);
        assert!(!map.has("user"));
        assert_eq!(map.get("user"), None);
        assert_eq!(
            map.get_as::<String>("user"),
            Err(FromCqlValError::ValIsNull)
        );

        map.set_ascii("user", "jarvis");
        assert_eq!(map.len(), 1);
        assert!(map.has("user"));
        assert_eq!(
            map.get("user"),
            Some(&CqlValue::Ascii("jarvis".to_string()))
        );
        assert_eq!(map.get_as::<String>("user"), Ok("jarvis".to_string()));

        map.set_ascii("user", "jarvis2");
        assert_eq!(map.len(), 1);
        assert!(map.has("user"));
        assert_eq!(
            map.get("user"),
            Some(&CqlValue::Ascii("jarvis2".to_string()))
        );
        assert_eq!(map.get_as::<String>("user"), Ok("jarvis2".to_string()));

        assert!(!map.has("embeddings"));
        assert_eq!(map.get("embeddings"), None);
        map.set_list_f32("embeddings", &vec![0.1f32, 0.2f32]);
        assert!(map.has("embeddings"));
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get_as::<Vec<f32>>("embeddings"),
            Ok(vec![0.1f32, 0.2f32]),
        );

        assert!(!map.has("tokens"));
        assert_eq!(map.get("tokens"), None);
        map.append_map_i32("tokens", "ada2", 999);
        assert!(map.has("tokens"));
        assert_eq!(map.len(), 3);
        assert_eq!(
            map.get_as::<BTreeMap<String, i32>>("tokens"),
            Ok(BTreeMap::from([("ada2".to_string(), 999i32)]))
        );

        map.append_map_i32("tokens", "gpt4", 1999);
        assert_eq!(map.len(), 3);
        assert_eq!(
            map.get_as::<BTreeMap<String, i32>>("tokens"),
            Ok(BTreeMap::from([
                ("ada2".to_string(), 999i32),
                ("gpt4".to_string(), 1999i32)
            ]))
        );

        map.append_map_i32("tokens", "ada2", 1999);
        assert_eq!(map.len(), 3);
        assert_eq!(
            map.get_as::<BTreeMap<String, i32>>("tokens"),
            Ok(BTreeMap::from([
                ("ada2".to_string(), 1999i32),
                ("gpt4".to_string(), 1999i32)
            ]))
        );

        assert!(!map.has("cbor"));
        assert_eq!(map.get("cbor"), None);
        assert_eq!(
            map.get_as::<Vec<u8>>("cbor"),
            Err(FromCqlValError::ValIsNull)
        );
        assert!(map.set_in_cbor("cbor", &vec![1i64, 2i64, 3i64]).is_ok()); // CBOR: 0x83010203
        assert!(map.has("cbor"));
        assert_eq!(map.len(), 4);
        assert_eq!(
            map.get_as::<Vec<u8>>("cbor"),
            Ok(vec![0x83, 0x01, 0x02, 0x03])
        );
        assert_eq!(
            map.get_as::<String>("cbor"),
            Err(FromCqlValError::BadCqlType)
        );

        let mut row: Row = Row {
            columns: Vec::new(),
        };

        let mut fields: Vec<&str> = Vec::new();
        for (k, v) in map.iter() {
            fields.push(k);
            row.columns.push(Some(v.to_owned()));
        }

        assert_eq!(fields.len(), 4);
        let mut map2 = ColumnsMap::new();
        assert!(map2
            .fill(
                Row {
                    columns: Vec::new(),
                },
                fields.clone()
            )
            .is_err());
        assert_ne!(map2, map);

        assert!(map2.fill(row, fields).is_ok());
        assert_eq!(map2, map);
    }
}
