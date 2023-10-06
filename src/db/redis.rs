use async_trait::async_trait;
use rustis::bb8::{CustomizeConnection, ErrorSink, Pool};
use rustis::{
    client::{Config, PooledClientManager, ServerConfig},
    commands::{SetCondition, SetExpiration, StringCommands},
    resp::{BulkString, Command, RespBuf},
};
use tokio::time::Duration;

use crate::conf;

pub struct Redis {
    pool: Pool<PooledClientManager>,
}

impl Redis {
    pub async fn new(cfg: conf::Redis) -> anyhow::Result<Self> {
        let config = Config {
            server: ServerConfig::Standalone {
                host: cfg.host,
                port: cfg.port,
            },
            username: Some(cfg.username).filter(|s| !s.is_empty()),
            password: Some(cfg.password).filter(|s| !s.is_empty()),
            connect_timeout: Duration::from_secs(3),
            command_timeout: Duration::from_millis(1000),
            keep_alive: Some(Duration::from_secs(600)),
            ..Config::default()
        };

        let max_size = if cfg.max_connections > 0 {
            cfg.max_connections as u32
        } else {
            10
        };
        let min_idle = if max_size <= 10 { 1 } else { max_size / 10 };

        let manager = PooledClientManager::new(config).unwrap();
        let pool = Pool::builder()
            .max_size(max_size)
            .min_idle(Some(min_idle))
            .max_lifetime(None)
            .idle_timeout(Some(Duration::from_secs(600)))
            .connection_timeout(Duration::from_secs(3))
            .error_sink(Box::new(RedisMonitor {}))
            .connection_customizer(Box::new(RedisMonitor {}))
            .build(manager)
            .await?;
        Ok(Redis { pool })
    }

    pub async fn send(
        &self,
        command: Command,
        retry_on_error: Option<bool>,
    ) -> anyhow::Result<RespBuf> {
        let conn = self.pool.get().await?;
        let res = conn.send(command, retry_on_error).await?;
        Ok(res)
    }

    pub async fn new_data(&self, key: &str, value: Vec<u8>, ttl_ms: u64) -> anyhow::Result<bool> {
        let conn = self.pool.get().await?;
        let res = conn
            .set_with_options(
                key,
                value,
                SetCondition::NX,
                SetExpiration::Px(ttl_ms),
                false,
            )
            .await?;
        Ok(res)
    }

    pub async fn update_data(&self, key: &str, value: Vec<u8>) -> anyhow::Result<bool> {
        let conn = self.pool.get().await?;
        let res = conn
            .set_with_options(key, value, SetCondition::XX, SetExpiration::None, true)
            .await?;
        Ok(res)
    }

    pub async fn get_data(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        let conn = self.pool.get().await?;
        let res: Option<BulkString> = conn.get(key).await?;
        match res {
            Some(data) => Ok(data.to_vec()),
            None => Err(anyhow::anyhow!("key {:?} not found", key)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RedisMonitor;

impl<E: std::fmt::Display> ErrorSink<E> for RedisMonitor {
    fn sink(&self, error: E) {
        log::error!(target: "redis", "{}", error);
    }

    fn boxed_clone(&self) -> Box<dyn ErrorSink<E>> {
        Box::new(*self)
    }
}

#[async_trait]
impl<C: Send + 'static, E: 'static> CustomizeConnection<C, E> for RedisMonitor {
    async fn on_acquire(&self, _connection: &mut C) -> Result<(), E> {
        log::info!(target: "redis", "connection acquired");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rustis::resp;

    use super::*;

    #[tokio::test]
    async fn redis_pool_works() -> anyhow::Result<()> {
        let cli = Redis::new(conf::Redis {
            host: "127.0.0.1".to_string(),
            port: 6379,
            username: String::new(),
            password: String::new(),
            max_connections: 10,
        })
        .await?;

        let data = cli.send(resp::cmd("PING"), None).await?;
        assert_eq!("PONG", data.to::<String>()?);

        Ok(())
    }
}
