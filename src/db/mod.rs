mod model_counter;
mod model_embedding;
mod model_translating;

pub mod scylladb;

pub use model_counter::Counter;
pub use model_embedding::Embedding;
pub use model_translating::Translating;

pub trait ToAnyhowError {
    fn to_anyhow_error(self) -> anyhow::Error;
}
