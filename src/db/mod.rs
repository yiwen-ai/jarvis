mod model_embedding;
mod model_summarizing;
mod model_translating;

pub mod qdrant;
pub mod scylladb;

pub use model_embedding::Embedding;
pub use model_summarizing::Summarizing;
pub use model_translating::Translating;

pub static USER_JARVIS: &str = "0000000000000jarvis0"; // system user
