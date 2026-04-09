//! riallm - Rust implementation of AirLLM
//!
//! A memory-optimized inference engine for large language models that enables
//! running 70B+ parameter models on consumer GPUs with limited VRAM.
//!
//! # Core Features
//! - Layer-by-layer model loading to minimize GPU memory usage
//! - Support for multiple model architectures (Llama, Qwen, Mistral, etc.)
//! - Model splitting and safetensors persistence
//! - Optional quantization (4-bit NF4, 8-bit)
//! - Automatic model type detection
//!
//! # Example
//! ```no_run
//! use riallm::AutoModel;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = AutoModel::from_pretrained("meta-llama/Llama-2-7b-hf", None).await?;
//!     // Use model for inference...
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod model;
pub mod persistence;
pub mod quantization;
pub mod utils;
pub mod profiler;
pub mod auto_model;
pub mod error;
pub mod adapters;
pub mod memory;

// Re-export main types for convenience
pub use auto_model::AutoModel;
pub use config::ModelConfig;
pub use error::Result;
pub use model::AirLLMBaseModel;
pub use persistence::SafetensorModelPersister;
pub use adapters::ModelAdapter;
