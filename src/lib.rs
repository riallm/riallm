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

pub mod adapters;
pub mod auto_model;
pub mod config;
pub mod error;
pub mod memory;
pub mod model;
pub mod persistence;
pub mod profiler;
pub mod quantization;
pub mod utils;

// Re-export main types for convenience
pub use adapters::ModelAdapter;
pub use auto_model::AutoModel;
pub use config::ModelConfig;
pub use error::Result;
pub use model::AirLLMBaseModel;
pub use persistence::SafetensorModelPersister;
