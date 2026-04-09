//! Error types for riallm

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RiallmError>;

#[derive(Error, Debug)]
pub enum RiallmError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(String),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("HuggingFace Hub error: {0}")]
    Hub(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),
    
    #[error("Progress bar error: {0}")]
    ProgressBar(#[from] std::num::TryFromIntError),
    
    #[error("CString error: {0}")]
    CString(#[from] std::ffi::NulError),
}

impl From<&str> for RiallmError {
    fn from(s: &str) -> Self {
        RiallmError::ModelLoading(s.to_string())
    }
}

impl From<String> for RiallmError {
    fn from(s: String) -> Self {
        RiallmError::ModelLoading(s)
    }
}

impl From<safetensors::SafeTensorError> for RiallmError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        RiallmError::Safetensors(e.to_string())
    }
}

impl From<hf_hub::api::sync::ApiError> for RiallmError {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        RiallmError::Hub(e.to_string())
    }
}
