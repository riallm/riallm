//! Model configuration and layer specifications

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{Result, RiallmError};

/// Configuration for a language model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type (e.g., "llama", "qwen", "mistral")
    pub model_type: String,
    
    /// Path to model files (local or cache)
    pub model_path: PathBuf,
    
    /// Path to split layer shards (if already split)
    pub split_path: Option<PathBuf>,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Hidden layer dimension
    pub hidden_size: usize,
    
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of key-value heads (for GQA/MQA)
    pub num_key_value_heads: Option<usize>,
    
    /// Intermediate layer dimension (FFN)
    pub intermediate_size: usize,
    
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    
    /// RMS Norm epsilon
    pub rms_norm_eps: f32,
    
    /// Rope theta (for rotary embeddings)
    pub rope_theta: Option<f32>,
    
    /// Whether to use sliding window attention
    pub sliding_window: Option<usize>,
    
    /// Additional configuration parameters
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Load configuration from a model directory
    pub fn from_path(model_path: PathBuf) -> Result<Self> {
        let config_file = model_path.join("config.json");
        
        if !config_file.exists() {
            return Err(RiallmError::Config(
                format!("config.json not found in {:?}", model_path)
            ));
        }
        
        let config_str = std::fs::read_to_string(&config_file)?;
        let mut config: ModelConfig = serde_json::from_str(&config_str)?;
        config.model_path = model_path;
        
        Ok(config)
    }
    
    /// Get the number of key-value heads (defaults to num_attention_heads)
    pub fn get_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    
    /// Check if model uses grouped query attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads.is_some() && 
        self.num_key_value_heads.unwrap() != self.num_attention_heads
    }
}

/// Layer name mappings for different model architectures
#[derive(Debug, Clone)]
pub struct LayerNames {
    /// Embedding layer name pattern
    pub embed: String,
    
    /// Layer prefix (e.g., "model.layers." for Llama)
    pub layer_prefix: String,
    
    /// Final layer norm name
    pub norm: String,
    
    /// LM head name
    pub lm_head: String,
    
    /// Whether to use post-attention layer norm
    pub use_post_attention_layernorm: bool,
    
    /// Rotary embedding dimension (if applicable)
    pub rotary_dim: Option<usize>,
}

impl LayerNames {
    /// Get layer names for a specific architecture
    pub fn for_arch(arch: &str) -> Result<Self> {
        match arch.to_lowercase().as_str() {
            "llama" | "llamaforcausallm" => Ok(Self {
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "qwen" | "qwenforcausallm" => Ok(Self {
                embed: "transformer.wte".to_string(),
                layer_prefix: "transformer.h.".to_string(),
                norm: "transformer.ln_f".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "qwen2" | "qwen2forcausallm" => Ok(Self {
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "mistral" | "mistralforcausallm" => Ok(Self {
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "mixtral" | "mixtralforcausallm" => Ok(Self {
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "chatglm" | "chatglmforcausallm" => Ok(Self {
                embed: "transformer.embedding.word_embeddings".to_string(),
                layer_prefix: "transformer.encoder.layers.".to_string(),
                norm: "transformer.encoder.final_layernorm".to_string(),
                lm_head: "transformer.output_layer".to_string(),
                use_post_attention_layernorm: true,
                rotary_dim: None,
            }),
            "baichuan" | "baichuanforcausallm" => Ok(Self {
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            "internlm" | "internlmforcausallm" => Ok(Self {
                embed: "model.tok_embeddings".to_string(),
                layer_prefix: "model.layers.".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "output".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: None,
            }),
            _ => Err(RiallmError::Config(
                format!("Unsupported architecture: {}", arch)
            )),
        }
    }
    
    /// Generate the full layer name for a specific layer index
    pub fn layer_name(&self, index: usize, suffix: &str) -> String {
        format!("{}{}{}", self.layer_prefix, index, suffix)
    }
}

/// Compression type for model quantization
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionType {
    /// No compression
    None,
    
    /// 4-bit NF4 quantization
    FourBit,
    
    /// 8-bit block-wise quantization
    EightBit,
}

impl CompressionType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" | "" => Ok(CompressionType::None),
            "4bit" => Ok(CompressionType::FourBit),
            "8bit" => Ok(CompressionType::EightBit),
            _ => Err(RiallmError::Config(
                format!("Invalid compression type: {}. Use 'none', '4bit', or '8bit'", s)
            )),
        }
    }
}

/// Device specification for model placement
#[derive(Debug, Clone)]
pub enum DeviceSpec {
    /// CPU device
    Cpu,
    
    /// CUDA device with optional device ID
    Cuda(usize),
    
    /// Metal device (Apple Silicon)
    Metal,
}

impl DeviceSpec {
    pub fn is_cuda(&self) -> bool {
        matches!(self, DeviceSpec::Cuda(_))
    }
    
    pub fn device_id(&self) -> Option<usize> {
        match self {
            DeviceSpec::Cuda(id) => Some(*id),
            _ => None,
        }
    }
}

/// Options for model loading
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Compression type
    pub compression: CompressionType,
    
    /// Device to use for inference
    pub device: DeviceSpec,
    
    /// Maximum sequence length
    pub max_seq_len: Option<usize>,
    
    /// Enable profiling
    pub profiling_mode: bool,
    
    /// Enable async prefetching
    pub prefetch_layers: bool,
    
    /// Number of layers to prefetch ahead
    pub prefetch_buffer_size: usize,
    
    /// Data type for computation
    pub dtype: String,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            compression: CompressionType::None,
            device: DeviceSpec::Cuda(0),
            max_seq_len: None,
            profiling_mode: false,
            prefetch_layers: true,
            prefetch_buffer_size: 2,
            dtype: "float16".to_string(),
        }
    }
}
