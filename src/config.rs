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
            return Err(RiallmError::Config(format!(
                "config.json not found in {:?}",
                model_path
            )));
        }

        let config_str = std::fs::read_to_string(&config_file)?;
        let value: serde_json::Value = serde_json::from_str(&config_str)?;

        Self::from_value(value, model_path)
    }

    /// Build a model configuration from a HuggingFace config.json value.
    ///
    /// Newer multimodal configs, including Qwen3.6, keep the decoder config in
    /// `text_config`; older text-only configs keep the same fields at top level.
    pub fn from_value(value: serde_json::Value, model_path: PathBuf) -> Result<Self> {
        let model_type = value
            .get("model_type")
            .and_then(|val| val.as_str())
            .map(ToString::to_string)
            .or_else(|| {
                Self::config_value(&value, "model_type")
                    .and_then(|val| val.as_str())
                    .map(ToString::to_string)
            })
            .ok_or_else(|| RiallmError::Config("Missing string field: model_type".to_string()))?;
        let vocab_size = Self::usize_field(&value, "vocab_size")?;
        let hidden_size = Self::usize_field(&value, "hidden_size")?;
        let num_hidden_layers = Self::usize_field(&value, "num_hidden_layers")?;
        let num_attention_heads = Self::usize_field(&value, "num_attention_heads")?;
        let num_key_value_heads = Self::optional_usize_field(&value, "num_key_value_heads");
        let intermediate_size = Self::usize_field_any(
            &value,
            &[
                "intermediate_size",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ],
        )?;
        let max_position_embeddings =
            Self::usize_field_any(&value, &["max_position_embeddings", "seq_length"])?;
        let rms_norm_eps = Self::f32_field_any(&value, &["rms_norm_eps", "layer_norm_epsilon"])?;
        let rope_theta = Self::optional_f32_field(&value, "rope_theta").or_else(|| {
            Self::config_value(&value, "rope_parameters")
                .and_then(|rope| rope.get("rope_theta"))
                .and_then(|theta| theta.as_f64())
                .map(|theta| theta as f32)
        });
        let sliding_window = Self::optional_usize_field(&value, "sliding_window");

        let mut extra = HashMap::new();
        if let Some(root) = value.as_object() {
            for (key, val) in root {
                extra.insert(key.clone(), val.clone());
            }
        }
        if let Some(text_config) = value.get("text_config").and_then(|v| v.as_object()) {
            for (key, val) in text_config {
                extra.entry(key.clone()).or_insert_with(|| val.clone());
            }
        }

        Ok(Self {
            model_type,
            model_path,
            split_path: None,
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            sliding_window,
            extra,
        })
    }

    /// Get the number of key-value heads (defaults to num_attention_heads)
    pub fn get_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if model uses grouped query attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads.is_some()
            && self.num_key_value_heads.unwrap() != self.num_attention_heads
    }

    /// Whether this config is for Qwen3.5/Qwen3.6 MoE text decoder weights.
    pub fn is_qwen3_5_moe(&self) -> bool {
        self.model_type.to_lowercase().contains("qwen3_5_moe")
    }

    fn config_value<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
        value
            .get("text_config")
            .and_then(|text| text.get(key))
            .or_else(|| value.get(key))
    }

    fn usize_field(value: &serde_json::Value, key: &str) -> Result<usize> {
        Self::optional_usize_field(value, key)
            .ok_or_else(|| RiallmError::Config(format!("Missing integer field: {}", key)))
    }

    fn usize_field_any(value: &serde_json::Value, keys: &[&str]) -> Result<usize> {
        keys.iter()
            .find_map(|key| Self::optional_usize_field(value, key))
            .ok_or_else(|| {
                RiallmError::Config(format!(
                    "Missing integer field; expected one of: {}",
                    keys.join(", ")
                ))
            })
    }

    fn optional_usize_field(value: &serde_json::Value, key: &str) -> Option<usize> {
        Self::config_value(value, key)
            .and_then(|val| val.as_u64())
            .and_then(|val| usize::try_from(val).ok())
    }

    fn f32_field_any(value: &serde_json::Value, keys: &[&str]) -> Result<f32> {
        keys.iter()
            .find_map(|key| Self::optional_f32_field(value, key))
            .ok_or_else(|| {
                RiallmError::Config(format!(
                    "Missing float field; expected one of: {}",
                    keys.join(", ")
                ))
            })
    }

    fn optional_f32_field(value: &serde_json::Value, key: &str) -> Option<f32> {
        Self::config_value(value, key)
            .and_then(|val| val.as_f64())
            .map(|val| val as f32)
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
            "qwen3_5_moe"
            | "qwen3_5moe"
            | "qwen3_5_moe_text"
            | "qwen3_5moeforconditionalgeneration"
            | "qwen3_5_moeforconditionalgeneration" => Ok(Self {
                embed: "model.language_model.embed_tokens".to_string(),
                layer_prefix: "model.language_model.layers.".to_string(),
                norm: "model.language_model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                use_post_attention_layernorm: false,
                rotary_dim: Some(64),
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
            _ => Err(RiallmError::Config(format!(
                "Unsupported architecture: {}",
                arch
            ))),
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
            _ => Err(RiallmError::Config(format!(
                "Invalid compression type: {}. Use 'none', '4bit', or '8bit'",
                s
            ))),
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
