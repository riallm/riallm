//! Model-specific adapters for different architectures

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

use crate::config::{LayerNames, ModelConfig, ModelOptions};
use crate::error::Result;
use crate::model::AirLLMBaseModel;

/// Trait for model-specific customizations
pub trait ModelAdapter: Send + Sync {
    /// Get model-specific layer names
    fn get_layer_names(&self) -> LayerNames;

    /// Prepare position IDs for the model
    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// Prepare attention mask arguments
    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>>;

    /// Prepare position embedding arguments
    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>>;

    /// Get model type name
    fn model_type_name(&self) -> &str;
}

/// Llama model adapter
pub struct LlamaAdapter {
    config: ModelConfig,
}

impl LlamaAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for LlamaAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("llama").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        // Llama uses causal mask by default during generation
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "llama"
    }
}

/// Qwen model adapter
pub struct QwenAdapter {
    config: ModelConfig,
}

impl QwenAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for QwenAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("qwen").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "qwen"
    }
}

/// Qwen2 model adapter
pub struct Qwen2Adapter {
    config: ModelConfig,
}

impl Qwen2Adapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for Qwen2Adapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("qwen2").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "qwen2"
    }
}

/// Qwen3.5/Qwen3.6 MoE adapter
pub struct Qwen35MoeAdapter {
    config: ModelConfig,
}

impl Qwen35MoeAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for Qwen35MoeAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("qwen3_5_moe").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        let _ = seq_len;
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "qwen3_5_moe"
    }
}

/// Mistral model adapter
pub struct MistralAdapter {
    config: ModelConfig,
}

impl MistralAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for MistralAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("mistral").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "mistral"
    }
}

/// Mixtral model adapter (Mixture of Experts)
pub struct MixtralAdapter {
    config: ModelConfig,
}

impl MixtralAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for MixtralAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("mixtral").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "mixtral"
    }
}

/// ChatGLM model adapter
pub struct ChatGLMAdapter {
    config: ModelConfig,
}

impl ChatGLMAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for ChatGLMAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("chatglm").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        // ChatGLM has specific attention mask format
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "chatglm"
    }
}

/// Baichuan model adapter
pub struct BaichuanAdapter {
    config: ModelConfig,
}

impl BaichuanAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for BaichuanAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("baichuan").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "baichuan"
    }
}

/// InternLM model adapter
pub struct InternLMAdapter {
    config: ModelConfig,
}

impl InternLMAdapter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelAdapter for InternLMAdapter {
    fn get_layer_names(&self) -> LayerNames {
        LayerNames::for_arch("internlm").unwrap()
    }

    fn prepare_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        let device = input_ids.device();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    fn prepare_attention_mask_args(
        &self,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(attention_mask.cloned())
    }

    fn prepare_position_embedding_args(
        &self,
        position_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut args = HashMap::new();
        args.insert("position_ids".to_string(), position_ids.clone());
        Ok(args)
    }

    fn model_type_name(&self) -> &str {
        "internlm"
    }
}

/// Create a model adapter from architecture name
pub fn create_adapter(arch: &str, config: ModelConfig) -> Result<Box<dyn ModelAdapter>> {
    match arch.to_lowercase().as_str() {
        "llama" | "llamaforcausallm" => Ok(Box::new(LlamaAdapter::new(config))),
        "qwen" | "qwenforcausallm" => Ok(Box::new(QwenAdapter::new(config))),
        "qwen2" | "qwen2forcausallm" => Ok(Box::new(Qwen2Adapter::new(config))),
        "qwen3_5_moe"
        | "qwen3_5_moe_text"
        | "qwen3_5moe"
        | "qwen3_5moeforconditionalgeneration"
        | "qwen3_5_moeforconditionalgeneration" => Ok(Box::new(Qwen35MoeAdapter::new(config))),
        "mistral" | "mistralforcausallm" => Ok(Box::new(MistralAdapter::new(config))),
        "mixtral" | "mixtralforcausallm" => Ok(Box::new(MixtralAdapter::new(config))),
        "chatglm" | "chatglmforcausallm" => Ok(Box::new(ChatGLMAdapter::new(config))),
        "baichuan" | "baichuanforcausallm" => Ok(Box::new(BaichuanAdapter::new(config))),
        "internlm" | "internlmforcausallm" => Ok(Box::new(InternLMAdapter::new(config))),
        _ => Err(crate::error::RiallmError::Config(format!(
            "Unsupported architecture: {}",
            arch
        ))),
    }
}
