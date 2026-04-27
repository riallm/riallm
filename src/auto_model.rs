//! AutoModel factory for automatic model type detection and instantiation

use std::path::PathBuf;

use crate::config::{LayerNames, ModelConfig, ModelOptions};
use crate::error::{Result, RiallmError};
use crate::model::AirLLMBaseModel;
use crate::utils::find_or_create_split_path;

/// Supported model architectures
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    Llama,
    Qwen,
    Qwen2,
    Qwen35Moe,
    Mistral,
    Mixtral,
    ChatGLM,
    Baichuan,
    InternLM,
    Unknown,
}

impl ModelArchitecture {
    /// Detect architecture from config
    pub fn from_config(config: &ModelConfig) -> Self {
        let arch = config.model_type.to_lowercase();

        if arch.contains("llama") {
            ModelArchitecture::Llama
        } else if arch.contains("qwen3_5_moe") {
            ModelArchitecture::Qwen35Moe
        } else if arch.contains("qwen2") {
            ModelArchitecture::Qwen2
        } else if arch.contains("qwen") {
            ModelArchitecture::Qwen
        } else if arch.contains("mistral") {
            ModelArchitecture::Mistral
        } else if arch.contains("mixtral") {
            ModelArchitecture::Mixtral
        } else if arch.contains("chatglm") {
            ModelArchitecture::ChatGLM
        } else if arch.contains("baichuan") {
            ModelArchitecture::Baichuan
        } else if arch.contains("internlm") {
            ModelArchitecture::InternLM
        } else {
            ModelArchitecture::Unknown
        }
    }

    /// Get architecture name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelArchitecture::Llama => "llama",
            ModelArchitecture::Qwen => "qwen",
            ModelArchitecture::Qwen2 => "qwen2",
            ModelArchitecture::Qwen35Moe => "qwen3_5_moe",
            ModelArchitecture::Mistral => "mistral",
            ModelArchitecture::Mixtral => "mixtral",
            ModelArchitecture::ChatGLM => "chatglm",
            ModelArchitecture::Baichuan => "baichuan",
            ModelArchitecture::InternLM => "internlm",
            ModelArchitecture::Unknown => "unknown",
        }
    }
}

/// AutoModel - automatically detects and loads the appropriate model
pub struct AutoModel;

impl AutoModel {
    /// Load a model from a pretrained source (HuggingFace model ID or local path)
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf") or local path
    /// * `options` - Optional model loading options
    ///
    /// # Returns
    /// Loaded AirLLMBaseModel ready for inference
    pub async fn from_pretrained(
        model_id: &str,
        options: Option<ModelOptions>,
    ) -> Result<AirLLMBaseModel> {
        let options = options.unwrap_or_default();

        // Check if model_id is a local path
        let model_path = if std::path::Path::new(model_id).exists() {
            PathBuf::from(model_id)
        } else {
            // TODO: Download from HuggingFace Hub
            return Err(RiallmError::ModelLoading(format!(
                "Model not found locally: '{}'. HuggingFace download not yet implemented.",
                model_id
            )));
        };

        // Load configuration
        let config = ModelConfig::from_path(model_path.clone())?;

        // Detect architecture
        let arch = ModelArchitecture::from_config(&config);

        // Get layer name mappings
        let layer_names = LayerNames::for_arch(arch.as_str())?;

        // Find or create split path
        let split_path = find_or_create_split_path(model_id, None)?;

        // If model not split yet, split it
        if !split_path.exists() || !split_path.join("split_metadata.json").exists() {
            println!("Model not split yet. Splitting {}...", model_id);
            crate::utils::split_model(
                model_path.clone(),
                split_path.clone(),
                &layer_names,
                &config,
            )?;
        }

        // Create model
        let model = AirLLMBaseModel::new(config, layer_names, options, split_path)?;

        println!("Loaded {} model from {:?}", arch.as_str(), model_path);

        Ok(model)
    }

    /// Get the architecture class for a model type
    ///
    /// This maps architecture names to the appropriate AirLLM implementation
    pub fn get_model_class(architectures: &[String]) -> Result<ModelArchitecture> {
        if architectures.is_empty() {
            return Err(RiallmError::Config(
                "No architectures specified in config".to_string(),
            ));
        }

        let arch = &architectures[0];

        if arch.contains("Qwen3_5Moe") || arch.contains("Qwen3_5_Moe") {
            Ok(ModelArchitecture::Qwen35Moe)
        } else if arch.contains("Llama") || arch.contains("Mistral") {
            Ok(ModelArchitecture::Llama)
        } else if arch.contains("Qwen2") {
            Ok(ModelArchitecture::Qwen2)
        } else if arch.contains("Qwen") {
            Ok(ModelArchitecture::Qwen)
        } else if arch.contains("Mistral") {
            Ok(ModelArchitecture::Mistral)
        } else if arch.contains("Mixtral") {
            Ok(ModelArchitecture::Mixtral)
        } else if arch.contains("ChatGLM") {
            Ok(ModelArchitecture::ChatGLM)
        } else if arch.contains("Baichuan") {
            Ok(ModelArchitecture::Baichuan)
        } else if arch.contains("InternLM") {
            Ok(ModelArchitecture::InternLM)
        } else {
            Err(RiallmError::Config(format!(
                "Unsupported architecture: {}",
                arch
            )))
        }
    }
}

/// Split a model if not already split
pub fn split_model(
    model_path: PathBuf,
    split_path: PathBuf,
    layer_names: &LayerNames,
    config: &ModelConfig,
) -> Result<()> {
    use crate::utils::ModelSplitter;

    let splitter = ModelSplitter::new(model_path, split_path, layer_names.clone(), config.clone());

    splitter.split()
}
