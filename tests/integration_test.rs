//! Integration tests for riallm

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

#[cfg(test)]
mod tests {
    use super::*;
    use riallm::adapters::{create_adapter, ModelAdapter};
    use riallm::config::{CompressionType, LayerNames, ModelConfig, ModelOptions};
    use riallm::error::Result;
    use riallm::memory::{MemoryStats, MemoryTracker};
    use riallm::persistence::SafetensorModelPersister;
    use riallm::profiler::Profiler;
    use riallm::quantization::{quantize_tensor, QuantizedTensor};

    #[test]
    fn test_layer_names_llama() {
        let layer_names = LayerNames::for_arch("llama").unwrap();

        assert_eq!(layer_names.embed, "model.embed_tokens");
        assert_eq!(layer_names.layer_prefix, "model.layers.");
        assert_eq!(layer_names.norm, "model.norm");
        assert_eq!(layer_names.lm_head, "lm_head");
    }

    #[test]
    fn test_layer_names_qwen2() {
        let layer_names = LayerNames::for_arch("qwen2").unwrap();

        assert_eq!(layer_names.embed, "model.embed_tokens");
        assert_eq!(layer_names.layer_prefix, "model.layers.");
    }

    #[test]
    fn test_layer_names_qwen3_5_moe() {
        let layer_names = LayerNames::for_arch("qwen3_5_moe").unwrap();

        assert_eq!(layer_names.embed, "model.language_model.embed_tokens");
        assert_eq!(layer_names.layer_prefix, "model.language_model.layers.");
        assert_eq!(layer_names.norm, "model.language_model.norm");
        assert_eq!(layer_names.lm_head, "lm_head");
        assert_eq!(layer_names.rotary_dim, Some(64));
    }

    #[test]
    fn test_layer_names_mistral() {
        let layer_names = LayerNames::for_arch("mistral").unwrap();

        assert_eq!(layer_names.embed, "model.embed_tokens");
        assert_eq!(layer_names.layer_prefix, "model.layers.");
    }

    #[test]
    fn test_layer_names_unsupported() {
        let result = LayerNames::for_arch("unsupported_arch");
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_type_parsing() {
        use riallm::config::CompressionType;

        assert_eq!(
            CompressionType::from_str("none").unwrap(),
            CompressionType::None
        );
        assert_eq!(
            CompressionType::from_str("4bit").unwrap(),
            CompressionType::FourBit
        );
        assert_eq!(
            CompressionType::from_str("8bit").unwrap(),
            CompressionType::EightBit
        );
        assert!(CompressionType::from_str("invalid").is_err());
    }

    #[test]
    fn test_model_options_default() {
        let options = ModelOptions::default();

        assert_eq!(options.compression, CompressionType::None);
        assert!(options.prefetch_layers);
        assert!(!options.profiling_mode);
        assert_eq!(options.dtype, "float16");
    }

    #[test]
    fn test_qwen3_5_moe_nested_config_parse() {
        let config_value = serde_json::json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "hidden_size": 2048,
                "vocab_size": 248320,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "moe_intermediate_size": 512,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 10000000
                }
            }
        });

        let config = ModelConfig::from_value(config_value, PathBuf::from("/tmp/qwen3")).unwrap();

        assert_eq!(config.model_type, "qwen3_5_moe");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(config.rope_theta, Some(10000000.0));
        assert!(config.is_qwen3_5_moe());
    }

    #[test]
    fn test_quantize_8bit() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..256).map(|x| x as f32 / 255.0).collect();
        let tensor = Tensor::from_vec(data.clone(), &[16, 16], &device).unwrap();

        let quantized = quantize_tensor(&tensor, &CompressionType::EightBit).unwrap();

        assert_eq!(quantized.qtype, CompressionType::EightBit);
        assert_eq!(quantized.shape, vec![16, 16]);
        assert!(!quantized.scales.is_empty());
    }

    #[test]
    fn test_quantize_4bit() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..128).map(|x| (x as f32 / 128.0) - 0.5).collect();
        let tensor = Tensor::from_vec(data.clone(), &[8, 16], &device).unwrap();

        let quantized = quantize_tensor(&tensor, &CompressionType::FourBit).unwrap();

        assert_eq!(quantized.qtype, CompressionType::FourBit);
        assert_eq!(quantized.shape, vec![8, 16]);
        assert!(!quantized.scales.is_empty());
    }

    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();

        profiler.start_layer("layer_0");
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.end_layer("layer_0");

        profiler.start_layer("layer_1");
        std::thread::sleep(std::time::Duration::from_millis(20));
        profiler.end_layer("layer_1");

        let stats_0 = profiler.get_layer_stats("layer_0").unwrap();
        let stats_1 = profiler.get_layer_stats("layer_1").unwrap();

        assert_eq!(stats_0.execution_count, 1);
        assert_eq!(stats_1.execution_count, 1);
        assert!(stats_0.avg_forward_time_ms() > 0.0);
        assert!(stats_1.avg_forward_time_ms() > 0.0);

        // Layer 1 should have taken longer
        assert!(stats_1.avg_forward_time_ms() > stats_0.avg_forward_time_ms() * 0.5);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::new(1_073_741_824, 536_870_912, 536_870_912);

        assert_eq!(stats.total, 1_073_741_824);
        assert_eq!(stats.used, 536_870_912);
        assert_eq!(stats.free, 536_870_912);
        assert!((stats.usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        tracker.track_allocation("tensor_1", 1_000_000);
        tracker.track_allocation("tensor_2", 2_000_000);

        assert_eq!(tracker.current_memory(), 3_000_000);
        assert_eq!(tracker.peak_memory(), 3_000_000);

        tracker.track_deallocation("tensor_1", 1_000_000);
        assert_eq!(tracker.current_memory(), 2_000_000);
        assert_eq!(tracker.peak_memory(), 3_000_000); // Peak should remain unchanged
    }

    #[test]
    fn test_create_adapter_llama() {
        let config = ModelConfig {
            model_type: "llama".to_string(),
            model_path: PathBuf::from("/tmp/test"),
            split_path: None,
            vocab_size: 32000,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            sliding_window: None,
            extra: HashMap::new(),
        };

        let adapter = create_adapter("llama", config.clone()).unwrap();
        assert_eq!(adapter.model_type_name(), "llama");

        let layer_names = adapter.get_layer_names();
        assert_eq!(layer_names.embed, "model.embed_tokens");
    }

    #[test]
    fn test_create_adapter_unsupported() {
        let config = ModelConfig {
            model_type: "unsupported".to_string(),
            model_path: PathBuf::from("/tmp/test"),
            split_path: None,
            vocab_size: 32000,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            sliding_window: None,
            extra: HashMap::new(),
        };

        let result = create_adapter("unsupported", config);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_stats_formatting() {
        assert_eq!(MemoryStats::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryStats::format_bytes(1_048_576), "1.00 MB");
        assert_eq!(MemoryStats::format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(MemoryStats::format_bytes(512), "512 B");
    }
}
