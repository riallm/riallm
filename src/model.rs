//! Core AirLLM model implementation with layer-by-layer loading

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor, D};

use crate::config::{CompressionType, DeviceSpec, LayerNames, ModelConfig, ModelOptions};
use crate::error::{Result, RiallmError};
use crate::persistence::{check_layers_exist, ModelPersister, SafetensorModelPersister};
use crate::profiler::Profiler;

/// KV Cache for efficient generation
#[derive(Debug, Clone)]
pub struct KVCACHE {
    /// Key cache per layer
    pub key_cache: Vec<Option<Tensor>>,

    /// Value cache per layer
    pub value_cache: Vec<Option<Tensor>>,

    /// Current sequence length
    pub seq_len: usize,
}

impl KVCACHE {
    pub fn new(num_layers: usize) -> Self {
        Self {
            key_cache: vec![None; num_layers],
            value_cache: vec![None; num_layers],
            seq_len: 0,
        }
    }

    pub fn update(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
    }
}

/// Layer state held in CPU memory between GPU passes
struct LayerState {
    /// Tensor weights for this layer
    tensors: HashMap<String, Tensor>,

    /// Whether this layer is currently loaded to GPU
    on_gpu: bool,
}

/// Core AirLLM model with memory-optimized inference
pub struct AirLLMBaseModel {
    /// Model configuration
    config: ModelConfig,

    /// Layer name mappings
    layer_names: LayerNames,

    /// Model loading options
    options: ModelOptions,

    /// Device for inference
    device: Device,

    /// CPU device
    cpu_device: Device,

    /// Path to split layer shards
    split_path: PathBuf,

    /// Model persister for loading layers
    persister: Box<dyn ModelPersister>,

    /// Layer states (CPU memory)
    layer_states: HashMap<String, LayerState>,

    /// Ordered list of layer names
    layer_order: Vec<String>,

    /// Profiler for timing/memory tracking
    profiler: Option<Profiler>,

    /// Tokenizer (handled externally)
    tokenizer_path: Option<PathBuf>,

    /// KV cache
    kv_cache: Option<KVCACHE>,

    /// CUDA stream for async operations (if applicable)
    #[allow(dead_code)]
    stream: Option<()>, // TODO: Implement CUDA streams in candle
}

impl AirLLMBaseModel {
    /// Create a new AirLLM model
    pub fn new(
        config: ModelConfig,
        layer_names: LayerNames,
        options: ModelOptions,
        split_path: PathBuf,
    ) -> Result<Self> {
        // Determine device
        let device = match &options.device {
            DeviceSpec::Cuda(id) => Device::new_cuda(*id)?,
            DeviceSpec::Cpu | DeviceSpec::Metal => Device::Cpu,
        };

        let cpu_device = Device::Cpu;

        // Create persister
        let persister = Box::new(SafetensorModelPersister::new(device.clone()));

        // Check if layers exist
        let all_layer_names = Self::generate_layer_names(&layer_names, config.num_hidden_layers);

        if !check_layers_exist(&split_path, &all_layer_names) {
            return Err(RiallmError::ModelLoading(format!(
                "Model layers not found at {:?}. Run model splitting first.",
                split_path
            )));
        }

        // Initialize layer states
        let mut layer_states = HashMap::new();
        let mut layer_order = Vec::new();

        // Add embedding layer
        layer_order.push("embed".to_string());
        layer_states.insert(
            "embed".to_string(),
            LayerState {
                tensors: HashMap::new(),
                on_gpu: false,
            },
        );

        // Add transformer layers
        for i in 0..config.num_hidden_layers {
            let layer_name = format!("layer_{}", i);
            layer_order.push(layer_name.clone());
            layer_states.insert(
                layer_name,
                LayerState {
                    tensors: HashMap::new(),
                    on_gpu: false,
                },
            );
        }

        // Add final norm
        layer_order.push("final_norm".to_string());
        layer_states.insert(
            "final_norm".to_string(),
            LayerState {
                tensors: HashMap::new(),
                on_gpu: false,
            },
        );

        // Add LM head
        layer_order.push("lm_head".to_string());
        layer_states.insert(
            "lm_head".to_string(),
            LayerState {
                tensors: HashMap::new(),
                on_gpu: false,
            },
        );

        let profiler = if options.profiling_mode {
            Some(Profiler::new())
        } else {
            None
        };

        let kv_cache = Some(KVCACHE::new(config.num_hidden_layers));

        Ok(Self {
            config,
            layer_names,
            options,
            device,
            cpu_device,
            split_path,
            persister,
            layer_states,
            layer_order,
            profiler,
            tokenizer_path: None,
            kv_cache,
            stream: None,
        })
    }

    /// Generate the full list of layer names
    fn generate_layer_names(layer_names: &LayerNames, num_layers: usize) -> Vec<String> {
        let mut names = Vec::new();

        // Embedding
        names.push("embed".to_string());

        // Transformer layers
        for i in 0..num_layers {
            names.push(format!("layer_{}", i));
        }

        // Final norm
        names.push("final_norm".to_string());

        // LM head
        names.push("lm_head".to_string());

        names
    }

    /// Forward pass with layer-by-layer loading
    ///
    /// This is the core algorithm that loads one layer at a time to minimize GPU memory
    pub fn forward(
        &mut self,
        hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut current_hidden = hidden_states.to_device(&self.cpu_device)?;

        // Track which layer is currently on GPU
        let mut current_gpu_layer: Option<String> = None;

        // Clone layer_order to avoid borrow conflicts
        let layer_order = self.layer_order.clone();

        // Process each layer sequentially
        for layer_name in &layer_order {
            if let Some(profiler) = &mut self.profiler {
                profiler.start_layer(layer_name);
            }

            // Unload previous layer if different
            if let Some(prev_layer) = &current_gpu_layer {
                if prev_layer != layer_name {
                    self.unload_layer_from_gpu(prev_layer)?;
                }
            }

            // Load current layer to GPU if not already loaded
            if !self.is_layer_on_gpu(layer_name) {
                self.load_layer_to_device(layer_name)?;
                current_gpu_layer = Some(layer_name.clone());
            }

            // Run forward pass for this layer
            current_hidden = self.forward_layer(layer_name, &current_hidden, attention_mask)?;

            if let Some(profiler) = &mut self.profiler {
                profiler.end_layer(layer_name);
            }

            // Clean memory after each layer
            self.clean_memory()?;
        }

        Ok(current_hidden)
    }

    /// Forward pass for a single layer
    fn forward_layer(
        &mut self,
        layer_name: &str,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let layer_state = self
            .layer_states
            .get(layer_name)
            .ok_or_else(|| RiallmError::LayerNotFound(layer_name.to_string()))?;

        match layer_name {
            "embed" => {
                // Embedding lookup: convert token IDs to hidden states
                if hidden_states.dims().len() == 2 {
                    // Input is token IDs [batch_size, seq_len]
                    let embed_weight = layer_state.tensors.get("weight").ok_or_else(|| {
                        RiallmError::ModelLoading("Embedding weight not found".to_string())
                    })?;

                    // Use candle's gather operation for embedding lookup
                    // embed_weight shape: [vocab_size, hidden_size]
                    // hidden_states shape: [batch_size, seq_len] (token IDs)
                    // Result shape: [batch_size, seq_len, hidden_size]
                    let hidden_size = embed_weight.dim(D::Minus1)?;

                    // Flatten token IDs, gather embeddings, reshape
                    let input_shape = hidden_states.shape();
                    let flat_ids = hidden_states.flatten_all()?;

                    // Gather rows from embedding matrix
                    let embedded = embed_weight.index_select(&flat_ids, 0)?;

                    // Reshape to [batch_size, seq_len, hidden_size]
                    let batch_size = input_shape.dim(0)?;
                    let seq_len = input_shape.dim(1)?;
                    Ok(embedded.reshape((batch_size, seq_len, hidden_size))?)
                } else {
                    // Input is already embedded [batch_size, seq_len, hidden_size]
                    Ok(hidden_states.clone())
                }
            }

            "final_norm" => {
                // Apply final RMS norm
                let weight = layer_state.tensors.get("weight").ok_or_else(|| {
                    RiallmError::ModelLoading("Final norm weight not found".to_string())
                })?;

                if self.config.is_qwen3_5_moe() {
                    self.apply_rms_norm_one_plus(hidden_states, weight, self.config.rms_norm_eps)
                } else {
                    self.apply_rms_norm(hidden_states, weight, self.config.rms_norm_eps)
                }
            }

            "lm_head" => {
                // Apply LM head (linear projection to vocab)
                let weight = layer_state.tensors.get("weight").ok_or_else(|| {
                    RiallmError::ModelLoading("LM head weight not found".to_string())
                })?;

                let bias = layer_state.tensors.get("bias").cloned();

                // Linear layer: hidden @ weight.T + bias
                let logits = hidden_states.matmul(&weight.t()?)?;

                if let Some(b) = bias {
                    Ok(logits.broadcast_add(&b.reshape((1, 1, b.dim(D::Minus1)?))?)?)
                } else {
                    Ok(logits)
                }
            }

            // Transformer layer
            _ if layer_name.starts_with("layer_") => {
                let layer_idx: usize = layer_name
                    .strip_prefix("layer_")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);

                let tensors = &layer_state.tensors;

                if self.config.is_qwen3_5_moe() {
                    return self.forward_qwen3_5_layer(
                        layer_idx,
                        tensors,
                        hidden_states,
                        attention_mask,
                    );
                }

                // Residual connection
                let mut hidden = hidden_states.clone();

                // Attention norm
                let attn_norm_weight = tensors.get("input_layernorm.weight").ok_or_else(|| {
                    RiallmError::ModelLoading("Attention norm weight not found".to_string())
                })?;

                let normed =
                    self.apply_rms_norm(&hidden, attn_norm_weight, self.config.rms_norm_eps)?;

                // Self-attention
                let q_weight = tensors
                    .get("self_attn.q_proj.weight")
                    .ok_or_else(|| RiallmError::ModelLoading("Q weight not found".to_string()))?;
                let k_weight = tensors
                    .get("self_attn.k_proj.weight")
                    .ok_or_else(|| RiallmError::ModelLoading("K weight not found".to_string()))?;
                let v_weight = tensors
                    .get("self_attn.v_proj.weight")
                    .ok_or_else(|| RiallmError::ModelLoading("V weight not found".to_string()))?;
                let o_weight = tensors
                    .get("self_attn.o_proj.weight")
                    .ok_or_else(|| RiallmError::ModelLoading("O weight not found".to_string()))?;

                // Create position IDs for this layer
                let seq_len = normed.dim(D::Minus2)?;
                let pos_offset = self.kv_cache.as_ref().map(|c| c.seq_len).unwrap_or(0);
                let position_ids =
                    self.create_position_ids(seq_len, pos_offset, hidden_states.device())?;

                // Extract KV cache for this layer
                let kv_cache_layer = if let Some(cache) = &self.kv_cache {
                    if let (Some(k), Some(v)) =
                        (&cache.key_cache[layer_idx], &cache.value_cache[layer_idx])
                    {
                        Some((k, v))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let (attn_output, new_kv) = self.apply_attention(
                    &normed,
                    q_weight,
                    k_weight,
                    v_weight,
                    o_weight,
                    kv_cache_layer,
                    Some(&position_ids),
                    attention_mask,
                )?;

                // Update KV cache
                if let (Some(cache), Some((nk, nv))) = (&mut self.kv_cache, new_kv) {
                    cache.key_cache[layer_idx] = Some(nk.to_device(&self.cpu_device)?);
                    cache.value_cache[layer_idx] = Some(nv.to_device(&self.cpu_device)?);
                }

                // Add attention residual
                hidden = hidden.add(&attn_output)?;

                // Post-attention norm (if applicable)
                if self.layer_names.use_post_attention_layernorm {
                    let ffn_norm_weight = tensors
                        .get("post_attention_layernorm.weight")
                        .ok_or_else(|| {
                            RiallmError::ModelLoading("FFN norm weight not found".to_string())
                        })?;

                    hidden =
                        self.apply_rms_norm(&hidden, ffn_norm_weight, self.config.rms_norm_eps)?;
                }

                // Feed-forward network
                let gate_weight = tensors.get("mlp.gate_proj.weight").ok_or_else(|| {
                    RiallmError::ModelLoading("Gate weight not found".to_string())
                })?;
                let up_weight = tensors
                    .get("mlp.up_proj.weight")
                    .ok_or_else(|| RiallmError::ModelLoading("Up weight not found".to_string()))?;
                let down_weight = tensors.get("mlp.down_proj.weight").ok_or_else(|| {
                    RiallmError::ModelLoading("Down weight not found".to_string())
                })?;

                let ffn_output = self.apply_mlp(&hidden, gate_weight, up_weight, down_weight)?;

                // Add FFN residual
                hidden = hidden.add(&ffn_output)?;

                Ok(hidden)
            }

            _ => Err(RiallmError::LayerNotFound(layer_name.to_string())),
        }
    }

    /// Apply RMS normalization
    fn apply_rms_norm(&self, hidden: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let hidden_f32 = hidden.to_dtype(candle_core::DType::F32)?;
        let variance = hidden_f32.sqr()?.sum_keepdim(D::Minus1)?;
        let hidden_size = hidden.dim(D::Minus1)? as f64;
        let variance = (variance / hidden_size)?;
        let eps_tensor = Tensor::full(eps as f32, variance.shape(), variance.device())?;
        let rsqrt = variance.add(&eps_tensor)?.sqrt()?.recip()?;
        let normalized = hidden_f32.broadcast_mul(&rsqrt)?;
        let weighted = normalized.broadcast_mul(&weight.to_dtype(candle_core::DType::F32)?)?;

        // Convert back to original dtype
        Ok(weighted.to_dtype(hidden.dtype())?)
    }

    /// Qwen3.5/Qwen3.6 stores RMSNorm weights as an offset from 1.0.
    fn apply_rms_norm_one_plus(
        &self,
        hidden: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let hidden_f32 = hidden.to_dtype(candle_core::DType::F32)?;
        let variance = hidden_f32.sqr()?.sum_keepdim(D::Minus1)?;
        let hidden_size = hidden.dim(D::Minus1)? as f64;
        let variance = (variance / hidden_size)?;
        let eps_tensor = Tensor::full(eps as f32, variance.shape(), variance.device())?;
        let rsqrt = variance.add(&eps_tensor)?.sqrt()?.recip()?;
        let normalized = hidden_f32.broadcast_mul(&rsqrt)?;
        let weight = weight.to_dtype(candle_core::DType::F32)?;
        let weight = weight.ones_like()?.add(&weight)?;
        let weighted = normalized.broadcast_mul(&weight)?;

        Ok(weighted.to_dtype(hidden.dtype())?)
    }

    fn forward_qwen3_5_layer(
        &self,
        layer_idx: usize,
        tensors: &HashMap<String, Tensor>,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let input_norm = self.apply_rms_norm_one_plus(
            hidden_states,
            get_tensor(tensors, "input_layernorm.weight")?,
            self.config.rms_norm_eps,
        )?;

        let mixed = match self.qwen3_5_layer_type(layer_idx).as_str() {
            "linear_attention" => {
                self.apply_qwen3_5_gated_delta_net(tensors, &input_norm, attention_mask)?
            }
            "full_attention" => self.apply_qwen3_5_full_attention(tensors, &input_norm)?,
            layer_type => {
                return Err(RiallmError::ModelLoading(format!(
                    "Unsupported Qwen3.6 layer type at layer {}: {}",
                    layer_idx, layer_type
                )));
            }
        };

        let hidden = residual.add(&mixed)?;
        let residual = hidden.clone();
        let ffn_norm = self.apply_rms_norm_one_plus(
            &hidden,
            get_tensor(tensors, "post_attention_layernorm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let ffn_output = self.apply_qwen3_5_moe(tensors, &ffn_norm)?;

        residual.add(&ffn_output).map_err(Into::into)
    }

    fn apply_qwen3_5_full_attention(
        &self,
        tensors: &HashMap<String, Tensor>,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden.dims3()?;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.get_num_key_value_heads();
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = self.qwen3_5_usize("head_dim", self.config.hidden_size / num_heads);
        let attn_dim = num_heads * head_dim;

        let q_proj = hidden.matmul(&get_tensor(tensors, "self_attn.q_proj.weight")?.t()?)?;
        let q_proj = q_proj.reshape((batch_size, seq_len, num_heads, head_dim * 2))?;
        let q_chunks = q_proj.chunk(2, D::Minus1)?;
        let query = self
            .apply_rms_norm_one_plus(
                &q_chunks[0],
                get_tensor(tensors, "self_attn.q_norm.weight")?,
                self.config.rms_norm_eps,
            )?
            .transpose(1, 2)?;
        let gate = q_chunks[1].reshape((batch_size, seq_len, attn_dim))?;

        let key = hidden
            .matmul(&get_tensor(tensors, "self_attn.k_proj.weight")?.t()?)?
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
        let key = self
            .apply_rms_norm_one_plus(
                &key,
                get_tensor(tensors, "self_attn.k_norm.weight")?,
                self.config.rms_norm_eps,
            )?
            .transpose(1, 2)?;
        let value = hidden
            .matmul(&get_tensor(tensors, "self_attn.v_proj.weight")?.t()?)?
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;

        let position_ids = self.create_position_ids(seq_len, 0, hidden.device())?;
        let (query, key) = self.apply_rope(
            &query,
            &key,
            &position_ids,
            self.layer_names.rotary_dim.unwrap_or(head_dim),
            self.config.rope_theta.unwrap_or(10000.0),
        )?;

        let key = repeat_heads(&key, num_kv_groups)?;
        let value = repeat_heads(&value, num_kv_groups)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut attn_weights = (query.matmul(&key.t()?)? * scale)?;
        let mask = self.create_full_causal_mask(seq_len, hidden.device(), attn_weights.dtype())?;
        attn_weights = attn_weights.add(&mask)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, attn_dim))?;
        let attn_output = attn_output.mul(&candle_nn::ops::sigmoid(&gate)?)?;

        attn_output
            .matmul(&get_tensor(tensors, "self_attn.o_proj.weight")?.t()?)
            .map_err(Into::into)
    }

    fn apply_qwen3_5_gated_delta_net(
        &self,
        tensors: &HashMap<String, Tensor>,
        hidden: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden.dims3()?;
        let num_v_heads = self.qwen3_5_usize("linear_num_value_heads", 32);
        let num_k_heads = self.qwen3_5_usize("linear_num_key_heads", 16);
        let head_k_dim = self.qwen3_5_usize("linear_key_head_dim", 128);
        let head_v_dim = self.qwen3_5_usize("linear_value_head_dim", 128);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;

        let mixed_qkv = hidden
            .matmul(&get_tensor(tensors, "linear_attn.in_proj_qkv.weight")?.t()?)?
            .transpose(1, 2)?;
        let mixed_qkv = self.depthwise_causal_conv1d(
            &mixed_qkv,
            get_tensor(tensors, "linear_attn.conv1d.weight")?,
        )?;
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let query = mixed_qkv.narrow(D::Minus1, 0, key_dim)?.reshape((
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        ))?;
        let key = mixed_qkv.narrow(D::Minus1, key_dim, key_dim)?.reshape((
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        ))?;
        let value = mixed_qkv
            .narrow(D::Minus1, key_dim * 2, value_dim)?
            .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?;

        let z = hidden
            .matmul(&get_tensor(tensors, "linear_attn.in_proj_z.weight")?.t()?)?
            .reshape((batch_size, seq_len, num_v_heads, head_v_dim))?;
        let beta = candle_nn::ops::sigmoid(
            &hidden.matmul(&get_tensor(tensors, "linear_attn.in_proj_b.weight")?.t()?)?,
        )?;
        let a = hidden.matmul(&get_tensor(tensors, "linear_attn.in_proj_a.weight")?.t()?)?;
        let dt_bias = get_tensor(tensors, "linear_attn.dt_bias")?
            .to_dtype(candle_core::DType::F32)?
            .reshape((1, 1, num_v_heads))?;
        let a = a
            .to_dtype(candle_core::DType::F32)?
            .broadcast_add(&dt_bias)?;
        let g = softplus(&a)?
            .broadcast_mul(
                &get_tensor(tensors, "linear_attn.A_log")?
                    .to_dtype(candle_core::DType::F32)?
                    .exp()?
                    .reshape((1, 1, num_v_heads))?,
            )?
            .neg()?;

        let repeat = num_v_heads / num_k_heads;
        let query = if repeat > 1 {
            repeat_linear_heads(&query, repeat)?
        } else {
            query
        };
        let key = if repeat > 1 {
            repeat_linear_heads(&key, repeat)?
        } else {
            key
        };

        let core = self.torch_recurrent_gated_delta_rule(&query, &key, &value, &g, &beta)?;
        let core = core.reshape((batch_size * seq_len * num_v_heads, head_v_dim))?;
        let z = z.reshape((batch_size * seq_len * num_v_heads, head_v_dim))?;
        let core = self.apply_qwen3_5_gated_rms_norm(
            &core,
            &z,
            get_tensor(tensors, "linear_attn.norm.weight")?,
        )?;
        let core = core.reshape((batch_size, seq_len, value_dim))?;

        core.matmul(&get_tensor(tensors, "linear_attn.out_proj.weight")?.t()?)
            .map_err(Into::into)
    }

    fn apply_qwen3_5_gated_rms_norm(
        &self,
        hidden: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.apply_rms_norm(hidden, weight, self.config.rms_norm_eps)?;
        normed
            .mul(
                &candle_nn::ops::silu(&gate.to_dtype(candle_core::DType::F32)?)?
                    .to_dtype(normed.dtype())?,
            )
            .map_err(Into::into)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, num_heads, head_k_dim) = query.dims4()?;
        let head_v_dim = value.dim(D::Minus1)?;
        let output_dtype = value.dtype();
        let query = l2norm(query, D::Minus1, 1e-6)?.to_dtype(candle_core::DType::F32)?;
        let key = l2norm(key, D::Minus1, 1e-6)?.to_dtype(candle_core::DType::F32)?;
        let value = value.to_dtype(candle_core::DType::F32)?;
        let beta = beta.to_dtype(candle_core::DType::F32)?;
        let g = g.to_dtype(candle_core::DType::F32)?;
        let query = (query * (1.0 / (head_k_dim as f64).sqrt()))?;

        let mut state = Tensor::zeros(
            (batch_size, num_heads, head_k_dim, head_v_dim),
            candle_core::DType::F32,
            query.device(),
        )?;
        let mut outputs = Vec::with_capacity(seq_len);

        for index in 0..seq_len {
            let q_t = query.narrow(1, index, 1)?.squeeze(1)?;
            let k_t = key.narrow(1, index, 1)?.squeeze(1)?;
            let v_t = value.narrow(1, index, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, index, 1)?.squeeze(1)?.exp()?;
            let beta_t = beta.narrow(1, index, 1)?.squeeze(1)?;

            state = state.broadcast_mul(&g_t.reshape((batch_size, num_heads, 1, 1))?)?;
            let kv_mem = state
                .broadcast_mul(&k_t.reshape((batch_size, num_heads, head_k_dim, 1))?)?
                .sum(D::Minus2)?;
            let delta = v_t
                .add(&kv_mem.neg()?)?
                .broadcast_mul(&beta_t.reshape((batch_size, num_heads, 1))?)?;
            let update = k_t
                .reshape((batch_size, num_heads, head_k_dim, 1))?
                .broadcast_mul(&delta.reshape((batch_size, num_heads, 1, head_v_dim))?)?;
            state = state.add(&update)?;
            let out = state
                .broadcast_mul(&q_t.reshape((batch_size, num_heads, head_k_dim, 1))?)?
                .sum(D::Minus2)?;
            outputs.push(out);
        }

        Ok(Tensor::stack(&outputs, 1)?.to_dtype(output_dtype)?)
    }

    fn apply_qwen3_5_moe(
        &self,
        tensors: &HashMap<String, Tensor>,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = hidden.dims3()?;
        let flat = hidden.reshape((batch_size * seq_len, hidden_dim))?;
        let top_k = self.qwen3_5_usize("num_experts_per_tok", 8);
        let num_experts = self.qwen3_5_usize("num_experts", 256);

        let shared = self.apply_qwen3_5_shared_expert(tensors, &flat)?;
        let router_logits = flat.matmul(&get_tensor(tensors, "mlp.gate.weight")?.t()?)?;
        let routing_probs = candle_nn::ops::softmax(&router_logits, D::Minus1)?;
        let selected_experts = routing_probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, top_k)?
            .contiguous()?;
        let routing_weights = routing_probs.gather(&selected_experts, D::Minus1)?;
        let routing_weights =
            routing_weights.broadcast_div(&routing_weights.sum_keepdim(D::Minus1)?)?;

        let routing_weights_vec = routing_weights
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;
        let selected_experts_vec = selected_experts.to_vec2::<u32>()?;
        let mut token_indices = vec![Vec::new(); num_experts];
        let mut token_weights = vec![Vec::new(); num_experts];

        for (row_idx, (weights, experts)) in routing_weights_vec
            .iter()
            .zip(selected_experts_vec.iter())
            .enumerate()
        {
            for (&weight, &expert_idx) in weights.iter().zip(experts.iter()) {
                let expert_idx = expert_idx as usize;
                token_indices[expert_idx].push(row_idx as u32);
                token_weights[expert_idx].push(weight);
            }
        }

        let gate_up_proj = get_tensor(tensors, "mlp.experts.gate_up_proj")?;
        let down_proj = get_tensor(tensors, "mlp.experts.down_proj")?;
        let mut output = flat.zeros_like()?;
        for expert_idx in 0..num_experts {
            if token_indices[expert_idx].is_empty() {
                continue;
            }

            let indices = Tensor::new(token_indices[expert_idx].as_slice(), flat.device())?;
            let weights = Tensor::new(token_weights[expert_idx].as_slice(), flat.device())?
                .reshape(((), 1))?
                .to_dtype(flat.dtype())?;
            let current = flat.index_select(&indices, 0)?;
            let gate_up = gate_up_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;
            let projected = current.matmul(&gate_up.t()?)?;
            let chunks = projected.chunk(2, D::Minus1)?;
            let expert_hidden = candle_nn::ops::silu(&chunks[0])?.mul(&chunks[1])?;
            let down = down_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;
            let expert_hidden = expert_hidden.matmul(&down.t()?)?;
            let expert_hidden = expert_hidden.broadcast_mul(&weights)?;
            output = output.index_add(&indices, &expert_hidden, 0)?;
        }

        output
            .add(&shared)?
            .reshape((batch_size, seq_len, hidden_dim))
            .map_err(Into::into)
    }

    fn apply_qwen3_5_shared_expert(
        &self,
        tensors: &HashMap<String, Tensor>,
        flat: &Tensor,
    ) -> Result<Tensor> {
        let gate = flat.matmul(&get_tensor(tensors, "mlp.shared_expert.gate_proj.weight")?.t()?)?;
        let up = flat.matmul(&get_tensor(tensors, "mlp.shared_expert.up_proj.weight")?.t()?)?;
        let shared = candle_nn::ops::silu(&gate)?
            .mul(&up)?
            .matmul(&get_tensor(tensors, "mlp.shared_expert.down_proj.weight")?.t()?)?;
        let shared_gate = candle_nn::ops::sigmoid(
            &flat.matmul(&get_tensor(tensors, "mlp.shared_expert_gate.weight")?.t()?)?,
        )?;
        shared.broadcast_mul(&shared_gate).map_err(Into::into)
    }

    fn depthwise_causal_conv1d(&self, xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let (batch_size, channels, seq_len) = xs.dims3()?;
        let weight = if weight.dims().len() == 3 {
            weight.squeeze(1)?
        } else {
            weight.clone()
        };
        let kernel = weight.dim(D::Minus1)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for out_index in 0..seq_len {
            let mut acc = Tensor::zeros(
                (batch_size, channels, 1),
                candle_core::DType::F32,
                xs.device(),
            )?;
            for kernel_index in 0..kernel {
                let input_index = out_index as isize + kernel_index as isize + 1 - kernel as isize;
                if input_index < 0 || input_index >= seq_len as isize {
                    continue;
                }
                let input = xs
                    .narrow(2, input_index as usize, 1)?
                    .to_dtype(candle_core::DType::F32)?;
                let kernel_weight = weight
                    .narrow(D::Minus1, kernel_index, 1)?
                    .reshape((1, channels, 1))?
                    .to_dtype(candle_core::DType::F32)?;
                acc = acc.add(&input.broadcast_mul(&kernel_weight)?)?;
            }
            outputs.push(candle_nn::ops::silu(&acc)?.to_dtype(xs.dtype())?);
        }

        Ok(Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?)
    }

    fn create_full_causal_mask(
        &self,
        seq_len: usize,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();

        Ok(Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?.to_dtype(dtype)?)
    }

    fn qwen3_5_usize(&self, key: &str, default: usize) -> usize {
        self.config
            .extra
            .get(key)
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(default)
    }

    fn qwen3_5_layer_type(&self, layer_idx: usize) -> String {
        self.config
            .extra
            .get("layer_types")
            .and_then(|value| value.as_array())
            .and_then(|layers| layers.get(layer_idx))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| {
                let interval = self.qwen3_5_usize("full_attention_interval", 4);
                if (layer_idx + 1) % interval == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
    }

    /// Apply self-attention
    fn apply_attention(
        &self,
        hidden: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        o_weight: &Tensor,
        kv_cache: Option<(&Tensor, &Tensor)>,
        position_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (batch_size, seq_len, _) = hidden.dims3()?;
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        let num_kv_heads = self.config.get_num_key_value_heads();
        let num_q_heads = self.config.num_attention_heads;

        // Project Q, K, V
        let q = hidden.matmul(&q_weight.t()?)?;
        let k = hidden.matmul(&k_weight.t()?)?;
        let v = hidden.matmul(&v_weight.t()?)?;

        // Reshape to (batch, heads, seq, head_dim)
        let mut q = q
            .reshape((batch_size, seq_len, num_q_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut k = k
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let v = v
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE (rotary position embeddings)
        if let Some(pos_ids) = position_ids {
            let rope_theta = self.config.rope_theta.unwrap_or(10000.0);
            let rope_dim = self.layer_names.rotary_dim.unwrap_or(head_dim);
            (q, k) = self.apply_rope(&q, &k, pos_ids, rope_dim, rope_theta)?;
        }

        // Handle KV cache
        let (k_to_use, v_to_use) = if let Some((k_cache, v_cache)) = kv_cache {
            // Concatenate with cached KV
            let k_cat = Tensor::cat(&[k_cache, &k], 2)?;
            let v_cat = Tensor::cat(&[v_cache, &v], 2)?;
            (k_cat, v_cat)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut attn_weights = q.matmul(&k_to_use.t()?.to_dtype(candle_core::DType::F32)?)?;
        attn_weights = (attn_weights * scale)?;

        // Apply attention mask (causal or provided)
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.add(mask)?;
        } else if seq_len > 1 || kv_cache.is_some() {
            // Apply causal mask for generation
            let causal_mask = self.create_causal_mask(seq_len, k_to_use.dim(2)?)?;
            if let Some(mask) = causal_mask {
                attn_weights = attn_weights.add(&mask)?;
            }
        }

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v_to_use)?;

        // Reshape back to (batch, seq, hidden)
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch_size, seq_len, self.config.hidden_size))?;

        // Output projection
        let attn_output = attn_output.matmul(&o_weight.t()?)?;

        Ok((attn_output, Some((k_to_use, v_to_use))))
    }

    /// Apply RoPE to Q and K tensors
    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
        rope_dim: usize,
        rope_theta: f32,
    ) -> Result<(Tensor, Tensor)> {
        // Create inverse frequency tensor
        let inv_freq: Vec<f32> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1.0 / rope_theta.powf(i as f32 / rope_dim as f32))
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq_tensor = Tensor::from_vec(inv_freq, &[1, 1, inv_freq_len], q.device())?;

        // Get position_ids shape
        let pos_shape = position_ids.shape();
        let seq_len = pos_shape.dim(D::Minus1)?;

        // Expand position_ids to [1, seq_len, rope_dim/2]
        let pos_expanded = position_ids.reshape((1, seq_len, 1))?;
        let pos_broadcast = pos_expanded.broadcast_as((1, seq_len, inv_freq_len))?;

        // Compute freq matrix: [1, seq_len, rope_dim/2]
        let freqs = pos_broadcast.mul(&inv_freq_tensor)?;

        // Create cos and sin: [1, seq_len, rope_dim/2]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Broadcast to [1, 1, seq_len, rope_dim/2] for q/k heads
        let cos_4d = cos.reshape((1, 1, seq_len, inv_freq_len))?;
        let sin_4d = sin.reshape((1, 1, seq_len, inv_freq_len))?;

        // Apply RoPE to q and k
        let q_rope = self.rotate_half_and_apply(q, &cos_4d, &sin_4d, rope_dim)?;
        let k_rope = self.rotate_half_and_apply(k, &cos_4d, &sin_4d, rope_dim)?;

        Ok((q_rope, k_rope))
    }

    /// Apply rotary transformation: x * cos + rotate_half(x) * sin
    fn rotate_half_and_apply(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope_dim: usize,
    ) -> Result<Tensor> {
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let head_dim = dims[3];
        let half = rope_dim / 2;

        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let out1 = x1.mul(cos)?.add(&x2.mul(sin)?.neg()?)?;
        let out2 = x2.mul(cos)?.add(&x1.mul(sin)?)?;
        let result = Tensor::cat(&[&out1, &out2], 3)?;

        if rope_dim < head_dim {
            let unchanged = x.narrow(3, rope_dim, head_dim - rope_dim)?;
            Ok(Tensor::cat(&[&result, &unchanged], 3)?)
        } else {
            Ok(result)
        }
    }

    /// Create causal attention mask
    fn create_causal_mask(&self, seq_len: usize, kv_seq_len: usize) -> Result<Option<Tensor>> {
        if seq_len >= kv_seq_len {
            return Ok(None); // No masking needed
        }

        // Create lower triangular mask
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_seq_len).map(move |j| {
                    if j > i + (kv_seq_len - seq_len) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let mask = Tensor::from_vec(mask_data, &[1, 1, seq_len, kv_seq_len], &Device::Cpu)?;
        Ok(Some(mask))
    }

    /// Create position IDs tensor [seq_len]
    fn create_position_ids(
        &self,
        seq_len: usize,
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let position_ids: Vec<u32> = (offset as u32..(offset + seq_len) as u32).collect();
        Ok(Tensor::from_vec(position_ids, &[seq_len], device)?)
    }

    /// Apply MLP (feed-forward network)
    fn apply_mlp(
        &self,
        hidden: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
    ) -> Result<Tensor> {
        // Gate projection with SiLU activation
        let gate = hidden.matmul(&gate_weight.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;

        // Up projection
        let up = hidden.matmul(&up_weight.t()?)?;

        // Element-wise multiplication
        let hidden = gate.mul(&up)?;

        // Down projection
        let output = hidden.matmul(&down_weight.t()?)?;

        Ok(output)
    }

    /// Load a layer from disk to CPU
    fn load_layer_to_cpu(&mut self, layer_name: &str) -> Result<()> {
        let layer_path = SafetensorModelPersister::layer_path(&self.split_path, layer_name);

        if !self.persister.layer_exists(&layer_path) {
            return Err(RiallmError::LayerNotFound(format!(
                "Layer {} not found at {:?}",
                layer_name, layer_path
            )));
        }

        let tensors = self.persister.load_layer(&layer_path)?;

        // Apply quantization if enabled
        let tensors = if self.options.compression != CompressionType::None {
            // TODO: Implement quantization
            tensors
        } else {
            tensors
        };

        // Update layer state
        if let Some(state) = self.layer_states.get_mut(layer_name) {
            state.tensors = tensors;
        }

        Ok(())
    }

    /// Move a layer from CPU to GPU
    fn load_layer_to_device(&mut self, layer_name: &str) -> Result<()> {
        // Load to CPU first if not already loaded
        if !self.layer_states.contains_key(layer_name)
            || self.layer_states[layer_name].tensors.is_empty()
        {
            self.load_layer_to_cpu(layer_name)?;
        }

        // Move tensors to GPU device
        if let Some(state) = self.layer_states.get_mut(layer_name) {
            let mut gpu_tensors = HashMap::new();

            for (name, tensor) in &state.tensors {
                gpu_tensors.insert(name.clone(), tensor.to_device(&self.device)?);
            }

            state.tensors = gpu_tensors;
            state.on_gpu = true;
        }

        Ok(())
    }

    /// Unload a layer from GPU to free memory
    fn unload_layer_from_gpu(&mut self, layer_name: &str) -> Result<()> {
        if let Some(state) = self.layer_states.get_mut(layer_name) {
            if state.on_gpu {
                // Move tensors back to CPU
                let mut cpu_tensors = HashMap::new();

                for (name, tensor) in &state.tensors {
                    cpu_tensors.insert(name.clone(), tensor.to_device(&self.cpu_device)?);
                }

                state.tensors = cpu_tensors;
                state.on_gpu = false;
            }
        }

        Ok(())
    }

    /// Check if a layer is currently on GPU
    fn is_layer_on_gpu(&self, layer_name: &str) -> bool {
        self.layer_states
            .get(layer_name)
            .map(|state| state.on_gpu)
            .unwrap_or(false)
    }

    /// Clean up memory after layer processing
    fn clean_memory(&self) -> Result<()> {
        // Trigger CUDA memory cleanup if applicable
        if self.device.is_cuda() {
            // TODO: Implement proper CUDA memory management
            // candle_core::cuda::synchronize()?;
        }

        Ok(())
    }

    /// Generate text from input IDs
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        max_new_tokens: usize,
        _temperature: f64,
        _top_p: Option<f64>,
    ) -> Result<Vec<u32>> {
        if self.config.is_qwen3_5_moe() {
            return self.generate_full_context(input_ids, max_new_tokens);
        }

        let mut tokens = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let mut current_ids = input_ids.clone();

        // Initial forward pass with full input
        let mut logits = self.forward(current_ids.to_device(&self.cpu_device)?, None)?;

        for _ in 0..max_new_tokens {
            // Get logits for last token
            let seq_len = logits.dim(D::Minus2)?;
            let last_token_logits = logits.narrow(D::Minus2, seq_len - 1, 1)?;
            let last_token_logits = last_token_logits.squeeze(0)?.squeeze(0)?;

            // Simple argmax sampling
            let logits_f32 = last_token_logits.to_vec1::<f32>()?;
            let next_token = logits_f32
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            tokens.push(next_token);

            // Update KV cache seq_len
            if let Some(cache) = &mut self.kv_cache {
                cache.seq_len += current_ids.dim(D::Minus1)?;
            }

            // Prepare next input (just the new token)
            current_ids = Tensor::new(&[next_token], &self.cpu_device)?.reshape((1, 1))?;

            // Forward pass for next token
            logits = self.forward(current_ids.clone(), None)?;
        }

        Ok(tokens)
    }

    fn generate_full_context(
        &mut self,
        input_ids: &Tensor,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.flatten_all()?.to_vec1::<u32>()?;

        for _ in 0..max_new_tokens {
            self.reset_kv_cache();
            let current_ids =
                Tensor::new(tokens.as_slice(), &self.cpu_device)?.reshape((1, tokens.len()))?;
            let logits = self.forward(current_ids, None)?;
            let seq_len = logits.dim(D::Minus2)?;
            let last_token_logits = logits.narrow(D::Minus2, seq_len - 1, 1)?;
            let last_token_logits = last_token_logits.squeeze(0)?.squeeze(0)?;
            let logits_f32 = last_token_logits.to_vec1::<f32>()?;
            let next_token = logits_f32
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Reset cached key/value tensors before starting a fresh prompt.
    pub fn reset_kv_cache(&mut self) {
        self.kv_cache = Some(KVCACHE::new(self.config.num_hidden_layers));
    }

    /// Get the profiler for inspection
    pub fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }
}

fn get_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| RiallmError::ModelLoading(format!("Tensor not found: {}", name)))
}

fn repeat_heads(hidden_states: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch_size, heads, seq_len, head_dim) = hidden_states.dims4()?;
    hidden_states
        .reshape((batch_size, heads, 1, seq_len, head_dim))?
        .broadcast_as((batch_size, heads, repeats, seq_len, head_dim))?
        .reshape((batch_size, heads * repeats, seq_len, head_dim))
        .map_err(Into::into)
}

fn repeat_linear_heads(hidden_states: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch_size, seq_len, heads, head_dim) = hidden_states.dims4()?;
    hidden_states
        .reshape((batch_size, seq_len, heads, 1, head_dim))?
        .broadcast_as((batch_size, seq_len, heads, repeats, head_dim))?
        .reshape((batch_size, seq_len, heads * repeats, head_dim))
        .map_err(Into::into)
}

fn l2norm(x: &Tensor, dim: D, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let norm = x_f32.sqr()?.sum_keepdim(dim)?;
    let eps = Tensor::full(eps, norm.shape(), norm.device())?;
    x_f32
        .broadcast_mul(&norm.add(&eps)?.sqrt()?.recip()?)
        .map_err(Into::into)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.add(&x.ones_like()?)?.log().map_err(Into::into)
}
