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

                self.apply_rms_norm(hidden_states, weight, self.config.rms_norm_eps)
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
            (q, k) = self.apply_rope(&q, &k, pos_ids, head_dim, rope_theta)?;
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
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<(Tensor, Tensor)> {
        let rope_dim = head_dim; // Use full head dim for RoPE

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
        // Split x into two halves for rotation
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let head_dim = dims[3];

        // Split x into x1 and x2 (each half of the rope_dim)
        let x1 = x.narrow(3, 0, rope_dim / 2)?;
        let x2 = x.narrow(3, rope_dim / 2, rope_dim / 2)?;

        // rotate_half: [-x2, x1]
        let neg_x2 = x2.neg()?;
        let rotated = Tensor::cat(&[&neg_x2, &x1], 3)?;

        // x * cos + rotated * sin
        let x_cos = x.narrow(3, 0, rope_dim)?.mul(cos)?;
        let rotated_sin = rotated.mul(sin)?;
        let result = x_cos.add(&rotated_sin)?;

        // If rope_dim < head_dim, concatenate the unchanged part
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
        temperature: f64,
        _top_p: Option<f64>,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec1::<u32>()?;
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

    /// Get the profiler for inspection
    pub fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }
}
