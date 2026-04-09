//! Quantization utilities for model compression

use std::collections::HashMap;

use candle_core::{Tensor, D};

use crate::error::{Result, RiallmError};
use crate::config::CompressionType;

/// Quantized tensor storage
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (4-bit or 8-bit)
    pub data: Vec<u8>,
    
    /// Original shape
    pub shape: Vec<usize>,
    
    /// Quantization type
    pub qtype: CompressionType,
    
    /// Scaling factors for dequantization
    pub scales: Vec<f32>,
    
    /// Zero points for dequantization
    pub zero_points: Vec<f32>,
    
    /// Block size for block-wise quantization
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Dequantize the tensor back to full precision
    pub fn dequantize(&self) -> Result<Vec<f32>> {
        match self.qtype {
            CompressionType::FourBit => self.dequantize_4bit(),
            CompressionType::EightBit => self.dequantize_8bit(),
            CompressionType::None => Err(RiallmError::Quantization(
                "Tensor is not quantized".to_string()
            )),
        }
    }
    
    /// Dequantize 4-bit NF4 quantization
    fn dequantize_4bit(&self) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        let mut scale_idx = 0;
        
        for (i, &byte) in self.data.iter().enumerate() {
            // Each byte contains two 4-bit values
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;
            
            // NF4 dequantization with normal float 4-bit format
            let scale = self.scales.get(scale_idx).copied().unwrap_or(1.0);
            let zero = self.zero_points.get(scale_idx).copied().unwrap_or(0.0);
            
            // Dequantize
            result.push((low - zero) * scale);
            
            // Check if we need to update scale index
            let element_idx = i * 2;
            if (element_idx + 2) % self.block_size == 0 {
                scale_idx += 1;
            }
            
            // High nibble
            if element_idx + 1 < self.shape.iter().product::<usize>() {
                let scale = self.scales.get(scale_idx).copied().unwrap_or(1.0);
                let zero = self.zero_points.get(scale_idx).copied().unwrap_or(0.0);
                result.push((high - zero) * scale);
            }
        }
        
        Ok(result)
    }
    
    /// Dequantize 8-bit block-wise quantization
    fn dequantize_8bit(&self) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        
        for (i, &byte) in self.data.iter().enumerate() {
            let value = byte as f32;
            
            // Calculate which block we're in
            let block_idx = i / self.block_size;
            let scale = self.scales.get(block_idx).copied().unwrap_or(1.0);
            let zero = self.zero_points.get(block_idx).copied().unwrap_or(0.0);
            
            // Dequantize
            result.push((value - zero) * scale);
        }
        
        Ok(result)
    }
}

/// Quantize a tensor
pub fn quantize_tensor(tensor: &Tensor, compression: &CompressionType) -> Result<QuantizedTensor> {
    match compression {
        CompressionType::FourBit => quantize_4bit_nf4(tensor),
        CompressionType::EightBit => quantize_8bit_blockwise(tensor),
        CompressionType::None => Err(RiallmError::Quantization(
            "No compression specified".to_string()
        )),
    }
}

/// 4-bit Normalized Float 4-bit (NF4) quantization
/// Based on bitsandbytes implementation
fn quantize_4bit_nf4(tensor: &Tensor) -> Result<QuantizedTensor> {
    let shape = tensor.shape().dims().to_vec();
    let num_elements = shape.iter().product::<usize>();
    
    // Flatten tensor to f32
    let data_f32 = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    // NF4 quantization parameters
    let block_size = 64;
    let num_blocks = (num_elements + block_size - 1) / block_size;
    
    let mut quantized = Vec::with_capacity(num_elements / 2);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut zero_points = Vec::with_capacity(num_blocks);
    
    // NF4 quantization table (approximately normal distribution)
    const NF4_TABLE: [f32; 16] = [
        -1.0, -0.6962, -0.5250, -0.3900, -0.2750, -0.1750, -0.0850, -0.0150,
        0.0, 0.0150, 0.0850, 0.1750, 0.2750, 0.3900, 0.5250, 0.6962,
    ];
    
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);
        let block = &data_f32[start..end];
        
        // Calculate absolute max for scaling
        let abs_max = block.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = abs_max / 1.0; // Normalize to [-1, 1]
        
        // Calculate zero point (offset for asymmetric quantization)
        let sum: f32 = block.iter().sum();
        let mean = sum / block.len() as f32;
        let zero_point = mean;
        
        scales.push(scale);
        zero_points.push(zero_point);
        
        // Quantize block
        for i in (0..block.len()).step_by(2) {
            let val1 = block[i];
            let val2 = if i + 1 < block.len() { block[i + 1] } else { 0.0 };
            
            // Normalize and quantize to 4-bit
            let q1 = normalize_and_quantize_nf4(val1, &NF4_TABLE);
            let q2 = normalize_and_quantize_nf4(val2, &NF4_TABLE);
            
            // Pack two 4-bit values into one byte
            let packed = (q2 << 4) | q1;
            quantized.push(packed);
        }
    }
    
    Ok(QuantizedTensor {
        data: quantized,
        shape,
        qtype: CompressionType::FourBit,
        scales,
        zero_points,
        block_size,
    })
}

/// Normalize a value and quantize to NF4
fn normalize_and_quantize_nf4(value: f32, table: &[f32; 16]) -> u8 {
    // Clamp to [-1, 1]
    let normalized = value.clamp(-1.0, 1.0);
    
    // Find closest NF4 value
    let mut min_dist = f32::MAX;
    let mut quantized = 0u8;
    
    for (i, &nf4_val) in table.iter().enumerate() {
        let dist = (normalized - nf4_val).abs();
        if dist < min_dist {
            min_dist = dist;
            quantized = i as u8;
        }
    }
    
    quantized
}

/// 8-bit block-wise quantization
fn quantize_8bit_blockwise(tensor: &Tensor) -> Result<QuantizedTensor> {
    let shape = tensor.shape().dims().to_vec();
    let num_elements = shape.iter().product::<usize>();
    
    // Flatten tensor to f32
    let data_f32 = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    // Block-wise quantization parameters
    let block_size = 128;
    let num_blocks = (num_elements + block_size - 1) / block_size;
    
    let mut quantized = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut zero_points = Vec::with_capacity(num_blocks);
    
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);
        let block = &data_f32[start..end];
        
        // Calculate min and max for the block
        let min = block.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = block.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate scale and zero point
        let scale = (max - min) / 255.0;
        let zero_point = min;
        
        scales.push(scale);
        zero_points.push(zero_point);
        
        // Quantize block
        for &val in block {
            let q = ((val - zero_point) / scale).round().clamp(0.0, 255.0) as u8;
            quantized.push(q);
        }
    }
    
    Ok(QuantizedTensor {
        data: quantized,
        shape,
        qtype: CompressionType::EightBit,
        scales,
        zero_points,
        block_size,
    })
}

/// Dequantize a layer's state dict
pub fn dequantize_layer_state_dict(
    state_dict: &HashMap<String, Tensor>,
    quant_metadata: &HashMap<String, QuantizedTensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut dequantized = HashMap::new();
    
    for (name, quant_tensor) in quant_metadata {
        let dequant_data = quant_tensor.dequantize()?;
        
        // Convert back to tensor
        let tensor = Tensor::from_vec(
            dequant_data,
            &*quant_tensor.shape,
            &candle_core::Device::Cpu,
        )?;
        
        dequantized.insert(name.clone(), tensor);
    }
    
    // Copy non-quantized tensors
    for (name, tensor) in state_dict {
        if !quant_metadata.contains_key(name) {
            dequantized.insert(name.clone(), tensor.clone());
        }
    }
    
    Ok(dequantized)
}

/// Apply compression to layer state dict
pub fn compress_layer_state_dict(
    state_dict: &HashMap<String, Tensor>,
    compression: &CompressionType,
) -> Result<(HashMap<String, Tensor>, HashMap<String, QuantizedTensor>)> {
    if *compression == CompressionType::None {
        return Ok((state_dict.clone(), HashMap::new()));
    }
    
    let mut compressed = HashMap::new();
    let mut metadata = HashMap::new();
    
    for (name, tensor) in state_dict {
        // Skip small tensors (biases, etc.)
        let num_elements = tensor.shape().elem_count();
        if num_elements < 4096 {
            compressed.insert(name.clone(), tensor.clone());
            continue;
        }
        
        // Quantize large tensors
        let quant_tensor = quantize_tensor(tensor, compression)?;
        metadata.insert(name.clone(), quant_tensor.clone());
        
        // Store quantized data as a tensor for serialization
        let quant_data_tensor = Tensor::from_vec(
            quant_tensor.data.clone(),
            &[quant_tensor.data.len()],
            &candle_core::Device::Cpu,
        )?;
        
        compressed.insert(name.clone(), quant_data_tensor);
    }
    
    Ok((compressed, metadata))
}

/// Save quantization state for later use
pub fn save_quant_state(
    metadata: &HashMap<String, QuantizedTensor>,
    path: &std::path::Path,
) -> Result<()> {
    // Serialize quantization metadata to JSON
    let mut serializable = HashMap::new();
    
    for (name, qt) in metadata {
        serializable.insert(
            name.clone(),
            serde_json::json!({
                "shape": qt.shape,
                "qtype": format!("{:?}", qt.qtype),
                "scales": qt.scales,
                "zero_points": qt.zero_points,
                "block_size": qt.block_size,
            }),
        );
    }
    
    let json_str = serde_json::to_string_pretty(&serializable)?;
    std::fs::write(path, json_str)?;
    
    Ok(())
}

/// Load quantization state
pub fn load_quant_state(path: &std::path::Path) -> Result<HashMap<String, QuantizedTensor>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    
    let json_str = std::fs::read_to_string(path)?;
    let serializable: HashMap<String, serde_json::Value> = serde_json::from_str(&json_str)?;
    
    let mut metadata = HashMap::new();
    
    for (name, data) in serializable {
        let qt = QuantizedTensor {
            data: Vec::new(), // Will be loaded separately
            shape: data["shape"]
                .as_array()
                .ok_or_else(|| RiallmError::Quantization("Invalid shape".to_string()))?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            qtype: CompressionType::EightBit, // Default, will be set from data
            scales: data["scales"]
                .as_array()
                .ok_or_else(|| RiallmError::Quantization("Invalid scales".to_string()))?
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect(),
            zero_points: data["zero_points"]
                .as_array()
                .ok_or_else(|| RiallmError::Quantization("Invalid zero points".to_string()))?
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect(),
            block_size: data["block_size"].as_u64().unwrap_or(128) as usize,
        };
        
        metadata.insert(name, qt);
    }
    
    Ok(metadata)
}
