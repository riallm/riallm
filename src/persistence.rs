//! Model persistence layer - handles saving and loading model shards

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;

use crate::error::{Result, RiallmError};

/// Trait for model persistence operations
pub trait ModelPersister: Send + Sync {
    /// Check if a layer exists at the given path
    fn layer_exists(&self, layer_path: &Path) -> bool;
    
    /// Load a layer from disk
    fn load_layer(&self, layer_path: &Path) -> Result<HashMap<String, Tensor>>;
    
    /// Save a layer to disk
    fn save_layer(&self, layer_path: &Path, tensors: &HashMap<String, Tensor>) -> Result<()>;
}

/// Safetensors-based model persister
pub struct SafetensorModelPersister {
    /// Device to load tensors to
    device: Device,
}

impl SafetensorModelPersister {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    
    /// Get the path to a layer shard file
    pub fn layer_path(base_path: &Path, layer_name: &str) -> PathBuf {
        base_path.join(format!("{}.safetensors", layer_name))
    }
}

impl ModelPersister for SafetensorModelPersister {
    fn layer_exists(&self, layer_path: &Path) -> bool {
        layer_path.exists()
    }
    
    fn load_layer(&self, layer_path: &Path) -> Result<HashMap<String, Tensor>> {
        if !layer_path.exists() {
            return Err(RiallmError::LayerNotFound(
                format!("Layer file not found: {:?}", layer_path)
            ));
        }
        
        let file = std::fs::File::open(layer_path)?;
        let file_size = file.metadata()?.len();
        
        // Memory map the file for efficient loading
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        
        // Parse safetensors
        let safetensors = SafeTensors::deserialize(&mmap)?;
        
        let mut tensors = HashMap::new();
        
        for tensor_view in safetensors.tensors() {
            let (name, view) = tensor_view;
            let dtype = view.dtype();
            let shape = view.shape();
            let data = view.data();
            
            // Convert safetensors dtype to candle dtype
            let candle_dtype = safetensors_dtype_to_candle(dtype)?;
            
            // Create tensor from raw bytes
            let tensor = Tensor::from_raw_buffer(
                data,
                candle_dtype,
                &shape,
                &self.device,
            )?;
            
            tensors.insert(name, tensor);
        }
        
        Ok(tensors)
    }
    
    fn save_layer(&self, layer_path: &Path, tensors: &HashMap<String, Tensor>) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = layer_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Convert candle tensors to safetensors format
        let mut tensor_data: HashMap<String, (Vec<u8>, candle_core::DType, Vec<usize>)> = HashMap::new();
        
        for (name, tensor) in tensors {
            let bytes = tensor.flatten_all()?.to_vec1::<u8>()?;
            let dtype = tensor.dtype();
            let shape = tensor.shape().dims().to_vec();
            tensor_data.insert(name.clone(), (bytes, dtype, shape));
        }
        
        // Create safetensors view
        let views: HashMap<String, safetensors::tensor::TensorView> = tensor_data
            .iter()
            .map(|(name, (data, dtype, shape))| {
                let view = safetensors::tensor::TensorView::new(
                    candle_dtype_to_safetensors(*dtype)?,
                    shape.clone(),
                    data.as_slice(),
                )?;
                Ok((name.clone(), view))
            })
            .collect::<Result<_>>()?;
        
        // Serialize to file
        safetensors::serialize_to_file(views, &None, layer_path)?;
        
        Ok(())
    }
}

/// Convert safetensors dtype to candle dtype
pub fn safetensors_dtype_to_candle(dtype: safetensors::Dtype) -> Result<candle_core::DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(candle_core::DType::F32),
        safetensors::Dtype::F16 => Ok(candle_core::DType::F16),
        safetensors::Dtype::BF16 => Ok(candle_core::DType::BF16),
        safetensors::Dtype::I64 => Ok(candle_core::DType::I64),
        safetensors::Dtype::U32 => Ok(candle_core::DType::U32),
        safetensors::Dtype::U8 => Ok(candle_core::DType::U8),
        safetensors::Dtype::BOOL => Ok(candle_core::DType::U8), // TODO: Handle bool properly
        _ => Err(RiallmError::ModelLoading(
            format!("Unsupported safetensors dtype: {:?}", dtype)
        )),
    }
}

/// Convert candle dtype to safetensors dtype
pub fn candle_dtype_to_safetensors(dtype: candle_core::DType) -> Result<safetensors::Dtype> {
    match dtype {
        candle_core::DType::F32 => Ok(safetensors::Dtype::F32),
        candle_core::DType::F16 => Ok(safetensors::Dtype::F16),
        candle_core::DType::BF16 => Ok(safetensors::Dtype::BF16),
        candle_core::DType::I64 => Ok(safetensors::Dtype::I64),
        candle_core::DType::U32 => Ok(safetensors::Dtype::U32),
        candle_core::DType::U8 => Ok(safetensors::Dtype::U8),
        _ => Err(RiallmError::ModelLoading(
            format!("Unsupported candle dtype: {:?}", dtype)
        )),
    }
}

/// Load a single layer from a safetensors file
pub fn load_layer_from_path(layer_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let persister = SafetensorModelPersister::new(device.clone());
    persister.load_layer(layer_path)
}

/// Save a single layer to a safetensors file
pub fn save_layer_to_path(layer_path: &Path, tensors: &HashMap<String, Tensor>) -> Result<()> {
    // Use CPU for saving to avoid GPU memory issues
    let cpu_device = Device::Cpu;
    let persister = SafetensorModelPersister::new(cpu_device);
    persister.save_layer(layer_path, tensors)
}

/// Check if all layers for a model exist in the split path
pub fn check_layers_exist(split_path: &Path, layer_names: &[String]) -> bool {
    layer_names.iter().all(|name| {
        let layer_file = split_path.join(format!("{}.safetensors", name));
        layer_file.exists()
    })
}
