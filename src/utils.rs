//! Model splitting utilities - split a model into per-layer shards

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use crate::config::{LayerNames, ModelConfig};
use crate::error::{Result, RiallmError};

struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Split a model checkpoint into per-layer safetensor files
pub struct ModelSplitter {
    /// Source model path
    source_path: PathBuf,

    /// Destination path for split layers
    dest_path: PathBuf,

    /// Layer name configuration
    layer_names: LayerNames,

    /// Model configuration
    config: ModelConfig,
}

impl ModelSplitter {
    pub fn new(
        source_path: PathBuf,
        dest_path: PathBuf,
        layer_names: LayerNames,
        config: ModelConfig,
    ) -> Self {
        Self {
            source_path,
            dest_path,
            layer_names,
            config,
        }
    }

    /// Split the model into per-layer shards
    pub fn split(&self) -> Result<()> {
        println!(
            "Splitting model from {:?} to {:?}",
            self.source_path, self.dest_path
        );

        // Ensure destination directory exists
        std::fs::create_dir_all(&self.dest_path)?;

        let mut safetensor_files = Vec::new();
        for entry in std::fs::read_dir(&self.source_path)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                safetensor_files.push(path);
            }
        }
        safetensor_files.sort();

        if safetensor_files.is_empty() {
            return Err(RiallmError::ModelLoading(format!(
                "No .safetensors files found in {:?}",
                self.source_path
            )));
        }

        let mut grouped: BTreeMap<String, BTreeMap<String, RawTensor>> = BTreeMap::new();
        let mut tensor_count = 0usize;
        let mut ignored_count = 0usize;

        for path in safetensor_files {
            let file = std::fs::File::open(&path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, view) in safetensors.tensors() {
                if let Some((layer_name, tensor_name)) = self.classify_tensor(&name) {
                    grouped.entry(layer_name).or_default().insert(
                        tensor_name,
                        RawTensor {
                            dtype: view.dtype(),
                            shape: view.shape().to_vec(),
                            data: view.data().to_vec(),
                        },
                    );
                    tensor_count += 1;
                } else {
                    ignored_count += 1;
                }
            }
        }

        if tensor_count == 0 {
            return Err(RiallmError::ModelLoading(format!(
                "No tensors matched the {:?} layer layout",
                self.layer_names
            )));
        }

        for (layer_name, tensors) in &grouped {
            self.write_layer(layer_name, tensors)?;
        }

        let missing_layers = self
            .required_layer_names()
            .into_iter()
            .filter(|name| {
                !self
                    .dest_path
                    .join(format!("{}.safetensors", name))
                    .exists()
            })
            .collect::<Vec<_>>();

        if !missing_layers.is_empty() {
            return Err(RiallmError::ModelLoading(format!(
                "Split completed but required layer shards are missing: {}",
                missing_layers.join(", ")
            )));
        }

        let metadata = serde_json::json!({
            "source_path": self.source_path,
            "model_type": self.config.model_type,
            "num_hidden_layers": self.config.num_hidden_layers,
            "tensor_count": tensor_count,
            "ignored_tensor_count": ignored_count,
        });
        std::fs::write(
            self.dest_path.join("split_metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;

        println!(
            "Split {} tensors into {} layer shards (ignored {} non-text/unmatched tensors)",
            tensor_count,
            grouped.len(),
            ignored_count
        );

        Ok(())
    }

    fn classify_tensor(&self, name: &str) -> Option<(String, String)> {
        if let Some(tensor_name) = strip_named_prefix(name, &self.layer_names.embed) {
            return Some(("embed".to_string(), tensor_name));
        }

        if let Some(tensor_name) = strip_named_prefix(name, &self.layer_names.norm) {
            return Some(("final_norm".to_string(), tensor_name));
        }

        if let Some(tensor_name) = strip_named_prefix(name, &self.layer_names.lm_head) {
            return Some(("lm_head".to_string(), tensor_name));
        }

        let rest = name.strip_prefix(&self.layer_names.layer_prefix)?;
        let (layer_index, tensor_name) = rest.split_once('.')?;
        let layer_index = layer_index.parse::<usize>().ok()?;

        if layer_index < self.config.num_hidden_layers {
            Some((format!("layer_{}", layer_index), tensor_name.to_string()))
        } else {
            None
        }
    }

    fn write_layer(&self, layer_name: &str, tensors: &BTreeMap<String, RawTensor>) -> Result<()> {
        let layer_path = self.dest_path.join(format!("{}.safetensors", layer_name));
        let views = tensors
            .iter()
            .map(|(name, tensor)| {
                TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.data.as_slice())
                    .map(|view| (name.as_str(), view))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        safetensors::serialize_to_file(views, &None, &layer_path)?;
        Ok(())
    }

    fn required_layer_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.config.num_hidden_layers + 3);
        names.push("embed".to_string());
        for index in 0..self.config.num_hidden_layers {
            names.push(format!("layer_{}", index));
        }
        names.push("final_norm".to_string());
        names.push("lm_head".to_string());
        names
    }
}

fn strip_named_prefix(name: &str, prefix: &str) -> Option<String> {
    let full_prefix = format!("{}.", prefix);
    name.strip_prefix(&full_prefix).map(ToString::to_string)
}

/// Find or create the local split path for a model
pub fn find_or_create_split_path(model_id: &str, cache_dir: Option<PathBuf>) -> Result<PathBuf> {
    // Determine cache directory
    let cache = cache_dir.unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("riallm")
    });

    // Create model-specific directory
    let model_dir = cache.join(model_id.replace('/', "_"));
    let split_dir = model_dir.join("split");

    // Check if already split
    if split_dir.exists() && split_dir.join("split_metadata.json").exists() {
        println!("Using existing split model at {:?}", split_dir);
        return Ok(split_dir);
    }

    std::fs::create_dir_all(&model_dir)?;

    Ok(split_dir)
}

/// Check if there's enough disk space for splitting
pub fn check_disk_space(path: &Path, required_bytes: u64) -> Result<()> {
    let available = disk_free_space(path)?;

    if available < required_bytes {
        return Err(RiallmError::Memory(format!(
            "Not enough disk space. Required: {} GB, Available: {} GB",
            required_bytes / 1_073_741_824,
            available / 1_073_741_824
        )));
    }

    Ok(())
}

/// Get free disk space in bytes
fn disk_free_space(path: &Path) -> Result<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;

        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| RiallmError::Config("Invalid path".to_string()))?,
        )?;

        let mut statfs = unsafe { mem::zeroed::<libc::statfs>() };

        if unsafe { libc::statfs(c_path.as_ptr(), &mut statfs) } != 0 {
            return Err(RiallmError::Io(std::io::Error::last_os_error()));
        }

        Ok(statfs.f_bavail * statfs.f_bsize as u64)
    }

    #[cfg(not(unix))]
    {
        // Fallback: assume plenty of space
        let _ = path;
        Ok(u64::MAX)
    }
}

/// Split a model (placeholder - full implementation needs safetensors parsing)
pub fn split_model(
    model_path: PathBuf,
    split_path: PathBuf,
    layer_names: &LayerNames,
    config: &ModelConfig,
) -> Result<()> {
    let splitter = ModelSplitter::new(model_path, split_path, layer_names.clone(), config.clone());

    splitter.split()
}
