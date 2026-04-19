//! Memory management utilities for GPU/CPU tensor operations

use candle_core::{Device, Tensor};

use crate::error::{Result, RiallmError};

/// Memory statistics for a device
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory in bytes
    pub total: u64,

    /// Used memory in bytes
    pub used: u64,

    /// Free memory in bytes
    pub free: u64,
}

impl MemoryStats {
    pub fn new(total: u64, used: u64, free: u64) -> Self {
        Self { total, used, free }
    }

    /// Get memory usage as percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Format bytes to human-readable string
    pub fn format_bytes(bytes: u64) -> String {
        if bytes >= 1_073_741_824 {
            format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.2} MB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Print memory statistics
    pub fn print(&self, device_name: &str) {
        println!(
            "{} Memory: {} / {} ({:.1}%)",
            device_name,
            Self::format_bytes(self.used),
            Self::format_bytes(self.total),
            self.usage_percent()
        );
    }
}

/// Get memory statistics for a device
pub fn get_memory_stats(device: &Device) -> Result<MemoryStats> {
    match device {
        Device::Cpu => {
            // CPU memory tracking is not straightforward
            // Return placeholder values
            Ok(MemoryStats::new(u64::MAX, 0, u64::MAX))
        }
        Device::Cuda(_id) => {
            // TODO: Implement CUDA memory tracking
            // This would require candle CUDA memory API
            Ok(MemoryStats::new(u64::MAX, 0, u64::MAX))
        }
        Device::Metal(_id) => {
            // Metal memory tracking
            Ok(MemoryStats::new(u64::MAX, 0, u64::MAX))
        }
    }
}

/// Clean memory by triggering garbage collection and clearing caches
pub fn clean_memory() -> Result<()> {
    // Trigger Rust garbage collection (not really needed in Rust, but useful for completeness)
    // In Rust, memory is freed deterministically when values go out of scope

    // For CUDA devices, we can clear the cache
    // TODO: Implement CUDA cache clearing when candle supports it
    // candle_core::cuda::empty_cache()?;

    Ok(())
}

/// Calculate tensor memory requirements in bytes
pub fn tensor_memory_bytes(tensor: &Tensor) -> Result<u64> {
    let elem_count = tensor.shape().elem_count();
    let dtype_size = dtype_size_bytes(tensor.dtype());

    Ok((elem_count * dtype_size) as u64)
}

/// Get size of a data type in bytes
fn dtype_size_bytes(dtype: candle_core::DType) -> usize {
    match dtype {
        candle_core::DType::U8 => 1,
        candle_core::DType::U32 => 4,
        candle_core::DType::I64 => 8,
        candle_core::DType::BF16 => 2,
        candle_core::DType::F32 => 4,
        candle_core::DType::F16 => 2,
        candle_core::DType::F64 => 8,
    }
}

/// Move tensors from CPU to GPU
pub fn move_tensors_to_device(
    tensors: &std::collections::HashMap<String, Tensor>,
    device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let mut gpu_tensors = std::collections::HashMap::new();

    for (name, tensor) in tensors {
        gpu_tensors.insert(name.clone(), tensor.to_device(device)?);
    }

    Ok(gpu_tensors)
}

/// Move tensors from GPU to CPU
pub fn move_tensors_to_cpu(
    tensors: &std::collections::HashMap<String, Tensor>,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let cpu_device = Device::Cpu;
    move_tensors_to_device(tensors, &cpu_device)
}

/// Calculate total memory for a set of tensors
pub fn total_tensor_memory(tensors: &std::collections::HashMap<String, Tensor>) -> Result<u64> {
    let mut total = 0u64;

    for tensor in tensors.values() {
        total += tensor_memory_bytes(tensor)?;
    }

    Ok(total)
}

/// Check if there's enough memory on the device
pub fn check_device_memory(
    tensors: &std::collections::HashMap<String, Tensor>,
    device: &Device,
) -> Result<bool> {
    let required_memory = total_tensor_memory(tensors)?;
    let stats = get_memory_stats(device)?;

    Ok(stats.free >= required_memory)
}

/// Print detailed memory information
pub fn print_memory_info(
    tensors: &std::collections::HashMap<String, Tensor>,
    device_name: &str,
) -> Result<()> {
    println!("\n{} Memory Details:", device_name);
    println!("{}", "-".repeat(60));

    let mut total_memory = 0u64;

    for (name, tensor) in tensors {
        let mem = tensor_memory_bytes(tensor)?;
        total_memory += mem;
        println!("{:<40} {}", name, MemoryStats::format_bytes(mem));
    }

    println!("{}", "-".repeat(60));
    println!(
        "{:<40} {}\n",
        "TOTAL",
        MemoryStats::format_bytes(total_memory)
    );

    Ok(())
}

/// Synchronize device operations (useful before memory measurements)
pub fn synchronize_device(device: &Device) -> Result<()> {
    match device {
        Device::Cpu => {
            // CPU is always synchronous
            Ok(())
        }
        Device::Cuda(_id) => {
            // TODO: CUDA synchronization
            Ok(())
        }
        Device::Metal(_id) => {
            // TODO: Metal synchronization
            Ok(())
        }
    }
}

/// Memory tracker for monitoring usage over time
pub struct MemoryTracker {
    /// Peak memory usage
    peak_memory: u64,

    /// Current memory usage
    current_memory: u64,

    /// History of memory allocations
    history: Vec<(String, u64)>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            history: Vec::new(),
        }
    }

    /// Track a memory allocation
    pub fn track_allocation(&mut self, name: &str, bytes: u64) {
        self.current_memory += bytes;
        self.peak_memory = self.peak_memory.max(self.current_memory);
        self.history.push((name.to_string(), bytes));
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&mut self, name: &str, bytes: u64) {
        self.current_memory = self.current_memory.saturating_sub(bytes);
        self.history.push((format!("[free] {}", name), bytes));
    }

    /// Get peak memory usage
    pub fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    /// Get current memory usage
    pub fn current_memory(&self) -> u64 {
        self.current_memory
    }

    /// Get allocation history
    pub fn history(&self) -> &[(String, u64)] {
        &self.history
    }

    /// Print memory tracking summary
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Memory Tracking Summary");
        println!("{}", "=".repeat(60));
        println!(
            "Peak Memory: {}",
            MemoryStats::format_bytes(self.peak_memory)
        );
        println!(
            "Current Memory: {}",
            MemoryStats::format_bytes(self.current_memory)
        );
        println!("\nAllocation History:");

        for (name, bytes) in &self.history {
            println!("  {:<40} {}", name, MemoryStats::format_bytes(*bytes));
        }

        println!("{}", "=".repeat(60));
    }

    /// Reset tracking data
    pub fn reset(&mut self) {
        self.peak_memory = 0;
        self.current_memory = 0;
        self.history.clear();
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}
