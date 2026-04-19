//! Profiler utilities for timing and memory tracking

use std::collections::HashMap;
use std::time::Instant;

/// Statistics for a single layer
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Time spent in forward pass (milliseconds)
    pub forward_time_ms: Vec<f64>,

    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,

    /// Number of times layer was executed
    pub execution_count: usize,
}

impl LayerStats {
    pub fn new() -> Self {
        Self {
            forward_time_ms: Vec::new(),
            peak_memory_mb: 0.0,
            execution_count: 0,
        }
    }

    /// Add a forward pass timing
    pub fn add_timing(&mut self, time_ms: f64) {
        self.forward_time_ms.push(time_ms);
        self.execution_count += 1;
    }

    /// Get average forward time
    pub fn avg_forward_time_ms(&self) -> f64 {
        if self.forward_time_ms.is_empty() {
            0.0
        } else {
            self.forward_time_ms.iter().sum::<f64>() / self.forward_time_ms.len() as f64
        }
    }

    /// Get total forward time
    pub fn total_forward_time_ms(&self) -> f64 {
        self.forward_time_ms.iter().sum()
    }
}

/// Profiler for tracking model performance
pub struct Profiler {
    /// Per-layer statistics
    layer_stats: HashMap<String, LayerStats>,

    /// Current layer being timed
    current_layer: Option<String>,

    /// Start time for current operation
    start_time: Option<Instant>,

    /// Overall start time
    overall_start: Instant,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            layer_stats: HashMap::new(),
            current_layer: None,
            start_time: None,
            overall_start: Instant::now(),
        }
    }

    /// Start timing a layer
    pub fn start_layer(&mut self, layer_name: &str) {
        self.current_layer = Some(layer_name.to_string());
        self.start_time = Some(Instant::now());
    }

    /// End timing a layer
    pub fn end_layer(&mut self, layer_name: &str) {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms

            let stats = self
                .layer_stats
                .entry(layer_name.to_string())
                .or_insert_with(LayerStats::new);

            stats.add_timing(elapsed);
        }

        self.current_layer = None;
        self.start_time = None;
    }

    /// Record memory usage for a layer
    pub fn record_memory(&mut self, layer_name: &str, memory_mb: f64) {
        let stats = self
            .layer_stats
            .entry(layer_name.to_string())
            .or_insert_with(LayerStats::new);

        stats.peak_memory_mb = stats.peak_memory_mb.max(memory_mb);
    }

    /// Get statistics for a specific layer
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_stats(&self) -> &HashMap<String, LayerStats> {
        &self.layer_stats
    }

    /// Print profiling summary
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("AirLLM Profiling Summary");
        println!("{}", "=".repeat(80));

        let total_time = self.overall_start.elapsed().as_secs_f64();

        println!("\nOverall Time: {:.2} seconds\n", total_time);

        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>10}",
            "Layer", "Count", "Avg (ms)", "Total (ms)", "Peak (MB)"
        );
        println!("{}", "-".repeat(80));

        let mut total_ms = 0.0;

        // Sort layers by total time
        let mut layers: Vec<_> = self.layer_stats.iter().collect();
        layers.sort_by(|a, b| {
            b.1.total_forward_time_ms()
                .partial_cmp(&a.1.total_forward_time_ms())
                .unwrap()
        });

        for (name, stats) in layers {
            println!(
                "{:<20} {:>10} {:>12.2} {:>12.2} {:>10.2}",
                name,
                stats.execution_count,
                stats.avg_forward_time_ms(),
                stats.total_forward_time_ms(),
                stats.peak_memory_mb,
            );

            total_ms += stats.total_forward_time_ms();
        }

        println!("{}", "-".repeat(80));
        println!("{:<20} {:>10} {:>12.2} {:>12}", "TOTAL", "", total_ms, "");
        println!("\n{}", "=".repeat(80));
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.current_layer = None;
        self.start_time = None;
        self.overall_start = Instant::now();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}
