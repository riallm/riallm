//! Example usage of riallm library

use riallm::config::ModelOptions;
use riallm::AutoModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("riallm - Memory-Optimized LLM Inference");
    println!("{}", "=".repeat(60));

    // Example 1: Load a model (requires pre-split model)
    // Note: Replace with actual model path
    let model_path =
        std::env::var("RIALLM_MODEL_PATH").unwrap_or_else(|_| "/tmp/models/llama-2-7b".to_string());

    println!("\nAttempting to load model from: {}", model_path);

    // Check if model exists
    if !std::path::Path::new(&model_path).exists() {
        println!("\nModel not found at {}. Please:", model_path);
        println!("1. Download a HuggingFace model");
        println!("2. Place it in the specified path");
        println!("3. The model will be automatically split into layers");
        println!("\nFor testing, you can use:");
        println!("  export RIALLM_MODEL_PATH=/path/to/your/model");
        return Ok(());
    }

    // Configure model loading
    let options = ModelOptions {
        profiling_mode: true,
        prefetch_layers: true,
        ..Default::default()
    };

    // Load model
    println!("\nLoading model...");
    let mut model = AutoModel::from_pretrained(&model_path, Some(options)).await?;

    // Example 2: Generate text
    println!("\n{}", "=".repeat(60));
    println!("Example Text Generation");
    println!("{}", "=".repeat(60));

    // Note: This requires a tokenizer which should be integrated
    // For now, we'll demonstrate the API structure
    println!("\nText generation example (requires tokenizer integration):");
    println!(
        "
    // Tokenize input
    let input_text = \"Hello, world!\";
    let input_tokens = tokenizer.encode(input_text, true)?;
    let input_ids = Tensor::new(input_tokens.get_ids(), &device)?;
    
    // Generate
    let output_tokens = model.generate(
        &input_ids,
        50,     // max_new_tokens
        0.7,    // temperature
        Some(0.9), // top_p
    )?;
    
    // Decode output
    let output_text = tokenizer.decode(&output_tokens, true)?;
    println!(\"Generated: {}\", output_text);
    "
    );

    // Example 3: Profiling
    if let Some(profiler) = model.profiler() {
        println!("\n{}", "=".repeat(60));
        println!("Profiling Results");
        println!("{}", "=".repeat(60));
        profiler.print_summary();
    }

    println!("\n{}", "=".repeat(60));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(60));

    Ok(())
}
