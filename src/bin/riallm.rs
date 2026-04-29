use std::io::{self, Write};
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use riallm::config::{DeviceSpec, ModelOptions};
use riallm::AutoModel;
use tokenizers::Tokenizer;

#[derive(Debug)]
struct ChatArgs {
    model_path: PathBuf,
    device: DeviceSpec,
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    system: Option<String>,
    enable_thinking: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut argv = std::env::args().skip(1).collect::<Vec<_>>();
    let command = argv.first().map(String::as_str).unwrap_or("help");

    match command {
        "chat" => {
            argv.remove(0);
            run_chat(parse_chat_args(&argv)?).await
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => anyhow::bail!("unknown command: {other}\n\nRun `riallm help` for usage."),
    }
}

async fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    if !args.model_path.exists() {
        anyhow::bail!("model path does not exist: {:?}", args.model_path);
    }

    let tokenizer_path = args.model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| anyhow::anyhow!("failed to load {:?}: {}", tokenizer_path, err))?;

    let options = ModelOptions {
        device: args.device.clone(),
        ..Default::default()
    };
    let mut model =
        AutoModel::from_pretrained(path_to_str(&args.model_path)?, Some(options)).await?;
    let mut messages = initial_messages(&args);

    println!("riallm native chat");
    println!("model: {}", args.model_path.display());
    println!("type :quit to exit, :reset to clear the conversation");

    loop {
        let Some(input) = read_user_input()? else {
            break;
        };

        match input.as_str() {
            "" => continue,
            ":quit" | ":exit" => break,
            ":reset" => {
                messages = initial_messages(&args);
                println!("conversation reset");
                continue;
            }
            _ => {}
        }

        messages.push(("user".to_string(), input));

        let prompt = render_qwen_chat_prompt(&messages, args.enable_thinking);
        let encoding = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {}", err))?;
        let input_ids =
            Tensor::new(encoding.get_ids(), &Device::Cpu)?.reshape((1, encoding.len()))?;

        model.reset_kv_cache();
        let output_tokens = model.generate(
            &input_ids,
            args.max_new_tokens,
            args.temperature,
            Some(args.top_p),
        )?;

        let prompt_len = encoding.get_ids().len();
        let new_tokens = &output_tokens[prompt_len.min(output_tokens.len())..];
        let reply = tokenizer
            .decode(new_tokens, true)
            .map_err(|err| anyhow::anyhow!("decoding failed: {}", err))?;
        let reply = trim_generation_stops(&reply).trim().to_string();

        println!("\nassistant> {}\n", reply);
        messages.push(("assistant".to_string(), reply));
    }

    Ok(())
}

fn parse_chat_args(argv: &[String]) -> anyhow::Result<ChatArgs> {
    let mut model_path = std::env::var("RIALLM_MODEL_PATH").ok().map(PathBuf::from);
    let mut device = DeviceSpec::Cpu;
    let mut max_new_tokens = 128usize;
    let mut temperature = 0.0f64;
    let mut top_p = 1.0f64;
    let mut system = None;
    let mut enable_thinking = true;
    let mut index = 0usize;

    while index < argv.len() {
        match argv[index].as_str() {
            "--model" | "-m" => {
                model_path = Some(PathBuf::from(next_value(argv, &mut index, "--model")?));
            }
            "--device" => {
                device = parse_device(&next_value(argv, &mut index, "--device")?)?;
            }
            "--max-new-tokens" => {
                max_new_tokens = parse_value(&next_value(argv, &mut index, "--max-new-tokens")?)?;
            }
            "--temperature" => {
                temperature = parse_value(&next_value(argv, &mut index, "--temperature")?)?;
            }
            "--top-p" => {
                top_p = parse_value(&next_value(argv, &mut index, "--top-p")?)?;
            }
            "--system" => {
                system = Some(next_value(argv, &mut index, "--system")?);
            }
            "--no-thinking" => {
                enable_thinking = false;
            }
            "--native" => {}
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown chat argument: {other}"),
        }

        index += 1;
    }

    let model_path = model_path.ok_or_else(|| {
        anyhow::anyhow!("--model /path/to/local/model or RIALLM_MODEL_PATH is required")
    })?;

    Ok(ChatArgs {
        model_path,
        device,
        max_new_tokens,
        temperature,
        top_p,
        system,
        enable_thinking,
    })
}

fn next_value(argv: &[String], index: &mut usize, flag: &str) -> anyhow::Result<String> {
    *index += 1;
    argv.get(*index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn parse_value<T>(value: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value.parse::<T>().map_err(Into::into)
}

fn parse_device(value: &str) -> anyhow::Result<DeviceSpec> {
    match value {
        "cpu" => Ok(DeviceSpec::Cpu),
        "metal" => Ok(DeviceSpec::Metal),
        "cuda" => Ok(DeviceSpec::Cuda(0)),
        other if other.starts_with("cuda:") => {
            let id = other.trim_start_matches("cuda:").parse::<usize>()?;
            Ok(DeviceSpec::Cuda(id))
        }
        other => anyhow::bail!("invalid device `{other}`; expected cpu, metal, cuda, or cuda:N"),
    }
}

fn initial_messages(args: &ChatArgs) -> Vec<(String, String)> {
    args.system
        .as_ref()
        .map(|content| vec![("system".to_string(), content.clone())])
        .unwrap_or_default()
}

fn read_user_input() -> anyhow::Result<Option<String>> {
    print!("riallm> ");
    io::stdout().flush()?;

    let mut input = String::new();
    let bytes = io::stdin().read_line(&mut input)?;
    if bytes == 0 {
        return Ok(None);
    }

    Ok(Some(input.trim().to_string()))
}

fn render_qwen_chat_prompt(messages: &[(String, String)], enable_thinking: bool) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

fn trim_generation_stops(text: &str) -> &str {
    let mut end = text.len();
    for marker in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(index) = text.find(marker) {
            end = end.min(index);
        }
    }
    &text[..end]
}

fn path_to_str(path: &PathBuf) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8: {:?}", path))
}

fn print_usage() {
    println!(
        "riallm\n\n\
         Usage:\n  \
           riallm chat --model /path/to/local/model [--device cpu]\n\n\
         Chat options:\n  \
           --model, -m <PATH>           Local Hugging Face model directory\n  \
           --device <DEVICE>            cpu, metal, cuda, or cuda:N\n  \
           --max-new-tokens <N>         Default: 128\n  \
           --temperature <FLOAT>        Parsed for API compatibility; generation is greedy today\n  \
           --top-p <FLOAT>              Parsed for API compatibility; generation is greedy today\n  \
           --system <TEXT>              Optional system message\n  \
           --no-thinking                Insert Qwen no-thinking prompt marker"
    );
}
