use std::io::{self, Write};
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use riallm::config::{DeviceSpec, ModelOptions};
use riallm::AutoModel;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug)]
struct ChatArgs {
    model: String,
    api_base: Option<String>,
    api_key: String,
    native: bool,
    device: DeviceSpec,
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    system: Option<String>,
    enable_thinking: bool,
}

impl Default for ChatArgs {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen3.6-35B-A3B".to_string(),
            api_base: std::env::var("OPENAI_BASE_URL").ok(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "EMPTY".to_string()),
            native: false,
            device: DeviceSpec::Cpu,
            max_new_tokens: 512,
            temperature: 0.6,
            top_p: 0.95,
            system: None,
            enable_thinking: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    stream: bool,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut argv = std::env::args().skip(1).collect::<Vec<_>>();
    let command = argv.first().map(String::as_str).unwrap_or("help");

    match command {
        "chat" => {
            argv.remove(0);
            let args = parse_chat_args(&argv)?;
            run_chat(args).await
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => anyhow::bail!("unknown command: {other}\n\nRun `riallm help` for usage."),
    }
}

async fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    if args.api_base.is_some() && !args.native {
        run_api_chat(args).await
    } else {
        run_native_chat(args).await
    }
}

async fn run_api_chat(args: ChatArgs) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let api_base = args
        .api_base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--api-base or OPENAI_BASE_URL is required"))?;
    let endpoint = chat_completion_endpoint(api_base);
    let mut messages = initial_messages(&args);

    println!("riallm chat API mode");
    println!("model: {}", args.model);
    println!("endpoint: {}", endpoint);
    println!("type :quit to exit, :reset to clear the conversation");

    loop {
        let Some(input) = read_user_input()? else {
            break;
        };

        if handle_control(&input, &mut messages, &args) {
            continue;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });

        let request = ChatCompletionRequest {
            model: args.model.clone(),
            messages: messages.clone(),
            max_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            stream: false,
            chat_template_kwargs: ChatTemplateKwargs {
                enable_thinking: args.enable_thinking,
            },
        };

        let response = client
            .post(&endpoint)
            .bearer_auth(&args.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("chat completion request failed with {status}: {body}");
        }

        let response = response.json::<ChatCompletionResponse>().await?;
        let reply = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content.or(choice.message.reasoning_content))
            .unwrap_or_default();

        println!("\nassistant> {}\n", reply.trim());
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply,
        });
    }

    Ok(())
}

async fn run_native_chat(args: ChatArgs) -> anyhow::Result<()> {
    let model_path = PathBuf::from(&args.model);
    if !model_path.exists() {
        anyhow::bail!(
            "native chat requires a local model path. For Qwen3.6, start a vLLM/SGLang/Transformers \
             OpenAI-compatible server and pass --api-base or set OPENAI_BASE_URL."
        );
    }

    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| anyhow::anyhow!("failed to load {:?}: {}", tokenizer_path, err))?;

    let options = ModelOptions {
        device: args.device.clone(),
        ..Default::default()
    };
    let mut model = AutoModel::from_pretrained(&args.model, Some(options)).await?;
    let mut messages = initial_messages(&args);

    println!("riallm chat native mode");
    println!("model: {}", args.model);
    println!("type :quit to exit, :reset to clear the conversation");

    loop {
        let Some(input) = read_user_input()? else {
            break;
        };

        if handle_control(&input, &mut messages, &args) {
            continue;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });

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
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply,
        });
    }

    Ok(())
}

fn parse_chat_args(argv: &[String]) -> anyhow::Result<ChatArgs> {
    let mut args = ChatArgs::default();
    let mut index = 0usize;

    while index < argv.len() {
        match argv[index].as_str() {
            "--model" | "-m" => {
                args.model = next_value(argv, &mut index, "--model")?;
            }
            "--api-base" => {
                args.api_base = Some(next_value(argv, &mut index, "--api-base")?);
            }
            "--api-key" => {
                args.api_key = next_value(argv, &mut index, "--api-key")?;
            }
            "--native" => {
                args.native = true;
            }
            "--device" => {
                args.device = parse_device(&next_value(argv, &mut index, "--device")?)?;
            }
            "--max-new-tokens" => {
                args.max_new_tokens =
                    parse_value(&next_value(argv, &mut index, "--max-new-tokens")?)?;
            }
            "--temperature" => {
                args.temperature = parse_value(&next_value(argv, &mut index, "--temperature")?)?;
            }
            "--top-p" => {
                args.top_p = parse_value(&next_value(argv, &mut index, "--top-p")?)?;
            }
            "--system" => {
                args.system = Some(next_value(argv, &mut index, "--system")?);
            }
            "--no-thinking" => {
                args.enable_thinking = false;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown chat argument: {other}"),
        }

        index += 1;
    }

    Ok(args)
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

fn initial_messages(args: &ChatArgs) -> Vec<ChatMessage> {
    args.system
        .as_ref()
        .map(|content| {
            vec![ChatMessage {
                role: "system".to_string(),
                content: content.clone(),
            }]
        })
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

fn handle_control(input: &str, messages: &mut Vec<ChatMessage>, args: &ChatArgs) -> bool {
    match input {
        "" => true,
        ":quit" | ":exit" => std::process::exit(0),
        ":reset" => {
            *messages = initial_messages(args);
            println!("conversation reset");
            true
        }
        _ => false,
    }
}

fn chat_completion_endpoint(api_base: &str) -> String {
    let base = api_base.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        base.to_string()
    } else {
        format!("{base}/chat/completions")
    }
}

fn render_qwen_chat_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
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

fn print_usage() {
    println!(
        "riallm\n\n\
         Usage:\n  \
           riallm chat --model /path/to/local/model [--native]\n  \
           riallm chat --model Qwen/Qwen3.6-35B-A3B --api-base http://127.0.0.1:8000/v1\n\n\
         Chat options:\n  \
           --model, -m <MODEL>          Local model path or API model id\n  \
           --api-base <URL>             OpenAI-compatible base URL, also reads OPENAI_BASE_URL\n  \
           --api-key <KEY>              API key, also reads OPENAI_API_KEY, defaults to EMPTY\n  \
           --native                     Force native local riallm inference\n  \
           --device <DEVICE>            cpu, metal, cuda, or cuda:N for native mode\n  \
           --max-new-tokens <N>         Default: 512\n  \
           --temperature <FLOAT>        Default: 0.6\n  \
           --top-p <FLOAT>              Default: 0.95\n  \
           --system <TEXT>              Optional system message\n  \
           --no-thinking                Request no thinking blocks when the backend supports it"
    );
}
