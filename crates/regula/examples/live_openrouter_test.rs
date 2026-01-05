//! Live OpenRouter Integration Test
//!
//! This example tests the REGULA framework with a real LLM call to OpenRouter.
//! It demonstrates the simple chat pattern with actual API responses.
//!
//! # Running
//!
//! ```bash
//! # Set your OpenRouter API key
//! set OPENROUTER_API_KEY=your-api-key
//!
//! # Or pass it interactively when prompted
//! cargo run --example live_openrouter_test
//! ```

use regula::prelude::*;
use std::env;
use std::io::{self, Write};
use std::time::Duration;

/// The state for our test chat.
#[derive(Clone, Default, Debug, Serialize, Deserialize, GraphState)]
struct ChatState {
    /// The conversation history
    messages: Vec<Message>,
    /// The response content (for verification)
    response_content: String,
}

/// Get the API key from environment or prompt the user
fn get_api_key() -> String {
    // Try environment variable first
    if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        if !key.is_empty() {
            println!("[OK] Using API key from OPENROUTER_API_KEY environment variable");
            return key;
        }
    }

    // Prompt user for API key
    print!("Enter your OpenRouter API key: ");
    io::stdout().flush().unwrap();
    
    let mut key = String::new();
    io::stdin().read_line(&mut key).unwrap();
    key.trim().to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("================================================================");
    println!("       REGULA Live OpenRouter Integration Test                  ");
    println!("================================================================");
    println!();

    // Get the API key
    let api_key = get_api_key();
    if api_key.is_empty() {
        eprintln!("[ERROR] No API key provided");
        std::process::exit(1);
    }

    println!();
    println!("Building graph...");

    // Create LLM config for OpenRouter
    // OpenRouter uses the OpenAI-compatible API format
    let llm_config = LlmConfig::openai(&api_key)
        .with_base_url("https://openrouter.ai/api/v1")
        .with_model("xiaomi/mimo-v2-flash:free") // Free model for testing
        .with_timeout(Duration::from_secs(30));

    // Build the state graph
    let graph = StateGraph::<ChatState>::new()
        .add_node("agent", node_fn(move |state: &ChatState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            let config = llm_config.clone();
            async move {
                println!("  -> Calling OpenRouter API...");
                
                let client = OpenAiClient::new(config);
                let response = client.complete(&messages).await?;
                
                let content = response.message.content.clone();
                println!("  <- Received response ({} chars)", content.len());
                
                let mut new_messages = messages.clone();
                new_messages.push(response.message);
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "response_content": content
                })))
            }
        }))
        .add_edge(start(), "agent")
        .add_edge("agent", end())
        .compile(Default::default())?;

    println!("[OK] Graph compiled successfully");
    println!();

    // Create the executor
    let executor = GraphExecutor::new(graph);

    // Test message
    let test_prompt = "What is 2 + 2? Reply with just the number.";
    println!("Test prompt: \"{}\"", test_prompt);
    println!();

    // Create initial state with a simple test message
    let initial_state = ChatState {
        messages: vec![
            Message::system("You are a helpful assistant. Keep responses brief."),
            Message::user(test_prompt),
        ],
        response_content: String::new(),
    };

    println!("Executing graph...");
    println!("----------------------------------------------------------------");

    // Execute the graph
    let start_time = std::time::Instant::now();
    let result = executor
        .invoke(initial_state, RunnableConfig::new())
        .await;
    let elapsed = start_time.elapsed();

    println!("----------------------------------------------------------------");
    println!();

    match result {
        Ok(final_state) => {
            println!("================================================================");
            println!("                      TEST RESULTS                              ");
            println!("================================================================");
            println!();
            println!("[OK] Execution completed successfully!");
            println!();
            println!("Response content:");
            println!("  \"{}\"", final_state.response_content.trim());
            println!();
            println!("Message count: {}", final_state.messages.len());
            println!("Execution time: {:?}", elapsed);
            println!();
            
            // Verify we got a response
            if final_state.response_content.is_empty() {
                println!("[WARN] Response content is empty");
            } else {
                println!("[OK] Response content received");
            }
            
            // Check if the response contains "4" (expected answer)
            if final_state.response_content.contains("4") {
                println!("[OK] Response contains expected answer (4)");
            } else {
                println!("[WARN] Response may not contain expected answer");
            }
            
            println!();
            println!("================================================================");
            println!("                    TEST PASSED                                 ");
            println!("================================================================");
        }
        Err(e) => {
            println!("================================================================");
            println!("                      TEST FAILED                               ");
            println!("================================================================");
            println!();
            println!("[ERROR] {}", e);
            println!();
            println!("Possible causes:");
            println!("  - Invalid API key");
            println!("  - Network connectivity issues");
            println!("  - OpenRouter service unavailable");
            println!("  - Rate limiting");
            println!();
            std::process::exit(1);
        }
    }

    Ok(())
}
