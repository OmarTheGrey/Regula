//! Simple Chat Example
//!
//! This example demonstrates a basic chatbot using REGULA.
//! It creates a simple graph with a single agent node that processes
//! user messages and generates responses using an LLM.
//!
//! # Running
//!
//! ```bash
//! # Set your API key
//! export OPENAI_API_KEY="your-api-key"
//!
//! # Run the example
//! cargo run --example simple_chat
//! ```

use regula::prelude::*;
use std::io::{self, Write};

/// The state for our simple chat agent.
#[derive(Clone, Default, Serialize, Deserialize, GraphState)]
struct ChatState {
    /// The conversation history
    messages: Vec<Message>,
    
    /// Whether the conversation has ended
    done: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== REGULA Simple Chat Example ===\n");
    println!("This is a simple chatbot demo. Type your message and press Enter.");
    println!("Type 'quit' to exit.\n");

    // Build the state graph
    let graph = StateGraph::<ChatState>::new()
        .add_node("agent", node_fn(|state: &ChatState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            async move {
                // Create an OpenAI client
                let client = OpenAiClient::new(LlmConfig::default());
                
                // Call the LLM with the conversation history
                let response = client.complete(&messages).await?;
                
                // Add the assistant's response to the messages
                let mut new_messages = messages.clone();
                new_messages.push(response.message);
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages
                })))
            }
        }))
        .add_edge(start(), "agent")
        .add_edge("agent", end())
        .compile(Default::default())?;

    // Create the executor
    let executor = GraphExecutor::new(graph);

    // Chat loop
    loop {
        // Get user input
        print!("You: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("quit") {
            println!("\nGoodbye!");
            break;
        }
        
        if input.is_empty() {
            continue;
        }
        
        // Create the initial state with the user's message
        let initial_state = ChatState {
            messages: vec![
                Message::system("You are a helpful AI assistant. Be concise and friendly."),
                Message::user(input),
            ],
            done: false,
        };
        
        // Run the graph
        let config = RunnableConfig::new();
        let result = executor.invoke(initial_state, config).await?;
        
        // Print the assistant's response
        if let Some(last_message) = result.messages.last() {
            if last_message.role == Role::Assistant {
                println!("Assistant: {}\n", &last_message.content);
            }
        }
    }

    Ok(())
}
