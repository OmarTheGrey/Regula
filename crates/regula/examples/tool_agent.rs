//! Tool Agent Example (ReAct Pattern)
//!
//! This example demonstrates a ReAct-style agent that can use tools.
//! The agent decides whether to call a tool or respond directly,
//! and loops until it has a final answer.
//!
//! # Running
//!
//! ```bash
//! # Set your API key
//! export OPENAI_API_KEY="your-api-key"
//!
//! # Run the example
//! cargo run --example tool_agent
//! ```

use regula::prelude::*;

/// Tool call that the agent can make
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// The state for our tool-calling agent.
#[derive(Clone, Default, Serialize, Deserialize, GraphState)]
struct AgentState {
    /// The conversation history
    messages: Vec<Message>,
    
    /// Pending tool calls to execute
    #[reducer(append)]
    pending_tool_calls: Vec<PendingToolCall>,
    
    /// Iteration count to prevent infinite loops
    iteration: u32,
}

/// Available tools for the agent
fn get_tools() -> Vec<Tool> {
    vec![
        Tool::function(
            "get_weather",
            "Get the current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        ),
        Tool::function(
            "calculate",
            "Perform a mathematical calculation",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate, e.g., '2 + 2'"
                    }
                },
                "required": ["expression"]
            }),
        ),
    ]
}

/// Execute a tool call and return the result
fn execute_tool(name: &str, arguments: &str) -> String {
    match name {
        "get_weather" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                let location = args["location"].as_str().unwrap_or("Unknown");
                format!("The weather in {} is sunny, 72°F (22°C) with light winds.", location)
            } else {
                "Error: Could not parse location argument".to_string()
            }
        }
        "calculate" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                let expr = args["expression"].as_str().unwrap_or("");
                match expr {
                    "2 + 2" => "4".to_string(),
                    "10 * 5" => "50".to_string(),
                    "100 / 4" => "25".to_string(),
                    _ => format!("Evaluated: {} = [result]", expr),
                }
            } else {
                "Error: Could not parse expression argument".to_string()
            }
        }
        _ => format!("Unknown tool: {}", name),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== REGULA Tool Agent Example (ReAct Pattern) ===\n");

    // Build the state graph
    let graph = StateGraph::<AgentState>::new()
        .add_node("agent", node_fn(|state: &AgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            let iteration = state.iteration;
            async move {
                println!("  [Agent] Thinking... (iteration {})", iteration + 1);
                
                let client = OpenAiClient::new(LlmConfig::default());
                let tools = get_tools();
                let response = client.complete_with_tools(&messages, &tools).await?;
                
                // Check if there are tool calls
                let pending_tool_calls: Vec<PendingToolCall> = response.message.tool_calls
                    .as_ref()
                    .map(|tcs| tcs.iter().map(|tc| PendingToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    }).collect())
                    .unwrap_or_default();
                
                let mut new_messages = messages.clone();
                new_messages.push(response.message);
                
                if !pending_tool_calls.is_empty() {
                    println!("  [Agent] Decided to call {} tool(s)", pending_tool_calls.len());
                } else {
                    println!("  [Agent] Generated final response");
                }
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "pending_tool_calls": pending_tool_calls,
                    "iteration": iteration + 1
                })))
            }
        }))
        .add_node("tools", node_fn(|state: &AgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            let pending_tool_calls = state.pending_tool_calls.clone();
            async move {
                println!("  [Tools] Executing {} tool call(s)", pending_tool_calls.len());
                
                let mut new_messages = messages.clone();
                
                for tool_call in &pending_tool_calls {
                    let result = execute_tool(&tool_call.name, &tool_call.arguments);
                    println!("    - {} -> {}", tool_call.name, result);
                    new_messages.push(Message::tool(&tool_call.id, &result));
                }
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "pending_tool_calls": Vec::<PendingToolCall>::new()
                })))
            }
        }))
        .add_edge(start(), "agent")
        .add_conditional_edges(
            "agent",
            router_fn(|state: &AgentState| {
                if state.iteration >= 10 {
                    return RouteOutput::End;
                }
                if !state.pending_tool_calls.is_empty() {
                    RouteOutput::One("tools".into())
                } else {
                    RouteOutput::End
                }
            }),
        )
        .add_edge("tools", "agent")
        .compile(Default::default())?;

    let executor = GraphExecutor::new(graph);

    // Test queries
    let queries = [
        "What's the weather like in San Francisco?",
        "Can you calculate 2 + 2 for me?",
    ];

    for query in queries {
        println!("─────────────────────────────────────────────");
        println!("User: {}\n", query);

        let initial_state = AgentState {
            messages: vec![
                Message::system(
                    "You are a helpful AI assistant with access to tools. \
                     Use the available tools when needed to answer questions."
                ),
                Message::user(query),
            ],
            pending_tool_calls: vec![],
            iteration: 0,
        };

        let config = RunnableConfig::new();
        let result = executor.invoke(initial_state, config).await?;

        if let Some(last_message) = result.messages.iter().rev()
            .find(|m| m.role == Role::Assistant && !m.content.is_empty())
        {
            println!("\nAssistant: {}", &last_message.content);
        }
        println!();
    }

    println!("─────────────────────────────────────────────");
    println!("Done!");

    Ok(())
}
