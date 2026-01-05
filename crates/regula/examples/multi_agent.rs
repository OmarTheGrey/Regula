//! Multi-Agent Example
//!
//! This example demonstrates a multi-agent workflow with a supervisor
//! that routes tasks to specialized worker agents.
//!
//! # Running
//!
//! ```bash
//! # Set your API key (optional for this demo)
//! export OPENAI_API_KEY="your-api-key"
//!
//! # Run the example
//! cargo run --example multi_agent
//! ```

use regula::prelude::*;

/// Represents which agent was last active
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
enum ActiveAgent {
    #[default]
    None,
    Supervisor,
    Researcher,
    Coder,
    Writer,
}

/// The state shared across all agents
#[derive(Clone, Default, Serialize, Deserialize, GraphState)]
struct MultiAgentState {
    /// The conversation/task history
    messages: Vec<Message>,
    
    /// The next agent to run (set by supervisor)
    next_agent: Option<String>,
    
    /// Which agent was last active
    last_agent: ActiveAgent,
    
    /// Accumulated work from agents
    #[reducer(append)]
    work_log: Vec<String>,
    
    /// Whether the task is complete
    task_complete: bool,
    
    /// Iteration count
    iteration: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== REGULA Multi-Agent Example ===\n");
    println!("This example demonstrates a supervisor-worker pattern.\n");

    // Build the state graph
    let graph = StateGraph::<MultiAgentState>::new()
        .add_node("supervisor", node_fn(|state: &MultiAgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            let iteration = state.iteration;
            async move {
                println!("  [Supervisor] Analyzing task...");
                
                // Simulate routing based on keywords
                let user_message = messages.iter()
                    .find(|m| m.role == Role::User)
                    .map(|m| m.content.as_str())
                    .unwrap_or("");
                
                let next_agent = if user_message.to_lowercase().contains("research") 
                    || user_message.to_lowercase().contains("find")
                {
                    "researcher"
                } else if user_message.to_lowercase().contains("code")
                    || user_message.to_lowercase().contains("function")
                {
                    "coder"
                } else {
                    "writer"
                };
                
                println!("  [Supervisor] Routing to: {}", next_agent);
                
                Ok(NodeOutput::update(json!({
                    "next_agent": next_agent,
                    "last_agent": ActiveAgent::Supervisor,
                    "work_log": vec![format!("Supervisor routed task to {}", next_agent)],
                    "iteration": iteration + 1
                })))
            }
        }))
        .add_node("researcher", node_fn(|state: &MultiAgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            async move {
                println!("  [Researcher] Conducting research...");
                
                let mut new_messages = messages.clone();
                new_messages.push(Message::assistant(
                    "[Researcher Agent]\n\nI've completed the research task. \
                     Found relevant information and compiled findings."
                ));
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "last_agent": ActiveAgent::Researcher,
                    "work_log": vec!["Research completed"],
                    "task_complete": true,
                    "next_agent": null
                })))
            }
        }))
        .add_node("coder", node_fn(|state: &MultiAgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            async move {
                println!("  [Coder] Writing code...");
                
                let mut new_messages = messages.clone();
                new_messages.push(Message::assistant(
                    "[Coder Agent]\n\n```rust\nfn solution() {\n    println!(\"Task completed!\");\n}\n```"
                ));
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "last_agent": ActiveAgent::Coder,
                    "work_log": vec!["Code implemented"],
                    "task_complete": true,
                    "next_agent": null
                })))
            }
        }))
        .add_node("writer", node_fn(|state: &MultiAgentState, _config: &RunnableConfig| {
            let messages = state.messages.clone();
            async move {
                println!("  [Writer] Drafting content...");
                
                let mut new_messages = messages.clone();
                new_messages.push(Message::assistant(
                    "[Writer Agent]\n\nI've completed the writing task. \
                     Content has been drafted with attention to clarity and structure."
                ));
                
                Ok(NodeOutput::update(json!({
                    "messages": new_messages,
                    "last_agent": ActiveAgent::Writer,
                    "work_log": vec!["Content drafted"],
                    "task_complete": true,
                    "next_agent": null
                })))
            }
        }))
        .add_edge(start(), "supervisor")
        .add_conditional_edges(
            "supervisor",
            router_fn(|state: &MultiAgentState| {
                if state.iteration >= 5 {
                    return RouteOutput::End;
                }
                match state.next_agent.as_deref() {
                    Some("researcher") => RouteOutput::One("researcher".into()),
                    Some("coder") => RouteOutput::One("coder".into()),
                    Some("writer") => RouteOutput::One("writer".into()),
                    _ => RouteOutput::End,
                }
            }),
        )
        .add_conditional_edges("researcher", router_fn(|state: &MultiAgentState| {
            if state.task_complete { RouteOutput::End } else { RouteOutput::One("supervisor".into()) }
        }))
        .add_conditional_edges("coder", router_fn(|state: &MultiAgentState| {
            if state.task_complete { RouteOutput::End } else { RouteOutput::One("supervisor".into()) }
        }))
        .add_conditional_edges("writer", router_fn(|state: &MultiAgentState| {
            if state.task_complete { RouteOutput::End } else { RouteOutput::One("supervisor".into()) }
        }))
        .compile(Default::default())?;

    let executor = GraphExecutor::new(graph);

    // Test with different types of tasks
    let tasks = [
        "Research the latest trends in AI.",
        "Write a function to calculate fibonacci numbers.",
        "Draft an email to the team.",
    ];

    for task in tasks {
        println!("═══════════════════════════════════════════════════");
        println!("Task: {}\n", task);

        let initial_state = MultiAgentState {
            messages: vec![Message::user(task)],
            next_agent: None,
            last_agent: ActiveAgent::None,
            work_log: vec![],
            task_complete: false,
            iteration: 0,
        };

        let config = RunnableConfig::new();
        let result = executor.invoke(initial_state, config).await?;

        println!("\n--- Work Log ---");
        for entry in &result.work_log {
            println!("  • {}", entry);
        }
        
        if let Some(last_message) = result.messages.iter().rev()
            .find(|m| m.role == Role::Assistant)
        {
            println!("\n--- Final Response ---");
            println!("{}", &last_message.content);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════");
    println!("All tasks completed!");

    Ok(())
}
