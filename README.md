# REGULA: Rust Execution Graph for Unified LLM Agents

[![Crates.io](https://img.shields.io/badge/{REGULA}-{HEX-COLOR}?style=for-the-badge&logo={REGULA}&logoColor=white)](https://crates.io/crates/regula)
[![Documentation](https://docs.rs/regula/badge.svg)](https://docs.rs/regula)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**REGULA** is a high-performance, type-safe framework for building stateful, multi-agent LLM applications in Rust. It implements the "Pregel" graph computational model—similar to LangGraph but leveraging Rust's ownership model, type system, and concurrency primitives for superior reliability and efficiency.

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Why REGULA?](#why-regula)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Deep Dive: The Graph Model](#deep-dive-the-graph-model)
    - [Nodes](#nodes)
    - [Edges & Control Flow](#edges--control-flow)
    - [The Pregel Loop](#the-pregel-loop)
- [Deep Dive: State Management](#deep-dive-state-management)
    - [The `GraphState` Trait](#the-graphstate-trait)
    - [Channels & Reducers](#channels--reducers)
    - [Macro Magic: `#[derive(GraphState)]`](#macro-magic-derivegraphstate)
- [Deep Dive: Execution Logic](#deep-dive-execution-logic)
    - [The `GraphExecutor`](#the-graphexecutor)
    - [Streaming Events](#streaming-events)
    - [Concurrency & Async](#concurrency--async)
- [LLM Integration](#llm-integration)
    - [Configuration](#configuration)
    - [Messages & Roles](#messages--roles)
    - [Tools & Function Calling](#tools--function-calling)
- [Persistence & Checkpointing](#persistence--checkpointing)
    - [Memory Checkpointer](#memory-checkpointer)
    - [Human-in-the-Loop](#human-in-the-loop)
- [Architecture Patterns](#architecture-patterns)
    - [Pattern 1: The ReAct Loop](#pattern-1-the-react-loop)
    - [Pattern 2: Multi-Agent Orchestration](#pattern-2-multi-agent-orchestration)
    - [Pattern 3: Tool-Using Agent](#pattern-3-tool-using-agent)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Building reliable LLM agents requires managing complex state transitions, handling failures gracefully, and coordinating multiple "actors" (the LLM, tools, human inputs). Traditional linear chains are often insufficient for these cyclic, stateful workflows.

**REGULA** structures your agent application as a **State Graph**. 
- **Nodes** represent units of work (reasoning, tool execution, user input).
- **Edges** represent the flow of control (normal or conditional).
- **State** is a shared data structure that persists across the graph execution.

This approach allows you to build agents that can loop, retry, wait for human input, and maintain long-term memory, all within a strictly typed Rust environment.

## Why REGULA?

1.  **Type Safety**: State transitions are verified at compile time. No more runtime errors because an agent expected a dictionary but got a string.
2.  **Performance**: Built on Tokio and optimized for low-overhead async execution. Ideal for high-throughput agent systems.
3.  **Explicit Control Flow**: The graph structure makes the agent's logic visible and debuggable. You can see exactly why an agent moved from "Reasoning" to "Tool Expectation".
4.  **Flexible State**: Define exactly how your state merges updates. Append messages to a history, overwrite the last summary, or perform custom reduction logic.

## Key Features

- **🕸️ Graph-Based Architecture**: Define workflows as nodes and edges. Support for cycles (loops) and conditional branching.
- **💾 Advanced State Management**: `GraphState` trait with channel support (`LastValue`, `Append`, `Reducer`) for handling concurrent updates.
- **🧠 LLM Integration**: First-class support for OpenAI, OpenRouter, and compatible APIs. Structured tool usage and message history management.
- **⏸️ Persistence & Interrupts**: Built-in checkpointing system allows you to pause execution, save state, and resume later—perfect for "Human-in-the-loop" workflows.
- **⚡ Async Runtime**: Fully asynchronous execution using Tokio, supporting parallel node execution within supersteps.
- **📝 Streaming**: Stream real-time events from your agent as it reasons and acts.

## Installation

Add the following to your `Cargo.toml`. To get started quickly, we recommend using the main workspace crate which re-exports the core modules.

```toml
[dependencies]
regula = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

If you prefer to pick and choose components:

```toml
[dependencies]
regula-core = "0.1.0"      # Traits, types, graph builder
regula-runtime = "0.1.0"   # Execution engine
regula-macros = "0.1.0"    # Derive macros
regula-checkpoint = "0.1.0" # State persistence
regula-llm = "0.1.0"       # LLM clients and types
```

## Quick Start

Here is a complete, compilable example of a simple agent that echoes a message and adds a "processed" tag.

```rust
use regula::{
    StateGraph, GraphState, RunnableConfig,
    node_fn, start, end, 
    regula_core::error::Result
};
use serde::{Serialize, Deserialize};

// 1. Define your State
// We use the #[derive(GraphState)] macro to auto-implement the trait.
#[derive(Clone, Debug, Default, Serialize, Deserialize, GraphState)]
struct AgentState {
    // By default, fields use "LastValue" semantics (overwrite).
    input: String,
    
    // We can use #[reducer(append)] to specify merge strategies.
    // 'append' works on Vec<T>.
    #[reducer(append)]
    logs: Vec<String>,
}

// 2. Define Nodes
// Nodes are async functions that take the state and return a NodeOutput.

// The 'input' node processes the user input
async fn process_input(state: AgentState) -> Result<serde_json::Value> {
    let response = format!("Processed: {}", state.input);
    
    // Return a partial state update
    Ok(serde_json::json!({
        "input": response,
        "logs": vec![format!("Handled input: {}", state.input)]
    }))
}

// A simple logging node
async fn logger(state: AgentState) -> Result<serde_json::Value> {
    println!("Current state: {:?}", state);
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<()> {
    // 3. Keep it simple: Build the Graph
    let graph = StateGraph::<AgentState>::new()
        // Add nodes to the graph
        .add_node("process", node_fn(process_input))
        .add_node("log", node_fn(logger))
        
        // Define the flow
        .add_edge(start(), "process")
        .add_edge("process", "log")
        .add_edge("log", end())
        
        // Compile the graph
        .compile(RunnableConfig::default())?;

    // 4. Execute
    let initial_state = AgentState {
        input: "Hello Regula".to_string(),
        logs: vec![],
    };

    println!("Starting execution...");
    
    // invoke() runs until the graph reaches the 'end' node
    let final_state = graph.invoke(initial_state).await?;

    println!("Final Result: {}", final_state.input);
    println!("Logs: {:?}", final_state.logs);

    Ok(())
}
```

---

## Deep Dive: The Graph Model

The core abstraction of REGULA is the **StateGraph**. It is inspired by Google's Pregel paper and the Bulk Synchronous Parallel (BSP) model.

### Nodes

A **Node** is a unit of computation. In Rust, it is represented as an async function (or a closure) that:
1.  Receives the current snapshot of the `GraphState`.
2.  Performs work (calls LLMs, queries databases, processes data).
3.  Returns a `NodeOutput`.

Nodes do **not** mutate the state directly. Instead, they return *updates* describing how the state should change. This separation allows REGULA to handle parallelism safely.

### Edges & Control Flow

Edges determine which nodes run next. There are two types:

1.  **Normal Edge**: `A -> B`. When node A finishes, node B starts.
2.  **Conditional Edge**: `A -> (Router) -> [B, C, or End]`. When node A finishes, a **Router** function examines the state and decides where to go next.

This allows for dynamic workflows like:
*   *If the LLM provides a tool call -> Go to ToolNode.*
*   *If the LLM provides a final answer -> Go to End.*

### The Pregel Loop

Execution happens in **Supersteps**:

1.  **Read Phase**: All active nodes read the state from the previous step.
2.  **Execute Phase**: Nodes run in parallel. They produce update operations.
3.  **Write Phase**: The framework collects all updates. Using the `ChannelSpec` defined in your state, it merges these updates into a new state snapshot.
4.  **Route Phase**: Based on edge definitions, the framework determines the next set of nodes to activate.

This loop continues until the graph reaches the special `end` node or a recursion limit is hit.

---

## Deep Dive: State Management

State in REGULA is more than just a `struct`. It is a collection of **Channels**.

### The `GraphState` Trait

Any type used as state must implement `GraphState`. This trait defines how data flows through the system.

```rust
pub trait GraphState: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {
    fn channels() -> HashMap<String, ChannelSpec>;
    // ... helper methods provided by default
}
```

### Channels & Reducers

When multiple nodes update the same field in the state, or when a node updates a field that already has data, how should that conflict be resolved? This is defined by `ChannelSpec`.

| Channel Type | Description | Use Case |
|--------------|-------------|----------|
| `LastValue` | (Default) The new value overwrites the old one. If two nodes write to this simultaneously, it throws an error. | Simple variables, current status flags. |
| `Reducer(Append)` | Appends the new value(s) to a list. | Chat history, logs, artifact collections. |
| `Reducer(Add)` | Adds the new value to the existing one. | Counters, voting scores. |
| `Ephemeral` | The value exists only for the current superstep, then is cleared. | Trigger signals, temporary messages. |
| `AnyValue` | Last writer wins, but concurrent writes are allowed (nondeterministic execution order). | Shared scratchpads where order implies minor importance. |

### Macro Magic: `#[derive(GraphState)]`

You rarely need to implement `GraphState` manually. The derive macro handles the boilerplate.

```rust
#[derive(Clone, GraphState, Serialize, Deserialize)]
struct MyComplexState {
    // Default: LastValue
    current_agent: String,

    // Reducer: Append (State is Vec<Msg>, update is Vec<Msg>)
    #[reducer(append)]
    messages: Vec<Message>,

    // Reducer: Add (State is i32, update is i32)
    #[reducer(add)]
    steps_taken: i32,
    
    // Custom logic?
    // You can also implement the trait manually if you need specific reduction logic.
}
```

---

## Deep Dive: Execution Logic

### The `GraphExecutor`

The `StateGraph::compile()` method converts your builder into a `CompiledStateGraph`, which is immutable and validated. The `GraphExecutor` then wraps this graph to manage a specific run.

```rust
let executor = GraphExecutor::new(compiled_graph)
    .with_checkpointer(checkpointer); // Optional persistence

// Run it
let result = executor.invoke(input_state, config).await?;
```

### Streaming Events

For interactive applications (like chat interfaces), you don't want to wait for the entire graph to finish. Use `.stream()` to get real-time updates.

```rust
use regula_runtime::stream::StreamMode;

let mut stream = executor.stream(input_state, config, StreamMode::Values);

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(event) => match event {
            StreamChunk::NodeStart { node } => println!("Starting {}", node),
            StreamChunk::NodeEnd { node, output } => println!("Node {} finished", node),
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Concurrency & Async

REGULA is built on Tokio.
- **Parallel Nodes**: If your graph has `A -> B` and `A -> C`, then `B` and `C` will run simultaneously in the same superstep after `A` completes.
- **Async I/O**: Nodes should be async. Waiting for an LLM response or a database query does not block the executor thread.

---

## LLM Integration

REGULA provides the `regula-llm` crate to abstract basic LLM interactions.

### Configuration

```rust
use regula_llm::{LlmClient, LlmConfig};

let config = LlmConfig::builder()
    .model("gpt-4")
    .temperature(0.7)
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .build();

let client = LlmClient::new(config);
```

### Messages & Roles

We provide a standardized `Message` struct compatible with OpenAI's Chat Completion format.

```rust
use regula_llm::{Message, Role};

let msgs = vec![
    Message::system("You are a helpful assistant."),
    Message::user("Calculate the sum of 2 + 2"),
];

let response = client.chat(msgs).await?;
```

### Tools & Function Calling

The client handles tool definitions and responses.

```rust
// Define a tool
let tools = vec![
    Tool::new("calculator", "Perform math", json_schema!({ ... }))
];

// Call with tools
let response = client.chat_with_tools(msgs, tools).await?;

if let Some(tool_calls) = response.tool_calls {
    for call in tool_calls {
        println!("LLM wants to call: {} with args {}", call.function.name, call.function.arguments);
    }
}
```

Note: The `regula-llm` crate abstracts the *interface*, but your nodes are responsible for actually executing the tools and appending the `Message::tool` responses to the history.

---

## Persistence & Checkpointing

One of REGULA's most powerful features is **Checkpointing**. This allows "time travel" for your agents.

Every step of the graph execution can be saved. This enables:
1.  **Resuming**: If the server crashes, you can load the last checkpoint and continue.
2.  **Human-in-the-Loop**: The graph can pause (via `interrupt_before`), ask a human for approval, and then the human can update the state and resume execution.

### Memory Checkpointer

Included out of the box is an in-memory checkpointer, mostly for testing or short-lived sessions.

```rust
use regula_checkpoint::MemoryCheckpointer;
use std::sync::Arc;

let checkpointer = Arc::new(MemoryCheckpointer::new());
let config = RunnableConfig::default().with_thread_id("conversation-123");

let executor = GraphExecutor::new(graph).with_checkpointer(checkpointer.clone());

// Run
let _ = executor.invoke(state, config.clone()).await?;

// Later, retrieve history
let history = checkpointer.list(&config).await?;
for checkpoint in history {
    println!("Step {}: {:?}", checkpoint.metadata.step, checkpoint.checkpoint);
}
```

### Human-in-the-Loop

To implement a "human approval" step:

1.  Build graph with `.interrupt_before("execute_tool")`.
2.  Run graph. It will stop *just before* executing the tool node and return `Error::Interrupted`.
3.  The system (your API server, CLI) catches this error.
4.  Present options to the user.
5.  User approves or edits the state (e.g., changing the tool arguments).
6.  Run the graph again with the **same thread ID**. It loads the state and resumes.

---

## Architecture Patterns

### Pattern 1: The ReAct Loop

Concepts: **Reasoning + Acting**.
Structure:
- **Agent Node**: Calls LLM with tools. Decides to call a tool or finish.
- **Router**: Checks if LLM output has `tool_calls`.
    - If yes -> **Tool Node**.
    - If no -> **End**.
- **Tool Node**: Executes tool, adds result to history. Edges back to **Agent Node**.

### Pattern 2: Multi-Agent Orchestration

Concepts: **Supervisor**.
Structure:
- **Supervisor Node**: Looks at user request, selects a worker (Researcher, Coder, Reviewer).
- **Router**: directs to the selected worker node.
- **Worker Nodes**: Perform task, return result to Supervisor.
- **Supervisor**: Aggregates results, decides if done.

### Pattern 3: Tool-Using Agent

A simpler version where specific nodes handle specific tools, rather than a generic tool executor.

```rust
// Graph
.add_node("agent", agent)
.add_node("search_tool", run_search)
.add_node("calc_tool", run_calc)
.add_conditional_edges("agent", router_fn(|state| {
    match state.last_cmd {
        "search" => RouteOutput::one("search_tool"),
        "calc" => RouteOutput::one("calc_tool"),
        _ => RouteOutput::end(),
    }
}))
```

---

## API Reference

### `regula::StateGraph`
- `new()`: Create builder.
- `add_node(name, fn)`: Register a node.
- `add_edge(from, to)`: Register a fixed transition.
- `add_conditional_edges(from, router)`: Register dynamic transition.
- `compile(config)`: Build the executable graph.

### `regula::GraphState`
- `channels()`: Define memory model.
- `apply_update(json)`: Merge logic.

### `regula::NodeOutput`
- `update(val)`: Simple state merge.
- `command(cmd)`: Complex control (update state + force routing).

### `regula::Command`
- `goto(node)`: Force the graph to jump to a specific node, ignoring standard edges.

---

## Contributing

We welcome contributions! Please follow standard Rust conventions.
1.  Fork the repo.
2.  Create a feature branch.
3.  Add tests for your feature.
4.  Submit a PR.

## License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
