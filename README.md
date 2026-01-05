Here is the comprehensive `README.md` for **REGULA**, written assuming Version 1.0 is complete and ready for use.

---

# REGULA

**Rust Execution Graph for Unified LLM Agents**

**Regula** is a production-grade orchestration framework for building stateful, multi-agent LLM applications in Rust.

Built on the **Pregel** message-passing model (inspired by LangGraph), Regula allows you to define complex agentic workflows as **Directed Cyclic Graphs (DCGs)**. Unlike generic DAG runners, Regula is designed specifically for LLM agents, featuring persistent state, cycles (loops), controllable interrupts, and typesafe state management.

## 🌟 Key Features

* **Stateful Graph Execution:** Define workflows where nodes (agents/tools) communicate via a shared, strongly-typed state.
* **Cyclic Control Flow:** Native support for loops (e.g., *Planner* → *Executor* → *Critic* → *Planner*), essential for resilient agents.
* **Persistence & Memory:** Built-in checkpointing allows you to pause, save, resume, and "time-travel" through agent execution threads.
* **Type-Safe Channels:** Manage state updates using specific merge strategies (LastValue, Reducers/Aggregators) enforced by Rust's type system.
* **Async Runtime:** Built on `tokio`, optimized for high-concurrency environments.
* **LLM Integration:** Includes `regula-llm`, an OpenAI-compatible client with support for tool calling and streaming.

---

## 📦 Installation

Add `regula` to your `Cargo.toml`.

```toml
[dependencies]
regula = { version = "0.1", features = ["full"] }

```

Or pick specific modules:

```toml
[dependencies]
regula = { version = "0.1", default-features = false }
regula-core = "0.1"
regula-runtime = "0.1"

```

---

## 🚀 Quick Start

Here is a simple example of a Chatbot that maintains a conversation history.

```rust
use regula::prelude::*;
use serde::{Deserialize, Serialize};

// 1. Define your State
// Derive GraphState to automatically handle channel merge logic
#[derive(Clone, Default, GraphState, Serialize, Deserialize)]
struct ChatState {
    // 'messages' uses the default LastValue channel (overwrites on update)
    // To append instead, we could use #[reducer(append)]
    messages: Vec<Message>,
}

// 2. Define a Node (The Agent)
async fn chatbot(
    state: &ChatState,
    config: &RunnableConfig,
) -> Result<NodeOutput<ChatState>> {
    let client = config.get::<OpenAiClient>()?;
    
    // Call LLM
    let response = client.complete(&state.messages).await?;
    
    // Return a partial state update
    Ok(NodeOutput::update(json!({
        "messages": [response.message]
    })))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup LLM Client
    let client = OpenAiClient::new(LlmConfig::openai(env!("OPENAI_API_KEY")));

    // 3. Build the Graph
    let graph = StateGraph::<ChatState>::new()
        .add_node("chatbot", node_fn(chatbot))
        .add_edge(START, "chatbot") // Entry point
        .add_edge("chatbot", END)   // Exit point
        .compile(Default::default())?;

    // 4. Run
    let input = ChatState {
        messages: vec![Message::user("Hello, how are you?")],
    };

    let config = RunnableConfig::new().with("client", client);
    let result = graph.invoke(input, config).await?;

    println!("Agent response: {}", result.messages.last().unwrap().content);
    Ok(())
}

```

---

## 🧠 Core Concepts

Regula maps the logic of your application into a **StateGraph**.

### 1. State & Channels

State is not just a mutable object; it is a collection of **Channels**. Each field in your state struct corresponds to a channel with a specific merge strategy.

```rust
#[derive(GraphState)]
struct AgentState {
    // LastValue: The last node to write to this field overwrites it.
    pub current_goal: String, 
    
    // Reducer: Writes to this field are aggregated (e.g., appended).
    // Ideal for conversation history or log accumulation.
    #[reducer(append)]
    pub history: Vec<String>, 
}

```

### 2. Nodes

Nodes are async Rust functions that receive the current `State` and return a `NodeOutput`. A node generally performs an action (calling an LLM, searching the web, calculating math) and returns a **Partial State Update**.

### 3. Edges & Routing

Control flow is determined by edges.

* **Normal Edge:** `A -> B` (Always go to B after A).
* **Conditional Edge:** `A -> Router -> (B or C)`. The router logic inspects the state to decide the next step dynamically (e.g., "If tool called, go to ToolNode, else go to END").

### 4. Checkpointing

Regula allows you to save the state of the graph at every step. This enables:

* **Human-in-the-loop:** Pause execution, wait for user approval, then resume.
* **Time Travel:** Rewind to a previous step to retry with different parameters.
* **Fault Tolerance:** Resume execution after a crash.

---

## 🏗️ Architecture

Regula is designed as a workspace of modular crates to keep dependencies lightweight.

| Crate | Description |
| --- | --- |
| `regula` | The facade crate. Include this to use the framework. |
| `regula-core` | Defines `GraphState`, `Node`, `Edge` traits and Channel logic. |
| `regula-macros` | Procedural macros for `#[derive(GraphState)]`. |
| `regula-runtime` | The **Pregel** execution engine (loop, parallelism, synchronization). |
| `regula-checkpoint` | Persistence layer (In-Memory, Postgres, Redis adapters). |
| `regula-llm` | Structs and traits for LLM interactions (OpenAI, Anthropic, etc.). |

### Execution Model

```
┌───────┐      ┌─────────┐      ┌─────────┐
│ START │─────▶│  Node A │─────▶│  Node B │
└───────┘      │ (Agent) │      │ (Tools) │
               └─────────┘      └─────────┘
                    │                │
                    ▼                ▼
          ┌─────────────────────────────┐
          │     Shared State (State)    │
          │ ┌─────────┐   ┌───────────┐ │
          │ │ messages│   │tool_calls │ │
          │ └─────────┘   └───────────┘ │
          └─────────────────────────────┘

```

---

## 📚 Examples

Check the [examples/](https://www.google.com/search?q=examples/) directory for more complex patterns:

* **`tool_agent.rs`**: A ReAct-style agent that can use tools in a loop.
* **`multi_agent.rs`**: Two agents (Research & Writer) handing off tasks to one another.
* **`human_loop.rs`**: Pausing execution for user input before proceeding.
* **`streaming.rs`**: Streaming tokens from the graph as they are generated.

---

## 🤝 Contributing

Contributions are welcome! Please check out the [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) guide.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
