# REGULA

**R**ust **E**xecution **G**raph for **U**nified **L**LM **A**gents

REGULA is a LangGraph-style agentic orchestration framework in Rust, built on the Pregel message-passing model. It enables the creation of stateful, multi-step workflows with cycles, controllable side-effects, and persistence.

## Features

- 🕸️ **Graph-Based Orchestration**: Define your application logic as a graph of nodes (functions) and edges (transitions).
- 🔄 **Stateful Execution**: Maintain and evolve state across the graph lifecycle with strictly typed state management.
- ♻️ **Cycles & Feedback Loops**: Native support for cyclic graphs, enabling agents to reason, act, and observe in loops.
- 💾 **Persistence**: Built-in checkpointing to pause, resume, and inspect workflow state.
- ⚡ **Async Runtime**: Built on Tokio for high-performance, asynchronous execution.
- 🧠 **LLM Optimised**: First-class support for LLM interactions (OpenAI, Anthropic, etc. via compatibility layers).

## Installation

Add `regula` to your `Cargo.toml`:

```toml
[dependencies]
regula = "0.1.0"
```

## Quick Start

Here's a simple example of a chat agent that responds to user input:

```rust
use regula::prelude::*;
use std::time::Duration;

#[derive(Clone, Default, Debug, Serialize, Deserialize, GraphState)]
struct ChatState {
    messages: Vec<Message>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Define an LLM agent node
    let agent_node = node_fn(|state: &ChatState, _| async move {
        // ... call LLM ...
        println!("Processing: {:?}", state.messages.last());
        Ok(NodeOutput::update(state.clone()))
    });

    // Build the graph
    let graph = StateGraph::<ChatState>::new()
        .add_node("agent", agent_node)
        .add_edge(start(), "agent")
        .add_edge("agent", end())
        .compile(Default::default())?;

    // Execute
    let executor = GraphExecutor::new(graph);
    let initial_state = ChatState::default();
    executor.invoke(initial_state, RunnableConfig::default()).await?;

    Ok(())
}
```

## Architecture

REGULA follows a modular architecture:

- `regula`: The main facade crate.
- `regula-core`: Core traits and type definitions.
- `regula-runtime`: The Pregel-style execution engine.
- `regula-macros`: Derive macros for state management.
- `regula-checkpoint`: Persistence and checkpointing.
- `regula-llm`: LLM client integrations.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
