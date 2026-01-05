//! REGULA - A LangGraph-style agentic orchestration framework in Rust.
//!
//! REGULA provides a graph-based approach for building AI agent workflows,
//! inspired by LangGraph's Pregel-style message passing model.
//!
//! # Features
//!
//! - **State Graphs**: Define agent workflows as directed graphs with typed state
//! - **Conditional Routing**: Dynamic control flow based on state
//! - **Parallel Execution**: Run multiple nodes concurrently within super-steps
//! - **Checkpointing**: Save and restore execution state for persistence
//! - **LLM Integration**: OpenAI-compatible client for chat completions
//! - **Streaming**: Real-time updates during graph execution
//!
//! # Example
//!
//! ```ignore
//! use regula::prelude::*;
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct AgentState {
//!     messages: Vec<Message>,
//!     done: bool,
//! }
//!
//! impl GraphState for AgentState {
//!     fn channels() -> HashMap<String, ChannelSpec> {
//!         let mut channels = HashMap::new();
//!         channels.insert("messages".to_string(), ChannelSpec::LastValue);
//!         channels.insert("done".to_string(), ChannelSpec::LastValue);
//!         channels
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let graph = StateGraph::<AgentState>::new()
//!         .add_node("agent", node_fn(|state, config| async {
//!             // Agent logic here
//!             Ok(NodeOutput::update(json!({"done": true})))
//!         }))
//!         .add_edge(start(), "agent")
//!         .add_edge("agent", end())
//!         .compile(Default::default())?;
//!
//!     let executor = GraphExecutor::new(graph);
//!     let result = executor.invoke(initial_state, RunnableConfig::new()).await?;
//!
//!     Ok(())
//! }
//! ```

// Re-export all sub-crates
pub use regula_checkpoint as checkpoint;
pub use regula_core as core;
pub use regula_llm as llm;
pub use regula_macros::GraphState;
pub use regula_runtime as runtime;

// Re-export commonly used types from core
pub use regula_core::{
    async_router_fn, end, node_fn, node_fn_named, partial_state, router_fn, start,
    BoxedNode, Channel, ChannelSpec, Command, CommandGoto, CompileConfig,
    CompiledStateGraph, ConfigSnapshot, DynState, Edge, EdgeRouter, GraphState,
    Node, NodeId, NodeOutput, RegulaError, Result, RetryPolicy, RouteOutput,
    RunnableConfig, StateGraph, END_NODE, START_NODE,
};

// Re-export from runtime
pub use regula_runtime::{GraphExecutor, StreamChunk, StreamMode};

// Re-export from checkpoint
pub use regula_checkpoint::{Checkpoint, CheckpointMetadata, CheckpointTuple, Checkpointer, InMemorySaver};

// Re-export from llm
pub use regula_llm::{
    FunctionCall, LlmClient, LlmConfig, Message, OpenAiClient, Role, Tool, ToolCall, Usage,
};

// Re-export serde and serde_json for convenience
pub use serde;
pub use serde_json;
pub use serde_json::json;

/// Prelude module for convenient imports.
///
/// Import with `use regula::prelude::*;` to get all commonly used types.
pub mod prelude {
    pub use crate::{
        async_router_fn, end, node_fn, router_fn, start, partial_state,
        ChannelSpec, Command, CommandGoto, CompileConfig, CompiledStateGraph,
        Edge, EdgeRouter, GraphExecutor, GraphState, InMemorySaver, LlmClient,
        LlmConfig, Message, Node, NodeId, NodeOutput, OpenAiClient, RegulaError,
        Result, Role, RouteOutput, RunnableConfig, StateGraph, StreamChunk,
        StreamMode, Tool, ToolCall,
    };

    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
    pub use serde_json::json;

    // Re-export futures for streaming
    pub use futures::StreamExt;
}
