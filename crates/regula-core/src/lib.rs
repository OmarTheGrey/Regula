//! REGULA Core - Core abstractions for the REGULA agentic orchestration framework.
//!
//! This crate provides the fundamental building blocks for constructing
//! LangGraph-style agent graphs in Rust:
//!
//! - **State**: The `GraphState` trait and channel system for managing graph state.
//! - **Nodes**: The `Node` trait for defining graph nodes (agents, tools, etc.).
//! - **Edges**: Normal and conditional edges for graph control flow.
//! - **Graph**: The `StateGraph` builder and `CompiledStateGraph` for execution.
//!
//! # Example
//!
//! ```ignore
//! use regula_core::prelude::*;
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct MyState {
//!     messages: Vec<String>,
//!     done: bool,
//! }
//!
//! impl GraphState for MyState {
//!     fn channels() -> HashMap<String, ChannelSpec> {
//!         let mut channels = HashMap::new();
//!         channels.insert("messages".to_string(), ChannelSpec::LastValue);
//!         channels.insert("done".to_string(), ChannelSpec::LastValue);
//!         channels
//!     }
//! }
//!
//! let graph = StateGraph::<MyState>::new()
//!     .add_node("agent", node_fn(|state, config| async {
//!         Ok(NodeOutput::update(json!({"messages": ["Hello!"]})))
//!     }))
//!     .add_edge(start(), "agent")
//!     .add_edge("agent", end())
//!     .compile(Default::default())?;
//! ```

// Re-export commonly used external crates
pub use async_trait::async_trait;
pub use serde;
pub use serde_json;

// Module declarations
pub mod channel;
pub mod config;
pub mod constants;
pub mod edge;
pub mod error;
pub mod graph;
pub mod node;
pub mod state;

// Re-exports for convenience
pub use channel::{Channel, ChannelSpec};
pub use config::{ConfigSnapshot, RunnableConfig};
pub use constants::{end, start, NodeId, END_NODE, START_NODE};
pub use edge::{async_router_fn, router_fn, Edge, EdgeRouter, RouteOutput};
pub use error::{RegulaError, Result};
pub use graph::{CompileConfig, CompiledStateGraph, StateGraph};
pub use node::{node_fn, node_fn_named, BoxedNode, Command, CommandGoto, Node, NodeOutput, RetryPolicy};
pub use state::{DynState, GraphState};

/// Prelude module for convenient imports.
///
/// Import with `use regula_core::prelude::*;` to get all commonly used types.
pub mod prelude {
    pub use crate::channel::{Channel, ChannelSpec};
    pub use crate::config::RunnableConfig;
    pub use crate::constants::{end, start, NodeId};
    pub use crate::edge::{async_router_fn, router_fn, Edge, EdgeRouter, RouteOutput};
    pub use crate::error::{RegulaError, Result};
    pub use crate::graph::{CompileConfig, CompiledStateGraph, StateGraph};
    pub use crate::node::{node_fn, Command, CommandGoto, Node, NodeOutput};
    pub use crate::state::GraphState;
    pub use crate::partial_state;

    // Re-export useful external types
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
    pub use serde_json::json;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use std::collections::HashMap;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestState {
        messages: Vec<String>,
        counter: i32,
    }

    impl GraphState for TestState {
        fn channels() -> HashMap<String, ChannelSpec> {
            let mut channels = HashMap::new();
            channels.insert("messages".to_string(), ChannelSpec::LastValue);
            channels.insert("counter".to_string(), ChannelSpec::LastValue);
            channels
        }
    }

    #[test]
    fn test_integration_build_graph() {
        let agent_node = node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": counter + 1
                })))
            }
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", agent_node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .compile(Default::default());

        assert!(result.is_ok());
        let graph = result.unwrap();
        assert_eq!(graph.node_names().count(), 1);
    }

    #[test]
    fn test_integration_conditional_graph() {
        let agent_node = node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": 1
                })))
            }
        });

        let tools_node = node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": 2
                })))
            }
        });

        let router = router_fn(|state: &TestState| {
            if state.counter >= 5 {
                RouteOutput::end()
            } else {
                RouteOutput::one("tools")
            }
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", agent_node)
            .add_node("tools", tools_node)
            .add_edge(start(), "agent")
            .add_conditional_edges("agent", router)
            .add_edge("tools", "agent")
            .compile(Default::default());

        assert!(result.is_ok());
    }

    #[test]
    fn test_partial_state_macro() {
        let update = partial_state! {
            messages: vec!["hello"],
            counter: 42,
        };

        assert_eq!(update["messages"][0], "hello");
        assert_eq!(update["counter"], 42);
    }
}
