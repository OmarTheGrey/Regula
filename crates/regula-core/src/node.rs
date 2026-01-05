//! Node trait and related types.
//!
//! Nodes are the fundamental building blocks of a REGULA graph. Each node
//! is an async function that receives state and produces updates.

use crate::constants::NodeId;
use crate::error::Result;
use crate::state::GraphState;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

/// The output from a node execution.
///
/// Nodes can return either a simple state update or a command that
/// includes routing information.
#[derive(Debug, Clone)]
pub enum NodeOutput {
    /// A partial state update to be merged into the current state.
    Update(serde_json::Value),

    /// A command that includes both state update and routing.
    Command(Command),

    /// No output - the state remains unchanged.
    None,
}

impl NodeOutput {
    /// Create an update output from a JSON value.
    pub fn update(value: serde_json::Value) -> Self {
        Self::Update(value)
    }

    /// Create an update output, serializing the given value.
    pub fn update_from<T: Serialize>(value: &T) -> Result<Self> {
        Ok(Self::Update(serde_json::to_value(value)?))
    }

    /// Create a command output.
    pub fn command(cmd: Command) -> Self {
        Self::Command(cmd)
    }

    /// Create a no-op output.
    pub fn none() -> Self {
        Self::None
    }

    /// Check if this output has state updates.
    pub fn has_update(&self) -> bool {
        match self {
            Self::Update(v) => !v.is_null() && v != &serde_json::Value::Object(Default::default()),
            Self::Command(cmd) => cmd.update.is_some(),
            Self::None => false,
        }
    }

    /// Get the state update, if any.
    pub fn get_update(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Update(v) => Some(v),
            Self::Command(cmd) => cmd.update.as_ref(),
            Self::None => None,
        }
    }

    /// Get routing from a command, if any.
    pub fn get_goto(&self) -> Option<&CommandGoto> {
        match self {
            Self::Command(cmd) => cmd.goto.as_ref(),
            _ => None,
        }
    }
}

/// A command that combines state update with explicit routing.
///
/// Commands allow nodes to control which node(s) execute next,
/// overriding the graph's edge definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// State update to apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<serde_json::Value>,

    /// Explicit next node(s) to execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goto: Option<CommandGoto>,

    /// Value to resume with (for interrupt handling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<serde_json::Value>,
}

impl Command {
    /// Create a new empty command.
    pub fn new() -> Self {
        Self {
            update: None,
            goto: None,
            resume: None,
        }
    }

    /// Create a command with a state update.
    pub fn with_update(update: serde_json::Value) -> Self {
        Self {
            update: Some(update),
            goto: None,
            resume: None,
        }
    }

    /// Create a command with a goto target.
    pub fn with_goto(goto: impl Into<CommandGoto>) -> Self {
        Self {
            update: None,
            goto: Some(goto.into()),
            resume: None,
        }
    }

    /// Add a state update to this command.
    pub fn update(mut self, update: serde_json::Value) -> Self {
        self.update = Some(update);
        self
    }

    /// Add a goto target to this command.
    pub fn goto(mut self, target: impl Into<CommandGoto>) -> Self {
        self.goto = Some(target.into());
        self
    }

    /// Add a resume value to this command.
    pub fn resume(mut self, value: serde_json::Value) -> Self {
        self.resume = Some(value);
        self
    }
}

impl Default for Command {
    fn default() -> Self {
        Self::new()
    }
}

/// Target for command routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommandGoto {
    /// Go to a single node.
    One(NodeId),

    /// Go to multiple nodes in parallel.
    Many(Vec<NodeId>),
}

impl CommandGoto {
    /// Create a goto to a single node.
    pub fn one(node: impl Into<NodeId>) -> Self {
        Self::One(node.into())
    }

    /// Create a goto to multiple nodes.
    pub fn many(nodes: impl IntoIterator<Item = impl Into<NodeId>>) -> Self {
        Self::Many(nodes.into_iter().map(Into::into).collect())
    }

    /// Check if this goto ends the graph.
    pub fn is_end(&self) -> bool {
        match self {
            Self::One(id) => id.is_end(),
            Self::Many(ids) => ids.iter().all(|id| id.is_end()),
        }
    }

    /// Get all target node IDs.
    pub fn targets(&self) -> Vec<&NodeId> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids.iter().collect(),
        }
    }
}

impl From<NodeId> for CommandGoto {
    fn from(id: NodeId) -> Self {
        Self::One(id)
    }
}

impl From<&str> for CommandGoto {
    fn from(s: &str) -> Self {
        Self::One(NodeId::from(s))
    }
}

impl From<String> for CommandGoto {
    fn from(s: String) -> Self {
        Self::One(NodeId::from(s))
    }
}

impl From<Vec<NodeId>> for CommandGoto {
    fn from(ids: Vec<NodeId>) -> Self {
        Self::Many(ids)
    }
}

/// Configuration passed to nodes during execution.
///
/// This is a re-export from the config module, included here
/// for convenience in node signatures.
pub use crate::config::RunnableConfig;

/// Trait for graph nodes.
///
/// Nodes are async functions that receive the current state and configuration,
/// and return a state update or command.
///
/// # Type Parameters
///
/// - `S`: The state type for this graph.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{Node, NodeOutput, GraphState, RunnableConfig, Result};
/// use async_trait::async_trait;
///
/// struct MyNode;
///
/// #[async_trait]
/// impl<S: GraphState> Node<S> for MyNode {
///     async fn execute(&self, state: &S, config: &RunnableConfig) -> Result<NodeOutput> {
///         // Process state and return update
///         Ok(NodeOutput::update(serde_json::json!({
///             "counter": 42
///         })))
///     }
/// }
/// ```
#[async_trait]
pub trait Node<S: GraphState>: Send + Sync {
    /// Execute the node with the current state.
    ///
    /// # Arguments
    ///
    /// - `state`: The current graph state.
    /// - `config`: Runtime configuration and context.
    ///
    /// # Returns
    ///
    /// A `NodeOutput` containing state updates and/or routing commands.
    async fn execute(&self, state: &S, config: &RunnableConfig) -> Result<NodeOutput>;

    /// Get the name of this node (for debugging/tracing).
    fn name(&self) -> Option<&str> {
        None
    }

    /// Check if this node should be retried on failure.
    fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy::None
    }
}

/// Retry policy for node execution.
#[derive(Debug, Clone, Default)]
pub enum RetryPolicy {
    /// No retries.
    #[default]
    None,

    /// Retry with exponential backoff.
    Exponential {
        /// Maximum number of retries.
        max_retries: usize,
        /// Initial delay in milliseconds.
        initial_delay_ms: u64,
        /// Maximum delay in milliseconds.
        max_delay_ms: u64,
    },

    /// Retry with fixed delay.
    Fixed {
        /// Maximum number of retries.
        max_retries: usize,
        /// Delay between retries in milliseconds.
        delay_ms: u64,
    },
}

/// A node created from an async function.
pub struct FnNode<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S, &RunnableConfig) -> Fut + Send + Sync,
    Fut: Future<Output = Result<NodeOutput>> + Send,
{
    func: F,
    name: Option<String>,
    _marker: PhantomData<fn(S) -> Fut>,
}

impl<S, F, Fut> FnNode<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S, &RunnableConfig) -> Fut + Send + Sync,
    Fut: Future<Output = Result<NodeOutput>> + Send,
{
    /// Create a new function node.
    pub fn new(func: F) -> Self {
        Self {
            func,
            name: None,
            _marker: PhantomData,
        }
    }

    /// Create a new function node with a name.
    pub fn with_name(func: F, name: impl Into<String>) -> Self {
        Self {
            func,
            name: Some(name.into()),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S, F, Fut> Node<S> for FnNode<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S, &RunnableConfig) -> Fut + Send + Sync,
    Fut: Future<Output = Result<NodeOutput>> + Send,
{
    async fn execute(&self, state: &S, config: &RunnableConfig) -> Result<NodeOutput> {
        (self.func)(state, config).await
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Create a node from an async function.
///
/// This is a convenience function to create nodes from closures or
/// async functions without implementing the `Node` trait manually.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{node_fn, NodeOutput, Result};
///
/// let my_node = node_fn(|state: &MyState, config: &RunnableConfig| async move {
///     Ok(NodeOutput::update(serde_json::json!({
///         "counter": state.counter + 1
///     })))
/// });
/// ```
pub fn node_fn<S, F, Fut>(f: F) -> impl Node<S>
where
    S: GraphState,
    F: Fn(&S, &RunnableConfig) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<NodeOutput>> + Send + 'static,
{
    FnNode::new(f)
}

/// Create a named node from an async function.
pub fn node_fn_named<S, F, Fut>(name: impl Into<String>, f: F) -> impl Node<S>
where
    S: GraphState,
    F: Fn(&S, &RunnableConfig) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<NodeOutput>> + Send + 'static,
{
    FnNode::with_name(f, name)
}

/// A boxed node for type erasure.
pub type BoxedNode<S> = Arc<dyn Node<S>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelSpec;
    use std::collections::HashMap;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct TestState {
        counter: i32,
    }

    impl GraphState for TestState {
        fn channels() -> HashMap<String, ChannelSpec> {
            let mut channels = HashMap::new();
            channels.insert("counter".to_string(), ChannelSpec::LastValue);
            channels
        }
    }

    #[test]
    fn test_node_output_update() {
        let output = NodeOutput::update(serde_json::json!({"counter": 42}));
        assert!(output.has_update());
        assert!(matches!(output, NodeOutput::Update(_)));
    }

    #[test]
    fn test_node_output_none() {
        let output = NodeOutput::none();
        assert!(!output.has_update());
    }

    #[test]
    fn test_command_builder() {
        let cmd = Command::new()
            .update(serde_json::json!({"counter": 100}))
            .goto("next_node");

        assert!(cmd.update.is_some());
        assert!(cmd.goto.is_some());
    }

    #[test]
    fn test_command_goto_one() {
        let goto = CommandGoto::one("agent");
        assert!(!goto.is_end());
        assert_eq!(goto.targets().len(), 1);
    }

    #[test]
    fn test_command_goto_many() {
        let goto = CommandGoto::many(vec!["a", "b", "c"]);
        assert!(!goto.is_end());
        assert_eq!(goto.targets().len(), 3);
    }

    #[test]
    fn test_command_goto_from_str() {
        let goto: CommandGoto = "agent".into();
        assert!(matches!(goto, CommandGoto::One(_)));
    }

    #[tokio::test]
    async fn test_fn_node() {
        let node = node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": counter + 1
                })))
            }
        });

        let state = TestState { counter: 10 };
        let config = RunnableConfig::new();

        let output = node.execute(&state, &config).await.unwrap();
        assert!(output.has_update());

        if let NodeOutput::Update(value) = output {
            assert_eq!(value["counter"], 11);
        }
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert!(matches!(policy, RetryPolicy::None));
    }
}
