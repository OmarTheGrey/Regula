//! Edge types and routing logic.
//!
//! Edges define the control flow between nodes in a REGULA graph.
//! Normal edges provide fixed routing, while conditional edges use
//! a router function to determine the next node(s) dynamically.

use crate::constants::{end, NodeId};
use crate::error::Result;
use crate::state::GraphState;
use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

/// An edge in the state graph.
///
/// Edges connect nodes and define the control flow of the graph.
#[derive(Clone)]
pub enum Edge<S: GraphState> {
    /// A fixed transition from one node to another.
    Normal {
        /// Source node.
        from: NodeId,
        /// Target node.
        to: NodeId,
    },

    /// A conditional transition where the target is determined at runtime.
    Conditional {
        /// Source node.
        from: NodeId,
        /// Router that determines the target(s).
        router: Arc<dyn EdgeRouter<S>>,
        /// Optional mapping from router output strings to node IDs.
        path_map: Option<HashMap<String, NodeId>>,
    },
}

impl<S: GraphState> Edge<S> {
    /// Create a normal edge.
    pub fn normal(from: impl Into<NodeId>, to: impl Into<NodeId>) -> Self {
        Self::Normal {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Create a conditional edge.
    pub fn conditional<R: EdgeRouter<S> + 'static>(from: impl Into<NodeId>, router: R) -> Self {
        Self::Conditional {
            from: from.into(),
            router: Arc::new(router),
            path_map: None,
        }
    }

    /// Create a conditional edge with a path map.
    pub fn conditional_with_map<R: EdgeRouter<S> + 'static>(
        from: impl Into<NodeId>,
        router: R,
        path_map: HashMap<String, NodeId>,
    ) -> Self {
        Self::Conditional {
            from: from.into(),
            router: Arc::new(router),
            path_map: Some(path_map),
        }
    }

    /// Get the source node of this edge.
    pub fn from(&self) -> &NodeId {
        match self {
            Self::Normal { from, .. } => from,
            Self::Conditional { from, .. } => from,
        }
    }

    /// Check if this is a conditional edge.
    pub fn is_conditional(&self) -> bool {
        matches!(self, Self::Conditional { .. })
    }
}

impl<S: GraphState> std::fmt::Debug for Edge<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal { from, to } => {
                f.debug_struct("Edge::Normal")
                    .field("from", from)
                    .field("to", to)
                    .finish()
            }
            Self::Conditional { from, path_map, .. } => {
                f.debug_struct("Edge::Conditional")
                    .field("from", from)
                    .field("path_map", path_map)
                    .finish()
            }
        }
    }
}

/// The result of a routing decision.
#[derive(Debug, Clone)]
pub enum RouteOutput {
    /// Go to a single node.
    One(NodeId),

    /// Go to multiple nodes in parallel.
    Many(Vec<NodeId>),

    /// End the graph execution.
    End,
}

impl RouteOutput {
    /// Create a route to a single node.
    pub fn one(node: impl Into<NodeId>) -> Self {
        Self::One(node.into())
    }

    /// Create a route to multiple nodes.
    pub fn many(nodes: impl IntoIterator<Item = impl Into<NodeId>>) -> Self {
        Self::Many(nodes.into_iter().map(Into::into).collect())
    }

    /// Create a route to end the graph.
    pub fn end() -> Self {
        Self::End
    }

    /// Check if this route ends the graph.
    pub fn is_end(&self) -> bool {
        match self {
            Self::End => true,
            Self::One(id) => id.is_end(),
            Self::Many(ids) => ids.iter().all(|id| id.is_end()),
        }
    }

    /// Get all target node IDs.
    pub fn targets(&self) -> Vec<NodeId> {
        match self {
            Self::One(id) => vec![id.clone()],
            Self::Many(ids) => ids.clone(),
            Self::End => vec![end()],
        }
    }

    /// Convert to node IDs, replacing END sentinel with actual end.
    pub fn into_nodes(self) -> Vec<NodeId> {
        self.targets()
    }
}

impl From<NodeId> for RouteOutput {
    fn from(id: NodeId) -> Self {
        Self::One(id)
    }
}

impl From<&str> for RouteOutput {
    fn from(s: &str) -> Self {
        Self::One(NodeId::from(s))
    }
}

impl From<String> for RouteOutput {
    fn from(s: String) -> Self {
        Self::One(NodeId::from(s))
    }
}

impl From<Vec<NodeId>> for RouteOutput {
    fn from(ids: Vec<NodeId>) -> Self {
        if ids.is_empty() {
            Self::End
        } else {
            Self::Many(ids)
        }
    }
}

/// Trait for edge routing logic.
///
/// Implement this trait to create custom routing logic for conditional edges.
/// The router receives the current state and returns the next node(s).
///
/// # Examples
///
/// ```ignore
/// use regula_core::{EdgeRouter, RouteOutput, GraphState, Result};
/// use async_trait::async_trait;
///
/// struct MyRouter;
///
/// #[async_trait]
/// impl<S: GraphState> EdgeRouter<S> for MyRouter {
///     async fn route(&self, state: &S) -> Result<RouteOutput> {
///         // Decide based on state
///         Ok(RouteOutput::one("next_node"))
///     }
/// }
/// ```
#[async_trait]
pub trait EdgeRouter<S: GraphState>: Send + Sync {
    /// Determine the next node(s) based on the current state.
    async fn route(&self, state: &S) -> Result<RouteOutput>;

    /// Get a description of this router (for debugging).
    fn description(&self) -> Option<&str> {
        None
    }
}

/// A router created from a sync function.
pub struct FnRouter<S, F>
where
    S: GraphState,
    F: Fn(&S) -> RouteOutput + Send + Sync,
{
    func: F,
    description: Option<String>,
    _marker: PhantomData<S>,
}

impl<S, F> FnRouter<S, F>
where
    S: GraphState,
    F: Fn(&S) -> RouteOutput + Send + Sync,
{
    /// Create a new function router.
    pub fn new(func: F) -> Self {
        Self {
            func,
            description: None,
            _marker: PhantomData,
        }
    }

    /// Create a new function router with a description.
    pub fn with_description(func: F, description: impl Into<String>) -> Self {
        Self {
            func,
            description: Some(description.into()),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S, F> EdgeRouter<S> for FnRouter<S, F>
where
    S: GraphState,
    F: Fn(&S) -> RouteOutput + Send + Sync,
{
    async fn route(&self, state: &S) -> Result<RouteOutput> {
        Ok((self.func)(state))
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// A router created from an async function.
pub struct AsyncFnRouter<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RouteOutput>> + Send,
{
    func: F,
    description: Option<String>,
    _marker: PhantomData<fn(S) -> Fut>,
}

impl<S, F, Fut> AsyncFnRouter<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RouteOutput>> + Send,
{
    /// Create a new async function router.
    pub fn new(func: F) -> Self {
        Self {
            func,
            description: None,
            _marker: PhantomData,
        }
    }

    /// Create a new async function router with a description.
    pub fn with_description(func: F, description: impl Into<String>) -> Self {
        Self {
            func,
            description: Some(description.into()),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S, F, Fut> EdgeRouter<S> for AsyncFnRouter<S, F, Fut>
where
    S: GraphState,
    F: Fn(&S) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RouteOutput>> + Send,
{
    async fn route(&self, state: &S) -> Result<RouteOutput> {
        (self.func)(state).await
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Create a router from a sync function.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{router_fn, RouteOutput};
///
/// let router = router_fn(|state: &MyState| {
///     if state.done {
///         RouteOutput::end()
///     } else {
///         RouteOutput::one("continue")
///     }
/// });
/// ```
pub fn router_fn<S, F>(f: F) -> impl EdgeRouter<S>
where
    S: GraphState,
    F: Fn(&S) -> RouteOutput + Send + Sync + 'static,
{
    FnRouter::new(f)
}

/// Create a named router from a sync function.
pub fn router_fn_named<S, F>(description: impl Into<String>, f: F) -> impl EdgeRouter<S>
where
    S: GraphState,
    F: Fn(&S) -> RouteOutput + Send + Sync + 'static,
{
    FnRouter::with_description(f, description)
}

/// Create a router from an async function.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{async_router_fn, RouteOutput, Result};
///
/// let router = async_router_fn(|state: &MyState| async move {
///     // Can do async operations here
///     Ok(RouteOutput::one("next"))
/// });
/// ```
pub fn async_router_fn<S, F, Fut>(f: F) -> impl EdgeRouter<S>
where
    S: GraphState,
    F: Fn(&S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<RouteOutput>> + Send + 'static,
{
    AsyncFnRouter::new(f)
}

/// A boxed router for type erasure.
pub type BoxedRouter<S> = Arc<dyn EdgeRouter<S>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelSpec;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct TestState {
        done: bool,
    }

    impl GraphState for TestState {
        fn channels() -> std::collections::HashMap<String, ChannelSpec> {
            let mut channels = std::collections::HashMap::new();
            channels.insert("done".to_string(), ChannelSpec::LastValue);
            channels
        }
    }

    #[test]
    fn test_edge_normal() {
        let edge: Edge<TestState> = Edge::normal("a", "b");
        assert_eq!(edge.from(), &NodeId::from("a"));
        assert!(!edge.is_conditional());
    }

    #[test]
    fn test_edge_conditional() {
        let router = router_fn(|state: &TestState| {
            if state.done {
                RouteOutput::end()
            } else {
                RouteOutput::one("continue")
            }
        });

        let edge: Edge<TestState> = Edge::conditional("a", router);
        assert_eq!(edge.from(), &NodeId::from("a"));
        assert!(edge.is_conditional());
    }

    #[test]
    fn test_route_output_one() {
        let route = RouteOutput::one("agent");
        assert!(!route.is_end());
        assert_eq!(route.targets().len(), 1);
    }

    #[test]
    fn test_route_output_many() {
        let route = RouteOutput::many(vec!["a", "b", "c"]);
        assert!(!route.is_end());
        assert_eq!(route.targets().len(), 3);
    }

    #[test]
    fn test_route_output_end() {
        let route = RouteOutput::end();
        assert!(route.is_end());
    }

    #[test]
    fn test_route_output_from_str() {
        let route: RouteOutput = "agent".into();
        assert!(matches!(route, RouteOutput::One(_)));
    }

    #[tokio::test]
    async fn test_fn_router() {
        let router = router_fn(|state: &TestState| {
            if state.done {
                RouteOutput::end()
            } else {
                RouteOutput::one("continue")
            }
        });

        let state = TestState { done: false };
        let result = router.route(&state).await.unwrap();
        assert!(matches!(result, RouteOutput::One(_)));

        let state = TestState { done: true };
        let result = router.route(&state).await.unwrap();
        assert!(result.is_end());
    }

    #[tokio::test]
    async fn test_async_fn_router() {
        let router = async_router_fn(|state: &TestState| {
            let done = state.done;
            async move {
                if done {
                    Ok(RouteOutput::end())
                } else {
                    Ok(RouteOutput::one("continue"))
                }
            }
        });

        let state = TestState { done: false };
        let result = router.route(&state).await.unwrap();
        assert!(matches!(result, RouteOutput::One(_)));
    }
}
