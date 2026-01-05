//! StateGraph builder and compilation.
//!
//! The `StateGraph` struct provides a fluent builder API for constructing
//! graphs. Once built, `compile()` validates the graph and produces a
//! `CompiledStateGraph` ready for execution.

use crate::constants::{end, start, NodeId};
use crate::edge::{Edge, EdgeRouter};
use crate::error::{RegulaError, Result};
use crate::node::{BoxedNode, Node};
use crate::state::GraphState;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Builder for state graphs.
///
/// Use the fluent API to add nodes, edges, and compile the graph.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{StateGraph, start, end, node_fn, router_fn, RouteOutput};
///
/// let graph = StateGraph::<MyState>::new()
///     .add_node("agent", agent_node)
///     .add_node("tools", tools_node)
///     .add_edge(start(), "agent")
///     .add_conditional_edges("agent", router_fn(|state| {
///         if state.done {
///             RouteOutput::end()
///         } else {
///             RouteOutput::one("tools")
///         }
///     }))
///     .add_edge("tools", "agent")
///     .compile(Default::default())?;
/// ```
#[derive(Clone)]
pub struct StateGraph<S: GraphState> {
    /// Nodes in the graph, keyed by name.
    nodes: IndexMap<NodeId, BoxedNode<S>>,

    /// Edges in the graph.
    edges: Vec<Edge<S>>,

    /// Entry point node (if set explicitly).
    entry_point: Option<NodeId>,

    /// Nodes to interrupt before execution.
    interrupt_before: HashSet<NodeId>,

    /// Nodes to interrupt after execution.
    interrupt_after: HashSet<NodeId>,
}

impl<S: GraphState> StateGraph<S> {
    /// Create a new empty state graph.
    pub fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
            edges: Vec::new(),
            entry_point: None,
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
        }
    }

    /// Add a node to the graph.
    ///
    /// # Arguments
    ///
    /// - `name`: The unique name for this node.
    /// - `node`: The node implementation.
    ///
    /// # Returns
    ///
    /// The builder for chaining.
    ///
    /// # Panics
    ///
    /// Panics if a node with this name already exists. Use `try_add_node`
    /// for fallible insertion.
    pub fn add_node<N: Node<S> + 'static>(mut self, name: impl Into<NodeId>, node: N) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            panic!("Node '{}' already exists in graph", name);
        }
        self.nodes.insert(name, Arc::new(node));
        self
    }

    /// Try to add a node to the graph.
    ///
    /// Returns an error if a node with this name already exists.
    pub fn try_add_node<N: Node<S> + 'static>(
        mut self,
        name: impl Into<NodeId>,
        node: N,
    ) -> Result<Self> {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            return Err(RegulaError::DuplicateNode(name.to_string()));
        }
        self.nodes.insert(name, Arc::new(node));
        Ok(self)
    }

    /// Add a normal (fixed) edge between two nodes.
    ///
    /// # Arguments
    ///
    /// - `from`: The source node (or `start()` for entry point).
    /// - `to`: The target node (or `end()` for termination).
    pub fn add_edge(mut self, from: impl Into<NodeId>, to: impl Into<NodeId>) -> Self {
        self.edges.push(Edge::normal(from, to));
        self
    }

    /// Add conditional edges with a router function.
    ///
    /// The router determines the next node(s) based on the current state.
    ///
    /// # Arguments
    ///
    /// - `from`: The source node.
    /// - `router`: The routing logic.
    pub fn add_conditional_edges<R: EdgeRouter<S> + 'static>(
        mut self,
        from: impl Into<NodeId>,
        router: R,
    ) -> Self {
        self.edges.push(Edge::conditional(from, router));
        self
    }

    /// Add conditional edges with a path map.
    ///
    /// The path map translates router string outputs to node IDs.
    pub fn add_conditional_edges_with_map<R: EdgeRouter<S> + 'static>(
        mut self,
        from: impl Into<NodeId>,
        router: R,
        path_map: HashMap<String, NodeId>,
    ) -> Self {
        self.edges
            .push(Edge::conditional_with_map(from, router, path_map));
        self
    }

    /// Set the entry point explicitly.
    ///
    /// This is equivalent to `add_edge(start(), node)`.
    pub fn set_entry_point(mut self, node: impl Into<NodeId>) -> Self {
        let node_id = node.into();
        self.entry_point = Some(node_id.clone());
        self.edges.push(Edge::normal(start(), node_id));
        self
    }

    /// Set a finish point explicitly.
    ///
    /// This is equivalent to `add_edge(node, end())`.
    pub fn set_finish_point(mut self, node: impl Into<NodeId>) -> Self {
        self.edges.push(Edge::normal(node, end()));
        self
    }

    /// Add a node to interrupt before execution.
    ///
    /// When the graph reaches this node, it will pause before executing
    /// and return to the caller.
    pub fn interrupt_before(mut self, node: impl Into<NodeId>) -> Self {
        self.interrupt_before.insert(node.into());
        self
    }

    /// Add a node to interrupt after execution.
    ///
    /// When this node completes, the graph will pause and return to the caller.
    pub fn interrupt_after(mut self, node: impl Into<NodeId>) -> Self {
        self.interrupt_after.insert(node.into());
        self
    }

    /// Get the list of node names.
    pub fn node_names(&self) -> impl Iterator<Item = &NodeId> {
        self.nodes.keys()
    }

    /// Check if a node exists.
    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.keys().any(|k| k.as_str() == name)
    }

    /// Validate the graph structure.
    fn validate(&self) -> Result<()> {
        // Check for at least one node
        if self.nodes.is_empty() {
            return Err(RegulaError::NoEntryPoint);
        }

        // Check for entry point
        let has_entry = self.edges.iter().any(|e| e.from().is_start());
        if !has_entry && self.entry_point.is_none() {
            return Err(RegulaError::NoEntryPoint);
        }

        // Check all edge sources and targets exist
        for edge in &self.edges {
            let from = edge.from();
            if !from.is_start() && !self.nodes.contains_key(from) {
                return Err(RegulaError::InvalidEdgeSource(from.to_string()));
            }

            if let Edge::Normal { to, .. } = edge {
                if !to.is_end() && !self.nodes.contains_key(to) {
                    return Err(RegulaError::InvalidEdgeTarget(to.to_string()));
                }
            }
        }

        // Check all non-sentinel nodes have at least one outgoing edge
        for node_id in self.nodes.keys() {
            let has_outgoing = self.edges.iter().any(|e| e.from() == node_id);
            if !has_outgoing {
                // Node has no outgoing edges - this could be valid if it uses Command
                // to specify routing, so we just warn in debug mode
                #[cfg(debug_assertions)]
                tracing::debug!(
                    "Node '{}' has no outgoing edges - ensure it returns a Command with goto",
                    node_id
                );
            }
        }

        // Check for unreachable nodes
        let mut reachable = HashSet::new();
        self.collect_reachable_nodes(&mut reachable);

        for node_id in self.nodes.keys() {
            if !reachable.contains(node_id) {
                return Err(RegulaError::UnreachableNode(node_id.to_string()));
            }
        }

        Ok(())
    }

    /// Collect all nodes reachable from START.
    fn collect_reachable_nodes(&self, reachable: &mut HashSet<NodeId>) {
        let mut queue: Vec<NodeId> = Vec::new();

        // Start from entry points
        for edge in &self.edges {
            if edge.from().is_start() {
                if let Edge::Normal { to, .. } = edge {
                    queue.push(to.clone());
                }
            }
        }

        // BFS to find all reachable nodes
        while let Some(node) = queue.pop() {
            if node.is_end() || reachable.contains(&node) {
                continue;
            }
            reachable.insert(node.clone());

            // Find outgoing edges
            for edge in &self.edges {
                if edge.from() == &node {
                    match edge {
                        Edge::Normal { to, .. } => {
                            if !to.is_end() {
                                queue.push(to.clone());
                            }
                        }
                        Edge::Conditional { path_map, .. } => {
                            // For conditional edges, include all possible targets from path_map
                            if let Some(map) = path_map {
                                for target in map.values() {
                                    if !target.is_end() {
                                        queue.push(target.clone());
                                    }
                                }
                            }
                            // Note: We can't statically determine all router outputs,
                            // so we assume all nodes might be reachable through conditional edges
                            for other_node in self.nodes.keys() {
                                queue.push(other_node.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compile the graph into an executable form.
    ///
    /// This validates the graph structure and produces a `CompiledStateGraph`
    /// ready for execution.
    pub fn compile(self, config: CompileConfig) -> Result<CompiledStateGraph<S>> {
        self.validate()?;
        Ok(CompiledStateGraph {
            nodes: self.nodes,
            edges: self.edges,
            entry_point: self.entry_point,
            interrupt_before: self.interrupt_before,
            interrupt_after: self.interrupt_after,
            config,
        })
    }
}

impl<S: GraphState> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GraphState> std::fmt::Debug for StateGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("entry_point", &self.entry_point)
            .finish()
    }
}

/// Configuration for graph compilation.
#[derive(Clone, Debug, Default)]
pub struct CompileConfig {
    /// Maximum iterations before timeout.
    pub max_iterations: Option<usize>,

    /// Whether to validate the graph structure.
    pub validate: bool,

    /// Enable debug mode with extra tracing.
    pub debug: bool,
}

impl CompileConfig {
    /// Create a new compile configuration with defaults.
    pub fn new() -> Self {
        Self {
            max_iterations: Some(100),
            validate: true,
            debug: false,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Disable validation.
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Enable debug mode.
    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }
}

/// A compiled state graph ready for execution.
///
/// This is the output of `StateGraph::compile()`. Use the runtime crate
/// to execute this graph.
#[derive(Clone)]
pub struct CompiledStateGraph<S: GraphState> {
    /// Nodes in the graph.
    pub(crate) nodes: IndexMap<NodeId, BoxedNode<S>>,

    /// Edges in the graph.
    pub(crate) edges: Vec<Edge<S>>,

    /// Entry point node.
    pub(crate) entry_point: Option<NodeId>,

    /// Nodes to interrupt before.
    pub(crate) interrupt_before: HashSet<NodeId>,

    /// Nodes to interrupt after.
    pub(crate) interrupt_after: HashSet<NodeId>,

    /// Compilation configuration.
    pub(crate) config: CompileConfig,
}

impl<S: GraphState> CompiledStateGraph<S> {
    /// Get a node by name.
    pub fn get_node(&self, name: &NodeId) -> Option<&BoxedNode<S>> {
        self.nodes.get(name)
    }

    /// Get all node names.
    pub fn node_names(&self) -> impl Iterator<Item = &NodeId> {
        self.nodes.keys()
    }

    /// Get the edges for a node.
    pub fn edges_from(&self, node: &NodeId) -> Vec<&Edge<S>> {
        self.edges.iter().filter(|e| e.from() == node).collect()
    }

    /// Get the entry points (nodes with edges from START).
    pub fn entry_points(&self) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.from().is_start() {
                    if let Edge::Normal { to, .. } = e {
                        return Some(to.clone());
                    }
                }
                None
            })
            .collect()
    }

    /// Check if a node should interrupt before execution.
    pub fn should_interrupt_before(&self, node: &NodeId) -> bool {
        self.interrupt_before.contains(node)
    }

    /// Check if a node should interrupt after execution.
    pub fn should_interrupt_after(&self, node: &NodeId) -> bool {
        self.interrupt_after.contains(node)
    }

    /// Get the maximum iterations setting.
    pub fn max_iterations(&self) -> Option<usize> {
        self.config.max_iterations
    }

    /// Find the next nodes to execute after a given node completes.
    ///
    /// For normal edges, returns the fixed target.
    /// For conditional edges, returns None (router must be invoked at runtime).
    pub fn next_nodes(&self, from: &NodeId) -> Option<Vec<NodeId>> {
        let edges: Vec<_> = self.edges_from(from);
        
        if edges.is_empty() {
            return None;
        }

        // If there's a conditional edge, we can't determine next statically
        if edges.iter().any(|e| e.is_conditional()) {
            return None;
        }

        // Collect all normal edge targets
        let targets: Vec<NodeId> = edges
            .iter()
            .filter_map(|e| {
                if let Edge::Normal { to, .. } = e {
                    Some(to.clone())
                } else {
                    None
                }
            })
            .collect();

        if targets.is_empty() {
            None
        } else {
            Some(targets)
        }
    }

    /// Generate a Mermaid diagram of the graph.
    pub fn to_mermaid(&self) -> String {
        let mut lines = vec!["graph TD".to_string()];

        // Add nodes
        for name in self.nodes.keys() {
            lines.push(format!("    {}[{}]", name, name));
        }

        // Add edges
        for edge in &self.edges {
            match edge {
                Edge::Normal { from, to } => {
                    let from_str = if from.is_start() {
                        "START"
                    } else {
                        from.as_str()
                    };
                    let to_str = if to.is_end() { "END" } else { to.as_str() };
                    lines.push(format!("    {} --> {}", from_str, to_str));
                }
                Edge::Conditional { from, .. } => {
                    lines.push(format!("    {} -.-> |?| ...", from));
                }
            }
        }

        // Add special nodes
        lines.push("    START((START))".to_string());
        lines.push("    END((END))".to_string());

        lines.join("\n")
    }
}

impl<S: GraphState> std::fmt::Debug for CompiledStateGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledStateGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges.len())
            .field("entry_point", &self.entry_point)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelSpec;
    use crate::edge::router_fn;
    use crate::node::{node_fn, NodeOutput};
    use crate::RunnableConfig;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct TestState {
        counter: i32,
        done: bool,
    }

    impl GraphState for TestState {
        fn channels() -> HashMap<String, ChannelSpec> {
            let mut channels = HashMap::new();
            channels.insert("counter".to_string(), ChannelSpec::LastValue);
            channels.insert("done".to_string(), ChannelSpec::LastValue);
            channels
        }
    }

    #[test]
    fn test_state_graph_new() {
        let graph = StateGraph::<TestState>::new();
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_state_graph_add_node() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new().add_node("agent", node);
        assert!(graph.has_node("agent"));
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_state_graph_duplicate_node() {
        let node1 = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });
        let node2 = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let _ = StateGraph::<TestState>::new()
            .add_node("agent", node1)
            .add_node("agent", node2); // Should panic
    }

    #[test]
    fn test_state_graph_try_add_node_duplicate() {
        let node1 = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });
        let node2 = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", node1)
            .try_add_node("agent", node2);

        assert!(result.is_err());
    }

    #[test]
    fn test_state_graph_add_edges() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end());

        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_state_graph_conditional_edges() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let router = router_fn(|state: &TestState| {
            if state.done {
                RouteOutput::end()
            } else {
                RouteOutput::one("agent")
            }
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_conditional_edges("agent", router);

        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_state_graph_compile_success() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .compile(Default::default());

        assert!(result.is_ok());
    }

    #[test]
    fn test_state_graph_compile_no_entry() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge("agent", end())
            .compile(Default::default());

        assert!(result.is_err());
    }

    #[test]
    fn test_state_graph_compile_invalid_edge_source() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("nonexistent", "agent") // Invalid source
            .compile(Default::default());

        assert!(result.is_err());
    }

    #[test]
    fn test_state_graph_compile_invalid_edge_target() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let result = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", "nonexistent") // Invalid target
            .compile(Default::default());

        assert!(result.is_err());
    }

    #[test]
    fn test_compiled_graph_entry_points() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .compile(Default::default())
            .unwrap();

        let entry_points = graph.entry_points();
        assert_eq!(entry_points.len(), 1);
        assert_eq!(entry_points[0], "agent");
    }

    #[test]
    fn test_compiled_graph_to_mermaid() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .compile(Default::default())
            .unwrap();

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("START"));
        assert!(mermaid.contains("END"));
        assert!(mermaid.contains("agent"));
    }

    #[test]
    fn test_set_entry_point() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .set_entry_point("agent")
            .add_edge("agent", end())
            .compile(Default::default())
            .unwrap();

        assert_eq!(graph.entry_points(), vec![NodeId::from("agent")]);
    }

    #[test]
    fn test_interrupt_points() {
        let node = node_fn(|_: &TestState, _: &RunnableConfig| async {
            Ok(NodeOutput::none())
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .interrupt_before("agent")
            .interrupt_after("agent")
            .compile(Default::default())
            .unwrap();

        assert!(graph.should_interrupt_before(&NodeId::from("agent")));
        assert!(graph.should_interrupt_after(&NodeId::from("agent")));
    }
}
