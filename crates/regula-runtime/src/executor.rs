//! Graph execution engine.

use futures::future::join_all;
use regula_checkpoint::{Checkpoint, CheckpointMetadata, Checkpointer};
use regula_core::{
    CompiledStateGraph, DynState, Edge, GraphState, NodeId, NodeOutput, RegulaError, Result,
    RouteOutput, RunnableConfig, end,
};
use std::collections::HashSet;
use std::sync::Arc;

use crate::stream::{StreamChunk, StreamMode};

/// Executor for running compiled state graphs.
pub struct GraphExecutor<S: GraphState> {
    /// The compiled graph.
    graph: Arc<CompiledStateGraph<S>>,

    /// Optional checkpointer for persistence.
    checkpointer: Option<Arc<dyn Checkpointer>>,
}

impl<S: GraphState> GraphExecutor<S> {
    /// Create a new executor for a graph.
    pub fn new(graph: CompiledStateGraph<S>) -> Self {
        Self {
            graph: Arc::new(graph),
            checkpointer: None,
        }
    }

    /// Set the checkpointer for this executor.
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Run the graph to completion.
    pub async fn invoke(&self, input: S, config: RunnableConfig) -> Result<S> {
        let mut state = DynState::from_state(&input)?;
        let mut current_nodes = self.graph.entry_points();
        let max_iterations = config
            .max_iterations()
            .or(self.graph.max_iterations())
            .unwrap_or(100);
        let mut iteration = 0;

        // Load from checkpoint if available
        if let Some(ref checkpointer) = self.checkpointer {
            if let Some(tuple) = checkpointer.get(&config).await? {
                state.restore(tuple.checkpoint.values);
                // Resume from pending nodes
                current_nodes = tuple
                    .checkpoint
                    .pending
                    .into_iter()
                    .map(NodeId::from)
                    .collect();
            }
        }

        loop {
            if current_nodes.is_empty() || current_nodes.iter().all(|n| n.is_end()) {
                break;
            }

            if iteration >= max_iterations {
                return Err(RegulaError::MaxIterationsExceeded(max_iterations));
            }

            // Check for interrupts before
            for node in &current_nodes {
                if self.graph.should_interrupt_before(node) {
                    return Err(RegulaError::Interrupted(node.to_string()));
                }
            }

            // Execute current nodes in parallel
            let typed_state: S = state.to_state()?;
            let results = self
                .execute_nodes(&current_nodes, &typed_state, &config)
                .await?;

            // Apply updates
            for (_node_id, output) in &results {
                if let Some(update) = output.get_update() {
                    state.apply_update(update.clone())?;
                }
            }

            // Clear modified flags
            state.clear_modified();

            // Check for interrupts after
            for node in &current_nodes {
                if self.graph.should_interrupt_after(node) {
                    // Save checkpoint before interrupting
                    if let Some(ref checkpointer) = self.checkpointer {
                        let checkpoint = Checkpoint::new(
                            config.thread_id().unwrap_or("default"),
                            state.to_json(),
                        );
                        let metadata = CheckpointMetadata::new().with_step(iteration);
                        checkpointer.put(&config, checkpoint, metadata).await?;
                    }
                    return Err(RegulaError::Interrupted(node.to_string()));
                }
            }

            // Determine next nodes
            current_nodes = self
                .determine_next_nodes(&current_nodes, &results, &state)
                .await?;

            // Save checkpoint
            if let Some(ref checkpointer) = self.checkpointer {
                let mut checkpoint = Checkpoint::new(
                    config.thread_id().unwrap_or("default"),
                    state.to_json(),
                );
                checkpoint.pending = current_nodes.iter().map(|n| n.to_string()).collect();
                let metadata = CheckpointMetadata::new().with_step(iteration);
                checkpointer.put(&config, checkpoint, metadata).await?;
            }

            iteration += 1;
        }

        state.to_state()
    }

    /// Stream execution updates.
    pub fn stream(
        &self,
        input: S,
        config: RunnableConfig,
        mode: StreamMode,
    ) -> impl futures::Stream<Item = Result<StreamChunk<S>>> + '_ {
        async_stream::stream! {
            let mut state = match DynState::from_state(&input) {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut current_nodes = self.graph.entry_points();
            let max_iterations = config
                .max_iterations()
                .or(self.graph.max_iterations())
                .unwrap_or(100);
            let mut iteration = 0;

            loop {
                if current_nodes.is_empty() || current_nodes.iter().all(|n| n.is_end()) {
                    break;
                }

                if iteration >= max_iterations {
                    yield Err(RegulaError::MaxIterationsExceeded(max_iterations));
                    return;
                }

                // Yield node start events
                for node in &current_nodes {
                    yield Ok(StreamChunk::NodeStart { node: node.clone() });
                }

                // Execute nodes
                let typed_state: S = match state.to_state() {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                let results = match self.execute_nodes(&current_nodes, &typed_state, &config).await {
                    Ok(r) => r,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                // Yield node end events and apply updates
                for (node_id, output) in &results {
                    yield Ok(StreamChunk::NodeEnd {
                        node: node_id.clone(),
                        output: output.get_update().cloned(),
                    });

                    if let Some(update) = output.get_update() {
                        if let Err(e) = state.apply_update(update.clone()) {
                            yield Err(e);
                            return;
                        }
                    }
                }

                state.clear_modified();

                // Yield state update
                match mode {
                    StreamMode::Updates => {
                        match state.to_state() {
                            Ok(s) => yield Ok(StreamChunk::StateUpdate { state: s }),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    StreamMode::Values => {
                        match state.to_state() {
                            Ok(s) => yield Ok(StreamChunk::StateUpdate { state: s }),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                }

                // Determine next nodes
                current_nodes = match self.determine_next_nodes(&current_nodes, &results, &state).await {
                    Ok(n) => n,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                iteration += 1;
            }

            // Yield final state
            match state.to_state() {
                Ok(s) => yield Ok(StreamChunk::Done { final_state: s }),
                Err(e) => yield Err(e),
            }
        }
    }

    /// Execute a set of nodes in parallel.
    async fn execute_nodes(
        &self,
        nodes: &[NodeId],
        state: &S,
        config: &RunnableConfig,
    ) -> Result<Vec<(NodeId, NodeOutput)>> {
        let futures: Vec<_> = nodes
            .iter()
            .filter(|n| !n.is_end())
            .map(|node_id| {
                let node = self.graph.get_node(node_id).cloned();
                let node_id = node_id.clone();
                let state = state.clone();
                let config = config.clone();

                async move {
                    match node {
                        Some(n) => {
                            let output = n.execute(&state, &config).await?;
                            Ok((node_id, output))
                        }
                        None => Err(RegulaError::NodeNotFound(node_id.to_string())),
                    }
                }
            })
            .collect();

        let results: Vec<Result<(NodeId, NodeOutput)>> = join_all(futures).await;
        results.into_iter().collect()
    }

    /// Determine the next nodes to execute based on edges and command outputs.
    async fn determine_next_nodes(
        &self,
        _current: &[NodeId],
        results: &[(NodeId, NodeOutput)],
        state: &DynState,
    ) -> Result<Vec<NodeId>> {
        let mut next_nodes = HashSet::new();

        for (node_id, output) in results {
            // Check if the output has explicit routing
            if let Some(goto) = output.get_goto() {
                for target in goto.targets() {
                    next_nodes.insert(target.clone());
                }
                continue;
            }

            // Otherwise, use graph edges
            let edges = self.graph.edges_from(node_id);
            for edge in edges {
                match edge {
                    Edge::Normal { to, .. } => {
                        next_nodes.insert(to.clone());
                    }
                    Edge::Conditional { router, path_map, .. } => {
                        let typed_state: S = state.to_state()?;
                        let route = router.route(&typed_state).await?;

                        let targets = match route {
                            RouteOutput::One(id) => vec![id],
                            RouteOutput::Many(ids) => ids,
                            RouteOutput::End => vec![end()],
                        };

                        for target in targets {
                            // Apply path_map if present
                            let final_target = if let Some(map) = path_map {
                                map.get(target.as_str())
                                    .cloned()
                                    .unwrap_or(target)
                            } else {
                                target
                            };
                            next_nodes.insert(final_target);
                        }
                    }
                }
            }
        }

        Ok(next_nodes.into_iter().collect())
    }

    /// Get the current state for a thread (from checkpoint).
    pub async fn get_state(&self, config: &RunnableConfig) -> Result<Option<S>> {
        if let Some(ref checkpointer) = self.checkpointer {
            if let Some(tuple) = checkpointer.get(config).await? {
                let state: S = serde_json::from_value(tuple.checkpoint.values)?;
                return Ok(Some(state));
            }
        }
        Ok(None)
    }
}

impl<S: GraphState> Clone for GraphExecutor<S> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            checkpointer: self.checkpointer.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regula_core::{node_fn, router_fn, ChannelSpec, StateGraph, start};
    use std::collections::HashMap;

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

    #[tokio::test]
    async fn test_executor_simple_graph() {
        let node = node_fn(|state: &TestState, _: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": counter + 1,
                    "done": true
                })))
            }
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", node)
            .add_edge(start(), "agent")
            .add_edge("agent", end())
            .compile(Default::default())
            .unwrap();

        let executor = GraphExecutor::new(graph);
        let input = TestState {
            counter: 0,
            done: false,
        };

        let result = executor.invoke(input, RunnableConfig::new()).await.unwrap();
        assert_eq!(result.counter, 1);
        assert!(result.done);
    }

    #[tokio::test]
    async fn test_executor_loop_graph() {
        let agent = node_fn(|state: &TestState, _: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(serde_json::json!({
                    "counter": counter + 1
                })))
            }
        });

        let router = router_fn(|state: &TestState| {
            if state.counter >= 3 {
                RouteOutput::end()
            } else {
                RouteOutput::one("agent")
            }
        });

        let graph = StateGraph::<TestState>::new()
            .add_node("agent", agent)
            .add_edge(start(), "agent")
            .add_conditional_edges("agent", router)
            .compile(Default::default())
            .unwrap();

        let executor = GraphExecutor::new(graph);
        let input = TestState {
            counter: 0,
            done: false,
        };

        let result = executor.invoke(input, RunnableConfig::new()).await.unwrap();
        assert_eq!(result.counter, 3);
    }
}
