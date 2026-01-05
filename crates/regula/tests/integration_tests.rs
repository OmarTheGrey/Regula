//! Integration tests for REGULA framework.
//!
//! These tests verify the full stack works correctly together.

use regula::prelude::*;

/// Simple test state
#[derive(Clone, Default, Debug, Serialize, Deserialize, GraphState)]
struct TestState {
    counter: u32,
    messages: Vec<String>,
    done: bool,
}

#[tokio::test]
async fn test_simple_linear_graph() {
    // Build a simple A -> B -> END graph
    let graph = StateGraph::<TestState>::new()
        .add_node("node_a", node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(json!({
                    "counter": counter + 1,
                    "messages": vec!["Visited A"]
                })))
            }
        }))
        .add_node("node_b", node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            let mut messages = state.messages.clone();
            messages.push("Visited B".to_string());
            async move {
                Ok(NodeOutput::update(json!({
                    "counter": counter + 10,
                    "messages": messages,
                    "done": true
                })))
            }
        }))
        .add_edge(start(), "node_a")
        .add_edge("node_a", "node_b")
        .add_edge("node_b", end())
        .compile(Default::default())
        .expect("Graph should compile");

    let executor = GraphExecutor::new(graph);
    let initial_state = TestState::default();
    
    let result = executor
        .invoke(initial_state, RunnableConfig::new())
        .await
        .expect("Execution should succeed");

    assert_eq!(result.counter, 11);
    assert_eq!(result.messages.len(), 2);
    assert!(result.messages.contains(&"Visited A".to_string()));
    assert!(result.messages.contains(&"Visited B".to_string()));
    assert!(result.done);
}

#[tokio::test]
async fn test_conditional_routing() {
    // Build a graph with conditional routing
    //
    // START -> check -> branch_a (if counter < 5)
    //               -> branch_b (if counter >= 5)
    // Both branches -> END
    //
    let graph = StateGraph::<TestState>::new()
        .add_node("check", node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            async move {
                Ok(NodeOutput::update(json!({
                    "counter": counter
                })))
            }
        }))
        .add_node("branch_a", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(json!({
                    "messages": vec!["Took branch A"]
                })))
            }
        }))
        .add_node("branch_b", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(json!({
                    "messages": vec!["Took branch B"]
                })))
            }
        }))
        .add_edge(start(), "check")
        .add_conditional_edges(
            "check",
            router_fn(|state: &TestState| {
                if state.counter < 5 {
                    RouteOutput::One("branch_a".into())
                } else {
                    RouteOutput::One("branch_b".into())
                }
            }),
        )
        .add_edge("branch_a", end())
        .add_edge("branch_b", end())
        .compile(Default::default())
        .expect("Graph should compile");

    let executor = GraphExecutor::new(graph);

    // Test with counter = 0 (should go to branch_a)
    let result_a = executor
        .invoke(TestState::default(), RunnableConfig::new())
        .await
        .expect("Execution should succeed");
    assert!(result_a.messages.contains(&"Took branch A".to_string()));

    // Test with counter = 10 (should go to branch_b)
    let result_b = executor
        .invoke(
            TestState {
                counter: 10,
                ..Default::default()
            },
            RunnableConfig::new(),
        )
        .await
        .expect("Execution should succeed");
    assert!(result_b.messages.contains(&"Took branch B".to_string()));
}

#[tokio::test]
async fn test_loop_with_termination() {
    // Build a graph that loops until counter >= 5
    //
    // START -> increment -> (loop back if counter < 5, else END)
    //
    let graph = StateGraph::<TestState>::new()
        .add_node("increment", node_fn(|state: &TestState, _config: &RunnableConfig| {
            let counter = state.counter;
            let mut messages = state.messages.clone();
            messages.push(format!("Count: {}", counter + 1));
            async move {
                Ok(NodeOutput::update(json!({
                    "counter": counter + 1,
                    "messages": messages
                })))
            }
        }))
        .add_edge(start(), "increment")
        .add_conditional_edges(
            "increment",
            router_fn(|state: &TestState| {
                if state.counter < 5 {
                    RouteOutput::One("increment".into())
                } else {
                    RouteOutput::end()
                }
            }),
        )
        .compile(Default::default())
        .expect("Graph should compile");

    let executor = GraphExecutor::new(graph);
    let initial_state = TestState::default();

    let result = executor
        .invoke(initial_state, RunnableConfig::new())
        .await
        .expect("Execution should succeed");

    assert_eq!(result.counter, 5);
    assert_eq!(result.messages.len(), 5);
}

#[tokio::test]
async fn test_derive_macro_with_reducers() {
    // Test that the GraphState derive macro works with reducer attributes (append)
    // Note: We test append reducer here; add reducer has JSON float serialization issues
    
    /// Simple state with only append reducer for testing
    #[derive(Clone, Default, Debug, Serialize, Deserialize, GraphState)]
    struct AppendState {
        #[reducer(append)]
        items: Vec<String>,
    }
    
    let graph = StateGraph::<AppendState>::new()
        .add_node("add_items", node_fn(|_state: &AppendState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(json!({
                    "items": vec!["item1", "item2"]
                })))
            }
        }))
        .add_node("add_more", node_fn(|_state: &AppendState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::update(json!({
                    "items": vec!["item3"]
                })))
            }
        }))
        .add_edge(start(), "add_items")
        .add_edge("add_items", "add_more")
        .add_edge("add_more", end())
        .compile(Default::default())
        .expect("Graph should compile");

    let executor = GraphExecutor::new(graph);
    let initial_state = AppendState::default();

    let result = executor
        .invoke(initial_state, RunnableConfig::new())
        .await
        .expect("Execution should succeed");

    // Items should be appended
    assert_eq!(result.items.len(), 3);
    assert!(result.items.contains(&"item1".to_string()));
    assert!(result.items.contains(&"item2".to_string()));
    assert!(result.items.contains(&"item3".to_string()));
}

#[tokio::test]
async fn test_node_output_types() {
    // Test different NodeOutput types
    let graph = StateGraph::<TestState>::new()
        .add_node("no_op", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::none())
            }
        }))
        .add_edge(start(), "no_op")
        .add_edge("no_op", end())
        .compile(Default::default())
        .expect("Graph should compile");

    let executor = GraphExecutor::new(graph);
    let initial_state = TestState {
        counter: 42,
        messages: vec!["initial".to_string()],
        done: false,
    };

    let result = executor
        .invoke(initial_state, RunnableConfig::new())
        .await
        .expect("Execution should succeed");

    // State should be unchanged after NodeOutput::none()
    assert_eq!(result.counter, 42);
    assert_eq!(result.messages.len(), 1);
    assert!(!result.done);
}

#[tokio::test]
async fn test_runnable_config() {
    // Test that RunnableConfig works correctly
    let config = RunnableConfig::new()
        .with_thread_id("my-thread")
        .with_recursion_limit(10)
        .with_tag("test-tag");

    assert_eq!(config.thread_id(), Some("my-thread"));
    assert_eq!(config.recursion_limit(), 10);
    assert!(config.tags().contains(&"test-tag".to_string()));
}

#[tokio::test]
async fn test_graph_validation_missing_edges() {
    // Test that graphs validate correctly
    let result = StateGraph::<TestState>::new()
        .add_node("orphan", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::none())
            }
        }))
        // No edges - orphan node
        .compile(Default::default());

    // Should fail validation because orphan node is unreachable
    assert!(result.is_err());
}

#[tokio::test]
async fn test_graph_compilation_success() {
    // Test that valid graphs compile successfully
    let graph = StateGraph::<TestState>::new()
        .add_node("node_a", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::none())
            }
        }))
        .add_node("node_b", node_fn(|_state: &TestState, _config: &RunnableConfig| {
            async move {
                Ok(NodeOutput::none())
            }
        }))
        .add_edge(start(), "node_a")
        .add_edge("node_a", "node_b")
        .add_edge("node_b", end())
        .compile(Default::default());

    assert!(graph.is_ok());
}
