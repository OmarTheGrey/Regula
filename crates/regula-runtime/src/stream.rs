//! Streaming types for graph execution.

use regula_core::{GraphState, NodeId};

/// Mode for streaming execution updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamMode {
    /// Stream full state after each step.
    #[default]
    Values,

    /// Stream state deltas after each step.
    Updates,
}

/// A chunk of streaming output from graph execution.
#[derive(Clone, Debug)]
pub enum StreamChunk<S: GraphState> {
    /// A node is starting execution.
    NodeStart {
        /// The node being executed.
        node: NodeId,
    },

    /// A node has completed execution.
    NodeEnd {
        /// The node that completed.
        node: NodeId,
        /// The output from the node, if any.
        output: Option<serde_json::Value>,
    },

    /// State has been updated.
    StateUpdate {
        /// The current state.
        state: S,
    },

    /// Graph execution has completed.
    Done {
        /// The final state.
        final_state: S,
    },

    /// An error occurred during execution.
    Error {
        /// The error message.
        message: String,
    },
}

impl<S: GraphState> StreamChunk<S> {
    /// Check if this is a completion chunk.
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done { .. })
    }

    /// Check if this is an error chunk.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the state if this is a state update or done chunk.
    pub fn state(&self) -> Option<&S> {
        match self {
            Self::StateUpdate { state } => Some(state),
            Self::Done { final_state } => Some(final_state),
            _ => None,
        }
    }
}
