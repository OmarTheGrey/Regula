//! Error types for REGULA framework.
//!
//! This module defines the comprehensive error enum used throughout the framework,
//! covering graph construction, execution, state management, checkpointing, and LLM errors.

use std::time::Duration;
use thiserror::Error;

/// The main error type for REGULA operations.
///
/// This enum covers all possible error conditions that can occur during
/// graph construction, compilation, and execution.
#[derive(Error, Debug)]
pub enum RegulaError {
    // =========================================================================
    // Graph Construction Errors
    // =========================================================================
    /// A node with this name already exists in the graph.
    #[error("Node '{0}' already exists in graph")]
    DuplicateNode(String),

    /// The specified node was not found in the graph.
    #[error("Node '{0}' not found in graph")]
    NodeNotFound(String),

    /// An edge references a non-existent source node.
    #[error("Invalid edge: source node '{0}' does not exist")]
    InvalidEdgeSource(String),

    /// An edge references a non-existent target node.
    #[error("Invalid edge: target node '{0}' does not exist")]
    InvalidEdgeTarget(String),

    /// The graph has no entry point (no edge from START).
    #[error("Graph has no entry point (no edge from START)")]
    NoEntryPoint,

    /// The graph contains a cycle without a valid exit condition.
    #[error("Graph contains a cycle without exit condition: {0}")]
    InvalidCycle(String),

    /// A node has no incoming edges and is unreachable.
    #[error("Unreachable node: '{0}' has no incoming edges")]
    UnreachableNode(String),

    /// Multiple edges from the same source without conditional routing.
    #[error("Ambiguous routing: node '{0}' has multiple outgoing edges without conditional routing")]
    AmbiguousRouting(String),

    /// A conditional edge has no valid targets defined.
    #[error("Conditional edge from '{0}' has no valid targets")]
    NoConditionalTargets(String),

    // =========================================================================
    // Execution Errors
    // =========================================================================
    /// A node's execution failed.
    #[error("Node '{node}' execution failed: {message}")]
    NodeExecution {
        /// The name of the node that failed.
        node: String,
        /// The error message.
        message: String,
        /// The underlying error, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A router returned an invalid or unknown target node.
    #[error("Router for node '{0}' returned invalid target: '{1}'")]
    InvalidRouteTarget(String, String),

    /// The graph exceeded the maximum number of iterations.
    #[error("Maximum iterations ({0}) exceeded - possible infinite loop")]
    MaxIterationsExceeded(usize),

    /// Graph execution was interrupted by a breakpoint or interrupt() call.
    #[error("Graph execution was interrupted at node '{0}'")]
    Interrupted(String),

    /// Graph execution was cancelled.
    #[error("Graph execution was cancelled")]
    Cancelled,

    /// The graph has already completed and cannot continue.
    #[error("Graph has already completed execution")]
    AlreadyCompleted,

    /// No pending tasks to execute.
    #[error("No pending tasks to execute")]
    NoPendingTasks,

    // =========================================================================
    // State / Channel Errors
    // =========================================================================
    /// Attempted to read from a channel that has no value.
    #[error("Channel '{0}' is empty (no value set)")]
    EmptyChannel(String),

    /// Type mismatch when reading or writing a channel.
    #[error("Channel '{channel}' type mismatch: expected {expected}, got {actual}")]
    ChannelTypeMismatch {
        /// The channel name.
        channel: String,
        /// The expected type.
        expected: String,
        /// The actual type encountered.
        actual: String,
    },

    /// A reducer function failed to combine values.
    #[error("Failed to apply reducer on channel '{channel}': {message}")]
    ReducerFailed {
        /// The channel name.
        channel: String,
        /// The error message.
        message: String,
    },

    /// State serialization or deserialization failed.
    #[error("State serialization failed: {0}")]
    StateSerialization(#[from] serde_json::Error),

    /// A required state field is missing.
    #[error("Required state field '{0}' is missing")]
    MissingStateField(String),

    /// State update contained an unknown field.
    #[error("Unknown field in state update: '{0}'")]
    UnknownStateField(String),

    // =========================================================================
    // Checkpoint Errors
    // =========================================================================
    /// No checkpoint found for the given thread ID.
    #[error("Checkpoint not found for thread '{0}'")]
    CheckpointNotFound(String),

    /// Checkpoint storage operation failed.
    #[error("Checkpoint storage error: {0}")]
    CheckpointStorage(String),

    /// Failed to deserialize a checkpoint.
    #[error("Checkpoint deserialization failed: {0}")]
    CheckpointDeserialization(String),

    /// Checkpoint version mismatch.
    #[error("Checkpoint version mismatch: expected {expected}, got {actual}")]
    CheckpointVersionMismatch {
        /// The expected version.
        expected: String,
        /// The actual version.
        actual: String,
    },

    /// No checkpointer configured but checkpoint operation requested.
    #[error("No checkpointer configured")]
    NoCheckpointer,

    // =========================================================================
    // LLM Errors
    // =========================================================================
    /// HTTP request to LLM API failed.
    #[error("LLM API request failed: {0}")]
    LlmRequest(String),

    /// LLM API returned an error response.
    #[error("LLM API error (status {status}): {message}")]
    LlmApiError {
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },

    /// Failed to parse LLM response.
    #[error("LLM response parsing failed: {0}")]
    LlmResponseParse(String),

    /// LLM API rate limit exceeded.
    #[error("LLM rate limited, retry after {retry_after:?}")]
    LlmRateLimited {
        /// Suggested retry delay, if provided.
        retry_after: Option<Duration>,
    },

    /// Error during LLM streaming.
    #[error("LLM streaming error: {0}")]
    LlmStreaming(String),

    /// LLM API key is missing or invalid.
    #[error("LLM API key is missing or invalid")]
    LlmApiKeyMissing,

    /// LLM request timed out.
    #[error("LLM request timed out after {0:?}")]
    LlmTimeout(Duration),

    // =========================================================================
    // Configuration Errors
    // =========================================================================
    /// Invalid configuration value.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Required configuration is missing.
    #[error("Missing required configuration: {0}")]
    MissingConfig(String),

    /// Configuration key not found in context.
    #[error("Configuration key '{0}' not found in context")]
    ConfigKeyNotFound(String),

    /// Configuration value type mismatch.
    #[error("Configuration key '{key}' type mismatch: expected {expected}")]
    ConfigTypeMismatch {
        /// The configuration key.
        key: String,
        /// The expected type.
        expected: String,
    },

    // =========================================================================
    // Generic Errors
    // =========================================================================
    /// Internal framework error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Lock acquisition failed (concurrency error).
    #[error("Failed to acquire lock: {0}")]
    LockError(String),

    /// Generic I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapper for other error types.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl RegulaError {
    /// Create a node execution error with a source error.
    pub fn node_execution(
        node: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        let message = source.to_string();
        Self::NodeExecution {
            node: node.into(),
            message,
            source: Some(Box::new(source)),
        }
    }

    /// Create a node execution error with just a message.
    pub fn node_execution_msg(node: impl Into<String>, message: impl Into<String>) -> Self {
        Self::NodeExecution {
            node: node.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::LlmRateLimited { .. }
                | Self::LlmTimeout(_)
                | Self::LlmRequest(_)
                | Self::LockError(_)
        )
    }

    /// Check if this error is a cancellation.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Interrupted(_))
    }
}

/// Result type alias for REGULA operations.
pub type Result<T> = std::result::Result<T, RegulaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RegulaError::DuplicateNode("agent".to_string());
        assert_eq!(err.to_string(), "Node 'agent' already exists in graph");
    }

    #[test]
    fn test_node_execution_error() {
        let err = RegulaError::node_execution_msg("agent", "connection refused");
        assert!(err.to_string().contains("agent"));
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_is_retryable() {
        assert!(RegulaError::LlmRateLimited { retry_after: None }.is_retryable());
        assert!(RegulaError::LlmTimeout(Duration::from_secs(30)).is_retryable());
        assert!(!RegulaError::Cancelled.is_retryable());
    }
}
