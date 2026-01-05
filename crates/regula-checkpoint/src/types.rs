//! Checkpoint types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A checkpoint of graph execution state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique identifier for this checkpoint.
    pub id: String,

    /// The thread ID this checkpoint belongs to.
    pub thread_id: String,

    /// The parent checkpoint ID, if any.
    pub parent_id: Option<String>,

    /// The serialized state values.
    pub values: serde_json::Value,

    /// Channel versions for conflict detection.
    pub channel_versions: HashMap<String, u64>,

    /// Pending node executions.
    pub pending: Vec<String>,

    /// Timestamp when checkpoint was created.
    pub created_at: DateTime<Utc>,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new(thread_id: impl Into<String>, values: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            parent_id: None,
            values,
            channel_versions: HashMap::new(),
            pending: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Create a child checkpoint.
    pub fn child(&self, values: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: self.thread_id.clone(),
            parent_id: Some(self.id.clone()),
            values,
            channel_versions: self.channel_versions.clone(),
            pending: Vec::new(),
            created_at: Utc::now(),
        }
    }
}

/// Metadata about a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// The node that created this checkpoint.
    pub source: Option<String>,

    /// The step number in the execution.
    pub step: usize,

    /// Custom metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            source: None,
            step: 0,
            metadata: HashMap::new(),
        }
    }
}

impl CheckpointMetadata {
    /// Create new metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source node.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the step number.
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Add custom metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A checkpoint with its metadata.
#[derive(Clone, Debug)]
pub struct CheckpointTuple {
    /// The checkpoint data.
    pub checkpoint: Checkpoint,

    /// The checkpoint metadata.
    pub metadata: CheckpointMetadata,
}

impl CheckpointTuple {
    /// Create a new checkpoint tuple.
    pub fn new(checkpoint: Checkpoint, metadata: CheckpointMetadata) -> Self {
        Self {
            checkpoint,
            metadata,
        }
    }
}
