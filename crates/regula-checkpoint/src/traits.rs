//! Checkpointer trait definition.

use crate::types::{Checkpoint, CheckpointMetadata, CheckpointTuple};
use async_trait::async_trait;
use regula_core::{Result, RunnableConfig};

/// Trait for checkpoint storage backends.
///
/// Implement this trait to create custom storage backends for checkpoints,
/// such as PostgreSQL, SQLite, Redis, or file-based storage.
#[async_trait]
pub trait Checkpointer: Send + Sync {
    /// Get the latest checkpoint for a thread.
    async fn get(&self, config: &RunnableConfig) -> Result<Option<CheckpointTuple>>;

    /// Get a specific checkpoint by ID.
    async fn get_by_id(
        &self,
        config: &RunnableConfig,
        checkpoint_id: &str,
    ) -> Result<Option<CheckpointTuple>>;

    /// Save a new checkpoint.
    async fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<String>;

    /// List checkpoints for a thread.
    async fn list(
        &self,
        config: &RunnableConfig,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>>;

    /// Delete a checkpoint.
    async fn delete(&self, config: &RunnableConfig, checkpoint_id: &str) -> Result<()>;

    /// Delete all checkpoints for a thread.
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}
