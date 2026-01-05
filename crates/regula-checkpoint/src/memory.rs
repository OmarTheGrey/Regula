//! In-memory checkpoint storage.

use crate::traits::Checkpointer;
use crate::types::{Checkpoint, CheckpointMetadata, CheckpointTuple};
use async_trait::async_trait;
use indexmap::IndexMap;
use regula_core::{RegulaError, Result, RunnableConfig};
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory checkpoint storage for development and testing.
///
/// This implementation stores all checkpoints in memory and is not
/// persistent across process restarts. Use it for development, testing,
/// or short-lived applications.
///
/// # Thread Safety
///
/// `InMemorySaver` uses `RwLock` internally and is safe to share
/// across threads.
///
/// # Example
///
/// ```
/// use regula_checkpoint::InMemorySaver;
///
/// let saver = InMemorySaver::new();
/// // Use with CompiledGraph
/// ```
pub struct InMemorySaver {
    /// Checkpoints indexed by thread_id -> checkpoint_id -> tuple
    storage: RwLock<HashMap<String, IndexMap<String, CheckpointTuple>>>,
}

impl InMemorySaver {
    /// Create a new in-memory saver.
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(HashMap::new()),
        }
    }

    /// Get the number of checkpoints stored.
    pub fn len(&self) -> usize {
        self.storage
            .read()
            .unwrap()
            .values()
            .map(|m| m.len())
            .sum()
    }

    /// Check if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all checkpoints.
    pub fn clear(&self) {
        self.storage.write().unwrap().clear();
    }
}

impl Default for InMemorySaver {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InMemorySaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemorySaver")
            .field("checkpoints", &self.len())
            .finish()
    }
}

#[async_trait]
impl Checkpointer for InMemorySaver {
    async fn get(&self, config: &RunnableConfig) -> Result<Option<CheckpointTuple>> {
        let thread_id = config
            .thread_id()
            .ok_or_else(|| RegulaError::MissingConfig("thread_id".to_string()))?;

        let storage = self.storage.read().unwrap();
        Ok(storage
            .get(thread_id)
            .and_then(|m| m.values().last())
            .cloned())
    }

    async fn get_by_id(
        &self,
        config: &RunnableConfig,
        checkpoint_id: &str,
    ) -> Result<Option<CheckpointTuple>> {
        let thread_id = config
            .thread_id()
            .ok_or_else(|| RegulaError::MissingConfig("thread_id".to_string()))?;

        let storage = self.storage.read().unwrap();
        Ok(storage
            .get(thread_id)
            .and_then(|m| m.get(checkpoint_id))
            .cloned())
    }

    async fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<String> {
        let thread_id = config
            .thread_id()
            .ok_or_else(|| RegulaError::MissingConfig("thread_id".to_string()))?;

        let checkpoint_id = checkpoint.id.clone();
        let tuple = CheckpointTuple::new(checkpoint, metadata);

        let mut storage = self.storage.write().unwrap();
        storage
            .entry(thread_id.to_string())
            .or_insert_with(IndexMap::new)
            .insert(checkpoint_id.clone(), tuple);

        Ok(checkpoint_id)
    }

    async fn list(
        &self,
        config: &RunnableConfig,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>> {
        let thread_id = config
            .thread_id()
            .ok_or_else(|| RegulaError::MissingConfig("thread_id".to_string()))?;

        let storage = self.storage.read().unwrap();
        let checkpoints: Vec<_> = storage
            .get(thread_id)
            .map(|m| {
                let iter = m.values().rev();
                match limit {
                    Some(n) => iter.take(n).cloned().collect(),
                    None => iter.cloned().collect(),
                }
            })
            .unwrap_or_default();

        Ok(checkpoints)
    }

    async fn delete(&self, config: &RunnableConfig, checkpoint_id: &str) -> Result<()> {
        let thread_id = config
            .thread_id()
            .ok_or_else(|| RegulaError::MissingConfig("thread_id".to_string()))?;

        let mut storage = self.storage.write().unwrap();
        if let Some(thread_checkpoints) = storage.get_mut(thread_id) {
            thread_checkpoints.shift_remove(checkpoint_id);
        }

        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.remove(thread_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_saver_new() {
        let saver = InMemorySaver::new();
        assert!(saver.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_saver_put_get() {
        let saver = InMemorySaver::new();
        let config = RunnableConfig::new().with_thread_id("thread-1");

        let checkpoint = Checkpoint::new("thread-1", serde_json::json!({"counter": 42}));
        let metadata = CheckpointMetadata::new().with_step(1);

        let id = saver.put(&config, checkpoint, metadata).await.unwrap();
        assert!(!id.is_empty());

        let result = saver.get(&config).await.unwrap();
        assert!(result.is_some());

        let tuple = result.unwrap();
        assert_eq!(tuple.checkpoint.values["counter"], 42);
        assert_eq!(tuple.metadata.step, 1);
    }

    #[tokio::test]
    async fn test_in_memory_saver_list() {
        let saver = InMemorySaver::new();
        let config = RunnableConfig::new().with_thread_id("thread-1");

        // Add multiple checkpoints
        for i in 0..5 {
            let checkpoint = Checkpoint::new("thread-1", serde_json::json!({"step": i}));
            let metadata = CheckpointMetadata::new().with_step(i);
            saver.put(&config, checkpoint, metadata).await.unwrap();
        }

        let all = saver.list(&config, None).await.unwrap();
        assert_eq!(all.len(), 5);

        let limited = saver.list(&config, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_in_memory_saver_delete() {
        let saver = InMemorySaver::new();
        let config = RunnableConfig::new().with_thread_id("thread-1");

        let checkpoint = Checkpoint::new("thread-1", serde_json::json!({}));
        let id = saver
            .put(&config, checkpoint, CheckpointMetadata::new())
            .await
            .unwrap();

        assert!(!saver.is_empty());

        saver.delete(&config, &id).await.unwrap();
        let result = saver.get(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_saver_delete_thread() {
        let saver = InMemorySaver::new();
        let config = RunnableConfig::new().with_thread_id("thread-1");

        let checkpoint = Checkpoint::new("thread-1", serde_json::json!({}));
        saver
            .put(&config, checkpoint, CheckpointMetadata::new())
            .await
            .unwrap();

        saver.delete_thread("thread-1").await.unwrap();
        assert!(saver.is_empty());
    }
}
