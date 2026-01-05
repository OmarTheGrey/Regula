//! Runtime configuration for graph execution.
//!
//! `RunnableConfig` carries context, thread identification, and metadata
//! through node execution. It provides a type-safe way to pass additional
//! data to nodes without polluting the state.

use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for a graph execution run.
///
/// `RunnableConfig` provides context and metadata to nodes during execution.
/// It includes thread identification for checkpointing, custom data storage,
/// and execution parameters.
///
/// # Examples
///
/// ```
/// use regula_core::RunnableConfig;
///
/// let config = RunnableConfig::new()
///     .with_thread_id("conversation-1")
///     .with_metadata("user_id", "user-123");
///
/// assert_eq!(config.thread_id(), Some("conversation-1"));
/// ```
#[derive(Clone, Debug)]
pub struct RunnableConfig {
    /// Unique identifier for this execution thread.
    thread_id: Option<String>,

    /// Checkpoint ID to resume from.
    checkpoint_id: Option<String>,

    /// Custom metadata (string key-value pairs).
    metadata: HashMap<String, String>,

    /// Type-erased context values.
    context: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,

    /// Tags for filtering/categorization.
    tags: Vec<String>,

    /// Maximum number of iterations before timeout.
    max_iterations: Option<usize>,

    /// Recursion limit for nested graphs.
    recursion_limit: usize,
}

impl RunnableConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self {
            thread_id: None,
            checkpoint_id: None,
            metadata: HashMap::new(),
            context: HashMap::new(),
            tags: Vec::new(),
            max_iterations: None,
            recursion_limit: 25,
        }
    }

    /// Create a configuration with a random thread ID.
    pub fn with_random_thread() -> Self {
        Self::new().with_thread_id(Uuid::new_v4().to_string())
    }

    /// Set the thread ID.
    ///
    /// The thread ID is used to identify a conversation or session for
    /// checkpointing and state persistence.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Set the checkpoint ID to resume from.
    pub fn with_checkpoint_id(mut self, checkpoint_id: impl Into<String>) -> Self {
        self.checkpoint_id = Some(checkpoint_id.into());
        self
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add multiple metadata entries.
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata.extend(metadata);
        self
    }

    /// Add a typed context value.
    ///
    /// Context values can be retrieved in nodes using `config.get::<T>()`.
    pub fn with_context<T: Send + Sync + 'static>(mut self, value: T) -> Self {
        self.context.insert(TypeId::of::<T>(), Arc::new(value));
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Set the maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Set the recursion limit.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Get the thread ID.
    pub fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_deref()
    }

    /// Get the checkpoint ID.
    pub fn checkpoint_id(&self) -> Option<&str> {
        self.checkpoint_id.as_deref()
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Get all metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get a typed context value.
    ///
    /// Returns `None` if the value was not set or is the wrong type.
    pub fn get_context<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.context
            .get(&TypeId::of::<T>())
            .and_then(|v| v.clone().downcast::<T>().ok())
    }

    /// Get a typed context value, returning an error if not found.
    pub fn require_context<T: Send + Sync + 'static>(&self) -> crate::error::Result<Arc<T>> {
        self.get_context::<T>().ok_or_else(|| {
            crate::error::RegulaError::ConfigKeyNotFound(std::any::type_name::<T>().to_string())
        })
    }

    /// Check if a tag is present.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Get all tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Get the maximum iterations limit.
    pub fn max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }

    /// Get the recursion limit.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Create a child configuration for nested graph execution.
    ///
    /// This decrements the recursion limit and preserves other settings.
    pub fn child(&self) -> crate::error::Result<Self> {
        if self.recursion_limit == 0 {
            return Err(crate::error::RegulaError::MaxIterationsExceeded(0));
        }

        Ok(Self {
            thread_id: self.thread_id.clone(),
            checkpoint_id: None, // Don't inherit checkpoint for nested execution
            metadata: self.metadata.clone(),
            context: self.context.clone(),
            tags: self.tags.clone(),
            max_iterations: self.max_iterations,
            recursion_limit: self.recursion_limit - 1,
        })
    }

    /// Merge another configuration into this one.
    ///
    /// Values from `other` take precedence.
    pub fn merge(mut self, other: Self) -> Self {
        if other.thread_id.is_some() {
            self.thread_id = other.thread_id;
        }
        if other.checkpoint_id.is_some() {
            self.checkpoint_id = other.checkpoint_id;
        }
        self.metadata.extend(other.metadata);
        self.context.extend(other.context);
        self.tags.extend(other.tags);
        if other.max_iterations.is_some() {
            self.max_iterations = other.max_iterations;
        }
        self.recursion_limit = other.recursion_limit;
        self
    }
}

impl Default for RunnableConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable version of RunnableConfig for checkpointing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    /// Thread ID.
    pub thread_id: Option<String>,
    /// Checkpoint ID.
    pub checkpoint_id: Option<String>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// Tags.
    pub tags: Vec<String>,
}

impl From<&RunnableConfig> for ConfigSnapshot {
    fn from(config: &RunnableConfig) -> Self {
        Self {
            thread_id: config.thread_id.clone(),
            checkpoint_id: config.checkpoint_id.clone(),
            metadata: config.metadata.clone(),
            tags: config.tags.clone(),
        }
    }
}

impl From<ConfigSnapshot> for RunnableConfig {
    fn from(snapshot: ConfigSnapshot) -> Self {
        Self {
            thread_id: snapshot.thread_id,
            checkpoint_id: snapshot.checkpoint_id,
            metadata: snapshot.metadata,
            context: HashMap::new(),
            tags: snapshot.tags,
            max_iterations: None,
            recursion_limit: 25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = RunnableConfig::new();
        assert!(config.thread_id().is_none());
        assert!(config.metadata().is_empty());
    }

    #[test]
    fn test_config_with_thread_id() {
        let config = RunnableConfig::new().with_thread_id("thread-1");
        assert_eq!(config.thread_id(), Some("thread-1"));
    }

    #[test]
    fn test_config_with_metadata() {
        let config = RunnableConfig::new()
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        assert_eq!(config.get_metadata("key1"), Some("value1"));
        assert_eq!(config.get_metadata("key2"), Some("value2"));
        assert_eq!(config.get_metadata("key3"), None);
    }

    #[test]
    fn test_config_with_context() {
        #[derive(Clone)]
        struct MyContext {
            value: i32,
        }

        let config = RunnableConfig::new().with_context(MyContext { value: 42 });

        let ctx = config.get_context::<MyContext>().unwrap();
        assert_eq!(ctx.value, 42);
    }

    #[test]
    fn test_config_require_context() {
        let config = RunnableConfig::new().with_context(42i32);

        let result = config.require_context::<i32>();
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), 42);

        let result = config.require_context::<String>();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_tags() {
        let config = RunnableConfig::new()
            .with_tag("tag1")
            .with_tags(vec!["tag2", "tag3"]);

        assert!(config.has_tag("tag1"));
        assert!(config.has_tag("tag2"));
        assert!(config.has_tag("tag3"));
        assert!(!config.has_tag("tag4"));
    }

    #[test]
    fn test_config_child() {
        let config = RunnableConfig::new()
            .with_thread_id("thread-1")
            .with_recursion_limit(5);

        let child = config.child().unwrap();
        assert_eq!(child.thread_id(), Some("thread-1"));
        assert_eq!(child.recursion_limit(), 4);
    }

    #[test]
    fn test_config_child_recursion_limit() {
        let config = RunnableConfig::new().with_recursion_limit(0);
        let result = config.child();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_merge() {
        let config1 = RunnableConfig::new()
            .with_thread_id("thread-1")
            .with_metadata("key1", "value1");

        let config2 = RunnableConfig::new()
            .with_thread_id("thread-2")
            .with_metadata("key2", "value2");

        let merged = config1.merge(config2);
        assert_eq!(merged.thread_id(), Some("thread-2")); // config2 takes precedence
        assert_eq!(merged.get_metadata("key1"), Some("value1"));
        assert_eq!(merged.get_metadata("key2"), Some("value2"));
    }

    #[test]
    fn test_config_snapshot_roundtrip() {
        let config = RunnableConfig::new()
            .with_thread_id("thread-1")
            .with_metadata("key", "value")
            .with_tag("tag1");

        let snapshot = ConfigSnapshot::from(&config);
        let restored = RunnableConfig::from(snapshot);

        assert_eq!(restored.thread_id(), Some("thread-1"));
        assert_eq!(restored.get_metadata("key"), Some("value"));
        assert!(restored.has_tag("tag1"));
    }

    #[test]
    fn test_config_with_random_thread() {
        let config = RunnableConfig::with_random_thread();
        assert!(config.thread_id().is_some());
        assert!(!config.thread_id().unwrap().is_empty());
    }
}
