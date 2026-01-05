//! Channel system for state management.
//!
//! Channels define how state values are stored, updated, and merged when
//! multiple nodes write to the same field during a super-step.
//!
//! # Channel Types
//!
//! - [`LastValue`]: Keeps the most recently written value (default).
//! - [`Reducer`]: Applies a reducer function to combine values (e.g., append to list).
//!
//! # Example
//!
//! ```ignore
//! // In state definition:
//! struct AgentState {
//!     messages: Vec<Message>,     // LastValue (default)
//!     #[reducer(append)]
//!     history: Vec<String>,       // Uses Reducer with append
//! }
//! ```

mod last_value;
mod reducer;

pub use last_value::LastValue;
pub use reducer::{Reducer, ReducerFn, ReducerType};

use crate::error::{RegulaError, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

/// Specification for how a channel behaves.
///
/// This is used during graph construction to determine how state
/// fields should handle concurrent updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelSpec {
    /// Keep the last written value. If multiple values are written
    /// in the same step, an error is raised.
    LastValue,

    /// Apply a reducer function to combine values.
    Reducer(ReducerType),

    /// Value is cleared after each step (not persisted).
    /// This is useful for temporary computation results.
    Ephemeral,

    /// Allow any value, last writer wins (no error on concurrent writes).
    AnyValue,
}

impl Default for ChannelSpec {
    fn default() -> Self {
        Self::LastValue
    }
}

/// Trait for channel implementations.
///
/// Channels manage individual state fields, handling value storage,
/// updates, and checkpoint serialization.
pub trait Channel: Send + Sync + Debug {
    /// The type of value stored in this channel.
    type Value: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Update the channel with new values.
    ///
    /// For LastValue channels, this expects exactly one value.
    /// For Reducer channels, values are combined using the reducer function.
    ///
    /// Returns `true` if the channel value changed.
    fn update(&mut self, values: Vec<Self::Value>) -> Result<bool>;

    /// Get the current value, if set.
    fn get(&self) -> Result<Self::Value>;

    /// Check if the channel has a value.
    fn is_set(&self) -> bool;

    /// Reset the channel to empty state.
    fn reset(&mut self);

    /// Get the channel specification.
    fn spec(&self) -> ChannelSpec;

    /// Serialize the channel value for checkpointing.
    fn checkpoint(&self) -> Result<Option<serde_json::Value>>;

    /// Restore the channel from a checkpoint.
    fn restore(&mut self, value: Option<serde_json::Value>) -> Result<()>;
}

/// A boxed channel that can hold any channel type.
pub type BoxedChannel<V> = Box<dyn Channel<Value = V>>;

/// Wrapper for dynamic channel access with type erasure.
///
/// This allows storing channels of different value types in a single collection.
#[derive(Debug, Clone)]
pub struct DynChannel {
    /// The channel specification.
    pub spec: ChannelSpec,
    /// The channel value as JSON (for type-erased storage).
    value: Option<serde_json::Value>,
    /// Whether the channel has been modified this step.
    modified: bool,
}

impl DynChannel {
    /// Create a new dynamic channel.
    pub fn new(spec: ChannelSpec) -> Self {
        Self {
            spec,
            value: None,
            modified: false,
        }
    }

    /// Create a new dynamic channel with an initial value.
    pub fn with_value<V: Serialize>(spec: ChannelSpec, value: V) -> Result<Self> {
        Ok(Self {
            spec,
            value: Some(serde_json::to_value(value)?),
            modified: false,
        })
    }

    /// Check if the channel has a value.
    pub fn is_set(&self) -> bool {
        self.value.is_some()
    }

    /// Get the value, deserializing to the expected type.
    pub fn get<V: DeserializeOwned>(&self) -> Result<V> {
        match &self.value {
            Some(v) => Ok(serde_json::from_value(v.clone())?),
            None => Err(RegulaError::EmptyChannel("unknown".to_string())),
        }
    }

    /// Set the value, serializing from the given type.
    pub fn set<V: Serialize>(&mut self, value: V) -> Result<()> {
        self.value = Some(serde_json::to_value(value)?);
        self.modified = true;
        Ok(())
    }

    /// Update the channel with a new JSON value.
    pub fn update(&mut self, value: serde_json::Value) -> Result<bool> {
        match self.spec {
            ChannelSpec::LastValue => {
                if self.modified {
                    return Err(RegulaError::ChannelTypeMismatch {
                        channel: "unknown".to_string(),
                        expected: "single write".to_string(),
                        actual: "multiple writes".to_string(),
                    });
                }
                self.value = Some(value);
                self.modified = true;
                Ok(true)
            }
            ChannelSpec::AnyValue | ChannelSpec::Ephemeral => {
                self.value = Some(value);
                self.modified = true;
                Ok(true)
            }
            ChannelSpec::Reducer(ref reducer_type) => {
                // Apply reducer logic
                let new_value = match (&self.value, reducer_type) {
                    (Some(current), ReducerType::Append) => {
                        // Both must be arrays
                        let mut arr = current
                            .as_array()
                            .cloned()
                            .unwrap_or_default();
                        if let Some(new_arr) = value.as_array() {
                            arr.extend(new_arr.iter().cloned());
                        } else {
                            arr.push(value);
                        }
                        serde_json::Value::Array(arr)
                    }
                    (None, ReducerType::Append) => {
                        if value.is_array() {
                            value
                        } else {
                            serde_json::Value::Array(vec![value])
                        }
                    }
                    (Some(current), ReducerType::Add) => {
                        // Both must be numbers
                        let c = current.as_f64().unwrap_or(0.0);
                        let v = value.as_f64().unwrap_or(0.0);
                        serde_json::Value::from(c + v)
                    }
                    (None, ReducerType::Add) => value,
                    (_, ReducerType::Custom(_)) => {
                        // Custom reducers need special handling
                        // For now, just use last value
                        value
                    }
                };
                self.value = Some(new_value);
                self.modified = true;
                Ok(true)
            }
        }
    }

    /// Reset the modified flag for the next step.
    pub fn clear_modified(&mut self) {
        self.modified = false;
        if self.spec == ChannelSpec::Ephemeral {
            self.value = None;
        }
    }

    /// Get the raw JSON value.
    pub fn raw_value(&self) -> Option<&serde_json::Value> {
        self.value.as_ref()
    }

    /// Checkpoint the channel value.
    pub fn checkpoint(&self) -> Option<serde_json::Value> {
        if self.spec == ChannelSpec::Ephemeral {
            None
        } else {
            self.value.clone()
        }
    }

    /// Restore from a checkpoint.
    pub fn restore(&mut self, value: Option<serde_json::Value>) {
        self.value = value;
        self.modified = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_spec_default() {
        assert_eq!(ChannelSpec::default(), ChannelSpec::LastValue);
    }

    #[test]
    fn test_dyn_channel_last_value() {
        let mut channel = DynChannel::new(ChannelSpec::LastValue);
        assert!(!channel.is_set());

        channel.update(serde_json::json!("hello")).unwrap();
        assert!(channel.is_set());

        let value: String = channel.get().unwrap();
        assert_eq!(value, "hello");
    }

    #[test]
    fn test_dyn_channel_reducer_append() {
        let mut channel = DynChannel::new(ChannelSpec::Reducer(ReducerType::Append));
        
        channel.update(serde_json::json!(["a"])).unwrap();
        channel.update(serde_json::json!(["b", "c"])).unwrap();

        let value: Vec<String> = channel.get().unwrap();
        assert_eq!(value, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_dyn_channel_ephemeral() {
        let mut channel = DynChannel::new(ChannelSpec::Ephemeral);
        
        channel.update(serde_json::json!("temp")).unwrap();
        assert!(channel.is_set());

        channel.clear_modified();
        assert!(!channel.is_set()); // Ephemeral values are cleared
    }
}
