//! GraphState trait and related types.
//!
//! The `GraphState` trait defines the requirements for types that can be
//! used as the state in a REGULA graph. State flows through nodes, which
//! read current values and produce updates.

use crate::channel::{ChannelSpec, DynChannel};
use crate::error::Result;
use indexmap::IndexMap;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;

/// Trait for types that can be used as graph state.
///
/// Implement this trait for your state struct to use it with `StateGraph`.
/// The derive macro `#[derive(GraphState)]` can auto-implement this trait.
///
/// # Requirements
///
/// - `Clone`: State is cloned during checkpointing and branching.
/// - `Send + Sync`: State must be thread-safe for parallel node execution.
/// - `Serialize + DeserializeOwned`: State must be serializable for checkpoints.
///
/// # Examples
///
/// ```ignore
/// use regula_core::{GraphState, ChannelSpec};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct MyState {
///     messages: Vec<String>,
///     counter: i32,
/// }
///
/// impl GraphState for MyState {
///     fn channels() -> std::collections::HashMap<String, ChannelSpec> {
///         let mut channels = std::collections::HashMap::new();
///         channels.insert("messages".to_string(), ChannelSpec::LastValue);
///         channels.insert("counter".to_string(), ChannelSpec::LastValue);
///         channels
///     }
/// }
/// ```
pub trait GraphState: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {
    /// Returns the channel specifications for each field in the state.
    ///
    /// The key is the field name, and the value describes how updates
    /// to that field should be handled (last value, reducer, etc.).
    fn channels() -> HashMap<String, ChannelSpec>;

    /// Get the list of field names in the state.
    fn field_names() -> Vec<&'static str> {
        vec![]
    }

    /// Convert the state to a JSON value.
    fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    /// Create a state instance from a JSON value.
    fn from_json(value: serde_json::Value) -> Result<Self> {
        Ok(serde_json::from_value(value)?)
    }

    /// Apply a partial update to the state.
    ///
    /// The update is a JSON object where keys are field names and
    /// values are the new values for those fields.
    fn apply_update(&mut self, update: serde_json::Value) -> Result<()> {
        if let serde_json::Value::Object(updates) = update {
            let mut current = self.to_json()?;
            if let serde_json::Value::Object(ref mut current_obj) = current {
                for (key, value) in updates {
                    current_obj.insert(key, value);
                }
            }
            *self = Self::from_json(current)?;
        }
        Ok(())
    }
}

/// A dynamic state container that stores values in JSON format.
///
/// This is used internally by the runtime to manage state with
/// type-erased channels.
#[derive(Debug, Clone)]
pub struct DynState {
    /// The channel values, keyed by field name.
    channels: IndexMap<String, DynChannel>,
}

impl DynState {
    /// Create a new empty dynamic state.
    pub fn new() -> Self {
        Self {
            channels: IndexMap::new(),
        }
    }

    /// Create a dynamic state from channel specifications.
    pub fn from_specs(specs: HashMap<String, ChannelSpec>) -> Self {
        let mut channels = IndexMap::new();
        for (name, spec) in specs {
            channels.insert(name, DynChannel::new(spec));
        }
        Self { channels }
    }

    /// Create a dynamic state from a GraphState instance.
    pub fn from_state<S: GraphState>(state: &S) -> Result<Self> {
        let specs = S::channels();
        let json = state.to_json()?;

        let mut dyn_state = Self::from_specs(specs);

        if let serde_json::Value::Object(obj) = json {
            for (key, value) in obj {
                if let Some(channel) = dyn_state.channels.get_mut(&key) {
                    channel.set(value)?;
                }
            }
        }

        // Clear modified flags so state is ready for new super-step
        dyn_state.clear_modified();

        Ok(dyn_state)
    }

    /// Convert the dynamic state back to a typed GraphState.
    pub fn to_state<S: GraphState>(&self) -> Result<S> {
        let mut obj = serde_json::Map::new();
        for (name, channel) in &self.channels {
            if let Some(value) = channel.raw_value() {
                obj.insert(name.clone(), value.clone());
            }
        }
        S::from_json(serde_json::Value::Object(obj))
    }

    /// Get a channel by name.
    pub fn get_channel(&self, name: &str) -> Option<&DynChannel> {
        self.channels.get(name)
    }

    /// Get a mutable channel by name.
    pub fn get_channel_mut(&mut self, name: &str) -> Option<&mut DynChannel> {
        self.channels.get_mut(name)
    }

    /// Update a channel with a new value.
    pub fn update_channel(&mut self, name: &str, value: serde_json::Value) -> Result<bool> {
        if let Some(channel) = self.channels.get_mut(name) {
            channel.update(value)
        } else {
            Ok(false)
        }
    }

    /// Apply a partial update (JSON object with field updates).
    pub fn apply_update(&mut self, update: serde_json::Value) -> Result<()> {
        if let serde_json::Value::Object(obj) = update {
            for (key, value) in obj {
                self.update_channel(&key, value)?;
            }
        }
        Ok(())
    }

    /// Clear the modified flag on all channels (called between super-steps).
    pub fn clear_modified(&mut self) {
        for channel in self.channels.values_mut() {
            channel.clear_modified();
        }
    }

    /// Get all channel values as a JSON object.
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        for (name, channel) in &self.channels {
            if let Some(value) = channel.raw_value() {
                obj.insert(name.clone(), value.clone());
            }
        }
        serde_json::Value::Object(obj)
    }

    /// Create a checkpoint of the current state.
    pub fn checkpoint(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        for (name, channel) in &self.channels {
            if let Some(value) = channel.checkpoint() {
                obj.insert(name.clone(), value);
            }
        }
        serde_json::Value::Object(obj)
    }

    /// Restore state from a checkpoint.
    pub fn restore(&mut self, checkpoint: serde_json::Value) {
        if let serde_json::Value::Object(obj) = checkpoint {
            for (name, value) in obj {
                if let Some(channel) = self.channels.get_mut(&name) {
                    channel.restore(Some(value));
                }
            }
        }
    }

    /// Get the list of channel names.
    pub fn channel_names(&self) -> impl Iterator<Item = &String> {
        self.channels.keys()
    }
}

impl Default for DynState {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to create a partial state update.
///
/// This macro creates a `serde_json::Value` object that can be
/// returned from a node to update specific state fields.
///
/// # Examples
///
/// ```ignore
/// use regula_core::partial_state;
///
/// let update = partial_state! {
///     messages: vec![new_message],
///     counter: 42,
/// };
/// ```
#[macro_export]
macro_rules! partial_state {
    ($($field:ident : $value:expr),* $(,)?) => {
        serde_json::json!({
            $(stringify!($field): $value),*
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
    struct TestState {
        messages: Vec<String>,
        counter: i32,
    }

    impl GraphState for TestState {
        fn channels() -> HashMap<String, ChannelSpec> {
            let mut channels = HashMap::new();
            channels.insert("messages".to_string(), ChannelSpec::LastValue);
            channels.insert("counter".to_string(), ChannelSpec::LastValue);
            channels
        }
    }

    #[test]
    fn test_graph_state_to_json() {
        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let json = state.to_json().unwrap();
        assert_eq!(json["messages"][0], "hello");
        assert_eq!(json["counter"], 42);
    }

    #[test]
    fn test_graph_state_from_json() {
        let json = serde_json::json!({
            "messages": ["hello", "world"],
            "counter": 10
        });

        let state: TestState = GraphState::from_json(json).unwrap();
        assert_eq!(state.messages, vec!["hello", "world"]);
        assert_eq!(state.counter, 10);
    }

    #[test]
    fn test_graph_state_apply_update() {
        let mut state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let update = serde_json::json!({
            "counter": 100
        });

        state.apply_update(update).unwrap();
        assert_eq!(state.counter, 100);
        assert_eq!(state.messages, vec!["hello"]); // Unchanged
    }

    #[test]
    fn test_dyn_state_from_state() {
        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let dyn_state = DynState::from_state(&state).unwrap();
        
        assert!(dyn_state.get_channel("messages").is_some());
        assert!(dyn_state.get_channel("counter").is_some());
    }

    #[test]
    fn test_dyn_state_to_state() {
        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let dyn_state = DynState::from_state(&state).unwrap();
        let restored: TestState = dyn_state.to_state().unwrap();

        assert_eq!(restored, state);
    }

    #[test]
    fn test_dyn_state_update_channel() {
        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let mut dyn_state = DynState::from_state(&state).unwrap();
        
        dyn_state.update_channel("counter", serde_json::json!(100)).unwrap();
        
        let restored: TestState = dyn_state.to_state().unwrap();
        assert_eq!(restored.counter, 100);
    }

    #[test]
    fn test_dyn_state_checkpoint_restore() {
        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 42,
        };

        let dyn_state = DynState::from_state(&state).unwrap();
        let checkpoint = dyn_state.checkpoint();

        let mut new_dyn_state = DynState::from_specs(TestState::channels());
        new_dyn_state.restore(checkpoint);

        let restored: TestState = new_dyn_state.to_state().unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn test_partial_state_macro() {
        let update = partial_state! {
            counter: 100,
            messages: vec!["new message"],
        };

        assert_eq!(update["counter"], 100);
        assert_eq!(update["messages"][0], "new message");
    }
}
