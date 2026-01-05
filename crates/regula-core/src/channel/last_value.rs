//! LastValue channel implementation.
//!
//! This channel keeps only the most recently written value.
//! It is the default channel type for state fields.

use super::{Channel, ChannelSpec};
use crate::error::{RegulaError, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;

/// A channel that stores the last written value.
///
/// This is the default channel type. If multiple values are written
/// during a single super-step without using `AnyValue` semantics,
/// an error will be raised.
///
/// # Type Parameters
///
/// - `V`: The value type stored in this channel.
///
/// # Examples
///
/// ```ignore
/// use regula_core::channel::{LastValue, Channel};
///
/// let mut channel: LastValue<String> = LastValue::new();
/// assert!(!channel.is_set());
///
/// channel.set("hello".to_string());
/// assert_eq!(channel.get_value().unwrap(), "hello");
/// ```
#[derive(Debug)]
pub struct LastValue<V> {
    value: Option<V>,
    _marker: PhantomData<V>,
}

impl<V> LastValue<V> {
    /// Create a new empty LastValue channel.
    pub fn new() -> Self {
        Self {
            value: None,
            _marker: PhantomData,
        }
    }

    /// Create a new LastValue channel with an initial value.
    pub fn with_value(value: V) -> Self {
        Self {
            value: Some(value),
            _marker: PhantomData,
        }
    }

    /// Set the channel value directly.
    pub fn set(&mut self, value: V) {
        self.value = Some(value);
    }

    /// Get a reference to the current value.
    pub fn get_value(&self) -> Option<&V> {
        self.value.as_ref()
    }

    /// Take the current value, leaving the channel empty.
    pub fn take(&mut self) -> Option<V> {
        self.value.take()
    }
}

impl<V> Default for LastValue<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Clone for LastValue<V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            _marker: PhantomData,
        }
    }
}

impl<V> Channel for LastValue<V>
where
    V: Clone + Send + Sync + Serialize + DeserializeOwned + Debug + 'static,
{
    type Value = V;

    fn update(&mut self, values: Vec<Self::Value>) -> Result<bool> {
        match values.len() {
            0 => Ok(false),
            1 => {
                self.value = Some(values.into_iter().next().unwrap());
                Ok(true)
            }
            _ => Err(RegulaError::ChannelTypeMismatch {
                channel: "LastValue".to_string(),
                expected: "single value".to_string(),
                actual: format!("{} values", values.len()),
            }),
        }
    }

    fn get(&self) -> Result<Self::Value> {
        self.value
            .clone()
            .ok_or_else(|| RegulaError::EmptyChannel("LastValue".to_string()))
    }

    fn is_set(&self) -> bool {
        self.value.is_some()
    }

    fn reset(&mut self) {
        self.value = None;
    }

    fn spec(&self) -> ChannelSpec {
        ChannelSpec::LastValue
    }

    fn checkpoint(&self) -> Result<Option<serde_json::Value>> {
        match &self.value {
            Some(v) => Ok(Some(serde_json::to_value(v)?)),
            None => Ok(None),
        }
    }

    fn restore(&mut self, value: Option<serde_json::Value>) -> Result<()> {
        self.value = match value {
            Some(v) => Some(serde_json::from_value(v)?),
            None => None,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_value_new() {
        let channel: LastValue<i32> = LastValue::new();
        assert!(!channel.is_set());
    }

    #[test]
    fn test_last_value_with_value() {
        let channel = LastValue::with_value(42);
        assert!(channel.is_set());
        assert_eq!(channel.get_value(), Some(&42));
    }

    #[test]
    fn test_last_value_set() {
        let mut channel: LastValue<String> = LastValue::new();
        channel.set("hello".to_string());
        assert!(channel.is_set());
        assert_eq!(channel.get_value(), Some(&"hello".to_string()));
    }

    #[test]
    fn test_last_value_update_single() {
        let mut channel: LastValue<i32> = LastValue::new();
        let changed = channel.update(vec![42]).unwrap();
        assert!(changed);
        assert_eq!(channel.get().unwrap(), 42);
    }

    #[test]
    fn test_last_value_update_empty() {
        let mut channel: LastValue<i32> = LastValue::new();
        let changed = channel.update(vec![]).unwrap();
        assert!(!changed);
        assert!(!channel.is_set());
    }

    #[test]
    fn test_last_value_update_multiple_error() {
        let mut channel: LastValue<i32> = LastValue::new();
        let result = channel.update(vec![1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_last_value_reset() {
        let mut channel = LastValue::with_value(42);
        assert!(channel.is_set());
        channel.reset();
        assert!(!channel.is_set());
    }

    #[test]
    fn test_last_value_checkpoint_restore() {
        let mut channel = LastValue::with_value("test".to_string());
        
        let checkpoint = channel.checkpoint().unwrap();
        assert!(checkpoint.is_some());

        let mut new_channel: LastValue<String> = LastValue::new();
        new_channel.restore(checkpoint).unwrap();
        assert_eq!(new_channel.get().unwrap(), "test");
    }

    #[test]
    fn test_last_value_take() {
        let mut channel = LastValue::with_value(42);
        let value = channel.take();
        assert_eq!(value, Some(42));
        assert!(!channel.is_set());
    }
}
