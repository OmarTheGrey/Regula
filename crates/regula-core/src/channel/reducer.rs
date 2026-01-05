//! Reducer channel implementation.
//!
//! This channel applies a reducer function to combine multiple values
//! written during the same super-step, or to accumulate values across steps.

use super::{Channel, ChannelSpec};
use crate::error::{RegulaError, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::Arc;

/// The type of reducer to apply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReducerType {
    /// Append items to a vector/list.
    Append,
    /// Add numeric values.
    Add,
    /// Custom reducer function (identified by name).
    Custom(String),
}

impl ReducerType {
    /// Get the name of this reducer type.
    pub fn name(&self) -> &str {
        match self {
            ReducerType::Append => "append",
            ReducerType::Add => "add",
            ReducerType::Custom(name) => name,
        }
    }
}

/// A function that reduces two values into one.
pub type ReducerFn<V> = Arc<dyn Fn(V, V) -> V + Send + Sync>;

/// A channel that applies a reducer function to combine values.
///
/// Unlike `LastValue`, this channel can accept multiple values during
/// a single super-step and will combine them using the specified reducer.
///
/// # Type Parameters
///
/// - `V`: The value type stored in this channel.
///
/// # Examples
///
/// ```ignore
/// use regula_core::channel::{Reducer, ReducerType, Channel};
///
/// // Create a reducer channel that appends to a vector
/// let mut channel: Reducer<Vec<String>> = Reducer::new(ReducerType::Append, None);
///
/// // Multiple updates are combined
/// channel.update(vec![vec!["a".to_string()]]).unwrap();
/// channel.update(vec![vec!["b".to_string()]]).unwrap();
///
/// // Result: ["a", "b"]
/// ```
pub struct Reducer<V> {
    value: Option<V>,
    reducer_type: ReducerType,
    custom_reducer: Option<ReducerFn<V>>,
    _marker: PhantomData<V>,
}

impl<V: Debug> Debug for Reducer<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reducer")
            .field("value", &self.value)
            .field("reducer_type", &self.reducer_type)
            .field("has_custom_reducer", &self.custom_reducer.is_some())
            .finish()
    }
}

impl<V> Reducer<V> {
    /// Create a new Reducer channel with the specified reducer type.
    pub fn new(reducer_type: ReducerType, initial: Option<V>) -> Self {
        Self {
            value: initial,
            reducer_type,
            custom_reducer: None,
            _marker: PhantomData,
        }
    }

    /// Create a new Reducer channel with a custom reducer function.
    pub fn with_custom(reducer_fn: ReducerFn<V>, initial: Option<V>) -> Self {
        Self {
            value: initial,
            reducer_type: ReducerType::Custom("custom".to_string()),
            custom_reducer: Some(reducer_fn),
            _marker: PhantomData,
        }
    }

    /// Get the reducer type.
    pub fn reducer_type(&self) -> &ReducerType {
        &self.reducer_type
    }

    /// Get a reference to the current value.
    pub fn get_value(&self) -> Option<&V> {
        self.value.as_ref()
    }
}

impl<V: Clone> Clone for Reducer<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            reducer_type: self.reducer_type.clone(),
            custom_reducer: self.custom_reducer.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement Channel for Vec<T> with Append reducer
impl<T> Channel for Reducer<Vec<T>>
where
    T: Clone + Send + Sync + Serialize + DeserializeOwned + Debug + 'static,
{
    type Value = Vec<T>;

    fn update(&mut self, values: Vec<Self::Value>) -> Result<bool> {
        if values.is_empty() {
            return Ok(false);
        }

        match &self.reducer_type {
            ReducerType::Append => {
                let current = self.value.take().unwrap_or_default();
                let mut result = current;
                for v in values {
                    result.extend(v);
                }
                self.value = Some(result);
                Ok(true)
            }
            ReducerType::Custom(_) => {
                if let Some(ref reducer) = self.custom_reducer {
                    let mut result = self.value.take().unwrap_or_default();
                    for v in values {
                        result = reducer(result, v);
                    }
                    self.value = Some(result);
                    Ok(true)
                } else {
                    Err(RegulaError::ReducerFailed {
                        channel: "Reducer".to_string(),
                        message: "No custom reducer function provided".to_string(),
                    })
                }
            }
            _ => Err(RegulaError::ReducerFailed {
                channel: "Reducer<Vec<T>>".to_string(),
                message: format!(
                    "Reducer type {:?} not applicable to Vec<T>",
                    self.reducer_type
                ),
            }),
        }
    }

    fn get(&self) -> Result<Self::Value> {
        self.value
            .clone()
            .ok_or_else(|| RegulaError::EmptyChannel("Reducer".to_string()))
    }

    fn is_set(&self) -> bool {
        self.value.is_some()
    }

    fn reset(&mut self) {
        self.value = None;
    }

    fn spec(&self) -> ChannelSpec {
        ChannelSpec::Reducer(self.reducer_type.clone())
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

/// Numeric reducer for summing values.
#[derive(Debug)]
pub struct NumericReducer<V> {
    value: Option<V>,
    reducer_type: ReducerType,
    _marker: PhantomData<V>,
}

impl<V> NumericReducer<V> {
    /// Create a new numeric reducer with the Add reducer type.
    pub fn new(initial: Option<V>) -> Self {
        Self {
            value: initial,
            reducer_type: ReducerType::Add,
            _marker: PhantomData,
        }
    }
}

impl<V: Clone> Clone for NumericReducer<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            reducer_type: self.reducer_type.clone(),
            _marker: PhantomData,
        }
    }
}

macro_rules! impl_numeric_reducer {
    ($($t:ty),*) => {
        $(
            impl Channel for NumericReducer<$t> {
                type Value = $t;

                fn update(&mut self, values: Vec<Self::Value>) -> Result<bool> {
                    if values.is_empty() {
                        return Ok(false);
                    }

                    let current = self.value.unwrap_or(0 as $t);
                    let sum: $t = values.into_iter().fold(current, |acc, v| acc + v);
                    self.value = Some(sum);
                    Ok(true)
                }

                fn get(&self) -> Result<Self::Value> {
                    self.value
                        .ok_or_else(|| RegulaError::EmptyChannel("NumericReducer".to_string()))
                }

                fn is_set(&self) -> bool {
                    self.value.is_some()
                }

                fn reset(&mut self) {
                    self.value = None;
                }

                fn spec(&self) -> ChannelSpec {
                    ChannelSpec::Reducer(ReducerType::Add)
                }

                fn checkpoint(&self) -> Result<Option<serde_json::Value>> {
                    match self.value {
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
        )*
    };
}

impl_numeric_reducer!(i32, i64, u32, u64, f32, f64, usize, isize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reducer_type_name() {
        assert_eq!(ReducerType::Append.name(), "append");
        assert_eq!(ReducerType::Add.name(), "add");
        assert_eq!(ReducerType::Custom("my_fn".to_string()).name(), "my_fn");
    }

    #[test]
    fn test_vec_reducer_append() {
        let mut channel: Reducer<Vec<String>> = Reducer::new(ReducerType::Append, None);
        
        channel.update(vec![vec!["a".to_string()]]).unwrap();
        assert_eq!(channel.get().unwrap(), vec!["a"]);

        channel.update(vec![vec!["b".to_string(), "c".to_string()]]).unwrap();
        assert_eq!(channel.get().unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_vec_reducer_with_initial() {
        let mut channel: Reducer<Vec<i32>> = Reducer::new(
            ReducerType::Append,
            Some(vec![1, 2]),
        );
        
        channel.update(vec![vec![3, 4]]).unwrap();
        assert_eq!(channel.get().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_vec_reducer_multiple_updates() {
        let mut channel: Reducer<Vec<i32>> = Reducer::new(ReducerType::Append, None);
        
        // Multiple values in one update
        channel.update(vec![vec![1], vec![2], vec![3]]).unwrap();
        assert_eq!(channel.get().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_numeric_reducer_add() {
        let mut channel: NumericReducer<i32> = NumericReducer::new(None);
        
        channel.update(vec![5]).unwrap();
        assert_eq!(channel.get().unwrap(), 5);

        channel.update(vec![3, 2]).unwrap();
        assert_eq!(channel.get().unwrap(), 10);
    }

    #[test]
    fn test_numeric_reducer_with_initial() {
        let mut channel: NumericReducer<f64> = NumericReducer::new(Some(10.0));
        
        channel.update(vec![5.5]).unwrap();
        assert_eq!(channel.get().unwrap(), 15.5);
    }

    #[test]
    fn test_reducer_reset() {
        let mut channel: Reducer<Vec<i32>> = Reducer::new(
            ReducerType::Append,
            Some(vec![1, 2, 3]),
        );
        
        assert!(channel.is_set());
        channel.reset();
        assert!(!channel.is_set());
    }

    #[test]
    fn test_reducer_checkpoint_restore() {
        let mut channel: Reducer<Vec<String>> = Reducer::new(ReducerType::Append, None);
        channel.update(vec![vec!["a".to_string(), "b".to_string()]]).unwrap();

        let checkpoint = channel.checkpoint().unwrap();

        let mut new_channel: Reducer<Vec<String>> = Reducer::new(ReducerType::Append, None);
        new_channel.restore(checkpoint).unwrap();

        assert_eq!(new_channel.get().unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn test_custom_reducer() {
        // Custom reducer that keeps only unique values
        let reducer: ReducerFn<Vec<i32>> = Arc::new(|mut a, b| {
            for item in b {
                if !a.contains(&item) {
                    a.push(item);
                }
            }
            a
        });

        let mut channel: Reducer<Vec<i32>> = Reducer::with_custom(reducer, None);
        
        channel.update(vec![vec![1, 2, 3]]).unwrap();
        channel.update(vec![vec![2, 3, 4]]).unwrap();
        
        assert_eq!(channel.get().unwrap(), vec![1, 2, 3, 4]);
    }
}
