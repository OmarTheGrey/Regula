//! Constants and fundamental types for REGULA framework.
//!
//! This module defines the `NodeId` type and special sentinel nodes
//! `START` and `END` used to mark graph entry and exit points.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a node in the graph.
///
/// `NodeId` is a newtype wrapper around `String` that provides type safety
/// and convenient conversions from string-like types.
///
/// # Examples
///
/// ```
/// use regula_core::NodeId;
///
/// let id: NodeId = "agent".into();
/// assert_eq!(id.as_str(), "agent");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(String);

impl NodeId {
    /// Create a new `NodeId` from a string.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the node ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if this is the START sentinel node.
    pub fn is_start(&self) -> bool {
        self.0 == START_NAME
    }

    /// Check if this is the END sentinel node.
    pub fn is_end(&self) -> bool {
        self.0 == END_NAME
    }

    /// Check if this is a special sentinel node (START or END).
    pub fn is_sentinel(&self) -> bool {
        self.is_start() || self.is_end()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&String> for NodeId {
    fn from(s: &String) -> Self {
        Self(s.clone())
    }
}

impl AsRef<str> for NodeId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for NodeId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for NodeId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for NodeId {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

// ============================================================================
// Sentinel Nodes
// ============================================================================

/// Internal name for the START sentinel node.
const START_NAME: &str = "__start__";

/// Internal name for the END sentinel node.
const END_NAME: &str = "__end__";

/// The START sentinel node.
///
/// This represents the entry point of the graph. Use `start()` when adding
/// edges from the beginning of the graph to the first node(s).
///
/// # Examples
///
/// ```
/// use regula_core::{start, NodeId};
///
/// // Add an edge from START to the first node
/// let s = start();
///
/// assert!(s.is_start());
/// assert!(s.is_sentinel());
/// ```
pub static START: NodeId = NodeId(String::new()); // Placeholder, replaced at runtime

/// The END sentinel node.
///
/// This represents the termination point of the graph. Routes that return
/// `end()` will cause the graph to complete execution.
///
/// # Examples
///
/// ```
/// use regula_core::{end, NodeId};
///
/// // Route to END when done
/// let e = end();
///
/// assert!(e.is_end());
/// assert!(e.is_sentinel());
/// ```
pub static END: NodeId = NodeId(String::new()); // Placeholder, replaced at runtime

// We can't use const with String, so we use lazy initialization pattern
// The actual START and END are created via functions

/// Get the START node ID.
///
/// This is the entry point of every graph.
#[inline]
pub fn start() -> NodeId {
    NodeId::new(START_NAME)
}

/// Get the END node ID.
///
/// This represents graph termination.
#[inline]
pub fn end() -> NodeId {
    NodeId::new(END_NAME)
}

// Re-export as constants using lazy_static pattern would be ideal,
// but for simplicity we provide the functions above.
// Users can also use the string literals directly: "START", "END"

/// Constant for START node name - use in pattern matching.
pub const START_NODE: &str = START_NAME;

/// Constant for END node name - use in pattern matching.
pub const END_NODE: &str = END_NAME;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let id = NodeId::new("agent");
        assert_eq!(id.as_str(), "agent");
    }

    #[test]
    fn test_node_id_from_str() {
        let id: NodeId = "tools".into();
        assert_eq!(id.as_str(), "tools");
    }

    #[test]
    fn test_node_id_equality() {
        let id1 = NodeId::new("agent");
        let id2: NodeId = "agent".into();
        assert_eq!(id1, id2);
        assert_eq!(id1, "agent");
    }

    #[test]
    fn test_sentinel_nodes() {
        let start = start();
        let end = end();

        assert!(start.is_start());
        assert!(!start.is_end());
        assert!(start.is_sentinel());

        assert!(end.is_end());
        assert!(!end.is_start());
        assert!(end.is_sentinel());
    }

    #[test]
    fn test_regular_node_not_sentinel() {
        let id = NodeId::new("agent");
        assert!(!id.is_sentinel());
        assert!(!id.is_start());
        assert!(!id.is_end());
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::new("my_agent");
        assert_eq!(format!("{}", id), "my_agent");
    }

    #[test]
    fn test_node_id_serialization() {
        let id = NodeId::new("agent");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"agent\"");

        let parsed: NodeId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }
}
