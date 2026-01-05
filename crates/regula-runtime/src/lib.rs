//! REGULA Runtime - Execution engine for the REGULA framework.
//!
//! This crate provides the Pregel-style execution engine for running
//! compiled state graphs. It handles:
//!
//! - Super-step execution with parallel node processing
//! - State management and channel updates
//! - Streaming support for real-time updates
//! - Integration with checkpointing

pub mod executor;
pub mod stream;

pub use executor::GraphExecutor;
pub use stream::{StreamChunk, StreamMode};
