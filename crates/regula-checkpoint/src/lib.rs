//! REGULA Checkpoint - Persistence layer for the REGULA framework.
//!
//! This crate provides checkpointing capabilities for saving and restoring
//! graph execution state. It includes:
//!
//! - `Checkpointer` trait for implementing storage backends
//! - `InMemorySaver` for development and testing
//! - Checkpoint types and serialization

pub mod memory;
pub mod traits;
pub mod types;

pub use memory::InMemorySaver;
pub use traits::Checkpointer;
pub use types::{Checkpoint, CheckpointMetadata, CheckpointTuple};
