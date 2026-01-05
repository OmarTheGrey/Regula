//! REGULA LLM - LLM client integrations for the REGULA framework.
//!
//! This crate provides OpenAI-compatible LLM clients for use with REGULA agents.
//! It supports:
//!
//! - Chat completions with tool/function calling
//! - Streaming responses with SSE
//! - Custom base URLs for OpenAI, Azure, Ollama, vLLM, etc.

pub mod client;
pub mod message;
pub mod openai;
pub mod streaming;

pub use client::{LlmClient, LlmConfig};
pub use message::{FunctionCall, Message, Role, Tool, ToolCall, Usage};
pub use openai::OpenAiClient;
