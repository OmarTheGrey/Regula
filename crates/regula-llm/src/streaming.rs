//! Streaming support for LLM responses.
//!
//! This module provides SSE (Server-Sent Events) streaming for real-time
//! token output from LLM APIs.

use crate::message::{Role, ToolCall};
use serde::Deserialize;

/// A streaming chunk from the LLM.
#[derive(Clone, Debug)]
pub enum StreamChunk {
    /// A content delta (partial token).
    Content(String),

    /// A tool call delta.
    ToolCall(ToolCallDelta),

    /// The stream has ended.
    Done,

    /// An error occurred.
    Error(String),
}

/// A partial tool call from streaming.
#[derive(Clone, Debug, Default)]
pub struct ToolCallDelta {
    /// The index of this tool call.
    pub index: usize,
    /// The tool call ID (sent with first chunk).
    pub id: Option<String>,
    /// The function name (sent with first chunk).
    pub name: Option<String>,
    /// The partial arguments.
    pub arguments: Option<String>,
}

/// SSE data chunk from OpenAI-compatible APIs.
#[derive(Debug, Deserialize)]
pub struct SseData {
    pub choices: Vec<SseChoice>,
}

#[derive(Debug, Deserialize)]
pub struct SseChoice {
    pub delta: SseDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SseDelta {
    pub role: Option<Role>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<SseToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct SseToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<SseFunction>,
}

#[derive(Debug, Deserialize)]
pub struct SseFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

impl SseData {
    /// Parse from an SSE data line.
    pub fn parse(line: &str) -> Option<Self> {
        // SSE format: "data: {json}"
        let data = line.strip_prefix("data: ")?;
        
        if data == "[DONE]" {
            return None;
        }

        serde_json::from_str(data).ok()
    }

    /// Convert to a stream chunk.
    pub fn to_chunk(&self) -> Option<StreamChunk> {
        let choice = self.choices.first()?;

        // Check for finish
        if choice.finish_reason.is_some() {
            return Some(StreamChunk::Done);
        }

        // Check for content
        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                return Some(StreamChunk::Content(content.clone()));
            }
        }

        // Check for tool calls
        if let Some(ref tool_calls) = choice.delta.tool_calls {
            if let Some(tc) = tool_calls.first() {
                return Some(StreamChunk::ToolCall(ToolCallDelta {
                    index: tc.index,
                    id: tc.id.clone(),
                    name: tc.function.as_ref().and_then(|f| f.name.clone()),
                    arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                }));
            }
        }

        None
    }
}

/// Accumulator for building complete tool calls from streaming deltas.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    tool_calls: Vec<PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    /// Create a new accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a delta to the accumulator.
    pub fn add_delta(&mut self, delta: ToolCallDelta) {
        // Ensure we have enough slots
        while self.tool_calls.len() <= delta.index {
            self.tool_calls.push(PartialToolCall::default());
        }

        let tc = &mut self.tool_calls[delta.index];

        if let Some(id) = delta.id {
            tc.id = id;
        }
        if let Some(name) = delta.name {
            tc.name = name;
        }
        if let Some(args) = delta.arguments {
            tc.arguments.push_str(&args);
        }
    }

    /// Get the completed tool calls.
    pub fn finish(self) -> Vec<ToolCall> {
        self.tool_calls
            .into_iter()
            .filter(|tc| !tc.id.is_empty())
            .map(|tc| ToolCall::new(tc.id, tc.name, tc.arguments))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_data_parse_content() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let data = SseData::parse(line).unwrap();
        
        let chunk = data.to_chunk().unwrap();
        assert!(matches!(chunk, StreamChunk::Content(s) if s == "Hello"));
    }

    #[test]
    fn test_sse_data_parse_done() {
        let line = "data: [DONE]";
        let data = SseData::parse(line);
        assert!(data.is_none());
    }

    #[test]
    fn test_sse_data_parse_finish_reason() {
        let line = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let data = SseData::parse(line).unwrap();
        
        let chunk = data.to_chunk().unwrap();
        assert!(matches!(chunk, StreamChunk::Done));
    }

    #[test]
    fn test_tool_call_accumulator() {
        let mut acc = ToolCallAccumulator::new();

        // First delta with ID and name
        acc.add_delta(ToolCallDelta {
            index: 0,
            id: Some("call_123".to_string()),
            name: Some("get_weather".to_string()),
            arguments: Some("{\"location\":".to_string()),
        });

        // Second delta with more arguments
        acc.add_delta(ToolCallDelta {
            index: 0,
            id: None,
            name: None,
            arguments: Some("\"NYC\"}".to_string()),
        });

        let tool_calls = acc.finish();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"location\":\"NYC\"}");
    }
}
