//! OpenAI-compatible client implementation.

use crate::client::{LlmClient, LlmConfig, LlmResponse};
use crate::message::{Message, Role, Tool, ToolCall, Usage};
use async_trait::async_trait;
use regula_core::{RegulaError, Result};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible client.
///
/// This client works with OpenAI, Azure OpenAI, Ollama, vLLM, and other
/// providers that implement the OpenAI API format.
#[derive(Clone, Debug)]
pub struct OpenAiClient {
    config: LlmConfig,
    client: reqwest::Client,
}

impl OpenAiClient {
    /// Create a new OpenAI client.
    pub fn new(config: LlmConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create from an API key, using default OpenAI configuration.
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        Self::new(LlmConfig::openai(api_key))
    }

    /// Create from environment variable OPENAI_API_KEY.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| RegulaError::LlmApiKeyMissing)?;
        Ok(Self::from_api_key(api_key))
    }

    /// Build the request body for a completion.
    fn build_request(&self, messages: &[Message], tools: Option<&[Tool]>) -> ChatRequest {
        ChatRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            tools: tools.map(|t| t.to_vec()),
            stream: Some(false),
        }
    }

    /// Send a request to the API.
    async fn send_request(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut req = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.config.api_key));

        if let Some(ref org) = self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        let response = req
            .json(&request)
            .send()
            .await
            .map_err(|e| RegulaError::LlmRequest(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            
            if status.as_u16() == 429 {
                return Err(RegulaError::LlmRateLimited { retry_after: None });
            }

            return Err(RegulaError::LlmApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| RegulaError::LlmResponseParse(e.to_string()))?;

        Ok(chat_response)
    }

    /// Convert API response to LlmResponse.
    fn to_llm_response(&self, response: ChatResponse) -> Result<LlmResponse> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| RegulaError::LlmResponseParse("No choices in response".to_string()))?;

        let message = Message {
            role: choice.message.role,
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice.message.tool_calls,
            tool_call_id: None,
            name: None,
        };

        Ok(LlmResponse {
            message,
            usage: response.usage,
            finish_reason: choice.finish_reason,
        })
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        let request = self.build_request(messages, None);
        let response = self.send_request(request).await?;
        self.to_llm_response(response)
    }

    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<LlmResponse> {
        let request = self.build_request(messages, Some(tools));
        let response = self.send_request(request).await?;
        self.to_llm_response(response)
    }

    fn model(&self) -> &str {
        &self.config.model
    }
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: Role,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_client_new() {
        let client = OpenAiClient::new(LlmConfig::openai("test-key"));
        assert_eq!(client.model(), "gpt-4");
    }

    #[test]
    fn test_build_request_basic() {
        let client = OpenAiClient::new(LlmConfig::openai("test-key"));
        let messages = vec![Message::user("Hello")];
        let request = client.build_request(&messages, None);

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 1);
        assert!(request.tools.is_none());
    }

    #[test]
    fn test_build_request_with_tools() {
        let client = OpenAiClient::new(LlmConfig::openai("test-key"));
        let messages = vec![Message::user("What's the weather?")];
        let tools = vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({}),
        )];

        let request = client.build_request(&messages, Some(&tools));
        assert!(request.tools.is_some());
        assert_eq!(request.tools.unwrap().len(), 1);
    }

    #[test]
    fn test_config_ollama() {
        let config = LlmConfig::ollama("llama2");
        assert!(config.base_url.contains("11434"));
        assert_eq!(config.model, "llama2");
    }
}
