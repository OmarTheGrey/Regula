//! LLM client trait and configuration.

use crate::message::{Message, Tool};
use async_trait::async_trait;
use regula_core::Result;
use std::time::Duration;

/// Response from an LLM completion.
#[derive(Clone, Debug)]
pub struct LlmResponse {
    /// The response message.
    pub message: Message,
    /// Token usage statistics.
    pub usage: Option<crate::message::Usage>,
    /// The reason the completion finished.
    pub finish_reason: Option<String>,
}

/// Configuration for an LLM client.
#[derive(Clone, Debug)]
pub struct LlmConfig {
    /// Base URL for the API.
    pub base_url: String,
    /// API key for authentication.
    pub api_key: String,
    /// Model identifier.
    pub model: String,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Request timeout.
    pub timeout: Duration,
    /// Organization ID (for OpenAI).
    pub organization: Option<String>,
}

impl LlmConfig {
    /// Create a new configuration for OpenAI.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: api_key.into(),
            model: "gpt-4".into(),
            temperature: None,
            max_tokens: None,
            timeout: Duration::from_secs(60),
            organization: None,
        }
    }

    /// Create a new configuration for Azure OpenAI.
    pub fn azure(endpoint: impl Into<String>, api_key: impl Into<String>, deployment: impl Into<String>) -> Self {
        Self {
            base_url: endpoint.into(),
            api_key: api_key.into(),
            model: deployment.into(),
            temperature: None,
            max_tokens: None,
            timeout: Duration::from_secs(60),
            organization: None,
        }
    }

    /// Create a new configuration for Ollama.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            base_url: "http://localhost:11434/v1".into(),
            api_key: "ollama".into(),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            timeout: Duration::from_secs(300),
            organization: None,
        }
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum tokens.
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the organization.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self::openai("")
    }
}

/// Trait for LLM clients.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Complete a chat conversation.
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse>;

    /// Complete a chat with tool calling support.
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<LlmResponse>;

    /// Get the model name.
    fn model(&self) -> &str;
}
