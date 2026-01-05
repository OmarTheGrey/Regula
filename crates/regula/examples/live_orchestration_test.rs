//! Live Orchestration Test - Multi-Step Agent with Conditional Routing
//!
//! This example demonstrates a real agentic workflow with REGULA:
//! - Multiple nodes (researcher, analyzer, synthesizer)
//! - Conditional routing based on LLM decisions
//! - Multi-step orchestration with state accumulation
//!
//! The workflow:
//!   START -> researcher -> should_continue? -> analyzer -> synthesizer -> END
//!                              |
//!                              +-> (loop back to researcher if more research needed)
//!
//! # Running
//!
//! ```bash
//! $env:GOOGLE_API_KEY = "your-api-key"
//! cargo run --example live_orchestration_test
//! ```

use regula::prelude::*;
use std::env;
use std::io::{self, Write};
use std::time::Duration;

/// Research agent state with multiple fields for orchestration
#[derive(Clone, Default, Debug, Serialize, Deserialize, GraphState)]
struct ResearchState {
    /// The original research question
    question: String,
    /// Research findings accumulated across iterations
    findings: Vec<String>,
    /// The analysis of the findings
    analysis: String,
    /// The final synthesized answer
    final_answer: String,
    /// Current iteration count
    iteration: i32,
    /// Maximum iterations allowed
    max_iterations: i32,
    /// Whether research is complete
    research_complete: bool,
    /// Messages for LLM calls
    messages: Vec<Message>,
}

/// Provider selection
enum Provider {
    OpenRouter,
    Google,
}

/// Get the API key and provider from environment or prompt
fn get_api_key_and_provider() -> (String, Provider) {
    // Try OpenRouter first (more generous free tier)
    if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        if !key.is_empty() {
            println!("[OK] Using API key from OPENROUTER_API_KEY environment variable");
            return (key, Provider::OpenRouter);
        }
    }
    
    // Try Google
    if let Ok(key) = env::var("GOOGLE_API_KEY") {
        if !key.is_empty() {
            println!("[OK] Using API key from GOOGLE_API_KEY environment variable");
            return (key, Provider::Google);
        }
    }
    
    // Prompt user
    println!("Choose provider:");
    println!("  1. OpenRouter (recommended - better free tier)");
    println!("  2. Google Gemini");
    print!("Enter choice (1 or 2): ");
    io::stdout().flush().unwrap();
    
    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    
    let provider = if choice.trim() == "2" {
        print!("Enter your Google AI API key: ");
        Provider::Google
    } else {
        print!("Enter your OpenRouter API key: ");
        Provider::OpenRouter
    };
    
    io::stdout().flush().unwrap();
    let mut key = String::new();
    io::stdin().read_line(&mut key).unwrap();
    
    (key.trim().to_string(), provider)
}

/// Create the LLM client config based on provider
fn create_llm_config(api_key: &str, provider: &Provider) -> LlmConfig {
    match provider {
        Provider::OpenRouter => LlmConfig::openai(api_key)
            .with_base_url("https://openrouter.ai/api/v1")
            .with_model("z-ai/glm-4.7")  // GPT-4o mini - very cheap
            .with_timeout(Duration::from_secs(60)),
        Provider::Google => LlmConfig::openai(api_key)
            .with_base_url("https://generativelanguage.googleapis.com/v1beta/openai")
            .with_model("gemini-2.0-flash")
            .with_timeout(Duration::from_secs(60)),
    }
}

/// Call LLM with retry logic for rate limiting
async fn call_llm_with_retry(
    client: &OpenAiClient,
    messages: &[Message],
    max_retries: u32,
) -> Result<Message> {
    let mut attempt = 0;
    loop {
        match client.complete(messages).await {
            Ok(response) => return Ok(response.message),
            Err(e) => {
                attempt += 1;
                let error_str = format!("{}", e);
                if error_str.contains("rate limit") && attempt < max_retries {
                    let delay = 2u64.pow(attempt); // Exponential backoff: 2, 4, 8 seconds
                    println!("    [RETRY] Rate limited, waiting {} seconds...", delay);
                    tokio::time::sleep(Duration::from_secs(delay)).await;
                } else {
                    return Err(e);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("================================================================");
    println!("   REGULA Live Orchestration Test - Multi-Step Research Agent   ");
    println!("================================================================");
    println!();
    println!("This test demonstrates a multi-node agentic workflow:");
    println!("  1. RESEARCHER: Gathers information about the topic");
    println!("  2. ROUTER: Decides if more research is needed");
    println!("  3. ANALYZER: Analyzes the collected findings");
    println!("  4. SYNTHESIZER: Creates the final answer");
    println!();

    let (api_key, provider) = get_api_key_and_provider();
    if api_key.is_empty() {
        eprintln!("[ERROR] No API key provided");
        std::process::exit(1);
    }

    let provider_name = match provider {
        Provider::OpenRouter => "OpenRouter",
        Provider::Google => "Google Gemini 2.0 Flash",
    };
    println!();
    println!("Using provider: {}", provider_name);
    println!();
    println!("Building orchestration graph...");
    println!();

    let llm_config = create_llm_config(&api_key, &provider);

    // =========================================================================
    // NODE 1: RESEARCHER - Gathers information
    // =========================================================================
    let researcher_config = llm_config.clone();
    let researcher_node = node_fn(move |state: &ResearchState, _config: &RunnableConfig| {
        let question = state.question.clone();
        let iteration = state.iteration;
        let existing_findings = state.findings.clone();
        let config = researcher_config.clone();
        
        async move {
            println!();
            println!("  [RESEARCHER] Iteration {} - Gathering information...", iteration + 1);
            
            let context = if existing_findings.is_empty() {
                "This is your first research pass.".to_string()
            } else {
                format!("Previous findings:\n{}", existing_findings.join("\n"))
            };
            
            let messages = vec![
                Message::system(
                    "You are a research assistant. Provide ONE specific fact or insight about the topic. \
                     Be concise (1-2 sentences). If you have prior findings, add something NEW."
                ),
                Message::user(format!(
                    "Research topic: {}\n\n{}\n\nProvide one new finding:",
                    question, context
                )),
            ];
            
            let client = OpenAiClient::new(config);
            let response = call_llm_with_retry(&client, &messages, 3).await?;
            let finding = response.content.trim().to_string();
            
            println!("  [RESEARCHER] Found: \"{}\"", 
                if finding.len() > 60 { format!("{}...", &finding[..60]) } else { finding.clone() });
            
            // Append the new finding to existing findings
            let mut updated_findings = existing_findings.clone();
            updated_findings.push(finding);
            
            Ok(NodeOutput::update(json!({
                "findings": updated_findings,
                "iteration": iteration + 1
            })))
        }
    });

    // =========================================================================
    // NODE 2: ANALYZER - Analyzes collected findings
    // =========================================================================
    let analyzer_config = llm_config.clone();
    let analyzer_node = node_fn(move |state: &ResearchState, _config: &RunnableConfig| {
        let findings = state.findings.clone();
        let question = state.question.clone();
        let config = analyzer_config.clone();
        
        async move {
            println!();
            println!("  [ANALYZER] Analyzing {} findings...", findings.len());
            
            let messages = vec![
                Message::system(
                    "You are an analyst. Synthesize the research findings into key insights. \
                     Be concise (2-3 sentences max)."
                ),
                Message::user(format!(
                    "Question: {}\n\nFindings:\n{}\n\nProvide your analysis:",
                    question,
                    findings.iter().enumerate()
                        .map(|(i, f)| format!("{}. {}", i + 1, f))
                        .collect::<Vec<_>>()
                        .join("\n")
                )),
            ];
            
            let client = OpenAiClient::new(config);
            let response = call_llm_with_retry(&client, &messages, 3).await?;
            let analysis = response.content.trim().to_string();
            
            println!("  [ANALYZER] Analysis complete ({} chars)", analysis.len());
            
            Ok(NodeOutput::update(json!({
                "analysis": analysis
            })))
        }
    });

    // =========================================================================
    // NODE 3: SYNTHESIZER - Creates final answer
    // =========================================================================
    let synthesizer_config = llm_config.clone();
    let synthesizer_node = node_fn(move |state: &ResearchState, _config: &RunnableConfig| {
        let question = state.question.clone();
        let findings = state.findings.clone();
        let analysis = state.analysis.clone();
        let config = synthesizer_config.clone();
        
        async move {
            println!();
            println!("  [SYNTHESIZER] Creating final answer...");
            
            let messages = vec![
                Message::system(
                    "You are a helpful assistant. Provide a clear, comprehensive answer \
                     based on the research and analysis. Be informative but concise."
                ),
                Message::user(format!(
                    "Original question: {}\n\nResearch findings:\n{}\n\nAnalysis: {}\n\n\
                     Provide the final answer:",
                    question,
                    findings.join("\n- "),
                    analysis
                )),
            ];
            
            let client = OpenAiClient::new(config);
            let response = call_llm_with_retry(&client, &messages, 3).await?;
            let final_answer = response.content.trim().to_string();
            
            println!("  [SYNTHESIZER] Final answer ready ({} chars)", final_answer.len());
            
            Ok(NodeOutput::update(json!({
                "final_answer": final_answer
            })))
        }
    });

    // =========================================================================
    // BUILD THE GRAPH
    // =========================================================================
    let graph = StateGraph::<ResearchState>::new()
        .add_node("researcher", researcher_node)
        .add_node("analyzer", analyzer_node)
        .add_node("synthesizer", synthesizer_node)
        // Start with researcher
        .add_edge(start(), "researcher")
        // Conditional: continue research or analyze
        .add_conditional_edges("researcher", router_fn(|state: &ResearchState| {
            let dominated_iterations = state.iteration >= state.max_iterations;
            let has_enough_findings = state.findings.len() >= 2;
            
            if dominated_iterations || has_enough_findings {
                println!();
                println!("  [ROUTER] -> Moving to ANALYZER (collected {} findings)", state.findings.len());
                RouteOutput::one("analyzer")
            } else {
                println!();
                println!("  [ROUTER] -> Back to RESEARCHER (need more findings)");
                RouteOutput::one("researcher")
            }
        }))
        // After analysis, synthesize
        .add_edge("analyzer", "synthesizer")
        // End after synthesis
        .add_edge("synthesizer", end())
        .compile(Default::default())?;

    println!("[OK] Orchestration graph compiled successfully");
    println!();
    println!("Graph structure:");
    println!("  START -> researcher -> [conditional] -> analyzer -> synthesizer -> END");
    println!("                ^              |");
    println!("                +--------------+ (loop if more research needed)");
    println!();

    // Create executor
    let executor = GraphExecutor::new(graph);

    // Research question
    let research_question = "What are the main benefits of using Rust for building AI agent frameworks?";
    
    println!("================================================================");
    println!("Research Question:");
    println!("  \"{}\"", research_question);
    println!("================================================================");
    println!();

    // Initial state
    let initial_state = ResearchState {
        question: research_question.to_string(),
        findings: vec![],
        analysis: String::new(),
        final_answer: String::new(),
        iteration: 0,
        max_iterations: 3,
        research_complete: false,
        messages: vec![],
    };

    println!("Executing orchestration...");
    println!("----------------------------------------------------------------");

    let start_time = std::time::Instant::now();
    let result = executor.invoke(initial_state, RunnableConfig::new()).await;
    let elapsed = start_time.elapsed();

    println!();
    println!("----------------------------------------------------------------");
    println!();

    match result {
        Ok(final_state) => {
            println!("================================================================");
            println!("                   ORCHESTRATION RESULTS                        ");
            println!("================================================================");
            println!();
            println!("[OK] Orchestration completed successfully!");
            println!();
            println!("Statistics:");
            println!("  - Research iterations: {}", final_state.iteration);
            println!("  - Findings collected: {}", final_state.findings.len());
            println!("  - Total execution time: {:?}", elapsed);
            println!();
            println!("----------------------------------------------------------------");
            println!("FINDINGS:");
            println!("----------------------------------------------------------------");
            for (i, finding) in final_state.findings.iter().enumerate() {
                println!("  {}. {}", i + 1, finding);
                println!();
            }
            println!("----------------------------------------------------------------");
            println!("ANALYSIS:");
            println!("----------------------------------------------------------------");
            println!("{}", final_state.analysis);
            println!();
            println!("----------------------------------------------------------------");
            println!("FINAL ANSWER:");
            println!("----------------------------------------------------------------");
            println!("{}", final_state.final_answer);
            println!();
            println!("================================================================");
            println!("                  ORCHESTRATION PASSED                          ");
            println!("================================================================");
        }
        Err(e) => {
            println!("================================================================");
            println!("                  ORCHESTRATION FAILED                          ");
            println!("================================================================");
            println!();
            println!("[ERROR] {}", e);
            println!();
            std::process::exit(1);
        }
    }

    Ok(())
}
