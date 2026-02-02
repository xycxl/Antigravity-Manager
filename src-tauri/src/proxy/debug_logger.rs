use serde_json::Value;
use tokio::fs;
use std::path::PathBuf;
use futures::StreamExt;

use crate::proxy::config::DebugLoggingConfig;

/// Token 使用量统计结构体
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_tokens: u32,
    pub total_tokens: u32,
}

/// 获取 ISO 8601 格式时间戳
pub fn get_iso_timestamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// 计算请求耗时（毫秒）
pub fn calculate_duration_ms(start_time: std::time::Instant) -> u64 {
    start_time.elapsed().as_millis() as u64
}

fn build_filename(prefix: &str, trace_id: Option<&str>) -> String {
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%.3f");
    let tid = trace_id.unwrap_or("unknown");
    format!("{}_{}_{}.json", ts, tid, prefix)
}

fn resolve_output_dir(cfg: &DebugLoggingConfig) -> Option<PathBuf> {
    if let Some(dir) = cfg.output_dir.as_ref() {
        return Some(PathBuf::from(dir));
    }
    if let Ok(data_dir) = crate::modules::account::get_data_dir() {
        return Some(data_dir.join("debug_logs"));
    }
    None
}

pub async fn write_debug_payload(
    cfg: &DebugLoggingConfig,
    trace_id: Option<&str>,
    prefix: &str,
    payload: &Value,
) {
    if !cfg.enabled {
        return;
    }

    let output_dir = match resolve_output_dir(cfg) {
        Some(dir) => dir,
        None => {
            tracing::warn!("[Debug-Log] Enabled but output_dir is not available.");
            return;
        }
    };

    if let Err(e) = fs::create_dir_all(&output_dir).await {
        tracing::warn!("[Debug-Log] Failed to create output dir: {}", e);
        return;
    }

    let filename = build_filename(prefix, trace_id);
    let path = output_dir.join(filename);

    match serde_json::to_vec_pretty(payload) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&path, bytes).await {
                tracing::warn!("[Debug-Log] Failed to write file: {}", e);
            }
        }
        Err(e) => {
            tracing::warn!("[Debug-Log] Failed to serialize payload: {}", e);
        }
    }
}

pub fn is_enabled(cfg: &DebugLoggingConfig) -> bool {
    cfg.enabled
}


/// SSE 解析结果结构体
struct ParsedSseResult {
    thinking_content: String,
    response_content: String,
    token_usage: Option<TokenUsage>,
}

/// 解析 SSE 流式数据，提取 thinking、正文内容和 token 统计
fn parse_sse_stream(raw: &str) -> ParsedSseResult {
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    let mut final_usage: Option<TokenUsage> = None;

    for line in raw.lines() {
        let line = line.trim();
        if !line.starts_with("data: ") {
            continue;
        }
        let json_str = &line[6..]; // 去掉 "data: " 前缀
        if json_str.is_empty() || json_str == "[DONE]" {
            continue;
        }

        // 尝试解析 JSON
        if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
            // Gemini/v1internal 格式: response.candidates[0].content.parts[0]
            if let Some(response) = parsed.get("response") {
                // 解析 usageMetadata
                if let Some(usage) = response.get("usageMetadata") {
                    let input = usage.get("promptTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let output = usage.get("candidatesTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let cached = usage.get("cachedContentTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let total = usage.get("totalTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    
                    final_usage = Some(TokenUsage {
                        input_tokens: input,
                        output_tokens: output,
                        cached_tokens: cached,
                        total_tokens: total,
                    });
                }
                
                // 解析内容
                if let Some(candidates) = response.get("candidates").and_then(|c| c.as_array()) {
                    for candidate in candidates {
                        if let Some(parts) = candidate.get("content")
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                        {
                            for part in parts {
                                let text = part.get("text")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");
                                let is_thought = part.get("thought")
                                    .and_then(|t| t.as_bool())
                                    .unwrap_or(false);
                                
                                if !text.is_empty() {
                                    if is_thought {
                                        thinking_parts.push(text.to_string());
                                    } else {
                                        content_parts.push(text.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // OpenAI 格式兼容: choices[0].delta.content
            else if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array()) {
                for choice in choices {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            if !content.is_empty() {
                                content_parts.push(content.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    ParsedSseResult {
        thinking_content: thinking_parts.join(""),
        response_content: content_parts.join(""),
        token_usage: final_usage,
    }
}

pub fn wrap_reqwest_stream_with_debug(
    stream: std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    cfg: DebugLoggingConfig,
    trace_id: String,
    prefix: &'static str,
    meta: Value,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>> {
    if !is_enabled(&cfg) {
        return stream;
    }

    let wrapped = async_stream::stream! {
        let start_time = std::time::Instant::now();
        let mut collected: Vec<u8> = Vec::new();
        let mut inner = stream;
        while let Some(item) = inner.next().await {
            if let Ok(bytes) = &item {
                collected.extend_from_slice(bytes);
            }
            yield item;
        }

        let duration_ms = calculate_duration_ms(start_time);
        let timestamp = get_iso_timestamp();
        let raw_text = String::from_utf8_lossy(&collected).to_string();
        let parsed = parse_sse_stream(&raw_text);
        
        let mut payload = serde_json::json!({
            "kind": "upstream_response",
            "trace_id": trace_id,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "meta": meta,
        });
        
        // 添加 thinking 内容（如果有）
        if !parsed.thinking_content.is_empty() {
            payload["thinking_content"] = serde_json::Value::String(parsed.thinking_content);
        }
        // 添加响应内容（如果有）
        if !parsed.response_content.is_empty() {
            payload["response_content"] = serde_json::Value::String(parsed.response_content);
        }
        // 添加 token 统计（如果有）
        if let Some(usage) = parsed.token_usage {
            payload["token_usage"] = serde_json::json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_tokens": usage.cached_tokens,
                "total_tokens": usage.total_tokens,
            });
        }

        write_debug_payload(&cfg, Some(&trace_id), prefix, &payload).await;
    };

    Box::pin(wrapped)
}
