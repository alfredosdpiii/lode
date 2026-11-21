use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

const SENSITIVE_PARTS: [&str; 5] = ["password", "secret", "token", "key", "authorization"];

/// Redact values whose key looks like a credential, recursively.
pub fn sanitize(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| {
                    let lowered = key.to_lowercase();
                    if SENSITIVE_PARTS.iter().any(|part| lowered.contains(part)) {
                        (key.clone(), Value::String("[REDACTED]".into()))
                    } else {
                        (key.clone(), sanitize(item))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(sanitize).collect()),
        other => other.clone(),
    }
}

pub fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// One JSON line on stderr: `{"event": ..., "time": ..., **sanitized fields}`.
pub fn log_event(event: &str, fields: Map<String, Value>) {
    let mut payload = match sanitize(&Value::Object(fields)) {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    payload.insert("event".into(), event.into());
    payload.insert(
        "time".into(),
        ((unix_time() * 1000.0).round() / 1000.0).into(),
    );
    eprintln!("{}", crate::json::dumps(&payload));
}

/// Prometheus-style request counter for the daemon.
#[derive(Debug, Default)]
pub struct Metrics {
    counts: Mutex<BTreeMap<(String, String, u16), u64>>,
}

impl Metrics {
    pub fn record(&self, method: &str, path: &str, status: u16) {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *counts
            .entry((method.to_uppercase(), path.to_owned(), status))
            .or_default() += 1;
    }

    pub fn render(&self) -> String {
        let mut lines = vec![
            "# HELP loded_requests_total Total local daemon HTTP requests.".to_owned(),
            "# TYPE loded_requests_total counter".to_owned(),
        ];
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for ((method, path, status), count) in counts.iter() {
            lines.push(format!(
                "loded_requests_total{{method=\"{method}\",path=\"{path}\",status=\"{status}\"}} {count}"
            ));
        }
        lines.join("\n") + "\n"
    }
}
