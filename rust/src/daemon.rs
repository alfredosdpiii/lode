//! `loded` local HTTP daemon (`daemon.py`): same routes, status codes and JSON keys.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::cli::impact_results;
use crate::config::sqlite_path;
use crate::context::{ContextOptions, build_context_pack};
use crate::graph::Direction;
use crate::observability::{Metrics, log_event};
use crate::{Error, Result, storage};

const WORKER_THREADS: usize = 8;

/// A response before it is written: status, content type and body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub content_type: &'static str,
    pub body: String,
}

impl Reply {
    fn json(payload: &Value, status: u16) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: crate::json::dumps_pretty(payload),
        }
    }

    fn error(message: impl std::fmt::Display, status: u16) -> Self {
        Self::json(&json!({"ok": false, "error": message.to_string()}), status)
    }
}

/// Shared daemon state: data directory and request metrics.
#[derive(Debug, Default)]
pub struct App {
    pub data_dir: PathBuf,
    pub metrics: Metrics,
}

/// `parse_qs` with blank values dropped, first value per key.
fn query_params(query: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        if !value.is_empty() {
            params
                .entry(key.into_owned())
                .or_insert_with(|| value.into_owned());
        }
    }
    params
}

fn parse_int(raw: &str, name: &str) -> Result<i64> {
    raw.trim()
        .parse()
        .map_err(|_| Error::invalid(format!("{name} must be an integer")))
}

/// `str(value)` the way Python renders JSON scalars.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        other => other.to_string(),
    }
}

fn is_blank(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null)) || value.and_then(Value::as_str) == Some("")
}

fn body_int(body: &Map<String, Value>, key: &str) -> Result<Option<i64>> {
    let value = body.get(key);
    if is_blank(value) {
        return Ok(None);
    }
    parse_int(&py_str(value.unwrap_or(&Value::Null)), key).map(Some)
}

fn required<'a>(body: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    let value = body.get(key);
    if is_blank(value) {
        return Err(Error::invalid(format!("missing required field: {key}")));
    }
    value.ok_or_else(|| Error::invalid(format!("missing required field: {key}")))
}

fn direction(value: Option<String>) -> Result<Direction> {
    Direction::parse(value.as_deref().filter(|v| !v.is_empty()).unwrap_or("both"))
}

fn repo_arg(body: &Map<String, Value>) -> Option<String> {
    body.get("repo").filter(|v| !v.is_null()).map(py_str)
}

impl App {
    fn conn(&self) -> Result<rusqlite::Connection> {
        storage::connect(&sqlite_path(&self.data_dir))
    }

    /// Route one request. `raw_path` includes the query string.
    pub fn handle(&self, method: &str, raw_path: &str, body: &[u8]) -> Reply {
        let (path, query) = raw_path.split_once('?').unwrap_or((raw_path, ""));
        let result = match method {
            "GET" => self.handle_get(path, query),
            "POST" => self.handle_post(raw_path, body),
            _ => return Reply::error(format!("Unsupported method ('{method}')"), 501),
        };
        result.unwrap_or_else(|err| Reply::error(err, 400))
    }

    fn handle_get(&self, path: &str, query: &str) -> Result<Reply> {
        let params = query_params(query);
        let int = |key: &str, default: i64| match params.get(key) {
            Some(raw) => parse_int(raw, key),
            None => Ok(default),
        };
        match path {
            "/health" => Ok(Reply::json(&json!({"ok": true, "service": "lode"}), 200)),
            "/metrics" if crate::features::enabled("metrics")? => Ok(Reply {
                status: 200,
                content_type: "text/plain",
                body: self.metrics.render(),
            }),
            "/status" => Ok(Reply::json(
                &json!({"ok": true, "repos": storage::list_repos(&self.conn()?)?}),
                200,
            )),
            "/search" => {
                let query = params
                    .get("q")
                    .or_else(|| params.get("query"))
                    .cloned()
                    .unwrap_or_default();
                let limit = int("limit", 20)?;
                let conn = self.conn()?;
                let repo_id = storage::repo_filter(&conn, params.get("repo").map(String::as_str))?;
                let results = storage::search_nodes(&conn, &query, repo_id.as_deref(), limit)?;
                Ok(Reply::json(&json!({"ok": true, "results": results}), 200))
            }
            "/impact" => {
                let target = params
                    .get("target")
                    .or_else(|| params.get("q"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::invalid("missing required query parameter: target or q")
                    })?;
                let limit = int("limit", 5)?;
                let neighbor_limit = int("neighbor_limit", 200)?;
                let depth = params
                    .get("depth")
                    .map(|raw| parse_int(raw, "depth"))
                    .transpose()?;
                let max_nodes = int("max_nodes", 1000)?;
                let direction = direction(params.get("direction").cloned())?;
                let conn = self.conn()?;
                let repo_id = storage::repo_filter(&conn, params.get("repo").map(String::as_str))?;
                let results = impact_results(
                    &conn,
                    &target,
                    repo_id.as_deref(),
                    limit,
                    neighbor_limit,
                    depth,
                    max_nodes,
                    direction,
                )?;
                Ok(Reply::json(
                    &json!({"ok": true, "query": target, "results": results}),
                    200,
                ))
            }
            _ => Ok(Reply::error("not found", 404)),
        }
    }

    fn handle_post(&self, path: &str, raw_body: &[u8]) -> Result<Reply> {
        let body = if raw_body.is_empty() {
            Map::new()
        } else {
            match serde_json::from_slice::<Value>(raw_body)? {
                Value::Object(map) => map,
                _ => return Err(Error::invalid("request body must be a JSON object")),
            }
        };
        match path {
            "/index" => {
                let repo = match required(&body, "path")? {
                    Value::String(text) => text.clone(),
                    _ => return Err(Error::invalid("expected str, bytes or os.PathLike object")),
                };
                let stats =
                    crate::indexer::index_repo(Path::new(&repo), &sqlite_path(&self.data_dir))?;
                let mut payload = serde_json::to_value(stats)?;
                payload["ok"] = true.into();
                Ok(Reply::json(&payload, 200))
            }
            "/search" => {
                let query = py_str(required(&body, "query")?);
                let limit = body_int(&body, "limit")?.unwrap_or(20);
                let conn = self.conn()?;
                let repo_id = storage::repo_filter(&conn, repo_arg(&body).as_deref())?;
                let results = storage::search_nodes(&conn, &query, repo_id.as_deref(), limit)?;
                Ok(Reply::json(&json!({"ok": true, "results": results}), 200))
            }
            "/context" => {
                let query = py_str(required(&body, "query")?);
                let repo = repo_arg(&body);
                let budget = body_int(&body, "budget")?.unwrap_or(6000);
                let limit = body_int(&body, "limit")?.unwrap_or(10).max(0) as usize;
                let options = ContextOptions {
                    repo_path: repo.as_deref(),
                    budget,
                    limit,
                    ..ContextOptions::default()
                };
                let mut payload = build_context_pack(&self.conn()?, &query, &options)?;
                payload["ok"] = true.into();
                Ok(Reply::json(&payload, 200))
            }
            "/impact" => {
                let target = py_str(required(&body, "target")?);
                let limit = body_int(&body, "limit")?.unwrap_or(5);
                let neighbor_limit = body_int(&body, "neighbor_limit")?.unwrap_or(200);
                let depth = body_int(&body, "depth")?;
                let max_nodes = body_int(&body, "max_nodes")?.unwrap_or(1000);
                let direction =
                    direction(body.get("direction").filter(|v| !v.is_null()).map(py_str))?;
                let conn = self.conn()?;
                let repo_id = storage::repo_filter(&conn, repo_arg(&body).as_deref())?;
                let results = impact_results(
                    &conn,
                    &target,
                    repo_id.as_deref(),
                    limit,
                    neighbor_limit,
                    depth,
                    max_nodes,
                    direction,
                )?;
                Ok(Reply::json(
                    &json!({"ok": true, "query": target, "results": results}),
                    200,
                ))
            }
            _ => Ok(Reply::error("not found", 404)),
        }
    }

    fn record(&self, request_id: &str, method: &str, raw_path: &str, status: u16) {
        let path = raw_path.split_once('?').map_or(raw_path, |(path, _)| path);
        self.metrics.record(method, path, status);
        let mut fields = Map::new();
        fields.insert("request_id".into(), request_id.into());
        fields.insert("method".into(), method.into());
        fields.insert("path".into(), path.into());
        fields.insert("status".into(), status.into());
        log_event("http_request", fields);
    }
}

fn header(value: &str, name: &str) -> Option<tiny_http::Header> {
    tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()).ok()
}

fn serve_one(app: &App, mut request: tiny_http::Request) {
    let method = request.method().as_str().to_uppercase();
    let raw_path = request.url().to_owned();
    let request_id = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("x-request-id"))
        .map(|h| h.value.as_str().to_owned())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let mut body = Vec::new();
    let reply = match std::io::Read::read_to_end(request.as_reader(), &mut body) {
        Ok(_) => app.handle(&method, &raw_path, &body),
        Err(err) => Reply::error(err, 400),
    };
    let mut response = tiny_http::Response::from_string(reply.body).with_status_code(reply.status);
    for (name, value) in [
        ("content-type", reply.content_type),
        ("x-request-id", request_id.as_str()),
    ] {
        if let Some(h) = header(value, name) {
            response.add_header(h);
        }
    }
    // The client may already be gone; nothing useful to do with a write error.
    let _ = request.respond(response);
    app.record(&request_id, &method, &raw_path, reply.status);
}

/// Bind and serve until the process is stopped.
pub fn serve(host: &str, port: u16, data_dir: &Path) -> Result<()> {
    let server =
        tiny_http::Server::http((host, port)).map_err(|err| Error::runtime(err.to_string()))?;
    let server = Arc::new(server);
    let app = Arc::new(App {
        data_dir: data_dir.to_path_buf(),
        metrics: Metrics::default(),
    });
    println!(
        "loded listening on http://{host}:{port} data_dir={}",
        data_dir.display()
    );
    let workers: Vec<_> = (0..WORKER_THREADS)
        .map(|_| {
            let (server, app) = (Arc::clone(&server), Arc::clone(&app));
            std::thread::spawn(move || {
                for request in server.incoming_requests() {
                    serve_one(&app, request);
                }
            })
        })
        .collect();
    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> (tempfile::TempDir, App) {
        let dir = tempfile::tempdir().unwrap();
        let app = App {
            data_dir: dir.path().to_path_buf(),
            metrics: Metrics::default(),
        };
        (dir, app)
    }

    #[test]
    fn health_and_not_found() {
        let (_dir, app) = app();
        let reply = app.handle("GET", "/health", b"");
        assert_eq!(reply.status, 200);
        assert_eq!(
            serde_json::from_str::<Value>(&reply.body).unwrap(),
            json!({"ok": true, "service": "lode"})
        );
        assert_eq!(app.handle("GET", "/nope", b"").status, 404);
        assert_eq!(app.handle("POST", "/nope", b"{}").status, 404);
    }

    #[test]
    fn bad_input_is_400() {
        let (_dir, app) = app();
        let reply = app.handle("GET", "/impact", b"");
        assert_eq!(reply.status, 400);
        assert!(
            reply
                .body
                .contains("missing required query parameter: target or q")
        );
        assert_eq!(app.handle("GET", "/search?q=x&limit=abc", b"").status, 400);
        assert_eq!(app.handle("POST", "/search", b"[]").status, 400);
        assert_eq!(
            app.handle("POST", "/search", b"{\"query\": \"\"}").status,
            400
        );
        assert_eq!(
            app.handle(
                "POST",
                "/impact",
                b"{\"target\": \"x\", \"direction\": \"sideways\"}"
            )
            .status,
            400
        );
    }
}
