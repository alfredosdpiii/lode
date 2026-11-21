//! `lode` command line (`cli.py`): same subcommands, flags, JSON keys and exit codes.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use rusqlite::Connection;
use serde_json::{Map, Value, json};

use crate::config::{default_data_dir, kuzu_path, resolve_path, sqlite_path};
use crate::context::{ContextOptions, build_context_pack};
use crate::graph::{Direction, impact_report, impact_targets};
use crate::storage::{self, EmbeddingRow, NodeRow};
use crate::text::{char_len, take_chars};
use crate::{Error, Result};

pub const EMBEDDING_TEXT_MAX_INPUT_CHARS: usize = 384;

#[derive(Debug, Parser)]
#[command(
    name = "lode",
    about = "Fully local repository knowledge graph for agents."
)]
pub struct Cli {
    /// Lode data directory
    #[arg(long = "data-dir")]
    pub data_dir: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Index a repository
    Index {
        path: PathBuf,
        #[arg(long = "sync-kuzu")]
        sync_kuzu: bool,
        #[arg(long)]
        json: bool,
    },
    /// Show indexed repositories
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Search indexed symbols, docs, and files
    Search {
        query: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long, default_value_t = 20, allow_negative_numbers = true)]
        limit: i64,
        #[arg(long)]
        json: bool,
    },
    /// Find a symbol by exact or partial name
    Symbol {
        name: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long, default_value_t = 20, allow_negative_numbers = true)]
        limit: i64,
        #[arg(long)]
        json: bool,
    },
    /// Build a token-budgeted context pack for an agent task
    Context {
        query: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long, default_value_t = 6000, allow_negative_numbers = true)]
        budget: i64,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Boost files the given repo-relative file depends on (library option in Python)
        #[arg(long = "from-path")]
        from_path: Option<String>,
        /// Leave this repo-relative path out of the results; repeatable
        #[arg(long = "exclude-path")]
        exclude_path: Vec<String>,
        /// Skip the related-neighbors section
        #[arg(long = "no-related")]
        no_related: bool,
        /// Re-rank the top hits with the local code-embedding model (needs `--features embed`)
        #[arg(long)]
        rerank: bool,
        #[arg(long)]
        json: bool,
    },
    /// Show direct graph neighbors for a node ID
    Neighbors {
        node_id: String,
        #[arg(long, default_value_t = 80, allow_negative_numbers = true)]
        limit: i64,
        #[arg(long)]
        json: bool,
    },
    /// Show callers, callees, and blast radius around a symbol or node
    Impact {
        target: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long, default_value_t = 5, allow_negative_numbers = true)]
        limit: i64,
        #[arg(
            long = "neighbor-limit",
            default_value_t = 200,
            allow_negative_numbers = true
        )]
        neighbor_limit: i64,
        /// Optional BFS depth limit for noisy graphs. Omit, or pass 0, for full traversal
        #[arg(long, allow_negative_numbers = true)]
        depth: Option<i64>,
        /// Safety cap for nodes in blast radius
        #[arg(
            long = "max-nodes",
            default_value_t = 1000,
            allow_negative_numbers = true
        )]
        max_nodes: i64,
        /// Traversal direction: up (dependents), down (dependencies), or both
        #[arg(long, default_value = "both", value_parser = ["up", "down", "both"])]
        direction: String,
        #[arg(long)]
        json: bool,
    },
    /// Project SQLite facts into embedded Kuzu
    #[command(name = "kuzu-sync")]
    KuzuSync {
        #[arg(long)]
        json: bool,
    },
    /// Embed queued nodes with the local embeddings service
    Embed {
        #[arg(long, default_value_t = 32, allow_negative_numbers = true)]
        limit: i64,
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show or download the local rerank model (the only command that uses the network)
    #[command(name = "rerank-model")]
    RerankModel {
        /// Download the model into the Hugging Face cache if it is missing
        #[arg(long)]
        download: bool,
        #[arg(long)]
        json: bool,
    },
    /// Eval harness: read JSONL context requests on stdin, write one JSON pack per line
    #[command(name = "eval-batch", hide = true)]
    EvalBatch,
    /// Run the local HTTP daemon
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 7979)]
        port: u16,
    },
}

impl Command {
    fn json(&self) -> bool {
        match self {
            Self::Index { json, .. }
            | Self::Status { json }
            | Self::Search { json, .. }
            | Self::Symbol { json, .. }
            | Self::Context { json, .. }
            | Self::Neighbors { json, .. }
            | Self::Impact { json, .. }
            | Self::KuzuSync { json }
            | Self::Embed { json, .. }
            | Self::RerankModel { json, .. } => *json,
            Self::Serve { .. } | Self::EvalBatch => false,
        }
    }
}

/// Parse `args` (including the program name) and run; returns the process exit code.
pub fn main_with_args(args: impl IntoIterator<Item = String>) -> i32 {
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            return err.exit_code();
        }
    };
    let as_json = cli.command.json();
    match run(cli) {
        Ok(()) => 0,
        Err(err) => {
            if as_json {
                eprintln!(
                    "{}",
                    crate::json::dumps_pretty(&json!({"ok": false, "error": err.to_string()}))
                );
            } else {
                eprintln!("error: {err}");
            }
            1
        }
    }
}

fn open(data_dir: &Path) -> Result<Connection> {
    storage::connect(&sqlite_path(data_dir))
}

fn with_ok(value: Value) -> Value {
    let mut map = match value {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    map.insert("ok".into(), true.into());
    Value::Object(map)
}

pub fn run(cli: Cli) -> Result<()> {
    let data_dir = cli.data_dir.unwrap_or_else(default_data_dir);
    let as_json = cli.command.json();
    let output = match cli.command {
        Command::Index {
            path, sync_kuzu, ..
        } => {
            let stats = crate::indexer::index_repo(&path, &sqlite_path(&data_dir))?;
            let mut output = with_ok(serde_json::to_value(stats)?);
            if sync_kuzu {
                let kuzu = crate::kuzu::sync_from_sqlite(&open(&data_dir)?, &kuzu_path(&data_dir))?;
                output["nodes"] = kuzu["nodes"].clone();
                output["edges"] = kuzu["edges"].clone();
                output["kuzu"] = kuzu;
            }
            output
        }
        Command::Status { .. } => {
            let conn = open(&data_dir)?;
            json!({"ok": true, "data_dir": data_dir.to_string_lossy(), "repos": storage::list_repos(&conn)?})
        }
        Command::Search {
            query, repo, limit, ..
        } => {
            let conn = open(&data_dir)?;
            let repo_id =
                storage::repo_filter(&conn, normalize_repo_arg(repo.as_deref()).as_deref())?;
            json!({"ok": true, "results": storage::search_nodes(&conn, &query, repo_id.as_deref(), limit)?})
        }
        Command::Symbol {
            name, repo, limit, ..
        } => {
            let conn = open(&data_dir)?;
            let repo_id =
                storage::repo_filter(&conn, normalize_repo_arg(repo.as_deref()).as_deref())?;
            json!({"ok": true, "results": storage::find_symbol(&conn, &name, repo_id.as_deref(), limit)?})
        }
        Command::Context {
            query,
            repo,
            budget,
            limit,
            from_path,
            exclude_path,
            no_related,
            rerank,
            ..
        } => {
            let conn = open(&data_dir)?;
            let repo_path = normalize_repo_arg(repo.as_deref());
            let reranker = if rerank { Some(load_reranker()?) } else { None };
            let options = ContextOptions {
                repo_path: repo_path.as_deref(),
                budget,
                limit,
                include_related: !no_related,
                from_path: from_path.as_deref(),
                exclude_paths: exclude_path.into_iter().collect(),
                reranker: reranker.as_deref(),
                rerank_query: None,
            };
            with_ok(build_context_pack(&conn, &query, &options)?)
        }
        Command::Neighbors { node_id, limit, .. } => {
            let conn = open(&data_dir)?;
            with_ok(serde_json::to_value(storage::get_neighbors(
                &conn, &node_id, limit,
            )?)?)
        }
        Command::Impact {
            target,
            repo,
            limit,
            neighbor_limit,
            depth,
            max_nodes,
            direction,
            ..
        } => {
            let conn = open(&data_dir)?;
            let repo_id =
                storage::repo_filter(&conn, normalize_repo_arg(repo.as_deref()).as_deref())?;
            let direction = Direction::parse(&direction)?;
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
            json!({"ok": true, "query": target, "results": results})
        }
        Command::KuzuSync { .. } => {
            let kuzu = crate::kuzu::sync_from_sqlite(&open(&data_dir)?, &kuzu_path(&data_dir))?;
            json!({"ok": true, "kuzu": kuzu})
        }
        Command::Embed {
            limit, url, model, ..
        } => cmd_embed(&data_dir, limit, url.as_deref(), model)?,
        Command::RerankModel { download, .. } => rerank_model_status(download)?,
        Command::EvalBatch => return eval_batch(),
        Command::Serve { host, port } => {
            return crate::daemon::serve(&host, port, &data_dir);
        }
    };
    emit(&output, as_json);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the impact flags shared by CLI and daemon"
)]
pub fn impact_results(
    conn: &Connection,
    target: &str,
    repo_id: Option<&str>,
    limit: i64,
    neighbor_limit: i64,
    depth: Option<i64>,
    max_nodes: i64,
    direction: Direction,
) -> Result<Vec<Value>> {
    impact_targets(conn, target, repo_id, limit)?
        .iter()
        .map(|node| impact_report(conn, node, neighbor_limit, depth, max_nodes, direction))
        .collect()
}

/// Relative `--repo` values are taken against the current directory.
pub fn normalize_repo_arg(repo: Option<&str>) -> Option<String> {
    let stripped = repo?.trim();
    let cwd = || resolve_path(Path::new(".")).to_string_lossy().into_owned();
    Some(if stripped == "." {
        cwd()
    } else if let Some(rest) = stripped.strip_prefix("./") {
        format!("{}/{rest}", cwd()).trim_end_matches('/').to_owned()
    } else if stripped.starts_with('/') || stripped == "~" || stripped.starts_with("~/") {
        stripped.to_owned()
    } else {
        format!("{}/{stripped}", cwd())
            .trim_end_matches('/')
            .to_owned()
    })
}

#[derive(Debug, serde::Deserialize)]
struct EvalRequest {
    data_dir: PathBuf,
    query: String,
    rerank_query: Option<String>,
    from_path: Option<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default = "default_eval_limit")]
    limit: usize,
    #[serde(default)]
    rerank: bool,
}

fn default_eval_limit() -> usize {
    10
}

/// Answers many context requests with one process, so the rerank model loads once.
/// Each output line is `{"ok", "ms", "pack"}` or `{"ok": false, "error"}`.
fn eval_batch() -> Result<()> {
    use std::io::{BufRead, Write};
    let mut reranker: Option<Box<dyn crate::context::Rerank>> = None;
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match eval_one(&line, &mut reranker) {
            Ok((pack, ms)) => json!({"ok": true, "ms": ms, "pack": pack}),
            Err(err) => json!({"ok": false, "error": err.to_string()}),
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

fn eval_one(
    line: &str,
    reranker: &mut Option<Box<dyn crate::context::Rerank>>,
) -> Result<(Value, f64)> {
    let request: EvalRequest = serde_json::from_str(line)?;
    if request.rerank && reranker.is_none() {
        let started = std::time::Instant::now();
        *reranker = Some(load_reranker()?);
        eprintln!(
            "eval-batch: rerank model loaded in {:.0} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let conn = open(&request.data_dir)?;
    let options = ContextOptions {
        limit: request.limit,
        include_related: false,
        from_path: request.from_path.as_deref(),
        exclude_paths: request.exclude.into_iter().collect(),
        reranker: reranker.as_deref().filter(|_| request.rerank),
        rerank_query: request.rerank_query.as_deref(),
        ..ContextOptions::default()
    };
    let started = std::time::Instant::now();
    let pack = build_context_pack(&conn, &request.query, &options)?;
    Ok((pack, started.elapsed().as_secs_f64() * 1000.0))
}

#[cfg(feature = "embed")]
fn load_reranker() -> Result<Box<dyn crate::context::Rerank>> {
    Ok(Box::new(crate::rerank::Reranker::load_cached()?))
}

#[cfg(not(feature = "embed"))]
fn load_reranker() -> Result<Box<dyn crate::context::Rerank>> {
    Err(Error::runtime(RERANK_NOT_BUILT))
}

#[cfg(not(feature = "embed"))]
const RERANK_NOT_BUILT: &str = "This lode binary was built without the embed feature; rebuild with `--features embed` to rerank.";

#[cfg(feature = "embed")]
fn rerank_model_status(download: bool) -> Result<Value> {
    use crate::rerank::{MODEL_ID, cached_model_files, download_model};
    let files = match cached_model_files() {
        Some(files) => Some(files),
        None if download => Some(download_model()?),
        None => None,
    };
    Ok(json!({
        "ok": true,
        "model": MODEL_ID,
        "cached": files.is_some(),
        "files": files.unwrap_or_default().iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>(),
    }))
}

#[cfg(not(feature = "embed"))]
fn rerank_model_status(_download: bool) -> Result<Value> {
    Err(Error::runtime(RERANK_NOT_BUILT))
}

fn cmd_embed(
    data_dir: &Path,
    limit: i64,
    url: Option<&str>,
    model: Option<String>,
) -> Result<Value> {
    let mut conn = open(data_dir)?;
    let nodes = storage::pending_embedding_nodes(&conn, limit)?;
    let mut embedded = 0;
    if !nodes.is_empty() {
        let texts: Vec<String> = nodes.iter().map(embedding_text).collect();
        let vectors = crate::embeddings::embed_texts(&texts, url, 2)?;
        if vectors.len() != nodes.len() {
            return Err(Error::runtime(format!(
                "Embedding endpoint returned {} vectors for {} texts",
                vectors.len(),
                nodes.len()
            )));
        }
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or_else(crate::embeddings::embeddings_model);
        let rows = nodes
            .iter()
            .zip(vectors)
            .map(|(node, vector)| match vector {
                Value::Array(vector) => Ok(EmbeddingRow {
                    node_id: node.id.clone(),
                    repo_id: node.repo_id.clone(),
                    vector,
                    model: model.clone(),
                }),
                _ => Err(Error::runtime("Unexpected embeddings response shape")),
            })
            .collect::<Result<Vec<_>>>()?;
        storage::upsert_embeddings(&mut conn, &rows, None)?;
        embedded = rows.len();
    }
    let counts = storage::embedding_counts(&conn)?;
    Ok(json!({
        "ok": true,
        "queued": counts.queued,
        "total_embeddings": counts.embedded,
        "embedded": embedded,
    }))
}

pub fn embedding_text(node: &NodeRow) -> String {
    let head = if node.qname.is_empty() {
        &node.name
    } else {
        &node.qname
    };
    join_bounded_embedding_parts(&[head, &node.signature, &node.doc])
}

pub fn join_bounded_embedding_parts(parts: &[&str]) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut used = 0;
    for part in parts.iter().filter(|part| !part.is_empty()) {
        let separator = usize::from(!out.is_empty());
        let Some(available) = EMBEDDING_TEXT_MAX_INPUT_CHARS
            .checked_sub(used + separator)
            .filter(|a| *a > 0)
        else {
            break;
        };
        let segment = take_chars(part, available);
        used += separator + char_len(segment);
        out.push(segment);
    }
    out.join("\n")
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

/// Python `str()` of a JSON scalar, with `default` for missing values.
fn show(value: &Value, default: &str) -> String {
    match value {
        Value::Null => default.to_owned(),
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

fn emit(data: &Value, as_json: bool) {
    if as_json || !std::io::stdout().is_terminal() {
        println!("{}", crate::json::dumps_pretty(data));
        return;
    }
    let has_query = !field(data, "query").is_null();
    if let (Some(rows), false) = (data["results"].as_array(), has_query) {
        for row in rows {
            println!(
                "{:12} {} {}:{}",
                show(field(row, "kind"), ""),
                show(field(row, "qname"), ""),
                show(field(row, "path"), ""),
                show(field(row, "start_line"), "")
            );
        }
    } else if let Some(repos) = data["repos"].as_array() {
        for repo in repos {
            println!(
                "{:24} files={} nodes={} root={}",
                show(field(repo, "name"), ""),
                show(field(repo, "files"), ""),
                show(field(repo, "nodes"), ""),
                show(field(repo, "root"), "")
            );
        }
    } else if let Some(results) = data["results"].as_array() {
        for result in results {
            print_impact(result, &show(field(data, "query"), ""));
        }
    } else {
        println!("{}", crate::json::dumps_pretty(data));
    }
}

fn print_impact(result: &Value, query: &str) {
    let target = field(result, "target");
    let summary = field(result, "summary");
    let qname = [field(target, "qname"), field(target, "name")]
        .into_iter()
        .map(|v| show(v, ""))
        .find(|v| !v.is_empty())
        .unwrap_or_else(|| query.to_owned());
    let count = |key: &str| show(field(summary, key), "0");
    println!("  {} {qname}", show(field(target, "kind"), "?"));
    println!(
        "    callers={}  callees={}  files={}",
        count("callers"),
        count("callees"),
        count("files")
    );
    println!(
        "    upstream={}  downstream={}  entrypoints={}  depth={}",
        count("upstream"),
        count("downstream"),
        count("entrypoints"),
        show(field(summary, "depth"), "None")
    );
    let radius = field(result, "blast_radius");
    let entries = |key: &str| radius[key].as_array().cloned().unwrap_or_default();
    for (key, tag) in [("upstream", "up"), ("downstream", "dn")] {
        for entry in entries(key) {
            let node = field(&entry, "node");
            println!(
                "    [{tag} d={} {} {}] {} {}",
                show(field(&entry, "distance"), "?"),
                show(field(&entry, "via"), "?"),
                show(field(&entry, "confidence"), "?"),
                show(field(node, "kind"), "?"),
                show(field(node, "qname"), "?")
            );
        }
    }
    for entry in entries("files") {
        println!(
            "    [file d={}] {}",
            show(field(&entry, "distance"), "?"),
            show(field(&entry, "path"), "?")
        );
    }
    for entry in entries("entrypoints") {
        let node = field(&entry, "node");
        println!(
            "    [entrypoint] {} {}",
            show(field(node, "kind"), "?"),
            show(field(node, "qname"), "?")
        );
    }
    if field(summary, "truncated").as_bool() == Some(true) {
        println!("    (truncated - increase --max-nodes for more)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_parts_are_bounded() {
        let long = "x".repeat(500);
        let text = join_bounded_embedding_parts(&["a.b", "", &long]);
        assert_eq!(char_len(&text), EMBEDDING_TEXT_MAX_INPUT_CHARS);
        assert!(text.starts_with("a.b\nxxx"));
    }

    #[test]
    fn repo_args_resolve_against_cwd() {
        assert_eq!(
            normalize_repo_arg(Some("/abs/x")).as_deref(),
            Some("/abs/x")
        );
        assert_eq!(normalize_repo_arg(Some("~/x")).as_deref(), Some("~/x"));
        assert!(normalize_repo_arg(Some("sub/")).unwrap().ends_with("/sub"));
        assert_eq!(normalize_repo_arg(None), None);
    }
}
