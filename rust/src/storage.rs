//! SQLite storage: same schema, FTS5 table and queries as the Python `lode.storage` module.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension, Row, params, params_from_iter};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::{expand_user, repo_id_for_root, resolve_path};
use crate::model::{FileIndex, Node, dedupe_edges, dedupe_nodes, is_external};
use crate::observability::unix_time;
use crate::text::{WORD_RE, split_word_parts, static_regex};
use crate::{Error, Result};

static QUERY_TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"[A-Za-z0-9_./:-]+"));
static REPO_PATH_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^[A-Za-z0-9_./:\-~ ]+$"));

const FTS_RESERVED: [&str; 4] = ["and", "or", "not", "near"];
const BROAD_STOPWORDS: [&str; 31] = [
    "self", "cls", "none", "true", "false", "return", "def", "class", "import", "from", "as",
    "with", "for", "while", "try", "except", "finally", "pass", "if", "else", "elif", "in", "is",
    "and", "or", "not", "the", "to", "of", "a", "an",
];
pub const STRUCTURED_FTS_TOKEN_LIMIT: usize = 8;
pub const BROAD_FTS_TOKEN_LIMIT: usize = 48;
/// Column weights for `node_fts(node_id UNINDEXED, qname, name, signature, doc, path)`; the
/// first weight belongs to the unindexed `node_id` column and must stay 0.0.
const BM25_EXPR: &str = "bm25(node_fts, 0.0, 8.0, 12.0, 4.0, 1.5, 6.0)";
const EMBEDDABLE_KINDS: [&str; 5] = ["Function", "Method", "Class", "Route", "DocSection"];

/// Node columns in table order; every node query selects these, then a rank.
pub const NODE_COLUMNS: &str = "n.id, n.repo_id, n.owner_path, n.kind, n.name, n.qname, n.path, \
     n.start_line, n.end_line, n.signature, n.doc, n.confidence, n.content_hash, n.extra_json";
const NODE_COLUMN_COUNT: usize = 14;

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS repos (
    id TEXT PRIMARY KEY,
    root TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    repo_id TEXT NOT NULL,
    path TEXT NOT NULL,
    abspath TEXT NOT NULL,
    language TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    generated INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL,
    PRIMARY KEY (repo_id, path),
    FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    owner_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qname TEXT NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT NOT NULL DEFAULT '',
    doc TEXT NOT NULL DEFAULT '',
    confidence TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_repo_kind ON nodes(repo_id, kind);
CREATE INDEX IF NOT EXISTS idx_nodes_repo_name ON nodes(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_nodes_repo_path ON nodes(repo_id, path);
CREATE INDEX IF NOT EXISTS idx_nodes_lower_name ON nodes(lower(name));
CREATE INDEX IF NOT EXISTS idx_nodes_lower_qname ON nodes(lower(qname));

CREATE TABLE IF NOT EXISTS edges (
    repo_id TEXT NOT NULL,
    owner_path TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo_id, owner_path, src, dst, kind, detail),
    FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(repo_id, src, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(repo_id, dst, kind);

CREATE TABLE IF NOT EXISTS embedding_queue (
    node_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    queued_at REAL NOT NULL,
    embedded_at REAL
);

CREATE TABLE IF NOT EXISTS embeddings (
    node_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    embedded_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_queue_pending
ON embedding_queue(repo_id, embedded_at, queued_at);

CREATE INDEX IF NOT EXISTS idx_embeddings_repo
ON embeddings(repo_id);

CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(
node_id UNINDEXED, qname, name, signature, doc, path, tokenize='porter');
"#;

/// A node as the query layer returns it: the stored columns plus the search rank, with
/// `extra_json` decoded and line numbers clamped to at least 1.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NodeRow {
    pub id: String,
    pub repo_id: String,
    pub owner_path: String,
    pub kind: String,
    pub name: String,
    pub qname: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub signature: String,
    pub doc: String,
    pub confidence: String,
    pub content_hash: String,
    pub rank: f64,
    pub extra: Map<String, Value>,
}

impl NodeRow {
    /// Reads `NODE_COLUMNS` starting at column `offset`, then the rank column right after.
    pub fn from_row(row: &Row<'_>, offset: usize) -> rusqlite::Result<Self> {
        let text = |index: usize| -> rusqlite::Result<String> {
            Ok(row
                .get::<_, Option<String>>(offset + index)?
                .unwrap_or_default())
        };
        let start_line = row.get::<_, Option<i64>>(offset + 7)?.unwrap_or(0).max(1);
        let end_line = row
            .get::<_, Option<i64>>(offset + 8)?
            .unwrap_or(0)
            .max(start_line);
        let rank = row
            .get::<_, Option<f64>>(offset + NODE_COLUMN_COUNT)?
            .unwrap_or(0.0);
        Ok(Self {
            id: text(0)?,
            repo_id: text(1)?,
            owner_path: text(2)?,
            kind: text(3)?,
            name: text(4)?,
            qname: text(5)?,
            path: text(6)?,
            start_line,
            end_line,
            signature: text(9)?,
            doc: text(10)?,
            confidence: text(11)?,
            content_hash: text(12)?,
            rank,
            extra: decode_extra(&text(13)?),
        })
    }
}

pub fn decode_extra(raw: &str) -> Map<String, Value> {
    if raw.is_empty() || raw == "{}" {
        return Map::new();
    }
    match serde_json::from_str(raw) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

pub fn connect(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    init_db(&conn)?;
    Ok(conn)
}

pub fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA_SQL)?;
    // Databases created before the mtime column existed; a duplicate column error is expected.
    if let Err(err) = conn.execute(
        "ALTER TABLE files ADD COLUMN mtime REAL NOT NULL DEFAULT 0",
        [],
    ) {
        if !err.to_string().contains("duplicate column") {
            return Err(err.into());
        }
    }
    Ok(())
}

pub fn upsert_repo(conn: &Connection, root: &Path) -> Result<String> {
    let resolved = resolve_path(root);
    let repo_id = repo_id_for_root(&resolved);
    let name = resolved
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    conn.execute(
        "INSERT INTO repos(id, root, name, indexed_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET root=excluded.root, name=excluded.name, indexed_at=excluded.indexed_at",
        params![repo_id, resolved.to_string_lossy(), name, unix_time()],
    )?;
    Ok(repo_id)
}

fn text(value: &str) -> SqlValue {
    SqlValue::Text(value.to_owned())
}

fn placeholders(count: usize) -> String {
    vec!["?"; count].join(",")
}

fn node_ids_for_paths(conn: &Connection, repo_id: &str, paths: &[String]) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    for group in paths.chunks(400) {
        let sql = format!(
            "SELECT id FROM nodes WHERE repo_id = ? AND owner_path IN ({})",
            placeholders(group.len())
        );
        let mut stmt = conn.prepare(&sql)?;
        let args = std::iter::once(repo_id).chain(group.iter().map(String::as_str));
        let rows = stmt.query_map(params_from_iter(args), |row| row.get::<_, String>(0))?;
        for id in rows {
            ids.push(id?);
        }
    }
    Ok(ids)
}

fn delete_fts_rows(conn: &Connection, node_ids: &[String]) -> Result<()> {
    let mut stmt = conn.prepare_cached("DELETE FROM node_fts WHERE node_id = ?")?;
    for id in node_ids {
        stmt.execute([id])?;
    }
    Ok(())
}

/// Replace the stored nodes, edges, FTS rows and embedding queue entries of the given files.
/// Callers wrap this in a transaction.
pub fn replace_file_indexes(
    conn: &Connection,
    repo_id: &str,
    file_indexes: Vec<FileIndex>,
) -> Result<()> {
    if file_indexes.is_empty() {
        return Ok(());
    }
    let now = unix_time();
    let paths: Vec<String> = file_indexes.iter().map(|f| f.path.clone()).collect();
    let old_node_ids = node_ids_for_paths(conn, repo_id, &paths)?;
    delete_fts_rows(conn, &old_node_ids)?;
    let files: Vec<FileIndex> = file_indexes
        .into_iter()
        .map(|mut f| {
            f.nodes = dedupe_nodes(std::mem::take(&mut f.nodes));
            f.edges = dedupe_edges(std::mem::take(&mut f.edges));
            f
        })
        .collect();
    let new_ids: HashSet<&str> = files
        .iter()
        .flat_map(|f| f.nodes.iter().map(|n| n.id.as_str()))
        .collect();
    let stale: Vec<String> = old_node_ids
        .iter()
        .filter(|id| !new_ids.contains(id.as_str()))
        .cloned()
        .collect();
    delete_embedding_artifacts(conn, &stale)?;
    for group in paths.chunks(400) {
        let marks = placeholders(group.len());
        let args: Vec<&str> = std::iter::once(repo_id)
            .chain(group.iter().map(String::as_str))
            .collect();
        let delete_edges =
            format!("DELETE FROM edges WHERE repo_id = ? AND owner_path IN ({marks})");
        let delete_nodes =
            format!("DELETE FROM nodes WHERE repo_id = ? AND owner_path IN ({marks})");
        conn.execute(&delete_edges, params_from_iter(&args))?;
        conn.execute(&delete_nodes, params_from_iter(&args))?;
    }
    let mut file_rows = Vec::new();
    let mut node_rows = Vec::new();
    let mut fts_rows = Vec::new();
    let mut queue_rows = Vec::new();
    let mut edge_rows = Vec::new();
    for file in &files {
        file_rows.push(vec![
            text(repo_id),
            file.path.clone().into(),
            file.abspath.clone().into(),
            file.language.clone().into(),
            SqlValue::Integer(i64::try_from(file.size).unwrap_or(i64::MAX)),
            file.mtime.into(),
            file.content_hash.clone().into(),
            SqlValue::Integer(i64::from(file.generated)),
            now.into(),
        ]);
        for node in &file.nodes {
            let (node_row, fts_row) = node_storage_rows(repo_id, &file.path, node);
            node_rows.push(node_row);
            fts_rows.extend(fts_row);
            if EMBEDDABLE_KINDS.contains(&node.kind.as_str()) && !node.content_hash.is_empty() {
                queue_rows.push(vec![
                    node.id.clone().into(),
                    text(repo_id),
                    node.content_hash.clone().into(),
                    now.into(),
                ]);
            }
        }
        for edge in &file.edges {
            edge_rows.push(vec![
                text(repo_id),
                file.path.clone().into(),
                edge.src.clone().into(),
                edge.dst.clone().into(),
                edge.kind.clone().into(),
                edge.confidence.clone().into(),
                edge.detail.clone().into(),
            ]);
        }
    }
    execute_values(
        conn,
        "INSERT INTO files(repo_id, path, abspath, language, size, mtime, content_hash, generated, indexed_at)",
        &file_rows,
        "ON CONFLICT(repo_id, path) DO UPDATE SET abspath=excluded.abspath, language=excluded.language, \
         size=excluded.size, mtime=excluded.mtime, content_hash=excluded.content_hash, \
         generated=excluded.generated, indexed_at=excluded.indexed_at",
    )?;
    execute_values(
        conn,
        "INSERT INTO nodes(id, repo_id, owner_path, kind, name, qname, path, start_line, end_line, \
         signature, doc, confidence, content_hash, extra_json)",
        &node_rows,
        "",
    )?;
    execute_values(
        conn,
        "INSERT INTO node_fts(node_id, qname, name, signature, doc, path)",
        &fts_rows,
        "",
    )?;
    execute_values(
        conn,
        "INSERT INTO embedding_queue(node_id, repo_id, content_hash, queued_at)",
        &queue_rows,
        "ON CONFLICT(node_id) DO UPDATE SET content_hash=excluded.content_hash, \
         queued_at=CASE WHEN embedding_queue.content_hash != excluded.content_hash THEN excluded.queued_at \
         ELSE embedding_queue.queued_at END, \
         embedded_at=CASE WHEN embedding_queue.content_hash != excluded.content_hash THEN NULL \
         ELSE embedding_queue.embedded_at END",
    )?;
    execute_values(
        conn,
        "INSERT OR IGNORE INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)",
        &edge_rows,
        "",
    )?;
    Ok(())
}

type SqlRow = Vec<SqlValue>;

fn node_storage_rows(repo_id: &str, owner_path: &str, node: &Node) -> (SqlRow, Option<SqlRow>) {
    let fts_body = node
        .extra
        .get("fts_body")
        .map(json_text)
        .unwrap_or_default();
    let extra_json = if node.extra.contains_key("fts_body") {
        let mut payload = node.extra.clone();
        payload.remove("fts_body");
        crate::json::dumps(&payload)
    } else {
        crate::json::dumps(&node.extra)
    };
    let node_row = vec![
        node.id.clone().into(),
        text(repo_id),
        text(owner_path),
        node.kind.clone().into(),
        node.name.clone().into(),
        node.qname.clone().into(),
        node.path.clone().into(),
        node.start_line.into(),
        node.end_line.into(),
        node.signature.clone().into(),
        node.doc.clone().into(),
        node.confidence.clone().into(),
        node.content_hash.clone().into(),
        extra_json.into(),
    ];
    if is_external(&node.kind) {
        return (node_row, None);
    }
    let fts_doc = if fts_body.is_empty() {
        node.doc.clone()
    } else {
        format!("{} {fts_body}", node.doc).trim().to_owned()
    };
    let fts_row = vec![
        node.id.clone().into(),
        expand_fts_field(&node.qname).into(),
        expand_fts_field(&node.name).into(),
        expand_fts_field(&node.signature).into(),
        fts_doc.into(),
        expand_fts_field(&node.path).into(),
    ];
    (node_row, Some(fts_row))
}

/// `str(value or "")` for a JSON value that is normally a string.
fn json_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null | Value::Bool(false) => String::new(),
        other => other.to_string(),
    }
}

/// Multi-row `INSERT ... VALUES (...), (...)` in groups that stay under SQLite's variable limit.
fn execute_values(
    conn: &Connection,
    insert_sql: &str,
    rows: &[SqlRow],
    suffix_sql: &str,
) -> Result<()> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    let width = first.len().max(1);
    let row_marks = format!("({})", vec!["?"; width].join(", "));
    let max_rows = (12_000 / width).clamp(1, 400);
    for group in rows.chunks(max_rows) {
        let sql = format!(
            "{insert_sql} VALUES {} {suffix_sql}",
            vec![row_marks.as_str(); group.len()].join(", ")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        stmt.execute(params_from_iter(group.iter().flatten()))?;
    }
    Ok(())
}

pub fn delete_embedding_artifacts(conn: &Connection, node_ids: &[String]) -> Result<()> {
    for group in node_ids.chunks(400) {
        let marks = placeholders(group.len());
        let delete_queue = format!("DELETE FROM embedding_queue WHERE node_id IN ({marks})");
        let delete_vectors = format!("DELETE FROM embeddings WHERE node_id IN ({marks})");
        conn.execute(&delete_queue, params_from_iter(group))?;
        conn.execute(&delete_vectors, params_from_iter(group))?;
    }
    Ok(())
}

/// Delete every stored file of the repo whose path is not in `live_paths`; returns the count.
pub fn remove_missing_files(
    conn: &Connection,
    repo_id: &str,
    live_paths: &HashSet<String>,
) -> Result<usize> {
    let stored: Vec<String> = conn
        .prepare("SELECT path FROM files WHERE repo_id = ?")?
        .query_map([repo_id], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut removed = 0;
    for path in stored.into_iter().filter(|path| !live_paths.contains(path)) {
        let old_ids = node_ids_for_paths(conn, repo_id, std::slice::from_ref(&path))?;
        if !old_ids.is_empty() {
            delete_fts_rows(conn, &old_ids)?;
            delete_embedding_artifacts(conn, &old_ids)?;
        }
        conn.execute(
            "DELETE FROM edges WHERE repo_id = ? AND owner_path = ?",
            params![repo_id, path],
        )?;
        conn.execute(
            "DELETE FROM nodes WHERE repo_id = ? AND owner_path = ?",
            params![repo_id, path],
        )?;
        conn.execute(
            "DELETE FROM files WHERE repo_id = ? AND path = ?",
            params![repo_id, path],
        )?;
        removed += 1;
    }
    Ok(removed)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RepoSummary {
    pub id: String,
    pub root: String,
    pub name: String,
    pub indexed_at: f64,
    pub files: i64,
    pub nodes: i64,
}

pub fn list_repos(conn: &Connection) -> Result<Vec<RepoSummary>> {
    let mut stmt = conn.prepare(
        "SELECT r.id, r.root, r.name, r.indexed_at, COUNT(DISTINCT f.path) AS files, COUNT(DISTINCT n.id) AS nodes
         FROM repos r
         LEFT JOIN files f ON f.repo_id = r.id
         LEFT JOIN nodes n ON n.repo_id = r.id
         GROUP BY r.id
         ORDER BY r.indexed_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(RepoSummary {
            id: row.get(0)?,
            root: row.get(1)?,
            name: row.get(2)?,
            indexed_at: row.get(3)?,
            files: row.get(4)?,
            nodes: row.get(5)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Repo id for an absolute (or `~`) repository root, `None` for unknown or unsafe input.
pub fn repo_filter(conn: &Connection, repo_path: Option<&str>) -> Result<Option<String>> {
    let Some(root) = repo_path
        .filter(|p| !p.is_empty())
        .and_then(normalize_repo_path_input)
    else {
        return Ok(None);
    };
    Ok(conn
        .query_row("SELECT id FROM repos WHERE root = ?", [root], |row| {
            row.get(0)
        })
        .optional()?)
}

fn normalize_repo_path_input(repo_path: &str) -> Option<String> {
    let stripped = repo_path.trim();
    if !is_safe_repo_path_input(stripped) {
        return None;
    }
    let expanded = if stripped == "~" || stripped.starts_with("~/") {
        expand_user(stripped).to_string_lossy().into_owned()
    } else {
        stripped.to_owned()
    };
    if expanded.chars().count() > 1 {
        Some(expanded.trim_end_matches('/').to_owned())
    } else {
        Some(expanded)
    }
}

fn is_safe_repo_path_input(path: &str) -> bool {
    if path.is_empty()
        || path.chars().count() > 4096
        || path.contains('\0')
        || !REPO_PATH_RE.is_match(path)
    {
        return false;
    }
    if !(path.starts_with('/') || path == "~" || path.starts_with("~/")) {
        return false;
    }
    !path.split('/').any(|part| part == "..")
}

/// The original value plus its lowercased identifier parts not already present.
pub fn expand_fts_field(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    let mut extras: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw in WORD_RE.find_iter(value) {
        seen.insert(raw.as_str().to_lowercase());
        for part in split_word_parts(raw.as_str()) {
            let low = part.to_lowercase();
            if low.chars().count() > 1 && seen.insert(low.clone()) {
                extras.push(low);
            }
        }
    }
    if extras.is_empty() {
        value.to_owned()
    } else {
        format!("{value} {}", extras.join(" "))
    }
}

fn expand_query_tokens(tokens: &[String], limit: usize) -> Vec<String> {
    let mut expanded = Vec::new();
    let mut seen = HashSet::new();
    for raw in tokens {
        for word in WORD_RE.find_iter(raw) {
            let word = word.as_str();
            for candidate in std::iter::once(word).chain(split_word_parts(word)) {
                let low = candidate.to_lowercase();
                if low.chars().count() <= 1
                    || FTS_RESERVED.contains(&low.as_str())
                    || !seen.insert(low)
                {
                    continue;
                }
                expanded.push(candidate.to_owned());
                if expanded.len() >= limit {
                    return expanded;
                }
            }
        }
    }
    expanded
}

fn quote_fts_token(token: &str) -> String {
    let cleaned = token.replace('"', "");
    if cleaned.is_empty() {
        return cleaned;
    }
    if FTS_RESERVED.contains(&cleaned.to_lowercase().as_str())
        || cleaned.contains(['.', '/', ':', '-'])
    {
        format!("\"{cleaned}\"")
    } else {
        cleaned
    }
}

pub fn query_tokens(query: &str) -> Vec<String> {
    QUERY_TOKEN_RE
        .find_iter(query)
        .map(|m| m.as_str().to_owned())
        .collect()
}

/// How FTS terms are combined: `Or` is the broad query, `All` (implicit AND) the strict one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtsOperator {
    Or,
    All,
}

pub fn fts_query(query: &str, operator: FtsOperator) -> String {
    fts_query_from_tokens(&query_tokens(query), operator)
}

pub fn fts_query_from_tokens(tokens: &[String], operator: FtsOperator) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let chosen: Vec<String> = match operator {
        FtsOperator::Or => {
            let expanded = expand_query_tokens(tokens, BROAD_FTS_TOKEN_LIMIT);
            let kept: Vec<String> = expanded
                .iter()
                .filter(|t| !BROAD_STOPWORDS.contains(&t.to_lowercase().as_str()))
                .cloned()
                .collect();
            if kept.is_empty() { expanded } else { kept }
        }
        FtsOperator::All => tokens
            .iter()
            .take(12)
            .filter(|t| !FTS_RESERVED.contains(&t.to_lowercase().as_str()))
            .cloned()
            .collect(),
    };
    let quoted: Vec<String> = chosen
        .iter()
        .map(|t| quote_fts_token(t))
        .filter(|q| !q.is_empty())
        .collect();
    let separator = match operator {
        FtsOperator::Or => " OR ",
        FtsOperator::All => " ",
    };
    quoted.join(separator)
}

pub fn external_like_terms_from_tokens(tokens: &[String]) -> Vec<String> {
    if tokens.len() > STRUCTURED_FTS_TOKEN_LIMIT {
        return Vec::new();
    }
    tokens.iter().take(12).map(|t| t.to_lowercase()).collect()
}

fn query_nodes(conn: &Connection, sql: &str, args: &[SqlValue]) -> Result<Vec<NodeRow>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let rows = stmt.query_map(params_from_iter(args), |row| NodeRow::from_row(row, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn external_node_like_rows(
    conn: &Connection,
    query: &str,
    repo_id: Option<&str>,
    limit: i64,
    tokens: &[String],
) -> Result<Vec<NodeRow>> {
    let terms = external_like_terms_from_tokens(tokens);
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let filter_sql = vec![
        "(lower(n.name) LIKE ? OR lower(n.qname) LIKE ? OR lower(n.path) LIKE ?)";
        terms.len()
    ]
    .join(" OR ");
    let mut args: Vec<SqlValue> = repo_id.map(text).into_iter().collect();
    for term in &terms {
        let like = format!("%{term}%");
        args.extend([like.clone().into(), like.clone().into(), like.into()]);
    }
    args.push(query.to_lowercase().into());
    args.push(limit.into());
    let repo_clause = if repo_id.is_some() {
        "n.repo_id = ? AND"
    } else {
        ""
    };
    let sql = format!(
        "SELECT {NODE_COLUMNS}, 0.0 AS rank FROM nodes n
         WHERE {repo_clause} n.kind IN ('ExternalSymbol', 'ExternalDependency') AND ({filter_sql})
         ORDER BY CASE WHEN lower(n.name) = ? THEN 0 ELSE 1 END, length(n.qname), n.path, n.start_line
         LIMIT ?"
    );
    query_nodes(conn, &sql, &args)
}

/// Appends then checks the limit, so a limit of 0 still admits one node (as in Python).
fn append_unique_nodes(
    rows: Vec<NodeRow>,
    results: &mut Vec<NodeRow>,
    seen: &mut HashSet<String>,
    limit: usize,
) {
    for node in rows {
        if !seen.insert(node.id.clone()) {
            continue;
        }
        results.push(node);
        if results.len() >= limit {
            break;
        }
    }
}

fn search_path_priority(path: &str) -> u8 {
    if path.starts_with("src/") {
        0
    } else if path.starts_with("scripts/") || path.starts_with("benchmarks/") {
        1
    } else if path.starts_with("tests/") {
        2
    } else if path == "README.md"
        || path == "AGENTS.md"
        || path.starts_with("docs/")
        || path.starts_with("skills/")
    {
        3
    } else {
        1
    }
}

fn search_kind_priority(kind: &str) -> u8 {
    match kind {
        "File" => 0,
        "Class" => 1,
        "Function" | "Method" => 2,
        "Route" => 3,
        "DocSection" => 4,
        _ => 5,
    }
}

fn search_result_cmp(left: &NodeRow, right: &NodeRow) -> std::cmp::Ordering {
    search_path_priority(&left.path)
        .cmp(&search_path_priority(&right.path))
        .then(left.rank.total_cmp(&right.rank))
        .then(search_kind_priority(&left.kind).cmp(&search_kind_priority(&right.kind)))
        .then(left.path.cmp(&right.path))
        .then(left.start_line.cmp(&right.start_line))
        .then(left.qname.cmp(&right.qname))
}

fn unique_nodes_in_order(candidates: &[NodeRow], limit: usize) -> Vec<NodeRow> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    append_unique_nodes(candidates.to_vec(), &mut results, &mut seen, limit);
    results
}

fn unique_node_count(candidates: &[NodeRow], limit: usize) -> usize {
    let mut seen = HashSet::new();
    let mut count = 0;
    for node in candidates {
        if seen.insert(node.id.as_str()) {
            count += 1;
            if count >= limit {
                break;
            }
        }
    }
    count
}

/// Keeps BM25 order but promotes the best source-tree hit to the top when a lower-priority
/// path (tests, docs) ranked first.
fn stable_search_results(candidates: &[NodeRow], limit: usize) -> Vec<NodeRow> {
    let results = unique_nodes_in_order(candidates, limit);
    let Some(first) = results.first() else {
        return results;
    };
    let top_priority = search_path_priority(&first.path);
    let best_priority = candidates
        .iter()
        .map(|n| search_path_priority(&n.path))
        .min()
        .unwrap_or(top_priority);
    if best_priority >= top_priority {
        return results;
    }
    let mut promoted: Option<&NodeRow> = None;
    for node in candidates
        .iter()
        .filter(|n| search_path_priority(&n.path) == best_priority)
    {
        if promoted.is_none_or(|best| search_result_cmp(node, best).is_lt()) {
            promoted = Some(node);
        }
    }
    let Some(promoted) = promoted.cloned() else {
        return results;
    };
    let remaining: Vec<NodeRow> = candidates
        .iter()
        .filter(|n| n.id != promoted.id)
        .cloned()
        .collect();
    let mut out = vec![promoted];
    out.extend(unique_nodes_in_order(&remaining, limit.saturating_sub(1)));
    out
}

fn run_fts_bm25(
    conn: &Connection,
    fts_expr: &str,
    repo_id: Option<&str>,
    limit: i64,
    rank_bias: f64,
) -> Vec<NodeRow> {
    if fts_expr.is_empty() {
        return Vec::new();
    }
    let repo_clause = if repo_id.is_some() {
        "AND n.repo_id = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT {NODE_COLUMNS}, ({BM25_EXPR} + ?) AS rank
         FROM node_fts JOIN nodes n ON n.id = node_fts.node_id
         WHERE node_fts MATCH ? {repo_clause}
           AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
         ORDER BY rank LIMIT ?"
    );
    let mut args: Vec<SqlValue> = vec![rank_bias.into(), text(fts_expr)];
    args.extend(repo_id.map(text));
    args.push(limit.into());
    // A malformed MATCH expression is an empty result, not an error.
    query_nodes(conn, &sql, &args).unwrap_or_default()
}

fn like_search_nodes(
    conn: &Connection,
    query: &str,
    repo_id: Option<&str>,
    limit: i64,
) -> Result<Vec<NodeRow>> {
    let like = format!("%{query}%");
    let repo_clause = if repo_id.is_some() {
        "n.repo_id = ? AND"
    } else {
        ""
    };
    let sql = format!(
        "SELECT {NODE_COLUMNS}, 0.0 AS rank FROM nodes n
         WHERE {repo_clause} (n.qname LIKE ? OR n.name LIKE ? OR n.path LIKE ? OR n.doc LIKE ?)
         ORDER BY CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
                  CASE WHEN n.name = ? THEN 0 ELSE 1 END, length(n.qname)
         LIMIT ?"
    );
    let mut args: Vec<SqlValue> = repo_id.map(text).into_iter().collect();
    args.extend(std::iter::repeat_n(SqlValue::from(like), 4));
    args.push(text(query));
    args.push(limit.into());
    query_nodes(conn, &sql, &args)
}

/// Full-text search over nodes: a strict (all terms) BM25 pass, a broad (any term) pass,
/// external-symbol LIKE fill, then a plain LIKE fallback when FTS finds nothing.
pub fn search_nodes(
    conn: &Connection,
    query: &str,
    repo_id: Option<&str>,
    limit: i64,
) -> Result<Vec<NodeRow>> {
    let tokens = query_tokens(query);
    let strict = if tokens.len() <= STRUCTURED_FTS_TOKEN_LIMIT {
        fts_query_from_tokens(&tokens, FtsOperator::All)
    } else {
        String::new()
    };
    let broad = fts_query_from_tokens(&tokens, FtsOperator::Or);
    let ulimit = usize::try_from(limit).unwrap_or(0);
    if !strict.is_empty() || !broad.is_empty() {
        let mut candidates = Vec::new();
        if !strict.is_empty() {
            candidates.extend(run_fts_bm25(conn, &strict, repo_id, limit, -25.0));
        }
        if !broad.is_empty() && broad != strict && unique_node_count(&candidates, ulimit) < ulimit {
            candidates.extend(run_fts_bm25(conn, &broad, repo_id, limit, 0.0));
        }
        let mut results = stable_search_results(&candidates, ulimit);
        let mut seen: HashSet<String> = results.iter().map(|n| n.id.clone()).collect();
        if results.len() < ulimit && !broad.is_empty() {
            let fill = i64::try_from(ulimit - results.len()).unwrap_or(0);
            let external = external_node_like_rows(conn, query, repo_id, fill, &tokens)?;
            append_unique_nodes(external, &mut results, &mut seen, ulimit);
        }
        if !results.is_empty() {
            return Ok(results);
        }
    }
    like_search_nodes(conn, query, repo_id, limit)
}

/// Exact (case-insensitive) name or qname matches, else qname substring matches.
pub fn find_symbol(
    conn: &Connection,
    name: &str,
    repo_id: Option<&str>,
    limit: i64,
) -> Result<Vec<NodeRow>> {
    let lowered = name.to_lowercase();
    let repo_clause = if repo_id.is_some() {
        "n.repo_id = ? AND"
    } else {
        ""
    };
    let repo_arg: Vec<SqlValue> = repo_id.map(text).into_iter().collect();
    let exact_sql = format!(
        "SELECT {NODE_COLUMNS}, 0.0 AS rank FROM nodes n
         WHERE {repo_clause} (lower(n.name) = ? OR lower(n.qname) = ?)
         ORDER BY CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
                  CASE WHEN lower(n.name) = ? THEN 0 ELSE 1 END, length(n.qname)
         LIMIT ?"
    );
    let mut args = repo_arg.clone();
    args.extend([
        lowered.clone().into(),
        lowered.clone().into(),
        lowered.clone().into(),
        limit.into(),
    ]);
    let results = query_nodes(conn, &exact_sql, &args)?;
    if !results.is_empty() {
        return Ok(results);
    }
    let fallback_sql = format!(
        "SELECT {NODE_COLUMNS}, 0.0 AS rank FROM nodes n
         WHERE {repo_clause} lower(n.qname) LIKE ? AND lower(n.name) != ? AND lower(n.qname) != ?
         ORDER BY CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END, length(n.qname)
         LIMIT ?"
    );
    let mut args = repo_arg;
    args.extend([
        format!("%{lowered}%").into(),
        lowered.clone().into(),
        lowered.into(),
        limit.into(),
    ]);
    query_nodes(conn, &fallback_sql, &args)
}

pub fn get_node(conn: &Connection, node_id: &str) -> Result<Option<NodeRow>> {
    let sql = format!("SELECT {NODE_COLUMNS}, 0.0 AS rank FROM nodes n WHERE n.id = ?");
    Ok(query_nodes(conn, &sql, &[text(node_id)])?
        .into_iter()
        .next())
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EdgeInfo {
    pub kind: String,
    pub confidence: String,
    pub detail: String,
}

/// One neighbor: the connecting edge and the node at its other end (`None` when the edge
/// points at a node id that is not stored).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Neighbor {
    pub edge: EdgeInfo,
    pub node: Option<NodeRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Neighbors {
    pub node: Option<NodeRow>,
    pub outgoing: Vec<Neighbor>,
    pub incoming: Vec<Neighbor>,
}

const NEIGHBOR_ORDER_SQL: &str = "
    ORDER BY
      CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
      CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
      COALESCE(n.path, ''),
      CASE WHEN n.id IS NULL THEN 0 WHEN COALESCE(n.start_line, 0) < 1 THEN 1 ELSE n.start_line END";

fn neighbor_sql(outgoing: bool) -> String {
    let (join_col, where_col) = if outgoing {
        ("e.dst", "e.src")
    } else {
        ("e.src", "e.dst")
    };
    format!(
        "SELECT e.kind, e.confidence, e.detail, {NODE_COLUMNS}, 0.0
         FROM edges e LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = {join_col}
         WHERE e.repo_id = ? AND {where_col} = ?"
    )
}

fn neighbor_from_row(row: &Row<'_>) -> rusqlite::Result<Neighbor> {
    let node_id: Option<String> = row.get(3)?;
    Ok(Neighbor {
        edge: EdgeInfo {
            kind: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
            confidence: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            detail: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
        },
        node: if node_id.is_some() {
            Some(NodeRow::from_row(row, 3)?)
        } else {
            None
        },
    })
}

fn confidence_order(confidence: &str) -> u8 {
    match confidence {
        "resolved" => 0,
        "strong" => 1,
        _ => 2,
    }
}

/// Sort key shared by the Python `neighbor_sort_key` and the SQL ordering above.
pub fn neighbor_sort_key(item: &Neighbor) -> (u8, u8, String, i64) {
    let node = item.node.as_ref();
    (
        u8::from(node.is_some_and(|n| is_external(&n.kind))),
        confidence_order(&item.edge.confidence),
        node.map(|n| n.path.clone()).unwrap_or_default(),
        node.map_or(0, |n| n.start_line),
    )
}

/// Python list slicing `items[:limit]`, including the negative-limit case.
fn py_head<T>(mut items: Vec<T>, limit: i64) -> Vec<T> {
    let len = i64::try_from(items.len()).unwrap_or(i64::MAX);
    let keep = if limit < 0 {
        (len + limit).max(0)
    } else {
        limit.min(len)
    };
    items.truncate(usize::try_from(keep).unwrap_or(0));
    items
}

fn bounded_sorted_neighbors(
    conn: &Connection,
    outgoing: bool,
    repo_id: &str,
    node_id: &str,
    limit: i64,
) -> Result<Vec<Neighbor>> {
    let base = neighbor_sql(outgoing);
    let collect = |sql: &str, args: &[SqlValue]| -> Result<Vec<Neighbor>> {
        let mut stmt = conn.prepare_cached(sql)?;
        let rows = stmt.query_map(params_from_iter(args), neighbor_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    };
    let sort_and_cut = |mut rows: Vec<Neighbor>| {
        rows.sort_by_cached_key(neighbor_sort_key);
        py_head(rows, limit)
    };
    if limit <= 0 {
        return Ok(sort_and_cut(collect(
            &base,
            &[text(repo_id), text(node_id)],
        )?));
    }
    let probe = collect(
        &format!("{base} LIMIT ?"),
        &[text(repo_id), text(node_id), (limit + 1).into()],
    )?;
    if i64::try_from(probe.len()).unwrap_or(i64::MAX) <= limit {
        return Ok(sort_and_cut(probe));
    }
    collect(
        &format!("{base}{NEIGHBOR_ORDER_SQL} LIMIT ?"),
        &[text(repo_id), text(node_id), limit.into()],
    )
}

pub fn get_neighbors(conn: &Connection, node_id: &str, limit: i64) -> Result<Neighbors> {
    let Some(node) = get_node(conn, node_id)? else {
        return Ok(Neighbors {
            node: None,
            outgoing: Vec::new(),
            incoming: Vec::new(),
        });
    };
    let repo_id = node.repo_id.clone();
    let outgoing = bounded_sorted_neighbors(conn, true, &repo_id, node_id, limit)?;
    let incoming = bounded_sorted_neighbors(conn, false, &repo_id, node_id, limit)?;
    Ok(Neighbors {
        node: Some(node),
        outgoing,
        incoming,
    })
}

pub fn pending_embedding_nodes(conn: &Connection, limit: i64) -> Result<Vec<NodeRow>> {
    if limit < 0 {
        return Err(Error::invalid("Embedding limit must be non-negative"));
    }
    let sql = format!(
        "SELECT {NODE_COLUMNS}, 0.0 AS rank FROM embedding_queue q JOIN nodes n ON n.id = q.node_id
         WHERE q.embedded_at IS NULL ORDER BY q.queued_at LIMIT ?"
    );
    query_nodes(conn, &sql, &[limit.into()])
}

/// One vector to store: `(node_id, repo_id, vector, model)`.
pub struct EmbeddingRow {
    pub node_id: String,
    pub repo_id: String,
    pub vector: Vec<Value>,
    pub model: String,
}

pub fn upsert_embeddings(
    conn: &mut Connection,
    rows: &[EmbeddingRow],
    embedded_at: Option<f64>,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let now = embedded_at.unwrap_or_else(unix_time);
    let tx = conn.transaction()?;
    for row in rows {
        tx.execute(
            "INSERT INTO embeddings(node_id, repo_id, dims, vector_json, model, embedded_at) VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(node_id) DO UPDATE SET repo_id=excluded.repo_id, dims=excluded.dims,
               vector_json=excluded.vector_json, model=excluded.model, embedded_at=excluded.embedded_at",
            params![row.node_id, row.repo_id, row.vector.len(), crate::json::dumps(&row.vector), row.model, now],
        )?;
        tx.execute(
            "UPDATE embedding_queue SET embedded_at = ? WHERE node_id = ?",
            params![now, row.node_id],
        )?;
    }
    tx.commit()?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EmbeddingCounts {
    pub queued: i64,
    pub embedded: i64,
}

pub fn embedding_counts(conn: &Connection) -> Result<EmbeddingCounts> {
    let queued = conn.query_row(
        "SELECT COUNT(*) FROM embedding_queue WHERE embedded_at IS NULL",
        [],
        |r| r.get(0),
    )?;
    let embedded = conn.query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))?;
    Ok(EmbeddingCounts { queued, embedded })
}

/// Previously stored hash per file path, used to skip unchanged files on re-index.
pub fn stored_file_hashes(conn: &Connection, repo_id: &str) -> Result<HashMap<String, String>> {
    let mut stmt = conn.prepare("SELECT path, content_hash FROM files WHERE repo_id = ?")?;
    let rows = stmt.query_map([repo_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}
