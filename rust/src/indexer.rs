//! Repository walk and per-file parsing (`indexer.py`). Python goes through `pyast`; TS/JS,
//! Markdown and config files keep the regex heuristics of the Python implementation.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use rayon::prelude::*;
use regex::Regex;
use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_text_size::Ranged;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::{resolve_path, sha1_hex};
use crate::model::{Edge, FileIndex, Node, NodeAttrs, dedupe_edges, dedupe_nodes, make_node};
use crate::observability::unix_time;
use crate::pyast::{self, ImportOrCall, PyModule};
use crate::text::{camel_parts, splitlines, static_regex as regex, take_chars};
use crate::{Error, Result, storage};

pub const SOURCE_EXTENSIONS: [(&str, &str); 14] = [
    (".py", "python"),
    (".pyw", "python"),
    (".ts", "typescript"),
    (".tsx", "typescript"),
    (".js", "javascript"),
    (".jsx", "javascript"),
    (".mjs", "javascript"),
    (".cjs", "javascript"),
    (".md", "markdown"),
    (".mdx", "markdown"),
    (".json", "json"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
    (".toml", "toml"),
];

pub const EXCLUDED_DIRS: [&str; 20] = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    ".local-repo-kg",
    ".lode",
    ".kg",
    "bench-results",
    "droid-wiki",
];

const BODY_STOPWORDS: &[&str] = &[
    "self",
    "cls",
    "none",
    "true",
    "false",
    "return",
    "def",
    "class",
    "import",
    "from",
    "as",
    "with",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "pass",
    "if",
    "else",
    "elif",
    "in",
    "is",
    "and",
    "or",
    "not",
    "lambda",
    "yield",
    "raise",
    "assert",
    "async",
    "await",
    "const",
    "let",
    "var",
    "function",
    "export",
    "default",
    "this",
    "new",
    "null",
    "undefined",
];
const ROUTE_METHODS: [&str; 8] = [
    "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "ROUTE",
];
const TS_CALL_KEYWORDS: [&str; 6] = ["if", "for", "while", "switch", "return", "function"];

static BODY_TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"[A-Za-z_][A-Za-z0-9_]{1,}"));
static PY_FALLBACK_DEF_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^([ \t]*)(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)"));
static PY_FALLBACK_CLASS_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^([ \t]*)class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*:"));
static PY_FALLBACK_FROM_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^\s*from\s+(\.*[A-Za-z0-9_.]*)\s+import\s+(.+)$"));
static PY_FALLBACK_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"^\s*import\s+(.+)$"));
static PY_FALLBACK_ASSIGN_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*(.+)$"));
// The Python patterns close the module string with a `\2` backreference; the `regex` crate
// has none, so each quote style is its own alternative.
static TS_IMPORT_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r#"import\s+(.*?)\s+from\s+(?:'([^'"]+)'|"([^'"]+)")"#));
static TS_REQUIRE_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r#"require\(\s*(?:'([^'"]+)'|"([^'"]+)")\s*\)"#));
static TS_ROUTE_RE: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r#"\b(?:app|router)\.(get|post|put|patch|delete|options|head)\s*\(\s*(?:'([^'"]+)'|"([^'"]+)")\s*,\s*([A-Za-z_$][\w$]*)"#,
    )
});
static TS_CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\("));
static TS_SYMBOL_PATTERNS: LazyLock<[(&str, Regex); 4]> = LazyLock::new(|| {
    [
        ("Class", regex(r"(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        (
            "Function",
            regex(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)"),
        ),
        (
            "Function",
            regex(r"(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>"),
        ),
        (
            "Function",
            regex(
                r"(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?([A-Za-z_$][\w$]*)\s*=>",
            ),
        ),
    ]
});
static MD_HEADING_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"(?m)^(#{1,6})\s+(.+?)\s*$"));
static SLUG_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"[^a-z0-9]+"));
static CLAUSE_DEFAULT_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"^(\w+)"));
static CLAUSE_NAMESPACE_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"\*\s+as\s+(\w+)"));
static CLAUSE_NAMED_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"\{([^}]+)\}"));

/// Counters returned by `index_repo`; serialized with the same keys as the Python dataclass.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct IndexStats {
    pub repo_id: String,
    pub root: String,
    pub scanned: u64,
    pub indexed: u64,
    pub skipped_unchanged: u64,
    pub removed: u64,
    pub resolved_imports: u64,
    pub resolved_calls: u64,
    pub resolved_extends: u64,
    pub nodes: u64,
    pub edges: u64,
}

/// Tokens of `text` and their identifier parts, deduplicated case-insensitively, capped at
/// `max_chars` characters including one separator per token.
pub fn extract_fts_body(text: &str, max_chars: usize) -> String {
    if text.is_empty() || max_chars == 0 {
        return String::new();
    }
    let mut tokens: Vec<&str> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut used = 0;
    for raw in BODY_TOKEN_RE.find_iter(take_chars(text, max_chars * 8)) {
        let raw = raw.as_str();
        let parts = std::iter::once(raw).chain(
            raw.split('_')
                .filter(|chunk| !chunk.is_empty())
                .flat_map(camel_parts),
        );
        for candidate in parts {
            let low = candidate.to_lowercase();
            if low.len() <= 1 || BODY_STOPWORDS.contains(&low.as_str()) || seen.contains(&low) {
                continue;
            }
            seen.insert(low);
            used += candidate.len() + 1;
            if used > max_chars {
                return tokens.join(" ");
            }
            tokens.push(candidate);
        }
    }
    tokens.join(" ")
}

pub fn index_repo(repo_path: &Path, db_path: &Path) -> Result<IndexStats> {
    let root = resolve_path(repo_path);
    if !root.is_dir() {
        return Err(Error::NotFound(format!(
            "Repository path does not exist or is not a directory: {}",
            root.display()
        )));
    }
    let mut conn = storage::connect(db_path)?;
    let repo_id = storage::upsert_repo(&conn, &root)?;
    let mut stats = IndexStats {
        repo_id: repo_id.clone(),
        root: root.to_string_lossy().into_owned(),
        ..IndexStats::default()
    };
    let previous = storage::stored_file_hashes(&conn, &repo_id)?;
    let mut live_paths = HashSet::new();
    let mut jobs = Vec::new();
    let mut skipped_updates = Vec::new();
    for path in iter_source_files(&root) {
        let rel = relative_posix(&root, &path);
        stats.scanned += 1;
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        live_paths.insert(rel.clone());
        let Ok(raw) = std::fs::read(&path) else {
            live_paths.remove(&rel);
            continue;
        };
        let digest = sha1_hex(&raw);
        let mtime = mtime_of(&meta);
        if previous.get(&rel) == Some(&digest) {
            stats.skipped_unchanged += 1;
            skipped_updates.push((mtime, rel));
            continue;
        }
        let text = String::from_utf8_lossy(&raw).into_owned();
        jobs.push((path, digest, mtime, meta.len(), text));
    }
    let changed: Vec<FileIndex> = jobs
        .into_par_iter()
        .map(|(path, digest, mtime, size, text)| {
            parse_file(&root, &path, &digest, mtime, size, &text)
        })
        .collect();
    stats.indexed = changed.len() as u64;
    stats.nodes = changed.iter().map(|file| file.nodes.len() as u64).sum();
    stats.edges = changed.iter().map(|file| file.edges.len() as u64).sum();
    let tx = conn.transaction()?;
    {
        let mut update = tx
            .prepare("UPDATE files SET mtime = ?, indexed_at = ? WHERE repo_id = ? AND path = ?")?;
        for (mtime, rel) in &skipped_updates {
            update.execute(rusqlite::params![mtime, unix_time(), repo_id, rel])?;
        }
    }
    storage::replace_file_indexes(&tx, &repo_id, changed)?;
    stats.removed = storage::remove_missing_files(&tx, &repo_id, &live_paths)? as u64;
    tx.commit()?;
    if stats.indexed > 0 || stats.removed > 0 {
        let resolved = crate::graph::resolve_graph(&mut conn, &repo_id)?;
        stats.resolved_imports = resolved.imports;
        stats.resolved_calls = resolved.calls;
        stats.resolved_extends = resolved.extends;
    }
    Ok(stats)
}

fn mtime_of(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |duration| duration.as_secs_f64())
}

fn relative_posix(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn language_for(path: &Path) -> Option<&'static str> {
    let ext = format!(".{}", path.extension()?.to_string_lossy().to_lowercase());
    SOURCE_EXTENSIONS
        .iter()
        .find(|(suffix, _)| *suffix == ext)
        .map(|(_, language)| *language)
}

/// Source files under `root`, skipping excluded/hidden/gitignored paths and generated files.
/// Sorted for a stable order; the Python walk uses directory order.
pub fn iter_source_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut ignores = Vec::new();
    if let Some(exclude) =
        crate::gitignore::IgnoreFile::from_path(&root.join(".git/info/exclude"), "")
    {
        ignores.push(exclude);
    }
    walk_source_dir(root, "", &mut ignores, &mut out);
    out.sort();
    out
}

fn walk_source_dir(
    dir: &Path,
    dir_rel: &str,
    ignores: &mut Vec<crate::gitignore::IgnoreFile>,
    out: &mut Vec<PathBuf>,
) {
    let pushed = crate::gitignore::IgnoreFile::from_path(&dir.join(".gitignore"), dir_rel)
        .map(|file| ignores.push(file))
        .is_some();
    let Ok(entries) = std::fs::read_dir(dir) else {
        if pushed {
            ignores.pop();
        }
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(std::result::Result::ok).collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let rel = if dir_rel.is_empty() {
            name.clone()
        } else {
            format!("{dir_rel}/{name}")
        };
        let is_dir = file_type.is_dir() || (file_type.is_symlink() && path.is_dir());
        if is_dir {
            // os.walk does not descend into directory symlinks by default.
            if !file_type.is_symlink()
                && !EXCLUDED_DIRS.contains(&name.as_str())
                && !name.starts_with('.')
                && !crate::gitignore::is_ignored(ignores, &rel, true)
            {
                walk_source_dir(&path, &rel, ignores, out);
            }
            continue;
        }
        if language_for(&path).is_some()
            && !crate::gitignore::is_ignored(ignores, &rel, false)
            && !is_probably_generated(&path)
        {
            out.push(path);
        }
    }
    if pushed {
        ignores.pop();
    }
}

pub fn is_probably_generated(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.ends_with(".min.js") || name.ends_with(".bundle.js") {
        return true;
    }
    if name.contains("generated") || name.ends_with(".lock") {
        return true;
    }
    std::fs::metadata(path).map_or(true, |meta| meta.len() > 1_000_000)
}

fn file_name(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

fn line_count(text: &str) -> i64 {
    (text.matches('\n').count() as i64 + 1).max(1)
}

pub fn parse_file(
    root: &Path,
    path: &Path,
    digest: &str,
    mtime: f64,
    size: u64,
    text: &str,
) -> FileIndex {
    let rel = relative_posix(root, path);
    let language = language_for(path).unwrap_or("text");
    let content_hash = if digest.is_empty() {
        sha1_hex(text.as_bytes())
    } else {
        digest.to_owned()
    };
    let mut file_node = make_node(
        "File",
        file_name(&rel),
        &rel,
        &rel,
        1,
        line_count(text),
        &content_hash,
        NodeAttrs::default(),
    );
    let fts_body = extract_fts_body(text, 1600);
    if !fts_body.is_empty() {
        file_node.extra.insert("fts_body".into(), fts_body.into());
    }
    let file_id = file_node.id.clone();
    let parsed = match language {
        "python" => parse_python(&rel, text, &file_id),
        "typescript" | "javascript" => parse_ts_js(&rel, text, &file_id, language),
        "markdown" => parse_markdown(&rel, text, &file_id),
        _ => parse_config(&rel, text, &file_id, language),
    };
    if !parsed.imports.is_empty() {
        file_node
            .extra
            .insert("imports".into(), Value::Array(parsed.imports));
    }
    if !parsed.receivers.is_empty() {
        file_node
            .extra
            .insert("receivers".into(), Value::Object(parsed.receivers));
    }
    let mut nodes = vec![file_node];
    nodes.extend(parsed.nodes);
    FileIndex {
        path: rel,
        abspath: path.to_string_lossy().into_owned(),
        language: language.to_owned(),
        size,
        mtime,
        content_hash,
        generated: false,
        nodes: dedupe_nodes(nodes),
        edges: dedupe_edges(parsed.edges),
    }
}

/// Nodes, edges and import bindings found in one file (besides its File node).
#[derive(Debug, Default)]
pub struct Parsed {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub imports: Vec<Value>,
    pub receivers: serde_json::Map<String, Value>,
}

/// `Path(rel).with_suffix("")` with `/` turned into `.`.
pub fn module_qname(rel: &str) -> String {
    let (dir, name) = match rel.rfind('/') {
        Some(index) => (&rel[..=index], &rel[index + 1..]),
        None => ("", rel),
    };
    let stem = match name.rfind('.') {
        Some(index) if index > 0 && index + 1 < name.len() => &name[..index],
        _ => name,
    };
    format!("{dir}{stem}").replace('/', ".")
}

fn binding(module: &str, name: Option<&str>, alias: Option<&str>) -> Value {
    json!({"module": module, "name": name, "alias": alias})
}

/// `(name, alias)` pairs of a JS import clause (also used for Python fallback lines).
pub fn split_import_clause(clause: &str) -> Vec<(Option<String>, Option<String>)> {
    let mut parts = Vec::new();
    if clause.is_empty() {
        return parts;
    }
    let default = CLAUSE_DEFAULT_RE.captures(clause);
    if let Some(caps) = &default {
        parts.push((Some("default".to_owned()), Some(caps[1].to_owned())));
    }
    let namespace = CLAUSE_NAMESPACE_RE.captures(clause);
    if let Some(caps) = &namespace {
        parts.push((Some("*".to_owned()), Some(caps[1].to_owned())));
    }
    let named = CLAUSE_NAMED_RE.captures(clause);
    if let Some(caps) = &named {
        for item in caps[1].split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            match item.split_once(" as ") {
                Some((orig, alias)) => {
                    parts.push((Some(orig.trim().to_owned()), Some(alias.trim().to_owned())));
                }
                None => parts.push((Some(item.to_owned()), None)),
            }
        }
    }
    if parts.is_empty() && default.is_none() && namespace.is_none() && named.is_none() {
        parts.push((None, None));
    }
    parts
}

fn slice_lines(lines: &[&str], start: i64, end: i64) -> String {
    let len = lines.len() as i64;
    let from = (start - 1).clamp(0, len) as usize;
    let to = start.max(end).clamp(0, len) as usize;
    if from >= to {
        return String::new();
    }
    lines[from..to].join("\n")
}

fn node_with(
    kind: &str,
    name: &str,
    qname: &str,
    rel: &str,
    lines: (i64, i64),
    content: &str,
    attrs: NodeAttrs<'_>,
) -> Node {
    make_node(kind, name, qname, rel, lines.0, lines.1, content, attrs)
}

/// External nodes are created once per `(kind, name)` and appended on first use.
struct ExternalCache<'a> {
    rel: &'a str,
    ids: HashMap<(String, String), String>,
}

impl ExternalCache<'_> {
    fn id(&mut self, nodes: &mut Vec<Node>, kind: &str, name: &str) -> String {
        let key = (kind.to_owned(), name.to_owned());
        if let Some(id) = self.ids.get(&key) {
            return id.clone();
        }
        let node = crate::model::external_node(kind, name, self.rel);
        let id = node.id.clone();
        nodes.push(node);
        self.ids.insert(key, id.clone());
        id
    }
}

pub fn parse_python(rel: &str, text: &str, file_id: &str) -> Parsed {
    let module = module_qname(rel);
    let mut out = Parsed::default();
    let mut externals = ExternalCache {
        rel,
        ids: HashMap::new(),
    };
    let tree = match pyast::parse(text) {
        Ok(tree) => tree,
        Err(err) => {
            let err_node = make_node(
                "ParseError",
                "SyntaxError",
                &format!("{module}.SyntaxError"),
                rel,
                err.line,
                err.line,
                text,
                NodeAttrs {
                    doc: &err.message,
                    confidence: "exact",
                    ..NodeAttrs::default()
                },
            );
            out.edges.push(Edge::new(
                file_id,
                &err_node.id,
                "HAS_PARSE_ERROR",
                "exact",
                "",
            ));
            out.nodes.push(err_node);
            for (lineno, line) in splitlines(text).into_iter().enumerate() {
                fallback_line(
                    rel,
                    &module,
                    lineno as i64 + 1,
                    line,
                    file_id,
                    &mut externals,
                    &mut out,
                );
            }
            // indexer.py appends fallback externals to a list it then discards, so only their
            // IMPORTS edges survive (pointing at unstored nodes). Kept for parity; see PORTING.md.
            out.nodes
                .retain(|node| !crate::model::is_external(&node.kind));
            return out;
        }
    };
    let lines = splitlines(text);
    for (key, value) in pyast::collect_module_receivers(&tree.body) {
        out.receivers.insert(key, Value::String(value));
    }
    let mut symbols: HashMap<String, String> = HashMap::new();
    for item in &tree.body {
        match item {
            Stmt::FunctionDef(def) => {
                let node = python_function_node(&tree, &module, rel, item, def, &lines, "Function");
                symbols.insert(def.name.id.to_string(), node.id.clone());
                out.edges
                    .push(Edge::new(file_id, &node.id, "DEFINES", "exact", ""));
                route_facts(&tree, def, item, &node, rel, file_id, &mut out);
                out.nodes.push(node);
            }
            Stmt::ClassDef(class) => {
                let class_node = python_class_node(&tree, &module, rel, item, class, &lines);
                symbols.insert(class.name.id.to_string(), class_node.id.clone());
                out.edges
                    .push(Edge::new(file_id, &class_node.id, "DEFINES", "exact", ""));
                let class_id = class_node.id.clone();
                out.nodes.push(class_node);
                for base in class.bases() {
                    let base_name = pyast::dotted_name(base);
                    if !base_name.is_empty() {
                        let base_id = externals.id(&mut out.nodes, "ExternalSymbol", &base_name);
                        out.edges.push(Edge::new(
                            &class_id,
                            &base_id,
                            "EXTENDS",
                            "heuristic",
                            &base_name,
                        ));
                    }
                }
                let prefix = format!("{module}.{}", class.name.id);
                for child in &class.body {
                    if let Stmt::FunctionDef(def) = child {
                        let method =
                            python_function_node(&tree, &prefix, rel, child, def, &lines, "Method");
                        out.edges
                            .push(Edge::new(&class_id, &method.id, "CONTAINS", "exact", ""));
                        out.edges
                            .push(Edge::new(file_id, &method.id, "DEFINES", "exact", ""));
                        route_facts(&tree, def, child, &method, rel, file_id, &mut out);
                        out.nodes.push(method);
                    }
                }
            }
            Stmt::Assign(_) | Stmt::AnnAssign(_) => {
                for var in python_assign_nodes(&tree, &module, rel, item, &lines) {
                    symbols.insert(var.name.clone(), var.id.clone());
                    out.edges
                        .push(Edge::new(file_id, &var.id, "DEFINES", "exact", ""));
                    out.nodes.push(var);
                }
            }
            _ => {}
        }
    }
    let mut calls = Vec::new();
    for found in tree.imports_and_calls() {
        match found {
            ImportOrCall::Import(import) => {
                for alias in &import.names {
                    let name = alias.name.id.as_str();
                    out.imports.push(binding(
                        name,
                        None,
                        alias.asname.as_ref().map(|a| a.id.as_str()),
                    ));
                    let dep = externals.id(&mut out.nodes, "ExternalDependency", name);
                    out.edges
                        .push(Edge::new(file_id, &dep, "IMPORTS", "strong", name));
                }
            }
            ImportOrCall::ImportFrom(import) => {
                let from = import_from_module(import);
                for alias in &import.names {
                    let name = alias.name.id.as_str();
                    let alias_name = if name == "*" {
                        None
                    } else {
                        alias.asname.as_ref().map(|a| a.id.as_str())
                    };
                    out.imports.push(binding(&from, Some(name), alias_name));
                }
                if !from.is_empty() {
                    let dep = externals.id(&mut out.nodes, "ExternalDependency", &from);
                    out.edges
                        .push(Edge::new(file_id, &dep, "IMPORTS", "strong", &from));
                }
            }
            ImportOrCall::Call(call) => calls.push(call),
        }
    }
    let ranges = function_ranges(&out.nodes);
    for call in calls {
        let call_name = pyast::dotted_name(&call.func);
        if call_name.is_empty() {
            continue;
        }
        let Some(caller) = enclosing_node_for_line(&ranges, tree.line(call.start())) else {
            continue;
        };
        let target = (!call_name.contains('.'))
            .then(|| symbols.get(&call_name))
            .flatten();
        match target {
            Some(target) => out
                .edges
                .push(Edge::new(&caller, target, "CALLS", "strong", &call_name)),
            None => {
                let target = externals.id(&mut out.nodes, "ExternalSymbol", &call_name);
                out.edges.push(Edge::new(
                    &caller,
                    &target,
                    "CALLS",
                    "heuristic",
                    &call_name,
                ));
            }
        }
    }
    out
}

fn import_from_module(import: &ast::StmtImportFrom) -> String {
    let base = import
        .module
        .as_ref()
        .map(|m| m.id.to_string())
        .unwrap_or_default();
    format!("{}{base}", ".".repeat(import.level as usize))
}

/// Function and method ranges sorted narrowest first, then by start line.
fn function_ranges(nodes: &[Node]) -> Vec<(i64, i64, String)> {
    let mut ranges: Vec<(i64, i64, String)> = nodes
        .iter()
        .filter(|node| node.kind == "Function" || node.kind == "Method")
        .map(|node| (node.start_line, node.end_line, node.id.clone()))
        .collect();
    ranges.sort_by_key(|(start, end, _)| (end - start, *start));
    ranges
}

fn enclosing_node_for_line(ranges: &[(i64, i64, String)], line: i64) -> Option<String> {
    ranges
        .iter()
        .find(|(start, end, _)| *start <= line && line <= *end)
        .map(|(_, _, id)| id.clone())
}

fn with_fts_body(mut node: Node, body: &str) -> Node {
    let fts_body = extract_fts_body(body, 600);
    if !fts_body.is_empty() {
        node.extra.insert("fts_body".into(), fts_body.into());
    }
    node
}

fn python_function_node(
    tree: &PyModule,
    prefix: &str,
    rel: &str,
    stmt: &Stmt,
    def: &ast::StmtFunctionDef,
    lines: &[&str],
    kind: &str,
) -> Node {
    let params = &def.parameters;
    let mut args: Vec<String> = params
        .args
        .iter()
        .map(|p| p.parameter.name.id.to_string())
        .collect();
    if let Some(vararg) = &params.vararg {
        args.push(format!("*{}", vararg.name.id));
    }
    if let Some(kwarg) = &params.kwarg {
        args.push(format!("**{}", kwarg.name.id));
    }
    let name = def.name.id.as_str();
    let signature = format!("{name}({})", args.join(", "));
    let span = tree.stmt_lines(stmt);
    let doc = pyast::docstring(&def.body);
    let body = slice_lines(lines, span.0, span.1);
    let mut node = node_with(
        kind,
        name,
        &format!("{prefix}.{name}"),
        rel,
        span,
        &body,
        NodeAttrs {
            signature: &signature,
            doc: &doc,
            confidence: "exact",
            ..NodeAttrs::default()
        },
    );
    let receivers = pyast::collect_function_receivers(def);
    if !receivers.is_empty() {
        let map: serde_json::Map<String, Value> = receivers
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        node.extra.insert("receivers".into(), Value::Object(map));
    }
    with_fts_body(node, &body)
}

fn python_class_node(
    tree: &PyModule,
    module: &str,
    rel: &str,
    stmt: &Stmt,
    class: &ast::StmtClassDef,
    lines: &[&str],
) -> Node {
    let bases: Vec<String> = class
        .bases()
        .iter()
        .map(pyast::dotted_name)
        .filter(|base| !base.is_empty())
        .collect();
    let name = class.name.id.as_str();
    let signature = if bases.is_empty() {
        format!("class {name}")
    } else {
        format!("class {name}({})", bases.join(", "))
    };
    let span = tree.stmt_lines(stmt);
    let doc = pyast::docstring(&class.body);
    let body = slice_lines(lines, span.0, span.1);
    let mut node = node_with(
        "Class",
        name,
        &format!("{module}.{name}"),
        rel,
        span,
        &body,
        NodeAttrs {
            signature: &signature,
            doc: &doc,
            confidence: "exact",
            ..NodeAttrs::default()
        },
    );
    let attr_types = pyast::collect_class_attr_types(class);
    if !attr_types.is_empty() {
        let map: serde_json::Map<String, Value> = attr_types
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        node.extra.insert("attr_types".into(), Value::Object(map));
    }
    with_fts_body(node, &body)
}

fn python_assign_nodes(
    tree: &PyModule,
    module: &str,
    rel: &str,
    stmt: &Stmt,
    lines: &[&str],
) -> Vec<Node> {
    let targets: Vec<&Expr> = match stmt {
        Stmt::Assign(assign) => assign.targets.iter().collect(),
        Stmt::AnnAssign(assign) => vec![assign.target.as_ref()],
        _ => Vec::new(),
    };
    let span = tree.stmt_lines(stmt);
    let body = slice_lines(lines, span.0, span.1);
    let signature = splitlines(&body)
        .first()
        .map(|line| take_chars(line.trim(), 160).to_owned())
        .unwrap_or_default();
    targets
        .into_iter()
        .filter_map(|target| match target {
            Expr::Name(name) if !name.id.starts_with("__") => Some(node_with(
                "Variable",
                &name.id,
                &format!("{module}.{}", name.id),
                rel,
                span,
                &body,
                NodeAttrs {
                    signature: &signature,
                    confidence: "exact",
                    ..NodeAttrs::default()
                },
            )),
            _ => None,
        })
        .collect()
}

fn route_facts(
    tree: &PyModule,
    def: &ast::StmtFunctionDef,
    stmt: &Stmt,
    handler: &Node,
    rel: &str,
    file_id: &str,
    out: &mut Parsed,
) {
    let line = tree.stmt_lines(stmt).0;
    for decorator in &def.decorator_list {
        let Expr::Call(call) = &decorator.expression else {
            continue;
        };
        let method_name = pyast::dotted_name(&call.func);
        if method_name.is_empty() {
            continue;
        }
        let method = method_name
            .rsplit('.')
            .next()
            .unwrap_or_default()
            .to_uppercase();
        if !ROUTE_METHODS.contains(&method.as_str()) {
            continue;
        }
        let route_path = match call.arguments.args.first() {
            Some(Expr::StringLiteral(literal)) => literal.value.to_str().to_owned(),
            _ => String::new(),
        };
        if route_path.is_empty() {
            continue;
        }
        let label = format!("{method} {route_path}");
        let route = make_node(
            "Route",
            &label,
            &format!("route:{label}"),
            rel,
            line,
            line,
            &route_path,
            NodeAttrs {
                signature: &label,
                confidence: "strong",
                ..NodeAttrs::default()
            },
        );
        out.edges.push(Edge::new(
            file_id,
            &route.id,
            "DEFINES",
            "strong",
            &method_name,
        ));
        out.edges.push(Edge::new(
            &route.id,
            &handler.id,
            "HANDLES",
            "strong",
            &method_name,
        ));
        out.edges.push(Edge::new(
            &route.id,
            &handler.id,
            "CALLS",
            "heuristic",
            &method_name,
        ));
        out.edges.push(Edge::new(
            &handler.id,
            &route.id,
            "EXPOSES",
            "strong",
            &method_name,
        ));
        out.nodes.push(route);
    }
}

fn fallback_line(
    rel: &str,
    module: &str,
    lineno: i64,
    line: &str,
    file_id: &str,
    externals: &mut ExternalCache<'_>,
    out: &mut Parsed,
) {
    let define = |out: &mut Parsed, kind: &str, name: &str, signature: &str| {
        let node = make_node(
            kind,
            name,
            &format!("{module}.{name}"),
            rel,
            lineno,
            lineno,
            line,
            NodeAttrs {
                signature,
                confidence: "heuristic",
                ..NodeAttrs::default()
            },
        );
        out.edges
            .push(Edge::new(file_id, &node.id, "DEFINES", "heuristic", ""));
        out.nodes.push(node);
    };
    if let Some(caps) = PY_FALLBACK_DEF_RE.captures(line) {
        let name = &caps[2];
        define(
            out,
            "Function",
            name,
            &format!("{name}({})", caps[3].trim()),
        );
        return;
    }
    if let Some(caps) = PY_FALLBACK_CLASS_RE.captures(line) {
        let name = &caps[2];
        let bases = caps.get(3).map_or("", |m| m.as_str()).trim();
        let signature = if bases.is_empty() {
            format!("class {name}")
        } else {
            format!("class {name}({bases})")
        };
        define(out, "Class", name, &signature);
        return;
    }
    if let Some(caps) = PY_FALLBACK_FROM_RE.captures(line) {
        let from = &caps[1];
        let names = caps[2].split('#').next().unwrap_or_default();
        for (name, alias) in split_import_clause(names) {
            out.imports
                .push(binding(from, name.as_deref(), alias.as_deref()));
        }
        if !from.is_empty() {
            let dep = externals.id(&mut out.nodes, "ExternalDependency", from);
            out.edges
                .push(Edge::new(file_id, &dep, "IMPORTS", "heuristic", from));
        }
        return;
    }
    if let Some(caps) = PY_FALLBACK_IMPORT_RE.captures(line) {
        let clause = caps[1].split('#').next().unwrap_or_default();
        for (name, alias) in split_import_clause(clause) {
            // Python formats a missing name as "None" here.
            let from = name.unwrap_or_else(|| "None".to_owned());
            out.imports.push(binding(&from, None, alias.as_deref()));
            let dep = externals.id(&mut out.nodes, "ExternalDependency", &from);
            out.edges
                .push(Edge::new(file_id, &dep, "IMPORTS", "heuristic", &from));
        }
        return;
    }
    if let Some(caps) = PY_FALLBACK_ASSIGN_RE.captures(line) {
        let name = &caps[1];
        if !name.starts_with("__") {
            define(out, "Variable", name, take_chars(line.trim(), 160));
        }
    }
}

/// First non-empty capture among the quote alternatives.
fn quoted<'t>(caps: &regex::Captures<'t>, first: usize) -> &'t str {
    caps.get(first)
        .or_else(|| caps.get(first + 1))
        .map_or("", |m| m.as_str())
}

pub fn parse_ts_js(rel: &str, text: &str, file_id: &str, language: &str) -> Parsed {
    let mut out = Parsed::default();
    let module = module_qname(rel);
    let lines = splitlines(text);
    let add_dep = |out: &mut Parsed, from: &str| {
        let dep = crate::model::external_node("ExternalDependency", from, rel);
        out.edges
            .push(Edge::new(file_id, &dep.id, "IMPORTS", "strong", from));
        out.nodes.push(dep);
    };
    for caps in TS_IMPORT_RE.captures_iter(text) {
        let clause = caps[1].trim();
        let from = quoted(&caps, 2);
        for (name, alias) in split_import_clause(clause) {
            out.imports
                .push(binding(from, name.as_deref(), alias.as_deref()));
        }
        add_dep(&mut out, from);
    }
    for caps in TS_REQUIRE_RE.captures_iter(text) {
        let from = quoted(&caps, 1);
        out.imports.push(binding(from, None, None));
        add_dep(&mut out, from);
    }
    let mut symbols: HashMap<String, String> = HashMap::new();
    let mut extra = Map::new();
    extra.insert("language".into(), language.into());
    for (index, line) in lines.iter().enumerate() {
        let line_no = index as i64 + 1;
        for (kind, pattern) in TS_SYMBOL_PATTERNS.iter() {
            let Some(caps) = pattern.captures(line) else {
                continue;
            };
            let name = &caps[1];
            let end = find_block_end(&lines, line_no);
            let signature = take_chars(line.trim(), 240);
            let body = slice_lines(&lines, line_no, end);
            let node = make_node(
                kind,
                name,
                &format!("{module}.{name}"),
                rel,
                line_no,
                end,
                &body,
                NodeAttrs {
                    signature,
                    confidence: "heuristic",
                    extra: extra.clone(),
                    ..NodeAttrs::default()
                },
            );
            symbols.insert(name.to_owned(), node.id.clone());
            out.edges
                .push(Edge::new(file_id, &node.id, "DEFINES", "heuristic", ""));
            out.nodes.push(node);
        }
    }
    for (index, line) in lines.iter().enumerate() {
        let line_no = index as i64 + 1;
        let Some(caps) = TS_ROUTE_RE.captures(line) else {
            continue;
        };
        let method = caps[1].to_uppercase();
        let route_path = quoted(&caps, 2);
        let handler = &caps[4];
        let label = format!("{method} {route_path}");
        let route = make_node(
            "Route",
            &label,
            &format!("route:{label}"),
            rel,
            line_no,
            line_no,
            line,
            NodeAttrs {
                signature: &label,
                confidence: "heuristic",
                ..NodeAttrs::default()
            },
        );
        out.edges
            .push(Edge::new(file_id, &route.id, "DEFINES", "heuristic", ""));
        if let Some(target) = symbols.get(handler) {
            out.edges.push(Edge::new(
                &route.id,
                target,
                "HANDLES",
                "heuristic",
                handler,
            ));
        }
        out.nodes.push(route);
    }
    // Declaration order, not narrowest-first: the Python code does not sort these.
    let ranges: Vec<(i64, i64, String)> = out
        .nodes
        .iter()
        .filter(|node| node.kind == "Function")
        .map(|node| (node.start_line, node.end_line, node.id.clone()))
        .collect();
    for (index, line) in lines.iter().enumerate() {
        let Some(caller) = enclosing_node_for_line(&ranges, index as i64 + 1) else {
            continue;
        };
        for caps in TS_CALL_RE.captures_iter(line) {
            let call_name = &caps[1];
            let first = call_name.split('.').next().unwrap_or_default();
            if TS_CALL_KEYWORDS.contains(&first) {
                continue;
            }
            let target = (!call_name.contains('.'))
                .then(|| symbols.get(call_name))
                .flatten();
            if let Some(target) = target {
                if *target != caller {
                    out.edges
                        .push(Edge::new(&caller, target, "CALLS", "heuristic", call_name));
                }
                continue;
            }
            let node = crate::model::external_node("ExternalSymbol", call_name, rel);
            out.edges.push(Edge::new(
                &caller,
                &node.id,
                "CALLS",
                "heuristic",
                call_name,
            ));
            out.nodes.push(node);
        }
    }
    out
}

pub fn find_block_end(lines: &[&str], start_line: i64) -> i64 {
    let mut depth: i64 = 0;
    let mut seen_open = false;
    let from = (start_line - 1).max(0) as usize;
    let to = lines.len().min((start_line + 400).max(0) as usize);
    for (index, line) in lines.iter().enumerate().take(to).skip(from) {
        let opens = line.matches('{').count() as i64;
        depth += opens;
        seen_open |= opens > 0;
        depth -= line.matches('}').count() as i64;
        if seen_open && depth <= 0 {
            return index as i64 + 1;
        }
    }
    start_line
}

pub fn parse_markdown(rel: &str, text: &str, file_id: &str) -> Parsed {
    let mut out = Parsed::default();
    let headings: Vec<regex::Captures<'_>> = MD_HEADING_RE.captures_iter(text).collect();
    let lines = splitlines(text);
    let count_lines = |upto: usize| text[..upto].matches('\n').count() as i64 + 1;
    for (index, caps) in headings.iter().enumerate() {
        let whole = caps.get_match();
        let title = caps[2].trim();
        let start = count_lines(whole.start());
        let end = headings
            .get(index + 1)
            .and_then(|next| next.get(0))
            .map_or(text.len(), |m| m.start());
        let end_line = count_lines(end);
        let body = slice_lines(&lines, start, end_line);
        let slug_source = SLUG_RE.replace_all(&title.to_lowercase(), "-").into_owned();
        let slug = slug_source.trim_matches('-');
        let signature = format!("{} {title}", &caps[1]);
        let doc = take_chars(&body, 1000);
        let node = make_node(
            "DocSection",
            title,
            &format!("{rel}#{slug}"),
            rel,
            start,
            end_line,
            &body,
            NodeAttrs {
                signature: &signature,
                doc,
                confidence: "exact",
                ..NodeAttrs::default()
            },
        );
        out.edges
            .push(Edge::new(file_id, &node.id, "CONTAINS", "exact", ""));
        out.nodes.push(node);
    }
    out
}

pub fn parse_config(rel: &str, text: &str, file_id: &str, language: &str) -> Parsed {
    let node = make_node(
        "ConfigFile",
        file_name(rel),
        rel,
        rel,
        1,
        line_count(text),
        take_chars(text, 4000),
        NodeAttrs {
            signature: language,
            confidence: "exact",
            ..NodeAttrs::default()
        },
    );
    Parsed {
        edges: vec![Edge::new(file_id, &node.id, "CONTAINS", "exact", "")],
        nodes: vec![node],
        ..Parsed::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_qname_drops_last_suffix() {
        assert_eq!(module_qname("pkg/mod.py"), "pkg.mod");
        assert_eq!(module_qname("a/b.c.ts"), "a.b.c");
        assert_eq!(module_qname("README"), "README");
    }

    #[test]
    fn split_import_clause_matches_python() {
        assert_eq!(
            split_import_clause("React, { useState as us, Fragment }"),
            vec![
                (Some("default".into()), Some("React".into())),
                (Some("useState".into()), Some("us".into())),
                (Some("Fragment".into()), None),
            ]
        );
        assert_eq!(split_import_clause("'x'"), vec![(None, None)]);
    }

    #[test]
    fn fts_body_dedupes_and_splits() {
        assert_eq!(
            extract_fts_body("def getUserName(self): return x_y", 1600),
            "getUserName get User Name x_y"
        );
    }
}
