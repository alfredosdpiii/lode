//! Graph resolution and blast-radius traversal over the SQLite index (`graph.py`).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use rusqlite::{Connection, params};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::storage::{self, Neighbor, NodeRow, decode_extra};
use crate::{Error, Result};

const SELF_NAMES: [&str; 3] = ["self", "cls", "this"];
const CALLABLE_KINDS: &[&str] = &["Function", "Class", "Method"];
const CLASS_KINDS: &[&str] = &["Class"];
const EXPANDABLE_NODE_KINDS: [&str; 5] = ["File", "Function", "Method", "Class", "Route"];
const INTERNAL_NODE_KINDS: [&str; 6] =
    ["File", "Function", "Method", "Class", "Route", "DocSection"];
const JS_SOURCE_SUFFIXES: [&str; 6] = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"];

const WALK_SELECT: &str = "SELECT e.kind, e.confidence, e.detail, \
       n.id, n.kind, n.name, n.qname, n.path, n.start_line, n.end_line FROM edges e ";
const WALK_ORDER: &str = " AND e.kind IN ('CALLS','EXTENDS','HANDLES','CONTAINS','IMPORTS')
ORDER BY
  CASE e.confidence WHEN 'exact' THEN 0 WHEN 'strong' THEN 1 WHEN 'resolved' THEN 2 ELSE 3 END,
  n.path, n.start_line
LIMIT ?";

/// Number of resolved edges `resolve_graph` inserted, by kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResolveCounts {
    pub imports: u64,
    pub calls: u64,
    pub extends: u64,
}

#[derive(Debug, Clone)]
struct DefRow {
    id: String,
    kind: String,
    name: String,
    qname: String,
    path: String,
}

#[derive(Debug, Clone)]
struct ImportBinding {
    module: String,
    name: Option<String>,
    alias: String,
}

/// `str(value or "")` for a JSON value.
fn json_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null | Value::Bool(false)) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) if number.as_f64() == Some(0.0) => String::new(),
        Some(Value::Bool(true)) => "True".to_owned(),
        Some(other) => other.to_string(),
    }
}

fn parse_bindings(extra: &Map<String, Value>) -> Option<Vec<ImportBinding>> {
    let Some(Value::Array(items)) = extra.get("imports") else {
        return None;
    };
    Some(
        items
            .iter()
            .filter_map(Value::as_object)
            .map(|binding| ImportBinding {
                module: json_str(binding.get("module")).trim().to_owned(),
                name: match binding.get("name") {
                    None | Some(Value::Null) => None,
                    Some(value) => Some(json_str(Some(value)).trim().to_owned()),
                },
                alias: json_str(binding.get("alias")).trim().to_owned(),
            })
            .collect(),
    )
}

type AliasMap = HashMap<String, Vec<(String, Option<String>)>>;

fn build_import_aliases(bindings: &[ImportBinding]) -> (AliasMap, Vec<String>) {
    let mut aliases: AliasMap = HashMap::new();
    let mut stars = Vec::new();
    for binding in bindings {
        if binding.module.is_empty() {
            continue;
        }
        let name = binding.name.as_deref();
        if name == Some("*") {
            stars.push(binding.module.clone());
            continue;
        }
        let (key, target_name) = if !binding.alias.is_empty() {
            (binding.alias.clone(), binding.name.clone())
        } else if !matches!(name, None | Some("" | "default")) {
            (name.unwrap_or_default().to_owned(), binding.name.clone())
        } else {
            (binding.module.clone(), None)
        };
        if !key.is_empty() {
            aliases
                .entry(key)
                .or_default()
                .push((binding.module.clone(), target_name));
        }
    }
    (aliases, stars)
}

pub fn join_module_name(module: &str, name: &str) -> String {
    if module.is_empty() {
        name.to_owned()
    } else if module.ends_with('.') {
        format!("{module}{name}")
    } else {
        format!("{module}.{name}")
    }
}

pub fn resolve_python_module(
    module: &str,
    importer: &str,
    files: &HashSet<String>,
) -> Option<String> {
    let module = module.trim();
    if module.is_empty() {
        return None;
    }
    if module.starts_with('.') {
        let rest = module.trim_start_matches('.');
        let level = module.len() - rest.len();
        let mut parts: Vec<&str> = importer.split('/').collect();
        parts.pop();
        let ups = level - 1;
        if ups > parts.len() {
            return None;
        }
        parts.truncate(parts.len() - ups);
        parts.extend(rest.split('.').filter(|segment| !segment.is_empty()));
        let rel = parts.join("/");
        let candidates = if rel.is_empty() {
            vec!["__init__.py".to_owned()]
        } else {
            vec![format!("{rel}.py"), format!("{rel}/__init__.py")]
        };
        return candidates.into_iter().find(|c| files.contains(c));
    }
    let rel = module.replace('.', "/");
    let candidates = [format!("{rel}.py"), format!("{rel}/__init__.py")];
    if let Some(found) = candidates.iter().find(|c| files.contains(*c)) {
        return Some(found.clone());
    }
    candidates.iter().find_map(|candidate| {
        let suffix = format!("/{candidate}");
        let mut matches = files.iter().filter(|path| path.ends_with(&suffix));
        match (matches.next(), matches.next()) {
            (Some(only), None) => Some(only.clone()),
            _ => None,
        }
    })
}

/// `posixpath.normpath` for relative paths.
fn normpath(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." if parts.last().is_some_and(|last| *last != "..") => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        ".".to_owned()
    } else {
        parts.join("/")
    }
}

/// `posixpath.splitext`.
fn splitext(path: &str) -> (&str, &str) {
    let name_start = path.rfind('/').map_or(0, |i| i + 1);
    let name = &path[name_start..];
    let stripped = name.trim_start_matches('.');
    match stripped.rfind('.') {
        Some(dot) => {
            let at = name_start + (name.len() - stripped.len()) + dot;
            (&path[..at], &path[at..])
        }
        None => (path, ""),
    }
}

pub fn resolve_js_module(module: &str, importer: &str, files: &HashSet<String>) -> Option<String> {
    let module = module.trim();
    if !module.starts_with('.') {
        return None;
    }
    let base = importer.rsplit_once('/').map_or("", |(dir, _)| dir);
    let joined = if base.is_empty() {
        module.to_owned()
    } else {
        format!("{base}/{module}")
    };
    let target = normpath(&joined);
    if target.starts_with("..") {
        return None;
    }
    let mut candidates = vec![target.clone()];
    let (stem, ext) = splitext(&target);
    if matches!(ext, ".js" | ".mjs" | ".cjs") {
        candidates.push(format!("{stem}.ts"));
        candidates.push(format!("{stem}.tsx"));
    }
    candidates.extend(JS_SOURCE_SUFFIXES.iter().map(|s| format!("{target}{s}")));
    candidates.extend(
        JS_SOURCE_SUFFIXES
            .iter()
            .map(|s| format!("{target}/index{s}")),
    );
    candidates.into_iter().find(|c| files.contains(c))
}

/// Indexes over one repo's definitions, built once per `resolve_graph` call.
struct Resolver {
    languages: HashMap<String, String>,
    files: HashSet<String>,
    by_qname: HashMap<String, usize>,
    by_name: HashMap<String, Vec<usize>>,
    defs_by_file: HashMap<String, HashMap<String, Vec<usize>>>,
    methods_by_class: HashMap<(String, String, String), Vec<usize>>,
    suffix_matches: HashMap<String, Vec<usize>>,
    aliases: HashMap<String, (AliasMap, Vec<String>)>,
    defs: Vec<DefRow>,
    module_cache: HashMap<(String, String), Option<String>>,
    target_cache: HashMap<(String, String, bool, String), Option<usize>>,
}

/// Calling node of an unresolved CALLS edge, used for `self.method` lookups.
struct Caller {
    kind: String,
    qname: String,
    path: String,
}

fn pick_one(pool: &[usize], defs: &[DefRow], prefer: &[&str]) -> Option<usize> {
    let preferred: Vec<usize> = pool
        .iter()
        .copied()
        .filter(|&i| prefer.contains(&defs[i].kind.as_str()))
        .collect();
    let pool = if preferred.is_empty() {
        pool.to_vec()
    } else {
        preferred
    };
    (pool.len() == 1).then(|| pool[0])
}

impl Resolver {
    fn resolve_module(&mut self, module: &str, importer: &str) -> Option<String> {
        let key = (module.to_owned(), importer.to_owned());
        if let Some(hit) = self.module_cache.get(&key) {
            return hit.clone();
        }
        let found = match self.languages.get(importer).map(String::as_str) {
            Some("python") => resolve_python_module(module, importer, &self.files),
            Some("typescript" | "javascript") => resolve_js_module(module, importer, &self.files),
            _ => None,
        };
        self.module_cache.insert(key, found.clone());
        found
    }

    fn pick_def(&self, path: Option<String>, name: &str, prefer: &[&str]) -> Option<usize> {
        let path = path?;
        if name.is_empty() {
            return None;
        }
        let rows = self.defs_by_file.get(&path)?.get(name)?;
        pick_one(rows, &self.defs, prefer)
    }

    fn resolve_via_imports(
        &mut self,
        caller_path: &str,
        dotted: &str,
        prefer: &[&str],
    ) -> Option<usize> {
        let (alias_map, stars) = self.aliases.get(caller_path)?.clone();
        if alias_map.is_empty() && stars.is_empty() {
            return None;
        }
        let segments: Vec<&str> = dotted.split('.').collect();
        for index in (1..=segments.len()).rev() {
            let prefix = segments[..index].join(".");
            let Some(entries) = alias_map.get(&prefix) else {
                continue;
            };
            let remainder = &segments[index..];
            let mut found: Vec<usize> = Vec::new();
            for (module, name) in entries {
                let row = match name.as_deref() {
                    None | Some("") => match remainder.first() {
                        Some(first) => {
                            let path = self.resolve_module(module, caller_path);
                            self.pick_def(path, first, prefer)
                        }
                        None => None,
                    },
                    Some("default") => {
                        let symbol = remainder.first().copied().unwrap_or(prefix.as_str());
                        let path = self.resolve_module(module, caller_path);
                        self.pick_def(path, symbol, prefer)
                    }
                    Some(import_name) => match remainder.first() {
                        Some(first) => {
                            let sub = join_module_name(module, import_name);
                            let path = self.resolve_module(&sub, caller_path);
                            self.pick_def(path, first, prefer).or_else(|| {
                                let path = self.resolve_module(module, caller_path);
                                self.pick_def(path, first, prefer)
                            })
                        }
                        None => {
                            let path = self.resolve_module(module, caller_path);
                            self.pick_def(path, import_name, prefer)
                        }
                    },
                };
                if let Some(row) = row
                    && !found.iter().any(|&f| self.defs[f].id == self.defs[row].id)
                {
                    found.push(row);
                }
            }
            if found.len() == 1 {
                return Some(found[0]);
            }
            if !found.is_empty() {
                return None;
            }
        }
        if segments.len() == 1 && !stars.is_empty() {
            let mut star_found: Vec<usize> = Vec::new();
            for module in &stars {
                let path = self.resolve_module(module, caller_path);
                if let Some(row) = self.pick_def(path, segments[0], prefer)
                    && !star_found
                        .iter()
                        .any(|&f| self.defs[f].id == self.defs[row].id)
                {
                    star_found.push(row);
                }
            }
            if star_found.len() == 1 {
                return Some(star_found[0]);
            }
        }
        None
    }

    fn resolve_self_method(
        &self,
        caller: Option<&Caller>,
        dotted: &str,
        prefer: &[&str],
    ) -> Option<usize> {
        let caller = caller?;
        let segments: Vec<&str> = dotted.split('.').collect();
        if segments.len() != 2 || !SELF_NAMES.contains(&segments[0]) {
            return None;
        }
        if caller.kind != "Method" || !caller.qname.contains('.') {
            return None;
        }
        let class_qname = caller.qname.rsplit_once('.').map_or("", |(head, _)| head);
        let key = (
            caller.path.clone(),
            class_qname.to_owned(),
            segments[1].to_owned(),
        );
        pick_one(self.methods_by_class.get(&key)?, &self.defs, prefer)
    }

    fn resolve_target(
        &mut self,
        caller_path: &str,
        dotted: &str,
        prefer: &[&str],
        caller: Option<&Caller>,
    ) -> Option<usize> {
        let dotted = dotted.trim();
        if dotted.is_empty() {
            return None;
        }
        let first = dotted.split('.').next().unwrap_or_default();
        let is_self = SELF_NAMES.contains(&first);
        let context = match caller {
            Some(caller) if is_self => caller.qname.clone(),
            _ => String::new(),
        };
        let key = (
            caller_path.to_owned(),
            dotted.to_owned(),
            prefer == CLASS_KINDS,
            context,
        );
        if let Some(hit) = self.target_cache.get(&key) {
            return *hit;
        }
        let mut row = self.by_qname.get(dotted).copied();
        if row.is_none() {
            row = self.resolve_self_method(caller, dotted, prefer);
        }
        if row.is_none() && !is_self {
            row = self.resolve_via_imports(caller_path, dotted, prefer);
        }
        if row.is_none() && !is_self && dotted.contains('.') {
            row = match self
                .suffix_matches
                .get(&format!(".{dotted}"))
                .map(Vec::as_slice)
            {
                Some([only]) => Some(*only),
                _ => None,
            };
        }
        if row.is_none() && !is_self {
            let simple = dotted.rsplit('.').next().unwrap_or(dotted);
            if let Some(matches) = self.by_name.get(simple) {
                row = pick_one(matches, &self.defs, prefer);
            }
        }
        self.target_cache.insert(key, row);
        row
    }
}

/// File node id per path, and import bindings per path in storage order.
struct RepoFacts {
    file_nodes: HashMap<String, String>,
    imports_by_path: Vec<(String, Vec<ImportBinding>)>,
}

fn load_resolver(conn: &Connection, repo_id: &str) -> Result<(Resolver, RepoFacts)> {
    let languages: HashMap<String, String> = conn
        .prepare("SELECT path, language FROM files WHERE repo_id = ?")?
        .query_map([repo_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut file_nodes = HashMap::new();
    let mut imports_by_path = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id, path, extra_json FROM nodes WHERE repo_id = ? AND kind = 'File' ORDER BY rowid",
        )?;
        let rows = stmt.query_map([repo_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        })?;
        for row in rows {
            let (id, path, extra) = row?;
            if let Some(bindings) = parse_bindings(&decode_extra(&extra)) {
                // A later row for the same path replaces the earlier one, like a dict.
                imports_by_path.retain(|(p, _): &(String, Vec<ImportBinding>)| *p != path);
                imports_by_path.push((path.clone(), bindings));
            }
            file_nodes.insert(path, id);
        }
    }
    let defs: Vec<DefRow> = conn
        .prepare(
            "SELECT id, kind, name, qname, path FROM nodes
             WHERE repo_id = ? AND kind IN ('Function', 'Method', 'Class') ORDER BY rowid",
        )?
        .query_map([repo_id], |row| {
            Ok(DefRow {
                id: row.get(0)?,
                kind: row.get(1)?,
                name: row.get(2)?,
                qname: row.get(3)?,
                path: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut resolver = Resolver {
        files: languages.keys().cloned().collect(),
        languages,
        by_qname: HashMap::new(),
        by_name: HashMap::new(),
        defs_by_file: HashMap::new(),
        methods_by_class: HashMap::new(),
        suffix_matches: HashMap::new(),
        aliases: imports_by_path
            .iter()
            .map(|(path, bindings)| (path.clone(), build_import_aliases(bindings)))
            .collect(),
        defs: Vec::new(),
        module_cache: HashMap::new(),
        target_cache: HashMap::new(),
    };
    for (index, row) in defs.iter().enumerate() {
        resolver.by_qname.insert(row.qname.clone(), index);
        resolver
            .by_name
            .entry(row.name.clone())
            .or_default()
            .push(index);
        resolver
            .defs_by_file
            .entry(row.path.clone())
            .or_default()
            .entry(row.name.clone())
            .or_default()
            .push(index);
        if row.kind == "Method"
            && let Some((class_qname, _)) = row.qname.rsplit_once('.')
        {
            resolver
                .methods_by_class
                .entry((row.path.clone(), class_qname.to_owned(), row.name.clone()))
                .or_default()
                .push(index);
        }
    }
    for (index, row) in defs.iter().enumerate() {
        let parts: Vec<&str> = row.qname.split('.').collect();
        for start in 1..parts.len() {
            resolver
                .suffix_matches
                .entry(format!(".{}", parts[start..].join(".")))
                .or_default()
                .push(index);
        }
    }
    resolver.defs = defs;
    Ok((
        resolver,
        RepoFacts {
            file_nodes,
            imports_by_path,
        },
    ))
}

/// Rebuild all `confidence='resolved'` edges for a repo from the parse facts.
pub fn resolve_graph(conn: &mut Connection, repo_id: &str) -> Result<ResolveCounts> {
    let (
        mut resolver,
        RepoFacts {
            file_nodes,
            imports_by_path,
        },
    ) = load_resolver(conn, repo_id)?;
    let mut counts = ResolveCounts::default();
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM edges WHERE repo_id = ? AND confidence = 'resolved'",
        [repo_id],
    )?;
    let insert_sql = |kind: &str| {
        format!(
            "INSERT OR IGNORE INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)
             VALUES (?, ?, ?, ?, '{kind}', 'resolved', ?)"
        )
    };
    {
        let mut insert = tx.prepare(&insert_sql("IMPORTS"))?;
        for (path, bindings) in &imports_by_path {
            let Some(src_id) = file_nodes.get(path) else {
                continue;
            };
            for binding in bindings {
                if binding.module.is_empty() {
                    continue;
                }
                let mut detail = binding.module.clone();
                let mut target_path = None;
                if let Some(name) = binding.name.as_deref()
                    && !matches!(name, "" | "*" | "default")
                {
                    let joined = join_module_name(&binding.module, name);
                    target_path = resolver.resolve_module(&joined, path);
                    if target_path.is_some() {
                        detail = joined;
                    }
                }
                if target_path.is_none() {
                    target_path = resolver.resolve_module(&binding.module, path);
                }
                let Some(target_path) = target_path.filter(|t| t != path) else {
                    continue;
                };
                let Some(dst_id) = file_nodes.get(&target_path) else {
                    continue;
                };
                counts.imports +=
                    insert.execute(params![repo_id, path, src_id, dst_id, detail])? as u64;
            }
        }
    }
    let calls: Vec<(String, String, String, Caller)> = tx
        .prepare(
            "SELECT e.owner_path, e.src, COALESCE(NULLIF(e.detail, ''), target.name, ''),
                    caller.kind, caller.qname, caller.path
             FROM edges e
             JOIN nodes target ON target.repo_id = e.repo_id AND target.id = e.dst
             JOIN nodes caller ON caller.repo_id = e.repo_id AND caller.id = e.src
             WHERE e.repo_id = ? AND e.kind = 'CALLS' AND target.kind = 'ExternalSymbol'",
        )?
        .query_map([repo_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get::<_, String>(2)?.trim().to_owned(),
                Caller {
                    kind: row.get(3)?,
                    qname: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    path: row.get(5)?,
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    {
        let mut insert = tx.prepare(&insert_sql("CALLS"))?;
        for (owner, src, call_name, caller) in &calls {
            let Some(row) = resolver.resolve_target(owner, call_name, CALLABLE_KINDS, Some(caller))
            else {
                continue;
            };
            let dst = &resolver.defs[row].id;
            if dst == src {
                continue;
            }
            counts.calls += insert.execute(params![repo_id, owner, src, dst, call_name])? as u64;
        }
    }
    let extends: Vec<(String, String, String)> = tx
        .prepare(
            "SELECT e.owner_path, e.src, COALESCE(NULLIF(e.detail, ''), target.name, '')
             FROM edges e
             JOIN nodes target ON target.repo_id = e.repo_id AND target.id = e.dst
             WHERE e.repo_id = ? AND e.kind = 'EXTENDS' AND target.kind = 'ExternalSymbol'",
        )?
        .query_map([repo_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get::<_, String>(2)?.trim().to_owned(),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    {
        let mut insert = tx.prepare(&insert_sql("EXTENDS"))?;
        for (owner, src, base) in &extends {
            let Some(row) = resolver.resolve_target(owner, base, CLASS_KINDS, None) else {
                continue;
            };
            let dst = &resolver.defs[row].id;
            if dst == src {
                continue;
            }
            counts.extends += insert.execute(params![repo_id, owner, src, dst, base])? as u64;
        }
    }
    tx.commit()?;
    Ok(counts)
}

fn confidence_rank(confidence: &str) -> u8 {
    match confidence {
        "exact" => 0,
        "strong" => 1,
        "resolved" => 2,
        _ => 3,
    }
}

pub fn weakest_confidence(left: &str, right: &str) -> &'static str {
    match confidence_rank(left).max(confidence_rank(right)) {
        0 => "exact",
        1 => "strong",
        2 => "resolved",
        _ => "heuristic",
    }
}

fn edge_scope(edge_kind: &str) -> &'static str {
    if edge_kind == "IMPORTS" {
        "conservative_file"
    } else {
        "precise_symbol"
    }
}

/// Traversal direction of `blast_radius`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Both,
}

impl Direction {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            "both" => Ok(Self::Both),
            _ => Err(Error::invalid("direction must be 'up', 'down', or 'both'")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Both => "both",
        }
    }
}

struct WalkRow {
    edge_kind: String,
    edge_confidence: String,
    edge_detail: String,
    id: String,
    kind: String,
    qname: String,
    compact: Value,
}

fn filter_shadowed_rows(rows: Vec<WalkRow>) -> Vec<WalkRow> {
    let resolved: HashSet<String> = rows
        .iter()
        .filter(|r| r.edge_kind == "CALLS" && !crate::model::is_external(&r.kind))
        .map(|r| r.edge_detail.trim().to_owned())
        .filter(|d| !d.is_empty())
        .collect();
    if resolved.is_empty() {
        return rows;
    }
    rows.into_iter()
        .filter(|r| {
            !(r.edge_kind == "CALLS"
                && r.kind == "ExternalSymbol"
                && resolved.contains(r.edge_detail.trim()))
        })
        .collect()
}

fn walk(
    conn: &Connection,
    repo_id: &str,
    seeds: &[(String, String)],
    sql: &str,
    direction: &str,
    depth: Option<i64>,
    max_nodes: usize,
) -> Result<(Vec<Value>, bool)> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut visited: HashSet<String> = seeds.iter().map(|(id, _)| id.clone()).collect();
    let mut queue: VecDeque<(String, i64, &'static str, String)> = seeds
        .iter()
        .map(|(id, qname)| (id.clone(), 0, "exact", qname.clone()))
        .collect();
    let mut entries = Vec::new();
    while let Some((current, distance, confidence, current_qname)) = queue.pop_front() {
        if depth.is_some_and(|limit| distance >= limit) {
            continue;
        }
        let rows = stmt
            .query_map(params![repo_id, current, max_nodes as i64 + 1], |row| {
                let start = row.get::<_, Option<i64>>(8)?.unwrap_or(0).max(1);
                let end = row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(start);
                let text = |i: usize| -> rusqlite::Result<String> {
                    Ok(row.get::<_, Option<String>>(i)?.unwrap_or_default())
                };
                Ok(WalkRow {
                    edge_kind: text(0)?,
                    edge_confidence: text(1)?,
                    edge_detail: text(2)?,
                    id: text(3)?,
                    kind: text(4)?,
                    qname: text(6)?,
                    compact: json!({
                        "id": text(3)?, "kind": text(4)?, "name": text(5)?,
                        "qname": text(6)?, "path": text(7)?, "lines": [start, end],
                    }),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for row in filter_shadowed_rows(rows) {
            if !visited.insert(row.id.clone()) {
                continue;
            }
            let combined = weakest_confidence(confidence, &row.edge_confidence);
            if entries.len() >= max_nodes {
                return Ok((entries, true));
            }
            entries.push(json!({
                "node": row.compact,
                "distance": distance + 1,
                "direction": direction,
                "via": row.edge_kind,
                "via_qname": current_qname,
                "detail": row.edge_detail,
                "confidence": combined,
                "scope": edge_scope(&row.edge_kind),
            }));
            if EXPANDABLE_NODE_KINDS.contains(&row.kind.as_str()) {
                queue.push_back((row.id, distance + 1, combined, row.qname));
            }
        }
    }
    Ok((entries, false))
}

/// Bounded BFS over dependency edges around a node. Upstream nodes depend on the target,
/// downstream nodes are what the target depends on. Returns an empty object for unknown ids.
pub fn blast_radius(
    conn: &Connection,
    node_id: &str,
    depth: Option<i64>,
    max_nodes: i64,
    direction: Direction,
) -> Result<Value> {
    let Some(target) = storage::get_node(conn, node_id)? else {
        return Ok(Value::Object(Map::new()));
    };
    let repo_id = target.repo_id.clone();
    let depth_limit = depth.filter(|d| *d > 0);
    let max_nodes = max_nodes.clamp(1, 10_000) as usize;
    let qname = target.qname.clone();
    let mut seeds_up = vec![(node_id.to_owned(), qname.clone())];
    let mut seeds_down = seeds_up.clone();
    let push_seed = |seeds: &mut Vec<(String, String)>, id: String| match seeds
        .iter_mut()
        .find(|(seed, _)| *seed == id)
    {
        Some(existing) => existing.1 = qname.clone(),
        None => seeds.push((id, qname.clone())),
    };
    if target.kind == "File" {
        let mut stmt = conn
            .prepare("SELECT dst FROM edges WHERE repo_id = ? AND src = ? AND kind = 'DEFINES'")?;
        let dsts = stmt
            .query_map(params![repo_id, node_id], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for dst in dsts {
            push_seed(&mut seeds_up, dst.clone());
            push_seed(&mut seeds_down, dst);
        }
    } else if !target.path.is_empty() {
        let file_id: Option<String> = conn
            .query_row(
                "SELECT id FROM nodes WHERE repo_id = ? AND kind = 'File' AND path = ? LIMIT 1",
                params![repo_id, target.path],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|err| match err {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        if let Some(file_id) = file_id {
            push_seed(&mut seeds_up, file_id);
        }
    }
    let up_sql = format!(
        "{WALK_SELECT}JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src WHERE e.repo_id = ? AND e.dst = ?{WALK_ORDER}"
    );
    let down_sql = format!(
        "{WALK_SELECT}JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst WHERE e.repo_id = ? AND e.src = ?{WALK_ORDER}"
    );
    let (upstream, up_truncated) = if direction != Direction::Down {
        walk(
            conn,
            &repo_id,
            &seeds_up,
            &up_sql,
            "up",
            depth_limit,
            max_nodes,
        )?
    } else {
        (Vec::new(), false)
    };
    let (downstream, down_truncated) = if direction != Direction::Up {
        walk(
            conn,
            &repo_id,
            &seeds_down,
            &down_sql,
            "down",
            depth_limit,
            max_nodes,
        )?
    } else {
        (Vec::new(), false)
    };
    let mut files: BTreeMap<String, i64> = BTreeMap::new();
    if !target.path.is_empty() && INTERNAL_NODE_KINDS.contains(&target.kind.as_str()) {
        files.insert(target.path.clone(), 0);
    }
    for entry in upstream.iter().chain(&downstream) {
        let node = &entry["node"];
        let kind = node["kind"].as_str().unwrap_or_default();
        let path = node["path"].as_str().unwrap_or_default();
        if !INTERNAL_NODE_KINDS.contains(&kind) || path.is_empty() {
            continue;
        }
        let distance = entry["distance"].as_i64().unwrap_or_default();
        files
            .entry(path.to_owned())
            .and_modify(|d| *d = (*d).min(distance))
            .or_insert(distance);
    }
    let mut file_list: Vec<(String, i64)> = files.into_iter().collect();
    file_list.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let file_entries: Vec<Value> = file_list
        .into_iter()
        .map(|(path, distance)| json!({"path": path, "distance": distance}))
        .collect();
    let entrypoints: Vec<Value> = upstream
        .iter()
        .filter(|e| e["node"]["kind"] == "Route")
        .cloned()
        .collect();
    let conservative = upstream
        .iter()
        .chain(&downstream)
        .filter(|e| e["scope"] == "conservative_file")
        .count();
    Ok(json!({
        "target_id": node_id,
        "depth": depth_limit,
        "depth_label": depth_limit.map_or_else(|| "all".to_owned(), |d| d.to_string()),
        "direction": direction.as_str(),
        "upstream": upstream,
        "downstream": downstream,
        "files": file_entries,
        "entrypoints": entrypoints,
        "truncated": up_truncated || down_truncated,
        "stats": {
            "upstream": upstream.len(),
            "downstream": downstream.len(),
            "files": file_entries.len(),
            "entrypoints": entrypoints.len(),
            "conservative_file_expansions": conservative,
        },
    }))
}

pub fn filter_shadowed_neighbors(items: Vec<Neighbor>) -> Vec<Neighbor> {
    let resolved: HashSet<String> = items
        .iter()
        .filter(|item| {
            item.edge.kind == "CALLS"
                && !item
                    .node
                    .as_ref()
                    .is_some_and(|n| crate::model::is_external(&n.kind))
        })
        .map(|item| item.edge.detail.trim().to_owned())
        .filter(|d| !d.is_empty())
        .collect();
    if resolved.is_empty() {
        return items;
    }
    items
        .into_iter()
        .filter(|item| {
            !(item.edge.kind == "CALLS"
                && item
                    .node
                    .as_ref()
                    .is_some_and(|n| n.kind == "ExternalSymbol")
                && resolved.contains(item.edge.detail.trim()))
        })
        .collect()
}

/// Node id first, then exact symbol lookup, then search.
pub fn impact_targets(
    conn: &Connection,
    target: &str,
    repo_id: Option<&str>,
    limit: i64,
) -> Result<Vec<NodeRow>> {
    if let Some(node) = storage::get_node(conn, target)?
        && repo_id.is_none_or(|repo| node.repo_id == repo)
    {
        return Ok(vec![node]);
    }
    let symbols = storage::find_symbol(conn, target, repo_id, limit)?;
    if !symbols.is_empty() {
        return Ok(symbols);
    }
    storage::search_nodes(conn, target, repo_id, limit)
}

pub fn impact_report(
    conn: &Connection,
    target: &NodeRow,
    neighbor_limit: i64,
    depth: Option<i64>,
    max_nodes: i64,
    direction: Direction,
) -> Result<Value> {
    let neighbors = storage::get_neighbors(conn, &target.id, neighbor_limit)?;
    let incoming = filter_shadowed_neighbors(neighbors.incoming);
    let outgoing = filter_shadowed_neighbors(neighbors.outgoing);
    let callers: Vec<&Neighbor> = incoming.iter().filter(|i| i.edge.kind == "CALLS").collect();
    let callees: Vec<&Neighbor> = outgoing.iter().filter(|i| i.edge.kind == "CALLS").collect();
    let radius = blast_radius(conn, &target.id, depth, max_nodes, direction)?;
    let files: Vec<Value> = radius["files"]
        .as_array()
        .map(|items| items.iter().map(|e| e["path"].clone()).collect())
        .unwrap_or_default();
    let stat = |key: &str| radius["stats"][key].as_u64().unwrap_or(0);
    let summary = json!({
        "callers": callers.len(),
        "callees": callees.len(),
        "incoming": incoming.len(),
        "outgoing": outgoing.len(),
        "files": files.len(),
        "upstream": stat("upstream"),
        "downstream": stat("downstream"),
        "entrypoints": stat("entrypoints"),
        "depth": radius.get("depth").cloned().unwrap_or_else(|| json!(depth)),
        "depth_label": radius.get("depth_label").cloned().unwrap_or_else(|| json!("all")),
        "direction": direction.as_str(),
        "truncated": radius.get("truncated").cloned().unwrap_or(json!(false)),
    });
    Ok(json!({
        "target": target,
        "summary": summary,
        "files": files,
        "callers": callers,
        "callees": callees,
        "blast_radius": radius,
        "incoming": incoming,
        "outgoing": outgoing,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|p| (*p).to_owned()).collect()
    }

    #[test]
    fn python_relative_and_suffix_modules() {
        let set = files(&["pkg/a.py", "pkg/sub/__init__.py", "src/lib/util.py"]);
        assert_eq!(
            resolve_python_module(".a", "pkg/b.py", &set).as_deref(),
            Some("pkg/a.py")
        );
        assert_eq!(
            resolve_python_module(".sub", "pkg/b.py", &set).as_deref(),
            Some("pkg/sub/__init__.py")
        );
        assert_eq!(
            resolve_python_module("lib.util", "x.py", &set).as_deref(),
            Some("src/lib/util.py")
        );
        assert_eq!(resolve_python_module("...a", "pkg/b.py", &set), None);
    }

    #[test]
    fn js_modules_resolve_extensions() {
        let set = files(&["web/api.ts", "web/lib/index.tsx"]);
        assert_eq!(
            resolve_js_module("./api.js", "web/app.ts", &set).as_deref(),
            Some("web/api.ts")
        );
        assert_eq!(
            resolve_js_module("./lib", "web/app.ts", &set).as_deref(),
            Some("web/lib/index.tsx")
        );
        assert_eq!(resolve_js_module("react", "web/app.ts", &set), None);
        assert_eq!(resolve_js_module("../../x", "web/app.ts", &set), None);
    }

    #[test]
    fn weakest_confidence_picks_lowest() {
        assert_eq!(weakest_confidence("exact", "resolved"), "resolved");
        assert_eq!(weakest_confidence("strong", "bogus"), "heuristic");
    }
}
