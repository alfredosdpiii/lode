//! Context packs for agents (`context.py`): weighted BM25 hits re-ranked by identifier
//! specificity, graph proximity to `from_path`, and directory proximity.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;
use rusqlite::Connection;
use rusqlite::types::Value as SqlValue;
use serde_json::{Value, json};

use crate::Result;
use crate::model::is_external;
use crate::storage::{self, NODE_COLUMNS, NodeRow};
use crate::text::{camel_parts, char_len, py_repr_len, static_regex, take_chars};

static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"[A-Za-z0-9_]+"));

const CONTEXT_STOPWORDS: &[&str] = &[
    "self", "cls", "none", "true", "false", "return", "def", "class", "import", "from", "as",
    "with", "for", "while", "try", "except", "finally", "pass", "if", "else", "elif", "in", "is",
    "and", "or", "not", "the", "to", "of", "a", "an", "py", "ts", "js",
];
const MAX_QUERY_TERMS: usize = 48;

/// Re-orders ranked hits with a signal the lexical ranker lacks (see `rerank.rs`).
pub trait Rerank: std::fmt::Debug {
    fn rerank(&self, conn: &Connection, query: &str, hits: Vec<NodeRow>) -> Result<Vec<NodeRow>>;
}

/// Options of `build_context_pack`, with the Python defaults.
#[derive(Debug, Clone)]
pub struct ContextOptions<'a> {
    pub repo_path: Option<&'a str>,
    pub budget: i64,
    pub limit: usize,
    pub include_related: bool,
    pub from_path: Option<&'a str>,
    pub exclude_paths: HashSet<String>,
    pub reranker: Option<&'a dyn Rerank>,
    /// Text the reranker embeds instead of `query`, e.g. the code just before the cursor.
    pub rerank_query: Option<&'a str>,
}

impl Default for ContextOptions<'_> {
    fn default() -> Self {
        Self {
            repo_path: None,
            budget: 6000,
            limit: 10,
            include_related: true,
            from_path: None,
            exclude_paths: HashSet::new(),
            reranker: None,
            rerank_query: None,
        }
    }
}

pub fn build_context_pack(
    conn: &Connection,
    query: &str,
    options: &ContextOptions<'_>,
) -> Result<Value> {
    let repo_id = storage::repo_filter(conn, options.repo_path)?;
    let excluded = &options.exclude_paths;
    let limit = options.limit;
    let fetch_limit = (limit * 3).max(limit + excluded.len() + 10);
    let raw_hits = storage::search_nodes(conn, query, repo_id.as_deref(), fetch_limit as i64)?;
    let (boosts, graph_nodes) = graph_context_signals(
        conn,
        repo_id.as_deref(),
        options.from_path,
        &raw_hits,
        excluded,
        limit,
    )?;
    let hits = merge_and_rank_hits(
        query,
        raw_hits,
        &boosts,
        graph_nodes,
        options.from_path,
        excluded,
        limit,
    );
    let hits = match options.reranker {
        Some(reranker) => reranker.rerank(conn, options.rerank_query.unwrap_or(query), hits)?,
        None => hits,
    };

    let mut must_read: Vec<Value> = Vec::new();
    let mut seen: HashSet<(String, i64, i64)> = HashSet::new();
    let mut remaining = options.budget.max(1000);
    for hit in &hits {
        if remaining <= 0 {
            break;
        }
        if !seen.insert((hit.path.clone(), hit.start_line, hit.end_line)) {
            continue;
        }
        let item = context_item(hit, reason_for_hit(hit, query));
        let cost = estimate_item_cost(&item);
        if cost > remaining && !must_read.is_empty() {
            continue;
        }
        remaining -= cost;
        must_read.push(Value::Object(item.into_iter().collect()));
    }

    let mut related = Vec::new();
    if options.include_related {
        let top = &hits[..hits.len().min(5)];
        let mut neighbors = compact_neighbors_for_hits(conn, top, 16)?;
        for hit in top {
            let (incoming, outgoing) = neighbors.remove(&hit.id).unwrap_or_default();
            related.push(json!({
                "node_id": hit.id,
                "qname": hit.qname,
                "incoming": incoming,
                "outgoing": outgoing,
            }));
        }
    }
    Ok(json!({
        "query": query,
        "budget": options.budget,
        "summary": summarize_hits(&hits),
        "must_read": must_read,
        "top_hits": hits.iter().map(compact_node).collect::<Vec<_>>(),
        "related": related,
        "confidence": aggregate_confidence(&hits),
        "notes": [
            "Lode ranks weighted BM25 matches with identifier specificity and graph proximity.",
            "Embeddings are optional and not required for this context pack yet.",
        ],
    }))
}

type Boosts = HashMap<String, f64>;

fn graph_context_signals(
    conn: &Connection,
    repo_id: Option<&str>,
    from_path: Option<&str>,
    raw_hits: &[NodeRow],
    excluded: &HashSet<String>,
    limit: usize,
) -> Result<(Boosts, Vec<NodeRow>)> {
    let Some(from_path) = from_path.filter(|p| !p.is_empty()) else {
        return Ok((HashMap::new(), Vec::new()));
    };
    let mut boosts: Boosts = HashMap::new();
    let mut extra = Vec::new();
    let mut seen: HashSet<String> = raw_hits.iter().map(|hit| hit.id.clone()).collect();
    for (edge_kind, confidence, node) in
        load_from_path_targets(conn, repo_id, from_path, (limit * 2).max(24))?
    {
        if node.path.is_empty() || excluded.contains(&node.path) {
            continue;
        }
        let weight = edge_boost_weight(&edge_kind, &confidence);
        let by_id = boosts.entry(node.id.clone()).or_insert(0.0);
        *by_id = by_id.max(weight);
        let by_path = boosts.entry(format!("path:{}", node.path)).or_insert(0.0);
        *by_path = by_path.max(weight * 0.75);
        if seen.insert(node.id.clone()) {
            extra.push(node);
        }
    }
    Ok((boosts, extra))
}

fn load_from_path_targets(
    conn: &Connection,
    repo_id: Option<&str>,
    from_path: &str,
    limit: usize,
) -> Result<Vec<(String, String, NodeRow)>> {
    let repo_clause = if repo_id.is_some() {
        "AND e.repo_id = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT e.kind, e.confidence, {NODE_COLUMNS}, 0.0
         FROM edges e
         JOIN nodes src ON src.repo_id = e.repo_id AND src.id = e.src
         JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
         WHERE src.path = ?
           {repo_clause}
           AND n.path != ?
           AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency', 'ParseError')
           AND e.kind IN ('CALLS', 'IMPORTS', 'EXTENDS')
         ORDER BY
           CASE e.kind WHEN 'CALLS' THEN 0 WHEN 'EXTENDS' THEN 1 ELSE 2 END,
           CASE e.confidence
             WHEN 'exact' THEN 0 WHEN 'strong' THEN 1 WHEN 'resolved' THEN 2 ELSE 3
           END,
           n.path,
           n.start_line
         LIMIT ?"
    );
    let mut args = vec![SqlValue::Text(from_path.to_owned())];
    args.extend(repo_id.map(|id| SqlValue::Text(id.to_owned())));
    args.push(SqlValue::Text(from_path.to_owned()));
    args.push(SqlValue::Integer(limit as i64));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(args), |row| {
        Ok((row.get(0)?, row.get(1)?, NodeRow::from_row(row, 2)?))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn edge_boost_weight(edge_kind: &str, confidence: &str) -> f64 {
    let kind_base = match edge_kind {
        "CALLS" | "EXTENDS" => 4.0,
        "IMPORTS" => 3.5,
        _ => 2.0,
    };
    let scale = match confidence {
        "exact" | "strong" => 1.0,
        "resolved" => 0.9,
        "heuristic" => 0.6,
        _ => 0.5,
    };
    kind_base * scale
}

/// Name, path, signature tokens (expanded), and exact name+signature tokens.
struct Fields {
    name: HashSet<String>,
    path: HashSet<String>,
    signature: HashSet<String>,
    exact: HashSet<String>,
}

fn merge_and_rank_hits(
    query: &str,
    raw_hits: Vec<NodeRow>,
    boosts: &Boosts,
    graph_nodes: Vec<NodeRow>,
    from_path: Option<&str>,
    excluded: &HashSet<String>,
    limit: usize,
) -> Vec<NodeRow> {
    let (weights, exact_terms) = query_term_weights(query);
    let from_parts = from_path
        .filter(|p| !p.is_empty())
        .map(path_dir_parts)
        .unwrap_or_default();
    let mut seen = HashSet::new();
    let merged: Vec<NodeRow> = raw_hits
        .into_iter()
        .chain(graph_nodes)
        .filter(|node| {
            !node.id.is_empty() && !excluded.contains(&node.path) && seen.insert(node.id.clone())
        })
        .collect();
    if merged.is_empty() {
        return Vec::new();
    }
    let fields: Vec<Fields> = merged.iter().map(extract_node_fields).collect();
    let idf = candidate_pool_idf(&fields);
    let mut scored: Vec<((u8, f64, String), NodeRow)> = merged
        .into_iter()
        .zip(&fields)
        .map(|(node, fields)| {
            let rank = composite_rank(
                &node,
                fields,
                &weights,
                &exact_terms,
                &idf,
                boosts,
                &from_parts,
            );
            (rank, node)
        })
        .collect();
    scored.sort_by(|(a, _), (b, _)| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    diversify_by_path(scored.into_iter().map(|(_, node)| node).collect(), limit)
}

fn diversify_by_path(nodes: Vec<NodeRow>, limit: usize) -> Vec<NodeRow> {
    let mut primary = Vec::new();
    let mut overflow = Vec::new();
    let mut per_path: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        let count = per_path.entry(node.path.clone()).or_insert(0);
        if *count < 2 {
            *count += 1;
            primary.push(node);
        } else {
            overflow.push(node);
        }
    }
    primary.extend(overflow);
    primary.truncate(limit);
    primary
}

fn extract_node_fields(node: &NodeRow) -> Fields {
    let combined = format!("{} {}", node.name, node.signature);
    Fields {
        name: field_token_set(&node.name),
        path: field_token_set(&node.path),
        signature: field_token_set(&node.signature),
        exact: TOKEN_RE
            .find_iter(&combined)
            .map(|m| m.as_str())
            .filter(|tok| tok.len() > 1)
            .map(str::to_lowercase)
            .collect(),
    }
}

fn candidate_pool_idf(fields: &[Fields]) -> HashMap<String, f64> {
    let total = fields.len();
    if total <= 1 {
        return HashMap::new();
    }
    let mut doc_freq: HashMap<&str, usize> = HashMap::new();
    for field in fields {
        let union: HashSet<&str> = field
            .name
            .iter()
            .chain(&field.path)
            .chain(&field.signature)
            .map(String::as_str)
            .collect();
        for tok in union {
            *doc_freq.entry(tok).or_insert(0) += 1;
        }
    }
    doc_freq
        .into_iter()
        .map(|(tok, df)| {
            (
                tok.to_owned(),
                ((total as f64 + 1.0) / (df as f64 + 0.5)).ln() + 0.5,
            )
        })
        .collect()
}

fn composite_rank(
    node: &NodeRow,
    fields: &Fields,
    weights: &IndexMap<String, f64>,
    exact_terms: &IndexMap<String, f64>,
    idf: &HashMap<String, f64>,
    boosts: &Boosts,
    from_parts: &[String],
) -> (u8, f64, String) {
    let penalty = u8::from(is_external(&node.kind));
    let lookup = |key: &str| boosts.get(key).copied().unwrap_or(0.0);
    let mut boost = lookup(&node.id) + lookup(&format!("path:{}", node.path));
    boost += lexical_specificity_boost(fields, weights, exact_terms, idf);
    if !from_parts.is_empty() && !node.path.is_empty() {
        boost += path_proximity_boost(from_parts, &path_dir_parts(&node.path));
    }
    (penalty, node.rank - boost, node.qname.clone())
}

/// Per-term query weights decaying with position, and the weights of whole query words.
fn query_term_weights(query: &str) -> (IndexMap<String, f64>, IndexMap<String, f64>) {
    let mut weights: IndexMap<String, f64> = IndexMap::new();
    let mut exact: IndexMap<String, f64> = IndexMap::new();
    let mut pos = 0usize;
    for raw in TOKEN_RE.find_iter(query).map(|m| m.as_str()) {
        let low_raw = raw.to_lowercase();
        if !CONTEXT_STOPWORDS.contains(&low_raw.as_str()) && char_len(&low_raw) > 1 {
            let weight = 1.0 / (1.0 + 0.22 * pos as f64);
            let entry = exact.entry(low_raw).or_insert(0.0);
            *entry = entry.max(weight);
        }
        for token in expand_token(raw) {
            if CONTEXT_STOPWORDS.contains(&token.as_str()) || char_len(&token) <= 1 {
                continue;
            }
            let weight = 1.0 / (1.0 + 0.25 * pos as f64);
            if weight > weights.get(&token).copied().unwrap_or(0.0) {
                weights.insert(token, weight);
            }
            pos += 1;
            if pos >= MAX_QUERY_TERMS {
                return (weights, exact);
            }
        }
    }
    (weights, exact)
}

fn expand_token(raw: &str) -> Vec<String> {
    let low_raw = raw.to_lowercase();
    let mut out = vec![low_raw.clone()];
    for chunk in raw.split('_').filter(|chunk| !chunk.is_empty()) {
        out.extend(
            camel_parts(chunk)
                .into_iter()
                .map(str::to_lowercase)
                .filter(|part| *part != low_raw),
        );
    }
    out
}

fn field_token_set(value: &str) -> HashSet<String> {
    TOKEN_RE
        .find_iter(value)
        .flat_map(|m| expand_token(m.as_str()))
        .collect()
}

/// IDF for a query term no candidate contains (`term_idf.get(term, 1.5)` in Python).
const UNSEEN_TERM_IDF: f64 = 1.5;

fn lexical_specificity_boost(
    fields: &Fields,
    weights: &IndexMap<String, f64>,
    exact_terms: &IndexMap<String, f64>,
    idf: &HashMap<String, f64>,
) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let idf_of = |term: &str| idf.get(term).copied().unwrap_or(UNSEEN_TERM_IDF);
    let mut boost = 0.0;
    for (term, weight) in exact_terms {
        if fields.exact.contains(term) {
            boost += 5.0 * weight * idf_of(term);
        }
    }
    for (term, weight) in weights {
        let term_idf = idf_of(term);
        if fields.name.contains(term) {
            boost += 3.5 * weight * term_idf;
        }
        if fields.path.contains(term) {
            boost += 3.0 * weight * term_idf;
        }
        if fields.signature.contains(term) {
            boost += 1.5 * weight * term_idf;
        }
    }
    boost
}

/// `PurePosixPath(path).parent.parts`.
fn path_dir_parts(path: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    if path.starts_with('/') {
        parts.push("/".to_owned());
    }
    parts.extend(
        path.split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned),
    );
    if parts.last().is_some_and(|last| last != "/") {
        parts.pop();
    }
    parts
}

fn path_proximity_boost(from_parts: &[String], cand_parts: &[String]) -> f64 {
    if from_parts.is_empty() || cand_parts.is_empty() {
        return 0.0;
    }
    let shared = from_parts
        .iter()
        .zip(cand_parts)
        .take_while(|(a, b)| a == b)
        .count();
    if shared == 0 {
        0.0
    } else if from_parts == cand_parts {
        4.0
    } else {
        (shared as f64).min(3.0)
    }
}

fn or_unknown(value: &str) -> &str {
    if value.is_empty() { "unknown" } else { value }
}

/// Ordered like the Python dict so `estimate_item_cost` measures the same `repr`.
fn context_item(node: &NodeRow, why: &str) -> Vec<(String, Value)> {
    vec![
        ("node_id".into(), node.id.clone().into()),
        ("kind".into(), node.kind.clone().into()),
        ("name".into(), node.name.clone().into()),
        ("qname".into(), node.qname.clone().into()),
        ("path".into(), node.path.clone().into()),
        ("start_line".into(), node.start_line.into()),
        ("end_line".into(), node.end_line.into()),
        ("signature".into(), node.signature.clone().into()),
        ("doc".into(), truncate(&node.doc, 800).into()),
        ("why".into(), why.into()),
        ("confidence".into(), or_unknown(&node.confidence).into()),
    ]
}

/// `max(120, len(str(item)) // 4)` for a flat dict of strings and ints.
fn estimate_item_cost(item: &[(String, Value)]) -> i64 {
    let body: usize = item
        .iter()
        .map(|(key, value)| {
            let value_len = match value {
                Value::String(text) => py_repr_len(text),
                other => other.to_string().len(),
            };
            py_repr_len(key) + 2 + value_len
        })
        .sum();
    let len = 2 + body + 2 * item.len().saturating_sub(1);
    (len as i64 / 4).max(120)
}

fn compact_node(node: &NodeRow) -> Value {
    json!({
        "id": node.id,
        "kind": node.kind,
        "qname": node.qname,
        "path": node.path,
        "lines": [node.start_line, node.end_line],
        "confidence": or_unknown(&node.confidence),
    })
}

type CompactNeighbors = (Vec<Value>, Vec<Value>);

fn compact_neighbors_for_hits(
    conn: &Connection,
    hits: &[NodeRow],
    limit: usize,
) -> Result<HashMap<String, CompactNeighbors>> {
    let mut related: HashMap<String, CompactNeighbors> = hits
        .iter()
        .filter(|hit| !hit.id.is_empty() && !hit.repo_id.is_empty())
        .map(|hit| (hit.id.clone(), (Vec::new(), Vec::new())))
        .collect();
    let mut ids_by_repo: IndexMap<String, Vec<String>> = IndexMap::new();
    for hit in hits.iter().filter(|hit| related.contains_key(&hit.id)) {
        let ids = ids_by_repo.entry(hit.repo_id.clone()).or_default();
        if !ids.contains(&hit.id) {
            ids.push(hit.id.clone());
        }
    }
    for (repo_id, ids) in &ids_by_repo {
        load_compact_neighbor_rows(conn, &mut related, repo_id, ids, true, limit)?;
        load_compact_neighbor_rows(conn, &mut related, repo_id, ids, false, limit)?;
    }
    Ok(related)
}

fn load_compact_neighbor_rows(
    conn: &Connection,
    related: &mut HashMap<String, CompactNeighbors>,
    repo_id: &str,
    node_ids: &[String],
    outgoing: bool,
    limit: usize,
) -> Result<()> {
    if node_ids.is_empty() {
        return Ok(());
    }
    let (center, other) = if outgoing {
        ("e.src", "e.dst")
    } else {
        ("e.dst", "e.src")
    };
    let marks = vec!["?"; node_ids.len()].join(",");
    let sql = format!(
        "SELECT {center}, e.kind, e.confidence, n.kind, n.qname, n.path
         FROM edges e
         JOIN nodes n ON n.repo_id = e.repo_id AND n.id = {other}
         WHERE e.repo_id = ? AND {center} IN ({marks})
         ORDER BY
           {center},
           CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
           CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
           n.path,
           n.start_line"
    );
    let args = std::iter::once(repo_id).chain(node_ids.iter().map(String::as_str));
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(rusqlite::params_from_iter(args))?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    while let Some(row) = rows.next()? {
        let center_id: String = row.get(0)?;
        let qname: Option<String> = row.get(4)?;
        let count = counts.entry(center_id.clone()).or_insert(0);
        let Some(qname) = qname else { continue };
        if *count >= limit {
            continue;
        }
        let item = json!({
            "edge": row.get::<_, Option<String>>(1)?,
            "qname": qname,
            "kind": row.get::<_, Option<String>>(3)?,
            "path": row.get::<_, Option<String>>(5)?,
            "confidence": row.get::<_, Option<String>>(2)?,
        });
        if let Some((incoming, outgoing_items)) = related.get_mut(&center_id) {
            if outgoing {
                outgoing_items.push(item)
            } else {
                incoming.push(item)
            }
        }
        *count += 1;
    }
    Ok(())
}

fn reason_for_hit(node: &NodeRow, query: &str) -> &'static str {
    let query = query.to_lowercase();
    if node.name.to_lowercase().contains(&query) {
        "name match"
    } else if node.qname.to_lowercase().contains(&query) {
        "qualified-name match"
    } else if node.path.to_lowercase().contains(&query) {
        "path match"
    } else {
        "full-text match"
    }
}

fn summarize_hits(hits: &[NodeRow]) -> String {
    if hits.is_empty() {
        return "No indexed nodes matched the query.".to_owned();
    }
    let mut kinds: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut paths: HashSet<&str> = HashSet::new();
    for hit in hits {
        *kinds.entry(hit.kind.as_str()).or_insert(0) += 1;
        paths.insert(&hit.path);
    }
    let kind_summary: Vec<String> = kinds
        .iter()
        .map(|(kind, count)| format!("{kind}:{count}"))
        .collect();
    format!(
        "Found {} relevant nodes across {} files ({}).",
        hits.len(),
        paths.len(),
        kind_summary.join(", ")
    )
}

fn aggregate_confidence(hits: &[NodeRow]) -> &'static str {
    if hits.is_empty() {
        return "none";
    }
    let all_in = |allowed: &[&str]| {
        hits.iter()
            .all(|hit| allowed.contains(&hit.confidence.as_str()))
    };
    if all_in(&["exact"]) {
        if hits.len() == 1 { "exact" } else { "strong" }
    } else if all_in(&["exact", "strong"]) {
        "strong"
    } else {
        "mixed"
    }
}

fn truncate(value: &str, length: usize) -> String {
    if char_len(value) <= length {
        return value.to_owned();
    }
    format!("{}...", take_chars(value, length - 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_parts_match_pure_posix_path() {
        assert_eq!(path_dir_parts("a/b/c.py"), ["a", "b"]);
        assert!(path_dir_parts("c.py").is_empty());
        assert_eq!(
            path_proximity_boost(&path_dir_parts("a/b/x.py"), &path_dir_parts("a/b/y.py")),
            4.0
        );
        assert_eq!(
            path_proximity_boost(&path_dir_parts("a/b/x.py"), &path_dir_parts("a/c/y.py")),
            1.0
        );
    }

    #[test]
    fn query_weights_decay_by_position() {
        let (weights, exact) = query_term_weights("parseHTTP the x");
        assert_eq!(
            weights.keys().collect::<Vec<_>>(),
            ["parsehttp", "parse", "http"]
        );
        assert_eq!(weights["http"], 1.0 / 1.5);
        assert_eq!(exact.keys().collect::<Vec<_>>(), ["parsehttp"]);
    }

    #[test]
    fn truncate_counts_chars() {
        assert_eq!(truncate("abcdef", 5), "ab...");
        assert_eq!(truncate("abc", 5), "abc");
    }
}
