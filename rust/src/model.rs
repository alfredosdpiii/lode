use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::sha1_hex;

/// Node kinds that stand for symbols outside the indexed files.
pub const EXTERNAL_KINDS: [&str; 2] = ["ExternalSymbol", "ExternalDependency"];

pub fn is_external(kind: &str) -> bool {
    EXTERNAL_KINDS.contains(&kind)
}

/// One parsed graph node. `kind` and `confidence` stay strings because they are stored and
/// compared as SQLite text and other tools write the same database.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Node {
    pub id: String,
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
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Edge {
    pub src: String,
    pub dst: String,
    pub kind: String,
    pub confidence: String,
    pub detail: String,
}

impl Edge {
    pub fn new(src: &str, dst: &str, kind: &str, confidence: &str, detail: &str) -> Self {
        Self {
            src: src.to_owned(),
            dst: dst.to_owned(),
            kind: kind.to_owned(),
            confidence: confidence.to_owned(),
            detail: detail.to_owned(),
        }
    }

    pub fn dedupe_key(&self) -> (&str, &str, &str, &str) {
        (&self.src, &self.dst, &self.kind, &self.detail)
    }
}

/// Parse result for one file, ready to be written by `storage::replace_file_indexes`.
#[derive(Debug, Clone)]
pub struct FileIndex {
    pub path: String,
    pub abspath: String,
    pub language: String,
    pub size: u64,
    pub mtime: f64,
    pub content_hash: String,
    pub generated: bool,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// Fields of a node besides the identity parts that `make_node` hashes.
#[derive(Debug, Clone, Default)]
pub struct NodeAttrs<'a> {
    pub signature: &'a str,
    pub doc: &'a str,
    pub confidence: &'a str,
    pub extra: Map<String, Value>,
}

/// Node id is `sha1(kind \0 qname \0 path \0 start_line)[:24]`; content hash is `sha1(content)`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the identity tuple of the Python make_node"
)]
pub fn make_node(
    kind: &str,
    name: &str,
    qname: &str,
    path: &str,
    start_line: i64,
    end_line: i64,
    content: &str,
    attrs: NodeAttrs<'_>,
) -> Node {
    let key = format!("{kind}\0{qname}\0{path}\0{start_line}");
    let confidence = if attrs.confidence.is_empty() {
        "strong"
    } else {
        attrs.confidence
    };
    Node {
        id: sha1_hex(key.as_bytes())[..24].to_owned(),
        kind: kind.to_owned(),
        name: name.to_owned(),
        qname: qname.to_owned(),
        path: path.to_owned(),
        start_line,
        end_line,
        signature: attrs.signature.to_owned(),
        doc: attrs.doc.to_owned(),
        confidence: confidence.to_owned(),
        content_hash: sha1_hex(content.as_bytes()),
        extra: attrs.extra,
    }
}

pub fn external_node(kind: &str, name: &str, rel: &str) -> Node {
    make_node(
        kind,
        name,
        &format!("external:{name}"),
        rel,
        1,
        1,
        name,
        NodeAttrs {
            confidence: "heuristic",
            ..Default::default()
        },
    )
}

/// Keep the first node per id, in order.
pub fn dedupe_nodes(nodes: Vec<Node>) -> Vec<Node> {
    let mut seen = std::collections::HashSet::new();
    nodes
        .into_iter()
        .filter(|node| seen.insert(node.id.clone()))
        .collect()
}

/// Keep the first edge per `(src, dst, kind, detail)`, in order.
pub fn dedupe_edges(edges: Vec<Edge>) -> Vec<Edge> {
    let mut seen = std::collections::HashSet::new();
    edges
        .into_iter()
        .filter(|edge| {
            let key = (
                edge.src.clone(),
                edge.dst.clone(),
                edge.kind.clone(),
                edge.detail.clone(),
            );
            seen.insert(key)
        })
        .collect()
}
