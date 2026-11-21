//! Ports of tests/test_storage_neighbors.py. The two SQL-trace assertions (no `COUNT(*)`
//! queries for low degree, `LIMIT` on every edge query) are Python internals; those tests keep
//! only their observable results: neighbor counts, ordering and prefix stability.

mod common;

use std::path::Path;

use lode::storage::{Neighbors, connect, get_neighbors, neighbor_sort_key, upsert_repo};
use rusqlite::{Connection, params};
use tempfile::TempDir;

const HIGH_DEGREE_NEIGHBORS: usize = 650;

fn insert_neighbor_graph(conn: &mut Connection, repo: &Path, neighbor_count: usize) -> String {
    let repo_id = upsert_repo(conn, repo).unwrap();
    let center_id = format!("{repo_id}:center:{neighbor_count}");
    let tx = conn.transaction().unwrap();
    let insert_node =
        "INSERT INTO nodes(id, repo_id, owner_path, kind, name, qname, path, start_line,
        end_line, signature, doc, confidence, content_hash, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    let insert_edge = "INSERT INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)
        VALUES (?, ?, ?, ?, 'CALLS', ?, ?)";
    tx.execute(
        insert_node,
        params![
            center_id,
            repo_id,
            "center.py",
            "Function",
            "center",
            "center.center",
            "center.py",
            1,
            2,
            "def center(): ...",
            "",
            "strong",
            "center-hash",
            "{}"
        ],
    )
    .unwrap();
    let confidences = ["heuristic", "strong", "resolved"];
    for index in 0..neighbor_count {
        let target_id = format!("{repo_id}:target:{index}");
        let path = format!("pkg/{:04}_{index:04}.py", neighbor_count - index);
        let kind = if index % 17 == 0 {
            "ExternalSymbol"
        } else {
            "Function"
        };
        let confidence = confidences[index % confidences.len()];
        let line = index as i64 + 1;
        tx.execute(
            insert_node,
            params![
                target_id,
                repo_id,
                path,
                kind,
                format!("target_{index}"),
                format!("pkg.target_{index}"),
                path,
                line,
                line,
                format!("def target_{index}(): ..."),
                "",
                "strong",
                format!("target-hash-{index}"),
                format!("{{\"index\": {index}}}")
            ],
        )
        .unwrap();
        let outgoing_detail = format!("outgoing-{index}");
        let incoming_detail = format!("incoming-{index}");
        tx.execute(
            insert_edge,
            params![
                repo_id,
                "center.py",
                center_id,
                target_id,
                confidence,
                outgoing_detail
            ],
        )
        .unwrap();
        tx.execute(
            insert_edge,
            params![
                repo_id,
                path,
                target_id,
                center_id,
                confidence,
                incoming_detail
            ],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    center_id
}

fn assert_sorted(neighbors: &Neighbors) {
    for items in [&neighbors.outgoing, &neighbors.incoming] {
        let keys: Vec<_> = items.iter().map(neighbor_sort_key).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}

fn graph(neighbor_count: usize) -> (TempDir, TempDir, Connection, String) {
    let repo = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    let mut conn = connect(&lode::config::sqlite_path(data.path())).unwrap();
    let node_id = insert_neighbor_graph(&mut conn, repo.path(), neighbor_count);
    (repo, data, conn, node_id)
}

#[test]
fn low_degree_neighbors_skip_count_queries_when_limit_covers_rows() {
    let (_repo, _data, conn, node_id) = graph(3);
    let neighbors = get_neighbors(&conn, &node_id, 80).unwrap();
    assert_eq!(neighbors.outgoing.len(), 3);
    assert_eq!(neighbors.incoming.len(), 3);
    assert_sorted(&neighbors);
}

#[test]
fn high_degree_neighbors_preserve_public_ordering_when_limited() {
    let (_repo, _data, conn, node_id) = graph(HIGH_DEGREE_NEIGHBORS);
    let full = get_neighbors(&conn, &node_id, HIGH_DEGREE_NEIGHBORS as i64 + 1).unwrap();
    let limited = get_neighbors(&conn, &node_id, 13).unwrap();
    assert_eq!(full.node.as_ref().unwrap().id, node_id);
    assert_eq!(full.outgoing.len(), HIGH_DEGREE_NEIGHBORS);
    assert_eq!(limited.outgoing, full.outgoing[..13]);
    assert_eq!(limited.incoming, full.incoming[..13]);
    assert_sorted(&limited);
    for neighbor in limited.outgoing.iter().chain(&limited.incoming) {
        assert_eq!(neighbor.edge.kind, "CALLS");
        assert!(!neighbor.edge.confidence.is_empty());
        assert!(
            neighbor.edge.detail.starts_with("outgoing-")
                || neighbor.edge.detail.starts_with("incoming-")
        );
        assert!(neighbor.node.is_some());
    }
    let value = serde_json::to_value(&limited.outgoing[0]).unwrap();
    let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["edge", "node"]);
}

#[test]
fn high_degree_neighbors_execute_bounded_edge_queries_for_small_limits() {
    let (_repo, _data, conn, node_id) = graph(HIGH_DEGREE_NEIGHBORS);
    let neighbors = get_neighbors(&conn, &node_id, 7).unwrap();
    let full = get_neighbors(&conn, &node_id, HIGH_DEGREE_NEIGHBORS as i64).unwrap();
    assert_eq!(neighbors.outgoing.len(), 7);
    assert_eq!(neighbors.incoming.len(), 7);
    assert_eq!(neighbors.outgoing, full.outgoing[..7]);
    assert_eq!(neighbors.incoming, full.incoming[..7]);
}
