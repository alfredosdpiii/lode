//! Ports of tests/test_lode.py. SQL-trace assertions (statement counts through
//! `set_trace_callback`) are Python internals; those tests keep their observable results:
//! repeated_query_context_and_graph_lookups_execute_sqlite, long_search_query_uses_single_broad_fts_lookup.

mod common;

use std::fs;
use std::time::{Duration, SystemTime};

use common::{
    FakeEmbeddings, Fixture, count, find, first, run_lode, run_lode_json, write_sample_repo,
};
use lode::cli::{EMBEDDING_TEXT_MAX_INPUT_CHARS, embedding_text};
use lode::config::sha1_hex;
use lode::context::{ContextOptions, build_context_pack};
use lode::indexer::{iter_source_files, parse_file};
use lode::storage::{
    NodeRow, embedding_counts, external_like_terms_from_tokens, get_neighbors,
    replace_file_indexes, repo_filter, search_nodes,
};
use rusqlite::Connection;
use serde_json::{Value, json};

fn sample() -> Fixture {
    let fx = Fixture::new();
    write_sample_repo(fx.repo());
    fx
}

fn owner_count(conn: &Connection, table_column: &str, repo_id: &str, path: &str) -> i64 {
    let (table, column) = table_column.split_once('.').unwrap();
    conn.query_row(
        &format!("SELECT COUNT(*) FROM {table} WHERE repo_id = ? AND {column} = ?"),
        [repo_id, path],
        |row| row.get(0),
    )
    .unwrap()
}

fn context(conn: &Connection, query: &str, budget: i64, limit: usize) -> Value {
    let options = ContextOptions {
        budget,
        limit,
        ..ContextOptions::default()
    };
    build_context_pack(conn, query, &options).unwrap()
}

fn assert_has_resolved_caller(conn: &Connection, node_id: &str, caller_qname: &str) {
    let incoming = get_neighbors(conn, node_id, 80).unwrap().incoming;
    assert!(
        incoming.iter().any(|item| item.edge.kind == "CALLS"
            && item.edge.confidence == "resolved"
            && item
                .node
                .as_ref()
                .is_some_and(|node| node.qname == caller_qname)),
        "{incoming:?}"
    );
}

fn set_mtime(path: &std::path::Path, mtime: SystemTime) {
    fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
}

fn embed(fx: &Fixture, server: &FakeEmbeddings, model: &str, limit: Option<&str>) -> Value {
    let mut args = vec!["embed", "--url", &server.url, "--model", model, "--json"];
    if let Some(limit) = limit {
        args.extend(["--limit", limit]);
    }
    run_lode_json(fx.data(), &args, &[])
}

#[test]
fn embedding_text_caps_large_payloads_while_preserving_identity() {
    let node = NodeRow {
        id: String::new(),
        repo_id: String::new(),
        owner_path: String::new(),
        kind: String::new(),
        name: String::new(),
        qname: "docs.long_section".into(),
        path: String::new(),
        start_line: 1,
        end_line: 1,
        signature: "### Long Section".into(),
        doc: "x".repeat(EMBEDDING_TEXT_MAX_INPUT_CHARS * 4),
        confidence: String::new(),
        content_hash: String::new(),
        rank: 0.0,
        extra: serde_json::Map::new(),
    };
    let text = embedding_text(&node);
    assert!(text.chars().count() <= EMBEDDING_TEXT_MAX_INPUT_CHARS);
    assert!(text.starts_with("docs.long_section\n### Long Section\n"));
}

#[test]
fn parse_file_with_explicit_text_matches_disk() {
    let fx = Fixture::new();
    let text = "def foo():\n    return 1\n";
    fx.write("app.py", text);
    let path = fx.repo().join("app.py");
    let parsed = parse_file(fx.repo(), &path, "", 0.0, text.len() as u64, text);
    assert_eq!(parsed.content_hash, sha1_hex(text.as_bytes()));
    let indexed = parse_file(
        fx.repo(),
        &path,
        &sha1_hex(text.as_bytes()),
        0.0,
        text.len() as u64,
        text,
    );
    assert_eq!(parsed.content_hash, indexed.content_hash);
    let ids = |nodes: &[lode::model::Node]| {
        nodes
            .iter()
            .map(|n| (n.id.clone(), n.content_hash.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&parsed.nodes), ids(&indexed.nodes));
    assert_eq!(parsed.edges.len(), indexed.edges.len());
    assert!(parsed.nodes.iter().any(|n| n.name == "foo"));
}

#[test]
fn generated_artifact_directories_are_not_indexed() {
    let fx = Fixture::new();
    fx.write("src/app.py", "def app():\n    return 1\n");
    fx.write("bench-results/run.json", "{\"ok\": true}\n");
    fx.write("droid-wiki/overview.md", "# Generated wiki\n");
    let indexed: Vec<String> = iter_source_files(fx.repo())
        .iter()
        .map(|path| {
            path.strip_prefix(fx.repo())
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert_eq!(indexed, ["src/app.py"]);
}

#[test]
fn repo_filter_rejects_unsafe_path_inputs() {
    let parent = tempfile::TempDir::new().unwrap();
    let data = tempfile::TempDir::new().unwrap();
    let repo = parent.path().join("repo with spaces");
    fs::create_dir(&repo).unwrap();
    write_sample_repo(&repo);
    let db = lode::config::sqlite_path(data.path());
    let stats = lode::indexer::index_repo(&repo, &db).unwrap();
    let conn = lode::storage::connect(&db).unwrap();
    assert_eq!(
        repo_filter(&conn, Some(repo.to_str().unwrap())).unwrap(),
        Some(stats.repo_id)
    );
    let other = repo.join("..").join("other");
    assert_eq!(
        repo_filter(&conn, Some(other.to_str().unwrap())).unwrap(),
        None
    );
    assert_eq!(repo_filter(&conn, Some("bad\npath")).unwrap(), None);
    assert_eq!(repo_filter(&conn, Some("\0")).unwrap(), None);
}

#[test]
fn batch_replace_preserves_nodes_edges_and_queue() {
    let fx = sample();
    let repo_id = fx.index().repo_id;
    let text = "def new_func():\n    return 42\n";
    fx.write("app.py", text);
    let path = fx.repo().join("app.py");
    let file_index = parse_file(
        fx.repo(),
        &path,
        &sha1_hex(text.as_bytes()),
        0.0,
        text.len() as u64,
        text,
    );
    let node_ids: Vec<String> = file_index.nodes.iter().map(|n| n.id.clone()).collect();
    let edge_count = file_index.edges.len() as i64;
    let fts_expected = file_index
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind.as_str(), "ExternalSymbol" | "ExternalDependency"))
        .count();
    let queue_expected = file_index
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.kind.as_str(),
                "Function" | "Method" | "Class" | "Route" | "DocSection"
            )
        })
        .filter(|n| !n.content_hash.is_empty())
        .count();
    let mut conn = fx.conn();
    let tx = conn.transaction().unwrap();
    replace_file_indexes(&tx, &repo_id, vec![file_index]).unwrap();
    tx.commit().unwrap();

    let mut stmt = conn
        .prepare("SELECT id FROM nodes WHERE repo_id = ? AND owner_path = 'app.py'")
        .unwrap();
    let stored: Vec<String> = stmt
        .query_map([&repo_id], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(stored.len(), node_ids.len());
    assert!(stored.iter().all(|id| node_ids.contains(id)));
    assert_eq!(
        owner_count(&conn, "edges.owner_path", &repo_id, "app.py"),
        edge_count
    );
    let owned = "SELECT id FROM nodes WHERE repo_id = ?1 AND owner_path = 'app.py'";
    let fts: i64 = conn
        .query_row(
            &format!("SELECT COUNT(*) FROM node_fts WHERE node_id IN ({owned})"),
            [&repo_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fts as usize, fts_expected);
    let queued: i64 = conn
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM embedding_queue WHERE repo_id = ?1 AND node_id IN ({owned})"
            ),
            [&repo_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(queued as usize, queue_expected);
}

#[test]
fn indexes_python_symbols_routes_and_calls() {
    let fx = sample();
    let stats = fx.index();
    assert_eq!(stats.scanned, 2);
    assert!(stats.nodes >= 6);
    assert!(stats.edges >= 5);
    let conn = fx.conn();
    let function = first(&conn, "create_user");
    assert_eq!(function.kind, "Function");
    let routes = search_nodes(&conn, "POST /users", None, 5).unwrap();
    assert!(routes.iter().any(|row| row.kind == "Route"));
    let neighbors = get_neighbors(&conn, &function.id, 80).unwrap();
    assert!(
        neighbors
            .outgoing
            .iter()
            .any(|item| item.edge.kind == "CALLS")
    );
    let pack = context(&conn, "create user route", 2000, 10);
    assert!(!pack["must_read"].as_array().unwrap().is_empty());
    assert!(pack.get("confidence").is_some());
}

#[test]
fn find_symbol_prefers_local_definitions_over_external_calls() {
    let fx = Fixture::new();
    fx.write(
        "nested/deep/module.py",
        "def long_symbol_name():\n    return 1\n",
    );
    fx.write(
        "caller.py",
        "def caller():\n    return long_symbol_name()\n",
    );
    fx.index();
    let symbol = first(&fx.conn(), "long_symbol_name");
    assert_eq!(symbol.kind, "Function");
    assert_eq!(symbol.qname, "nested.deep.module.long_symbol_name");
}

#[test]
fn find_symbol_case_insensitive_and_substring_fallback() {
    let fx = Fixture::new();
    fx.write(
        "context_tools.py",
        "def Build_Context_Pack():\n    return 'ok'\n",
    );
    fx.index();
    let conn = fx.conn();
    let exact = lode::storage::find_symbol(&conn, "build_context_pack", None, 5).unwrap();
    assert_eq!(exact[0].name, "Build_Context_Pack");
    assert_eq!(exact[0].kind, "Function");
    let substring = lode::storage::find_symbol(&conn, "context_pack", None, 5).unwrap();
    assert!(
        substring
            .iter()
            .any(|item| item.qname == "context_tools.Build_Context_Pack")
    );
}

#[test]
fn python_parser_preserves_nested_call_attribution() {
    let fx = Fixture::new();
    fx.write(
        "app.py",
        "def helper():\n    return 1\n\ndef outer():\n    def inner():\n        return helper()\n    return inner()\n",
    );
    fx.index();
    let conn = fx.conn();
    let outer = first(&conn, "outer");
    let helper = first(&conn, "helper");
    let (confidence, detail): (String, String) = conn
        .query_row(
            "SELECT confidence, detail FROM edges WHERE src = ? AND dst = ? AND kind = 'CALLS'",
            [&outer.id, &helper.id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(confidence, "strong");
    assert_eq!(detail, "helper");
}

#[test]
fn function_local_imports_still_resolve_calls() {
    let fx = Fixture::new();
    fx.write("callee.py", "def target():\n    return 'ok'\n");
    fx.write(
        "caller.py",
        "def caller():\n    from callee import target\n    return target()\n",
    );
    let stats = fx.index();
    assert!(stats.resolved_calls >= 1);
    let conn = fx.conn();
    let target = first(&conn, "target");
    assert_has_resolved_caller(&conn, &target.id, "caller.caller");
}

#[test]
fn repeated_external_calls_are_persisted_once_per_owner() {
    let fx = Fixture::new();
    fx.write(
        "app.py",
        "def noisy():\n    print('one')\n    print('two')\n    print('three')\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let repo = &stats.repo_id;
    let nodes: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = 'app.py'
             AND kind = 'ExternalSymbol' AND name = 'print'",
            [repo],
            |r| r.get(0),
        )
        .unwrap();
    let edges: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND owner_path = 'app.py' AND kind = 'CALLS' AND detail = 'print'",
            [repo],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!((nodes, edges), (1, 1));
    let fts: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM node_fts f JOIN nodes n ON n.id = f.node_id
             WHERE n.repo_id = ? AND n.kind = 'ExternalSymbol' AND n.name = 'print'",
            [repo],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fts, 0);
    let results = search_nodes(&conn, "print", None, 5).unwrap();
    assert!(
        results
            .iter()
            .any(|row| row.kind == "ExternalSymbol" && row.name == "print")
    );
}

#[test]
fn context_related_matches_public_neighbor_compaction() {
    let fx = sample();
    fx.index();
    let conn = fx.conn();
    let pack = context(&conn, "create user route", 2000, 5);
    let related = pack["related"].as_array().unwrap();
    assert!(!related.is_empty());
    let compact = |items: Vec<lode::storage::Neighbor>| -> Value {
        items
            .into_iter()
            .filter_map(|item| {
                let node = item.node?;
                Some(
                    json!({"edge": item.edge.kind, "qname": node.qname, "kind": node.kind,
                            "path": node.path, "confidence": item.edge.confidence}),
                )
            })
            .collect()
    };
    for entry in related {
        let neighbors = get_neighbors(&conn, entry["node_id"].as_str().unwrap(), 16).unwrap();
        assert_eq!(entry["incoming"], compact(neighbors.incoming));
        assert_eq!(entry["outgoing"], compact(neighbors.outgoing));
    }
}

#[test]
fn repeated_query_context_and_graph_lookups_execute_sqlite() {
    let fx = sample();
    fx.index();
    let conn = fx.conn();
    let target = lode::storage::find_symbol(&conn, "create_user", None, 5).unwrap()[0].clone();
    let search = search_nodes(&conn, "create user", None, 5).unwrap();
    assert_eq!(search, search_nodes(&conn, "create user", None, 5).unwrap());
    assert!(!search.is_empty());
    let neighbors = get_neighbors(&conn, &target.id, 8).unwrap();
    assert_eq!(neighbors, get_neighbors(&conn, &target.id, 8).unwrap());
    let pack = context(&conn, "create user route", 2000, 5);
    assert_eq!(pack, context(&conn, "create user route", 2000, 5));
    // Repeated reads see fresh data: a re-index after an edit changes the next lookup.
    fx.write("app.py", "def create_user_v2():\n    return 1\n");
    fx.index();
    assert!(
        lode::storage::find_symbol(&conn, "create_user", None, 5)
            .unwrap()
            .iter()
            .all(|n| n.name != "create_user")
    );
}

#[test]
fn operational_search_and_context_preserve_source_ranking() {
    let fx = Fixture::new();
    fx.write(
        "src/lode/embeddings.py",
        "def embeddings_url():\n    return 'http://127.0.0.1:7980'\n\n\
         def embeddings_model():\n    return 'Snowflake/snowflake-arctic-embed-s'\n",
    );
    fx.write(
        "src/lode/context.py",
        "def build_context_pack():\n    \"\"\"Build context pack for agents.\"\"\"\n    return []\n",
    );
    fx.write(
        "README.md",
        "# Architecture\n\nThe operational docs mention the embedding queue explicitly.\n\n\
         # Context\n\nThe docs also describe how to build context pack outputs.\n",
    );
    fx.index();
    let conn = fx.conn();
    let hits = search_nodes(&conn, "embedding queue", None, 20).unwrap();
    assert_eq!(hits[0].path, "src/lode/embeddings.py");
    let pack = context(&conn, "build context pack", 4000, 10);
    assert_eq!(pack["confidence"], "strong");
    assert!(
        pack["top_hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["path"] == "src/lode/context.py")
    );
}

#[test]
fn long_search_query_uses_single_broad_fts_lookup() {
    let fx = sample();
    fx.index();
    let long_query = "create user service save account profile controller repository validate serialize response payload";
    let tokens: Vec<String> = long_query.split_whitespace().map(String::from).collect();
    assert!(external_like_terms_from_tokens(&tokens).is_empty());
    assert_eq!(
        external_like_terms_from_tokens(&["print".to_string()]),
        ["print"]
    );
    let results = search_nodes(&fx.conn(), long_query, None, 5).unwrap();
    assert!(results.iter().any(|row| row.path == "app.py"));
}

#[test]
fn neighbors_include_cross_file_callers_after_reindex() {
    let fx = Fixture::new();
    fx.write("callee.py", "def target_function():\n    return 'ok'\n");
    fx.write(
        "caller.py",
        "from callee import target_function\n\ndef call_target():\n    return target_function()\n",
    );
    assert!(fx.index().resolved_calls >= 1);
    let old_id = first(&fx.conn(), "target_function").id;
    assert_has_resolved_caller(&fx.conn(), &old_id, "caller.call_target");

    fx.write(
        "callee.py",
        "\n\ndef target_function():\n    return 'updated'\n",
    );
    assert!(fx.index().resolved_calls >= 1);
    let conn = fx.conn();
    let target = first(&conn, "target_function");
    assert_ne!(target.id, old_id);
    assert_has_resolved_caller(&conn, &target.id, "caller.call_target");
}

#[test]
fn hot_reindex_skips_all_unchanged() {
    let fx = sample();
    let cold = fx.index();
    assert!(cold.scanned > 0 && cold.indexed > 0 && cold.nodes > 0 && cold.edges > 0);
    assert_eq!((cold.skipped_unchanged, cold.removed), (0, 0));
    let hot = fx.index();
    assert_eq!(hot.scanned, cold.scanned);
    assert_eq!(hot.skipped_unchanged, cold.scanned);
    assert_eq!(
        (hot.indexed, hot.nodes, hot.edges, hot.removed),
        (0, 0, 0, 0)
    );
}

#[test]
fn reindex_skips_via_hash_fallback_when_mtime_changes() {
    let fx = sample();
    let cold = fx.index();
    assert!(cold.indexed > 0);
    let app = fx.repo().join("app.py");
    let mtime = fs::metadata(&app).unwrap().modified().unwrap();
    set_mtime(&app, mtime + Duration::from_secs(1));
    let hot = fx.index();
    assert_eq!(hot.skipped_unchanged, cold.scanned);
    assert_eq!(
        (hot.indexed, hot.nodes, hot.edges, hot.removed),
        (0, 0, 0, 0)
    );
}

#[test]
fn reindex_same_length_different_content_with_preserved_mtime() {
    let fx = Fixture::new();
    fx.write("app.py", "def foo():\n    return 1\n");
    let cold = fx.index();
    assert!(cold.indexed > 0 && cold.nodes > 0);
    assert!(!find(&fx.conn(), "foo").is_empty());
    let old_nodes = owner_count(&fx.conn(), "nodes.owner_path", &cold.repo_id, "app.py");
    assert!(old_nodes > 0);

    let app = fx.repo().join("app.py");
    let mtime = fs::metadata(&app).unwrap().modified().unwrap();
    fx.write("app.py", "def bar():\n    return 2\n");
    set_mtime(&app, mtime);
    let hot = fx.index();
    assert!(hot.indexed > 0 && hot.nodes > 0);
    assert_eq!(hot.removed, 0);
    let conn = fx.conn();
    assert!(!find(&conn, "bar").is_empty());
    assert!(find(&conn, "foo").is_empty());
    assert_eq!(
        owner_count(&conn, "nodes.owner_path", &cold.repo_id, "app.py"),
        old_nodes
    );
}

#[cfg(unix)]
#[test]
fn reindex_unreadable_file_removes_stale_data() {
    use std::os::unix::fs::PermissionsExt;
    let fx = Fixture::new();
    fx.write("app.py", "def foo():\n    return 1\n");
    let cold = fx.index();
    assert!(cold.indexed > 0);
    let app = fx.repo().join("app.py");
    fs::set_permissions(&app, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&app).is_ok() {
        // Running as root: permissions cannot make the file unreadable.
        fs::set_permissions(&app, fs::Permissions::from_mode(0o644)).unwrap();
        return;
    }
    let hot = fx.index();
    fs::set_permissions(&app, fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(hot.indexed, 0);
    assert!(hot.removed > 0);
    let conn = fx.conn();
    assert_eq!(
        owner_count(&conn, "nodes.owner_path", &cold.repo_id, "app.py"),
        0
    );
    assert_eq!(
        owner_count(&conn, "edges.owner_path", &cold.repo_id, "app.py"),
        0
    );
    assert_eq!(owner_count(&conn, "files.path", &cold.repo_id, "app.py"), 0);
}

#[test]
fn reindex_after_file_change_updates_and_no_stale_nodes() {
    let fx = sample();
    let cold = fx.index();
    assert!(cold.indexed > 0);
    fx.write("app.py", "def new_function():\n    return 42\n");
    let hot = fx.index();
    assert!(hot.indexed > 0);
    assert_eq!(hot.removed, 0);
    let conn = fx.conn();
    assert!(!find(&conn, "new_function").is_empty());
    assert!(find(&conn, "create_user").is_empty());
    assert_eq!(
        owner_count(&conn, "nodes.owner_path", &cold.repo_id, "app.py"),
        2
    );
}

#[test]
fn reindex_after_file_delete_removes_stale_data() {
    let fx = sample();
    let cold = fx.index();
    assert!(cold.indexed > 0);
    fs::remove_file(fx.repo().join("app.py")).unwrap();
    let hot = fx.index();
    assert_eq!(hot.indexed, 0);
    assert!(hot.removed > 0);
    let conn = fx.conn();
    assert_eq!(
        owner_count(&conn, "nodes.owner_path", &cold.repo_id, "app.py"),
        0
    );
    assert_eq!(
        owner_count(&conn, "edges.owner_path", &cold.repo_id, "app.py"),
        0
    );
    assert_eq!(owner_count(&conn, "files.path", &cold.repo_id, "app.py"), 0);
    let orphaned = count(
        &conn,
        "SELECT COUNT(*) FROM edges e LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst WHERE n.id IS NULL",
    );
    assert_eq!(orphaned, 0);
}

#[test]
fn reindex_coverage_does_not_shrink() {
    let fx = sample();
    assert!(fx.index().indexed > 0);
    assert_eq!(fx.index().indexed, 0);
    let conn = fx.conn();
    assert!(count(&conn, "SELECT COUNT(DISTINCT path) FROM files") >= 2);
    assert!(count(&conn, "SELECT COUNT(*) FROM nodes") >= 6);
    assert!(count(&conn, "SELECT COUNT(*) FROM edges") >= 5);
}

#[test]
fn cli_json_output() {
    let fx = sample();
    let repo = fx.repo().to_str().unwrap();
    let index = run_lode_json(fx.data(), &["index", repo, "--json"], &[]);
    assert_eq!(index["ok"], true);
    let search = run_lode_json(fx.data(), &["search", "UserService", "--json"], &[]);
    assert_eq!(search["ok"], true);
    assert!(!search["results"].as_array().unwrap().is_empty());
    let impact = run_lode_json(
        fx.data(),
        &["impact", "audit_user", "--repo", repo, "--json"],
        &[],
    );
    assert_eq!(impact["ok"], true);
    assert_eq!(
        impact["results"][0]["callers"][0]["node"]["qname"],
        "app.create_user"
    );
}

#[cfg(not(feature = "embed"))]
#[test]
fn cli_rerank_without_embed_feature_reports_rerank_not_built() {
    let fx = sample();
    fx.index();
    let (code, stdout, stderr) = run_lode(
        fx.data(),
        &["context", "UserService", "--rerank", "--json"],
        &[],
    );
    assert_eq!(code, 1, "{stdout}");
    let error: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(error["ok"], false);
    assert!(
        error["error"]
            .as_str()
            .unwrap()
            .contains("built without the embed feature"),
        "{stderr}"
    );
}

#[test]
fn cli_eval_batch_answers_each_jsonl_line() {
    use std::io::Write;
    let fx = sample();
    fx.index();
    let request = serde_json::json!({"data_dir": fx.data(), "query": "UserService", "limit": 5});
    let mut child = std::process::Command::new(common::lode_bin())
        .arg("eval-batch")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{request}\nnot json").unwrap();
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let replies: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["ok"], true);
    assert_eq!(
        replies[0]["pack"]["top_hits"][0]["qname"],
        "app.UserService"
    );
    assert_eq!(replies[1]["ok"], false);
}

#[test]
fn cli_embed_persists_vectors_from_local_endpoint() {
    let fx = sample();
    fx.index();
    let server = FakeEmbeddings::start();
    let payload = embed(&fx, &server, "fake-embedding-model", Some("3"));
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["embedded"], 3);
    let conn = fx.conn();
    assert_eq!(embedding_counts(&conn).unwrap().embedded, 3);
    let dims: (i64, i64) = conn
        .query_row("SELECT MIN(dims), MAX(dims) FROM embeddings", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(dims, (3, 3));
}

#[test]
fn cli_embed_honors_limit_and_no_pending_is_idempotent() {
    let fx = sample();
    fx.index();
    let server = FakeEmbeddings::start();
    let before = embedding_counts(&fx.conn()).unwrap();
    assert!(before.queued > 2);

    let first_run = embed(&fx, &server, "fake-embedding-model", Some("2"));
    assert_eq!(first_run["embedded"], 2);
    assert_eq!(server.request_count(), 1);
    assert_eq!(server.requests.lock().unwrap()[0].len(), 2);
    let after_first = embedding_counts(&fx.conn()).unwrap();
    assert_eq!(after_first.queued, before.queued - 2);
    assert_eq!(after_first.embedded, before.embedded + 2);

    let remaining = embed(&fx, &server, "fake-embedding-model", Some("100"));
    assert_eq!(remaining["embedded"], after_first.queued);
    assert_eq!(server.request_count(), 2);

    let no_pending = embed(&fx, &server, "fake-embedding-model", None);
    assert_eq!(no_pending["ok"], true);
    assert_eq!(no_pending["embedded"], 0);
    assert_eq!(no_pending["queued"], 0);
    assert_eq!(no_pending["total_embeddings"], before.queued);
    assert_eq!(server.request_count(), 2);
}

#[test]
fn cli_embed_fails_closed_without_local_endpoint() {
    let fx = sample();
    fx.index();
    let before = embedding_counts(&fx.conn()).unwrap();
    let (code, stdout, stderr) = run_lode(fx.data(), &["embed", "--json"], &[]);
    assert_ne!(code, 0);
    let payload: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(payload["ok"], false);
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("No local embeddings endpoint configured")
    );
    assert!(!(stdout + &stderr).contains("panicked"));
    assert_eq!(embedding_counts(&fx.conn()).unwrap(), before);
}

#[test]
fn cli_embed_rejects_hosted_endpoint_without_network_call() {
    let fx = sample();
    fx.index();
    let (code, _stdout, stderr) = run_lode(
        fx.data(),
        &["embed", "--url", "https://example.com", "--json"],
        &[],
    );
    assert_ne!(code, 0);
    let payload: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(payload["ok"], false);
    assert!(payload["error"].as_str().unwrap().contains("must be local"));
}

#[test]
fn changed_content_refreshes_embedding_without_duplicate_rows() {
    let fx = Fixture::new();
    fx.write("app.py", "def target():\n    return 1\n");
    fx.index();
    let server = FakeEmbeddings::start();
    assert_eq!(embed(&fx, &server, "first-model", Some("1"))["embedded"], 1);

    let row_sql = "SELECT dims, vector_json, model, embedded_at FROM embeddings WHERE node_id = ?";
    let read_row = |conn: &Connection, id: &str| -> Vec<(i64, String, String, f64)> {
        let mut stmt = conn.prepare(row_sql).unwrap();
        stmt.query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    let node_id = first(&fx.conn(), "target").id;
    let before = read_row(&fx.conn(), &node_id).pop().unwrap();

    fx.write("app.py", "def target():\n    return 2\n");
    fx.index();
    let conn = fx.conn();
    assert_eq!(first(&conn, "target").id, node_id);
    let queued_at: Option<f64> = conn
        .query_row(
            "SELECT embedded_at FROM embedding_queue WHERE node_id = ?",
            [&node_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(queued_at, None);
    assert_eq!(read_row(&conn, &node_id).len(), 1);

    *server.vector_base.lock().unwrap() = 100.0;
    assert_eq!(
        embed(&fx, &server, "second-model", Some("1"))["embedded"],
        1
    );
    assert_eq!(server.request_count(), 2);
    let rows = read_row(&fx.conn(), &node_id);
    assert_eq!(rows.len(), 1);
    let (dims, vector, model, embedded_at) = &rows[0];
    assert_eq!(*dims, 3);
    assert_eq!(model, "second-model");
    assert_ne!(*vector, before.1);
    assert!(*embedded_at >= before.3);
}

#[test]
fn deleted_nodes_do_not_leave_embedding_artifacts() {
    let fx = Fixture::new();
    fx.write("app.py", "def deleted_target():\n    return 1\n");
    fx.write("keep.py", "def kept_target():\n    return 2\n");
    fx.index();
    let server = FakeEmbeddings::start();
    assert!(
        embed(&fx, &server, "fake-embedding-model", Some("10"))["embedded"]
            .as_i64()
            .unwrap()
            >= 2
    );
    let deleted: Vec<String> = {
        let conn = fx.conn();
        let mut stmt = conn
            .prepare("SELECT id FROM nodes WHERE owner_path = 'app.py'")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert!(!deleted.is_empty());

    fs::remove_file(fx.repo().join("app.py")).unwrap();
    fx.index();
    let conn = fx.conn();
    let orphan_sql = |table: &str| {
        format!(
            "SELECT COUNT(*) FROM {table} t LEFT JOIN nodes n ON n.id = t.node_id WHERE n.id IS NULL"
        )
    };
    assert_eq!(count(&conn, &orphan_sql("embedding_queue")), 0);
    assert_eq!(count(&conn, &orphan_sql("embeddings")), 0);
    for id in &deleted {
        for table in ["embedding_queue", "embeddings"] {
            let rows: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE node_id = ?"),
                    [id],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(rows, 0, "{table} {id}");
        }
    }
}

#[test]
fn gitignored_files_and_directories_are_skipped() {
    let fx = Fixture::new();
    fx.write(
        ".gitignore",
        "ignored_dir/\n*.tmp.py\n!keep.tmp.py\n/root_skip.py\n",
    );
    fx.write("app.py", "def ok():\n    return 1\n");
    fx.write("root_skip.py", "def skipped():\n    return 2\n");
    fx.write("sub/root_skip.py", "def kept():\n    return 3\n");
    fx.write("drop.tmp.py", "def dropped():\n    return 4\n");
    fx.write("keep.tmp.py", "def unignored():\n    return 5\n");
    fx.write("ignored_dir/dump.py", "def buried():\n    return 6\n");
    fx.write("sub/.gitignore", "local_ignore.py\n");
    fx.write("sub/local_ignore.py", "def local_skip():\n    return 7\n");

    let paths: Vec<String> = iter_source_files(fx.repo())
        .into_iter()
        .map(|p| {
            p.strip_prefix(fx.repo())
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert!(paths.contains(&"app.py".to_owned()), "{paths:?}");
    assert!(paths.contains(&"sub/root_skip.py".to_owned()), "{paths:?}");
    assert!(paths.contains(&"keep.tmp.py".to_owned()), "{paths:?}");
    assert!(!paths.contains(&"root_skip.py".to_owned()), "{paths:?}");
    assert!(!paths.contains(&"drop.tmp.py".to_owned()), "{paths:?}");
    assert!(
        !paths.contains(&"ignored_dir/dump.py".to_owned()),
        "{paths:?}"
    );
    assert!(
        !paths.contains(&"sub/local_ignore.py".to_owned()),
        "{paths:?}"
    );
}
