//! Ports of tests/test_public_compatibility.py.
//! Skipped parts (Python-only surfaces): `kg`/`kgd` aliases and `python -m lode[.daemon]` help,
//! `scripts/bench_lode.py` data-dir isolation, and the `openapi/loded.openapi.yml` text checks.
//! The daemon contract test drives `App::handle` directly; request-id echo is checked against the
//! real `loded` process in lode_serve_mirrors_loded_for_core_api_shapes.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{free_port, http, lode_bin, run_lode, run_lode_json, write_file};
use lode::daemon::App;
use serde_json::{Value, json};
use tempfile::TempDir;

const APP_CODE: &str = "from services import UserService

def create_user(name: str):
    service = UserService()
    return service.save_user(name)
";

const SERVICE_CODE: &str = "class UserService:
    def save_user(self, name: str):
        return {\"name\": name}
";

const PUBLIC_NODE_FIELDS: [&str; 15] = [
    "id",
    "repo_id",
    "owner_path",
    "kind",
    "name",
    "qname",
    "path",
    "start_line",
    "end_line",
    "signature",
    "doc",
    "confidence",
    "content_hash",
    "rank",
    "extra",
];

const IMPACT_FIELDS: [&str; 8] = [
    "target",
    "summary",
    "files",
    "callers",
    "callees",
    "blast_radius",
    "incoming",
    "outgoing",
];

const EXTERNAL_KINDS: [&str; 2] = ["ExternalSymbol", "ExternalDependency"];

fn write_fixture_repo(repo: &Path, service_name: &str) {
    write_file(
        repo,
        "app.py",
        &APP_CODE.replace("UserService", service_name),
    );
    write_file(
        repo,
        "services.py",
        &SERVICE_CODE.replace("UserService", service_name),
    );
}

fn resolved(path: &Path) -> String {
    path.canonicalize().unwrap().to_string_lossy().into_owned()
}

fn same_dir(reported: &Value, expected: &Path) -> bool {
    PathBuf::from(reported.as_str().unwrap())
        .canonicalize()
        .unwrap()
        == expected.canonicalize().unwrap()
}

fn assert_node_schema(node: &Value) {
    for field in PUBLIC_NODE_FIELDS {
        assert!(node.get(field).is_some(), "{field} missing in {node}");
    }
    assert!(node.get("extra_json").is_none());
    assert!(!Path::new(node["path"].as_str().unwrap()).is_absolute());
    let start = node["start_line"].as_i64().unwrap();
    let end = node["end_line"].as_i64().unwrap();
    assert!(start >= 1 && end >= start, "{node}");
}

fn assert_index_stats(payload: &Value, repo: &Path) {
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["root"], resolved(repo));
    for field in [
        "repo_id",
        "root",
        "scanned",
        "indexed",
        "skipped_unchanged",
        "removed",
        "resolved_imports",
        "resolved_calls",
        "resolved_extends",
        "nodes",
        "edges",
    ] {
        assert!(payload.get(field).is_some(), "{field}");
    }
}

fn items(value: &Value) -> &Vec<Value> {
    value.as_array().unwrap()
}

fn search_tuples(payload: &Value) -> Vec<(Value, Value, Value, Value)> {
    items(&payload["results"])
        .iter()
        .map(|i| {
            (
                i["path"].clone(),
                i["kind"].clone(),
                i["qname"].clone(),
                i["start_line"].clone(),
            )
        })
        .collect()
}

fn paths(payload: &Value, key: &str) -> Vec<Value> {
    items(&payload[key])
        .iter()
        .map(|item| item["path"].clone())
        .collect()
}

fn radius_nodes(payload: &Value) -> Vec<Value> {
    let radius = &payload["results"][0]["blast_radius"];
    ["upstream", "downstream", "entrypoints"]
        .iter()
        .flat_map(|key| radius[key].as_array().cloned().unwrap_or_default())
        .map(|entry| entry["node"].clone())
        .collect()
}

/// Runs `lode` without `--data-dir`, so the data directory comes from the environment.
fn run_lode_env(args: &[&str], env: &[(&str, &str)]) -> Value {
    let mut cmd = Command::new(lode_bin());
    cmd.args(args)
        .env_remove("LODE_DATA_DIR")
        .env_remove("KG_DATA_DIR")
        .env_remove("LODE_EMBEDDINGS_URL");
    for (key, value) in env {
        cmd.env(key, value);
    }
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn help(bin: &str, args: &[&str]) -> String {
    let out = Command::new(bin).args(args).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(!text.contains("panicked"));
    text
}

#[test]
fn public_entrypoint_help_and_data_dir_environment_precedence() {
    let top = help(lode_bin(), &["--help"]);
    assert!(top.contains("--data-dir"));
    for sub in [
        "index",
        "status",
        "search",
        "symbol",
        "context",
        "impact",
        "neighbors",
        "kuzu-sync",
        "embed",
        "serve",
    ] {
        assert!(top.contains(sub), "{sub}");
    }
    let serve = help(lode_bin(), &["serve", "--help"]);
    assert!(serve.contains("--host") && serve.contains("--port"));
    let daemon = help(env!("CARGO_BIN_EXE_loded"), &["--help"]);
    assert!(
        daemon.contains("--host") && daemon.contains("--port") && daemon.contains("--data-dir")
    );

    let root = TempDir::new().unwrap();
    let repo = root.path().join("repo");
    write_fixture_repo(&repo, "UserService");
    let repo_arg = repo.to_str().unwrap();

    let lode_data = root.path().join("lode-env-data");
    let env = [("LODE_DATA_DIR", lode_data.to_str().unwrap())];
    assert_index_stats(&run_lode_env(&["index", repo_arg, "--json"], &env), &repo);
    let status = run_lode_env(&["status", "--json"], &env);
    assert!(same_dir(&status["data_dir"], &lode_data));
    assert_eq!(status["repos"][0]["root"], resolved(&repo));

    let kg_data = root.path().join("kg-env-data");
    let env = [("KG_DATA_DIR", kg_data.to_str().unwrap())];
    run_lode_env(&["index", repo_arg, "--json"], &env);
    assert!(same_dir(
        &run_lode_env(&["status", "--json"], &env)["data_dir"],
        &kg_data
    ));

    let unused = root.path().join("unused-env-data");
    let override_data = root.path().join("override-data");
    let env = [("LODE_DATA_DIR", unused.to_str().unwrap())];
    run_lode_json(&override_data, &["index", repo_arg, "--json"], &env);
    let status = run_lode_json(&override_data, &["status", "--json"], &env);
    assert!(same_dir(&status["data_dir"], &override_data));
    assert!(override_data.join("lode.sqlite").exists());
    assert!(!unused.join("lode.sqlite").exists());
}

#[test]
fn cli_json_contracts_for_index_query_context_graph_and_errors() {
    let repo_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    let (repo, data) = (repo_dir.path(), data_dir.path());
    write_fixture_repo(repo, "UserService");
    let repo_arg = repo.to_str().unwrap();

    let index = run_lode_json(data, &["index", repo_arg, "--json"], &[]);
    assert_index_stats(&index, repo);
    rusqlite::Connection::open(data.join("lode.sqlite"))
        .unwrap()
        .execute(
            "UPDATE nodes SET start_line = 0, end_line = 0 WHERE kind IN ('ExternalSymbol', 'ExternalDependency')",
            [],
        )
        .unwrap();

    let status = run_lode_json(data, &["status", "--json"], &[]);
    assert!(same_dir(&status["data_dir"], data));
    for field in ["id", "root", "name", "indexed_at", "files", "nodes"] {
        assert!(status["repos"][0].get(field).is_some(), "{field}");
    }

    let search = run_lode_json(
        data,
        &[
            "search",
            "UserService save_user",
            "--repo",
            repo_arg,
            "--limit",
            "5",
            "--json",
        ],
        &[],
    );
    assert_eq!(search["ok"], true);
    let results = items(&search["results"]);
    assert!(!results.is_empty() && results.len() <= 5);
    assert!(!EXTERNAL_KINDS.contains(&results[0]["kind"].as_str().unwrap()));
    for result in results {
        assert_node_schema(result);
        assert_eq!(result["repo_id"], index["repo_id"]);
    }

    let stale = run_lode_json(
        data,
        &[
            "search", "services", "--repo", repo_arg, "--limit", "20", "--json",
        ],
        &[],
    );
    assert!(
        items(&stale["results"])
            .iter()
            .any(|r| EXTERNAL_KINDS.contains(&r["kind"].as_str().unwrap()))
    );
    items(&stale["results"]).iter().for_each(assert_node_schema);

    let symbol = run_lode_json(
        data,
        &[
            "symbol",
            "UserService",
            "--repo",
            repo_arg,
            "--limit",
            "5",
            "--json",
        ],
        &[],
    );
    assert_eq!(symbol["ok"], true);
    assert!(items(&symbol["results"]).len() <= 5);
    assert_eq!(symbol["results"][0]["kind"], "Class");
    items(&symbol["results"])
        .iter()
        .for_each(assert_node_schema);

    let context = run_lode_json(
        data,
        &[
            "context",
            "UserService save_user",
            "--repo",
            repo_arg,
            "--budget",
            "2000",
            "--limit",
            "5",
            "--json",
        ],
        &[],
    );
    assert_eq!(context["ok"], true);
    assert_eq!(context["query"], "UserService save_user");
    assert_eq!(context["budget"], 2000);
    assert!(!items(&context["must_read"]).is_empty());
    assert!(items(&context["top_hits"]).len() <= 5);
    for field in ["summary", "related", "confidence", "notes"] {
        assert!(context.get(field).is_some(), "{field}");
    }
    let must_read_fields = [
        "node_id",
        "kind",
        "name",
        "qname",
        "path",
        "start_line",
        "end_line",
        "signature",
        "doc",
        "why",
        "confidence",
    ];
    for item in items(&context["must_read"]) {
        assert!(
            must_read_fields.iter().all(|f| item.get(f).is_some()),
            "{item}"
        );
    }
    for item in items(&context["top_hits"]) {
        assert!(
            ["id", "kind", "qname", "path", "lines", "confidence"]
                .iter()
                .all(|f| item.get(f).is_some()),
            "{item}"
        );
    }

    let create = run_lode_json(
        data,
        &["symbol", "create_user", "--repo", repo_arg, "--json"],
        &[],
    );
    let node_id = create["results"][0]["id"].as_str().unwrap().to_owned();
    let neighbors = run_lode_json(
        data,
        &["neighbors", &node_id, "--limit", "5", "--json"],
        &[],
    );
    assert_eq!(neighbors["ok"], true);
    assert_eq!(neighbors["node"]["id"], node_id.as_str());
    for neighbor in items(&neighbors["incoming"])
        .iter()
        .chain(items(&neighbors["outgoing"]))
    {
        assert!(
            ["kind", "confidence", "detail"]
                .iter()
                .all(|f| neighbor["edge"].get(f).is_some())
        );
        assert!(neighbor.get("node").is_some());
    }

    for target in ["create_user", node_id.as_str()] {
        let impact = run_lode_json(
            data,
            &[
                "impact",
                target,
                "--repo",
                repo_arg,
                "--direction",
                "both",
                "--depth",
                "2",
                "--max-nodes",
                "100",
                "--json",
            ],
            &[],
        );
        assert_eq!(impact["ok"], true);
        assert_eq!(impact["query"], target);
        assert!(
            IMPACT_FIELDS
                .iter()
                .all(|f| impact["results"][0].get(f).is_some()),
            "{impact}"
        );
        for node in radius_nodes(&impact) {
            let (start, end) = (
                node["lines"][0].as_i64().unwrap(),
                node["lines"][1].as_i64().unwrap(),
            );
            assert!(start >= 1 && end >= start, "{node}");
        }
    }

    let missing = repo.join("does-not-exist");
    let (code, stdout, stderr) =
        run_lode(data, &["index", missing.to_str().unwrap(), "--json"], &[]);
    assert_ne!(code, 0);
    let payload: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(payload["ok"], false);
    assert!(!payload["error"].as_str().unwrap().is_empty());
    assert!(!(stdout + &stderr).contains("panicked"));

    let usage = Command::new(lode_bin())
        .args(["index", "--json"])
        .output()
        .unwrap();
    assert!(!usage.status.success());
    assert!(!String::from_utf8_lossy(&usage.stderr).contains("panicked"));
}

#[test]
fn data_dir_isolation_in_cli_and_benchmark_flows() {
    let root = TempDir::new().unwrap();
    let (repo_a, repo_b, home) = (
        root.path().join("repo-a"),
        root.path().join("repo-b"),
        root.path().join("home"),
    );
    std::fs::create_dir(&home).unwrap();
    write_fixture_repo(&repo_a, "AlphaService");
    write_fixture_repo(&repo_b, "BetaService");
    let (data_a, data_b) = (root.path().join("data-a"), root.path().join("data-b"));
    let env = [("HOME", home.to_str().unwrap())];
    let env_b = [
        ("HOME", home.to_str().unwrap()),
        ("LODE_DATA_DIR", data_b.to_str().unwrap()),
    ];
    let (a, b) = (repo_a.to_str().unwrap(), repo_b.to_str().unwrap());

    run_lode_json(&data_a, &["index", a, "--json"], &env);
    run_lode_env(&["index", b, "--json"], &env_b);
    assert_eq!(
        run_lode_json(&data_a, &["status", "--json"], &env)["repos"][0]["root"],
        resolved(&repo_a)
    );
    assert_eq!(
        run_lode_env(&["status", "--json"], &env_b)["repos"][0]["root"],
        resolved(&repo_b)
    );

    let alpha = run_lode_json(
        &data_a,
        &["search", "AlphaService", "--repo", a, "--json"],
        &env,
    );
    let beta_from_a = run_lode_json(
        &data_a,
        &["symbol", "BetaService", "--repo", a, "--json"],
        &env,
    );
    let beta = run_lode_env(
        &[
            "context",
            "BetaService save_user",
            "--repo",
            b,
            "--limit",
            "5",
            "--json",
        ],
        &env_b,
    );
    assert!(!items(&alpha["results"]).is_empty());
    assert!(items(&beta_from_a["results"]).is_empty());
    assert!(!items(&beta["must_read"]).is_empty());
    assert!(!home.join(".local/share/lode/lode.sqlite").exists());
}

fn call(app: &App, method: &str, path: &str, body: &[u8]) -> (u16, &'static str, Value) {
    let reply = app.handle(method, path, body);
    let parsed = serde_json::from_str(&reply.body).unwrap_or(Value::Null);
    (reply.status, reply.content_type, parsed)
}

fn ok(app: &App, method: &str, path: &str, body: &Value) -> Value {
    let raw = if body.is_null() {
        Vec::new()
    } else {
        body.to_string().into_bytes()
    };
    let (status, content_type, payload) = call(app, method, path, &raw);
    assert_eq!(status, 200, "{method} {path}: {payload}");
    assert!(content_type.starts_with("application/json"));
    payload
}

fn query(pairs: &[(&str, &str)]) -> String {
    form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs)
        .finish()
}

#[test]
fn daemon_api_contracts_errors_openapi_and_cli_consistency() {
    let repo_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    let (repo, data) = (repo_dir.path(), data_dir.path());
    write_fixture_repo(repo, "UserService");
    let repo_arg = repo.to_str().unwrap();
    let cli_index = run_lode_json(data, &["index", repo_arg, "--json"], &[]);
    assert_index_stats(&cli_index, repo);
    let app = App {
        data_dir: data.to_path_buf(),
        ..App::default()
    };

    assert_eq!(
        ok(&app, "GET", "/health", &Value::Null),
        json!({"ok": true, "service": "lode"})
    );
    let metrics = app.handle("GET", "/metrics", b"");
    assert_eq!(metrics.status, 200);
    assert!(metrics.content_type.starts_with("text/plain"));
    assert!(metrics.body.contains("loded_requests_total"));

    assert_index_stats(
        &ok(&app, "POST", "/index", &json!({"path": repo_arg})),
        repo,
    );
    let status = ok(&app, "GET", "/status", &Value::Null);
    for field in ["id", "root", "name", "indexed_at", "files", "nodes"] {
        assert!(status["repos"][0].get(field).is_some(), "{field}");
    }

    let q = "UserService save_user";
    let search_q = ok(
        &app,
        "GET",
        &format!(
            "/search?{}",
            query(&[("q", q), ("repo", repo_arg), ("limit", "5")])
        ),
        &Value::Null,
    );
    let search_query = ok(
        &app,
        "GET",
        &format!(
            "/search?{}",
            query(&[("query", q), ("repo", repo_arg), ("limit", "5")])
        ),
        &Value::Null,
    );
    let search_post = ok(
        &app,
        "POST",
        "/search",
        &json!({"query": q, "repo": repo_arg, "limit": 5}),
    );
    for payload in [&search_q, &search_query, &search_post] {
        let results = items(&payload["results"]);
        assert!(!results.is_empty() && results.len() <= 5);
        for result in results {
            assert_node_schema(result);
            assert_eq!(result["repo_id"], cli_index["repo_id"]);
        }
    }

    let context = ok(
        &app,
        "POST",
        "/context",
        &json!({"query": q, "repo": repo_arg, "budget": 2000, "limit": 5}),
    );
    assert_eq!(context["query"], q);
    assert_eq!(context["budget"], 2000);
    assert!(items(&context["top_hits"]).len() <= 5);
    assert!(!items(&context["must_read"]).is_empty());

    let impact_query = query(&[
        ("target", "create_user"),
        ("repo", repo_arg),
        ("direction", "up"),
        ("depth", "1"),
        ("max_nodes", "100"),
    ]);
    let impact_get = ok(
        &app,
        "GET",
        &format!("/impact?{impact_query}"),
        &Value::Null,
    );
    assert_eq!(impact_get["results"][0]["summary"]["direction"], "up");
    assert!(items(&impact_get["results"][0]["blast_radius"]["downstream"]).is_empty());
    let impact_post = ok(
        &app,
        "POST",
        "/impact",
        &json!({"target": "create_user", "repo": repo_arg, "direction": "both", "depth": 2, "max_nodes": 100}),
    );
    assert!(
        IMPACT_FIELDS
            .iter()
            .all(|f| impact_post["results"][0].get(f).is_some())
    );

    let cli_search = run_lode_json(
        data,
        &["search", q, "--repo", repo_arg, "--limit", "5", "--json"],
        &[],
    );
    let cli_context = run_lode_json(
        data,
        &[
            "context", q, "--repo", repo_arg, "--budget", "2000", "--limit", "5", "--json",
        ],
        &[],
    );
    let cli_impact = run_lode_json(
        data,
        &[
            "impact",
            "create_user",
            "--repo",
            repo_arg,
            "--depth",
            "2",
            "--max-nodes",
            "100",
            "--json",
        ],
        &[],
    );
    assert_eq!(search_tuples(&cli_search), search_tuples(&search_q));
    assert_eq!(paths(&cli_context, "top_hits"), paths(&context, "top_hits"));
    assert_eq!(
        paths(&cli_context, "must_read"),
        paths(&context, "must_read")
    );
    for key in ["callers", "callees", "files", "upstream", "downstream"] {
        assert_eq!(
            cli_impact["results"][0]["summary"][key], impact_post["results"][0]["summary"][key],
            "{key}"
        );
    }

    for (method, body) in [("GET", &b""[..]), ("POST", &b"{}"[..])] {
        let (status, content_type, payload) = call(&app, method, "/missing", body);
        assert_eq!(status, 404);
        assert!(content_type.starts_with("application/json"));
        assert_eq!(payload, json!({"ok": false, "error": "not found"}));
    }

    let missing_index = json!({"path": repo.join("missing").to_str().unwrap()}).to_string();
    let invalid: [(&str, &str, &[u8]); 9] = [
        ("POST", "/search", b"{not-json"),
        ("POST", "/search", b"[]"),
        ("POST", "/context", b"{}"),
        ("POST", "/index", missing_index.as_bytes()),
        ("GET", "/search?limit=not-int", b""),
        ("GET", "/impact?target=create_user&depth=not-int", b""),
        ("GET", "/impact?target=create_user&max_nodes=not-int", b""),
        ("GET", "/impact?target=create_user&direction=sideways", b""),
        ("GET", "/impact", b""),
    ];
    for (method, path, body) in invalid {
        let (status, content_type, payload) = call(&app, method, path, body);
        assert_eq!(status, 400, "{method} {path}: {payload}");
        assert!(content_type.starts_with("application/json"));
        assert_eq!(payload["ok"], false);
        assert!(!payload["error"].as_str().unwrap().is_empty());
    }
    assert_eq!(ok(&app, "GET", "/health", &Value::Null)["ok"], true);
}

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_for_health(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok((200, _, _)) = http(port, "GET", "/health", &[]) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("daemon on port {port} did not become healthy");
}

fn get_json(port: u16, path: &str) -> Value {
    let (status, _, body) = http(port, "GET", path, &[]).unwrap();
    assert_eq!(status, 200, "{path}: {body}");
    serde_json::from_str(&body).unwrap()
}

#[test]
fn lode_serve_mirrors_loded_for_core_api_shapes() {
    let repo_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    let (repo, data) = (repo_dir.path(), data_dir.path());
    write_fixture_repo(repo, "UserService");
    run_lode_json(data, &["index", repo.to_str().unwrap(), "--json"], &[]);
    let data_arg = data.to_str().unwrap();
    let mut responses = Vec::new();
    for (bin, serve) in [(env!("CARGO_BIN_EXE_loded"), false), (lode_bin(), true)] {
        let port = free_port().to_string();
        let mut args = vec!["--data-dir", data_arg];
        if serve {
            args.push("serve");
        }
        args.extend(["--host", "127.0.0.1", "--port", &port]);
        let child = Command::new(bin)
            .args(&args)
            .env_remove("LODE_DATA_DIR")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let _daemon = Daemon(child);
        let port: u16 = port.parse().unwrap();
        wait_for_health(port);
        let (_, headers, _) = http(
            port,
            "GET",
            "/health",
            &[("x-request-id", "contract-test-health")],
        )
        .unwrap();
        assert!(
            headers.contains(&("x-request-id".into(), "contract-test-health".into())),
            "{headers:?}"
        );
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "content-type" && v.starts_with("application/json"))
        );
        let status = get_json(port, "/status");
        let mut status_keys: Vec<String> = status.as_object().unwrap().keys().cloned().collect();
        status_keys.sort();
        let search_path = format!(
            "/search?{}",
            query(&[
                ("q", "UserService"),
                ("repo", repo.to_str().unwrap()),
                ("limit", "5")
            ])
        );
        responses.push((
            get_json(port, "/health"),
            status_keys,
            items(&status["repos"])
                .iter()
                .map(|e| e["root"].clone())
                .collect::<Vec<_>>(),
            search_tuples(&get_json(port, &search_path)),
        ));
    }
    assert_eq!(responses[0], responses[1]);
    assert!(!responses[0].3.is_empty());
}
