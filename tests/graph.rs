mod common;

use common::{Fixture, by_path, by_qname, first, resolved_calls, write_multi_file_repo};
use lode::graph::{Direction, blast_radius, impact_report, impact_targets, resolve_graph};
use serde_json::Value;

fn multi() -> Fixture {
    let fx = Fixture::new();
    write_multi_file_repo(fx.repo());
    fx
}

fn qnames(entries: &Value) -> Vec<String> {
    entries
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|e| e["node"]["qname"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn len(value: &Value) -> usize {
    value.as_array().map_or(0, Vec::len)
}

#[test]
fn resolves_file_import_edges() {
    let fx = multi();
    let stats = fx.index();
    let mut conn = fx.conn();
    let counts = resolve_graph(&mut conn, &stats.repo_id).unwrap();
    assert!(counts.imports > 0);
}

#[test]
fn resolves_cross_file_calls_via_imports() {
    let fx = multi();
    let stats = fx.index();
    let conn = fx.conn();
    let target = first(&conn, "greet");
    assert!(resolved_calls(&conn, &stats.repo_id, None, Some(&target.id)) >= 1);
}

#[test]
fn resolves_extends_to_local_class() {
    let fx = Fixture::new();
    fx.write(
        "base.py",
        "class Animal:\n    def speak(self):\n        pass\n",
    );
    fx.write(
        "dog.py",
        "from base import Animal\n\nclass Dog(Animal):\n    def speak(self):\n        return 'woof'\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let dog = first(&conn, "Dog");
    let kind: String = conn
        .query_row(
            "SELECT n.kind FROM edges e JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
             WHERE e.repo_id = ? AND e.src = ? AND e.kind = 'EXTENDS' AND e.confidence = 'resolved'",
            [&stats.repo_id, &dog.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(kind, "Class");
}

#[test]
fn relative_imports_resolve() {
    let fx = Fixture::new();
    fx.write("pkg/__init__.py", "");
    fx.write("pkg/a.py", "def fn_a():\n    pass\n");
    fx.write(
        "pkg/b.py",
        "from .a import fn_a\n\ndef fn_b():\n    return fn_a()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let fn_a = first(&conn, "fn_a");
    assert!(resolved_calls(&conn, &stats.repo_id, None, Some(&fn_a.id)) >= 1);
}

#[test]
fn star_import_resolution() {
    let fx = Fixture::new();
    fx.write("lib.py", "def exported_func():\n    return 1\n");
    fx.write(
        "consumer.py",
        "from lib import *\n\ndef caller():\n    return exported_func()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let target = first(&conn, "exported_func");
    assert!(resolved_calls(&conn, &stats.repo_id, None, Some(&target.id)) >= 1);
}

#[test]
fn ts_js_named_import_calls_resolve() {
    let fx = Fixture::new();
    fx.write("lib.ts", "export function fn() { return 1; }\n");
    fx.write(
        "app.ts",
        "import { fn } from \"./lib\";\nexport function main() { return fn(); }\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let target = first(&conn, "fn");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, None, Some(&target.id)),
        1
    );
}

#[test]
fn python_import_syntaxes_resolve_calls() {
    let fx = Fixture::new();
    fx.write("pkg/__init__.py", "");
    fx.write("lib.py", "def fn():\n    return 1\n");
    fx.write("pkg/mod.py", "def inner():\n    return 2\n");
    fx.write("pkg/mod2.py", "def named():\n    return 3\n");
    fx.write(
        "consumer.py",
        "import lib\nimport lib as l\nfrom pkg import mod\nfrom pkg.mod2 import named as alias\n\n\
         def caller():\n    lib.fn()\n    l.fn()\n    mod.inner()\n    alias()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    for (name, minimum) in [("fn", 2), ("inner", 1), ("named", 1)] {
        let target = first(&conn, name);
        let resolved = resolved_calls(&conn, &stats.repo_id, None, Some(&target.id));
        assert!(resolved >= minimum, "{name}: {resolved}");
    }
}

#[test]
fn imports_disambiguate_duplicate_symbol_names() {
    let fx = Fixture::new();
    fx.write("a.py", "def render():\n    return 'a'\n");
    fx.write("b.py", "def render():\n    return 'b'\n");
    fx.write(
        "consumer.py",
        "from a import render\n\ndef caller():\n    return render()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let a_render = by_path(&conn, "render", "a.py");
    let b_render = by_path(&conn, "render", "b.py");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, None, Some(&a_render.id)),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, None, Some(&b_render.id)),
        0
    );
}

#[test]
fn self_method_resolves_to_containing_class() {
    let fx = Fixture::new();
    fx.write(
        "models.py",
        "class A:\n    def save(self):\n        return 'a'\n    def run(self):\n        return self.save()\n\n\
         class B:\n    def save(self):\n        return 'b'\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let run = by_qname(&conn, "run", "models.A.run");
    let a_save = by_qname(&conn, "save", "models.A.save");
    let b_save = by_qname(&conn, "save", "models.B.save");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&run.id), Some(&a_save.id)),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&run.id), Some(&b_save.id)),
        0
    );
}

#[test]
fn duplicate_methods_do_not_resolve_by_bare_name() {
    let fx = Fixture::new();
    fx.write(
        "models.py",
        "class A:\n    def save(self):\n        return 'a'\n\nclass B:\n    def save(self):\n        return 'b'\n\n\
         def caller(obj):\n    return obj.save()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let caller = first(&conn, "caller");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&caller.id), None),
        0
    );
}

#[test]
fn blast_radius_traverses_upstream() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let result = blast_radius(&conn, &greet.id, Some(3), 1000, Direction::Up).unwrap();
    assert!(qnames(&result["upstream"]).contains(&"services.auth.authenticate".to_string()));
}

#[test]
fn blast_radius_traverses_downstream() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let main_fn = first(&conn, "main");
    let result = blast_radius(&conn, &main_fn.id, Some(3), 1000, Direction::Down).unwrap();
    assert!(qnames(&result["downstream"]).contains(&"app.login".to_string()));
}

#[test]
fn blast_radius_respects_depth() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let r1 = blast_radius(&conn, &greet.id, Some(1), 1000, Direction::Up).unwrap();
    let r3 = blast_radius(&conn, &greet.id, Some(3), 1000, Direction::Up).unwrap();
    assert!(len(&r1["upstream"]) <= len(&r3["upstream"]));
}

#[test]
fn blast_radius_defaults_to_full_reachable_traversal() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let result = blast_radius(&conn, &greet.id, None, 1000, Direction::Up).unwrap();
    assert_eq!(result["depth"], Value::Null);
    assert_eq!(result["depth_label"], "all");
    assert!(qnames(&result["upstream"]).contains(&"runner.main".to_string()));
}

#[test]
fn blast_radius_finds_entrypoints() {
    let fx = Fixture::new();
    fx.write(
        "app.py",
        "from fastapi import FastAPI\n\napp = FastAPI()\n\ndef process(data):\n    return data\n\n\
         @app.post('/process')\ndef handle_process(data: str):\n    return process(data)\n",
    );
    fx.index();
    let conn = fx.conn();
    let process = first(&conn, "process");
    let result = blast_radius(&conn, &process.id, Some(2), 1000, Direction::Up).unwrap();
    let entrypoints = result["entrypoints"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        entrypoints.iter().any(|ep| ep["node"]["kind"] == "Route"),
        "{result}"
    );
}

#[test]
fn blast_radius_includes_files() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let result = blast_radius(&conn, &greet.id, Some(3), 1000, Direction::Both).unwrap();
    let files: Vec<&str> = result["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert!(files.contains(&"models/user.py"));
}

#[test]
fn blast_radius_max_nodes_truncates() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let result = blast_radius(&conn, &greet.id, Some(5), 1, Direction::Up).unwrap();
    assert_eq!(result["truncated"], true);
}

#[test]
fn blast_radius_on_nonexistent_node() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let result = blast_radius(&conn, "nonexistent_id", Some(3), 1000, Direction::Both).unwrap();
    assert_eq!(result, serde_json::json!({}));
}

#[test]
fn blast_radius_labels_file_import_expansion() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let result = blast_radius(&conn, &greet.id, None, 1000, Direction::Up).unwrap();
    let entries = result["upstream"].as_array().unwrap();
    assert!(
        entries
            .iter()
            .any(|e| e["via"] == "IMPORTS" && e["scope"] == "conservative_file"),
        "{entries:?}"
    );
}

#[test]
fn impact_report_includes_blast_radius() {
    let fx = multi();
    fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let report = impact_report(&conn, &greet, 200, Some(3), 1000, Direction::Both).unwrap();
    assert!(report.get("blast_radius").is_some());
    assert!(report["summary"]["upstream"].as_i64().unwrap() > 0);
    let files: Vec<&str> = report["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap())
        .collect();
    assert!(files.contains(&"models/user.py"));
}

#[test]
fn impact_targets_by_name() {
    let fx = multi();
    let stats = fx.index();
    let conn = fx.conn();
    let targets = impact_targets(&conn, "greet", Some(&stats.repo_id), 5).unwrap();
    assert_eq!(targets[0].name, "greet");
}

#[test]
fn impact_targets_by_node_id() {
    let fx = multi();
    let stats = fx.index();
    let conn = fx.conn();
    let greet = first(&conn, "greet");
    let targets = impact_targets(&conn, &greet.id, Some(&stats.repo_id), 5).unwrap();
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].id, greet.id);
}

#[test]
fn impact_report_filters_resolved_external_placeholder() {
    let fx = Fixture::new();
    fx.write("lib.py", "def fn():\n    return 1\n");
    fx.write(
        "consumer.py",
        "from lib import fn\n\ndef caller():\n    return fn()\n",
    );
    fx.index();
    let conn = fx.conn();
    let caller = first(&conn, "caller");
    let report = impact_report(&conn, &caller, 200, Some(1), 1000, Direction::Down).unwrap();
    let kinds: Vec<&str> = report["callees"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["node"]["kind"].as_str())
        .collect();
    assert!(kinds.contains(&"Function"));
    assert!(!kinds.contains(&"ExternalSymbol"));
}

#[test]
fn receiver_assignment_and_annotation_disambiguate_methods() {
    let fx = Fixture::new();
    fx.write(
        "a.py",
        "class A:\n    def save(self):\n        return 'a'\n",
    );
    fx.write(
        "b.py",
        "class B:\n    def save(self):\n        return 'b'\n",
    );
    fx.write(
        "consumer.py",
        "from typing import Optional\nfrom a import A\nfrom b import B\n\n\
         def via_ctor():\n    svc = A()\n    return svc.save()\n\n\
         def via_param(svc: Optional[B]):\n    return svc.save()\n\n\
         def via_ann(factory):\n    svc: A = factory()\n    return svc.save()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let a_save = by_qname(&conn, "save", "a.A.save");
    let b_save = by_qname(&conn, "save", "b.B.save");
    let via_ctor = first(&conn, "via_ctor");
    let via_param = first(&conn, "via_param");
    let via_ann = first(&conn, "via_ann");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&via_ctor.id), Some(&a_save.id)),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&via_ctor.id), Some(&b_save.id)),
        0
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&via_param.id), Some(&b_save.id)),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&via_ann.id), Some(&a_save.id)),
        1
    );
}

#[test]
fn self_attr_and_inherited_methods_resolve() {
    let fx = Fixture::new();
    fx.write(
        "base.py",
        "class Base:\n    def execute(self):\n        return 1\n",
    );
    fx.write(
        "other.py",
        "class Other:\n    def execute(self):\n        return 2\n",
    );
    fx.write(
        "app.py",
        "from base import Base\n\n\
         class Sub(Base):\n    def run(self):\n        return self.execute()\n\n\
         class Controller:\n    def __init__(self):\n        self.worker = Sub()\n    \
         def handle(self):\n        return self.worker.execute()\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let base_exec = by_qname(&conn, "execute", "base.Base.execute");
    let other_exec = by_qname(&conn, "execute", "other.Other.execute");
    let sub_run = by_qname(&conn, "run", "app.Sub.run");
    let handle = by_qname(&conn, "handle", "app.Controller.handle");
    assert_eq!(
        resolved_calls(
            &conn,
            &stats.repo_id,
            Some(&sub_run.id),
            Some(&base_exec.id)
        ),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&handle.id), Some(&base_exec.id)),
        1
    );
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, None, Some(&other_exec.id)),
        0
    );
}

#[test]
fn dotted_call_on_unknown_receiver_does_not_false_match_unique_method() {
    let fx = Fixture::new();
    fx.write(
        "store.py",
        "class LocalStore:\n    def execute(self):\n        return 1\n",
    );
    fx.write(
        "db.py",
        "import sqlite3\n\ndef query(conn):\n    return conn.execute('SELECT 1')\n",
    );
    let stats = fx.index();
    let conn = fx.conn();
    let query = first(&conn, "query");
    let store_exec = by_qname(&conn, "execute", "store.LocalStore.execute");
    assert_eq!(
        resolved_calls(&conn, &stats.repo_id, Some(&query.id), Some(&store_exec.id)),
        0
    );
}

#[test]
fn blast_radius_filters_unrelated_named_imports() {
    let fx = Fixture::new();
    fx.write(
        "utils.py",
        "def helper_a():\n    return 'a'\n\ndef helper_b():\n    return 'b'\n",
    );
    fx.write(
        "uses_a.py",
        "from utils import helper_a\n\ndef run_a():\n    return helper_a()\n",
    );
    fx.write(
        "uses_b.py",
        "from utils import helper_b\n\ndef run_b():\n    return helper_b()\n",
    );
    fx.write(
        "uses_mod.py",
        "import utils\n\ndef run_mod():\n    return utils.helper_b()\n",
    );
    fx.index();
    let conn = fx.conn();
    let helper_b = first(&conn, "helper_b");
    let result = blast_radius(&conn, &helper_b.id, None, 1000, Direction::Up).unwrap();
    let files: Vec<&str> = result["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert!(files.contains(&"uses_b.py"), "{files:?}");
    assert!(files.contains(&"uses_mod.py"), "{files:?}");
    assert!(!files.contains(&"uses_a.py"), "{files:?}");
}

#[test]
fn blast_radius_does_not_cascade_transitive_file_imports_for_symbol() {
    let fx = Fixture::new();
    fx.write("leaf.py", "def target_fn():\n    return 1\n");
    fx.write(
        "mid.py",
        "import leaf\n\ndef unrelated_mid():\n    return 2\n",
    );
    fx.write(
        "top.py",
        "from mid import unrelated_mid\n\ndef entry():\n    return unrelated_mid()\n",
    );
    fx.index();
    let conn = fx.conn();
    let target = first(&conn, "target_fn");
    let result = blast_radius(&conn, &target.id, None, 1000, Direction::Up).unwrap();
    let files: Vec<&str> = result["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert!(files.contains(&"mid.py"), "{files:?}");
    assert!(!files.contains(&"top.py"), "{files:?}");
}
