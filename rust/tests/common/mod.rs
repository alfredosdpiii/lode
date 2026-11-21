// Each test crate uses a different subset of these fixtures.
#![allow(dead_code)]

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use lode::config::sqlite_path;
use lode::indexer::{IndexStats, index_repo};
use lode::storage::{self, NodeRow};
use rusqlite::Connection;
use serde_json::Value;
use tempfile::TempDir;

/// A repository directory and a data directory, both removed on drop.
pub struct Fixture {
    pub repo_dir: TempDir,
    pub data_dir: TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        Self {
            repo_dir: TempDir::new().unwrap(),
            data_dir: TempDir::new().unwrap(),
        }
    }

    pub fn repo(&self) -> &Path {
        self.repo_dir.path()
    }

    pub fn data(&self) -> &Path {
        self.data_dir.path()
    }

    pub fn db(&self) -> PathBuf {
        sqlite_path(self.data())
    }

    pub fn write(&self, rel: &str, text: &str) {
        write_file(self.repo(), rel, text);
    }

    pub fn index(&self) -> IndexStats {
        index_repo(self.repo(), &self.db()).unwrap()
    }

    pub fn conn(&self) -> Connection {
        storage::connect(&self.db()).unwrap()
    }
}

pub fn write_file(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

pub fn find(conn: &Connection, name: &str) -> Vec<NodeRow> {
    storage::find_symbol(conn, name, None, 20).unwrap()
}

pub fn first(conn: &Connection, name: &str) -> NodeRow {
    find(conn, name)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("no symbol {name}"))
}

pub fn by_qname(conn: &Connection, name: &str, qname: &str) -> NodeRow {
    find(conn, name)
        .into_iter()
        .find(|n| n.qname == qname)
        .unwrap()
}

pub fn by_path(conn: &Connection, name: &str, path: &str) -> NodeRow {
    find(conn, name)
        .into_iter()
        .find(|n| n.path == path)
        .unwrap()
}

/// Resolved CALLS edges matching the optional source and destination.
pub fn resolved_calls(
    conn: &Connection,
    repo_id: &str,
    src: Option<&str>,
    dst: Option<&str>,
) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM edges WHERE repo_id = ?1 AND (?2 IS NULL OR src = ?2)
         AND (?3 IS NULL OR dst = ?3) AND kind = 'CALLS' AND confidence = 'resolved'",
        rusqlite::params![repo_id, src, dst],
        |row| row.get(0),
    )
    .unwrap()
}

pub fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

pub fn write_multi_file_repo(repo: &Path) {
    write_file(repo, "models/__init__.py", "");
    write_file(
        repo,
        "models/user.py",
        "class User:\n    def __init__(self, name):\n        self.name = name\n    def greet(self):\n        return f'hello {self.name}'\n",
    );
    write_file(repo, "services/__init__.py", "");
    write_file(
        repo,
        "services/auth.py",
        "from models.user import User\n\ndef authenticate(name):\n    user = User(name)\n    return user.greet()\n",
    );
    write_file(
        repo,
        "app.py",
        "from services.auth import authenticate\n\ndef login(name):\n    return authenticate(name)\n",
    );
    write_file(
        repo,
        "runner.py",
        "from app import login\n\ndef main():\n    return login('world')\n",
    );
    write_file(repo, "orphan.py", "def standalone():\n    return 42\n");
}

pub const SAMPLE_APP: &str = "from fastapi import FastAPI

app = FastAPI()

class UserService:
    def create(self, name: str):
        \"\"\"Create a user record.\"\"\"
        return {\"name\": name}

def audit_user(name):
    return name

@app.post(\"/users\")
def create_user(name: str):
    audit_user(name)
    return UserService().create(name)
";

pub fn lode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lode")
}

/// Runs the `lode` binary with `--data-dir` (callers pass `--json` after the subcommand); returns exit code, stdout, stderr.
pub fn run_lode(data_dir: &Path, args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let mut cmd = Command::new(lode_bin());
    cmd.arg("--data-dir").arg(data_dir).args(args);
    cmd.env_remove("LODE_DATA_DIR")
        .env_remove("LODE_EMBEDDINGS_URL")
        .env_remove("LODE_EMBEDDINGS_MODEL");
    for (key, value) in env {
        cmd.env(key, value);
    }
    let out = cmd.output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Runs the `lode` binary and parses stdout as JSON, failing on a non-zero exit.
pub fn run_lode_json(data_dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Value {
    let (code, stdout, stderr) = run_lode(data_dir, args, env);
    assert_eq!(code, 0, "lode {args:?} failed: {stderr}\n{stdout}");
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("bad json {e}: {stdout}"))
}

pub const SAMPLE_REPO_APP: &str = r#"from fastapi import FastAPI

app = FastAPI()


class UserService:
    """Creates and stores users."""
    def save_user(self, name):
        return {"name": name}


def audit_user(name):
    return name


@app.post("/users")
def create_user(name: str):
    service = UserService()
    audit_user(name)
    return service.save_user(name)
"#;

/// The `write_sample_repo` fixture of tests/test_lode.py.
pub fn write_sample_repo(repo: &Path) {
    write_file(repo, "app.py", SAMPLE_REPO_APP);
    write_file(
        repo,
        "README.md",
        "# Sample\n\nThe create user route calls the user service.\n",
    );
}

/// A loopback embeddings server that answers each POST with one 3-dim vector per input:
/// `[base + i, base + i + 1, len(text)]`, recording the inputs of every request.
pub struct FakeEmbeddings {
    pub url: String,
    pub requests: Arc<Mutex<Vec<Vec<String>>>>,
    pub vector_base: Arc<Mutex<f64>>,
}

impl FakeEmbeddings {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let vector_base = Arc::new(Mutex::new(0.0));
        let (seen, base) = (Arc::clone(&requests), Arc::clone(&vector_base));
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                serve_embed(stream, &seen, &base);
            }
        });
        Self {
            url,
            requests,
            vector_base,
        }
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

fn serve_embed(mut stream: TcpStream, seen: &Mutex<Vec<Vec<String>>>, base: &Mutex<f64>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        if line == "\r\n" {
            break;
        }
        if let Some((key, value)) = line.split_once(':')
            && key.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    let inputs: Vec<String> = parsed["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|text| text.as_str().unwrap().to_owned())
        .collect();
    let base = *base.lock().unwrap();
    let vectors: Vec<[f64; 3]> = inputs
        .iter()
        .enumerate()
        .map(|(index, text)| {
            [
                base + index as f64,
                base + index as f64 + 1.0,
                text.chars().count() as f64,
            ]
        })
        .collect();
    seen.lock().unwrap().push(inputs);
    let payload = serde_json::to_string(&vectors).unwrap();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(response.as_bytes()).unwrap();
}

/// Status, lower-cased headers and body.
pub type HttpReply = (u16, Vec<(String, String)>, String);

/// One blocking HTTP/1.1 request to `127.0.0.1:port`; returns status, lower-cased headers and body.
pub fn http(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
) -> std::io::Result<HttpReply> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
    let extra: String = headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}\r\n"))
        .collect();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{extra}content-length: 0\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header end"))?;
    let mut lines = head.lines();
    let status = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_owned()))
        .collect();
    Ok((status, headers, body.to_owned()))
}

/// A port that was free a moment ago.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
