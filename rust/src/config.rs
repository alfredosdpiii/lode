use std::path::{Path, PathBuf};

use sha1::{Digest, Sha1};

pub const APP_NAME: &str = "lode";

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// `~`-expansion as `Path.expanduser` does it for the current user.
pub fn expand_user(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir();
    }
    match path.strip_prefix("~/") {
        Some(rest) => home_dir().join(rest),
        None => PathBuf::from(path),
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// `LODE_DATA_DIR`, then `KG_DATA_DIR`, then `~/.local/share/lode`.
pub fn default_data_dir() -> PathBuf {
    match env_nonempty("LODE_DATA_DIR").or_else(|| env_nonempty("KG_DATA_DIR")) {
        Some(dir) => expand_user(&dir),
        None => home_dir().join(".local").join("share").join(APP_NAME),
    }
}

pub fn sqlite_path(data_dir: &Path) -> PathBuf {
    data_dir.join("lode.sqlite")
}

pub fn kuzu_path(data_dir: &Path) -> PathBuf {
    data_dir.join("lode.kuzu")
}

/// Absolute, symlink-free path like `Path.resolve()`; falls back to a lexical absolute path
/// when the path does not exist.
pub fn resolve_path(path: &Path) -> PathBuf {
    let expanded = expand_user(&path.to_string_lossy());
    std::fs::canonicalize(&expanded)
        .unwrap_or_else(|_| std::path::absolute(&expanded).unwrap_or(expanded))
}

pub fn repo_id_for_root(root: &Path) -> String {
    let resolved = resolve_path(root);
    let digest = Sha1::digest(resolved.to_string_lossy().as_bytes());
    hex(&digest)[..16].to_owned()
}

pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

pub fn sha1_hex(bytes: &[u8]) -> String {
    hex(&Sha1::digest(bytes))
}
