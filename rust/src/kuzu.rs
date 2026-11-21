//! Kuzu projection. The Python package treats Kuzu as an optional extra; the Rust port has
//! no Kuzu binding, so syncing always reports that Kuzu is not installed.

use std::path::Path;

use serde_json::Value;

use crate::{Error, Result};

pub fn sync_from_sqlite(_conn: &rusqlite::Connection, _path: &Path) -> Result<Value> {
    Err(Error::runtime(if cfg!(feature = "kuzu") {
        "Kuzu is not installed. The Rust port has no Kuzu binding yet; use the Python package with `uv sync --extra kuzu`."
    } else {
        "Kuzu is not installed. Install with `uv sync --extra kuzu` or `pip install 'lode-kg[kuzu]'`."
    }))
}
