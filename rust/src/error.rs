use std::fmt;

/// Errors surfaced by the library. `Display` is the message the Python CLI and daemon print.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// A path the caller named does not exist (Python `FileNotFoundError`).
    #[error("{0}")]
    NotFound(String),
    /// Bad caller input (Python `ValueError` / `TypeError`).
    #[error("{0}")]
    InvalidInput(String),
    /// Anything else the Python code raised as `RuntimeError`.
    #[error("{0}")]
    Runtime(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub fn invalid(message: impl fmt::Display) -> Self {
        Self::InvalidInput(message.to_string())
    }

    pub fn runtime(message: impl fmt::Display) -> Self {
        Self::Runtime(message.to_string())
    }
}
