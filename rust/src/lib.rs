//! Lode: a fully local repository knowledge graph (Rust port of the Python `lode` package).
//!
//! The SQLite schema, node ids and JSON shapes match the Python implementation so both
//! can read and write the same database.

pub mod cli;
pub mod config;
pub mod context;
pub mod daemon;
pub mod embeddings;
pub mod error;
pub mod features;
pub mod gitignore;
pub mod graph;
pub mod indexer;
pub mod json;
pub mod kuzu;
pub mod model;
pub mod observability;
pub mod pyast;
#[cfg(feature = "embed")]
pub mod rerank;
pub mod storage;
pub mod text;

pub use error::{Error, Result};

/// Package version, kept equal to the Python package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
