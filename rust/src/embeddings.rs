//! Local-only embeddings client (`embeddings.py`). The endpoint must be loopback or the
//! Docker service `embeddings`; requests are plain HTTP/1.1 over `std::net`.

use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream};
use std::time::Duration;

use serde_json::{Value, json};

use crate::{Error, Result};

const LOCAL_EMBEDDING_HOSTS: [&str; 2] = ["localhost", "embeddings"];
const TIMEOUT: Duration = Duration::from_secs(60);

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

pub fn embeddings_url() -> Option<String> {
    env_nonempty("LODE_EMBEDDINGS_URL").or_else(|| env_nonempty("KG_EMBEDDINGS_URL"))
}

pub fn embeddings_model() -> String {
    env_nonempty("LODE_EMBEDDINGS_MODEL")
        .or_else(|| env_nonempty("KG_EMBEDDINGS_MODEL"))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// A validated local endpoint: host, port and base path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub url: String,
    host: String,
    port: u16,
    path: String,
}

pub fn local_embeddings_endpoint(url: Option<&str>) -> Result<Endpoint> {
    let configured = url
        .map(str::to_owned)
        .or_else(embeddings_url)
        .unwrap_or_default();
    let endpoint = configured.trim_end_matches('/').to_owned();
    if endpoint.is_empty() {
        return Err(Error::runtime(
            "No local embeddings endpoint configured. Set LODE_EMBEDDINGS_URL.",
        ));
    }
    let parsed = url::Url::parse(&endpoint)
        .ok()
        .filter(|u| {
            matches!(u.scheme(), "http" | "https") && u.host_str().is_some_and(|h| !h.is_empty())
        })
        .ok_or_else(|| Error::runtime("Embeddings endpoint must be a local HTTP URL."))?;
    let host = parsed.host_str().unwrap_or_default().to_lowercase();
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let loopback = bare.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if !LOCAL_EMBEDDING_HOSTS.contains(&bare) && !loopback {
        return Err(Error::runtime(
            "Embeddings endpoint must be local, use loopback or Docker service 'embeddings'.",
        ));
    }
    if parsed.scheme() == "https" {
        return Err(Error::runtime(
            "HTTPS embeddings endpoints are not supported by the Rust port; use plain HTTP on loopback.",
        ));
    }
    Ok(Endpoint {
        port: parsed.port_or_known_default().unwrap_or(80),
        path: parsed.path().trim_end_matches('/').to_owned(),
        host,
        url: endpoint,
    })
}

fn post_json(endpoint: &Endpoint, body: &str) -> std::io::Result<(u16, Vec<u8>)> {
    let mut stream = TcpStream::connect((
        endpoint.host.trim_start_matches('[').trim_end_matches(']'),
        endpoint.port,
    ))?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    let request = format!(
        "POST {}/embed HTTP/1.1\r\nHost: {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nConnection: close\r\n\r\n{body}",
        endpoint.path,
        endpoint.host,
        body.len()
    );
    stream.write_all(request.as_bytes())?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    parse_response(&raw).ok_or_else(|| std::io::Error::other("malformed HTTP response"))
}

fn parse_response(raw: &[u8]) -> Option<(u16, Vec<u8>)> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    let status = head.split_whitespace().nth(1)?.parse().ok()?;
    let body = &raw[split + 4..];
    let chunked = head.lines().any(|line| {
        line.to_lowercase().starts_with("transfer-encoding:")
            && line.to_lowercase().contains("chunked")
    });
    Some((
        status,
        if chunked {
            dechunk(body)?
        } else {
            body.to_vec()
        },
    ))
}

fn dechunk(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = body.windows(2).position(|w| w == b"\r\n")?;
        let size_text = std::str::from_utf8(&body[..line_end]).ok()?;
        let size = usize::from_str_radix(size_text.split(';').next()?.trim(), 16).ok()?;
        body = &body[line_end + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(body.get(..size)?);
        body = body.get(size + 2..)?;
    }
}

/// POST `{"inputs": texts}` to `<endpoint>/embed`, retrying transport errors with backoff.
pub fn embed_texts(texts: &[String], url: Option<&str>, retries: u32) -> Result<Vec<Value>> {
    let endpoint = local_embeddings_endpoint(url)?;
    let payload = crate::json::dumps(&json!({"inputs": texts}));
    let mut attempt = 0;
    let (status, body) = loop {
        match post_json(&endpoint, &payload) {
            Ok(response) => break response,
            Err(_) if attempt < retries => {
                std::thread::sleep(Duration::from_millis(250 * (1 << attempt)));
                attempt += 1;
            }
            Err(_) => return Err(Error::runtime("Embedding endpoint failed after retries")),
        }
    };
    if !(200..300).contains(&status) {
        return Err(Error::runtime(format!("HTTP Error {status}")));
    }
    match serde_json::from_slice::<Value>(&body)? {
        Value::Array(items) => Ok(items),
        Value::Object(mut map) => match map.remove("embeddings") {
            Some(Value::Array(items)) => Ok(items),
            _ => Err(Error::runtime("Unexpected embeddings response shape")),
        },
        _ => Err(Error::runtime("Unexpected embeddings response shape")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_local_endpoints_are_accepted() {
        assert_eq!(
            local_embeddings_endpoint(Some("http://127.0.0.1:8080/"))
                .unwrap()
                .url,
            "http://127.0.0.1:8080"
        );
        assert!(local_embeddings_endpoint(Some("http://embeddings")).is_ok());
        assert!(local_embeddings_endpoint(Some("http://[::1]:9/")).is_ok());
        let remote = local_embeddings_endpoint(Some("http://example.com")).unwrap_err();
        assert!(remote.to_string().contains("must be local"));
        assert!(local_embeddings_endpoint(Some("ftp://localhost")).is_err());
    }

    #[test]
    fn chunked_bodies_are_decoded() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n[1]\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw), Some((200, b"[1]".to_vec())));
    }
}
