//! JSON text in the exact shape Python's `json.dumps` produces: ASCII-only output, `", "` and
//! `": "` separators for compact text, two-space indentation for pretty text. Object keys come
//! out sorted because `serde_json::Map` is a `BTreeMap` (the Python code always passes
//! `sort_keys=True`).

use std::io;

use serde::Serialize;
use serde_json::ser::Formatter;

struct PyCompact;

impl Formatter for PyCompact {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

fn to_value<T: Serialize + ?Sized>(value: &T) -> serde_json::Value {
    // Going through `Value` sorts every object's keys, like `sort_keys=True`.
    serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
}

/// `json.dumps(value, sort_keys=True)`.
pub fn dumps<T: Serialize + ?Sized>(value: &T) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyCompact);
    // Serializing a `Value` into a Vec cannot fail.
    let _ = to_value(value).serialize(&mut ser);
    ascii_escape(&String::from_utf8_lossy(&out))
}

/// `json.dumps(value, indent=2, sort_keys=True)`.
pub fn dumps_pretty<T: Serialize + ?Sized>(value: &T) -> String {
    let text = serde_json::to_string_pretty(&to_value(value)).unwrap_or_default();
    ascii_escape(&text)
}

/// Non-ASCII characters only occur inside JSON strings, so escaping the whole text is safe.
fn ascii_escape(text: &str) -> String {
    if text.is_ascii() && !text.contains('\x7f') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() + 16);
    for ch in text.chars() {
        if (ch as u32) < 0x7f {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_dumps() {
        let value = json!({"b": [1, "é"], "a": null, "c": {}});
        assert_eq!(dumps(&value), r#"{"a": null, "b": [1, "\u00e9"], "c": {}}"#);
        assert_eq!(dumps_pretty(&json!({"x": []})), "{\n  \"x\": []\n}");
    }
}
