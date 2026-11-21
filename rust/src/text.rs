//! Small text helpers that reproduce the Python standard-library behavior the index depends on
//! (identifier splitting, `str.splitlines`, character-based slicing, `repr` length).

use std::sync::LazyLock;

use regex::Regex;

/// Compile a pattern literal. Every pattern in the crate is a constant exercised by the unit
/// tests, so a failure here is a programming error.
#[allow(
    clippy::expect_used,
    reason = "pattern literals are fixed at compile time"
)]
pub fn static_regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("pattern literal is valid")
}

/// `[A-Za-z0-9_]+`
pub static WORD_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"[A-Za-z0-9_]+"));

/// Python's `[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+` with `findall`. The `regex` crate has no
/// lookahead, so the alternation is walked by hand. Only ASCII letters and digits match.
pub fn camel_parts(chunk: &str) -> Vec<&str> {
    let bytes = chunk.as_bytes();
    let len = bytes.len();
    let run =
        |from: usize, pred: fn(&u8) -> bool| bytes[from..].iter().take_while(|b| pred(b)).count();
    let mut parts = Vec::new();
    let mut index = 0;
    while index < len {
        let byte = bytes[index];
        if byte.is_ascii_lowercase() {
            let end = index + run(index, u8::is_ascii_lowercase);
            parts.push(&chunk[index..end]);
            index = end;
        } else if byte.is_ascii_uppercase() {
            let lower = run(index + 1, u8::is_ascii_lowercase);
            if lower > 0 {
                let end = index + 1 + lower;
                parts.push(&chunk[index..end]);
                index = end;
                continue;
            }
            let upper = run(index, u8::is_ascii_uppercase);
            let take = if index + upper == len {
                upper
            } else {
                upper - 1
            };
            if take == 0 {
                index += 1;
            } else {
                parts.push(&chunk[index..index + take]);
                index += take;
            }
        } else if byte.is_ascii_digit() {
            let end = index + run(index, u8::is_ascii_digit);
            parts.push(&chunk[index..end]);
            index = end;
        } else {
            index += chunk[index..].chars().next().map_or(1, char::len_utf8);
        }
    }
    parts
}

/// Camel/snake parts of every `_`-separated chunk of `word`.
pub fn split_word_parts(word: &str) -> Vec<&str> {
    word.split('_')
        .filter(|chunk| !chunk.is_empty())
        .flat_map(camel_parts)
        .collect()
}

/// `str.splitlines()`: splits on every line boundary Python recognizes, `\r\n` counting once.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let is_break = matches!(
            ch,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        lines.push(&text[start..index]);
        let mut next = index + ch.len_utf8();
        if ch == '\r'
            && let Some(&(_, '\n')) = chars.peek()
        {
            chars.next();
            next += 1;
        }
        start = next;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `value[:count]` counted in characters, as Python slices strings.
pub fn take_chars(value: &str, count: usize) -> &str {
    match value.char_indices().nth(count) {
        Some((index, _)) => &value[..index],
        None => value,
    }
}

pub fn char_len(value: &str) -> usize {
    value.chars().count()
}

/// Length of Python's `repr()` of a `str`.
pub fn py_repr_len(value: &str) -> usize {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    2 + value
        .chars()
        .map(|ch| match ch {
            '\\' | '\t' | '\n' | '\r' => 2,
            _ if ch == quote => 2,
            _ if (ch as u32) < 0x20 || ('\u{7f}'..='\u{a0}').contains(&ch) => 4,
            _ => 1,
        })
        .sum::<usize>()
}

/// `str.strip(chars)` for a set of ASCII characters.
pub fn strip_chars<'a>(value: &'a str, chars: &[char]) -> &'a str {
    value.trim_matches(|ch| chars.contains(&ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_parts_match_python_regex() {
        assert_eq!(camel_parts("HTTPServer"), ["HTTP", "Server"]);
        assert_eq!(camel_parts("parseHTTP"), ["parse", "HTTP"]);
        assert_eq!(camel_parts("getV2Name"), ["get", "2", "Name"]);
        assert_eq!(camel_parts("aB1"), ["a", "1"]);
        assert_eq!(
            split_word_parts("build_context_pack"),
            ["build", "context", "pack"]
        );
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines("a\r\nb\rc\n"), ["a", "b", "c"]);
        assert_eq!(splitlines("a\x0cb"), ["a", "b"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn repr_len_matches_python() {
        assert_eq!(py_repr_len("abc"), 5);
        assert_eq!(py_repr_len("it's"), 6);
        assert_eq!(py_repr_len("a\nb"), 6);
    }
}
