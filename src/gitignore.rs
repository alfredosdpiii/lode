//! Lightweight `.gitignore` matcher for directory walks (supports nested `.gitignore`,
//! `.git/info/exclude`, comments, `!` negations, anchored `/`, directory-only `/`, and globs).

use std::path::Path;

use regex::Regex;

#[derive(Debug)]
struct Rule {
    negated: bool,
    dir_only: bool,
    regex: Regex,
}

/// Compiled rules from one `.gitignore` (or `.git/info/exclude`) file, scoped to `base_rel`.
#[derive(Debug)]
pub struct IgnoreFile {
    base_rel: String,
    rules: Vec<Rule>,
}

impl IgnoreFile {
    pub fn from_path(path: &Path, base_rel: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let rules = parse_rules(&text);
        (!rules.is_empty()).then(|| Self {
            base_rel: base_rel.to_owned(),
            rules,
        })
    }

    pub fn from_str(text: &str, base_rel: &str) -> Self {
        Self {
            base_rel: base_rel.to_owned(),
            rules: parse_rules(text),
        }
    }

    fn matches(&self, rel_path: &str, is_dir: bool) -> Option<bool> {
        let local = if self.base_rel.is_empty() {
            rel_path
        } else {
            let prefix = format!("{}/", self.base_rel);
            rel_path.strip_prefix(&prefix)?
        };
        let mut decision = None;
        for rule in &self.rules {
            if rule.dir_only && !is_dir {
                continue;
            }
            if rule.regex.is_match(local) {
                decision = Some(!rule.negated);
            }
        }
        decision
    }
}

/// Evaluate a stack of `IgnoreFile`s from outermost (root) to innermost directory.
/// Later matching rules override earlier ones, matching Git's precedence.
pub fn is_ignored(stack: &[IgnoreFile], rel_path: &str, is_dir: bool) -> bool {
    let mut ignored = false;
    for file in stack {
        if let Some(decision) = file.matches(rel_path, is_dir) {
            ignored = decision;
        }
    }
    ignored
}

fn strip_trailing_unescaped_spaces(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b' ' {
        let mut backslashes = 0;
        let mut i = end - 1;
        while i > 0 && bytes[i - 1] == b'\\' {
            backslashes += 1;
            i -= 1;
        }
        if backslashes % 2 == 1 {
            break;
        }
        end -= 1;
    }
    &line[..end]
}

fn parse_rules(text: &str) -> Vec<Rule> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = strip_trailing_unescaped_spaces(raw.trim_end_matches('\r'));
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (negated, mut pat) = if let Some(rest) = line.strip_prefix('!') {
            (true, rest)
        } else {
            (false, line)
        };
        if pat.is_empty() {
            continue;
        }
        let dir_only = pat.ends_with('/');
        if dir_only {
            pat = pat.trim_end_matches('/');
        }
        if pat.is_empty() {
            continue;
        }
        let leading_slash = pat.starts_with('/');
        if leading_slash {
            pat = pat.trim_start_matches('/');
        }
        if pat.is_empty() {
            continue;
        }
        let anchored = leading_slash || pat.contains('/');
        let body = glob_to_regex(pat);
        let full = if anchored {
            format!("^{body}$")
        } else {
            format!("(?:^|/){body}$")
        };
        if let Ok(regex) = Regex::new(&full) {
            out.push(Rule {
                negated,
                dir_only,
                regex,
            });
        }
    }
    out
}

fn glob_to_regex(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' if i + 1 < chars.len() => {
                out.push_str(&regex::escape(&chars[i + 1].to_string()));
                i += 2;
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    let prev_slash = i == 0 || chars[i - 1] == '/';
                    let next_slash = i + 2 < chars.len() && chars[i + 2] == '/';
                    if prev_slash && next_slash {
                        out.push_str("(?:.*/)?");
                        i += 3;
                    } else {
                        out.push_str(".*");
                        i += 2;
                    }
                } else {
                    out.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                out.push_str("[^/]");
                i += 1;
            }
            '[' => {
                if let Some(close) = chars[i + 1..].iter().position(|&c| c == ']') {
                    let end = i + 1 + close;
                    out.push('[');
                    let mut j = i + 1;
                    if j < end && (chars[j] == '!' || chars[j] == '^') {
                        out.push('^');
                        j += 1;
                    }
                    while j < end {
                        if chars[j] == '\\' && j + 1 < end {
                            out.push('\\');
                            out.push(chars[j + 1]);
                            j += 2;
                        } else {
                            out.push(chars[j]);
                            j += 1;
                        }
                    }
                    out.push(']');
                    i = end + 1;
                } else {
                    out.push_str("\\[");
                    i += 1;
                }
            }
            other => {
                out.push_str(&regex::escape(&other.to_string()));
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_patterns_negation_and_anchoring() {
        let root = IgnoreFile::from_str(
            "# comment\n*.log\n!important.log\n/root_only.py\ndata/\ndocs/**/*.md\n",
            "",
        );
        let stack = [root];
        assert!(is_ignored(&stack, "a.log", false));
        assert!(is_ignored(&stack, "sub/a.log", false));
        assert!(!is_ignored(&stack, "important.log", false));
        assert!(!is_ignored(&stack, "sub/important.log", false));
        assert!(is_ignored(&stack, "root_only.py", false));
        assert!(!is_ignored(&stack, "sub/root_only.py", false));
        assert!(is_ignored(&stack, "data", true));
        assert!(!is_ignored(&stack, "data", false));
        assert!(is_ignored(&stack, "docs/a.md", false));
        assert!(is_ignored(&stack, "docs/a/b/c.md", false));
    }

    #[test]
    fn nested_gitignore_overrides_parent() {
        let root = IgnoreFile::from_str("*.py\n", "");
        let nested = IgnoreFile::from_str("!keep.py\n", "pkg");
        let stack = [root, nested];
        assert!(is_ignored(&stack, "drop.py", false));
        assert!(is_ignored(&stack, "pkg/drop.py", false));
        assert!(!is_ignored(&stack, "pkg/keep.py", false));
    }
}
