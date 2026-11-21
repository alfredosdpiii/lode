//! Python source extraction on top of `ruff_python_parser`, reproducing what `indexer.py`
//! reads from CPython's `ast` module: top-level definitions, docstrings, imports and calls.

use std::collections::BTreeMap;

use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{self as ast, Expr, Operator, Stmt};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextSize};

/// The parsed module plus a line index to turn byte offsets into 1-based line numbers.
pub struct PyModule {
    pub body: Vec<Stmt>,
    lines: LineIndex,
}

/// A syntax error with the 1-based line where it was found and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PySyntaxError {
    pub line: i64,
    pub message: String,
}

pub fn parse(text: &str) -> Result<PyModule, PySyntaxError> {
    let lines = LineIndex::from_source_text(text);
    match ruff_python_parser::parse_module(text) {
        Ok(parsed) => Ok(PyModule {
            body: parsed.into_syntax().body.into_iter().collect(),
            lines,
        }),
        Err(err) => {
            let (line, message) = cpython_error(text, &lines, &err);
            Err(PySyntaxError {
                line,
                message: format!("{message} (<unknown>, line {line})"),
            })
        }
    }
}

/// CPython's line and wording for the errors real repositories hit most; anything else gets
/// CPython's generic message, since ruff's wording never matches it.
fn cpython_error(
    text: &str,
    lines: &LineIndex,
    err: &ruff_python_parser::ParseError,
) -> (i64, String) {
    use ruff_python_parser::ParseErrorType as E;
    let line_count = i64::try_from(crate::text::splitlines(text).len()).unwrap_or(i64::MAX);
    let line = line_of(lines, err.location.start()).min(line_count.max(1));
    match &err.error {
        E::UnexpectedIndentation => (line, "unexpected indent".to_owned()),
        E::OtherError(message) if message.starts_with("Expected an indented block after ") => {
            let before = &text[..usize::from(err.location.start()).min(text.len())];
            let at_line_start = before
                .rsplit('\n')
                .next()
                .is_some_and(|head| head.trim().is_empty());
            // Ruff points at the next statement when there is one; the header is above it.
            let last_candidate = if at_line_start { line - 1 } else { line };
            let clause = &message["Expected an indented block after ".len()..];
            missing_block(text, last_candidate, clause)
        }
        E::OtherError(message) if message.starts_with("Expected `except` or `finally`") => {
            (line, "expected 'except' or 'finally' block".to_owned())
        }
        _ => unclosed_bracket(text, lines, err.location.start(), line)
            .unwrap_or_else(|| (line, "invalid syntax".to_owned())),
    }
}

/// CPython reports an error past a still-open bracket as that bracket never being closed, on
/// the line where the innermost open bracket sits.
fn unclosed_bracket(
    text: &str,
    lines: &LineIndex,
    error_at: TextSize,
    error_line: i64,
) -> Option<(i64, String)> {
    use ruff_python_ast::token::TokenKind as T;
    let parsed =
        ruff_python_parser::parse_unchecked_source(text, ruff_python_ast::PySourceType::Python);
    let mut open: Vec<(char, TextSize)> = Vec::new();
    for token in parsed.tokens().iter() {
        if token.start() >= error_at {
            break;
        }
        let (opener, closer) = match token.kind() {
            T::Lpar => (Some('('), None),
            T::Lsqb => (Some('['), None),
            T::Lbrace => (Some('{'), None),
            T::Rpar => (None, Some('(')),
            T::Rsqb => (None, Some('[')),
            T::Rbrace => (None, Some('{')),
            _ => (None, None),
        };
        if let Some(bracket) = opener {
            open.push((bracket, token.start()));
        } else if closer.is_some() && open.last().map(|(b, _)| Some(*b)) == Some(closer) {
            open.pop();
        }
    }
    let (bracket, at) = open.last()?;
    let line = line_of(lines, *at);
    let at_eof = text[usize::from(error_at).min(text.len())..]
        .trim()
        .is_empty();
    (error_line > line || at_eof).then(|| (line, format!("'{bracket}' was never closed")))
}

/// CPython names the line where the header statement starts and reports the first code line
/// after it, or the last line of the file when the block is cut off.
fn missing_block(text: &str, last_candidate: i64, clause: &str) -> (i64, String) {
    let source: Vec<&str> = text.split('\n').collect();
    let has_code = |number: i64| {
        usize::try_from(number - 1)
            .ok()
            .and_then(|index| source.get(index))
            .is_some_and(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
    };
    let header_end = (1..=last_candidate)
        .rev()
        .find(|n| has_code(*n) && line_ends_header(&source, *n))
        .unwrap_or(last_candidate);
    let header = header_start(&source, header_end);
    let line_count = i64::try_from(crate::text::splitlines(text).len()).unwrap_or(i64::MAX);
    let next = ((header_end + 1)..=line_count)
        .find(|n| has_code(*n))
        .unwrap_or_else(|| line_count.max(header_end));
    let clause = match clause {
        "function definition" => "function definition".to_owned(),
        "`class` definition" => "class definition".to_owned(),
        other => {
            let keyword = other.split('`').nth(1).unwrap_or(other);
            format!("'{keyword}' statement")
        }
    };
    (
        next,
        format!("expected an indented block after {clause} on line {header}"),
    )
}

/// First line of a header that may span lines inside brackets (`def f(a,\n      b):`).
fn header_start(source: &[&str], header_end: i64) -> i64 {
    let mut depth: i64 = 0;
    let mut number = header_end;
    while number >= 1 {
        let line = usize::try_from(number - 1)
            .ok()
            .and_then(|index| source.get(index))
            .copied()
            .unwrap_or_default();
        let code = line.split('#').next().unwrap_or_default();
        for ch in code.chars() {
            match ch {
                ')' | ']' | '}' => depth += 1,
                '(' | '[' | '{' => depth -= 1,
                _ => {}
            }
        }
        if depth <= 0 {
            return number;
        }
        number -= 1;
    }
    header_end
}

fn line_ends_header(source: &[&str], number: i64) -> bool {
    usize::try_from(number - 1)
        .ok()
        .and_then(|index| source.get(index))
        .is_some_and(|line| {
            line.split('#')
                .next()
                .unwrap_or_default()
                .trim_end()
                .ends_with(':')
        })
}

fn line_of(lines: &LineIndex, offset: TextSize) -> i64 {
    i64::try_from(lines.line_index(offset).get()).unwrap_or(i64::MAX)
}

impl PyModule {
    pub fn line(&self, offset: TextSize) -> i64 {
        line_of(&self.lines, offset)
    }

    /// `(lineno, end_lineno)` of a statement. CPython puts a decorated definition on its
    /// `def`/`class` line, while ruff's range starts at the first decorator.
    pub fn stmt_lines(&self, stmt: &Stmt) -> (i64, i64) {
        let start = match stmt {
            Stmt::FunctionDef(def) if !def.decorator_list.is_empty() => def.name.start(),
            Stmt::ClassDef(def) if !def.decorator_list.is_empty() => def.name.start(),
            other => other.start(),
        };
        (self.line(start), self.line(stmt.end()))
    }

    /// Imports and calls in source pre-order, as `iter_imports_and_calls` yields them.
    pub fn imports_and_calls(&self) -> Vec<ImportOrCall<'_>> {
        let mut collector = Collector { out: Vec::new() };
        collector.visit_body(&self.body);
        collector.out
    }
}

pub enum ImportOrCall<'a> {
    Import(&'a ast::StmtImport),
    ImportFrom(&'a ast::StmtImportFrom),
    Call(&'a ast::ExprCall),
}

struct Collector<'a> {
    out: Vec<ImportOrCall<'a>>,
}

impl<'a> Visitor<'a> for Collector<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::Import(import) => self.out.push(ImportOrCall::Import(import)),
            Stmt::ImportFrom(import) => self.out.push(ImportOrCall::ImportFrom(import)),
            _ => {}
        }
        visitor::walk_stmt(self, stmt);
    }

    // `iter_imports_and_calls` treats `ast.arg` as a leaf, so calls in parameter annotations
    // are never reached.
    fn visit_parameter(&mut self, _parameter: &'a ast::Parameter) {}

    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Call(call) = expr {
            self.out.push(ImportOrCall::Call(call));
        }
        visitor::walk_expr(self, expr);
    }
}

/// `dotted_name` from `indexer.py`: names, attributes, and the callee of calls and subscripts.
pub fn dotted_name(expr: &Expr) -> String {
    match expr {
        Expr::Name(name) => name.id.to_string(),
        Expr::Attribute(attr) => {
            let prefix = dotted_name(&attr.value);
            if prefix.is_empty() {
                attr.attr.id.to_string()
            } else {
                format!("{prefix}.{}", attr.attr.id)
            }
        }
        Expr::Call(call) => dotted_name(&call.func),
        Expr::Subscript(sub) => dotted_name(&sub.value),
        _ => String::new(),
    }
}

/// Extract a receiver type name from a Python type annotation, unwrapping `Optional[T]`,
/// `T | None`, `Union[T, None]`, `Annotated[T, ...]`, and string forward references.
pub fn annotation_type_name(expr: &Expr) -> String {
    match expr {
        Expr::Name(name) => {
            let id = name.id.as_str();
            if id == "None" {
                String::new()
            } else {
                id.to_owned()
            }
        }
        Expr::Attribute(_) => dotted_name(expr),
        Expr::StringLiteral(lit) => {
            let raw = lit.value.to_str().trim();
            if !raw.is_empty()
                && raw
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
            {
                raw.to_owned()
            } else {
                String::new()
            }
        }
        Expr::BinOp(bin) if bin.op == Operator::BitOr => {
            let left = annotation_type_name(&bin.left);
            let right = annotation_type_name(&bin.right);
            match (left.is_empty(), right.is_empty()) {
                (false, true) => left,
                (true, false) => right,
                _ => String::new(),
            }
        }
        Expr::Subscript(sub) => {
            let outer = dotted_name(&sub.value);
            let leaf = outer.rsplit('.').next().unwrap_or(&outer);
            match leaf {
                "Optional" => annotation_type_name(&sub.slice),
                "Annotated" => match sub.slice.as_ref() {
                    Expr::Tuple(tuple) => tuple
                        .elts
                        .first()
                        .map_or_else(String::new, annotation_type_name),
                    other => annotation_type_name(other),
                },
                "Union" => match sub.slice.as_ref() {
                    Expr::Tuple(tuple) => {
                        let non_empty: Vec<String> = tuple
                            .elts
                            .iter()
                            .map(annotation_type_name)
                            .filter(|name| !name.is_empty())
                            .collect();
                        if non_empty.len() == 1 {
                            non_empty.into_iter().next().unwrap_or_default()
                        } else {
                            String::new()
                        }
                    }
                    other => annotation_type_name(other),
                },
                _ => outer,
            }
        }
        _ => String::new(),
    }
}

fn receiver_target_key(target: &Expr) -> Option<String> {
    match target {
        Expr::Name(name) => Some(name.id.to_string()),
        Expr::Attribute(attr) => match attr.value.as_ref() {
            Expr::Name(base) if matches!(base.id.as_str(), "self" | "cls") => {
                Some(format!("self.{}", attr.attr.id))
            }
            _ => None,
        },
        _ => None,
    }
}

fn call_callee_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Call(call) => {
            let callee = dotted_name(&call.func);
            (!callee.is_empty()).then_some(callee)
        }
        _ => None,
    }
}

fn collect_stmt_receivers(stmts: &[Stmt], out: &mut BTreeMap<String, String>) {
    for stmt in stmts {
        match stmt {
            Stmt::AnnAssign(assign) => {
                if let Some(key) = receiver_target_key(&assign.target) {
                    let ty = annotation_type_name(&assign.annotation);
                    if !ty.is_empty() {
                        out.insert(key, ty);
                    } else if let Some(val) = assign.value.as_deref().and_then(call_callee_name) {
                        out.insert(key, val);
                    }
                }
            }
            Stmt::Assign(assign) => {
                if let Some(val) = call_callee_name(&assign.value) {
                    for target in &assign.targets {
                        if let Some(key) = receiver_target_key(target) {
                            out.insert(key, val.clone());
                        }
                    }
                }
            }
            Stmt::With(with_stmt) => {
                for item in &with_stmt.items {
                    if let Some(target) = item.optional_vars.as_deref()
                        && let Some(key) = receiver_target_key(target)
                        && let Some(val) = call_callee_name(&item.context_expr)
                    {
                        out.insert(key, val);
                    }
                }
                collect_stmt_receivers(&with_stmt.body, out);
            }
            Stmt::If(if_stmt) => {
                collect_stmt_receivers(&if_stmt.body, out);
                for clause in &if_stmt.elif_else_clauses {
                    collect_stmt_receivers(&clause.body, out);
                }
            }
            Stmt::For(for_stmt) => {
                collect_stmt_receivers(&for_stmt.body, out);
                collect_stmt_receivers(&for_stmt.orelse, out);
            }
            Stmt::While(while_stmt) => {
                collect_stmt_receivers(&while_stmt.body, out);
                collect_stmt_receivers(&while_stmt.orelse, out);
            }
            Stmt::Try(try_stmt) => {
                collect_stmt_receivers(&try_stmt.body, out);
                for handler in &try_stmt.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_stmt_receivers(&h.body, out);
                }
                collect_stmt_receivers(&try_stmt.orelse, out);
                collect_stmt_receivers(&try_stmt.finalbody, out);
            }
            Stmt::Match(match_stmt) => {
                for case in &match_stmt.cases {
                    collect_stmt_receivers(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Local variable and `self.<attr>` type bindings for a Python function or method.
pub fn collect_function_receivers(def: &ast::StmtFunctionDef) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let params = &def.parameters;
    for param in params
        .posonlyargs
        .iter()
        .chain(&params.args)
        .chain(&params.kwonlyargs)
    {
        if let Some(ann) = param.parameter.annotation.as_deref() {
            let ty = annotation_type_name(ann);
            if !ty.is_empty() {
                out.insert(param.parameter.name.id.to_string(), ty);
            }
        }
    }
    for extra in [&params.vararg, &params.kwarg].into_iter().flatten() {
        if let Some(ann) = extra.annotation.as_deref() {
            let ty = annotation_type_name(ann);
            if !ty.is_empty() {
                out.insert(extra.name.id.to_string(), ty);
            }
        }
    }
    collect_stmt_receivers(&def.body, &mut out);
    out
}

/// Module-level variable type bindings (top-level assignments and annotations).
pub fn collect_module_receivers(body: &[Stmt]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    collect_stmt_receivers(body, &mut out);
    out.retain(|key, _| !key.starts_with("self."));
    out
}

/// Class attribute type bindings from class body annotations/assignments and `self.<attr>` in methods.
pub fn collect_class_attr_types(class: &ast::StmtClassDef) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut class_body = BTreeMap::new();
    collect_stmt_receivers(&class.body, &mut class_body);
    for (key, value) in class_body {
        if !key.starts_with("self.") {
            out.insert(key, value);
        }
    }
    // Non-__init__ methods first so __init__ bindings take precedence.
    for stmt in &class.body {
        if let Stmt::FunctionDef(def) = stmt
            && def.name.id.as_str() != "__init__"
        {
            let mut local = BTreeMap::new();
            collect_stmt_receivers(&def.body, &mut local);
            for (key, value) in local {
                if let Some(attr) = key.strip_prefix("self.") {
                    out.entry(attr.to_owned()).or_insert(value);
                }
            }
        }
    }
    for stmt in &class.body {
        if let Stmt::FunctionDef(def) = stmt
            && def.name.id.as_str() == "__init__"
        {
            let mut local = BTreeMap::new();
            collect_stmt_receivers(&def.body, &mut local);
            for (key, value) in local {
                if let Some(attr) = key.strip_prefix("self.") {
                    out.insert(attr.to_owned(), value);
                }
            }
        }
    }
    out
}

/// `ast.get_docstring(node)` with `clean=True`.
pub fn docstring(body: &[Stmt]) -> String {
    match body.first() {
        Some(Stmt::Expr(stmt)) => match stmt.value.as_ref() {
            Expr::StringLiteral(literal) => cleandoc(literal.value.to_str()),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// `str.expandtabs()` with the default tab size of 8.
fn expand_tabs(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut column = 0;
    for ch in line.chars() {
        match ch {
            '\t' => {
                let pad = 8 - column % 8;
                out.extend(std::iter::repeat_n(' ', pad));
                column += pad;
            }
            '\n' | '\r' => {
                out.push(ch);
                column = 0;
            }
            _ => {
                out.push(ch);
                column += 1;
            }
        }
    }
    out
}

/// `inspect.cleandoc`.
pub fn cleandoc(doc: &str) -> String {
    let expanded = expand_tabs(doc);
    let mut lines: Vec<String> = expanded.split('\n').map(str::to_owned).collect();
    let margin = lines
        .iter()
        .skip(1)
        .filter_map(|line| {
            let content = line.trim_start_matches(' ').chars().count();
            (content > 0).then(|| line.chars().count() - content)
        })
        .min();
    if let Some(first) = lines.first_mut() {
        *first = first.trim_start_matches(' ').to_owned();
    }
    if let Some(margin) = margin {
        for line in lines.iter_mut().skip(1) {
            *line = line.chars().skip(margin).collect();
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    let leading = lines.iter().take_while(|line| line.is_empty()).count();
    lines.drain(..leading);
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleandoc_matches_inspect() {
        assert_eq!(
            cleandoc("  Summary.\n\n    Body\n      more\n  "),
            "Summary.\n\nBody\n  more"
        );
        assert_eq!(cleandoc("\n\n"), "");
    }

    #[test]
    fn decorated_function_starts_on_def_line() {
        let module = parse("@app.get('/x')\ndef handler():\n    \"Doc.\"\n    return 1\n").unwrap();
        assert_eq!(module.stmt_lines(&module.body[0]), (2, 4));
        let Stmt::FunctionDef(def) = &module.body[0] else {
            panic!("expected a function")
        };
        assert_eq!(docstring(&def.body), "Doc.");
    }

    #[test]
    fn missing_block_matches_cpython() {
        let err = parse(
            "class R:\n    def a(self):\n        def wrap(x):\n    def b(self):\n        pass\n",
        )
        .err()
        .unwrap();
        assert_eq!(err.line, 4);
        assert_eq!(
            err.message,
            "expected an indented block after function definition on line 3 (<unknown>, line 4)"
        );
    }

    #[test]
    fn cut_off_blocks_match_cpython() {
        let cases = [
            ("if x:\n", 1, "'if' statement on line 1"),
            ("if x:\n\n", 2, "'if' statement on line 1"),
            ("def f(a,\n      b):\n", 2, "function definition on line 1"),
            ("def a():\ndef b():\n", 2, "function definition on line 1"),
        ];
        for (source, line, header) in cases {
            let err = parse(source).err().unwrap();
            assert_eq!(err.line, line, "{source:?}");
            assert!(err.message.contains(header), "{source:?}: {}", err.message);
        }
    }

    #[test]
    fn unclosed_bracket_matches_cpython() {
        let err = parse("x = foo(1,\n    2\ny = 3\n").err().unwrap();
        assert_eq!(
            (err.line, err.message.as_str()),
            (1, "'(' was never closed (<unknown>, line 1)")
        );
    }

    #[test]
    fn syntax_error_reports_line() {
        let err = parse("x = 1\ndef broken(:\n").err().unwrap();
        assert_eq!(err.line, 2);
    }
}
