---
name: lode
description: Use the local lode repository knowledge graph CLI for codebase discovery, symbol search, context-pack generation, graph neighbors, impact exploration, and fast source-cited local answers before relying on broader search.
---

# Lode

Lode is a local repository knowledge graph for coding agents. Use it when you need
fast, source-cited answers from code that is already on disk: where a symbol lives,
which files are relevant to a task, what neighbors a node has, or what context to
read before editing.

Lode is local-first. It does not replace typecheckers, tests, LSP, or exact text
search; it complements them by building a compact index and context pack.

## When to use

Use Lode early when:

- exploring an unfamiliar repository or subsystem
- finding symbols, routes, docs, config files, or cross-file references
- gathering a bounded context pack before a coding change
- checking likely impact around a file or symbol
- needing JSON output with file paths and line citations

Do not use Lode for web facts or dependency docs. Use web/context docs for external
information and use the checked-out source as truth for local code behavior.

## Basic workflow

From the repository you are working on:

```bash
lode index "$PWD"
lode status --json
lode search "auth middleware" --repo "$PWD" --limit 10 --json
lode symbol "UserService" --repo "$PWD" --limit 10 --json
lode context "how does auth middleware work" --repo "$PWD" --budget 6000 --json
```

If results seem stale, re-index the repo before trusting search output.

## Command guide

### Index a repo

```bash
lode index /path/to/repo --json
```

Use before search/context in a repo that may not be indexed. Add `--sync-kuzu` only
when graph projection is needed and the optional Kuzu extra is installed.

### Search indexed code and docs

```bash
lode search "query terms" --repo /path/to/repo --limit 20 --json
```

Good for fuzzy discovery across symbols, docs, paths, and file content. Prefer
`rg` for exact string occurrences or regex-sensitive searches.

### Find a symbol

```bash
lode symbol "symbol_or_partial_name" --repo /path/to/repo --limit 20 --json
```

Use for exact-ish symbol lookup before opening files or jumping with LSP.

### Build an agent context pack

```bash
lode context "task or question" --repo /path/to/repo --budget 6000 --limit 10 --json
```

Use before implementation or review to get a compact set of likely relevant files,
symbols, and citations. Treat it as a reading shortlist, not a proof.

### Inspect graph neighbors

```bash
lode neighbors "node_id" --limit 80 --json
```

Use node IDs returned by search/symbol/context to inspect direct relationships.

### Run the daemon

```bash
loded --host 127.0.0.1 --port 7979
curl http://127.0.0.1:7979/health
```

Use the daemon when an external tool or repeated workflow benefits from a local HTTP
service. For ordinary agent work, the CLI is usually simpler.

## Output handling

- Prefer `--json` for agent workflows.
- Cite paths and line numbers from Lode results in your answer.
- Open files before making precise claims or edits.
- If Lode returns no results, fall back to `rg`, LSP, or direct file inspection.
- If Lode errors due to missing indexes, run `lode index "$PWD"` and retry once.

## Data and aliases

Default data directory:

```text
~/.local/share/lode/
```

Override with `--data-dir DIR` when isolating experiments or avoiding shared state.
`kg` and `kgd` are aliases for `lode` and `loded`.
