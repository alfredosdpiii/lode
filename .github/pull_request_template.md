## Summary

Describe the change and why it is needed.

## Testing

List the commands run, for example:

```bash
uv run ruff check .
uv run mypy src tests scripts benchmarks
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

## Context

Link related issues, releases, or follow-up work.

## Agent safety

- [ ] No secrets or local-only artifacts are included.
- [ ] Blast radius or relevant callers/callees were checked when code paths changed.
