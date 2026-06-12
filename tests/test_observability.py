from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from lode.features import enabled
from lode.observability import Metrics, sanitize


class ObservabilityTests(unittest.TestCase):
    def test_sanitize_redacts_sensitive_fields(self) -> None:
        payload = {
            "token": "secret-value",
            "nested": {"api_key": "secret-value", "safe": "ok"},
        }
        self.assertEqual(
            sanitize(payload),
            {
                "token": "[REDACTED]",
                "nested": {"api_key": "[REDACTED]", "safe": "ok"},
            },
        )

    def test_metrics_render_prometheus_counter(self) -> None:
        metrics = Metrics()
        metrics.record("get", "/health", 200)
        rendered = metrics.render()
        self.assertIn("# TYPE loded_requests_total counter", rendered)
        self.assertIn(
            'loded_requests_total{method="GET",path="/health",status="200"} 1',
            rendered,
        )

    def test_feature_flag_env_override(self) -> None:
        with patch.dict(os.environ, {"LODE_FEATURE_METRICS": "0"}):
            self.assertFalse(enabled("metrics"))

    def test_openapi_schema_and_generated_docs_exist(self) -> None:
        schema = Path("openapi/loded.openapi.yml").read_text(encoding="utf-8")
        docs = Path("docs/api/loded.md").read_text(encoding="utf-8")
        self.assertIn("/health:", schema)
        self.assertIn("/metrics:", schema)
        self.assertIn("`GET` | `/metrics`", docs)


if __name__ == "__main__":
    unittest.main()
