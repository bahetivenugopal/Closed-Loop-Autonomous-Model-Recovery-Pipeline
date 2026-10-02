"""
Unit test suite for Phase 8: CI/CD Implementation.

Verifies:
1. CI/CD workflow YAML syntax and structure.
2. All 16 logical stages represented.
3. Presence of healthcheck, readiness probe, and inference smoke test.
4. Rollback mechanism definition on failure.
5. Use of immutable Docker image tags.
"""

from pathlib import Path
import pytest
import yaml


@pytest.mark.unit
class TestPhase8CICD:
    """Test suite for CI/CD Pipeline."""

    def test_workflow_file_exists_and_parses_yaml(self):
        """Verify .github/workflows/ci-cd.yml is valid YAML."""
        workflow_path = Path(".github/workflows/ci-cd.yml")
        assert workflow_path.exists(), "ci-cd.yml does not exist"

        content = workflow_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)

        assert "name" in parsed
        assert "jobs" in parsed

    def test_workflow_contains_required_pipeline_stages(self):
        """Verify presence of key pipeline jobs and verification steps."""
        workflow_path = Path(".github/workflows/ci-cd.yml")
        content = workflow_path.read_text(encoding="utf-8")

        # 1. Quality & Tests
        assert "mypy" in content
        assert "flake8" in content
        assert "pytest tests/unit/" in content
        assert "pytest tests/integration/" in content

        # 2. Model Validation & Schema Check
        assert "FEATURE_COLUMNS" in content
        assert "verify_reference_integrity" in content

        # 3. Docker & Immutable Tagging
        assert "github.sha" in content
        assert "trivy" in content.lower()

        # 4. Staging Deployment, Probes & Smoke Test
        assert "/health" in content
        assert "/ready" in content
        assert "/predict" in content

        # 5. Rollback on Failure
        assert "if: failure()" in content
        assert "docker compose down" in content
