"""
pytest configuration for the FinSage test suite.

Adds the workspace root to sys.path so that:
  - `from finsage.constants import ...` works in unit tests without installing the package.
  - Integration tests can import shared fixtures defined here.

Markers:
  integration — requires a live Spark session + Databricks workspace.
                Skip in CI with:  pytest -m "not integration"
"""

import sys
import os
import pytest

# Make 'src/' importable as a package root so tests can do:
#   from finsage.constants import TARGET_CONCEPT_MAP
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring a live Databricks Spark session",
    )


# ---------------------------------------------------------------------------
# Spark fixture — only instantiated when an integration test is collected.
# Skips gracefully if PySpark is not available in the CI environment.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark():
    try:
        from pyspark.sql import SparkSession
        session = (
            SparkSession.builder
            .appName("finsage-integration-tests")
            .getOrCreate()
        )
        yield session
        session.stop()
    except ImportError:
        pytest.skip("PySpark not available — integration tests require a Databricks runtime")
