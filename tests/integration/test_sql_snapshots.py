"""SQL golden file snapshots and inline pattern assertions.

Tests verify SQL output stability via golden file comparison and check
key SQL patterns are present in generated output.

Use --update-snapshots to regenerate golden files:
    pytest tests/integration/test_sql_snapshots.py --update-snapshots
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_sql(sql: str) -> str:
    """Strip and normalize whitespace in SQL for stable comparison."""
    return sql.strip()


def _assert_snapshot(
    sql: str,
    snapshot_path: Path,
    update: bool,
) -> None:
    """Compare SQL to golden file, or update it."""
    normalized = _normalize_sql(sql)
    if update:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(normalized + "\n")
        pytest.skip(f"Snapshot updated: {snapshot_path.name}")
    assert snapshot_path.exists(), (
        f"Snapshot {snapshot_path.name} not found. Run with --update-snapshots to generate."
    )
    expected = snapshot_path.read_text().strip()
    assert normalized == expected, (
        f"SQL output differs from snapshot {snapshot_path.name}.\n"
        "Run with --update-snapshots to update."
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def numeric_conn() -> duckdb.DuckDBPyConnection:
    """2-column numeric data for StandardScaler snapshot."""
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            (1.0, 10.0),
            (2.0, 20.0),
            (3.0, 30.0),
            (4.0, 40.0),
            (5.0, 50.0)
        t(a, b)
    """)
    return c


@pytest.fixture
def numeric_nulls_conn() -> duckdb.DuckDBPyConnection:
    """Numeric data with NULLs for Imputer snapshots."""
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            (1.0, 10.0),
            (2.0, NULL),
            (NULL, 30.0),
            (4.0, 40.0),
            (5.0, 50.0)
        t(a, b)
    """)
    return c


@pytest.fixture
def categorical_conn() -> duckdb.DuckDBPyConnection:
    """Pure categorical data for OneHotEncoder snapshot."""
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            ('red',   'small'),
            ('blue',  'medium'),
            ('green', 'large'),
            ('red',   'small'),
            ('blue',  'medium')
        t(color, size)
    """)
    return c


@pytest.fixture
def mixed_conn() -> duckdb.DuckDBPyConnection:
    """Mixed numeric + categorical with NULLs for pipeline snapshots."""
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            (1.0, 10.0, 'London'),
            (2.0, NULL, 'Paris'),
            (NULL, 30.0, 'London'),
            (4.0, 40.0, 'Tokyo'),
            (5.0, 50.0, 'Paris')
        t(price, quantity, city)
    """)
    return c


# ── Snapshot Tests ───────────────────────────────────────────────────


class TestStandardScalerSnapshot:
    """SQL snapshot for solo StandardScaler on 2-column numeric data."""

    def test_snapshot(self, numeric_conn, snapshot_dir, update_snapshots) -> None:
        """Verify StandardScaler SQL matches golden file."""
        backend = DuckDBBackend(connection=numeric_conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "standard_scaler.sql", update_snapshots)

    def test_inline_patterns(self, numeric_conn) -> None:
        """SQL contains subtraction, division, and NULLIF."""
        backend = DuckDBBackend(connection=numeric_conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")
        sql_upper = sql.upper()

        assert "-" in sql, "Missing subtraction"
        assert "/" in sql, "Missing division"
        assert "NULLIF" in sql_upper, "Missing NULLIF for safe division"


class TestImputerMeanSnapshot:
    """SQL snapshot for Imputer(strategy='mean') on numeric data with NULLs."""

    def test_snapshot(self, numeric_nulls_conn, snapshot_dir, update_snapshots) -> None:
        """Verify Imputer(strategy='mean') SQL matches golden file."""
        backend = DuckDBBackend(connection=numeric_nulls_conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "imputer_mean.sql", update_snapshots)

    def test_inline_patterns(self, numeric_nulls_conn) -> None:
        """SQL contains COALESCE."""
        backend = DuckDBBackend(connection=numeric_nulls_conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb").upper()

        assert "COALESCE" in sql, "Missing COALESCE for null replacement"


class TestImputerAutoSnapshot:
    """SQL snapshot for Imputer() (auto) on mixed numeric+categorical data."""

    def test_snapshot(self, mixed_conn, snapshot_dir, update_snapshots) -> None:
        """Verify Imputer() auto SQL matches golden file."""
        backend = DuckDBBackend(connection=mixed_conn)
        pipe = Pipeline([Imputer()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "imputer_auto.sql", update_snapshots)

    def test_inline_patterns(self, mixed_conn) -> None:
        """SQL contains COALESCE for auto imputation."""
        backend = DuckDBBackend(connection=mixed_conn)
        pipe = Pipeline([Imputer()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb").upper()

        assert "COALESCE" in sql, "Missing COALESCE"


class TestOneHotEncoderSnapshot:
    """SQL snapshot for solo OneHotEncoder on categorical data."""

    def test_snapshot(self, categorical_conn, snapshot_dir, update_snapshots) -> None:
        """Verify OneHotEncoder SQL matches golden file."""
        backend = DuckDBBackend(connection=categorical_conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "onehot_encoder.sql", update_snapshots)

    def test_inline_patterns(self, categorical_conn) -> None:
        """SQL contains CASE WHEN THEN 1 ELSE 0."""
        backend = DuckDBBackend(connection=categorical_conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb").upper()

        assert "CASE" in sql, "Missing CASE"
        assert "WHEN" in sql, "Missing WHEN"
        assert "THEN 1" in sql, "Missing THEN 1"
        assert "ELSE 0" in sql, "Missing ELSE 0"


class TestFullPipelineSnapshot:
    """SQL snapshot for Pipeline([Imputer(), StandardScaler(), OneHotEncoder()])."""

    def test_snapshot(self, mixed_conn, snapshot_dir, update_snapshots) -> None:
        """Verify full pipeline SQL matches golden file."""
        backend = DuckDBBackend(connection=mixed_conn)
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "full_pipeline.sql", update_snapshots)

    def test_inline_patterns(self, mixed_conn) -> None:
        """Full pipeline SQL contains all transformer patterns."""
        backend = DuckDBBackend(connection=mixed_conn)
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("data", backend=backend)
        sql_raw = pipe.to_sql(dialect="duckdb")
        sql = sql_raw.upper()

        assert "COALESCE" in sql, "Missing COALESCE (Imputer)"
        assert "CASE" in sql, "Missing CASE (OneHotEncoder)"
        assert "WHEN" in sql, "Missing WHEN (OneHotEncoder)"
        assert "-" in sql_raw, "Missing subtraction (StandardScaler)"
        assert "/" in sql_raw, "Missing division (StandardScaler)"
        assert "NULLIF" in sql, "Missing NULLIF (StandardScaler)"
        assert "FROM __INPUT__" in sql.replace('"', ""), "Missing FROM __input__"


class TestCompositionSnapshot:
    """SQL snapshot for Imputer+StandardScaler showing expression nesting."""

    def test_snapshot(self, numeric_nulls_conn, snapshot_dir, update_snapshots) -> None:
        """Verify Imputer+StandardScaler composition SQL matches golden file."""
        backend = DuckDBBackend(connection=numeric_nulls_conn)
        pipe = Pipeline([Imputer(strategy="mean"), StandardScaler()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")

        _assert_snapshot(sql, snapshot_dir / "composition.sql", update_snapshots)

    def test_inline_patterns(self, numeric_nulls_conn) -> None:
        """Composition SQL has COALESCE nested inside arithmetic, no CTE."""
        backend = DuckDBBackend(connection=numeric_nulls_conn)
        pipe = Pipeline([Imputer(strategy="mean"), StandardScaler()], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(dialect="duckdb")
        sql_upper = sql.upper()

        assert "COALESCE" in sql_upper, "Missing COALESCE"
        assert "-" in sql, "Missing subtraction"
        assert "NULLIF" in sql_upper, "Missing NULLIF"
        # No CTE for expression-only composition
        assert "WITH" not in sql_upper, "Unexpected CTE for expression-only pipeline"
