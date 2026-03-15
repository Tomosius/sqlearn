"""Shared test fixtures and pytest configuration for sqlearn tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from sqlearn.core.backend import DuckDBBackend


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --update-snapshots CLI flag."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate SQL snapshot files instead of comparing.",
    )


@pytest.fixture
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    """Whether to update snapshot files instead of comparing."""
    return bool(request.config.getoption("--update-snapshots"))


@pytest.fixture
def snapshot_dir() -> Path:
    """Path to the integration test snapshots directory."""
    d = Path(__file__).parent / "integration" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_numeric_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a 7-row, 3-column numeric table with some NULLs.

    Odd row count ensures median tests avoid interpolation differences.
    NULLs in columns b and c for imputer tests.
    """
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE sample_numeric AS SELECT * FROM VALUES
            (1.0,  10.0, 100.0),
            (2.0,  20.0, NULL),
            (3.0,  NULL, 300.0),
            (4.0,  40.0, 400.0),
            (5.0,  50.0, 500.0),
            (6.0,  60.0, 600.0),
            (7.0,  70.0, 700.0)
        t(a, b, c)
    """)
    return conn


@pytest.fixture
def sample_categorical_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a pure categorical table (3+ categories per column).

    Each column has a clear mode (no ties) for most_frequent tests.
    """
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE sample_categorical AS SELECT * FROM VALUES
            ('red',   'small',  'circle'),
            ('blue',  'medium', 'square'),
            ('green', 'large',  'circle'),
            ('red',   'small',  'triangle'),
            ('blue',  'medium', 'square'),
            ('red',   'large',  'circle'),
            ('green', 'small',  'square')
        t(color, size, shape)
    """)
    return conn


@pytest.fixture
def sample_mixed_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with numeric + categorical columns and NULLs."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE sample_mixed AS SELECT * FROM VALUES
            (1.0,  10.0, 'London'),
            (2.0,  NULL, 'Paris'),
            (NULL,  30.0, 'London'),
            (4.0,  40.0, NULL),
            (5.0,  50.0, 'Tokyo'),
            (6.0,  60.0, 'Paris'),
            (7.0,  70.0, 'London')
        t(price, quantity, city)
    """)
    return conn


@pytest.fixture
def backend(sample_numeric_conn: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """DuckDBBackend from in-memory connection (wraps sample_numeric_conn)."""
    return DuckDBBackend(connection=sample_numeric_conn)


def _table_to_numpy(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str] | None = None,
) -> np.ndarray:
    """Fetch a DuckDB table as a float64 numpy array.

    Args:
        conn: DuckDB connection.
        table: Table name.
        columns: Specific columns to fetch. If None, fetches all.

    Returns:
        2D numpy array of float64.
    """
    cols = ", ".join(columns) if columns else "*"
    rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()  # noqa: S608
    return np.array(rows, dtype=np.float64)
