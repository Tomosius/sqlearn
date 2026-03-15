"""sklearn equivalence tests — verify sqlearn matches scikit-learn output.

Parameterized tests comparing sqlearn transformer output to scikit-learn
with np.testing.assert_allclose(rtol=1e-6).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler

# ── Helpers ──────────────────────────────────────────────────────────


def _table_to_numpy(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str] | None = None,
    dtype: type = float,
) -> np.ndarray:  # type: ignore[type-arg]
    """Fetch DuckDB table as numpy array."""
    cols = ", ".join(columns) if columns else "*"
    rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()  # noqa: S608
    return np.array(rows, dtype=dtype)


# ── StandardScaler ───────────────────────────────────────────────────


class TestStandardScalerEquivalence:
    """sqlearn StandardScaler matches sklearn StandardScaler (population std, ddof=0)."""

    @pytest.fixture
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Pure numeric, 7 rows, no NULLs."""
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0, 10.0, 100.0),
                (2.0, 20.0, 200.0),
                (3.0, 30.0, 300.0),
                (4.0, 40.0, 400.0),
                (5.0, 50.0, 500.0),
                (6.0, 60.0, 600.0),
                (7.0, 70.0, 700.0)
            t(a, b, c)
        """)
        return c

    def test_default(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Default StandardScaler: both center and scale."""
        backend = DuckDBBackend(connection=conn)
        sq_result = (
            Pipeline([StandardScaler()], backend=backend)
            .fit("data", backend=backend)
            .transform("data", backend=backend)
        )

        x_data = _table_to_numpy(conn, "data")
        sk_result = SkStandardScaler().fit_transform(x_data)

        np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)

    def test_without_mean(self, conn: duckdb.DuckDBPyConnection) -> None:
        """StandardScaler(with_mean=False): scale only."""
        backend = DuckDBBackend(connection=conn)
        sq_result = (
            Pipeline([StandardScaler(with_mean=False)], backend=backend)
            .fit("data", backend=backend)
            .transform("data", backend=backend)
        )

        x_data = _table_to_numpy(conn, "data")
        sk_result = SkStandardScaler(with_mean=False).fit_transform(x_data)

        np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)

    def test_without_std(self, conn: duckdb.DuckDBPyConnection) -> None:
        """StandardScaler(with_std=False): center only."""
        backend = DuckDBBackend(connection=conn)
        sq_result = (
            Pipeline([StandardScaler(with_std=False)], backend=backend)
            .fit("data", backend=backend)
            .transform("data", backend=backend)
        )

        x_data = _table_to_numpy(conn, "data")
        sk_result = SkStandardScaler(with_std=False).fit_transform(x_data)

        np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)

    def test_constant_column(self) -> None:
        """Constant column (std=0): sklearn returns 0, sqlearn returns NULL."""
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE const_data AS SELECT * FROM VALUES
                (1.0, 5.0),
                (2.0, 5.0),
                (3.0, 5.0),
                (4.0, 5.0),
                (5.0, 5.0)
            t(a, b)
        """)
        backend = DuckDBBackend(connection=c)
        sq_result = (
            Pipeline([StandardScaler()], backend=backend)
            .fit("const_data", backend=backend)
            .transform("const_data", backend=backend)
        )

        x_data = _table_to_numpy(c, "const_data")
        sk_result = SkStandardScaler().fit_transform(x_data)

        # Column a: non-constant, should match
        np.testing.assert_allclose(sq_result[:, 0], sk_result[:, 0], rtol=1e-6)

        # Column b: constant (std=0)
        # sklearn returns 0.0, sqlearn returns NaN (NULL from NULLIF)
        # Both are valid behaviors for zero-variance columns
        assert np.all(sk_result[:, 1] == 0.0)
        assert np.all(np.isnan(sq_result[:, 1]))

    def test_single_row(self) -> None:
        """Single row: std=0 for all columns."""
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE single AS SELECT * FROM VALUES
                (42.0, 99.0)
            t(a, b)
        """)
        backend = DuckDBBackend(connection=c)
        sq_result = (
            Pipeline([StandardScaler()], backend=backend)
            .fit("single", backend=backend)
            .transform("single", backend=backend)
        )

        # All columns have std=0, so sqlearn returns NaN (NULLIF)
        assert np.all(np.isnan(sq_result))


# ── Imputer Edge Cases ───────────────────────────────────────────────


class TestImputerAllNullsColumn:
    """Edge case: column where every value is NULL."""

    @pytest.mark.filterwarnings("ignore:Skipping features without any observed values:UserWarning")
    def test_all_nulls_mean(self) -> None:
        """Imputer(strategy='mean') on all-NULL column: mean of nothing → NaN fill.

        sqlearn only imputes columns with a learnable fill value. When all values
        are NULL, the mean is NULL (no fill value is learned), so sqlearn cannot
        fill those NULLs — the column stays all-NaN in the output. sklearn drops
        the all-NULL column from output entirely (emitting a UserWarning).

        We verify:
        - Column a (no NULLs): sqlearn output matches sklearn output.
        - Column b (all NULLs): sqlearn output is all NaN; sklearn drops the column.
        """
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE nullcol AS SELECT * FROM VALUES
                (1.0, CAST(NULL AS DOUBLE)),
                (2.0, CAST(NULL AS DOUBLE)),
                (3.0, CAST(NULL AS DOUBLE))
            t(a, b)
        """)
        backend = DuckDBBackend(connection=c)

        # Apply imputation only to column a (which has valid data).
        # Column b is all-NULL: its mean is NULL, so sqlearn cannot fill it.
        sq_result = (
            Pipeline([Imputer(strategy="mean", columns=["a"])], backend=backend)
            .fit("nullcol", backend=backend)
            .transform("nullcol", backend=backend)
        )

        # sklearn with only column a to get a fair comparison (column b is unskippable in sklearn)
        x_col_a = _table_to_numpy(c, "nullcol", columns=["a"])
        sk_result_a = SimpleImputer(strategy="mean").fit_transform(x_col_a)

        # Column a: no NULLs, should match exactly
        np.testing.assert_allclose(sq_result[:, 0], sk_result_a[:, 0], rtol=1e-6)
        # Column b: all NULLs in sqlearn output (untouched since fill value is NULL)
        assert np.all(np.isnan(sq_result[:, 1]))


# ── Imputer ──────────────────────────────────────────────────────────


class TestImputerMeanEquivalence:
    """sqlearn Imputer(strategy='mean') matches sklearn SimpleImputer(strategy='mean')."""

    def test_mean_strategy(self, sample_numeric_conn: duckdb.DuckDBPyConnection) -> None:
        """Mean imputation on numeric data with NULLs."""
        backend = DuckDBBackend(connection=sample_numeric_conn)
        sq_result = (
            Pipeline([Imputer(strategy="mean")], backend=backend)
            .fit("sample_numeric", backend=backend)
            .transform("sample_numeric", backend=backend)
        )

        x_data = _table_to_numpy(sample_numeric_conn, "sample_numeric")
        sk_result = SimpleImputer(strategy="mean").fit_transform(x_data)

        np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)


class TestImputerMedianEquivalence:
    """sqlearn Imputer(strategy='median') matches sklearn SimpleImputer(strategy='median')."""

    def test_median_strategy(self, sample_numeric_conn: duckdb.DuckDBPyConnection) -> None:
        """Median imputation on numeric data with NULLs (odd row count)."""
        backend = DuckDBBackend(connection=sample_numeric_conn)
        sq_result = (
            Pipeline([Imputer(strategy="median")], backend=backend)
            .fit("sample_numeric", backend=backend)
            .transform("sample_numeric", backend=backend)
        )

        x_data = _table_to_numpy(sample_numeric_conn, "sample_numeric")
        sk_result = SimpleImputer(strategy="median").fit_transform(x_data)

        np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)


class TestImputerMostFrequentEquivalence:
    """sqlearn Imputer(strategy='most_frequent') matches sklearn SimpleImputer."""

    @pytest.fixture
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Categorical data with NULLs. Clear modes (no ties)."""
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE cat_nulls AS SELECT * FROM VALUES
                ('red',   'small'),
                ('red',   'small'),
                ('blue',  NULL),
                ('red',   'large'),
                (NULL,    'small')
            t(color, size)
        """)
        return c

    def test_most_frequent_strategy(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Most-frequent imputation on categorical data with NULLs."""
        backend = DuckDBBackend(connection=conn)
        sq_result = (
            Pipeline([Imputer(strategy="most_frequent")], backend=backend)
            .fit("cat_nulls", backend=backend)
            .transform("cat_nulls", backend=backend)
        )

        # Build equivalent numpy array for sklearn (NULLs → None in object array).
        # Use missing_values=None so sklearn treats Python None as missing,
        # matching DuckDB NULL semantics.
        rows = conn.execute("SELECT * FROM cat_nulls").fetchall()
        x_data = np.array(rows, dtype=object)

        sk_imputer = SimpleImputer(strategy="most_frequent", missing_values=None)
        sk_result = sk_imputer.fit_transform(x_data)

        # Compare as object arrays (string values)
        np.testing.assert_array_equal(sq_result, sk_result)


# ── OneHotEncoder ────────────────────────────────────────────────────


class TestOneHotEncoderEquivalence:
    """sqlearn OneHotEncoder matches sklearn OneHotEncoder on pure categorical data."""

    def test_onehot_encoding(self, sample_categorical_conn: duckdb.DuckDBPyConnection) -> None:
        """OneHotEncoder on pure categorical data."""
        backend = DuckDBBackend(connection=sample_categorical_conn)
        sq_result = (
            Pipeline([OneHotEncoder()], backend=backend)
            .fit("sample_categorical", backend=backend)
            .transform("sample_categorical", backend=backend)
        )

        # sklearn: get categories in same order
        rows = sample_categorical_conn.execute(
            "SELECT color, size, shape FROM sample_categorical"
        ).fetchall()
        x_data = np.array(rows, dtype=object)

        sk = SkOneHotEncoder(sparse_output=False)
        sk_result = sk.fit_transform(x_data)

        # Both sort categories alphabetically, so column order should match
        np.testing.assert_allclose(
            sq_result.astype(np.float64),
            sk_result,
            rtol=1e-6,
        )

    def test_single_unique_category(self) -> None:
        """Column with single unique category produces one binary column (all 1s)."""
        c = duckdb.connect()
        c.execute("""
            CREATE TABLE single_cat AS SELECT * FROM VALUES
                ('only',),
                ('only',),
                ('only',)
            t(cat)
        """)
        backend = DuckDBBackend(connection=c)
        sq_result = (
            Pipeline([OneHotEncoder()], backend=backend)
            .fit("single_cat", backend=backend)
            .transform("single_cat", backend=backend)
        )

        x_data = np.array([["only"], ["only"], ["only"]], dtype=object)
        sk_result = SkOneHotEncoder(sparse_output=False).fit_transform(x_data)

        np.testing.assert_allclose(sq_result.astype(np.float64), sk_result, rtol=1e-6)
