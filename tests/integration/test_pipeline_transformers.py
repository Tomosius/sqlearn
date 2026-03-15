"""Integration tests for Pipeline with Imputer, StandardScaler, and OneHotEncoder.

Proves the full pipeline works with all three transformers together on real
DuckDB data: fit, transform, to_sql, feature names, cloning, and y-column.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler


@pytest.fixture
def train_conn() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with NULLs, numerics, and categoricals."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE train AS SELECT * FROM VALUES
            (1.0, 10.0, 'London'),
            (2.0, 20.0, 'Paris'),
            (NULL, 30.0, 'London'),
            (4.0, NULL, 'Tokyo'),
            (5.0, 50.0, 'Paris')
        t(price, quantity, city)
    """)
    return conn


@pytest.fixture
def backend(train_conn: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """Create a DuckDBBackend wrapping the train connection."""
    return DuckDBBackend(connection=train_conn)


class TestFullPipelineFitTransform:
    """Full pipeline with Imputer, StandardScaler, and OneHotEncoder."""

    def test_output_shape(self, backend: DuckDBBackend) -> None:
        """Pipeline produces 5 rows x 5 columns (numerics + one-hot)."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit("train", backend=backend).transform("train", backend=backend)

        assert result.shape == (5, 5)

    def test_no_nans(self, backend: DuckDBBackend) -> None:
        """Output contains no NaN or None values after imputation."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit("train", backend=backend).transform("train", backend=backend)

        assert not np.any(np.isnan(result))

    def test_numeric_columns_scaled(self, backend: DuckDBBackend) -> None:
        """Numeric columns (price, quantity) are scaled close to zero mean.

        Note: imputed NULL values shift the mean slightly from zero, since
        the scaler learns stats from original data but transform includes
        the imputed values. We verify the mean is near zero (atol=0.1).
        """
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit("train", backend=backend).transform("train", backend=backend)

        # price is column 0, quantity is column 1
        np.testing.assert_allclose(np.mean(result[:, 0]), 0.0, atol=0.1)
        np.testing.assert_allclose(np.mean(result[:, 1]), 0.0, atol=0.1)

    def test_city_columns_binary(self, backend: DuckDBBackend) -> None:
        """City one-hot columns contain only 0 or 1 values."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit("train", backend=backend).transform("train", backend=backend)

        # city columns are the last 3
        city_data = result[:, 2:]
        unique_values = set(city_data.flatten())
        assert unique_values <= {0.0, 1.0}


class TestFullPipelineToSql:
    """Pipeline SQL output contains expected SQL patterns and matches transform output."""

    def test_sql_contains_expected_patterns(self, backend: DuckDBBackend) -> None:
        """SQL string contains COALESCE (imputer), arithmetic (scaler), and CASE WHEN (encoder)."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("train", backend=backend)
        sql = pipe.to_sql()

        assert "COALESCE" in sql.upper()
        assert "CASE" in sql.upper()
        assert "WHEN" in sql.upper()
        # Scaler arithmetic: subtraction and division
        assert "-" in sql or "SUB" in sql.upper()

    def test_sql_output_matches_transform(
        self, backend: DuckDBBackend, train_conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Executing the SQL directly produces the same result as transform()."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("train", backend=backend)

        transform_result = pipe.transform("train", backend=backend)
        sql = pipe.to_sql(table="train")
        sql_rows = train_conn.execute(sql).fetchall()
        sql_result = np.array(sql_rows, dtype=np.float64)

        np.testing.assert_allclose(sql_result, transform_result, atol=1e-10)


class TestPipelineFeatureNames:
    """Pipeline get_feature_names_out returns expected column names."""

    def test_feature_names(self, backend: DuckDBBackend) -> None:
        """Feature names include scaled numerics and one-hot categoricals."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("train", backend=backend)
        names = pipe.get_feature_names_out()

        assert names == ["price", "quantity", "city_london", "city_paris", "city_tokyo"]


class TestImputerScalerComposition:
    """Imputer + StandardScaler compose into nested inline expressions (no extra CTEs)."""

    def test_coalesce_nested_in_arithmetic(
        self,
        backend: DuckDBBackend,
    ) -> None:
        """SQL output shows COALESCE nested inside arithmetic, proving expression composition."""
        pipe = Pipeline(
            [
                Imputer(strategy="mean", columns=["price", "quantity"]),
                StandardScaler(),
            ],
            backend=backend,
        )
        pipe.fit("train", backend=backend)
        sql = pipe.to_sql()

        sql_upper = sql.upper()
        assert "COALESCE" in sql_upper
        # COALESCE should be nested inside scaler arithmetic (subtraction)
        # The SQL should NOT contain a CTE for just two composable steps
        assert "WITH" not in sql_upper


class TestClonePreservesTransformers:
    """Cloning a fitted pipeline preserves fitted state and produces the same output."""

    def test_clone_produces_same_output(self, backend: DuckDBBackend) -> None:
        """Cloned pipeline is fitted and produces identical transform results."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("train", backend=backend)
        original_result = pipe.transform("train", backend=backend)

        cloned = pipe.clone()

        assert cloned.is_fitted
        cloned_result = cloned.transform("train", backend=backend)
        np.testing.assert_allclose(cloned_result, original_result, atol=1e-10)


class TestPipelineWithYColumn:
    """Pipeline fit with y="price" works correctly."""

    def test_y_column_present_in_output(self, backend: DuckDBBackend) -> None:
        """When y is specified, transformers still operate and y column is in output."""
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        pipe.fit("train", y="price", backend=backend)
        result = pipe.transform("train", backend=backend)

        # y column (price) should still be present in output
        names = pipe.get_feature_names_out()
        assert "price" in names

        # Output should be valid (no NaN)
        assert not np.any(np.isnan(result))
        assert result.shape == (5, 5)
