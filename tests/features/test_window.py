"""Tests for sqlearn.features.window -- Lag, Lead, RollingMean, RollingStd, Rank, RowNumber."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import NotFittedError
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.features.window import (
    Lag,
    Lead,
    Rank,
    RollingMean,
    RollingStd,
    RowNumber,
)

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def ts_backend() -> DuckDBBackend:
    """DuckDB backend with a sequential time-series-like table.

    Table ``t`` has columns: id (INTEGER), ts (INTEGER), value (DOUBLE),
    category (VARCHAR). 6 rows with sequential ts values.
    """
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE t AS SELECT * FROM VALUES
            (1, 1, 10.0, 'A'),
            (2, 2, 20.0, 'A'),
            (3, 3, 30.0, 'A'),
            (4, 4, 40.0, 'B'),
            (5, 5, 50.0, 'B'),
            (6, 6, 60.0, 'B')
        t(id, ts, value, category)
    """)
    return DuckDBBackend(connection=conn)


@pytest.fixture
def ts_numeric_backend() -> DuckDBBackend:
    """DuckDB backend with numeric-only columns for value correctness tests.

    Table ``t`` has columns: ts (INTEGER), value (DOUBLE). 6 rows.
    """
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE t AS SELECT * FROM VALUES
            (1, 10.0),
            (2, 20.0),
            (3, 30.0),
            (4, 40.0),
            (5, 50.0),
            (6, 60.0)
        t(ts, value)
    """)
    return DuckDBBackend(connection=conn)


@pytest.fixture
def ts_backend_with_nulls() -> DuckDBBackend:
    """DuckDB backend with NULLs in the value column."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE t AS SELECT * FROM VALUES
            (1, 1, 10.0, 'A'),
            (2, 2, NULL,  'A'),
            (3, 3, 30.0, 'A'),
            (4, 4, 40.0, 'B'),
            (5, 5, NULL,  'B'),
            (6, 6, 60.0, 'B')
        t(id, ts, value, category)
    """)
    return DuckDBBackend(connection=conn)


def _make_input_query() -> exp.Select:
    """Create a simple input query for unit tests."""
    return exp.select(exp.Star()).from_("t")  # pyright: ignore[reportUnknownMemberType]


# =============================================================================
# Lag
# =============================================================================


class TestLagConstructor:
    """Test Lag constructor validation and attributes."""

    def test_defaults(self) -> None:
        """Default periods=1, partition_by=None."""
        lag = Lag(order_by="ts")
        assert lag.periods == 1
        assert lag.order_by == "ts"
        assert lag.partition_by is None

    def test_custom_periods(self) -> None:
        """Custom periods stored correctly."""
        lag = Lag(periods=3, order_by="ts")
        assert lag.periods == 3

    def test_partition_by(self) -> None:
        """Partition by stored correctly."""
        lag = Lag(order_by="ts", partition_by="group")
        assert lag.partition_by == "group"

    def test_partition_by_list(self) -> None:
        """Partition by as list stored correctly."""
        lag = Lag(order_by="ts", partition_by=["g1", "g2"])
        assert lag.partition_by == ["g1", "g2"]

    def test_order_by_list(self) -> None:
        """Order by as list stored correctly."""
        lag = Lag(order_by=["ts", "id"])
        assert lag.order_by == ["ts", "id"]

    def test_zero_periods_raises(self) -> None:
        """Zero periods raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Lag(periods=0, order_by="ts")

    def test_negative_periods_raises(self) -> None:
        """Negative periods raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Lag(periods=-1, order_by="ts")

    def test_float_periods_raises(self) -> None:
        """Float periods raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Lag(periods=1.5, order_by="ts")  # type: ignore[arg-type]

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Lag(order_by=[])

    def test_classification_is_static(self) -> None:
        """Lag is classified as static."""
        assert Lag._classification == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns targets numeric columns."""
        assert Lag._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        lag = Lag(periods=2, order_by="ts", partition_by="grp")
        params = lag.get_params()
        assert params["periods"] == 2
        assert params["order_by"] == "ts"
        assert params["partition_by"] == "grp"

    def test_repr(self) -> None:
        """repr shows non-default params."""
        lag = Lag(periods=2, order_by="ts")
        r = repr(lag)
        assert "Lag(" in r
        assert "periods=2" in r


class TestLagQuery:
    """Test Lag.query() generates correct sqlglot ASTs."""

    def test_generates_lag_function(self) -> None:
        """Query contains LAG function."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "LAG" in sql

    def test_contains_order_by(self) -> None:
        """Query contains ORDER BY clause."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "ORDER BY" in sql

    def test_contains_alias(self) -> None:
        """Query contains correct column alias."""
        lag = Lag(periods=2, order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "value_lag2" in sql

    def test_contains_partition_by(self) -> None:
        """Partition by appears in SQL when specified."""
        lag = Lag(order_by="ts", partition_by="category")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "PARTITION BY" in sql

    def test_no_partition_by_when_none(self) -> None:
        """No PARTITION BY when not specified."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "PARTITION BY" not in sql

    def test_multiple_columns(self) -> None:
        """Multiple target columns produce multiple LAG expressions."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value", "id"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "value_lag1" in sql
        assert "id_lag1" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        assert isinstance(result, exp.Select)

    def test_wraps_input_as_subquery(self) -> None:
        """query() wraps input in a subquery with __input__ alias."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        result = lag.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "__input__" in sql


class TestLagPipeline:
    """Test Lag integrated with Pipeline (end-to-end)."""

    def test_output_adds_lag_column(self, ts_backend: DuckDBBackend) -> None:
        """Lag adds a new column to output."""
        pipe = Pipeline([Lag(order_by="ts", columns=["value"])], backend=ts_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "value_lag1" in names
        assert "value" in names

    def test_lag_values_correct(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Lag column contains shifted values."""
        pipe = Pipeline([Lag(order_by="ts", columns=["value"])], backend=ts_numeric_backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lag_idx = names.index("value_lag1")
        val_idx = names.index("value")
        # First row lag should be NULL (NaN in numpy)
        assert np.isnan(result[0, lag_idx])
        # Second row lag should be the first row's value
        assert result[1, lag_idx] == result[0, val_idx]

    def test_lag_periods_2(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Lag with periods=2 shifts by 2 rows."""
        pipe = Pipeline(
            [Lag(periods=2, order_by="ts", columns=["value"])],
            backend=ts_numeric_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lag_idx = names.index("value_lag2")
        # First two rows should be NULL
        assert np.isnan(result[0, lag_idx])
        assert np.isnan(result[1, lag_idx])
        # Third row should have first row's value
        val_idx = names.index("value")
        assert result[2, lag_idx] == result[0, val_idx]

    def test_lag_with_partition(self, ts_backend: DuckDBBackend) -> None:
        """Lag with partition_by resets at partition boundaries."""
        pipe = Pipeline(
            [Lag(order_by="ts", partition_by="category", columns=["value"])],
            backend=ts_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lag_idx = names.index("value_lag1")
        # Within each partition, first row's lag should be NULL (None in object array)
        null_count = sum(1 for i in range(result.shape[0]) if result[i, lag_idx] is None)
        assert null_count >= 2  # At least one per partition

    def test_to_sql_contains_lag(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains LAG."""
        pipe = Pipeline([Lag(order_by="ts", columns=["value"])], backend=ts_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "LAG" in sql

    def test_column_count(self, ts_backend: DuckDBBackend) -> None:
        """Output has original columns plus lag column."""
        pipe = Pipeline([Lag(order_by="ts", columns=["value"])], backend=ts_backend)
        result = pipe.fit_transform("t")
        # Original: id, ts, value, category -> 4 + 1 lag = 5
        assert result.shape[1] == 5

    def test_null_handling(self, ts_backend_with_nulls: DuckDBBackend) -> None:
        """Lag handles NULLs in source data."""
        pipe = Pipeline(
            [Lag(order_by="ts", columns=["value"])],
            backend=ts_backend_with_nulls,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6


class TestLagClonePickle:
    """Clone and pickle roundtrip for Lag."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Lag has same params but is independent."""
        lag = Lag(periods=2, order_by="ts", partition_by="grp")
        cloned = lag.clone()
        assert cloned.periods == 2
        assert cloned.order_by == "ts"
        assert cloned.partition_by == "grp"
        assert cloned is not lag

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        lag = Lag(periods=2, order_by="ts")
        cloned = lag.clone()
        cloned.set_params(periods=5)
        assert lag.periods == 2
        assert cloned.periods == 5

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        lag = Lag(periods=3, order_by=["ts", "id"], partition_by="cat")
        data = pickle.dumps(lag)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.periods == 3
        assert restored.order_by == ["ts", "id"]
        assert restored.partition_by == "cat"


# =============================================================================
# Lead
# =============================================================================


class TestLeadConstructor:
    """Test Lead constructor validation and attributes."""

    def test_defaults(self) -> None:
        """Default periods=1, partition_by=None."""
        lead = Lead(order_by="ts")
        assert lead.periods == 1
        assert lead.partition_by is None

    def test_custom_periods(self) -> None:
        """Custom periods stored correctly."""
        lead = Lead(periods=3, order_by="ts")
        assert lead.periods == 3

    def test_zero_periods_raises(self) -> None:
        """Zero periods raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Lead(periods=0, order_by="ts")

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Lead(order_by=[])

    def test_classification_is_static(self) -> None:
        """Lead is classified as static."""
        assert Lead._classification == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns targets numeric columns."""
        assert Lead._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        lead = Lead(periods=2, order_by="ts")
        params = lead.get_params()
        assert params["periods"] == 2
        assert params["order_by"] == "ts"

    def test_repr(self) -> None:
        """repr shows non-default params."""
        lead = Lead(periods=2, order_by="ts")
        r = repr(lead)
        assert "Lead(" in r
        assert "periods=2" in r


class TestLeadQuery:
    """Test Lead.query() generates correct sqlglot ASTs."""

    def test_generates_lead_function(self) -> None:
        """Query contains LEAD function."""
        lead = Lead(order_by="ts")
        lead.columns_ = ["value"]
        result = lead.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "LEAD" in sql

    def test_contains_alias(self) -> None:
        """Query contains correct column alias."""
        lead = Lead(periods=2, order_by="ts")
        lead.columns_ = ["value"]
        result = lead.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "value_lead2" in sql

    def test_contains_partition_by(self) -> None:
        """Partition by appears in SQL when specified."""
        lead = Lead(order_by="ts", partition_by="category")
        lead.columns_ = ["value"]
        result = lead.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "PARTITION BY" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        lead = Lead(order_by="ts")
        lead.columns_ = ["value"]
        result = lead.query(_make_input_query())
        assert isinstance(result, exp.Select)


class TestLeadPipeline:
    """Test Lead integrated with Pipeline (end-to-end)."""

    def test_output_adds_lead_column(self, ts_backend: DuckDBBackend) -> None:
        """Lead adds a new column to output."""
        pipe = Pipeline([Lead(order_by="ts", columns=["value"])], backend=ts_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "value_lead1" in names

    def test_lead_values_correct(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Lead column contains forward-shifted values."""
        pipe = Pipeline([Lead(order_by="ts", columns=["value"])], backend=ts_numeric_backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lead_idx = names.index("value_lead1")
        val_idx = names.index("value")
        # Last row lead should be NULL (NaN in numpy)
        assert np.isnan(result[-1, lead_idx])
        # First row lead should be second row's value
        assert result[0, lead_idx] == result[1, val_idx]

    def test_lead_with_partition(self, ts_backend: DuckDBBackend) -> None:
        """Lead with partition_by resets at partition boundaries."""
        pipe = Pipeline(
            [Lead(order_by="ts", partition_by="category", columns=["value"])],
            backend=ts_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lead_idx = names.index("value_lead1")
        # Last row in each partition should be None (object array due to VARCHAR)
        null_count = sum(1 for i in range(result.shape[0]) if result[i, lead_idx] is None)
        assert null_count >= 2

    def test_to_sql_contains_lead(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains LEAD."""
        pipe = Pipeline([Lead(order_by="ts", columns=["value"])], backend=ts_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "LEAD" in sql

    def test_null_handling(self, ts_backend_with_nulls: DuckDBBackend) -> None:
        """Lead handles NULLs in source data."""
        pipe = Pipeline(
            [Lead(order_by="ts", columns=["value"])],
            backend=ts_backend_with_nulls,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6


class TestLeadClonePickle:
    """Clone and pickle roundtrip for Lead."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Lead has same params but is independent."""
        lead = Lead(periods=2, order_by="ts", partition_by="grp")
        cloned = lead.clone()
        assert cloned.periods == 2
        assert cloned.order_by == "ts"
        assert cloned is not lead

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        lead = Lead(periods=3, order_by="ts")
        data = pickle.dumps(lead)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.periods == 3
        assert restored.order_by == "ts"


# =============================================================================
# RollingMean
# =============================================================================


class TestRollingMeanConstructor:
    """Test RollingMean constructor validation."""

    def test_basic(self) -> None:
        """Basic creation with required params."""
        rm = RollingMean(window=3, order_by="ts")
        assert rm.window == 3
        assert rm.order_by == "ts"
        assert rm.partition_by is None

    def test_partition_by(self) -> None:
        """Partition by stored correctly."""
        rm = RollingMean(window=3, order_by="ts", partition_by="grp")
        assert rm.partition_by == "grp"

    def test_window_1_raises(self) -> None:
        """Window < 2 raises ValueError."""
        with pytest.raises(ValueError, match="integer >= 2"):
            RollingMean(window=1, order_by="ts")

    def test_window_0_raises(self) -> None:
        """Window of 0 raises ValueError."""
        with pytest.raises(ValueError, match="integer >= 2"):
            RollingMean(window=0, order_by="ts")

    def test_window_float_raises(self) -> None:
        """Float window raises ValueError."""
        with pytest.raises(ValueError, match="integer >= 2"):
            RollingMean(window=3.5, order_by="ts")  # type: ignore[arg-type]

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            RollingMean(window=3, order_by=[])

    def test_classification_is_static(self) -> None:
        """RollingMean is classified as static."""
        assert RollingMean._classification == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns targets numeric columns."""
        assert RollingMean._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        rm = RollingMean(window=5, order_by="ts")
        params = rm.get_params()
        assert params["window"] == 5
        assert params["order_by"] == "ts"

    def test_repr(self) -> None:
        """repr shows params."""
        rm = RollingMean(window=3, order_by="ts")
        r = repr(rm)
        assert "RollingMean(" in r
        assert "window=3" in r


class TestRollingMeanQuery:
    """Test RollingMean.query() generates correct sqlglot ASTs."""

    def test_generates_avg_function(self) -> None:
        """Query contains AVG function."""
        rm = RollingMean(window=3, order_by="ts")
        rm.columns_ = ["value"]
        result = rm.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "AVG" in sql

    def test_contains_rows_frame(self) -> None:
        """Query contains ROWS BETWEEN ... PRECEDING AND CURRENT ROW."""
        rm = RollingMean(window=3, order_by="ts")
        rm.columns_ = ["value"]
        result = rm.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "ROWS BETWEEN" in sql
        assert "PRECEDING" in sql
        assert "CURRENT ROW" in sql

    def test_contains_alias(self) -> None:
        """Query contains correct column alias."""
        rm = RollingMean(window=3, order_by="ts")
        rm.columns_ = ["value"]
        result = rm.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "value_rmean3" in sql

    def test_window_frame_size(self) -> None:
        """PRECEDING value is window - 1."""
        rm = RollingMean(window=5, order_by="ts")
        rm.columns_ = ["value"]
        result = rm.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "4 PRECEDING" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        rm = RollingMean(window=3, order_by="ts")
        rm.columns_ = ["value"]
        result = rm.query(_make_input_query())
        assert isinstance(result, exp.Select)


class TestRollingMeanPipeline:
    """Test RollingMean integrated with Pipeline."""

    def test_output_adds_rmean_column(self, ts_backend: DuckDBBackend) -> None:
        """RollingMean adds a new column to output."""
        pipe = Pipeline(
            [RollingMean(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend,
        )
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "value_rmean3" in names

    def test_rolling_mean_values(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Rolling mean computes correct averages."""
        pipe = Pipeline(
            [RollingMean(window=3, order_by="ts", columns=["value"])],
            backend=ts_numeric_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rmean_idx = names.index("value_rmean3")
        # Row 0: avg(10) = 10 (partial window)
        assert result[0, rmean_idx] == pytest.approx(10.0)
        # Row 1: avg(10, 20) = 15 (partial window)
        assert result[1, rmean_idx] == pytest.approx(15.0)
        # Row 2: avg(10, 20, 30) = 20 (full window)
        assert result[2, rmean_idx] == pytest.approx(20.0)
        # Row 3: avg(20, 30, 40) = 30
        assert result[3, rmean_idx] == pytest.approx(30.0)

    def test_rolling_mean_with_partition(self, ts_backend: DuckDBBackend) -> None:
        """Rolling mean with partition resets at boundaries."""
        pipe = Pipeline(
            [
                RollingMean(
                    window=3,
                    order_by="ts",
                    partition_by="category",
                    columns=["value"],
                )
            ],
            backend=ts_backend,
        )
        result = pipe.fit_transform("t")
        # Each partition has 3 rows, so the rolling mean resets
        assert result.shape[0] == 6

    def test_to_sql_contains_avg(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains AVG."""
        pipe = Pipeline(
            [RollingMean(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend,
        )
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "AVG" in sql

    def test_null_handling(self, ts_backend_with_nulls: DuckDBBackend) -> None:
        """RollingMean handles NULLs in source data."""
        pipe = Pipeline(
            [RollingMean(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend_with_nulls,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6


class TestRollingMeanClonePickle:
    """Clone and pickle roundtrip for RollingMean."""

    def test_clone_roundtrip(self) -> None:
        """Cloned RollingMean has same params but is independent."""
        rm = RollingMean(window=5, order_by="ts", partition_by="grp")
        cloned = rm.clone()
        assert cloned.window == 5
        assert cloned.order_by == "ts"
        assert cloned is not rm

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        rm = RollingMean(window=3, order_by="ts")
        data = pickle.dumps(rm)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.window == 3
        assert restored.order_by == "ts"


# =============================================================================
# RollingStd
# =============================================================================


class TestRollingStdConstructor:
    """Test RollingStd constructor validation."""

    def test_basic(self) -> None:
        """Basic creation with required params."""
        rs = RollingStd(window=3, order_by="ts")
        assert rs.window == 3
        assert rs.order_by == "ts"

    def test_window_1_raises(self) -> None:
        """Window < 2 raises ValueError."""
        with pytest.raises(ValueError, match="integer >= 2"):
            RollingStd(window=1, order_by="ts")

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            RollingStd(window=3, order_by=[])

    def test_classification_is_static(self) -> None:
        """RollingStd is classified as static."""
        assert RollingStd._classification == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns targets numeric columns."""
        assert RollingStd._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        rs = RollingStd(window=4, order_by="ts")
        params = rs.get_params()
        assert params["window"] == 4

    def test_repr(self) -> None:
        """repr shows params."""
        rs = RollingStd(window=3, order_by="ts")
        r = repr(rs)
        assert "RollingStd(" in r
        assert "window=3" in r


class TestRollingStdQuery:
    """Test RollingStd.query() generates correct sqlglot ASTs."""

    def test_generates_stddev_function(self) -> None:
        """Query contains STDDEV_POP function."""
        rs = RollingStd(window=3, order_by="ts")
        rs.columns_ = ["value"]
        result = rs.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "STDDEV_POP" in sql

    def test_contains_rows_frame(self) -> None:
        """Query contains ROWS BETWEEN frame."""
        rs = RollingStd(window=3, order_by="ts")
        rs.columns_ = ["value"]
        result = rs.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "ROWS BETWEEN" in sql

    def test_contains_alias(self) -> None:
        """Query contains correct column alias."""
        rs = RollingStd(window=3, order_by="ts")
        rs.columns_ = ["value"]
        result = rs.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "value_rstd3" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        rs = RollingStd(window=3, order_by="ts")
        rs.columns_ = ["value"]
        result = rs.query(_make_input_query())
        assert isinstance(result, exp.Select)


class TestRollingStdPipeline:
    """Test RollingStd integrated with Pipeline."""

    def test_output_adds_rstd_column(self, ts_backend: DuckDBBackend) -> None:
        """RollingStd adds a new column to output."""
        pipe = Pipeline(
            [RollingStd(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend,
        )
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "value_rstd3" in names

    def test_rolling_std_first_row_zero(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Rolling std of a single value is 0 (population stddev)."""
        pipe = Pipeline(
            [RollingStd(window=3, order_by="ts", columns=["value"])],
            backend=ts_numeric_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rstd_idx = names.index("value_rstd3")
        # Row 0: stddev_pop(10) = 0
        assert result[0, rstd_idx] == pytest.approx(0.0)

    def test_rolling_std_values(self, ts_numeric_backend: DuckDBBackend) -> None:
        """Rolling std computes correct standard deviations."""
        pipe = Pipeline(
            [RollingStd(window=3, order_by="ts", columns=["value"])],
            backend=ts_numeric_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rstd_idx = names.index("value_rstd3")
        # Row 2: stddev_pop(10, 20, 30) = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3)
        # = sqrt(200/3) ~= 8.165
        assert result[2, rstd_idx] == pytest.approx(8.16496580927726, rel=1e-4)

    def test_to_sql_contains_stddev(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains STDDEV_POP."""
        pipe = Pipeline(
            [RollingStd(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend,
        )
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "STDDEV_POP" in sql

    def test_null_handling(self, ts_backend_with_nulls: DuckDBBackend) -> None:
        """RollingStd handles NULLs in source data."""
        pipe = Pipeline(
            [RollingStd(window=3, order_by="ts", columns=["value"])],
            backend=ts_backend_with_nulls,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6


class TestRollingStdClonePickle:
    """Clone and pickle roundtrip for RollingStd."""

    def test_clone_roundtrip(self) -> None:
        """Cloned RollingStd has same params."""
        rs = RollingStd(window=5, order_by="ts")
        cloned = rs.clone()
        assert cloned.window == 5
        assert cloned is not rs

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        rs = RollingStd(window=3, order_by="ts")
        data = pickle.dumps(rs)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.window == 3


# =============================================================================
# Rank
# =============================================================================


class TestRankConstructor:
    """Test Rank constructor validation."""

    def test_defaults(self) -> None:
        """Default method='rank', partition_by=None."""
        rank = Rank(order_by="score")
        assert rank.method == "rank"
        assert rank.partition_by is None

    def test_dense_rank(self) -> None:
        """method='dense_rank' accepted."""
        rank = Rank(order_by="score", method="dense_rank")
        assert rank.method == "dense_rank"

    def test_row_number_method(self) -> None:
        """method='row_number' accepted."""
        rank = Rank(order_by="score", method="row_number")
        assert rank.method == "row_number"

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            Rank(order_by="score", method="percentile")

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Rank(order_by=[])

    def test_classification_is_static(self) -> None:
        """Rank is classified as static."""
        assert Rank._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Default columns is None."""
        assert Rank._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        rank = Rank(order_by="score", method="dense_rank")
        params = rank.get_params()
        assert params["order_by"] == "score"
        assert params["method"] == "dense_rank"

    def test_repr(self) -> None:
        """repr shows non-default params."""
        rank = Rank(order_by="score", method="dense_rank")
        r = repr(rank)
        assert "Rank(" in r
        assert "method='dense_rank'" in r


class TestRankQuery:
    """Test Rank.query() generates correct sqlglot ASTs."""

    def test_generates_rank_function(self) -> None:
        """Query contains RANK function."""
        rank = Rank(order_by="score")
        rank.columns_ = []
        result = rank.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "RANK" in sql

    def test_generates_dense_rank(self) -> None:
        """Query contains DENSE_RANK function."""
        rank = Rank(order_by="score", method="dense_rank")
        rank.columns_ = []
        result = rank.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "DENSE_RANK" in sql

    def test_no_columns_creates_rank_alias(self) -> None:
        """No explicit columns produces a single 'rank' column."""
        rank = Rank(order_by="score")
        rank.columns_ = ["a", "b"]  # simulates pipeline auto-resolution
        result = rank.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        # Should create single "rank" column, not per-column
        assert "AS rank" in sql

    def test_with_columns_creates_col_rank(self) -> None:
        """Target columns produce {col}_rank aliases."""
        rank = Rank(order_by="score", columns=["score"])
        rank.columns_ = ["score"]
        result = rank.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "score_rank" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        rank = Rank(order_by="score")
        rank.columns_ = []
        result = rank.query(_make_input_query())
        assert isinstance(result, exp.Select)

    def test_partition_by(self) -> None:
        """Partition by appears in SQL when specified."""
        rank = Rank(order_by="score", partition_by="group")
        rank.columns_ = []
        result = rank.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "PARTITION BY" in sql


class TestRankPipeline:
    """Test Rank integrated with Pipeline."""

    def test_rank_without_columns(self) -> None:
        """Rank without columns creates a single 'rank' column."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1, 10.0), (2, 20.0), (3, 30.0) t(ts, value)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rank(order_by="value")], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "rank" in names

    def test_rank_values_ordered(self) -> None:
        """Rank values are sequential for unique values."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES "
            "(1, 10.0), (2, 20.0), (3, 30.0), "
            "(4, 40.0), (5, 50.0), (6, 60.0) t(ts, value)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rank(order_by="value")], backend=backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rank_idx = names.index("rank")
        ranks = sorted(result[:, rank_idx])
        assert list(ranks) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_dense_rank_with_ties(self) -> None:
        """Dense rank handles ties correctly."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE t AS SELECT * FROM VALUES
                (1, 10.0),
                (2, 20.0),
                (3, 20.0),
                (4, 30.0)
            t(id, score)
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Rank(order_by="score", method="dense_rank", columns=["score"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rank_idx = names.index("score_rank")
        ranks = sorted(result[:, rank_idx])
        # Dense rank: 10->1, 20->2, 20->2, 30->3
        assert list(ranks) == [1.0, 2.0, 2.0, 3.0]

    def test_rank_with_partition(self, ts_backend: DuckDBBackend) -> None:
        """Rank with partition resets at boundaries."""
        pipe = Pipeline(
            [
                Rank(
                    order_by="value",
                    partition_by="category",
                    columns=["value"],
                )
            ],
            backend=ts_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rank_idx = names.index("value_rank")
        # Each partition has 3 rows, ranks should be 1,2,3 in each
        ranks = sorted(result[:, rank_idx])
        assert list(ranks) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    def test_to_sql_contains_rank(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains RANK."""
        pipe = Pipeline([Rank(order_by="value")], backend=ts_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "RANK" in sql


class TestRankClonePickle:
    """Clone and pickle roundtrip for Rank."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Rank has same params."""
        rank = Rank(order_by="score", method="dense_rank")
        cloned = rank.clone()
        assert cloned.method == "dense_rank"
        assert cloned.order_by == "score"
        assert cloned is not rank

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        rank = Rank(order_by="score", method="row_number", partition_by="grp")
        data = pickle.dumps(rank)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.method == "row_number"
        assert restored.partition_by == "grp"


# =============================================================================
# RowNumber
# =============================================================================


class TestRowNumberConstructor:
    """Test RowNumber constructor validation."""

    def test_defaults(self) -> None:
        """Default partition_by=None."""
        rn = RowNumber(order_by="ts")
        assert rn.order_by == "ts"
        assert rn.partition_by is None

    def test_partition_by(self) -> None:
        """Partition by stored correctly."""
        rn = RowNumber(order_by="ts", partition_by="grp")
        assert rn.partition_by == "grp"

    def test_empty_order_by_raises(self) -> None:
        """Empty order_by raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            RowNumber(order_by=[])

    def test_classification_is_static(self) -> None:
        """RowNumber is classified as static."""
        assert RowNumber._classification == "static"

    def test_columns_is_none(self) -> None:
        """RowNumber ignores columns param."""
        rn = RowNumber(order_by="ts")
        assert rn.columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        rn = RowNumber(order_by="ts", partition_by="grp")
        params = rn.get_params()
        assert params["order_by"] == "ts"
        assert params["partition_by"] == "grp"

    def test_repr(self) -> None:
        """repr shows params."""
        rn = RowNumber(order_by="ts")
        r = repr(rn)
        assert "RowNumber(" in r


class TestRowNumberQuery:
    """Test RowNumber.query() generates correct sqlglot ASTs."""

    def test_generates_row_number(self) -> None:
        """Query contains ROW_NUMBER function."""
        rn = RowNumber(order_by="ts")
        result = rn.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "ROW_NUMBER" in sql

    def test_contains_alias(self) -> None:
        """Query contains row_number alias."""
        rn = RowNumber(order_by="ts")
        result = rn.query(_make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "row_number" in sql.lower()

    def test_contains_partition_by(self) -> None:
        """Partition by appears in SQL when specified."""
        rn = RowNumber(order_by="ts", partition_by="grp")
        result = rn.query(_make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "PARTITION BY" in sql

    def test_returns_select(self) -> None:
        """query() returns a Select expression."""
        rn = RowNumber(order_by="ts")
        result = rn.query(_make_input_query())
        assert isinstance(result, exp.Select)


class TestRowNumberPipeline:
    """Test RowNumber integrated with Pipeline."""

    def test_output_adds_row_number_column(self, ts_backend: DuckDBBackend) -> None:
        """RowNumber adds a row_number column."""
        pipe = Pipeline([RowNumber(order_by="ts")], backend=ts_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "row_number" in names

    def test_row_number_values(self, ts_backend: DuckDBBackend) -> None:
        """Row numbers are 1..N sequential."""
        pipe = Pipeline([RowNumber(order_by="ts")], backend=ts_backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rn_idx = names.index("row_number")
        row_nums = sorted(result[:, rn_idx])
        assert list(row_nums) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_row_number_with_partition(self, ts_backend: DuckDBBackend) -> None:
        """Row numbers restart per partition."""
        pipe = Pipeline(
            [RowNumber(order_by="ts", partition_by="category")],
            backend=ts_backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rn_idx = names.index("row_number")
        row_nums = sorted(result[:, rn_idx])
        # Each partition (A, B) has 3 rows: 1,2,3 + 1,2,3
        assert list(row_nums) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    def test_to_sql_contains_row_number(self, ts_backend: DuckDBBackend) -> None:
        """to_sql() output contains ROW_NUMBER."""
        pipe = Pipeline([RowNumber(order_by="ts")], backend=ts_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "ROW_NUMBER" in sql

    def test_column_count(self, ts_backend: DuckDBBackend) -> None:
        """Output has original columns + row_number."""
        pipe = Pipeline([RowNumber(order_by="ts")], backend=ts_backend)
        result = pipe.fit_transform("t")
        # Original: id, ts, value, category -> 4 + 1 = 5
        assert result.shape[1] == 5

    def test_row_count_unchanged(self, ts_backend: DuckDBBackend) -> None:
        """Row count is unchanged after adding row numbers."""
        pipe = Pipeline([RowNumber(order_by="ts")], backend=ts_backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6


class TestRowNumberClonePickle:
    """Clone and pickle roundtrip for RowNumber."""

    def test_clone_roundtrip(self) -> None:
        """Cloned RowNumber has same params."""
        rn = RowNumber(order_by="ts", partition_by="grp")
        cloned = rn.clone()
        assert cloned.order_by == "ts"
        assert cloned.partition_by == "grp"
        assert cloned is not rn

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves params."""
        rn = RowNumber(order_by="ts")
        data = pickle.dumps(rn)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.order_by == "ts"


# =============================================================================
# output_schema tests
# =============================================================================


class TestOutputSchema:
    """Test output_schema for all window transforms."""

    def _schema(self) -> Schema:
        """Create a test schema."""
        return Schema({"id": "INTEGER", "ts": "INTEGER", "value": "DOUBLE"})

    def test_lag_adds_column(self) -> None:
        """Lag output_schema adds lag column."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["value"]
        out = lag.output_schema(self._schema())
        assert "value_lag1" in out.columns
        assert len(out) == 4

    def test_lead_adds_column(self) -> None:
        """Lead output_schema adds lead column."""
        lead = Lead(periods=2, order_by="ts")
        lead.columns_ = ["value"]
        out = lead.output_schema(self._schema())
        assert "value_lead2" in out.columns
        assert len(out) == 4

    def test_rolling_mean_adds_column(self) -> None:
        """RollingMean output_schema adds rmean column."""
        rm = RollingMean(window=3, order_by="ts")
        rm.columns_ = ["value"]
        out = rm.output_schema(self._schema())
        assert "value_rmean3" in out.columns
        assert out.columns["value_rmean3"] == "DOUBLE"

    def test_rolling_std_adds_column(self) -> None:
        """RollingStd output_schema adds rstd column."""
        rs = RollingStd(window=3, order_by="ts")
        rs.columns_ = ["value"]
        out = rs.output_schema(self._schema())
        assert "value_rstd3" in out.columns
        assert out.columns["value_rstd3"] == "DOUBLE"

    def test_rank_no_columns_adds_rank(self) -> None:
        """Rank without explicit columns adds a single 'rank' column."""
        rank = Rank(order_by="value")
        rank.columns_ = ["id", "ts", "value"]  # auto-resolved by pipeline
        out = rank.output_schema(self._schema())
        assert "rank" in out.columns
        assert out.columns["rank"] == "BIGINT"

    def test_rank_with_columns_adds_col_rank(self) -> None:
        """Rank with explicit columns adds '{col}_rank' column."""
        rank = Rank(order_by="value", columns=["value"])
        rank.columns_ = ["value"]
        out = rank.output_schema(self._schema())
        assert "value_rank" in out.columns
        assert out.columns["value_rank"] == "BIGINT"

    def test_row_number_adds_column(self) -> None:
        """RowNumber output_schema adds row_number column."""
        rn = RowNumber(order_by="ts")
        out = rn.output_schema(self._schema())
        assert "row_number" in out.columns
        assert out.columns["row_number"] == "BIGINT"

    def test_lag_preserves_original_type(self) -> None:
        """Lag preserves the original column type for the lag column."""
        schema = Schema({"ts": "INTEGER", "score": "INTEGER"})
        lag = Lag(order_by="ts")
        lag.columns_ = ["score"]
        out = lag.output_schema(schema)
        assert out.columns["score_lag1"] == "INTEGER"

    def test_multiple_columns_output_schema(self) -> None:
        """Output schema with multiple target columns adds all lag columns."""
        lag = Lag(order_by="ts")
        lag.columns_ = ["id", "value"]
        out = lag.output_schema(self._schema())
        assert "id_lag1" in out.columns
        assert "value_lag1" in out.columns
        assert len(out) == 5  # 3 original + 2 lag


# =============================================================================
# Not-fitted guard tests
# =============================================================================


class TestNotFittedGuard:
    """Test that unfitted transformers raise NotFittedError."""

    def test_lag_not_fitted(self) -> None:
        """Lag get_feature_names_out raises when not fitted."""
        lag = Lag(order_by="ts")
        with pytest.raises(NotFittedError):
            lag.get_feature_names_out()

    def test_lead_not_fitted(self) -> None:
        """Lead get_feature_names_out raises when not fitted."""
        lead = Lead(order_by="ts")
        with pytest.raises(NotFittedError):
            lead.get_feature_names_out()

    def test_rolling_mean_not_fitted(self) -> None:
        """RollingMean get_feature_names_out raises when not fitted."""
        rm = RollingMean(window=3, order_by="ts")
        with pytest.raises(NotFittedError):
            rm.get_feature_names_out()

    def test_rolling_std_not_fitted(self) -> None:
        """RollingStd get_feature_names_out raises when not fitted."""
        rs = RollingStd(window=3, order_by="ts")
        with pytest.raises(NotFittedError):
            rs.get_feature_names_out()

    def test_rank_not_fitted(self) -> None:
        """Rank get_feature_names_out raises when not fitted."""
        rank = Rank(order_by="score")
        with pytest.raises(NotFittedError):
            rank.get_feature_names_out()

    def test_row_number_not_fitted(self) -> None:
        """RowNumber get_feature_names_out raises when not fitted."""
        rn = RowNumber(order_by="ts")
        with pytest.raises(NotFittedError):
            rn.get_feature_names_out()


# =============================================================================
# Composition tests
# =============================================================================


class TestComposition:
    """Test window transforms composing with other transformers."""

    def test_scaler_then_lag(self) -> None:
        """StandardScaler + Lag produces scaled values with lag."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0) t(ts, value)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(columns=["value"]), Lag(order_by="ts", columns=["value"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape[1] == 3  # ts, value, value_lag1

    def test_lag_then_lead(self) -> None:
        """Lag + Lead in sequence produces SQL with both window functions."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0) t(ts, value)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Lag(order_by="ts", columns=["value"]),
                Lead(order_by="ts", columns=["value"]),
            ],
            backend=backend,
        )
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        # Both window functions should appear in the composed SQL
        assert "LAG" in sql
        assert "LEAD" in sql
        # The SQL should execute successfully
        result = pipe.transform("t")
        assert result.shape[0] == 4

    def test_sample_then_row_number(self) -> None:
        """Sample + RowNumber adds row numbers to sampled data."""
        from sqlearn.ops.sample import Sample

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS ts, i * 10 AS value FROM range(20) tbl(i)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Sample(n=10), RowNumber(order_by="ts")],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 10
        names = pipe.get_feature_names_out()
        assert "row_number" in names


# =============================================================================
# expressions() no-op tests
# =============================================================================


class TestExpressionsNoOp:
    """Verify all window transforms return empty from expressions()."""

    def test_lag_expressions_empty(self) -> None:
        """Lag expressions() returns empty dict."""
        lag = Lag(order_by="ts")
        assert lag.expressions(["a"], {"a": exp.Column(this="a")}) == {}

    def test_lead_expressions_empty(self) -> None:
        """Lead expressions() returns empty dict."""
        lead = Lead(order_by="ts")
        assert lead.expressions(["a"], {"a": exp.Column(this="a")}) == {}

    def test_rolling_mean_expressions_empty(self) -> None:
        """RollingMean expressions() returns empty dict."""
        rm = RollingMean(window=3, order_by="ts")
        assert rm.expressions(["a"], {"a": exp.Column(this="a")}) == {}

    def test_rolling_std_expressions_empty(self) -> None:
        """RollingStd expressions() returns empty dict."""
        rs = RollingStd(window=3, order_by="ts")
        assert rs.expressions(["a"], {"a": exp.Column(this="a")}) == {}

    def test_rank_expressions_empty(self) -> None:
        """Rank expressions() returns empty dict."""
        rank = Rank(order_by="score")
        assert rank.expressions(["a"], {"a": exp.Column(this="a")}) == {}

    def test_row_number_expressions_empty(self) -> None:
        """RowNumber expressions() returns empty dict."""
        rn = RowNumber(order_by="ts")
        assert rn.expressions(["a"], {"a": exp.Column(this="a")}) == {}


# =============================================================================
# Static classification tests
# =============================================================================


class TestStaticClassification:
    """Verify all window transforms are classified as static."""

    def test_lag_classify(self) -> None:
        """Lag._classify() returns 'static'."""
        assert Lag(order_by="ts")._classify() == "static"

    def test_lead_classify(self) -> None:
        """Lead._classify() returns 'static'."""
        assert Lead(order_by="ts")._classify() == "static"

    def test_rolling_mean_classify(self) -> None:
        """RollingMean._classify() returns 'static'."""
        assert RollingMean(window=3, order_by="ts")._classify() == "static"

    def test_rolling_std_classify(self) -> None:
        """RollingStd._classify() returns 'static'."""
        assert RollingStd(window=3, order_by="ts")._classify() == "static"

    def test_rank_classify(self) -> None:
        """Rank._classify() returns 'static'."""
        assert Rank(order_by="score")._classify() == "static"

    def test_row_number_classify(self) -> None:
        """RowNumber._classify() returns 'static'."""
        assert RowNumber(order_by="ts")._classify() == "static"

    def test_lag_discover_empty(self) -> None:
        """Lag discover() returns empty dict."""
        lag = Lag(order_by="ts")
        schema = Schema({"ts": "INTEGER", "value": "DOUBLE"})
        assert lag.discover(["value"], schema) == {}

    def test_lead_discover_empty(self) -> None:
        """Lead discover() returns empty dict."""
        lead = Lead(order_by="ts")
        schema = Schema({"ts": "INTEGER", "value": "DOUBLE"})
        assert lead.discover(["value"], schema) == {}


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Edge cases for window transforms."""

    def test_single_row_lag(self) -> None:
        """Lag on single-row table produces NaN."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT 1 AS ts, 42.0 AS value")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lag(order_by="ts", columns=["value"])], backend=backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        lag_idx = names.index("value_lag1")
        assert np.isnan(result[0, lag_idx])

    def test_single_row_row_number(self) -> None:
        """RowNumber on single-row table returns 1."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT 1 AS ts, 42.0 AS value")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RowNumber(order_by="ts")], backend=backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rn_idx = names.index("row_number")
        assert result[0, rn_idx] == 1.0

    def test_single_row_rolling_mean(self) -> None:
        """RollingMean on single-row table returns the value itself."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT 1 AS ts, 42.0 AS value")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [RollingMean(window=3, order_by="ts", columns=["value"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        rmean_idx = names.index("value_rmean3")
        assert result[0, rmean_idx] == pytest.approx(42.0)

    def test_many_columns_lag(self) -> None:
        """Lag works with many target columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT i AS ts, i AS a, i*2 AS b, i*3 AS c, "
            "i*4 AS d, i*5 AS e FROM range(10) tbl(i)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Lag(order_by="ts", columns=["a", "b", "c", "d", "e"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        for col in ["a", "b", "c", "d", "e"]:
            assert f"{col}_lag1" in names
        assert result.shape[1] == 11  # 6 original + 5 lag

    def test_multiple_order_by_columns(self) -> None:
        """Multiple order_by columns work correctly."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE t AS SELECT * FROM VALUES
                (1, 'A', 10.0),
                (1, 'B', 20.0),
                (2, 'A', 30.0),
                (2, 'B', 40.0)
            t(ts, cat, value)
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Lag(order_by=["ts", "cat"], columns=["value"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape[0] == 4
        names = pipe.get_feature_names_out()
        assert "value_lag1" in names


# =============================================================================
# Import tests
# =============================================================================


class TestImports:
    """Verify window transforms are importable from sqlearn."""

    def test_import_from_features(self) -> None:
        """All transforms importable from sqlearn.features."""
        from sqlearn.features import (  # noqa: F401
            Lag,
            Lead,
            Rank,
            RollingMean,
            RollingStd,
            RowNumber,
        )

    def test_import_from_sqlearn(self) -> None:
        """All transforms importable from top-level sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "Lag")
        assert hasattr(sq, "Lead")
        assert hasattr(sq, "Rank")
        assert hasattr(sq, "RollingMean")
        assert hasattr(sq, "RollingStd")
        assert hasattr(sq, "RowNumber")
