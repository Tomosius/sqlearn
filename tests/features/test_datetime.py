"""Tests for sqlearn.features.datetime -- DateParts, DateDiff, IsWeekend, Quarter."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.features.datetime import DateDiff, DateParts, IsWeekend, Quarter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a table containing timestamp data."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE t (id INTEGER, ts TIMESTAMP, created_at TIMESTAMP)
    """)
    conn.execute("""
        INSERT INTO t VALUES
            (1, '2024-03-15 14:30:00', '2024-01-01 00:00:00'),
            (2, '2024-12-25 00:00:00', '2024-06-15 12:30:00'),
            (3, '2024-07-04 18:45:30', '2024-03-20 08:15:00'),
            (4, '2024-01-01 00:00:00', '2023-12-31 23:59:59'),
            (5, '2024-09-21 09:00:00', '2024-09-21 09:00:00')
    """)
    return conn


@pytest.fixture
def ts_backend(ts_conn: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """DuckDB backend wrapping timestamp test data."""
    return DuckDBBackend(connection=ts_conn)


@pytest.fixture
def weekend_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with known weekend and weekday dates."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (id INTEGER, ts TIMESTAMP)")
    conn.execute("""
        INSERT INTO t VALUES
            (1, '2024-03-11 10:00:00'),
            (2, '2024-03-12 10:00:00'),
            (3, '2024-03-13 10:00:00'),
            (4, '2024-03-14 10:00:00'),
            (5, '2024-03-15 10:00:00'),
            (6, '2024-03-16 10:00:00'),
            (7, '2024-03-17 10:00:00')
    """)
    return conn


@pytest.fixture
def weekend_backend(weekend_conn: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """DuckDB backend wrapping weekend test data."""
    return DuckDBBackend(connection=weekend_conn)


# ============================================================================
# DateParts
# ============================================================================


class TestDatePartsConstructor:
    """Test DateParts constructor and validation."""

    def test_default_parts(self) -> None:
        """Default parts are year, month, day, dayofweek, hour."""
        dp = DateParts()
        assert dp.parts == ["year", "month", "day", "dayofweek", "hour"]

    def test_custom_parts(self) -> None:
        """Custom parts list is stored."""
        dp = DateParts(parts=["year", "quarter"])
        assert dp.parts == ["year", "quarter"]

    def test_empty_parts_raises(self) -> None:
        """Empty parts list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            DateParts(parts=[])

    def test_invalid_part_raises(self) -> None:
        """Invalid part name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date parts"):
            DateParts(parts=["year", "invalid_part"])

    def test_classification_is_static(self) -> None:
        """DateParts is a static transformer."""
        assert DateParts._classification == "static"

    def test_default_columns_is_temporal(self) -> None:
        """Default columns targets temporal types."""
        assert DateParts._default_columns == "temporal"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        dp = DateParts()
        assert dp._classify() == "static"

    def test_get_params(self) -> None:
        """get_params returns parts and columns."""
        dp = DateParts(parts=["year", "month"])
        params = dp.get_params()
        assert params["parts"] == ["year", "month"]
        assert params["columns"] is None

    def test_set_params(self) -> None:
        """set_params updates parts and returns self."""
        dp = DateParts()
        result = dp.set_params(parts=["hour", "minute"])
        assert result is dp
        assert dp.parts == ["hour", "minute"]

    def test_all_valid_parts_accepted(self) -> None:
        """All documented part names are accepted."""
        all_parts = [
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "minute",
            "second",
            "quarter",
            "week",
            "dayofyear",
        ]
        dp = DateParts(parts=all_parts)
        assert dp.parts == all_parts


class TestDatePartsExpressions:
    """Test DateParts.expressions() generates correct sqlglot EXTRACT ASTs."""

    def test_single_part_single_column(self) -> None:
        """Single part for single column produces one EXTRACT."""
        dp = DateParts(parts=["year"])
        exprs = {"ts": exp.Column(this="ts")}
        result = dp.expressions(["ts"], exprs)
        assert "ts_year" in result
        assert isinstance(result["ts_year"], exp.Extract)

    def test_multiple_parts_single_column(self) -> None:
        """Multiple parts produce multiple EXTRACT expressions."""
        dp = DateParts(parts=["year", "month", "day"])
        exprs = {"ts": exp.Column(this="ts")}
        result = dp.expressions(["ts"], exprs)
        assert len(result) == 3
        assert "ts_year" in result
        assert "ts_month" in result
        assert "ts_day" in result

    def test_single_part_multiple_columns(self) -> None:
        """Single part applied to multiple columns."""
        dp = DateParts(parts=["hour"])
        exprs = {
            "ts": exp.Column(this="ts"),
            "created_at": exp.Column(this="created_at"),
        }
        result = dp.expressions(["ts", "created_at"], exprs)
        assert "ts_hour" in result
        assert "created_at_hour" in result

    def test_uses_exprs_not_raw_column(self) -> None:
        """expressions() composes with prior transforms."""
        dp = DateParts(parts=["year"])
        # Simulate prior transform wrapping ts in CAST
        prior = exp.Cast(this=exp.Column(this="ts"), to=exp.DataType.build("TIMESTAMP"))
        exprs = {"ts": prior}
        result = dp.expressions(["ts"], exprs)
        # The inner expression should be the Cast, not a raw Column
        assert isinstance(result["ts_year"].args["expression"], exp.Cast)

    def test_extract_sql_output(self) -> None:
        """Generated SQL contains EXTRACT."""
        dp = DateParts(parts=["year"])
        exprs = {"ts": exp.Column(this="ts")}
        result = dp.expressions(["ts"], exprs)
        sql = result["ts_year"].sql(dialect="duckdb")
        assert "EXTRACT" in sql.upper()
        assert "YEAR" in sql.upper()

    def test_dayofweek_uses_dow(self) -> None:
        """DayOfWeek uses DOW/DAYOFWEEK in SQL."""
        dp = DateParts(parts=["dayofweek"])
        exprs = {"ts": exp.Column(this="ts")}
        result = dp.expressions(["ts"], exprs)
        sql = result["ts_dayofweek"].sql(dialect="duckdb")
        assert "DOW" in sql.upper() or "DAYOFWEEK" in sql.upper()

    def test_empty_columns_returns_empty(self) -> None:
        """Empty columns list returns empty dict."""
        dp = DateParts(parts=["year"])
        result = dp.expressions([], {"ts": exp.Column(this="ts")})
        assert result == {}


class TestDatePartsOutputSchema:
    """Test DateParts.output_schema() adds new columns."""

    def test_adds_part_columns(self) -> None:
        """Output schema includes new part columns as INTEGER."""
        dp = DateParts(parts=["year", "month"])
        schema = Schema({"id": "INTEGER", "ts": "TIMESTAMP"})
        result = dp.output_schema(schema)
        assert "ts_year" in result.columns
        assert "ts_month" in result.columns
        assert result["ts_year"] == "INTEGER"
        assert result["ts_month"] == "INTEGER"

    def test_preserves_original_columns(self) -> None:
        """Original columns are preserved in output schema."""
        dp = DateParts(parts=["year"])
        schema = Schema({"id": "INTEGER", "ts": "TIMESTAMP"})
        result = dp.output_schema(schema)
        assert "id" in result.columns
        assert "ts" in result.columns

    def test_multiple_temporal_columns(self) -> None:
        """Multiple temporal columns each get part columns."""
        dp = DateParts(parts=["year"])
        schema = Schema({"ts": "TIMESTAMP", "created": "TIMESTAMP"})
        result = dp.output_schema(schema)
        assert "ts_year" in result.columns
        assert "created_year" in result.columns

    def test_schema_immutable(self) -> None:
        """Original schema is not modified."""
        dp = DateParts(parts=["year"])
        schema = Schema({"ts": "TIMESTAMP"})
        _ = dp.output_schema(schema)
        assert "ts_year" not in schema.columns


class TestDatePartsPipeline:
    """Test DateParts integrated with Pipeline."""

    def test_fit_transform_extracts_year(self, ts_backend: DuckDBBackend) -> None:
        """Pipeline extracts year values correctly."""
        pipe = Pipeline([DateParts(parts=["year"], columns=["ts"])], backend=ts_backend)
        result = pipe.fit_transform("t")
        # All rows are from 2024
        year_col_idx = pipe.get_feature_names_out().index("ts_year")
        years = result[:, year_col_idx]
        assert all(y == 2024.0 for y in years)

    def test_fit_transform_extracts_month(self, ts_backend: DuckDBBackend) -> None:
        """Pipeline extracts month values correctly."""
        pipe = Pipeline([DateParts(parts=["month"], columns=["ts"])], backend=ts_backend)
        result = pipe.fit_transform("t")
        month_col_idx = pipe.get_feature_names_out().index("ts_month")
        months = result[:, month_col_idx].astype(int)
        # Row 1: March(3), Row 2: Dec(12), Row 3: Jul(7), Row 4: Jan(1), Row 5: Sep(9)
        expected = [3, 12, 7, 1, 9]
        np.testing.assert_array_equal(months, expected)

    def test_fit_transform_output_shape(self, ts_backend: DuckDBBackend) -> None:
        """Output shape includes new part columns."""
        pipe = Pipeline([DateParts(parts=["year", "month"], columns=["ts"])], backend=ts_backend)
        result = pipe.fit_transform("t")
        # Original: id, ts, created_at (3 cols) + 2 new = 5 cols
        assert result.shape == (5, 5)

    def test_feature_names_include_parts(self, ts_backend: DuckDBBackend) -> None:
        """Feature names include the new part column names."""
        pipe = Pipeline([DateParts(parts=["year", "month"], columns=["ts"])], backend=ts_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "ts_year" in names
        assert "ts_month" in names
        assert "id" in names

    def test_to_sql_contains_extract(self, ts_backend: DuckDBBackend) -> None:
        """Generated SQL contains EXTRACT expressions."""
        pipe = Pipeline([DateParts(parts=["year"], columns=["ts"])], backend=ts_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "EXTRACT" in sql
        assert "YEAR" in sql


class TestDatePartsClonePickle:
    """Clone and pickle for DateParts."""

    def test_clone_preserves_parts(self) -> None:
        """Cloned DateParts has same parts."""
        dp = DateParts(parts=["year", "quarter"])
        dp._fitted = True
        cloned = dp.clone()
        assert cloned.parts == ["year", "quarter"]
        assert cloned._fitted is True

    def test_pickle_roundtrip(self) -> None:
        """Pickle and unpickle preserves state."""
        dp = DateParts(parts=["hour", "minute"])
        dp._fitted = True
        data = pickle.dumps(dp)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.parts == ["hour", "minute"]
        assert restored._fitted is True


# ============================================================================
# DateDiff
# ============================================================================


class TestDateDiffConstructor:
    """Test DateDiff constructor and validation."""

    def test_reference_and_unit_stored(self) -> None:
        """Reference and unit are stored."""
        dd = DateDiff(reference="2020-01-01")
        assert dd.reference == "2020-01-01"
        assert dd.unit == "day"

    def test_custom_unit(self) -> None:
        """Custom unit is stored."""
        dd = DateDiff(reference="2020-01-01", unit="hour")
        assert dd.unit == "hour"

    def test_invalid_unit_raises(self) -> None:
        """Invalid unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid unit"):
            DateDiff(reference="2020-01-01", unit="invalid")

    def test_classification_is_static(self) -> None:
        """DateDiff is a static transformer."""
        assert DateDiff._classification == "static"

    def test_default_columns_is_temporal(self) -> None:
        """Default columns targets temporal types."""
        assert DateDiff._default_columns == "temporal"

    def test_get_params(self) -> None:
        """get_params returns reference, unit, columns."""
        dd = DateDiff(reference="2020-01-01", unit="month")
        params = dd.get_params()
        assert params["reference"] == "2020-01-01"
        assert params["unit"] == "month"

    def test_column_reference_detection(self) -> None:
        """Column reference is distinguished from ISO date."""
        dd_date = DateDiff(reference="2020-01-01")
        assert not dd_date._is_column_reference()

        dd_col = DateDiff(reference="start_date")
        assert dd_col._is_column_reference()

    def test_all_valid_units(self) -> None:
        """All valid units are accepted."""
        for unit in ("day", "hour", "month", "year", "minute", "second", "week"):
            dd = DateDiff(reference="2020-01-01", unit=unit)
            assert dd.unit == unit


class TestDateDiffExpressions:
    """Test DateDiff.expressions() generates correct sqlglot DateDiff ASTs."""

    def test_date_reference_produces_datediff(self) -> None:
        """ISO date reference produces DateDiff with string literal."""
        dd = DateDiff(reference="2020-01-01")
        exprs = {"ts": exp.Column(this="ts")}
        result = dd.expressions(["ts"], exprs)
        assert "ts" in result
        assert isinstance(result["ts"], exp.DateDiff)

    def test_column_reference_produces_datediff(self) -> None:
        """Column reference produces DateDiff with column expression."""
        dd = DateDiff(reference="start_date")
        exprs = {
            "ts": exp.Column(this="ts"),
            "start_date": exp.Column(this="start_date"),
        }
        result = dd.expressions(["ts"], exprs)
        assert isinstance(result["ts"], exp.DateDiff)

    def test_sql_output(self) -> None:
        """Generated SQL contains DATEDIFF."""
        dd = DateDiff(reference="2020-01-01")
        exprs = {"ts": exp.Column(this="ts")}
        result = dd.expressions(["ts"], exprs)
        sql = result["ts"].sql(dialect="duckdb")
        assert "DATEDIFF" in sql.upper() or "DATE_DIFF" in sql.upper()

    def test_empty_columns_returns_empty(self) -> None:
        """Empty columns list returns empty dict."""
        dd = DateDiff(reference="2020-01-01")
        result = dd.expressions([], {"ts": exp.Column(this="ts")})
        assert result == {}


class TestDateDiffOutputSchema:
    """Test DateDiff.output_schema() changes types to INTEGER."""

    def test_temporal_becomes_integer(self) -> None:
        """Temporal column becomes INTEGER in output schema."""
        dd = DateDiff(reference="2020-01-01")
        schema = Schema({"id": "INTEGER", "ts": "TIMESTAMP"})
        result = dd.output_schema(schema)
        assert result["ts"] == "INTEGER"
        assert result["id"] == "INTEGER"

    def test_non_temporal_unchanged(self) -> None:
        """Non-temporal columns keep their types."""
        dd = DateDiff(reference="2020-01-01")
        schema = Schema({"name": "VARCHAR", "ts": "TIMESTAMP"})
        result = dd.output_schema(schema)
        assert result["name"] == "VARCHAR"


class TestDateDiffPipeline:
    """Test DateDiff integrated with Pipeline."""

    def test_fit_transform_days_from_epoch(self) -> None:
        """DateDiff computes days from reference date."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2020-01-01 00:00:00'),
                ('2020-01-02 00:00:00'),
                ('2020-01-11 00:00:00')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateDiff(reference="2020-01-01", unit="day")], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.array([[0], [1], [10]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_fit_transform_negative_diff(self) -> None:
        """Dates before reference produce negative differences."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2019-12-31 00:00:00'),
                ('2020-01-01 00:00:00')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateDiff(reference="2020-01-01", unit="day")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] < 0  # Dec 31 is before Jan 1
        assert result[1, 0] == 0  # Same date

    def test_to_sql_contains_datediff(self) -> None:
        """Generated SQL contains DATEDIFF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-01-01 00:00:00')")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateDiff(reference="2020-01-01")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "DATEDIFF" in sql or "DATE_DIFF" in sql

    def test_column_reference_pipeline(self) -> None:
        """DateDiff with column reference computes difference between columns."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (start_ts TIMESTAMP, end_ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2024-01-01 00:00:00', '2024-01-06 00:00:00'),
                ('2024-01-01 00:00:00', '2024-01-01 00:00:00'),
                ('2024-01-01 00:00:00', '2024-02-01 00:00:00')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [DateDiff(reference="start_ts", unit="day", columns=["end_ts"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        end_ts_idx = pipe.get_feature_names_out().index("end_ts")
        diffs = result[:, end_ts_idx].astype(int)
        assert diffs[0] == 5
        assert diffs[1] == 0
        assert diffs[2] == 31


class TestDateDiffClonePickle:
    """Clone and pickle for DateDiff."""

    def test_clone_preserves_state(self) -> None:
        """Cloned DateDiff has same reference and unit."""
        dd = DateDiff(reference="2020-01-01", unit="hour")
        dd._fitted = True
        cloned = dd.clone()
        assert cloned.reference == "2020-01-01"
        assert cloned.unit == "hour"
        assert cloned._fitted is True

    def test_pickle_roundtrip(self) -> None:
        """Pickle and unpickle preserves state."""
        dd = DateDiff(reference="2020-01-01", unit="month")
        data = pickle.dumps(dd)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.reference == "2020-01-01"
        assert restored.unit == "month"


# ============================================================================
# IsWeekend
# ============================================================================


class TestIsWeekendConstructor:
    """Test IsWeekend constructor."""

    def test_classification_is_static(self) -> None:
        """IsWeekend is a static transformer."""
        assert IsWeekend._classification == "static"

    def test_default_columns_is_temporal(self) -> None:
        """Default columns targets temporal types."""
        assert IsWeekend._default_columns == "temporal"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        iw = IsWeekend()
        assert iw._classify() == "static"

    def test_get_params(self) -> None:
        """get_params returns columns."""
        iw = IsWeekend()
        params = iw.get_params()
        assert params["columns"] is None


class TestIsWeekendExpressions:
    """Test IsWeekend.expressions() generates correct CASE WHEN ASTs."""

    def test_produces_case_expression(self) -> None:
        """Expression is a Case node."""
        iw = IsWeekend()
        exprs = {"ts": exp.Column(this="ts")}
        result = iw.expressions(["ts"], exprs)
        assert "ts" in result
        assert isinstance(result["ts"], exp.Case)

    def test_sql_contains_case_when(self) -> None:
        """Generated SQL contains CASE WHEN and DOW/DAYOFWEEK."""
        iw = IsWeekend()
        exprs = {"ts": exp.Column(this="ts")}
        result = iw.expressions(["ts"], exprs)
        sql = result["ts"].sql(dialect="duckdb")
        assert "CASE" in sql.upper()
        assert "WHEN" in sql.upper()
        assert "DOW" in sql.upper() or "DAYOFWEEK" in sql.upper()

    def test_multiple_columns(self) -> None:
        """IsWeekend applies to multiple columns."""
        iw = IsWeekend()
        exprs = {
            "ts": exp.Column(this="ts"),
            "created": exp.Column(this="created"),
        }
        result = iw.expressions(["ts", "created"], exprs)
        assert len(result) == 2
        assert isinstance(result["ts"], exp.Case)
        assert isinstance(result["created"], exp.Case)

    def test_empty_columns_returns_empty(self) -> None:
        """Empty columns list returns empty dict."""
        iw = IsWeekend()
        result = iw.expressions([], {"ts": exp.Column(this="ts")})
        assert result == {}


class TestIsWeekendOutputSchema:
    """Test IsWeekend.output_schema() changes types to INTEGER."""

    def test_temporal_becomes_integer(self) -> None:
        """Temporal column becomes INTEGER in output schema."""
        iw = IsWeekend()
        schema = Schema({"ts": "TIMESTAMP", "name": "VARCHAR"})
        result = iw.output_schema(schema)
        assert result["ts"] == "INTEGER"
        assert result["name"] == "VARCHAR"


class TestIsWeekendPipeline:
    """Test IsWeekend integrated with Pipeline."""

    def test_weekday_values(self, weekend_backend: DuckDBBackend) -> None:
        """Weekdays (Mon-Fri) produce 0."""
        pipe = Pipeline([IsWeekend(columns=["ts"])], backend=weekend_backend)
        result = pipe.fit_transform("t")
        ts_idx = pipe.get_feature_names_out().index("ts")
        values = result[:, ts_idx].astype(int)
        # Mon-Fri should be 0
        assert values[0] == 0  # Monday
        assert values[1] == 0  # Tuesday
        assert values[2] == 0  # Wednesday
        assert values[3] == 0  # Thursday
        assert values[4] == 0  # Friday

    def test_weekend_values(self, weekend_backend: DuckDBBackend) -> None:
        """Saturday and Sunday produce 1."""
        pipe = Pipeline([IsWeekend(columns=["ts"])], backend=weekend_backend)
        result = pipe.fit_transform("t")
        ts_idx = pipe.get_feature_names_out().index("ts")
        values = result[:, ts_idx].astype(int)
        assert values[5] == 1  # Saturday
        assert values[6] == 1  # Sunday

    def test_output_is_binary(self, weekend_backend: DuckDBBackend) -> None:
        """Output contains only 0 and 1."""
        pipe = Pipeline([IsWeekend(columns=["ts"])], backend=weekend_backend)
        result = pipe.fit_transform("t")
        ts_idx = pipe.get_feature_names_out().index("ts")
        unique = set(result[:, ts_idx].astype(int))
        assert unique <= {0, 1}

    def test_to_sql_contains_case_when(self, weekend_backend: DuckDBBackend) -> None:
        """Generated SQL contains CASE WHEN."""
        pipe = Pipeline([IsWeekend(columns=["ts"])], backend=weekend_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "CASE" in sql
        assert "WHEN" in sql

    def test_feature_names_unchanged(self, weekend_backend: DuckDBBackend) -> None:
        """IsWeekend replaces in-place, feature names stay the same."""
        pipe = Pipeline([IsWeekend(columns=["ts"])], backend=weekend_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "ts" in names
        assert "id" in names


class TestIsWeekendClonePickle:
    """Clone and pickle for IsWeekend."""

    def test_clone_preserves_state(self) -> None:
        """Cloned IsWeekend has same fitted state."""
        iw = IsWeekend()
        iw._fitted = True
        cloned = iw.clone()
        assert cloned._fitted is True

    def test_pickle_roundtrip(self) -> None:
        """Pickle and unpickle preserves state."""
        iw = IsWeekend()
        data = pickle.dumps(iw)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored, IsWeekend)
        assert restored._fitted is False


# ============================================================================
# Quarter
# ============================================================================


class TestQuarterConstructor:
    """Test Quarter constructor."""

    def test_classification_is_static(self) -> None:
        """Quarter is a static transformer."""
        assert Quarter._classification == "static"

    def test_default_columns_is_temporal(self) -> None:
        """Default columns targets temporal types."""
        assert Quarter._default_columns == "temporal"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        q = Quarter()
        assert q._classify() == "static"

    def test_get_params(self) -> None:
        """get_params returns columns."""
        q = Quarter()
        params = q.get_params()
        assert params["columns"] is None


class TestQuarterExpressions:
    """Test Quarter.expressions() generates correct EXTRACT ASTs."""

    def test_produces_extract(self) -> None:
        """Expression is an Extract node with QUARTER."""
        q = Quarter()
        exprs = {"ts": exp.Column(this="ts")}
        result = q.expressions(["ts"], exprs)
        assert "ts" in result
        assert isinstance(result["ts"], exp.Extract)

    def test_sql_contains_quarter(self) -> None:
        """Generated SQL contains EXTRACT and QUARTER."""
        q = Quarter()
        exprs = {"ts": exp.Column(this="ts")}
        result = q.expressions(["ts"], exprs)
        sql = result["ts"].sql(dialect="duckdb")
        assert "EXTRACT" in sql.upper()
        assert "QUARTER" in sql.upper()

    def test_multiple_columns(self) -> None:
        """Quarter applies to multiple columns."""
        q = Quarter()
        exprs = {
            "ts": exp.Column(this="ts"),
            "created": exp.Column(this="created"),
        }
        result = q.expressions(["ts", "created"], exprs)
        assert len(result) == 2

    def test_empty_columns_returns_empty(self) -> None:
        """Empty columns list returns empty dict."""
        q = Quarter()
        result = q.expressions([], {"ts": exp.Column(this="ts")})
        assert result == {}


class TestQuarterOutputSchema:
    """Test Quarter.output_schema() changes types to INTEGER."""

    def test_temporal_becomes_integer(self) -> None:
        """Temporal column becomes INTEGER in output schema."""
        q = Quarter()
        schema = Schema({"ts": "TIMESTAMP", "id": "INTEGER"})
        result = q.output_schema(schema)
        assert result["ts"] == "INTEGER"
        assert result["id"] == "INTEGER"


class TestQuarterPipeline:
    """Test Quarter integrated with Pipeline."""

    def test_quarter_values(self) -> None:
        """Quarter extracts correct quarter numbers (1-4)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2024-01-15 00:00:00'),
                ('2024-04-15 00:00:00'),
                ('2024-07-15 00:00:00'),
                ('2024-10-15 00:00:00')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Quarter()], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.array([[1], [2], [3], [4]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_q1_boundary(self) -> None:
        """Q1 boundary: Jan 1 and Mar 31 are both Q1."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2024-01-01 00:00:00'),
                ('2024-03-31 23:59:59')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Quarter()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == 1.0
        assert result[1, 0] == 1.0

    def test_to_sql_contains_extract(self) -> None:
        """Generated SQL contains EXTRACT QUARTER."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-01-01 00:00:00')")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Quarter()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "EXTRACT" in sql
        assert "QUARTER" in sql


class TestQuarterClonePickle:
    """Clone and pickle for Quarter."""

    def test_clone_preserves_state(self) -> None:
        """Cloned Quarter has same fitted state."""
        q = Quarter()
        q._fitted = True
        cloned = q.clone()
        assert cloned._fitted is True

    def test_pickle_roundtrip(self) -> None:
        """Pickle and unpickle preserves state."""
        q = Quarter()
        data = pickle.dumps(q)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored, Quarter)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases across all datetime transformers."""

    def test_null_timestamp_dateparts(self) -> None:
        """DateParts handles NULL timestamps."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-01-01 00:00:00'), (NULL)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateParts(parts=["year"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_null_timestamp_isweekend(self) -> None:
        """IsWeekend handles NULL timestamps."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-03-16 00:00:00'), (NULL)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([IsWeekend()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_null_timestamp_quarter(self) -> None:
        """Quarter handles NULL timestamps."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-07-01 00:00:00'), (NULL)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Quarter()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_null_timestamp_datediff(self) -> None:
        """DateDiff handles NULL timestamps."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-01-01 00:00:00'), (NULL)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateDiff(reference="2020-01-01")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_midnight_boundary(self) -> None:
        """DateParts at midnight extracts hour=0."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-01-01 00:00:00')")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateParts(parts=["hour"])], backend=backend)
        result = pipe.fit_transform("t")
        hour_idx = pipe.get_feature_names_out().index("ts_hour")
        assert result[0, hour_idx] == 0.0

    def test_year_boundary(self) -> None:
        """DateParts correctly handles year boundary (Dec 31 -> Jan 1)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2023-12-31 23:59:59'),
                ('2024-01-01 00:00:00')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateParts(parts=["year", "month", "day"])], backend=backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        year_idx = names.index("ts_year")
        month_idx = names.index("ts_month")
        day_idx = names.index("ts_day")
        # First row: 2023, 12, 31
        assert result[0, year_idx] == 2023.0
        assert result[0, month_idx] == 12.0
        assert result[0, day_idx] == 31.0
        # Second row: 2024, 1, 1
        assert result[1, year_idx] == 2024.0
        assert result[1, month_idx] == 1.0
        assert result[1, day_idx] == 1.0

    def test_leap_year_feb29(self) -> None:
        """DateParts correctly extracts Feb 29 in leap year."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-02-29 12:00:00')")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([DateParts(parts=["month", "day"])], backend=backend)
        result = pipe.fit_transform("t")
        names = pipe.get_feature_names_out()
        month_idx = names.index("ts_month")
        day_idx = names.index("ts_day")
        assert result[0, month_idx] == 2.0
        assert result[0, day_idx] == 29.0

    def test_single_row(self) -> None:
        """All transformers work with a single row."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.execute("INSERT INTO t VALUES ('2024-06-15 12:30:00')")
        backend = DuckDBBackend(connection=conn)

        for transformer in [
            DateParts(parts=["year"]),
            DateDiff(reference="2020-01-01"),
            IsWeekend(),
            Quarter(),
        ]:
            pipe = Pipeline([transformer], backend=backend)
            result = pipe.fit_transform("t")
            assert result.shape[0] == 1

    def test_date_type_column(self) -> None:
        """Transformers work with DATE type (not just TIMESTAMP)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (d DATE)")
        conn.execute("""
            INSERT INTO t VALUES
                ('2024-01-15'),
                ('2024-07-04')
        """)
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Quarter()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == 1.0
        assert result[1, 0] == 3.0

    def test_repr_dateparts(self) -> None:
        """DateParts repr shows non-default params."""
        dp = DateParts(parts=["year", "month"])
        r = repr(dp)
        assert "DateParts" in r
        assert "year" in r

    def test_repr_datediff(self) -> None:
        """DateDiff repr shows reference and unit."""
        dd = DateDiff(reference="2020-01-01")
        r = repr(dd)
        assert "DateDiff" in r
        assert "2020-01-01" in r

    def test_repr_isweekend_default(self) -> None:
        """IsWeekend repr with defaults."""
        iw = IsWeekend()
        r = repr(iw)
        assert "IsWeekend" in r

    def test_repr_quarter_default(self) -> None:
        """Quarter repr with defaults."""
        q = Quarter()
        r = repr(q)
        assert "Quarter" in r


class TestImportFromSqlearn:
    """Test imports from sqlearn top-level."""

    def test_import_dateparts(self) -> None:
        """DateParts is importable from sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "DateParts")

    def test_import_datediff(self) -> None:
        """DateDiff is importable from sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "DateDiff")

    def test_import_isweekend(self) -> None:
        """IsWeekend is importable from sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "IsWeekend")

    def test_import_quarter(self) -> None:
        """Quarter is importable from sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "Quarter")
