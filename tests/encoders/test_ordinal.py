"""Tests for sqlearn.encoders.ordinal — OrdinalEncoder."""

from __future__ import annotations

import pickle
from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.encoders.ordinal import OrdinalEncoder

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test OrdinalEncoder constructor and parameter validation."""

    def test_defaults(self) -> None:
        """Default categories='auto', handle_unknown='error', columns=None."""
        enc = OrdinalEncoder()
        assert enc.categories == "auto"
        assert enc.handle_unknown == "error"
        assert enc.unknown_value is None
        assert enc.columns is None

    def test_default_columns_is_categorical(self) -> None:
        """Class default routes to categorical columns."""
        assert OrdinalEncoder._default_columns == "categorical"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn categories)."""
        assert OrdinalEncoder._classification == "dynamic"

    def test_custom_columns(self) -> None:
        """Explicit columns override."""
        enc = OrdinalEncoder(columns=["city", "color"])
        assert enc.columns == ["city", "color"]

    def test_handle_unknown_use_encoded_value(self) -> None:
        """handle_unknown='use_encoded_value' with unknown_value is valid."""
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        assert enc.handle_unknown == "use_encoded_value"
        assert enc.unknown_value == -1

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            columns=["city"],
        )
        params = enc.get_params()
        assert params == {
            "categories": "auto",
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
            "columns": ["city"],
        }

    def test_repr_default(self) -> None:
        """Default repr shows class name with no params."""
        enc = OrdinalEncoder()
        assert repr(enc) == "OrdinalEncoder()"

    def test_repr_custom(self) -> None:
        """Custom params appear in repr."""
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        r = repr(enc)
        assert "handle_unknown='use_encoded_value'" in r
        assert "unknown_value=-1" in r


# ── Constructor validation tests ─────────────────────────────────


class TestConstructorValidation:
    """Test OrdinalEncoder constructor parameter validation."""

    def test_use_encoded_value_requires_unknown_value(self) -> None:
        """handle_unknown='use_encoded_value' without unknown_value raises."""
        with pytest.raises(ValueError, match="unknown_value must be set"):
            OrdinalEncoder(handle_unknown="use_encoded_value")

    def test_error_forbids_unknown_value(self) -> None:
        """handle_unknown='error' with unknown_value raises."""
        with pytest.raises(ValueError, match="unknown_value must be None"):
            OrdinalEncoder(handle_unknown="error", unknown_value=-1)

    def test_invalid_handle_unknown(self) -> None:
        """Invalid handle_unknown value raises."""
        with pytest.raises(ValueError, match="Invalid handle_unknown"):
            OrdinalEncoder(handle_unknown="ignore")


# ── discover_sets() tests ─────────────────────────────────────────


class TestDiscoverSets:
    """Test OrdinalEncoder.discover_sets() returns correct set queries."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with categorical columns."""
        return Schema({"city": "VARCHAR", "color": "VARCHAR", "price": "DOUBLE"})

    def test_returns_select_distinct_per_column(self, schema: Schema) -> None:
        """Each column gets a SELECT DISTINCT query."""
        enc = OrdinalEncoder()
        result = enc.discover_sets(["city", "color"], schema)
        assert "city__categories" in result
        assert "color__categories" in result
        assert len(result) == 2

    def test_ast_is_select_with_distinct(self, schema: Schema) -> None:
        """Each query is a Select containing a Distinct node."""
        enc = OrdinalEncoder()
        result = enc.discover_sets(["city"], schema)
        query = result["city__categories"]
        assert isinstance(query, exp.Select)
        distinct_nodes = list(query.find_all(exp.Distinct))
        assert len(distinct_nodes) > 0

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        enc = OrdinalEncoder()
        result = enc.discover_sets([], schema)
        assert result == {}


# ── _get_categories() tests ───────────────────────────────────────


class TestGetCategories:
    """Test OrdinalEncoder._get_categories() helper."""

    def _make_fitted_encoder(
        self,
        sets: dict[str, list[dict[str, Any]]],
    ) -> OrdinalEncoder:
        """Create a fitted OrdinalEncoder with given sets."""
        enc = OrdinalEncoder()
        enc.sets_ = sets
        enc._fitted = True
        return enc

    def test_sorted_alphabetically(self) -> None:
        """Categories are sorted alphabetically."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "Paris"}, {"city": "London"}, {"city": "Berlin"}]}
        )
        assert enc._get_categories("city") == ["Berlin", "London", "Paris"]

    def test_nulls_excluded(self) -> None:
        """None values are excluded from categories."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "Paris"}, {"city": None}, {"city": "London"}]}
        )
        cats = enc._get_categories("city")
        assert None not in cats
        assert len(cats) == 2

    def test_all_categories_included(self) -> None:
        """All non-null categories are returned (no truncation like OneHotEncoder)."""
        rows = [{"letter": chr(65 + i)} for i in range(26)]  # A through Z
        enc = self._make_fitted_encoder({"letter__categories": rows})
        cats = enc._get_categories("letter")
        assert len(cats) == 26


# ── expressions() tests ──────────────────────────────────────────


class TestExpressions:
    """Test OrdinalEncoder.expressions() generates correct CASE WHEN ASTs."""

    def _make_fitted_encoder(
        self,
        sets: dict[str, list[dict[str, Any]]],
        handle_unknown: str = "error",
        unknown_value: int | None = None,
    ) -> OrdinalEncoder:
        """Create a fitted OrdinalEncoder with given sets."""
        enc = OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value)
        enc.sets_ = sets
        enc._fitted = True
        enc.columns_ = list({k.replace("__categories", "") for k in sets})
        return enc

    def test_case_when_replaces_column(self) -> None:
        """CASE WHEN expression replaces the original column (same key)."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert "city" in result
        assert isinstance(result["city"], exp.Case)

    def test_ordinal_values_match_sorted_order(self) -> None:
        """Categories get indices matching alphabetical sort order."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "Paris"}, {"city": "London"}, {"city": "Berlin"}]}
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        ifs = case_expr.args["ifs"]
        # Berlin=0, London=1, Paris=2 (alphabetical)
        assert len(ifs) == 3
        # Check the literal values correspond to sorted indices
        literals = [int(if_expr.args["true"].this) for if_expr in ifs]
        assert literals == [0, 1, 2]

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own CASE expression."""
        enc = self._make_fitted_encoder(
            {
                "city__categories": [{"city": "London"}, {"city": "Paris"}],
                "color__categories": [{"color": "Red"}, {"color": "Blue"}],
            }
        )
        exprs = {"city": exp.Column(this="city"), "color": exp.Column(this="color")}
        result = enc.expressions(["city", "color"], exprs)
        assert "city" in result
        assert "color" in result
        assert isinstance(result["city"], exp.Case)
        assert isinstance(result["color"], exp.Case)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        enc = self._make_fitted_encoder({"city__categories": [{"city": "London"}]})
        # Simulate a prior transform (e.g., COALESCE from Imputer)
        prior = exp.Upper(this=exp.Column(this="city"))
        exprs = {"city": prior}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        if_expr = case_expr.args["ifs"][0]
        eq_expr = if_expr.this
        assert isinstance(eq_expr, exp.EQ)
        assert isinstance(eq_expr.this, exp.Upper)

    def test_default_is_null_for_error_mode(self) -> None:
        """handle_unknown='error': default is NULL sentinel."""
        enc = self._make_fitted_encoder({"city__categories": [{"city": "London"}]})
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        assert isinstance(case_expr.args["default"], exp.Null)

    def test_default_is_encoded_value(self) -> None:
        """handle_unknown='use_encoded_value': default is unknown_value."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "London"}]},
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        default = case_expr.args["default"]
        # sqlglot represents -1 as Neg(Literal(1))
        assert isinstance(default, exp.Neg)
        assert isinstance(default.this, exp.Literal)
        assert default.this.this == "1"

    def test_no_new_columns_added(self) -> None:
        """OrdinalEncoder does not add new columns (replaces in-place)."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "A"}, {"city": "B"}, {"city": "C"}]}
        )
        exprs = {"city": exp.Column(this="city"), "price": exp.Column(this="price")}
        result = enc.expressions(["city"], exprs)
        # Only city is in the result, no city_a, city_b, etc.
        assert set(result.keys()) == {"city"}


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test OrdinalEncoder integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with categorical test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), "
            "(3.0, 'London'), (4.0, 'Tokyo') t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape: 4 rows x 2 cols (price + city as integer)."""
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 2)

    def test_ordinal_encoding_correct(self, backend: DuckDBBackend) -> None:
        """Ordinal encoding: London=0, Paris=1, Tokyo=2 (alphabetical)."""
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # Column 1 is city (ordinal encoded)
        # Row 0: London -> 0
        assert result[0, 1] == 0
        # Row 1: Paris -> 1
        assert result[1, 1] == 1
        # Row 2: London -> 0
        assert result[2, 1] == 0
        # Row 3: Tokyo -> 2
        assert result[3, 1] == 2

    def test_to_sql_contains_case_when(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains CASE WHEN."""
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "CASE" in sql.upper()
        assert "WHEN" in sql.upper()

    def test_get_feature_names_out(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out returns original column names (no expansion)."""
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "city" in names
        assert len(names) == 2

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([OrdinalEncoder()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([OrdinalEncoder()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── sklearn equivalence tests ─────────────────────────────────────


class TestSklearnEquivalence:
    """Verify OrdinalEncoder produces same results as sklearn."""

    def test_matches_sklearn_ordinal_encoder(self) -> None:
        """Results match sklearn.preprocessing.OrdinalEncoder."""
        sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

        # Create test data
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('London',), ('Paris',), ('Berlin',), ('London',) t(city)"
        )
        backend = DuckDBBackend(connection=conn)

        # sqlearn
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        sq_result = pipe.fit_transform("t")

        # sklearn
        sk_enc = sklearn_preprocessing.OrdinalEncoder()
        sk_data = np.array([["London"], ["Paris"], ["Berlin"], ["London"]])
        sk_result = sk_enc.fit_transform(sk_data)

        np.testing.assert_allclose(sq_result, sk_result)

    def test_matches_sklearn_multiple_columns(self) -> None:
        """Multi-column results match sklearn."""
        sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('A', 'X'), ('B', 'Y'), ('C', 'Z'), ('A', 'X') t(col1, col2)"
        )
        backend = DuckDBBackend(connection=conn)

        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        sq_result = pipe.fit_transform("t")

        sk_enc = sklearn_preprocessing.OrdinalEncoder()
        sk_data = np.array([["A", "X"], ["B", "Y"], ["C", "Z"], ["A", "X"]])
        sk_result = sk_enc.fit_transform(sk_data)

        np.testing.assert_allclose(sq_result, sk_result)


# ── Clone and pickle tests ──────────────────────────────────────


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_sets_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned step sets does not affect original."""
        cloned = fitted_pipe.clone()
        original_enc = fitted_pipe.steps[0][1]
        cloned_enc = cloned.steps[0][1]
        assert cloned_enc.sets_ is not None
        cloned_enc.sets_["city__categories"] = []
        assert original_enc.sets_ is not None
        assert len(original_enc.sets_["city__categories"]) == 3

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual encoder preserves sets."""
        enc = OrdinalEncoder()
        enc.sets_ = {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        enc._fitted = True
        enc.columns_ = ["city"]
        enc.input_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        enc.output_schema_ = enc.output_schema(enc.input_schema_)
        data = pickle.dumps(enc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.sets_ == enc.sets_
        assert restored._fitted is True
        assert restored.handle_unknown == "error"


# ── Not-fitted guard tests ───────────────────────────────────────


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OrdinalEncoder()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OrdinalEncoder()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_category(self) -> None:
        """Column with one unique category produces code 0 for all rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('A'), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [0, 0, 0])

    def test_all_same_value(self) -> None:
        """Column where every row has the same value encodes to 0."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'X'), (2.0, 'X'), (3.0, 'X') t(price, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)
        # All cat values should be 0
        np.testing.assert_array_equal(result[:, 1], [0, 0, 0])

    def test_null_only_category_column(self) -> None:
        """Column with all NULLs produces all-NULL output."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, CAST(NULL AS VARCHAR)), "
            "(2.0, CAST(NULL AS VARCHAR)) t(price, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # Shape preserved: price + cat
        assert result.shape == (2, 2)

    def test_nulls_in_data(self) -> None:
        """NULL values get the ELSE branch (NULL for error mode)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        # A=0, B=1
        assert result[0, 0] == 0
        assert result[1, 0] == 1
        assert result[3, 0] == 0

    def test_numeric_passthrough(self) -> None:
        """Numeric columns are not encoded (pass through unchanged)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 100, 'A'), (2.0, 200, 'B') t(price, qty, color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # price, qty pass through + color encoded = still 3 columns
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 0].astype(float), [1.0, 2.0])
        np.testing.assert_allclose(result[:, 1].astype(float), [100.0, 200.0])

    def test_many_categories(self) -> None:
        """Column with many unique categories gets correct ordinals."""
        conn = duckdb.connect()
        values = ", ".join([f"('{chr(65 + i)}')" for i in range(26)])
        conn.execute(f"CREATE TABLE t AS SELECT * FROM VALUES {values} t(letter)")  # noqa: S608
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (26, 1)
        # A=0, B=1, ..., Z=25
        expected = np.arange(26, dtype=float)
        np.testing.assert_array_equal(result.flatten(), expected)


# ── handle_unknown='use_encoded_value' tests ─────────────────────


class TestHandleUnknown:
    """Test handle_unknown='use_encoded_value' behavior."""

    def test_unknown_value_in_sql(self) -> None:
        """SQL contains the unknown_value as ELSE branch."""
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.sets_ = {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        enc._fitted = True
        enc.columns_ = ["city"]

        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        default = case_expr.args["default"]
        # sqlglot represents -1 as Neg(Literal(1))
        assert isinstance(default, exp.Neg)
        assert isinstance(default.this, exp.Literal)
        assert default.this.this == "1"

    def test_unknown_value_zero(self) -> None:
        """unknown_value=0 is valid (edge case)."""
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0)
        enc.sets_ = {"city__categories": [{"city": "A"}]}
        enc._fitted = True
        enc.columns_ = ["city"]

        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        default = result["city"].args["default"]
        assert isinstance(default, exp.Literal)
        assert default.this == "0"


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """OrdinalEncoder composing with other transformers."""

    def test_imputer_then_encoder(self) -> None:
        """Imputer + OrdinalEncoder: NULLs filled before encoding."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OrdinalEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)

    def test_encoder_sql_composes_with_imputer(self) -> None:
        """SQL shows COALESCE nested inside CASE WHEN from composition."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL) t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OrdinalEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "CASE" in sql

    def test_full_pipeline_with_scaler(self) -> None:
        """Imputer + OrdinalEncoder + StandardScaler: full pipeline."""
        from sqlearn.imputers.imputer import Imputer
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'A'), (NULL, 'B'), (3.0, NULL), (4.0, 'A') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OrdinalEncoder(), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # num + cat = 2 columns (both now numeric after encoding)
        assert result.shape == (4, 2)


# ── SQL snapshot tests ──────────────────────────────────────────


class TestSqlSnapshot:
    """Verify SQL output structure and patterns."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create fitted pipeline for SQL verification."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OrdinalEncoder()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_single_case_per_column(self, fitted_pipe: Pipeline) -> None:
        """Each column gets exactly one CASE expression (not one per category)."""
        sql = fitted_pipe.to_sql().upper()
        # Unlike OneHotEncoder (3 CASE), OrdinalEncoder has 1 CASE per column
        assert sql.count("CASE") == 1

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_preserves_original_column_name(self, fitted_pipe: Pipeline) -> None:
        """Output uses original column name (no city_london etc.)."""
        sql = fitted_pipe.to_sql().lower()
        # Should NOT have onehot-style names
        assert "city_london" not in sql
        assert "city_paris" not in sql

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="raw_data")
        assert "raw_data" in sql
