"""Tests for sqlearn.core.columns."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.columns import Columns
from sqlearn.core.errors import InvalidStepError, NotFittedError, SchemaError
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema, categorical, matching, numeric
from sqlearn.core.transformer import Transformer

# --- Mock transformers ---


class _StaticDoubler(Transformer):
    """Static transformer that doubles values."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Double each column."""
        return {c: exp.Mul(this=exprs[c], expression=exp.Literal.number(2)) for c in columns}


_StaticDoubler.__module__ = "sqlearn.scalers.fake"


class _DynamicMeanCenter(Transformer):
    """Dynamic transformer that subtracts mean."""

    _classification = "dynamic"
    _default_columns = "numeric"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Request AVG for each column."""
        _ = schema, y_column
        return {f"{c}__mean": exp.Avg(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Subtract mean."""
        result: dict[str, exp.Expression] = {}
        for c in columns:
            mean = self.params_.get(f"{c}__mean", 0) if self.params_ else 0
            result[c] = exp.Sub(this=exprs[c], expression=exp.Literal.number(mean))
        return result


_DynamicMeanCenter.__module__ = "sqlearn.scalers.fake"


# --- Fixtures ---


@pytest.fixture
def backend(tmp_path: Any) -> DuckDBBackend:
    """Create a DuckDB backend with test data (numeric + categorical)."""
    import duckdb

    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        "CREATE TABLE data AS SELECT "
        "1.0 AS price, 10.0 AS quantity, 'London' AS city UNION ALL SELECT "
        "2.0, 20.0, 'Paris' UNION ALL SELECT "
        "3.0, 30.0, 'London' UNION ALL SELECT "
        "4.0, 40.0, 'Tokyo' UNION ALL SELECT "
        "5.0, 50.0, 'Paris'"
    )
    conn.close()
    return DuckDBBackend(db_path)


# ── Constructor validation ────────────────────────────────────────


class TestColumnsConstructor:
    """Test Columns constructor validation."""

    def test_empty_transformers_raises(self) -> None:
        """Empty transformers list raises InvalidStepError."""
        with pytest.raises(InvalidStepError, match="at least one"):
            Columns([])

    def test_non_transformer_raises(self) -> None:
        """Non-Transformer in group raises InvalidStepError."""
        with pytest.raises(InvalidStepError, match="not a Transformer"):
            Columns([("bad", "not_a_transformer", ["x"])])  # type: ignore[list-item]

    def test_duplicate_group_names_raises(self) -> None:
        """Duplicate group names raise InvalidStepError."""
        with pytest.raises(InvalidStepError, match="Duplicate group name"):
            Columns(
                [
                    ("same", _StaticDoubler(), ["x"]),
                    ("same", _DynamicMeanCenter(), ["y"]),
                ]
            )

    def test_invalid_remainder_raises(self) -> None:
        """Invalid remainder value raises ValueError."""
        with pytest.raises(ValueError, match="remainder must be"):
            Columns(
                [("a", _StaticDoubler(), ["x"])],
                remainder="invalid",
            )

    def test_wrong_tuple_length_raises(self) -> None:
        """Tuple with wrong number of elements raises InvalidStepError."""
        with pytest.raises(InvalidStepError, match="triple"):
            Columns([("name", _StaticDoubler())])  # type: ignore[list-item]

    def test_valid_construction(self) -> None:
        """Valid construction succeeds with correct group count."""
        cols = Columns(
            [
                ("scale", _StaticDoubler(), ["x"]),
                ("center", _DynamicMeanCenter(), ["y"]),
            ]
        )
        assert len(cols.transformers) == 2


# ── Basic usage ───────────────────────────────────────────────────


class TestColumnsBasicUsage:
    """Test basic Columns usage with real transformers."""

    def test_scale_numeric_encode_categorical(self, backend: DuckDBBackend) -> None:
        """Scale numeric columns and encode categorical columns."""
        from sqlearn.encoders.onehot import OneHotEncoder
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), numeric()),
                ("encode", OneHotEncoder(), categorical()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        assert result.shape[0] == 5
        assert pipe.is_fitted is True

        names = pipe.get_feature_names_out()
        # Numeric columns should still be present (scaled)
        assert "price" in names
        assert "quantity" in names
        # Categorical should be expanded
        assert any("city_" in n for n in names)

    def test_single_group(self, backend: DuckDBBackend) -> None:
        """Single group with remainder=drop drops unmatched columns."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), ["price"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price" in names
        # Other columns should be dropped
        assert "city" not in names
        assert "quantity" not in names


# ── Remainder behavior ────────────────────────────────────────────


class TestColumnsRemainder:
    """Test remainder parameter behavior."""

    def test_remainder_drop(self, backend: DuckDBBackend) -> None:
        """remainder='drop' excludes unmatched columns from output."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [("scale", StandardScaler(), ["price"])],
            remainder="drop",
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "quantity" not in names
        assert "city" not in names

    def test_remainder_passthrough(self, backend: DuckDBBackend) -> None:
        """remainder='passthrough' includes unmatched columns unchanged."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [("scale", StandardScaler(), ["price"])],
            remainder="passthrough",
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "quantity" in names
        assert "city" in names


# ── Column selectors ──────────────────────────────────────────────


class TestColumnsSelectors:
    """Test various column selector types."""

    def test_string_list_selector(self, backend: DuckDBBackend) -> None:
        """Explicit column name list works correctly."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), ["price", "quantity"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # Only price and quantity should be in the output
        names = pipe.get_feature_names_out()
        assert set(names) == {"price", "quantity"}
        assert result.shape == (5, 2)

    def test_numeric_selector(self, backend: DuckDBBackend) -> None:
        """sq.numeric() selector works correctly."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "quantity" in names
        assert "city" not in names

    def test_categorical_selector(self, backend: DuckDBBackend) -> None:
        """sq.categorical() selector works correctly."""
        from sqlearn.encoders.onehot import OneHotEncoder

        cols = Columns(
            [
                ("encode", OneHotEncoder(), categorical()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert any("city_" in n for n in names)
        # Original numeric columns should be dropped
        assert "price" not in names

    def test_pattern_selector(self, tmp_path: Any) -> None:
        """sq.matching() pattern selector works correctly."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS feat_a, 2.0 AS feat_b, 3.0 AS other_col")
        conn.close()

        be = DuckDBBackend(db_path)
        cols = Columns(
            [
                ("scale", _StaticDoubler(), matching("feat_*")),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=be)

        names = pipe.get_feature_names_out()
        assert "feat_a" in names
        assert "feat_b" in names
        assert "other_col" not in names


# ── Overlapping columns ──────────────────────────────────────────


class TestColumnsOverlap:
    """Test overlapping column group detection."""

    def test_overlapping_groups_raises(self) -> None:
        """Overlapping column groups raise SchemaError during resolve."""
        cols = Columns(
            [
                ("a", _StaticDoubler(), ["price", "quantity"]),
                ("b", _DynamicMeanCenter(), ["price"]),  # overlaps!
            ]
        )

        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE"})

        with pytest.raises(SchemaError, match="already assigned"):
            cols._resolve_groups(schema)


# ── Output schema ─────────────────────────────────────────────────


class TestColumnsOutputSchema:
    """Test output_schema correctness."""

    def test_output_schema_drop_remainder(self) -> None:
        """Output schema with remainder=drop only contains group outputs."""
        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE", "city": "VARCHAR"})
        cols = Columns(
            [
                ("scale", _StaticDoubler(), ["price"]),
            ]
        )

        out = cols.output_schema(schema)
        assert "price" in out.columns
        assert "quantity" not in out.columns
        assert "city" not in out.columns

    def test_output_schema_passthrough_remainder(self) -> None:
        """Output schema with remainder=passthrough includes unmatched columns."""
        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE", "city": "VARCHAR"})
        cols = Columns(
            [("scale", _StaticDoubler(), ["price"])],
            remainder="passthrough",
        )

        out = cols.output_schema(schema)
        assert "price" in out.columns
        assert "quantity" in out.columns
        assert "city" in out.columns

    def test_output_schema_encoder_replaces_columns(self, backend: DuckDBBackend) -> None:
        """OneHotEncoder in a group removes original and adds binary columns post-fit."""
        from sqlearn.encoders.onehot import OneHotEncoder

        cols = Columns(
            [
                ("encode", OneHotEncoder(), ["city"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        # Post-fit: original column dropped, binary columns added
        names = pipe.get_feature_names_out()
        assert "city" not in names
        assert any("city_" in n for n in names)


# ── Pipeline integration ──────────────────────────────────────────


class TestColumnsPipelineIntegration:
    """Test Columns as a step in Pipeline."""

    def test_pipeline_fit_transform(self, backend: DuckDBBackend) -> None:
        """Columns works as a Pipeline step with fit_transform()."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        result = pipe.fit_transform("data", backend=backend)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    def test_composition_imputer_then_columns(self, backend: DuckDBBackend) -> None:
        """Imputer before Columns composes correctly."""
        from sqlearn.imputers.imputer import Imputer
        from sqlearn.scalers.standard import StandardScaler

        pipe = Pipeline(
            [
                Imputer(),
                Columns(
                    [
                        ("scale", StandardScaler(), numeric()),
                    ]
                ),
            ]
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        assert result.shape[0] == 5
        names = pipe.get_feature_names_out()
        assert "price" in names

    def test_to_sql_combined_output(self, backend: DuckDBBackend) -> None:
        """to_sql() generates correct combined SQL."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql()

        assert isinstance(sql, str)
        assert "SELECT" in sql
        assert "price" in sql
        assert "quantity" in sql

    def test_to_sql_custom_table(self, backend: DuckDBBackend) -> None:
        """to_sql() respects custom table parameter."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql(table="my_table")
        assert "my_table" in sql

    def test_multi_group_transform(self, backend: DuckDBBackend) -> None:
        """Multiple groups produce correct transform output."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), ["price"]),
                ("double", _StaticDoubler(), ["quantity"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "quantity" in names
        assert result.shape == (5, 2)


# ── Clone and pickle ──────────────────────────────────────────────


class TestColumnsClonePickle:
    """Test clone and pickle roundtrip."""

    def test_clone_unfitted(self) -> None:
        """Clone unfitted Columns creates independent copy."""
        cols = Columns(
            [
                ("a", _StaticDoubler(), ["x"]),
            ]
        )
        cloned = cols.clone()

        assert cloned is not cols
        assert cloned.remainder == cols.remainder
        assert len(cloned.transformers) == len(cols.transformers)
        assert cloned.transformers[0][1] is not cols.transformers[0][1]

    def test_clone_fitted(self, backend: DuckDBBackend) -> None:
        """Clone fitted Columns preserves fitted state."""
        cols = Columns(
            [
                ("double", _StaticDoubler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        cloned = cols.clone()
        assert cloned._fitted == cols._fitted
        assert cloned.output_schema_ == cols.output_schema_

    def test_pickle_roundtrip(self, backend: DuckDBBackend) -> None:
        """Pickle roundtrip preserves state."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), numeric()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)

        pickled = pickle.dumps(cols)
        restored = pickle.loads(pickled)  # noqa: S301

        assert restored._fitted == cols._fitted
        assert restored.remainder == cols.remainder

    def test_deepcopy(self) -> None:
        """copy.deepcopy works on Columns."""
        cols = Columns(
            [
                ("a", _StaticDoubler(), ["x"]),
            ]
        )
        copied = copy.deepcopy(cols)

        assert copied is not cols
        assert copied.transformers[0][1] is not cols.transformers[0][1]


# ── Not-fitted guard ──────────────────────────────────────────────


class TestColumnsNotFitted:
    """Test not-fitted guards."""

    def test_get_feature_names_out_before_fit_raises(self) -> None:
        """get_feature_names_out() before fit raises NotFittedError."""
        cols = Columns(
            [
                ("a", _StaticDoubler(), ["x"]),
            ]
        )
        with pytest.raises(NotFittedError, match="not fitted"):
            cols.get_feature_names_out()


# ── Repr ──────────────────────────────────────────────────────────


class TestColumnsRepr:
    """Test Columns __repr__."""

    def test_repr_default_remainder(self) -> None:
        """Repr with default remainder omits the parameter."""
        cols = Columns(
            [
                ("scale", _StaticDoubler(), ["x"]),
                ("center", _DynamicMeanCenter(), ["y"]),
            ]
        )
        r = repr(cols)
        assert r == "Columns(scale=_StaticDoubler, center=_DynamicMeanCenter)"
        assert "remainder" not in r

    def test_repr_passthrough_remainder(self) -> None:
        """Repr with passthrough remainder includes the parameter."""
        cols = Columns(
            [("scale", _StaticDoubler(), ["x"])],
            remainder="passthrough",
        )
        r = repr(cols)
        assert "remainder='passthrough'" in r


# ── get_params ────────────────────────────────────────────────────


class TestColumnsGetParams:
    """Test get_params for sklearn compatibility."""

    def test_get_params_shallow(self) -> None:
        """get_params(deep=False) returns top-level params only."""
        cols = Columns(
            [("a", _StaticDoubler(), ["x"])],
            remainder="passthrough",
        )
        params = cols.get_params(deep=False)
        assert "transformers" in params
        assert "remainder" in params
        assert params["remainder"] == "passthrough"

    def test_get_params_deep(self) -> None:
        """get_params(deep=True) includes sub-transformer params."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(with_mean=False), ["x"]),
            ]
        )
        params = cols.get_params(deep=True)
        assert "scale__with_mean" in params
        assert params["scale__with_mean"] is False


# ── Multiple groups targeting different types ─────────────────────


class TestColumnsMultipleGroups:
    """Test multiple groups with different column types."""

    def test_three_groups(self, tmp_path: Any) -> None:
        """Three groups targeting different column types."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE data AS SELECT 1.0 AS price, 2.0 AS score, 'a' AS city, 'x' AS color"
        )
        conn.close()

        be = DuckDBBackend(db_path)
        cols = Columns(
            [
                ("price_scale", _StaticDoubler(), ["price"]),
                ("score_scale", _DynamicMeanCenter(), ["score"]),
                ("encode_city", _StaticDoubler(), ["city"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=be)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "score" in names
        assert "city" in names
        # color should be dropped (remainder=drop)
        assert "color" not in names

    def test_all_columns_covered_no_remainder(self, tmp_path: Any) -> None:
        """All columns covered by groups, no remainder columns."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        be = DuckDBBackend(db_path)
        cols = Columns(
            [
                ("a", _StaticDoubler(), ["x"]),
                ("b", _StaticDoubler(), ["y"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=be)

        names = pipe.get_feature_names_out()
        assert set(names) == {"x", "y"}


# ── Edge cases ────────────────────────────────────────────────────


class TestColumnsEdgeCases:
    """Test edge cases."""

    def test_empty_column_group(self, tmp_path: Any) -> None:
        """Group with no matching columns is handled gracefully."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS price, 2.0 AS quantity")
        conn.close()

        be = DuckDBBackend(db_path)
        # categorical() will find no columns in an all-numeric schema
        cols = Columns(
            [
                ("scale", _StaticDoubler(), numeric()),
                ("encode", _StaticDoubler(), categorical()),
            ],
            remainder="passthrough",
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=be)

        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "quantity" in names

    def test_passthrough_with_all_columns_covered(self, tmp_path: Any) -> None:
        """remainder='passthrough' with all columns covered adds nothing extra."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        be = DuckDBBackend(db_path)
        cols = Columns(
            [("a", _StaticDoubler(), ["x", "y"])],
            remainder="passthrough",
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=be)

        names = pipe.get_feature_names_out()
        assert set(names) == {"x", "y"}


# ── StandardScaler + OneHotEncoder end-to-end ─────────────────────


class TestColumnsEndToEnd:
    """End-to-end integration tests with real transformers."""

    def test_standard_scaler_values(self, backend: DuckDBBackend) -> None:
        """StandardScaler through Columns produces correctly scaled values."""
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), ["price"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # Mean of [1,2,3,4,5] = 3.0, std = sqrt(2) ~ 1.414
        # Scaled values should have mean ~0
        assert abs(result.mean()) < 1e-10

    def test_onehot_through_columns(self, backend: DuckDBBackend) -> None:
        """OneHotEncoder through Columns produces binary columns."""
        from sqlearn.encoders.onehot import OneHotEncoder

        cols = Columns(
            [
                ("encode", OneHotEncoder(), ["city"]),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "city_london" in names
        assert "city_paris" in names
        assert "city_tokyo" in names
        # Each row should have exactly one 1 in the city columns
        city_idx = [i for i, n in enumerate(names) if n.startswith("city_")]
        for row in result:
            city_vals = [row[i] for i in city_idx]
            assert sum(city_vals) == 1.0

    def test_combined_scale_and_encode(self, backend: DuckDBBackend) -> None:
        """Combined StandardScaler + OneHotEncoder produces valid output."""
        from sqlearn.encoders.onehot import OneHotEncoder
        from sqlearn.scalers.standard import StandardScaler

        cols = Columns(
            [
                ("scale", StandardScaler(), numeric()),
                ("encode", OneHotEncoder(), categorical()),
            ]
        )
        pipe = Pipeline([cols])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        # All values should be numeric (float64)
        assert result.dtype == np.float64
        # Should have scaled numeric + one-hot encoded columns
        assert len(names) >= 5
