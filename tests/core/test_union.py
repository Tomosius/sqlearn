"""Tests for sqlearn.core.union."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import InvalidStepError, NotFittedError
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.core.transformer import Transformer
from sqlearn.core.union import Union

# --- Mock transformers (reused across union tests) ---


class _StaticStep(Transformer):
    """Static transformer that doubles numeric values."""

    _classification = "static"
    _default_columns = "numeric"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Double each column."""
        return {c: exp.Mul(this=exprs[c], expression=exp.Literal.number(2)) for c in columns}


_StaticStep.__module__ = "sqlearn.scalers.fake"


class _DynamicStep(Transformer):
    """Dynamic transformer that subtracts mean."""

    _classification = "dynamic"
    _default_columns = "numeric"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Request AVG for each column."""
        return {f"{c}__mean": exp.Avg(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Subtract mean (requires params_)."""
        result: dict[str, exp.Expression] = {}
        for c in columns:
            mean = self.params_.get(f"{c}__mean", 0) if self.params_ else 0
            result[c] = exp.Sub(this=exprs[c], expression=exp.Literal.number(mean))
        return result


_DynamicStep.__module__ = "sqlearn.scalers.fake"


class _CategoricalStep(Transformer):
    """Static transformer that passes through categorical columns."""

    _classification = "static"
    _default_columns = "categorical"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Pass through as-is."""
        return {c: exprs[c] for c in columns}


_CategoricalStep.__module__ = "sqlearn.encoders.fake"


# ── Constructor tests ──────────────────────────────────────────────


class TestUnionConstructor:
    """Test Union constructor validation."""

    def test_valid_tuple_list(self) -> None:
        """Union accepts list of (name, transformer) tuples."""
        union = Union([("a", _StaticStep()), ("b", _DynamicStep())])
        assert len(union.transformers) == 2
        assert union.transformers[0][0] == "a"
        assert union.transformers[1][0] == "b"

    def test_single_branch(self) -> None:
        """Union accepts a single branch."""
        union = Union([("only", _StaticStep())])
        assert len(union.transformers) == 1

    def test_empty_list_raises(self) -> None:
        """Empty transformer list raises InvalidStepError."""
        with pytest.raises(InvalidStepError, match="at least one"):
            Union([])

    def test_non_transformer_raises(self) -> None:
        """Non-Transformer element raises InvalidStepError."""
        with pytest.raises(InvalidStepError, match="not a Transformer"):
            Union([("bad", 42)])  # type: ignore[list-item]

    def test_duplicate_names_raises(self) -> None:
        """Duplicate branch names raise InvalidStepError."""
        with pytest.raises(InvalidStepError, match="Duplicate branch name"):
            Union([("same", _StaticStep()), ("same", _DynamicStep())])

    def test_non_tuple_raises(self) -> None:
        """Non-tuple elements raise InvalidStepError."""
        with pytest.raises(InvalidStepError, match="tuples"):
            Union([_StaticStep()])  # type: ignore[list-item]

    def test_non_string_name_raises(self) -> None:
        """Non-string branch names raise InvalidStepError."""
        with pytest.raises(InvalidStepError, match="string"):
            Union([(42, _StaticStep())])  # type: ignore[list-item]


# ── Repr tests ─────────────────────────────────────────────────────


class TestUnionRepr:
    """Test Union display methods."""

    def test_repr_format(self) -> None:
        """Repr shows branch names and class names."""
        union = Union([("scale", _StaticStep()), ("center", _DynamicStep())])
        r = repr(union)
        assert r == "Union(scale=_StaticStep, center=_DynamicStep)"

    def test_repr_html(self) -> None:
        """HTML repr includes branch info and fitted status."""
        union = Union([("a", _StaticStep())])
        html = union._repr_html_()
        assert "Union" in html
        assert "not fitted" in html
        assert "_StaticStep" in html


# ── Classification tests ───────────────────────────────────────────


class TestUnionClassification:
    """Test Union classification logic."""

    def test_all_static_branches(self) -> None:
        """Union with all static branches classifies as static."""
        union = Union([("a", _StaticStep()), ("b", _StaticStep())])
        assert union._classify() == "static"

    def test_any_dynamic_branch(self) -> None:
        """Union with any dynamic branch classifies as dynamic."""
        union = Union([("a", _StaticStep()), ("b", _DynamicStep())])
        assert union._classify() == "dynamic"

    def test_all_dynamic_branches(self) -> None:
        """Union with all dynamic branches classifies as dynamic."""
        union = Union([("a", _DynamicStep()), ("b", _DynamicStep())])
        assert union._classify() == "dynamic"


# ── Discovery tests ────────────────────────────────────────────────


class TestUnionDiscovery:
    """Test Union discover and discover_sets methods."""

    def test_discover_merges_branches(self) -> None:
        """discover() merges aggregations from all branches with prefixes."""
        union = Union([("a", _DynamicStep()), ("b", _DynamicStep())])
        schema = Schema({"x": "DOUBLE", "y": "DOUBLE"})
        result = union.discover(["x", "y"], schema)
        # Both branches resolve to numeric columns and produce AVG
        assert "a__x__mean" in result
        assert "a__y__mean" in result
        assert "b__x__mean" in result
        assert "b__y__mean" in result

    def test_discover_static_branch_empty(self) -> None:
        """Static branches contribute no aggregations."""
        union = Union([("s", _StaticStep()), ("d", _DynamicStep())])
        schema = Schema({"x": "DOUBLE"})
        result = union.discover(["x"], schema)
        # Static branch contributes nothing, dynamic contributes AVG
        assert "d__x__mean" in result
        assert not any(k.startswith("s__") for k in result)

    def test_discover_sets_merges_branches(self) -> None:
        """discover_sets() merges set queries from all branches."""
        from sqlearn.encoders.onehot import OneHotEncoder

        enc = OneHotEncoder(columns=["city"])
        union = Union([("enc", enc)])
        schema = Schema({"city": "VARCHAR", "x": "DOUBLE"})
        result = union.discover_sets(["city", "x"], schema)
        assert "enc__city__categories" in result


# ── Expressions tests ──────────────────────────────────────────────


class TestUnionExpressions:
    """Test Union expression generation."""

    def test_expressions_prefixes_columns(self) -> None:
        """expressions() prefixes all output columns with branch name."""
        step = _StaticStep()
        step.columns_ = ["x"]
        step.input_schema_ = Schema({"x": "DOUBLE"})
        step.output_schema_ = step.output_schema(step.input_schema_)
        step._fitted = True

        union = Union([("dbl", step)])
        union.columns_ = ["x"]
        union.input_schema_ = Schema({"x": "DOUBLE"})
        union._fitted = True

        exprs = {"x": exp.Column(this="x")}
        result = union.expressions(["x"], exprs)
        assert "dbl_x" in result
        assert "x" not in result

    def test_expressions_multiple_branches(self) -> None:
        """expressions() combines outputs from multiple branches."""
        s1 = _StaticStep()
        s1.columns_ = ["x"]
        s1.input_schema_ = Schema({"x": "DOUBLE"})
        s1.output_schema_ = s1.output_schema(s1.input_schema_)
        s1._fitted = True

        s2 = _StaticStep()
        s2.columns_ = ["x"]
        s2.input_schema_ = Schema({"x": "DOUBLE"})
        s2.output_schema_ = s2.output_schema(s2.input_schema_)
        s2._fitted = True

        union = Union([("a", s1), ("b", s2)])
        union.columns_ = ["x"]
        union.input_schema_ = Schema({"x": "DOUBLE"})
        union._fitted = True

        exprs = {"x": exp.Column(this="x")}
        result = union.expressions(["x"], exprs)
        assert "a_x" in result
        assert "b_x" in result


# ── Output schema tests ───────────────────────────────────────────


class TestUnionOutputSchema:
    """Test Union output schema generation."""

    def test_output_schema_prefixed(self) -> None:
        """output_schema() returns prefixed column names."""
        union = Union([("a", _StaticStep()), ("b", _DynamicStep())])
        schema = Schema({"x": "DOUBLE", "y": "DOUBLE"})
        out = union.output_schema(schema)
        # Both branches default to numeric, both see x and y
        assert "a_x" in out.columns
        assert "a_y" in out.columns
        assert "b_x" in out.columns
        assert "b_y" in out.columns
        # Original columns are NOT in output
        assert "x" not in out.columns
        assert "y" not in out.columns

    def test_output_schema_mixed_branches(self) -> None:
        """output_schema() handles branches with different column sets."""
        union = Union(
            [
                ("num", _StaticStep()),
                ("cat", _CategoricalStep()),
            ]
        )
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        out = union.output_schema(schema)
        assert "num_price" in out.columns
        assert "cat_city" in out.columns


# ── Pipeline integration tests ─────────────────────────────────────


class TestUnionPipelineIntegration:
    """Test Union inside Pipeline (full fit/transform cycle)."""

    @pytest.fixture
    def backend(self, tmp_path: Any) -> DuckDBBackend:
        """Create a DuckDB backend with test data."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE train AS SELECT "
            "1.0 AS price, 10.0 AS qty UNION ALL SELECT "
            "2.0, 20.0 UNION ALL SELECT "
            "3.0, 30.0"
        )
        conn.close()
        return DuckDBBackend(db_path)

    def test_static_union_fit_transform(self, backend: DuckDBBackend) -> None:
        """Pipeline with Union of static branches fits and transforms."""
        union = Union([("a", _StaticStep()), ("b", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("train", backend=backend)
        result = pipe.transform("train", backend=backend)
        assert result.shape == (3, 4)  # 2 branches x 2 columns each
        names = pipe.get_feature_names_out()
        assert "a_price" in names
        assert "a_qty" in names
        assert "b_price" in names
        assert "b_qty" in names

    def test_dynamic_union_fit_transform(self, backend: DuckDBBackend) -> None:
        """Pipeline with Union of dynamic branches fits and transforms."""
        union = Union([("avg", _DynamicStep())])
        pipe = Pipeline([union])
        pipe.fit("train", backend=backend)
        assert pipe.is_fitted

        result = pipe.transform("train", backend=backend)
        assert result.shape[0] == 3
        names = pipe.get_feature_names_out()
        assert "avg_price" in names
        assert "avg_qty" in names

    def test_union_to_sql(self, backend: DuckDBBackend) -> None:
        """to_sql() generates valid SQL for Union pipeline."""
        union = Union([("dbl", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("train", backend=backend)
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql
        assert "dbl_price" in sql
        assert "dbl_qty" in sql

    def test_union_after_step(self, backend: DuckDBBackend) -> None:
        """Union works after another step in the pipeline."""
        pipe = Pipeline(
            [
                _StaticStep(),
                Union([("branch", _StaticStep())]),
            ]
        )
        pipe.fit("train", backend=backend)
        result = pipe.transform("train", backend=backend)
        assert result.shape[0] == 3
        names = pipe.get_feature_names_out()
        assert all(n.startswith("branch_") for n in names)


class TestUnionWithRealTransformers:
    """Test Union with real sqlearn transformers."""

    @pytest.fixture
    def backend(self, tmp_path: Any) -> DuckDBBackend:
        """Create a DuckDB backend with mixed-type data."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE data AS SELECT "
            "1.0 AS price, 'a' AS city UNION ALL SELECT "
            "2.0, 'b' UNION ALL SELECT "
            "3.0, 'a'"
        )
        conn.close()
        return DuckDBBackend(db_path)

    def test_standard_scaler_union(self, backend: DuckDBBackend) -> None:
        """Union with StandardScaler produces prefixed columns for all schema cols."""
        from sqlearn.scalers.standard import StandardScaler

        union = Union([("scaled", StandardScaler())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)
        # StandardScaler's output_schema returns full schema (numeric+categorical)
        # so branch output includes both price (scaled) and city (passthrough)
        assert result.shape == (3, 2)
        names = pipe.get_feature_names_out()
        assert "scaled_price" in names
        assert "scaled_city" in names

    def test_standard_scaler_and_imputer_union(self, backend: DuckDBBackend) -> None:
        """Union with StandardScaler and Imputer branches."""
        from sqlearn.imputers.imputer import Imputer
        from sqlearn.scalers.standard import StandardScaler

        union = Union(
            [
                ("scaled", StandardScaler()),
                ("imputed", Imputer(strategy="mean", columns=["price"])),
            ]
        )
        pipe = Pipeline([union])
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)
        assert result.shape[0] == 3
        names = pipe.get_feature_names_out()
        assert "scaled_price" in names
        assert "imputed_price" in names


# ── get_feature_names_out tests ────────────────────────────────────


class TestUnionGetFeatureNamesOut:
    """Test Union.get_feature_names_out()."""

    def test_before_fit_raises(self) -> None:
        """get_feature_names_out() before fit raises NotFittedError."""
        union = Union([("a", _StaticStep())])
        with pytest.raises(NotFittedError):
            union.get_feature_names_out()

    def test_returns_prefixed_names(self, tmp_path: Any) -> None:
        """get_feature_names_out() returns prefixed column names."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        union = Union([("a", _StaticStep()), ("b", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        names = union.get_feature_names_out()
        assert "a_x" in names
        assert "a_y" in names
        assert "b_x" in names
        assert "b_y" in names


# ── Clone tests ────────────────────────────────────────────────────


class TestUnionClone:
    """Test Union.clone() method."""

    def test_clone_creates_independent_copy(self) -> None:
        """clone() creates independent Union with cloned branches."""
        union = Union([("a", _StaticStep()), ("b", _DynamicStep())])
        cloned = union.clone()
        assert cloned is not union
        assert len(cloned.transformers) == len(union.transformers)
        assert cloned.transformers[0][1] is not union.transformers[0][1]
        assert cloned.transformers[1][1] is not union.transformers[1][1]

    def test_clone_preserves_names(self) -> None:
        """clone() preserves branch names."""
        union = Union([("scale", _StaticStep())])
        cloned = union.clone()
        assert cloned.transformers[0][0] == "scale"

    def test_clone_preserves_fitted_state(self, tmp_path: Any) -> None:
        """clone() preserves fitted state from branches."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        union = Union([("a", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=DuckDBBackend(db_path))

        cloned = union.clone()
        assert cloned._fitted is True
        assert cloned.output_schema_ is not None

    def test_clone_resets_thread_ownership(self) -> None:
        """clone() resets thread ownership for thread safety."""
        union = Union([("a", _StaticStep())])
        union._owner_thread = 12345
        union._owner_pid = 99999
        cloned = union.clone()
        assert cloned._owner_thread is None
        assert cloned._owner_pid is None


# ── Pickle tests ───────────────────────────────────────────────────


class TestUnionPickle:
    """Test Union pickle serialization."""

    def test_pickle_roundtrip(self) -> None:
        """Union survives pickle roundtrip with real transformers."""
        from sqlearn.scalers.standard import StandardScaler

        union = Union([("a", StandardScaler()), ("b", StandardScaler(with_mean=False))])
        data = pickle.dumps(union)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored, Union)
        assert len(restored.transformers) == 2
        assert restored.transformers[0][0] == "a"

    def test_pickle_fitted(self, tmp_path: Any) -> None:
        """Fitted Union survives pickle roundtrip."""
        import duckdb

        from sqlearn.scalers.standard import StandardScaler

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        union = Union([("a", StandardScaler())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=DuckDBBackend(db_path))

        data = pickle.dumps(union)
        restored = pickle.loads(data)  # noqa: S301
        assert restored._fitted is True
        assert restored._connection is None


# ── Get params tests ───────────────────────────────────────────────


class TestUnionGetParams:
    """Test Union.get_params() for sklearn introspection."""

    def test_get_params_shallow(self) -> None:
        """get_params(deep=False) returns transformers list."""
        s1, s2 = _StaticStep(), _DynamicStep()
        union = Union([("a", s1), ("b", s2)])
        params = union.get_params(deep=False)
        assert "transformers" in params
        assert params["transformers"] == [("a", s1), ("b", s2)]

    def test_get_params_deep(self) -> None:
        """get_params(deep=True) includes nested transformer params."""
        from sqlearn.scalers.standard import StandardScaler

        scaler = StandardScaler(with_mean=False)
        union = Union([("scaled", scaler)])
        params = union.get_params(deep=True)
        assert "scaled__with_mean" in params
        assert params["scaled__with_mean"] is False
        assert "scaled__with_std" in params
        assert params["scaled__with_std"] is True


# ── Not-fitted guard tests ─────────────────────────────────────────


class TestUnionNotFittedGuard:
    """Test not-fitted error behavior."""

    def test_get_feature_names_out_not_fitted(self) -> None:
        """get_feature_names_out() raises NotFittedError when not fitted."""
        union = Union([("a", _StaticStep())])
        with pytest.raises(NotFittedError, match="not fitted"):
            union.get_feature_names_out()


# ── Single branch tests ───────────────────────────────────────────


class TestUnionSingleBranch:
    """Test Union with a single branch (trivial case)."""

    def test_single_branch_passthrough(self, tmp_path: Any) -> None:
        """Single branch Union works like the branch itself but with prefix."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        be = DuckDBBackend(db_path)
        union = Union([("only", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=be)
        pipe.transform("data", backend=be)
        names = pipe.get_feature_names_out()
        assert names == ["only_x", "only_y"]

    def test_single_branch_to_sql(self, tmp_path: Any) -> None:
        """Single branch Union produces valid SQL."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x")
        conn.close()

        union = Union([("only", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        sql = pipe.to_sql()
        assert "only_x" in sql


# ── Duplicate column handling tests ────────────────────────────────


class TestUnionDuplicateColumns:
    """Test that Union handles duplicate column names via prefixing."""

    def test_same_columns_different_branches(self, tmp_path: Any) -> None:
        """Two branches processing the same columns get distinct prefixed names."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        union = Union([("a", _StaticStep()), ("b", _StaticStep())])
        pipe = Pipeline([union])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        names = pipe.get_feature_names_out()
        # Both branches produce x and y, but with different prefixes
        assert "a_x" in names
        assert "a_y" in names
        assert "b_x" in names
        assert "b_y" in names
        assert len(names) == 4
