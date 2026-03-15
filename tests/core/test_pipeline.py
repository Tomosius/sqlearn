"""Tests for sqlearn.core.pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import InvalidStepError, NotFittedError
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema


# --- Mock transformers (reused across all tests) ---


class _StaticStep(Transformer):
    """Static transformer that doubles values."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Double each column."""
        return {c: exp.Mul(this=exprs[c], expression=exp.Literal.number(2)) for c in columns}


_StaticStep.__module__ = "sqlearn.scalers.fake"


class _DynamicStep(Transformer):
    """Dynamic transformer that needs AVG."""

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


# ── Constructor tests ──────────────────────────────────────────────


class TestPipelineConstructor:
    """Test Pipeline constructor with all three input formats."""

    def test_bare_list(self) -> None:
        """Bare list auto-names as step_00, step_01, ..."""
        from sqlearn.core.pipeline import Pipeline

        s1, s2 = _StaticStep(), _DynamicStep()
        pipe = Pipeline([s1, s2])
        assert pipe.steps == [("step_00", s1), ("step_01", s2)]

    def test_bare_list_single(self) -> None:
        """Single-element bare list gets step_00 (min width 2)."""
        from sqlearn.core.pipeline import Pipeline

        s = _StaticStep()
        pipe = Pipeline([s])
        assert pipe.steps == [("step_00", s)]

    def test_tuple_list(self) -> None:
        """Tuple list used as-is."""
        from sqlearn.core.pipeline import Pipeline

        s1, s2 = _StaticStep(), _DynamicStep()
        pipe = Pipeline([("scale", s1), ("center", s2)])
        assert pipe.steps == [("scale", s1), ("center", s2)]

    def test_dict_input(self) -> None:
        """Dict preserves insertion order."""
        from sqlearn.core.pipeline import Pipeline

        s1, s2 = _StaticStep(), _DynamicStep()
        pipe = Pipeline({"scale": s1, "center": s2})
        assert pipe.steps == [("scale", s1), ("center", s2)]

    def test_empty_steps_raises(self) -> None:
        """Empty steps raises InvalidStepError."""
        from sqlearn.core.pipeline import Pipeline

        with pytest.raises(InvalidStepError, match="at least one step"):
            Pipeline([])

    def test_non_transformer_raises(self) -> None:
        """Non-Transformer element raises InvalidStepError."""
        from sqlearn.core.pipeline import Pipeline

        with pytest.raises(InvalidStepError, match="not a Transformer"):
            Pipeline(["not_a_transformer"])  # type: ignore[list-item]

    def test_non_transformer_in_tuple_raises(self) -> None:
        """Non-Transformer in tuple raises InvalidStepError."""
        from sqlearn.core.pipeline import Pipeline

        with pytest.raises(InvalidStepError, match="not a Transformer"):
            Pipeline([("bad", 42)])  # type: ignore[list-item]

    def test_duplicate_names_raises(self) -> None:
        """Duplicate step names raise InvalidStepError."""
        from sqlearn.core.pipeline import Pipeline

        s1, s2 = _StaticStep(), _DynamicStep()
        with pytest.raises(InvalidStepError, match="Duplicate step name"):
            Pipeline([("same", s1), ("same", s2)])

    def test_empty_dict_raises(self) -> None:
        """Empty dict raises InvalidStepError."""
        from sqlearn.core.pipeline import Pipeline

        with pytest.raises(InvalidStepError, match="at least one step"):
            Pipeline({})


class TestPipelineProperties:
    """Test Pipeline read-only properties."""

    def test_steps_returns_copy(self) -> None:
        """steps property returns defensive copy."""
        from sqlearn.core.pipeline import Pipeline

        s = _StaticStep()
        pipe = Pipeline([s])
        steps1 = pipe.steps
        steps2 = pipe.steps
        assert steps1 == steps2
        assert steps1 is not steps2

    def test_named_steps(self) -> None:
        """named_steps returns dict access."""
        from sqlearn.core.pipeline import Pipeline

        s1, s2 = _StaticStep(), _DynamicStep()
        pipe = Pipeline([("scale", s1), ("center", s2)])
        assert pipe.named_steps == {"scale": s1, "center": s2}

    def test_is_fitted_false_initially(self) -> None:
        """is_fitted is False before fit()."""
        from sqlearn.core.pipeline import Pipeline

        pipe = Pipeline([_StaticStep()])
        assert pipe.is_fitted is False


class TestPipelineRepr:
    """Test Pipeline __repr__."""

    def test_repr_format(self) -> None:
        """Repr shows step names and class names."""
        from sqlearn.core.pipeline import Pipeline

        pipe = Pipeline([("scale", _StaticStep()), ("center", _DynamicStep())])
        r = repr(pipe)
        assert r == "Pipeline(scale=_StaticStep, center=_DynamicStep)"

    def test_repr_auto_named(self) -> None:
        """Repr with auto-named steps."""
        from sqlearn.core.pipeline import Pipeline

        pipe = Pipeline([_StaticStep()])
        assert repr(pipe) == "Pipeline(step_00=_StaticStep)"


class TestPipelineFit:
    """Test Pipeline.fit() workflow."""

    @pytest.fixture
    def backend(self, tmp_path: Any) -> DuckDBBackend:
        """Create a DuckDB backend with test data."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE test_data AS SELECT "
            "1.0 AS price, 'a' AS city UNION ALL SELECT "
            "2.0, 'b' UNION ALL SELECT "
            "3.0, 'a'"
        )
        conn.close()
        return DuckDBBackend(db_path)

    def test_fit_returns_self(self, backend: DuckDBBackend) -> None:
        """fit() returns self for chaining."""
        pipe = Pipeline([_StaticStep()])
        result = pipe.fit("test_data", backend=backend)
        assert result is pipe

    def test_fit_sets_fitted(self, backend: DuckDBBackend) -> None:
        """fit() sets is_fitted to True."""
        pipe = Pipeline([_StaticStep()])
        pipe.fit("test_data", backend=backend)
        assert pipe.is_fitted is True

    def test_fit_static_only(self, backend: DuckDBBackend) -> None:
        """fit() with static-only steps marks them fitted."""
        step = _StaticStep()
        pipe = Pipeline([step])
        pipe.fit("test_data", backend=backend)
        assert step._fitted is True
        assert step.columns_ is not None

    def test_fit_dynamic_populates_params(self, backend: DuckDBBackend) -> None:
        """fit() with dynamic step populates params_ from aggregate query."""
        step = _DynamicStep(columns="numeric")
        pipe = Pipeline([step])
        pipe.fit("test_data", backend=backend)
        assert step.params_ is not None
        assert step._fitted is True

    def test_fit_auto_creates_backend(self, tmp_path: Any) -> None:
        """fit() auto-creates in-memory DuckDB when no backend given."""
        import duckdb

        path = str(tmp_path / "data.parquet")
        conn = duckdb.connect()
        conn.execute(f"COPY (SELECT 1.0 AS x, 2.0 AS y) TO '{path}' (FORMAT PARQUET)")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit(path)
        assert pipe.is_fitted is True
        assert pipe._owns_backend is True

    def test_fit_stores_schemas(self, backend: DuckDBBackend) -> None:
        """fit() stores input and output schemas."""
        pipe = Pipeline([_StaticStep()])
        pipe.fit("test_data", backend=backend)
        assert pipe._schema_in is not None
        assert pipe._schema_out is not None


class TestPipelineTransform:
    """Test Pipeline.transform() workflow."""

    @pytest.fixture
    def fitted_pipe(self, tmp_path: Any) -> tuple[Pipeline, str]:
        """Create a fitted Pipeline with test data."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE train AS SELECT "
            "1.0 AS x, 2.0 AS y UNION ALL SELECT "
            "3.0, 4.0 UNION ALL SELECT "
            "5.0, 6.0"
        )
        conn.close()

        backend = DuckDBBackend(db_path)
        pipe = Pipeline([_StaticStep()])
        pipe.fit("train", backend=backend)
        return pipe, db_path

    def test_transform_returns_numpy(self, fitted_pipe: tuple[Pipeline, str]) -> None:
        """transform() returns numpy array."""
        pipe, db_path = fitted_pipe
        result = pipe.transform("train", backend=DuckDBBackend(db_path))
        assert isinstance(result, np.ndarray)

    def test_transform_shape(self, fitted_pipe: tuple[Pipeline, str]) -> None:
        """transform() returns correct shape."""
        pipe, db_path = fitted_pipe
        result = pipe.transform("train", backend=DuckDBBackend(db_path))
        assert result.shape == (3, 2)

    def test_transform_before_fit_raises(self) -> None:
        """transform() before fit() raises NotFittedError."""
        pipe = Pipeline([_StaticStep()])
        with pytest.raises(NotFittedError):
            pipe.transform("some_table")

    def test_transform_float64_dtype(self, fitted_pipe: tuple[Pipeline, str]) -> None:
        """transform() returns float64 by default."""
        pipe, db_path = fitted_pipe
        result = pipe.transform("train", backend=DuckDBBackend(db_path))
        assert result.dtype == np.float64


class TestPipelineFitTransform:
    """Test Pipeline.fit_transform()."""

    def test_fit_transform_equivalent(self, tmp_path: Any) -> None:
        """fit_transform() equals fit().transform()."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y UNION ALL SELECT 3.0, 4.0")
        conn.close()

        backend1 = DuckDBBackend(db_path)
        pipe1 = Pipeline([_StaticStep()])
        result1 = pipe1.fit_transform("data", backend=backend1)

        backend2 = DuckDBBackend(db_path)
        pipe2 = Pipeline([_StaticStep()])
        pipe2.fit("data", backend=backend2)
        result2 = pipe2.transform("data", backend=backend2)

        np.testing.assert_array_equal(result1, result2)


class TestPipelineToSql:
    """Test Pipeline.to_sql()."""

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() before fit() raises NotFittedError."""
        pipe = Pipeline([_StaticStep()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()

    def test_to_sql_returns_string(self, tmp_path: Any) -> None:
        """to_sql() returns SQL string."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql

    def test_to_sql_custom_dialect(self, tmp_path: Any) -> None:
        """to_sql() respects dialect parameter."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        sql = pipe.to_sql(dialect="postgres")
        assert isinstance(sql, str)

    def test_to_sql_custom_table(self, tmp_path: Any) -> None:
        """to_sql() respects table parameter."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        sql = pipe.to_sql(table="my_table")
        assert "my_table" in sql


class TestPipelineGetFeatureNamesOut:
    """Test Pipeline.get_feature_names_out()."""

    def test_before_fit_raises(self) -> None:
        """get_feature_names_out() before fit() raises NotFittedError."""
        pipe = Pipeline([_StaticStep()])
        with pytest.raises(NotFittedError):
            pipe.get_feature_names_out()

    def test_returns_column_names(self, tmp_path: Any) -> None:
        """get_feature_names_out() returns output column names."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit("data", backend=DuckDBBackend(db_path))
        names = pipe.get_feature_names_out()
        assert isinstance(names, list)
        assert "x" in names
        assert "y" in names
