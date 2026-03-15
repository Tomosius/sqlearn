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


class TestPipelineOperators:
    """Test Pipeline + and += operators."""

    def test_pipeline_plus_transformer(self) -> None:
        """Pipeline + Transformer appends step with auto-generated name."""
        pipe = Pipeline([("a", _StaticStep())])
        new = pipe + _DynamicStep()
        assert len(new.steps) == 2
        assert new.steps[0][0] == "a"
        assert new.steps[1][0].startswith("step_")

    def test_pipeline_plus_pipeline(self) -> None:
        """Pipeline + Pipeline flattens steps from both into one Pipeline."""
        pipe1 = Pipeline([("a", _StaticStep())])
        pipe2 = Pipeline([("b", _DynamicStep())])
        combined = pipe1 + pipe2
        assert len(combined.steps) == 2
        assert combined.steps[0][0] == "a"
        assert combined.steps[1][0] == "b"

    def test_pipeline_plus_name_collision_raises(self) -> None:
        """Pipeline + Pipeline with name collision raises InvalidStepError."""
        pipe1 = Pipeline([("same", _StaticStep())])
        pipe2 = Pipeline([("same", _DynamicStep())])
        with pytest.raises(InvalidStepError, match="Duplicate step name"):
            pipe1 + pipe2

    def test_pipeline_plus_non_transformer(self) -> None:
        """Pipeline + non-Transformer returns NotImplemented."""
        pipe = Pipeline([_StaticStep()])
        result = pipe.__add__(42)  # type: ignore[arg-type]
        assert result is NotImplemented

    def test_iadd_returns_new_pipeline(self) -> None:
        """Pipeline += returns NEW Pipeline, non-mutating."""
        pipe = Pipeline([("a", _StaticStep())])
        original_id = id(pipe)
        pipe += _DynamicStep()  # type: ignore[assignment]
        assert id(pipe) != original_id
        assert len(pipe.steps) == 2

    def test_radd_transformer_plus_pipeline(self) -> None:
        """Transformer + Pipeline via __radd__ prepends step."""
        pipe = Pipeline([("b", _DynamicStep())])
        step = _StaticStep()
        combined = step + pipe  # type: ignore[operator]
        assert len(combined.steps) == 2
        assert combined.steps[1][0] == "b"

    def test_pipeline_plus_pipeline_multi_step(self) -> None:
        """Pipeline(2 steps) + Pipeline(2 steps) produces Pipeline with 4 steps."""
        pipe1 = Pipeline([("a", _StaticStep()), ("b", _DynamicStep())])
        pipe2 = Pipeline([("c", _StaticStep()), ("d", _DynamicStep())])
        combined = pipe1 + pipe2
        assert len(combined.steps) == 4
        assert [n for n, _ in combined.steps] == ["a", "b", "c", "d"]

    def test_three_way_add(self) -> None:
        """a + b + c chains left-to-right producing a flat 3-step Pipeline."""
        s1, s2, s3 = _StaticStep(), _DynamicStep(), _StaticStep()
        result = s1 + s2 + s3
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 3

    def test_pipeline_plus_transformer_preserves_order(self) -> None:
        """Pipeline + Transformer preserves existing step order with new step at end."""
        pipe = Pipeline([("x", _StaticStep()), ("y", _DynamicStep())])
        combined = pipe + _StaticStep()
        assert combined.steps[0][0] == "x"
        assert combined.steps[1][0] == "y"
        assert len(combined.steps) == 3

    def test_transformer_plus_pipeline_preserves_order(self) -> None:
        """Transformer + Pipeline prepends step before existing pipeline steps."""
        pipe = Pipeline([("x", _StaticStep()), ("y", _DynamicStep())])
        step = _StaticStep()
        combined = step + pipe
        assert combined.steps[1][0] == "x"
        assert combined.steps[2][0] == "y"
        assert len(combined.steps) == 3

    def test_add_non_mutating_original_pipeline_unchanged(self) -> None:
        """Pipeline + X does not mutate the original Pipeline."""
        s1 = _StaticStep()
        pipe = Pipeline([("a", s1)])
        original_steps = pipe.steps[:]
        _ = pipe + _DynamicStep()
        assert pipe.steps == original_steps
        assert len(pipe.steps) == 1

    def test_add_non_mutating_both_pipelines_unchanged(self) -> None:
        """Pipeline + Pipeline leaves both original pipelines unchanged."""
        pipe1 = Pipeline([("a", _StaticStep())])
        pipe2 = Pipeline([("b", _DynamicStep())])
        _ = pipe1 + pipe2
        assert len(pipe1.steps) == 1
        assert len(pipe2.steps) == 1

    def test_iadd_pipeline_plus_pipeline(self) -> None:
        """Pipeline += Pipeline flattens and returns new Pipeline."""
        pipe1 = Pipeline([("a", _StaticStep())])
        pipe2 = Pipeline([("b", _DynamicStep())])
        original_id = id(pipe1)
        pipe1 += pipe2  # type: ignore[assignment]
        assert id(pipe1) != original_id
        assert len(pipe1.steps) == 2
        assert pipe1.steps[0][0] == "a"
        assert pipe1.steps[1][0] == "b"

    def test_add_clones_steps_for_independence(self) -> None:
        """Combined pipeline has cloned steps independent from originals."""
        step = _StaticStep()
        pipe = Pipeline([("a", step)])
        combined = pipe + _DynamicStep()
        # The step in combined is a clone, not the same object
        assert combined.steps[0][1] is not step

    def test_add_pipeline_plus_pipeline_clones_both_sides(self) -> None:
        """Pipeline + Pipeline clones steps from both operands."""
        step1 = _StaticStep()
        step2 = _DynamicStep()
        pipe1 = Pipeline([("a", step1)])
        pipe2 = Pipeline([("b", step2)])
        combined = pipe1 + pipe2
        assert combined.steps[0][1] is not step1
        assert combined.steps[1][1] is not step2

    def test_radd_clones_steps_for_independence(self) -> None:
        """Transformer + Pipeline clones all steps for independence."""
        step1 = _StaticStep()
        step2 = _DynamicStep()
        pipe = Pipeline([("b", step2)])
        combined = step1 + pipe
        assert combined.steps[0][1] is not step1
        assert combined.steps[1][1] is not step2

    def test_single_step_pipeline_plus_transformer(self) -> None:
        """Single-step Pipeline + Transformer produces 2-step flat Pipeline."""
        pipe = Pipeline([_StaticStep()])
        combined = pipe + _DynamicStep()
        assert len(combined.steps) == 2
        assert not isinstance(combined.steps[0][1], Pipeline)
        assert not isinstance(combined.steps[1][1], Pipeline)

    def test_single_step_pipeline_plus_single_step_pipeline(self) -> None:
        """Single-step Pipeline + single-step Pipeline produces 2-step Pipeline."""
        pipe1 = Pipeline([_StaticStep()])
        pipe2 = Pipeline([_DynamicStep()])
        combined = pipe1 + pipe2
        assert len(combined.steps) == 2

    def test_radd_non_transformer_returns_not_implemented(self) -> None:
        """Pipeline.__radd__(non-Transformer) returns NotImplemented."""
        pipe = Pipeline([_StaticStep()])
        result = pipe.__radd__(42)  # type: ignore[arg-type]
        assert result is NotImplemented


class TestTransformerOperators:
    """Test Transformer.__add__ and __iadd__ creating Pipelines."""

    def test_transformer_plus_transformer(self) -> None:
        """Transformer + Transformer creates a 2-step Pipeline."""
        s1, s2 = _StaticStep(), _DynamicStep()
        result = s1 + s2
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2

    def test_transformer_plus_pipeline(self) -> None:
        """Transformer + Pipeline prepends step and flattens."""
        pipe = Pipeline([("b", _DynamicStep())])
        step = _StaticStep()
        result = step + pipe
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2
        assert result.steps[1][0] == "b"

    def test_transformer_iadd(self) -> None:
        """Transformer += creates Pipeline."""
        s1 = _StaticStep()
        result = s1.__iadd__(_DynamicStep())
        assert isinstance(result, Pipeline)

    def test_transformer_plus_transformer_clones(self) -> None:
        """Transformer + Transformer clones both for independence."""
        s1, s2 = _StaticStep(), _DynamicStep()
        result = s1 + s2
        assert result.steps[0][1] is not s1
        assert result.steps[1][1] is not s2

    def test_transformer_plus_non_transformer(self) -> None:
        """Transformer + int returns NotImplemented."""
        result = _StaticStep().__add__(42)
        assert result is NotImplemented


class TestPipelineOperatorIntegration:
    """Test that pipelines created via + actually fit/transform/to_sql."""

    def test_combined_pipeline_fit_transform(self, tmp_path: Any) -> None:
        """Pipeline built with + can fit and transform data end-to-end."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE data AS SELECT "
            "1.0 AS x, 2.0 AS y UNION ALL SELECT "
            "3.0, 4.0 UNION ALL SELECT "
            "5.0, 6.0"
        )
        conn.close()

        pipe1 = Pipeline([_StaticStep()])
        pipe2 = Pipeline([_StaticStep()])
        combined = pipe1 + pipe2

        backend = DuckDBBackend(db_path)
        combined.fit("data", backend=backend)
        result = combined.transform("data", backend=backend)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        # Static step doubles: 1*2*2=4, 2*2*2=8, etc.
        np.testing.assert_array_equal(result[0], [4.0, 8.0])

    def test_combined_pipeline_to_sql(self, tmp_path: Any) -> None:
        """Pipeline built with + generates valid SQL via to_sql()."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")
        conn.close()

        pipe1 = Pipeline([_StaticStep()])
        pipe2 = Pipeline([_StaticStep()])
        combined = pipe1 + pipe2

        backend = DuckDBBackend(db_path)
        combined.fit("data", backend=backend)
        sql = combined.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql

    def test_transformer_plus_transformer_fit_transform(self, tmp_path: Any) -> None:
        """Pipeline built via Transformer + Transformer can fit and transform."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y UNION ALL SELECT 3.0, 4.0")
        conn.close()

        combined = _StaticStep() + _StaticStep()
        backend = DuckDBBackend(db_path)
        combined.fit("data", backend=backend)
        result = combined.transform("data", backend=backend)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [4.0, 8.0])


class TestMergeSteps:
    """Test _merge_steps helper for auto-name collision resolution."""

    def test_no_collision(self) -> None:
        """Non-colliding names are preserved as-is."""
        from sqlearn.core.pipeline import _merge_steps

        left = [("a", _StaticStep())]
        right = [("b", _DynamicStep())]
        result = _merge_steps(left, right)
        assert [n for n, _ in result] == ["a", "b"]

    def test_auto_name_collision_renumbered(self) -> None:
        """Colliding step_NN names on the right are renumbered."""
        from sqlearn.core.pipeline import _merge_steps

        left = [("step_00", _StaticStep())]
        right = [("step_00", _DynamicStep())]
        result = _merge_steps(left, right)
        assert len(result) == 2
        assert result[0][0] == "step_00"
        assert result[1][0] == "step_01"

    def test_user_name_collision_preserved(self) -> None:
        """User-given names (non step_NN) are NOT renamed on collision.

        User-given name collisions are intentionally left for
        _normalize_steps to catch with a clear error message.
        """
        from sqlearn.core.pipeline import _merge_steps

        left = [("my_step", _StaticStep())]
        right = [("my_step", _DynamicStep())]
        result = _merge_steps(left, right)
        # Both kept — _normalize_steps will catch the duplicate
        assert len(result) == 2
        assert result[0][0] == "my_step"
        assert result[1][0] == "my_step"

    def test_multiple_auto_name_collisions(self) -> None:
        """Multiple colliding auto-names are each renumbered sequentially."""
        from sqlearn.core.pipeline import _merge_steps

        left = [("step_00", _StaticStep()), ("step_01", _DynamicStep())]
        right = [("step_00", _StaticStep()), ("step_01", _DynamicStep())]
        result = _merge_steps(left, right)
        assert len(result) == 4
        names = [n for n, _ in result]
        assert len(set(names)) == 4  # all unique


class TestPipelineClone:
    """Test Pipeline.clone()."""

    def test_clone_independent_copy(self) -> None:
        """clone() creates independent copy."""
        pipe = Pipeline([("a", _StaticStep()), ("b", _DynamicStep())])
        cloned = pipe.clone()
        assert cloned is not pipe
        assert len(cloned.steps) == len(pipe.steps)

    def test_clone_preserves_names(self) -> None:
        """clone() preserves step names."""
        pipe = Pipeline([("scale", _StaticStep())])
        cloned = pipe.clone()
        assert cloned.steps[0][0] == "scale"

    def test_clone_independent_steps(self) -> None:
        """clone() steps are independent objects."""
        step = _StaticStep()
        pipe = Pipeline([("a", step)])
        cloned = pipe.clone()
        assert cloned.steps[0][1] is not step

    def test_clone_no_backend(self) -> None:
        """clone() does not copy backend."""
        pipe = Pipeline([_StaticStep()], backend="test.duckdb")
        cloned = pipe.clone()
        assert cloned._backend is None
        assert cloned._owns_backend is False

    def test_clone_preserves_fitted_state(self, tmp_path: Any) -> None:
        """clone() preserves is_fitted when original is fitted."""
        import duckdb

        path = str(tmp_path / "data.parquet")
        conn = duckdb.connect()
        conn.execute(f"COPY (SELECT 1.0 AS x, 2.0 AS y) TO '{path}' (FORMAT PARQUET)")
        conn.close()

        pipe = Pipeline([_StaticStep()])
        pipe.fit(path)
        cloned = pipe.clone()
        assert cloned.is_fitted is True
        assert cloned.get_feature_names_out() == pipe.get_feature_names_out()


class TestPipelineContextManager:
    """Test Pipeline context manager."""

    def test_context_manager_returns_self(self) -> None:
        """__enter__ returns Pipeline."""
        pipe = Pipeline([_StaticStep()])
        with pipe as p:
            assert p is pipe

    def test_context_manager_closes_owned_backend(self, tmp_path: Any) -> None:
        """__exit__ closes auto-created (owned) backend."""
        import duckdb

        path = str(tmp_path / "data.parquet")
        conn = duckdb.connect()
        conn.execute(f"COPY (SELECT 1.0 AS x) TO '{path}' (FORMAT PARQUET)")
        conn.close()

        with Pipeline([_StaticStep()]) as pipe:
            pipe.fit(path)
            assert pipe._owns_backend is True
        assert pipe._backend_instance is not None
        assert pipe._backend_instance._connection is None

    def test_context_manager_does_not_close_user_backend(self, tmp_path: Any) -> None:
        """__exit__ does NOT close user-provided backend."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x")
        conn.close()

        user_backend = DuckDBBackend(db_path)
        with Pipeline([_StaticStep()]) as pipe:
            pipe.fit("data", backend=user_backend)
            assert pipe._owns_backend is False
        assert user_backend._connection is not None

    def test_context_manager_no_backend_no_error(self) -> None:
        """__exit__ works even if no backend was created."""
        with Pipeline([_StaticStep()]):
            pass


class TestAutoNameWithExistingSteps:
    """Test _auto_name increments past existing step_NN names."""

    def test_auto_name_after_auto_named(self) -> None:
        """Adding to a pipeline with auto-named steps increments the counter.

        When existing steps use step_00, step_01, the next auto-name must be
        step_02, not step_00 again (which would cause a duplicate name error).
        """
        pipe = Pipeline([_StaticStep(), _StaticStep()])
        # pipe.steps are step_00, step_01
        new = pipe + _DynamicStep()
        assert new.steps[2][0] == "step_02"

    def test_auto_name_gap_in_numbering(self) -> None:
        """Auto-name picks max+1 even with gaps in existing numbering.

        If steps are named step_00 and step_05 (gap), the next auto-name
        should be step_06, not step_01 or step_02.
        """
        pipe = Pipeline([("step_00", _StaticStep()), ("step_05", _DynamicStep())])
        new = pipe + _StaticStep()
        assert new.steps[2][0] == "step_06"


class TestResolveBackendVariants:
    """Test _resolve_backend with raw DuckDB connections."""

    def test_fit_with_raw_connection(self, tmp_path: Any) -> None:
        """fit(backend=raw_connection) wraps in DuckDBBackend automatically.

        Users should be able to pass a plain duckdb.DuckDBPyConnection to
        fit() without manually wrapping it in DuckDBBackend.
        """
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x, 2.0 AS y")

        pipe = Pipeline([_StaticStep()])
        pipe.fit("data", backend=conn)
        assert pipe.is_fitted is True
        conn.close()

    def test_pipeline_stored_raw_connection(self, tmp_path: Any) -> None:
        """Pipeline(steps, backend=raw_connection) stores and wraps on first use.

        When a raw connection is passed at construction time, it should be
        lazily wrapped into a DuckDBBackend on the first fit() call.
        """
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE data AS SELECT 1.0 AS x")

        pipe = Pipeline([_StaticStep()], backend=conn)
        pipe.fit("data")
        assert pipe.is_fitted is True
        assert pipe._backend_instance is not None
        conn.close()


class TestMultiLayerFit:
    """Test Pipeline.fit() with schema-changing dynamic steps (multi-layer)."""

    def test_multi_layer_creates_temp_views(self, tmp_path: Any) -> None:
        """Schema-changing dynamic step triggers multi-layer compilation.

        When a dynamic step changes the schema, the compiler creates a layer
        boundary. Pipeline.fit() materializes intermediate layers as temp
        views before fitting subsequent steps. Requires a step AFTER the
        schema-changing step to actually create 2+ layers.
        Covers lines 322-338.
        """
        import duckdb

        from sqlearn.encoders.onehot import OneHotEncoder
        from sqlearn.scalers.standard import StandardScaler

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE data AS SELECT "
            "1.0 AS price, 'a' AS city UNION ALL SELECT "
            "2.0, 'b' UNION ALL SELECT "
            "3.0, 'a'"
        )
        conn.close()

        backend = DuckDBBackend(db_path)
        # OneHotEncoder (dynamic + schema-changing → layer boundary)
        # → StandardScaler (dynamic → second layer, needs temp view from first)
        pipe = Pipeline(
            [OneHotEncoder(columns=["city"]), StandardScaler(columns=["price"])],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)
        assert result.shape[0] == 3
        names = pipe.get_feature_names_out()
        assert "price" in names


class TestTransformEmptyTable:
    """Test Pipeline.transform() on a table with zero rows."""

    def test_transform_empty_returns_empty_array(self, tmp_path: Any) -> None:
        """transform() on an empty table returns an empty (0, N) numpy array.

        The pipeline should handle the zero-row case gracefully, returning
        a float64 array with shape (0, num_columns) instead of crashing.
        """
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE train AS SELECT 1.0 AS x, 2.0 AS y")
        conn.execute("CREATE TABLE empty AS SELECT * FROM train WHERE FALSE")
        conn.close()

        backend = DuckDBBackend(db_path)
        pipe = Pipeline([_StaticStep()])
        pipe.fit("train", backend=backend)
        result = pipe.transform("empty", backend=backend)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64


class TestPipelineRaddDirect:
    """Test Pipeline.__radd__ called directly (bypasses Transformer.__add__)."""

    def test_radd_with_transformer(self) -> None:
        """Pipeline.__radd__(Transformer) prepends step.

        When called directly (not through the + operator), __radd__ should
        create a new Pipeline with the transformer prepended before the
        existing steps.
        """
        pipe = Pipeline([("b", _DynamicStep())])
        result = pipe.__radd__(_StaticStep())
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2
        assert result.steps[1][0] == "b"

    def test_radd_non_transformer_returns_not_implemented(self) -> None:
        """Pipeline.__radd__(non-Transformer) returns NotImplemented.

        When the left operand is not a Transformer, __radd__ cannot combine
        them and must return NotImplemented per Python's data model.
        """
        pipe = Pipeline([_StaticStep()])
        result = pipe.__radd__(42)  # type: ignore[arg-type]
        assert result is NotImplemented
