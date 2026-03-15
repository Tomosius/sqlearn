"""Tests for sqlearn.core.pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlglot.expressions as exp

from sqlearn.core.errors import InvalidStepError
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
