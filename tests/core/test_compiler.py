"""Tests for sqlearn.core.compiler."""

from __future__ import annotations

from typing import Any

import pytest
import sqlglot.expressions as exp

from sqlearn.core.compiler import (
    classify_step,
    compose_transform,
    detect_schema_change,
    plan_fit,
)
from sqlearn.core.errors import ClassificationError, CompilationError, SchemaError
from sqlearn.core.schema import Schema, resolve_columns
from sqlearn.core.transformer import Transformer

# --- Mock transformers for testing ---


class _BuiltinStatic(Transformer):
    """Simulates a built-in static transformer (Tier 1)."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Double each column."""
        return {c: exp.Mul(this=exprs[c], expression=exp.Literal.number(2)) for c in columns}


# Pretend it's built-in by patching __module__
_BuiltinStatic.__module__ = "sqlearn.scalers.fake"


class _BuiltinDynamic(Transformer):
    """Simulates a built-in dynamic transformer (Tier 1)."""

    _classification = "dynamic"
    _default_columns = "numeric"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn AVG per column."""
        return {f"{c}__mean": exp.Avg(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Subtract mean."""
        return {c: exp.Sub(this=exprs[c], expression=exp.Literal.number(0)) for c in columns}


_BuiltinDynamic.__module__ = "sqlearn.scalers.fake"


class _CustomDeclaredStatic(Transformer):
    """Custom transformer declaring static (Tier 2)."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDeclaredStaticBadDiscover(Transformer):
    """Custom declaring static but discover() returns non-empty (Tier 2 mismatch)."""

    _classification = "static"
    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Returns aggregations — contradicts static declaration."""
        return {f"{c}__mean": exp.Avg(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDeclaredStaticBadSets(Transformer):
    """Custom declaring static but discover_sets() returns non-empty (Tier 2 mismatch)."""

    _classification = "static"
    _default_columns = "all"

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Returns set queries — contradicts static declaration."""
        return {"cats": exp.select("city").from_("data").distinct()}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDeclaredDynamicEmpty(Transformer):
    """Custom declaring dynamic but discover() returns {} (Tier 2 wasteful)."""

    _classification = "dynamic"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomUndeclaredStatic(Transformer):
    """Custom with no _classification, discover() returns {} (Tier 3 → static)."""

    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomUndeclaredDynamic(Transformer):
    """Custom with no _classification, discover() returns non-empty (Tier 3 → dynamic)."""

    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Returns aggregations."""
        return {f"{c}__mean": exp.Avg(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDiscoverRaises(Transformer):
    """Custom where discover() raises (Tier 3 → dynamic fallback)."""

    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Raises."""
        msg = "broken"
        raise RuntimeError(msg)

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDiscoverReturnsNone(Transformer):
    """Custom where discover() returns None (Tier 3 → dynamic fallback)."""

    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> Any:
        """Returns None instead of dict."""
        return None

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomDiscoverReturnsNonDict(Transformer):
    """Custom where discover() returns a list (Tier 3 → dynamic fallback)."""

    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> Any:
        """Returns list instead of dict."""
        return [1, 2, 3]

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomWithSets(Transformer):
    """Custom with no _classification, discover_sets() returns non-empty (Tier 3 → dynamic)."""

    _default_columns = "all"

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Returns set queries."""
        return {"cats": exp.select("city").from_("data").distinct()}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


class _CustomOverridesFit(Transformer):
    """Custom that overrides fit() directly (Tier 3 → dynamic)."""

    _default_columns = "all"

    def fit(  # type: ignore[override]
        self,
        data: Any,
        y: str | None = None,
        *,
        backend: Any = None,
    ) -> _CustomOverridesFit:
        """Custom fit."""
        return self

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


# --- Additional mock transformers for schema change / plan_fit ---


class _SchemaChangingDynamic(Transformer):
    """Dynamic transformer that changes schema (adds columns)."""

    _classification = "dynamic"
    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn distinct values."""
        return {f"{c}__count": exp.Count(this=exp.Column(this=c)) for c in columns}

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Add new columns."""
        result: dict[str, exp.Expression] = {}
        for c in columns:
            result[f"{c}_new"] = exp.Literal.number(1)
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Add new columns to schema."""
        new_cols = {f"{c}_new": "INTEGER" for c in schema.columns}
        return schema.add(new_cols)


_SchemaChangingDynamic.__module__ = "sqlearn.scalers.fake"


class _SchemaChangingStatic(Transformer):
    """Static transformer that changes schema (adds columns)."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Add new columns."""
        result: dict[str, exp.Expression] = {}
        for c in columns:
            result[f"{c}_log"] = exp.Ln(this=exprs[c])
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Add log columns to schema."""
        new_cols = {f"{c}_log": "DOUBLE" for c in schema.columns}
        return schema.add(new_cols)


_SchemaChangingStatic.__module__ = "sqlearn.features.fake"


class _OutputSchemaRaises(Transformer):
    """Transformer where output_schema() raises."""

    _classification = "static"
    _default_columns = "all"

    def output_schema(self, schema: Schema) -> Schema:
        """Raises."""
        msg = "broken"
        raise RuntimeError(msg)

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


_OutputSchemaRaises.__module__ = "sqlearn.features.fake"


class _OutputSchemaReturnsNone(Transformer):
    """Transformer where output_schema() returns None."""

    _classification = "static"
    _default_columns = "all"

    def output_schema(self, schema: Schema) -> Any:
        """Returns None."""
        return None

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Noop."""
        return {c: exprs[c] for c in columns}


_OutputSchemaReturnsNone.__module__ = "sqlearn.features.fake"


# --- Test fixtures ---


@pytest.fixture
def schema() -> Schema:
    """Simple schema for testing."""
    return Schema({"price": "DOUBLE", "city": "VARCHAR"})


@pytest.fixture
def columns() -> list[str]:
    """Column list for testing."""
    return ["price", "city"]


# --- classify_step tests ---


class TestClassifyStepTier1:
    """Tier 1: built-in transformers with _classification set."""

    def test_builtin_static(self, columns: list[str], schema: Schema) -> None:
        """Built-in static → trusted, returns 'static', tier 1."""
        step = _BuiltinStatic()
        result = classify_step(step, columns, schema)
        assert result.kind == "static"
        assert result.tier == 1

    def test_builtin_dynamic(self, columns: list[str], schema: Schema) -> None:
        """Built-in dynamic → trusted, returns 'dynamic', tier 1."""
        step = _BuiltinDynamic()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 1


class TestClassifyStepTier2:
    """Tier 2: custom transformers with explicit _classification."""

    def test_declared_static_consistent(self, columns: list[str], schema: Schema) -> None:
        """Custom declared static, discover() returns {} → verified, tier 2."""
        step = _CustomDeclaredStatic()
        result = classify_step(step, columns, schema)
        assert result.kind == "static"
        assert result.tier == 2

    def test_declared_static_discover_mismatch(self, columns: list[str], schema: Schema) -> None:
        """Custom declared static, discover() non-empty → ClassificationError."""
        step = _CustomDeclaredStaticBadDiscover()
        with pytest.raises(ClassificationError):
            classify_step(step, columns, schema)

    def test_declared_static_discover_sets_mismatch(
        self, columns: list[str], schema: Schema
    ) -> None:
        """Custom declared static, discover_sets() non-empty → ClassificationError."""
        step = _CustomDeclaredStaticBadSets()
        with pytest.raises(ClassificationError):
            classify_step(step, columns, schema)

    def test_declared_dynamic_empty_discover(self, columns: list[str], schema: Schema) -> None:
        """Custom declared dynamic, discover() returns {} → warning, honors declaration."""
        step = _CustomDeclaredDynamicEmpty()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 2
        assert len(result.warnings) > 0

    def test_verified_flag_skips_reverification(self, columns: list[str], schema: Schema) -> None:
        """After first verification, _classification_verified=True skips re-check."""
        step = _CustomDeclaredStatic()
        classify_step(step, columns, schema)
        assert getattr(step, "_classification_verified", False) is True
        # Second call should still return same result (no re-verification)
        result = classify_step(step, columns, schema)
        assert result.kind == "static"
        assert result.tier == 2


class TestClassifyStepTier3:
    """Tier 3: custom transformers without _classification."""

    def test_undeclared_static(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover() returns {} → static, tier 3."""
        step = _CustomUndeclaredStatic()
        result = classify_step(step, columns, schema)
        assert result.kind == "static"
        assert result.tier == 3

    def test_undeclared_dynamic(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover() returns non-empty → dynamic, tier 3."""
        step = _CustomUndeclaredDynamic()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3

    def test_discover_raises(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover() raises → dynamic fallback, tier 3."""
        step = _CustomDiscoverRaises()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3

    def test_discover_returns_none(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover() returns None → dynamic fallback, tier 3."""
        step = _CustomDiscoverReturnsNone()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3

    def test_discover_sets_nonempty(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover_sets() returns non-empty → dynamic, tier 3."""
        step = _CustomWithSets()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3

    def test_overrides_fit(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, overrides fit() → dynamic, tier 3."""
        step = _CustomOverridesFit()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3

    def test_discover_returns_non_dict(self, columns: list[str], schema: Schema) -> None:
        """Undeclared, discover() returns list → dynamic fallback, tier 3."""
        step = _CustomDiscoverReturnsNonDict()
        result = classify_step(step, columns, schema)
        assert result.kind == "dynamic"
        assert result.tier == 3


# --- detect_schema_change tests ---


class TestDetectSchemaChange:
    """Test schema change detection."""

    def test_same_schema(self, schema: Schema) -> None:
        """Identical input/output → no change."""
        step = _BuiltinStatic()
        result = detect_schema_change(step, schema)
        assert result.changes is False

    def test_added_columns(self, schema: Schema) -> None:
        """Output has extra columns → change detected."""
        step = _SchemaChangingDynamic()
        result = detect_schema_change(step, schema)
        assert result.changes is True

    def test_removed_columns(self) -> None:
        """Output missing columns → change detected."""
        full_schema = Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})

        class _Dropper(Transformer):
            _classification = "static"
            _default_columns = "all"

            def output_schema(self, schema: Schema) -> Schema:
                return schema.drop(["c"])

            def expressions(
                self, columns: list[str], exprs: dict[str, exp.Expression]
            ) -> dict[str, exp.Expression]:
                return {c: exprs[c] for c in columns if c != "c"}

        _Dropper.__module__ = "sqlearn.ops.fake"

        step = _Dropper()
        result = detect_schema_change(step, full_schema)
        assert result.changes is True

    def test_retyped_columns(self) -> None:
        """Output has different type for same column → change detected."""
        schema = Schema({"a": "INTEGER"})

        class _Retyper(Transformer):
            _classification = "static"
            _default_columns = "all"

            def output_schema(self, schema: Schema) -> Schema:
                return schema.cast({"a": "DOUBLE"})

            def expressions(
                self, columns: list[str], exprs: dict[str, exp.Expression]
            ) -> dict[str, exp.Expression]:
                return {
                    c: exp.Cast(
                        this=exprs[c],
                        to=exp.DataType(this=exp.DataType.Type.DOUBLE),
                    )
                    for c in columns
                }

        _Retyper.__module__ = "sqlearn.ops.fake"

        step = _Retyper()
        result = detect_schema_change(step, schema)
        assert result.changes is True

    def test_output_schema_raises(self, schema: Schema) -> None:
        """output_schema() raises → conservative: assume change."""
        step = _OutputSchemaRaises()
        result = detect_schema_change(step, schema)
        assert result.changes is True

    def test_output_schema_returns_none(self, schema: Schema) -> None:
        """output_schema() returns None → conservative: assume change."""
        step = _OutputSchemaReturnsNone()
        result = detect_schema_change(step, schema)
        assert result.changes is True


# --- plan_fit tests ---


class TestPlanFit:
    """Test plan_fit layer grouping."""

    def test_all_static(self, schema: Schema) -> None:
        """All static steps → one layer."""
        steps = [_BuiltinStatic(), _BuiltinStatic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 1

    def test_all_dynamic_same_schema(self, schema: Schema) -> None:
        """All dynamic same-schema → one layer."""
        steps = [_BuiltinDynamic(), _BuiltinDynamic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 1

    def test_dynamic_schema_changing_creates_boundary(self, schema: Schema) -> None:
        """Dynamic schema-changing step creates layer boundary."""
        steps = [_BuiltinDynamic(), _SchemaChangingDynamic(), _BuiltinDynamic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 2
        # First layer has 2 steps (dynamic + schema-changing dynamic)
        assert len(plan.layers[0].steps) == 2
        # Second layer has 1 step
        assert len(plan.layers[1].steps) == 1

    def test_static_schema_changing_no_boundary(self, schema: Schema) -> None:
        """Static schema-changing does NOT create boundary."""
        steps = [_BuiltinDynamic(), _SchemaChangingStatic(), _BuiltinDynamic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 1

    def test_mixed_pipeline(self, schema: Schema) -> None:
        """Mixed: static + dynamic + schema-changing dynamic → correct layers."""
        steps = [
            _BuiltinStatic(),  # static, same schema
            _BuiltinDynamic(),  # dynamic, same schema
            _SchemaChangingDynamic(),  # dynamic, changes schema → boundary
            _BuiltinStatic(),  # static, new layer
        ]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 2
        assert len(plan.layers[0].steps) == 3
        assert len(plan.layers[1].steps) == 1

    def test_single_step(self, schema: Schema) -> None:
        """Single step → one layer."""
        steps = [_BuiltinDynamic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 1
        assert len(plan.layers[0].steps) == 1

    def test_empty_steps_raises(self, schema: Schema) -> None:
        """Empty steps → CompilationError."""
        with pytest.raises(CompilationError):
            plan_fit([], schema)

    def test_layer_schema_propagation(self, schema: Schema) -> None:
        """Layer 1's input_schema equals last step of Layer 0's output."""
        steps = [_SchemaChangingDynamic(), _BuiltinDynamic()]
        plan = plan_fit(steps, schema)
        assert len(plan.layers) == 2
        # Layer 1's input schema should have the new columns
        layer1_input_cols = set(plan.layers[1].input_schema.columns.keys())
        assert "price_new" in layer1_input_cols
        assert "city_new" in layer1_input_cols


# --- build_fit_queries tests ---

from sqlearn.core.compiler import Layer, build_fit_queries  # noqa: E402


class TestBuildFitQueries:
    """Test build_fit_queries aggregation batching."""

    def _make_layer(self, steps: list[Transformer], schema: Schema) -> Layer:
        """Helper: create a Layer via plan_fit (single layer)."""
        plan = plan_fit(steps, schema)
        return plan.layers[0]

    def _bare_exprs(self, schema: Schema) -> dict[str, exp.Expression]:
        """Helper: bare column references."""
        return {c: exp.Column(this=c) for c in schema.columns}

    def test_single_dynamic_step(self, schema: Schema) -> None:
        """Single dynamic step → one aggregate query."""
        layer = self._make_layer([_BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is not None
        sql = result.aggregate_query.sql(dialect="duckdb")
        assert "AVG" in sql

    def test_multiple_dynamic_batched(self, schema: Schema) -> None:
        """Multiple dynamic steps → batched into one query."""
        layer = self._make_layer([_BuiltinDynamic(), _BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is not None
        # Both steps contribute to param_mapping — step indices 0 and 1
        step_indices = {v[0] for v in result.param_mapping.values()}
        assert len(step_indices) == 2, f"Expected 2 contributing steps, got {step_indices}"

    def test_all_static_layer(self, schema: Schema) -> None:
        """All static layer → aggregate_query is None."""
        layer = self._make_layer([_BuiltinStatic(), _BuiltinStatic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is None

    def test_discover_sets_produces_set_queries(self, schema: Schema) -> None:
        """Step with discover_sets() → set_queries entry."""
        layer = self._make_layer([_CustomWithSets()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert len(result.set_queries) > 0

    def test_expression_inlining(self, schema: Schema) -> None:
        """Static before dynamic → static expr composed into aggregation."""
        # _BuiltinStatic multiplies by 2, _BuiltinDynamic computes AVG
        # So the aggregation should be AVG(col * 2), not AVG(col)
        layer = self._make_layer([_BuiltinStatic(), _BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is not None
        sql = result.aggregate_query.sql(dialect="duckdb")
        assert "AVG" in sql
        # Verify inlining: the multiplication (from static step) should be
        # INSIDE the AVG, not a bare column reference
        assert "* 2" in sql or "2 *" in sql

    def test_multiple_set_queries(self, schema: Schema) -> None:
        """Multiple steps with discover_sets() → one entry per step."""

        class _SetStep1(Transformer):
            """First set step."""

            _default_columns = "all"
            _classification = "dynamic"

            def discover_sets(
                self, columns: list[str], schema: Schema, y_column: str | None = None
            ) -> dict[str, exp.Expression]:
                """Return set query a."""
                return {"set_a": exp.select("city").from_("data").distinct()}

            def expressions(
                self, columns: list[str], exprs: dict[str, exp.Expression]
            ) -> dict[str, exp.Expression]:
                """Noop."""
                return {c: exprs[c] for c in columns}

        class _SetStep2(Transformer):
            """Second set step."""

            _default_columns = "all"
            _classification = "dynamic"

            def discover_sets(
                self, columns: list[str], schema: Schema, y_column: str | None = None
            ) -> dict[str, exp.Expression]:
                """Return set query b."""
                return {"set_b": exp.select("price").from_("data").distinct()}

            def expressions(
                self, columns: list[str], exprs: dict[str, exp.Expression]
            ) -> dict[str, exp.Expression]:
                """Noop."""
                return {c: exprs[c] for c in columns}

        _SetStep1.__module__ = "sqlearn.fake"
        _SetStep2.__module__ = "sqlearn.fake"

        layer = self._make_layer([_SetStep1(), _SetStep2()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert len(result.set_queries) == 2

    def test_param_mapping(self, schema: Schema) -> None:
        """param_mapping maps aliases back to step names."""
        layer = self._make_layer([_BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        # Each entry should be (step_index_str, param_name)
        for alias, (step_idx, param_name) in result.param_mapping.items():
            assert isinstance(alias, str)
            assert isinstance(step_idx, str)
            assert isinstance(param_name, str)

    def test_empty_discover_skipped(self, schema: Schema) -> None:
        """Dynamic step with empty discover() → no aggregations from that step."""
        layer = self._make_layer([_CustomDeclaredDynamicEmpty()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        # declared dynamic but discover returns {} → no aggregate query needed
        assert result.aggregate_query is None

    def test_mixed_static_dynamic(self, schema: Schema) -> None:
        """Mixed layer → only dynamic contribute to aggregation."""
        layer = self._make_layer([_BuiltinStatic(), _BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is not None

    def test_source_as_string(self, schema: Schema) -> None:
        """Source as plain string works."""
        layer = self._make_layer([_BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        result = build_fit_queries(layer, "data", exprs)
        assert result.aggregate_query is not None
        sql = result.aggregate_query.sql(dialect="duckdb")
        assert "data" in sql

    def test_source_as_expression(self, schema: Schema) -> None:
        """Source as sqlglot expression works."""
        layer = self._make_layer([_BuiltinDynamic()], schema)
        exprs = self._bare_exprs(schema)
        source_expr = exp.to_table("my_view")
        result = build_fit_queries(layer, source_expr, exprs)
        assert result.aggregate_query is not None
        sql = result.aggregate_query.sql(dialect="duckdb")
        assert "my_view" in sql


# --- Mock transformers for compose_transform tests ---


class _QueryStep(Transformer):
    """Transformer that uses query() instead of expressions()."""

    _classification = "static"
    _default_columns = "all"

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Wrap input in a SELECT with window function."""
        return exp.select(
            "*",
            exp.Window(
                this=exp.Avg(this=exp.Column(this="price")),
                partition_by=[exp.Column(this="city")],
            ).as_("price_avg"),
        ).from_(input_query.subquery("__sub"))

    def output_schema(self, schema: Schema) -> Schema:
        """Add price_avg column if not already present."""
        if "price_avg" in schema.columns:
            return schema
        return schema.add({"price_avg": "DOUBLE"})

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Not used — query() takes precedence."""
        return {c: exprs[c] for c in columns}


_QueryStep.__module__ = "sqlearn.features.fake"


class _CollisionStep(Transformer):
    """Transformer that adds __collision__ column — two of these cause SchemaError."""

    _classification = "static"
    _default_columns = "all"

    def expressions(
        self, columns: list[str], exprs: dict[str, exp.Expression]
    ) -> dict[str, exp.Expression]:
        """Return same columns plus __collision__."""
        result = {c: exprs[c] for c in columns}
        result["__collision__"] = exp.Literal.number(1)
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Add collision column if not already present."""
        if "__collision__" in schema.columns:
            return schema
        return schema.add({"__collision__": "INTEGER"})


_CollisionStep.__module__ = "sqlearn.ops.fake"


class _NonAstStep(Transformer):
    """Transformer where expressions() returns a string (invalid)."""

    _classification = "static"
    _default_columns = "all"

    def expressions(self, columns: list[str], exprs: dict[str, exp.Expression]) -> Any:
        """Returns strings instead of AST nodes."""
        return dict.fromkeys(columns, "bad_string")


_NonAstStep.__module__ = "sqlearn.ops.fake"


def _fit_step(step: Transformer, schema: Schema) -> None:
    """Helper: minimally fit a step so compose_transform works."""
    col_spec = step._resolve_columns_spec()  # pyright: ignore[reportPrivateUsage]
    if col_spec is None:
        columns = list(schema.columns.keys())
    else:
        columns = resolve_columns(schema, col_spec)
    step.columns_ = columns
    step.input_schema_ = schema
    step.output_schema_ = step.output_schema(schema)
    step._fitted = True  # pyright: ignore[reportPrivateUsage]
    step.params_ = {}
    step.sets_ = {}


class TestComposeTransform:
    """Test compose_transform expression composition."""

    def test_single_expression_step(self, schema: Schema) -> None:
        """Single expression step → SELECT with composed expressions."""
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        assert "FROM" in sql
        assert "data" in sql

    def test_two_steps_nested(self, schema: Schema) -> None:
        """Two expression steps → nested composition."""
        s1 = _BuiltinStatic()
        s2 = _BuiltinStatic()
        _fit_step(s1, schema)
        _fit_step(s2, schema)
        result = compose_transform([s1, s2], "data")
        sql = result.sql(dialect="duckdb")
        assert "FROM" in sql
        # Both steps multiply by 2, so expect nested multiplication in output
        assert sql.count("*") >= 2 or "2" in sql

    def test_query_step_cte(self, schema: Schema) -> None:
        """query() step → CTE promotion."""
        step = _QueryStep()
        _fit_step(step, schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        # CTE should be present
        assert result.find(exp.With) is not None or "__cte_" in sql

    def test_expression_query_expression(self, schema: Schema) -> None:
        """Expression → query → expression → CTE in middle."""
        s1 = _BuiltinStatic()
        s2 = _QueryStep()
        s3 = _BuiltinStatic()
        output_schema = schema.add({"price_avg": "DOUBLE"})
        _fit_step(s1, schema)
        _fit_step(s2, schema)
        _fit_step(s3, output_schema)
        result = compose_transform([s1, s2, s3], "data")
        sql = result.sql(dialect="duckdb")
        assert isinstance(result, exp.Select)
        # CTE should be present from the query() step
        assert result.find(exp.With) is not None or "__cte_" in sql

    def test_multiple_query_steps(self, schema: Schema) -> None:
        """Multiple query() steps → chained CTEs."""
        s1 = _QueryStep()
        s2 = _QueryStep()
        output_schema = schema.add({"price_avg": "DOUBLE"})
        _fit_step(s1, schema)
        _fit_step(s2, output_schema)
        result = compose_transform([s1, s2], "data")
        sql = result.sql(dialect="duckdb")
        assert isinstance(result, exp.Select)
        # Should have 2 CTEs
        assert sql.count("__cte_") >= 2

    def test_passthrough(self, schema: Schema) -> None:
        """Unmodified columns pass through in output."""
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        assert "price" in sql

    def test_empty_pipeline_raises(self) -> None:
        """Empty pipeline → CompilationError."""
        with pytest.raises(CompilationError):
            compose_transform([], "data")

    def test_all_static_no_ctes(self, schema: Schema) -> None:
        """All-static pipeline → valid SELECT, no CTEs."""
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        assert "WITH" not in sql

    def test_source_in_from(self, schema: Schema) -> None:
        """Source name appears in FROM clause."""
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "my_table")
        sql = result.sql(dialect="duckdb")
        assert "my_table" in sql

    def test_aliases_match_output(self, schema: Schema) -> None:
        """Final SELECT aliases match output column names."""
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        # Aliases should include original column names
        assert "price" in sql
        assert "city" in sql

    def test_schema_changing_step(self, schema: Schema) -> None:
        """Schema-changing step: new columns appear in output."""
        step = _SchemaChangingStatic()
        _fit_step(step, schema)
        # output_schema adds *_log columns
        step.output_schema_ = step.output_schema(schema)
        result = compose_transform([step], "data")
        sql = result.sql(dialect="duckdb")
        # New columns from output_schema should appear
        assert "price_log" in sql or "city_log" in sql

    def test_auto_cte_at_depth(self, schema: Schema) -> None:
        """Auto CTE when expression depth exceeds threshold."""
        # Use cte_depth=1 to trigger auto-CTE easily
        step = _BuiltinStatic()
        _fit_step(step, schema)
        result = compose_transform([step], "data", cte_depth=1)
        sql = result.sql(dialect="duckdb")
        # Low threshold should force CTE extraction
        assert "WITH" in sql or "__cte_" in sql

    def test_depth_tracking(self) -> None:
        """Depth is measured correctly across nested expressions."""
        from sqlearn.core.compiler import (  # pyright: ignore[reportPrivateUsage]
            _expression_depth,
            _max_depth,
        )

        # Bare column: depth 1
        bare = exp.Column(this="price")
        assert _expression_depth(bare) >= 1

        # Nested: depth increases
        nested = exp.Mul(this=bare, expression=exp.Literal.number(2))
        assert _expression_depth(nested) > _expression_depth(bare)

        # _max_depth across dict
        exprs = {"a": bare, "b": nested}
        assert _max_depth(exprs) == _expression_depth(nested)

    def test_column_collision_raises(self, schema: Schema) -> None:
        """Column name collision → SchemaError."""
        # Two steps that both add "__collision__" — second step triggers collision
        s1 = _CollisionStep()
        s2 = _CollisionStep()
        _fit_step(s1, schema)
        out_schema = s1.output_schema(schema)
        _fit_step(s2, out_schema)
        with pytest.raises(SchemaError, match="__collision__"):
            compose_transform([s1, s2], "data")

    def test_non_ast_expressions_raises(self, schema: Schema) -> None:
        """expressions() returns non-AST → CompilationError."""
        step = _NonAstStep()
        _fit_step(step, schema)
        with pytest.raises(CompilationError):
            compose_transform([step], "data")

    def test_unfitted_step_raises(self, schema: Schema) -> None:
        """Unfitted step (columns_ is None) → CompilationError."""
        step = _BuiltinDynamic()
        # Leave columns_ as None (unfitted) — compose_transform checks this
        with pytest.raises(CompilationError, match="columns_"):
            compose_transform([step], "data")
