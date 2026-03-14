"""Tests for sqlearn.core.compiler."""

from __future__ import annotations

from typing import Any

import pytest
import sqlglot.expressions as exp

from sqlearn.core.compiler import (
    classify_step,
    detect_schema_change,
    plan_fit,
)
from sqlearn.core.errors import ClassificationError, CompilationError
from sqlearn.core.schema import Schema
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
