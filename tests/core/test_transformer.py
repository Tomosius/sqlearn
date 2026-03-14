"""Tests for sqlearn.core.transformer."""

from __future__ import annotations

import pytest
import sqlglot.expressions as exp

from sqlearn.core.schema import ColumnSelector, Schema, numeric
from sqlearn.core.transformer import Transformer

# ---------------------------------------------------------------------------
# Test helpers — concrete subclasses for testing
# ---------------------------------------------------------------------------


class _StaticTransformer(Transformer):
    """Minimal static transformer for testing."""

    _default_columns = "numeric"
    _classification = "static"

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {}


class _DynamicTransformer(Transformer):
    """Dynamic transformer with custom __init__ params."""

    _default_columns = "numeric"
    _classification = "dynamic"

    def __init__(
        self,
        *,
        scale: float = 1.0,
        columns: str | list[str] | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.scale = scale

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        return {f"{col}__mean": exp.Avg(this=exp.column(col)) for col in columns}

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {
            col: exp.Sub(
                this=exprs[col],
                expression=exp.Literal.number(self.scale),
            )
            for col in columns
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransformerInit:
    """Test Transformer.__init__ and instance attributes."""

    def test_default_init(self) -> None:
        t = _StaticTransformer()
        assert t.columns is None
        assert t._fitted is False
        assert t._owner_thread is None
        assert t._owner_pid is None
        assert t._connection is None

    def test_columns_stored(self) -> None:
        t = _StaticTransformer(columns=["price"])
        assert t.columns == ["price"]

    def test_columns_string(self) -> None:
        t = _StaticTransformer(columns="numeric")
        assert t.columns == "numeric"

    def test_fitted_attributes_none(self) -> None:
        t = _StaticTransformer()
        assert t.params_ is None
        assert t.sets_ is None
        assert t.columns_ is None
        assert t.input_schema_ is None
        assert t.output_schema_ is None
        assert t._y_column is None

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            _StaticTransformer("numeric")  # type: ignore[misc]

    def test_subclass_init(self) -> None:
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        assert t.scale == 2.0
        assert t.columns == ["price"]
        assert t._fitted is False


class TestTransformerIsFitted:
    """Test is_fitted property and __sklearn_is_fitted__."""

    def test_is_fitted_false(self) -> None:
        t = _StaticTransformer()
        assert t.is_fitted is False

    def test_is_fitted_after_manual_set(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        assert t.is_fitted is True

    def test_sklearn_is_fitted(self) -> None:
        t = _StaticTransformer()
        assert t.__sklearn_is_fitted__() is False
        t._fitted = True
        assert t.__sklearn_is_fitted__() is True


class TestTransformerOverrides:
    """Test default implementations of subclass override methods."""

    def test_discover_default_empty(self) -> None:
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        assert t.discover(["price"], schema) == {}

    def test_discover_sets_default_empty(self) -> None:
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        assert t.discover_sets(["price"], schema) == {}

    def test_expressions_raises(self) -> None:
        t = Transformer()
        with pytest.raises(NotImplementedError):
            t.expressions(["price"], {"price": exp.column("price")})

    def test_expressions_subclass_override(self) -> None:
        t = _StaticTransformer()
        result = t.expressions(["price"], {"price": exp.column("price")})
        assert result == {}

    def test_query_default_none(self) -> None:
        t = _StaticTransformer()
        assert t.query(exp.select("*")) is None

    def test_output_schema_default_passthrough(self) -> None:
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert t.output_schema(schema) is schema

    def test_discover_subclass_override(self) -> None:
        t = _DynamicTransformer()
        schema = Schema({"price": "DOUBLE"})
        result = t.discover(["price"], schema)
        assert "price__mean" in result
        assert isinstance(result["price__mean"], exp.Avg)

    def test_discover_with_y_column(self) -> None:
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        assert t.discover(["price"], schema, y_column="target") == {}


# ---------------------------------------------------------------------------
# Task 2 helpers
# ---------------------------------------------------------------------------


class _AutoDetectDynamic(Transformer):
    """No _classification — overrides discover(), should auto-detect as dynamic."""

    _default_columns = "all"

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        return {f"{col}__count": exp.Count(this=exp.column(col)) for col in columns}

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {}


class _AutoDetectStatic(Transformer):
    """No _classification — no discover() override, should auto-detect as static."""

    _default_columns = "all"

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {}


class _AutoDetectDynamicSets(Transformer):
    """No _classification — overrides discover_sets(), should auto-detect as dynamic."""

    _default_columns = "all"

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        return {"categories": exp.select(exp.column("col")).from_("t")}

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {}


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------


class TestResolveColumnsSpec:
    """Test _resolve_columns_spec() method."""

    def test_user_override_takes_precedence(self) -> None:
        t = _StaticTransformer(columns=["price"])
        assert t._resolve_columns_spec() == ["price"]

    def test_falls_back_to_default(self) -> None:
        t = _StaticTransformer()
        assert t._resolve_columns_spec() == "numeric"

    def test_none_when_no_default(self) -> None:
        t = Transformer()
        assert t._resolve_columns_spec() is None

    def test_user_string_override(self) -> None:
        t = _StaticTransformer(columns="all")
        assert t._resolve_columns_spec() == "all"

    def test_column_selector_override(self) -> None:
        sel = numeric()
        t = _StaticTransformer(columns=sel)
        result = t._resolve_columns_spec()
        assert isinstance(result, ColumnSelector)
        assert result is sel


class TestClassify:
    """Test _classify() static/dynamic detection."""

    def test_tier1_static(self) -> None:
        t = _StaticTransformer()
        assert t._classify() == "static"

    def test_tier1_dynamic(self) -> None:
        t = _DynamicTransformer()
        assert t._classify() == "dynamic"

    def test_tier3_auto_dynamic_discover(self) -> None:
        t = _AutoDetectDynamic()
        assert t._classify() == "dynamic"

    def test_tier3_auto_dynamic_discover_sets(self) -> None:
        t = _AutoDetectDynamicSets()
        assert t._classify() == "dynamic"

    def test_tier3_auto_static(self) -> None:
        t = _AutoDetectStatic()
        assert t._classify() == "static"

    def test_base_transformer_auto_static(self) -> None:
        t = Transformer()
        assert t._classify() == "static"


class TestGetParams:
    """Test get_params() sklearn-compatible introspection."""

    def test_base_transformer(self) -> None:
        t = _StaticTransformer()
        params = t.get_params()
        assert params == {"columns": None}

    def test_with_columns(self) -> None:
        t = _StaticTransformer(columns=["price"])
        params = t.get_params()
        assert params == {"columns": ["price"]}

    def test_subclass_params(self) -> None:
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        params = t.get_params()
        assert params == {"scale": 2.0, "columns": ["price"]}

    def test_subclass_defaults(self) -> None:
        t = _DynamicTransformer()
        params = t.get_params()
        assert params == {"scale": 1.0, "columns": None}


class TestSetParams:
    """Test set_params() sklearn-compatible parameter setting."""

    def test_set_columns(self) -> None:
        t = _StaticTransformer()
        result = t.set_params(columns=["price"])
        assert t.columns == ["price"]
        assert result is t

    def test_set_subclass_param(self) -> None:
        t = _DynamicTransformer()
        t.set_params(scale=3.0)
        assert t.scale == 3.0

    def test_set_multiple(self) -> None:
        t = _DynamicTransformer()
        t.set_params(scale=5.0, columns=["qty"])
        assert t.scale == 5.0
        assert t.columns == ["qty"]

    def test_invalid_param_raises(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(ValueError, match="Invalid parameter"):
            t.set_params(nonexistent=True)

    def test_roundtrip(self) -> None:
        t = _DynamicTransformer(scale=2.5, columns=["a", "b"])
        params = t.get_params()
        t2 = _DynamicTransformer()
        t2.set_params(**params)
        assert t2.get_params() == params


class TestTransformerRepr:
    """Test __repr__ and _repr_html_ display methods."""

    def test_repr_no_params(self) -> None:
        t = _StaticTransformer()
        assert repr(t) == "_StaticTransformer()"

    def test_repr_with_columns(self) -> None:
        t = _StaticTransformer(columns=["price"])
        assert repr(t) == "_StaticTransformer(columns=['price'])"

    def test_repr_subclass(self) -> None:
        t = _DynamicTransformer(scale=2.0)
        assert repr(t) == "_DynamicTransformer(scale=2.0)"

    def test_repr_all_defaults(self) -> None:
        t = _DynamicTransformer()
        assert repr(t) == "_DynamicTransformer()"

    def test_repr_multiple_non_defaults(self) -> None:
        t = _DynamicTransformer(scale=3.0, columns="all")
        r = repr(t)
        assert "scale=3.0" in r
        assert "columns='all'" in r

    def test_repr_html(self) -> None:
        t = _StaticTransformer()
        html = t._repr_html_()
        assert "_StaticTransformer" in html
        assert "not fitted" in html

    def test_repr_html_fitted(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        html = t._repr_html_()
        assert "fitted" in html
        assert "not fitted" not in html


class TestGetFeatureNamesOut:
    """Test get_feature_names_out() method."""

    def test_returns_column_names(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        t.output_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert t.get_feature_names_out() == ["price", "city"]

    def test_not_fitted_raises(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(ValueError, match="not fitted"):
            t.get_feature_names_out()

    def test_preserves_order(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        t.output_schema_ = Schema({"b": "INT", "a": "VARCHAR", "c": "DOUBLE"})
        assert t.get_feature_names_out() == ["b", "a", "c"]
