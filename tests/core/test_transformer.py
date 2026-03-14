"""Tests for sqlearn.core.transformer."""

from __future__ import annotations

import os
import pickle
import threading

import pytest
import sqlglot.expressions as exp

from sqlearn.core.errors import NotFittedError
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
# Task 6 helpers — _apply_expressions tests
# ---------------------------------------------------------------------------


class _AddColumnTransformer(Transformer):
    """Transformer that adds a new column (for _apply_expressions tests)."""

    _default_columns = "numeric"
    _classification = "static"

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        result: dict[str, exp.Expression] = {}
        for col in columns:
            result[col] = exprs[col]
            result[f"{col}_doubled"] = exp.Mul(
                this=exprs[col],
                expression=exp.Literal.number(2),
            )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare added columns in output."""
        new_cols = {f"{col}_doubled": "DOUBLE" for col in schema.numeric()}
        return schema.add(new_cols)


class _UndeclaredColumnTransformer(Transformer):
    """Transformer that adds a column without declaring it in output_schema."""

    _default_columns = "numeric"
    _classification = "static"

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        return {"secret_col": exp.Literal.number(42)}

    # Does NOT override output_schema — secret_col is undeclared


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
        with pytest.raises(NotFittedError, match="not fitted"):
            t.get_feature_names_out()

    def test_preserves_order(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        t.output_schema_ = Schema({"b": "INT", "a": "VARCHAR", "c": "DOUBLE"})
        assert t.get_feature_names_out() == ["b", "a", "c"]


class TestCheckThread:
    """Test _check_thread() thread/process safety guard."""

    def test_first_call_sets_owner(self) -> None:
        t = _StaticTransformer()
        t._check_thread()
        assert t._owner_thread == threading.current_thread().ident
        assert t._owner_pid == os.getpid()

    def test_same_thread_ok(self) -> None:
        t = _StaticTransformer()
        t._check_thread()
        t._check_thread()  # should not raise

    def test_different_thread_raises(self) -> None:
        t = _StaticTransformer()
        t._check_thread()  # set owner to main thread

        error: BaseException | None = None

        def call_from_thread() -> None:
            nonlocal error
            try:
                t._check_thread()
            except RuntimeError as e:
                error = e

        thread = threading.Thread(target=call_from_thread)
        thread.start()
        thread.join()
        assert error is not None
        assert "thread" in str(error).lower()
        assert "clone" in str(error).lower()

    def test_different_pid_raises(self) -> None:
        t = _StaticTransformer()
        t._check_thread()
        t._owner_pid = -1  # simulate different process
        with pytest.raises(RuntimeError, match="process"):
            t._check_thread()


class TestClone:
    """Test clone() creates independent thread-safe copy."""

    def test_clone_unfitted(self) -> None:
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        c = t.clone()
        assert c.get_params() == t.get_params()
        assert c is not t

    def test_clone_fitted_state(self) -> None:
        t = _DynamicTransformer()
        t._fitted = True
        t.params_ = {"price__mean": 42.0}
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        t.output_schema_ = Schema({"price": "DOUBLE"})
        c = t.clone()
        assert c._fitted is True
        assert c.params_ == {"price__mean": 42.0}
        assert c.columns_ == ["price"]

    def test_clone_deep_copies_params(self) -> None:
        t = _DynamicTransformer()
        t._fitted = True
        t.params_ = {"price__mean": 42.0}
        c = t.clone()
        c.params_["price__mean"] = 99.0
        assert t.params_["price__mean"] == 42.0  # original unchanged

    def test_clone_resets_thread_owner(self) -> None:
        t = _StaticTransformer()
        t._check_thread()
        c = t.clone()
        assert c._owner_thread is None
        assert c._owner_pid is None

    def test_clone_type_preserved(self) -> None:
        t = _DynamicTransformer(scale=3.0)
        c = t.clone()
        assert type(c) is _DynamicTransformer
        assert c.scale == 3.0


class TestCopy:
    """Test copy() deep copy."""

    def test_copy_creates_independent_instance(self) -> None:
        t = _DynamicTransformer(scale=2.0)
        c = t.copy()
        assert c is not t
        assert c.scale == 2.0

    def test_copy_deep_copies_params(self) -> None:
        t = _DynamicTransformer()
        t._fitted = True
        t.params_ = {"x": 1.0}
        c = t.copy()
        c.params_["x"] = 99.0
        assert t.params_["x"] == 1.0


class TestSerialization:
    """Test __getstate__ / __setstate__ for pickle."""

    def test_pickle_roundtrip(self) -> None:
        t = _DynamicTransformer(scale=2.0)
        t._fitted = True
        t.params_ = {"price__mean": 42.0}
        data = pickle.dumps(t)
        t2 = pickle.loads(data)  # noqa: S301
        assert t2.scale == 2.0
        assert t2.params_ == {"price__mean": 42.0}
        assert t2._fitted is True

    def test_pickle_nulls_connection(self) -> None:
        t = _StaticTransformer()
        t._connection = "fake_connection"
        data = pickle.dumps(t)
        t2 = pickle.loads(data)  # noqa: S301
        assert t2._connection is None

    def test_pickle_preserves_type(self) -> None:
        t = _DynamicTransformer(scale=3.0)
        t2 = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert type(t2) is _DynamicTransformer


class TestApplyExpressions:
    """Test _apply_expressions() base class wrapper."""

    def test_passthrough_unmodified(self) -> None:
        t = _StaticTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        exprs = {
            "price": exp.column("price"),
            "city": exp.column("city"),
        }
        result = t._apply_expressions(exprs)
        assert "price" in result
        assert "city" in result

    def test_modified_columns_merged(self) -> None:
        t = _DynamicTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        result = t._apply_expressions(exprs)
        assert "price" in result
        assert isinstance(result["price"], exp.Sub)

    def test_new_columns_added(self) -> None:
        t = _AddColumnTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        result = t._apply_expressions(exprs)
        assert "price" in result
        assert "price_doubled" in result

    def test_output_schema_filters_columns(self) -> None:
        t = _UndeclaredColumnTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        with pytest.warns(UserWarning, match="output_schema"):
            result = t._apply_expressions(exprs)
        assert "secret_col" not in result

    def test_not_fitted_raises(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotFittedError, match="columns_"):
            t._apply_expressions({"price": exp.column("price")})


class TestOperators:
    """Test __add__ and __iadd__ pipeline composition."""

    def test_add_raises_not_implemented(self) -> None:
        a = _StaticTransformer()
        b = _DynamicTransformer()
        with pytest.raises(NotImplementedError, match="Pipeline"):
            _ = a + b

    def test_iadd_raises_not_implemented(self) -> None:
        a = _StaticTransformer()
        b = _DynamicTransformer()
        with pytest.raises(NotImplementedError, match="Pipeline"):
            a += b


class TestStubs:
    """Test stub methods raise NotImplementedError."""

    def test_fit_stub(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.fit("data.parquet")

    def test_transform_stub(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.transform("data.parquet")

    def test_fit_transform_stub(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.fit_transform("data.parquet")

    def test_to_sql_stub(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.to_sql()

    def test_freeze_stub(self) -> None:
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.freeze()

    def test_fit_signature(self) -> None:
        import inspect

        sig = inspect.signature(Transformer.fit)
        params = list(sig.parameters.keys())
        assert "data" in params
        assert "y" in params
        assert "backend" in params

    def test_transform_signature(self) -> None:
        import inspect

        sig = inspect.signature(Transformer.transform)
        params = list(sig.parameters.keys())
        assert "data" in params
        assert "out" in params
        assert "backend" in params
        assert "batch_size" in params
        assert "dtype" in params
        assert "exclude_target" in params


class TestExports:
    """Test that Transformer is exported correctly."""

    def test_from_core(self) -> None:
        import sqlearn.core

        assert sqlearn.core.Transformer is Transformer

    def test_from_package(self) -> None:
        import sqlearn

        assert sqlearn.Transformer is Transformer
