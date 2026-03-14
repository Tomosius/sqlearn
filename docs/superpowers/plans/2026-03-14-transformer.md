# Transformer Base Class Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Transformer base class that all sqlearn transformers extend, with subclass override methods, sklearn-compatible introspection, thread safety, and pipeline composition stubs.

**Architecture:** Single class in `src/sqlearn/core/transformer.py` with ~25 methods. Subclasses override `discover()`, `expressions()`, and optionally `discover_sets()`, `query()`, `output_schema()` to define behavior. `fit()`/`transform()`/`to_sql()` are stubs until Pipeline (#7) and Compiler (#6) land.

**Tech Stack:** Python 3.10+, sqlglot (TYPE_CHECKING only), sqlearn.core.schema (Schema, ColumnSelector, resolve_columns)

**Spec:** `docs/superpowers/specs/2026-03-14-transformer-design.md`

**Spec deviation:** The spec uses `_columns_spec` for the stored `columns=` parameter. This plan uses `self.columns` instead (matching the `__init__` parameter name) so that `get_params()` works via simple `getattr(self, param_name)` — the standard sklearn convention. The fitted attribute `self.columns_` (trailing underscore) holds the resolved column list.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/sqlearn/core/transformer.py` | Transformer base class — all methods |
| Create | `tests/core/test_transformer.py` | Tests for Transformer |
| Modify | `src/sqlearn/core/__init__.py` | Add Transformer export |
| Modify | `src/sqlearn/__init__.py` | Add Transformer export |

---

## Chunk 1: Foundation

### Task 1: Class skeleton, __init__, properties, and subclass overrides

**Files:**
- Create: `src/sqlearn/core/transformer.py`
- Create: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_transformer.py
"""Tests for sqlearn.core.transformer."""

from __future__ import annotations

import inspect

import sqlglot.expressions as exp
import pytest

from sqlearn.core.schema import ColumnSelector, Schema
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
        """Base Transformer can be instantiated with defaults."""
        t = _StaticTransformer()
        assert t.columns is None
        assert t._fitted is False
        assert t._owner_thread is None
        assert t._owner_pid is None
        assert t._connection is None

    def test_columns_stored(self) -> None:
        """columns= parameter is stored."""
        t = _StaticTransformer(columns=["price"])
        assert t.columns == ["price"]

    def test_columns_string(self) -> None:
        """columns= accepts string literals."""
        t = _StaticTransformer(columns="numeric")
        assert t.columns == "numeric"

    def test_fitted_attributes_none(self) -> None:
        """All fitted attributes start as None."""
        t = _StaticTransformer()
        assert t.params_ is None
        assert t.sets_ is None
        assert t.columns_ is None
        assert t.input_schema_ is None
        assert t.output_schema_ is None
        assert t._y_column is None

    def test_keyword_only(self) -> None:
        """columns must be keyword-only."""
        with pytest.raises(TypeError):
            _StaticTransformer("numeric")  # type: ignore[misc]

    def test_subclass_init(self) -> None:
        """Subclass __init__ stores its own params via super()."""
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        assert t.scale == 2.0
        assert t.columns == ["price"]
        assert t._fitted is False


class TestTransformerIsFitted:
    """Test is_fitted property and __sklearn_is_fitted__."""

    def test_is_fitted_false(self) -> None:
        """is_fitted starts False."""
        t = _StaticTransformer()
        assert t.is_fitted is False

    def test_is_fitted_after_manual_set(self) -> None:
        """is_fitted reflects _fitted state."""
        t = _StaticTransformer()
        t._fitted = True
        assert t.is_fitted is True

    def test_sklearn_is_fitted(self) -> None:
        """__sklearn_is_fitted__ matches is_fitted."""
        t = _StaticTransformer()
        assert t.__sklearn_is_fitted__() is False
        t._fitted = True
        assert t.__sklearn_is_fitted__() is True


class TestTransformerOverrides:
    """Test default implementations of subclass override methods."""

    def test_discover_default_empty(self) -> None:
        """Base discover() returns empty dict."""
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        assert t.discover(["price"], schema) == {}

    def test_discover_sets_default_empty(self) -> None:
        """Base discover_sets() returns empty dict."""
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        assert t.discover_sets(["price"], schema) == {}

    def test_expressions_raises(self) -> None:
        """Base expressions() raises NotImplementedError."""
        t = Transformer()
        with pytest.raises(NotImplementedError):
            t.expressions(["price"], {"price": exp.column("price")})

    def test_expressions_subclass_override(self) -> None:
        """Subclass can override expressions()."""
        t = _StaticTransformer()
        result = t.expressions(["price"], {"price": exp.column("price")})
        assert result == {}

    def test_query_default_none(self) -> None:
        """Base query() returns None."""
        t = _StaticTransformer()
        assert t.query(exp.select("*")) is None

    def test_output_schema_default_passthrough(self) -> None:
        """Base output_schema() returns input schema unchanged."""
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert t.output_schema(schema) is schema

    def test_discover_subclass_override(self) -> None:
        """Subclass discover() returns aggregates."""
        t = _DynamicTransformer()
        schema = Schema({"price": "DOUBLE"})
        result = t.discover(["price"], schema)
        assert "price__mean" in result
        assert isinstance(result["price__mean"], exp.Avg)

    def test_discover_with_y_column(self) -> None:
        """discover() receives y_column parameter."""
        t = _StaticTransformer()
        schema = Schema({"price": "DOUBLE"})
        # Default ignores y_column, just verify it accepts it
        assert t.discover(["price"], schema, y_column="target") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sqlearn.core.transformer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sqlearn/core/transformer.py
"""Transformer base class for all sqlearn transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlearn.core.schema import ColumnSelector, Schema

if TYPE_CHECKING:
    import sqlglot.expressions as exp


class Transformer:
    """Base class for all sqlearn transformers.

    Subclasses override discover(), expressions(), and optionally
    discover_sets(), query(), and output_schema() to define behavior.
    """

    # --- Class attributes (set by subclasses) ---
    _default_columns: str | None = None
    _classification: str | None = None

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize transformer.

        Args:
            columns: Column specification override. If provided, takes
                precedence over _default_columns. Accepts column names,
                type strings, or ColumnSelector objects. Resolved against
                schema at fit time.
        """
        # Init params (stored as-is for get_params compatibility)
        self.columns = columns

        # Internal state
        self._fitted: bool = False
        self._owner_thread: int | None = None
        self._owner_pid: int | None = None
        self._connection: Any = None  # DuckDB connection, lazy

        # Fitted attributes (set by Pipeline.fit)
        self.params_: dict[str, Any] | None = None
        self.sets_: dict[str, list[dict[str, Any]]] | None = None
        self.columns_: list[str] | None = None
        self.input_schema_: Schema | None = None
        self.output_schema_: Schema | None = None
        self._y_column: str | None = None

    # --- Properties ---

    @property
    def is_fitted(self) -> bool:
        """Whether this transformer has been fitted.

        Returns:
            True if fit() has been called successfully.
        """
        return self._fitted

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearn compatibility: enables sklearn's check_is_fitted().

        Returns:
            True if this transformer has been fitted.
        """
        return self._fitted

    # --- Subclass overrides ---

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn scalar statistics from data via SQL aggregates.

        Override to return {param_name: sqlglot_aggregate} mappings.
        Results are executed as SQL and stored in self.params_.

        Default returns {} (static — no learning).

        Param naming convention: '{col}__{stat}' (e.g. 'price__mean').
        Must return sqlglot AST nodes, never raw strings or Python values.

        Args:
            columns: Target columns this transformer operates on.
            schema: Current table schema.
            y_column: Target column name, if provided to fit().

        Returns:
            Mapping of parameter names to sqlglot aggregate expressions.
        """
        return {}

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn set-valued (multi-row) data via SQL queries.

        Override to return {param_name: sqlglot_select_query} mappings.
        Results are executed and stored in self.sets_ as lists of dicts.

        Default returns {} (no set learning).

        Args:
            columns: Target columns this transformer operates on.
            schema: Current table schema.
            y_column: Target column name, if provided to fit().

        Returns:
            Mapping of parameter names to sqlglot SELECT queries.
        """
        return {}

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL column expressions.

        Subclasses must override this or query(). Returns ONLY modified/new
        columns. Unmentioned columns pass through automatically via
        _apply_expressions().

        Must return sqlglot AST nodes, never raw strings.

        Args:
            columns: Target columns this transformer operates on.
            exprs: Current expression dict for ALL columns.

        Returns:
            Dict of modified/new column expressions.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression | None:
        """Generate a full query wrapping input.

        Alternative to expressions() for transforms needing query-level
        control (window functions, joins, CTEs). Returns None to fall
        back to expressions().

        Default returns None.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT wrapping the input, or None.
        """
        return None

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after this step.

        Override when adding, removing, renaming, or retyping columns.
        Default returns input schema unchanged.

        Args:
            schema: Input schema.

        Returns:
            Output schema after this transformation step.
        """
        return schema
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run linters**

Run: `make check`
Expected: All checks pass. Fix any ruff/pyright/mypy/interrogate issues.

- [ ] **Step 6: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add Transformer skeleton with init, properties, overrides"
```

---

### Task 2: _resolve_columns_spec() and _classify()

**Files:**
- Modify: `src/sqlearn/core/transformer.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_transformer.py`:

```python
# Add test helper at module level (after existing helpers):

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


# --- Tests ---


class TestResolveColumnsSpec:
    """Test _resolve_columns_spec() method."""

    def test_user_override_takes_precedence(self) -> None:
        """User columns= overrides class _default_columns."""
        t = _StaticTransformer(columns=["price"])
        assert t._resolve_columns_spec() == ["price"]

    def test_falls_back_to_default(self) -> None:
        """No user override returns class _default_columns."""
        t = _StaticTransformer()
        assert t._resolve_columns_spec() == "numeric"

    def test_none_when_no_default(self) -> None:
        """Returns None when both user and class default are None."""
        t = Transformer()
        assert t._resolve_columns_spec() is None

    def test_user_string_override(self) -> None:
        """String override works."""
        t = _StaticTransformer(columns="all")
        assert t._resolve_columns_spec() == "all"

    def test_column_selector_override(self) -> None:
        """ColumnSelector override is returned as-is."""
        from sqlearn.core.schema import numeric

        sel = numeric()
        t = _StaticTransformer(columns=sel)
        result = t._resolve_columns_spec()
        assert isinstance(result, ColumnSelector)
        assert result is sel


class TestClassify:
    """Test _classify() static/dynamic detection."""

    def test_tier1_static(self) -> None:
        """Declared static returns 'static'."""
        t = _StaticTransformer()
        assert t._classify() == "static"

    def test_tier1_dynamic(self) -> None:
        """Declared dynamic returns 'dynamic'."""
        t = _DynamicTransformer()
        assert t._classify() == "dynamic"

    def test_tier3_auto_dynamic_discover(self) -> None:
        """Auto-detect: discover() overridden -> dynamic."""
        t = _AutoDetectDynamic()
        assert t._classify() == "dynamic"

    def test_tier3_auto_dynamic_discover_sets(self) -> None:
        """Auto-detect: discover_sets() overridden -> dynamic."""
        t = _AutoDetectDynamicSets()
        assert t._classify() == "dynamic"

    def test_tier3_auto_static(self) -> None:
        """Auto-detect: no discover overrides -> static."""
        t = _AutoDetectStatic()
        assert t._classify() == "static"

    def test_base_transformer_auto_static(self) -> None:
        """Base Transformer with no overrides classifies as static."""
        t = Transformer()
        assert t._classify() == "static"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestResolveColumnsSpec tests/core/test_transformer.py::TestClassify -v`
Expected: FAIL — `AttributeError: 'Transformer' object has no attribute '_resolve_columns_spec'`

- [ ] **Step 3: Write implementation**

Add to `Transformer` class in `src/sqlearn/core/transformer.py`:

```python
    # --- Column resolution ---

    def _resolve_columns_spec(self) -> str | list[str] | ColumnSelector | None:
        """Return the effective column spec (user override or class default).

        Returns _columns_spec if user passed columns=, else _default_columns.
        Actual resolution against schema happens at fit time via
        resolve_columns().

        Returns:
            Column specification to resolve, or None.
        """
        if self.columns is not None:
            return self.columns
        return self._default_columns

    # --- Classification ---

    def _classify(self) -> str:
        """Classify this transformer as 'static' or 'dynamic'.

        Tier 1: If _classification is set, trust it.
        Tier 3: Check if discover() or discover_sets() are overridden.

        Safety rule: if in doubt, classify as dynamic.

        Returns:
            ``'static'`` or ``'dynamic'``.
        """
        # Tier 1: explicit declaration
        if self._classification is not None:
            return self._classification

        # Tier 3: auto-detect by checking method overrides
        has_discover = type(self).discover is not Transformer.discover
        has_discover_sets = type(self).discover_sets is not Transformer.discover_sets

        if has_discover or has_discover_sets:
            return "dynamic"
        return "static"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add _resolve_columns_spec() and _classify() to Transformer"
```

---

### Task 3: get_params() and set_params()

**Files:**
- Modify: `src/sqlearn/core/transformer.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_transformer.py`:

```python
class TestGetParams:
    """Test get_params() sklearn-compatible introspection."""

    def test_base_transformer(self) -> None:
        """Base Transformer returns columns param."""
        t = _StaticTransformer()
        params = t.get_params()
        assert params == {"columns": None}

    def test_with_columns(self) -> None:
        """Transformer with columns= returns it."""
        t = _StaticTransformer(columns=["price"])
        params = t.get_params()
        assert params == {"columns": ["price"]}

    def test_subclass_params(self) -> None:
        """Subclass returns its own params plus inherited."""
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        params = t.get_params()
        assert params == {"scale": 2.0, "columns": ["price"]}

    def test_subclass_defaults(self) -> None:
        """Subclass defaults are returned."""
        t = _DynamicTransformer()
        params = t.get_params()
        assert params == {"scale": 1.0, "columns": None}


class TestSetParams:
    """Test set_params() sklearn-compatible parameter setting."""

    def test_set_columns(self) -> None:
        """set_params updates columns."""
        t = _StaticTransformer()
        result = t.set_params(columns=["price"])
        assert t.columns == ["price"]
        assert result is t  # returns self

    def test_set_subclass_param(self) -> None:
        """set_params updates subclass params."""
        t = _DynamicTransformer()
        t.set_params(scale=3.0)
        assert t.scale == 3.0

    def test_set_multiple(self) -> None:
        """set_params sets multiple params at once."""
        t = _DynamicTransformer()
        t.set_params(scale=5.0, columns=["qty"])
        assert t.scale == 5.0
        assert t.columns == ["qty"]

    def test_invalid_param_raises(self) -> None:
        """set_params raises ValueError for unknown params."""
        t = _StaticTransformer()
        with pytest.raises(ValueError, match="Invalid parameter"):
            t.set_params(nonexistent=True)

    def test_roundtrip(self) -> None:
        """get_params -> set_params roundtrip preserves values."""
        t = _DynamicTransformer(scale=2.5, columns=["a", "b"])
        params = t.get_params()
        t2 = _DynamicTransformer()
        t2.set_params(**params)
        assert t2.get_params() == params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestGetParams tests/core/test_transformer.py::TestSetParams -v`
Expected: FAIL — `AttributeError: 'Transformer' object has no attribute 'get_params'`

- [ ] **Step 3: Write implementation**

Add `import inspect` at top of `src/sqlearn/core/transformer.py`, then add to the class:

```python
    # --- sklearn introspection ---

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return __init__ parameters as dict. sklearn-compatible.

        Introspects the subclass __init__ signature. Parameters are
        retrieved via getattr, matching sklearn convention.

        Args:
            deep: If True, returns params for nested transformers
                using '__' separator. Not used until Pipeline lands.

        Returns:
            Dict of parameter names to current values.
        """
        sig = inspect.signature(type(self).__init__)
        params: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> Transformer:
        """Set parameters. Returns self. sklearn-compatible.

        Args:
            **params: Parameter names and values to set.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If any parameter name is not a valid __init__ param.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                msg = f"Invalid parameter {key!r} for {type(self).__name__}"
                raise ValueError(msg)
            setattr(self, key, value)
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add get_params()/set_params() to Transformer"
```

---

### Task 4: __repr__, _repr_html_, get_feature_names_out()

**Files:**
- Modify: `src/sqlearn/core/transformer.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_transformer.py`:

```python
class TestTransformerRepr:
    """Test __repr__ and _repr_html_ display methods."""

    def test_repr_no_params(self) -> None:
        """Repr with all defaults shows empty parens."""
        t = _StaticTransformer()
        assert repr(t) == "_StaticTransformer()"

    def test_repr_with_columns(self) -> None:
        """Repr shows non-default columns."""
        t = _StaticTransformer(columns=["price"])
        assert repr(t) == "_StaticTransformer(columns=['price'])"

    def test_repr_subclass(self) -> None:
        """Repr shows subclass name and non-default params."""
        t = _DynamicTransformer(scale=2.0)
        assert repr(t) == "_DynamicTransformer(scale=2.0)"

    def test_repr_all_defaults(self) -> None:
        """Subclass with all defaults shows empty parens."""
        t = _DynamicTransformer()
        assert repr(t) == "_DynamicTransformer()"

    def test_repr_multiple_non_defaults(self) -> None:
        """Repr shows all non-default params."""
        t = _DynamicTransformer(scale=3.0, columns="all")
        r = repr(t)
        assert "scale=3.0" in r
        assert "columns='all'" in r

    def test_repr_html(self) -> None:
        """_repr_html_ returns HTML string."""
        t = _StaticTransformer()
        html = t._repr_html_()
        assert "_StaticTransformer" in html
        assert "not fitted" in html

    def test_repr_html_fitted(self) -> None:
        """_repr_html_ shows fitted state."""
        t = _StaticTransformer()
        t._fitted = True
        html = t._repr_html_()
        assert "fitted" in html
        assert "not fitted" not in html


class TestGetFeatureNamesOut:
    """Test get_feature_names_out() method."""

    def test_returns_column_names(self) -> None:
        """Returns output schema column names when fitted."""
        t = _StaticTransformer()
        t._fitted = True
        t.output_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert t.get_feature_names_out() == ["price", "city"]

    def test_not_fitted_raises(self) -> None:
        """Raises ValueError when not fitted."""
        t = _StaticTransformer()
        with pytest.raises(ValueError, match="not fitted"):
            t.get_feature_names_out()

    def test_preserves_order(self) -> None:
        """Column names preserve insertion order."""
        t = _StaticTransformer()
        t._fitted = True
        t.output_schema_ = Schema({"b": "INT", "a": "VARCHAR", "c": "DOUBLE"})
        assert t.get_feature_names_out() == ["b", "a", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestTransformerRepr tests/core/test_transformer.py::TestGetFeatureNamesOut -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `Transformer` class in `src/sqlearn/core/transformer.py`:

```python
    # --- Output ---

    def get_feature_names_out(self) -> list[str]:
        """Return output column names. Requires fitted state.

        Returns:
            List of output column names.

        Raises:
            ValueError: If not fitted or output_schema_ not set.
        """
        if not self._fitted:
            msg = f"{type(self).__name__} is not fitted"
            raise ValueError(msg)
        if self.output_schema_ is None:
            msg = "output_schema_ is not set"
            raise ValueError(msg)
        return list(self.output_schema_.columns.keys())

    # --- Display ---

    def __repr__(self) -> str:
        """Readable repr: ClassName(non_default_params).

        Shows class name and only parameters whose values differ
        from their __init__ defaults.
        """
        sig = inspect.signature(type(self).__init__)
        parts: list[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            value = getattr(self, name)
            if value != param.default:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _repr_html_(self) -> str:
        """Rich HTML repr for Jupyter notebooks.

        Shows transformer name, params, fitted state, and column routing.

        Returns:
            HTML string for Jupyter display.
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return (
            f"<div><strong>{type(self).__name__}</strong>"
            f"({params_str}) [{fitted_str}]</div>"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add repr, _repr_html_, get_feature_names_out to Transformer"
```

---

## Chunk 2: Safety, Composition, and Integration

### Task 5: _check_thread(), clone(), copy(), serialization

**Files:**
- Modify: `src/sqlearn/core/transformer.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_transformer.py`:

```python
import copy
import os
import pickle
import threading


class TestCheckThread:
    """Test _check_thread() thread/process safety guard."""

    def test_first_call_sets_owner(self) -> None:
        """First call records thread and process IDs."""
        t = _StaticTransformer()
        t._check_thread()
        assert t._owner_thread == threading.current_thread().ident
        assert t._owner_pid == os.getpid()

    def test_same_thread_ok(self) -> None:
        """Repeated calls from same thread succeed."""
        t = _StaticTransformer()
        t._check_thread()
        t._check_thread()  # should not raise

    def test_different_thread_raises(self) -> None:
        """Call from different thread raises RuntimeError."""
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
        """Simulated different PID raises RuntimeError."""
        t = _StaticTransformer()
        t._check_thread()
        t._owner_pid = -1  # simulate different process
        with pytest.raises(RuntimeError, match="process"):
            t._check_thread()


class TestClone:
    """Test clone() creates independent thread-safe copy."""

    def test_clone_unfitted(self) -> None:
        """Clone of unfitted transformer has same params."""
        t = _DynamicTransformer(scale=2.0, columns=["price"])
        c = t.clone()
        assert c.get_params() == t.get_params()
        assert c is not t

    def test_clone_fitted_state(self) -> None:
        """Clone copies fitted state."""
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
        """Clone deep copies mutable fitted state."""
        t = _DynamicTransformer()
        t._fitted = True
        t.params_ = {"price__mean": 42.0}
        c = t.clone()
        c.params_["price__mean"] = 99.0
        assert t.params_["price__mean"] == 42.0  # original unchanged

    def test_clone_resets_thread_owner(self) -> None:
        """Clone resets thread ownership."""
        t = _StaticTransformer()
        t._check_thread()
        c = t.clone()
        assert c._owner_thread is None
        assert c._owner_pid is None

    def test_clone_type_preserved(self) -> None:
        """Clone preserves the subclass type."""
        t = _DynamicTransformer(scale=3.0)
        c = t.clone()
        assert type(c) is _DynamicTransformer
        assert c.scale == 3.0


class TestCopy:
    """Test copy() deep copy with shared connection."""

    def test_copy_creates_independent_instance(self) -> None:
        """copy() creates a new instance."""
        t = _DynamicTransformer(scale=2.0)
        c = t.copy()
        assert c is not t
        assert c.scale == 2.0

    def test_copy_deep_copies_params(self) -> None:
        """copy() deep copies mutable state."""
        t = _DynamicTransformer()
        t._fitted = True
        t.params_ = {"x": 1.0}
        c = t.copy()
        c.params_["x"] = 99.0
        assert t.params_["x"] == 1.0


class TestSerialization:
    """Test __getstate__ / __setstate__ for pickle."""

    def test_pickle_roundtrip(self) -> None:
        """Transformer survives pickle roundtrip."""
        t = _DynamicTransformer(scale=2.0)
        t._fitted = True
        t.params_ = {"price__mean": 42.0}
        data = pickle.dumps(t)
        t2 = pickle.loads(data)  # noqa: S301
        assert t2.scale == 2.0
        assert t2.params_ == {"price__mean": 42.0}
        assert t2._fitted is True

    def test_pickle_nulls_connection(self) -> None:
        """Connection is nulled during pickling."""
        t = _StaticTransformer()
        t._connection = "fake_connection"
        data = pickle.dumps(t)
        t2 = pickle.loads(data)  # noqa: S301
        assert t2._connection is None

    def test_pickle_preserves_type(self) -> None:
        """Pickle preserves subclass type."""
        t = _DynamicTransformer(scale=3.0)
        t2 = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert type(t2) is _DynamicTransformer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestCheckThread tests/core/test_transformer.py::TestClone tests/core/test_transformer.py::TestCopy tests/core/test_transformer.py::TestSerialization -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write implementation**

Add `import copy`, `import os`, `import threading` at top of `src/sqlearn/core/transformer.py`, then add to the class:

```python
    # --- Thread safety ---

    def _check_thread(self) -> None:
        """Guard against cross-thread/cross-process access.

        Stores _owner_thread and _owner_pid on first call. Raises
        RuntimeError on subsequent calls from different thread/process.

        Raises:
            RuntimeError: If accessed from different thread or process.
        """
        current_thread = threading.current_thread().ident
        current_pid = os.getpid()

        if self._owner_pid is None:
            self._owner_pid = current_pid
            self._owner_thread = current_thread
        elif self._owner_pid != current_pid:
            msg = (
                f"{type(self).__name__} accessed from a different process "
                f"(original pid={self._owner_pid}, current pid={current_pid}). "
                "DuckDB connections cannot be shared across processes."
            )
            raise RuntimeError(msg)
        elif self._owner_thread != current_thread:
            msg = (
                f"{type(self).__name__} accessed from a different thread. "
                "Pipelines are not thread-safe. Use .clone() to create "
                "a thread-safe copy with the same fitted parameters."
            )
            raise RuntimeError(msg)

    # --- Copying ---

    def clone(self) -> Transformer:
        """Create independent copy. Thread-safe (new connection).

        Deep copies params_, sets_, columns_. Resets thread ownership.
        Used by sq.Search for parallel training.

        Returns:
            New Transformer of the same type with same params and
            fitted state, but independent thread ownership.
        """
        params = self.get_params()
        new = type(self)(**params)
        new._fitted = self._fitted
        new.params_ = copy.deepcopy(self.params_)
        new.sets_ = copy.deepcopy(self.sets_)
        new.columns_ = copy.deepcopy(self.columns_)
        new.input_schema_ = self.input_schema_
        new.output_schema_ = self.output_schema_
        new._y_column = self._y_column
        new._owner_thread = None
        new._owner_pid = None
        new._connection = None
        return new

    def copy(self) -> Transformer:
        """Deep copy via copy.deepcopy(). Shares connection reference.

        NOT thread-safe. Use clone() for cross-thread independence.

        Returns:
            Deep copy of this transformer.
        """
        return copy.deepcopy(self)

    # --- Serialization ---

    def __getstate__(self) -> dict[str, Any]:
        """Null out DuckDB connection before pickling.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle. Connection lazily recreated.

        Args:
            state: Instance state dict from __getstate__.
        """
        self.__dict__.update(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add thread safety, clone, copy, serialization to Transformer"
```

---

### Task 6: _apply_expressions(), __add__, __iadd__, and stubs

**Files:**
- Modify: `src/sqlearn/core/transformer.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_transformer.py`:

```python
# Add test helper at module level:

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


# --- Tests ---


class TestApplyExpressions:
    """Test _apply_expressions() base class wrapper."""

    def test_passthrough_unmodified(self) -> None:
        """Unmodified columns pass through."""
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
        """Modified columns overlay input expressions."""
        t = _DynamicTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        result = t._apply_expressions(exprs)
        assert "price" in result
        # Should be a Sub expression, not a bare Column
        assert isinstance(result["price"], exp.Sub)

    def test_new_columns_added(self) -> None:
        """New columns from expressions() are added to result."""
        t = _AddColumnTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        result = t._apply_expressions(exprs)
        assert "price" in result
        assert "price_doubled" in result

    def test_output_schema_filters_columns(self) -> None:
        """Columns not in output_schema are removed."""
        t = _UndeclaredColumnTransformer()
        t._fitted = True
        t.columns_ = ["price"]
        t.input_schema_ = Schema({"price": "DOUBLE"})
        exprs = {"price": exp.column("price")}
        with pytest.warns(UserWarning, match="output_schema"):
            result = t._apply_expressions(exprs)
        assert "secret_col" not in result

    def test_not_fitted_raises(self) -> None:
        """_apply_expressions raises if columns_ not set."""
        t = _StaticTransformer()
        with pytest.raises(RuntimeError, match="columns_"):
            t._apply_expressions({"price": exp.column("price")})


class TestOperators:
    """Test __add__ and __iadd__ pipeline composition."""

    def test_add_raises_not_implemented(self) -> None:
        """__add__ raises NotImplementedError until Pipeline lands."""
        a = _StaticTransformer()
        b = _DynamicTransformer()
        with pytest.raises(NotImplementedError, match="[Pp]ipeline"):
            _ = a + b

    def test_iadd_raises_not_implemented(self) -> None:
        """__iadd__ raises NotImplementedError until Pipeline lands."""
        a = _StaticTransformer()
        b = _DynamicTransformer()
        with pytest.raises(NotImplementedError, match="[Pp]ipeline"):
            a += b


class TestStubs:
    """Test stub methods raise NotImplementedError."""

    def test_fit_stub(self) -> None:
        """fit() raises NotImplementedError."""
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.fit("data.parquet")

    def test_transform_stub(self) -> None:
        """transform() raises NotImplementedError."""
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.transform("data.parquet")

    def test_fit_transform_stub(self) -> None:
        """fit_transform() raises NotImplementedError."""
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.fit_transform("data.parquet")

    def test_to_sql_stub(self) -> None:
        """to_sql() raises NotImplementedError."""
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.to_sql()

    def test_freeze_stub(self) -> None:
        """freeze() raises NotImplementedError."""
        t = _StaticTransformer()
        with pytest.raises(NotImplementedError):
            t.freeze()

    def test_fit_signature(self) -> None:
        """fit() has correct parameter names."""
        sig = inspect.signature(Transformer.fit)
        params = list(sig.parameters.keys())
        assert "data" in params
        assert "y" in params
        assert "backend" in params

    def test_transform_signature(self) -> None:
        """transform() has correct parameter names."""
        sig = inspect.signature(Transformer.transform)
        params = list(sig.parameters.keys())
        assert "data" in params
        assert "out" in params
        assert "backend" in params
        assert "batch_size" in params
        assert "dtype" in params
        assert "exclude_target" in params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestApplyExpressions tests/core/test_transformer.py::TestOperators tests/core/test_transformer.py::TestStubs -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write implementation**

Add `import warnings` at top of `src/sqlearn/core/transformer.py`, then add to the class:

```python
    # --- Expression composition (internal) ---

    def _apply_expressions(
        self,
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Base class wrapper around expressions().

        Called by the compiler, not by users. Handles auto-passthrough
        of unmodified columns, detects undeclared new columns, and
        filters output to match output_schema().

        Args:
            exprs: Current expression dict for all columns.

        Returns:
            Expression dict after applying this transformer.

        Raises:
            RuntimeError: If columns_ is not set (not fitted).
        """
        if self.columns_ is None:
            msg = "columns_ not set — call fit() first"
            raise RuntimeError(msg)

        modified = self.expressions(self.columns_, exprs)
        result = dict(exprs)  # passthrough all input columns
        new_cols = set(modified.keys()) - set(exprs.keys())
        result.update(modified)  # overlay modifications and additions

        # Filter to output schema columns
        if self.input_schema_ is not None:
            output_cols = set(
                self.output_schema(self.input_schema_).columns.keys()
            )
            undeclared = new_cols - output_cols
            if undeclared:
                warnings.warn(
                    f"{type(self).__name__}.expressions() created columns "
                    f"{undeclared} but output_schema() doesn't declare them. "
                    "Override output_schema() to include new columns.",
                    UserWarning,
                    stacklevel=2,
                )
            result = {k: v for k, v in result.items() if k in output_cols}

        return result

    # --- Operators ---

    def __add__(self, other: Transformer) -> Transformer:
        """Sequential composition: a + b -> Pipeline([a, b]).

        Returns a new Pipeline containing both transformers.
        Flattens nested Pipeline operands.

        Args:
            other: Transformer to compose after self.

        Returns:
            A new Pipeline (once Pipeline is implemented).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        msg = "Pipeline composition requires sqlearn.core.pipeline (issue #7)"
        raise NotImplementedError(msg)

    def __iadd__(self, other: Transformer) -> Transformer:
        """Incremental composition: pipe += step -> NEW Pipeline.

        Non-mutating — follows Python numeric convention.
        Returns a new Pipeline, does not modify self.

        Args:
            other: Transformer to append.

        Returns:
            A new Pipeline (once Pipeline is implemented).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        return self.__add__(other)

    # --- Stubs (implemented when Pipeline/Compiler/Backend land) ---

    def fit(
        self,
        data: Any,
        y: str | None = None,
        *,
        backend: Any = None,
    ) -> Transformer:
        """Learn parameters from data.

        Calls discover() and discover_sets() internally. Resolves
        column specifications against the data schema.

        Args:
            data: Input data (file path, table name, or DataFrame).
            y: Target column name, or None.
            backend: Backend override. Default uses DuckDB.

        Returns:
            self (for method chaining).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def transform(
        self,
        data: Any,
        *,
        out: str = "numpy",
        backend: Any = None,
        batch_size: int | None = None,
        dtype: Any = None,
        exclude_target: bool = True,
    ) -> Any:
        """Apply transformation to data.

        Calls expressions() or query() internally. Compiles to SQL
        and executes via the backend.

        Args:
            data: Input data (file path, table name, or DataFrame).
            out: Output format (``'numpy'``, ``'pandas'``, ``'polars'``,
                ``'arrow'``).
            backend: Backend override.
            batch_size: Process data in batches of this size.
            dtype: NumPy dtype for output array.
            exclude_target: Exclude target column(s) from output.

        Returns:
            TransformResult (numpy-compatible).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def fit_transform(
        self,
        data: Any,
        y: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience: fit then transform.

        Args:
            data: Input data.
            y: Target column name.
            **kwargs: Passed to transform().

        Returns:
            TransformResult (numpy-compatible).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def to_sql(
        self,
        *,
        dialect: str = "duckdb",
        table: str = "__input__",
    ) -> str:
        """Compile to SQL string without executing.

        Args:
            dialect: SQL dialect for output (``'duckdb'``, ``'postgres'``,
                ``'snowflake'``, etc.).
            table: Input table name placeholder.

        Returns:
            SQL query string.

        Raises:
            NotImplementedError: Until Compiler (issue #6) is implemented.
        """
        raise NotImplementedError

    def freeze(self) -> Any:
        """Return a FrozenPipeline: immutable, pre-compiled, deployment-ready.

        Returns:
            FrozenPipeline instance.

        Raises:
            NotImplementedError: Until FrozenPipeline (Milestone 7).
        """
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full linter suite**

Run: `make check`
Expected: All checks pass. Fix any issues.

- [ ] **Step 6: Commit**

```bash
git add src/sqlearn/core/transformer.py tests/core/test_transformer.py
git commit -m "feat(core): add _apply_expressions, operators, stubs to Transformer"
```

---

### Task 7: Package exports and final verification

**Files:**
- Modify: `src/sqlearn/core/__init__.py`
- Modify: `src/sqlearn/__init__.py`
- Modify: `tests/core/test_transformer.py`

- [ ] **Step 1: Write failing export tests**

Add to `tests/core/test_transformer.py`:

```python
class TestExports:
    """Test that Transformer is exported correctly."""

    def test_from_core(self) -> None:
        """Transformer is importable from sqlearn.core."""
        import sqlearn.core

        assert sqlearn.core.Transformer is Transformer

    def test_from_package(self) -> None:
        """Transformer is importable from sqlearn."""
        import sqlearn

        assert sqlearn.Transformer is Transformer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_transformer.py::TestExports -v`
Expected: FAIL — `AttributeError: module 'sqlearn.core' has no attribute 'Transformer'`

- [ ] **Step 3: Update __init__.py files**

Update `src/sqlearn/core/__init__.py`:

```python
"""sqlearn core — Transformer base, Pipeline, Compiler, Backend, Schema."""

from sqlearn.core.schema import (
    ColumnSelector,
    Schema,
    boolean,
    categorical,
    dtype,
    matching,
    numeric,
    temporal,
)
from sqlearn.core.transformer import Transformer

__all__ = [
    "ColumnSelector",
    "Schema",
    "Transformer",
    "boolean",
    "categorical",
    "dtype",
    "matching",
    "numeric",
    "temporal",
]
```

Update `src/sqlearn/__init__.py`:

```python
"""sqlearn — Compile ML preprocessing pipelines to SQL."""

from sqlearn.core.schema import (
    ColumnSelector,
    Schema,
    boolean,
    categorical,
    dtype,
    matching,
    numeric,
    temporal,
)
from sqlearn.core.transformer import Transformer

__all__ = [
    "ColumnSelector",
    "Schema",
    "Transformer",
    "boolean",
    "categorical",
    "dtype",
    "matching",
    "numeric",
    "temporal",
]

__version__ = "0.0.1"
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (both schema and transformer)

- [ ] **Step 5: Run full check suite**

Run: `make check`
Expected: All checks pass (ruff, pyright, mypy, interrogate, vulture, tests)

- [ ] **Step 6: Commit**

```bash
git add src/sqlearn/core/__init__.py src/sqlearn/__init__.py tests/core/test_transformer.py
git commit -m "feat(core): export Transformer from sqlearn package"
```

- [ ] **Step 7: Update BACKLOG.md**

In `BACKLOG.md`, change:
```
- [ ] `transformer.py` — Transformer base class with `_validate_custom()` method
```
to:
```
- [x] `transformer.py` — Transformer base class with `_validate_custom()` method
```

- [ ] **Step 8: Close GitHub issue**

```bash
gh issue close 5 --comment "Transformer base class implemented with all methods per spec."
```

- [ ] **Step 9: Final commit**

```bash
git add BACKLOG.md
git commit -m "chore: mark transformer.py complete in backlog"
```
