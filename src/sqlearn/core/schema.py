"""Schema dataclass and column type system.

Provides the Schema class for representing table structure, type category
constants for column classification, column selectors for type-aware routing,
and resolve_columns() for unified column resolution.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlearn.core.errors import MissingColumnError, SchemaError

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Type categories — source of truth for auto column routing
# ---------------------------------------------------------------------------

NUMERIC_TYPES: frozenset[str] = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "INT",
        "INT4",
        "INT2",
        "INT1",
        "BIGINT",
        "INT8",
        "LONG",
        "HUGEINT",
        "SIGNED",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "FLOAT4",
        "DOUBLE",
        "FLOAT8",
        "DECIMAL",
        "NUMERIC",
        "REAL",
    }
)

CATEGORICAL_TYPES: frozenset[str] = frozenset(
    {
        "VARCHAR",
        "TEXT",
        "STRING",
        "CHAR",
        "BPCHAR",
        "ENUM",
    }
)

TEMPORAL_TYPES: frozenset[str] = frozenset(
    {
        "DATE",
        "TIME",
        "TIMETZ",
        "TIME WITH TIME ZONE",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP_S",
        "TIMESTAMP_MS",
        "TIMESTAMP_NS",
        "INTERVAL",
    }
)

BOOLEAN_TYPES: frozenset[str] = frozenset(
    {
        "BOOLEAN",
        "BOOL",
        "LOGICAL",
    }
)

_CATEGORY_MAP: dict[str, frozenset[str]] = {
    "numeric": NUMERIC_TYPES,
    "categorical": CATEGORICAL_TYPES,
    "temporal": TEMPORAL_TYPES,
    "boolean": BOOLEAN_TYPES,
}


def _normalize_type(sql_type: str) -> str:
    """Strip parameters from SQL type: 'DECIMAL(18,3)' -> 'DECIMAL'."""
    paren = sql_type.find("(")
    base = sql_type[:paren] if paren != -1 else sql_type
    return base.strip().upper()


def _classify_type(sql_type: str) -> str:
    """Classify a SQL type string into a category name."""
    normalized = _normalize_type(sql_type)
    for category, types in _CATEGORY_MAP.items():
        if normalized in types:
            return category
    return "other"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schema:
    """Immutable table schema mapping column names to SQL types.

    All mutation methods return new Schema instances. Column order is
    preserved (Python dict insertion order).

    Args:
        columns: Mapping of column names to DuckDB SQL type strings.
            Example: ``{"price": "DOUBLE", "city": "VARCHAR"}``

    Raises:
        SchemaError: If column names use the reserved ``__sq_*__`` prefix.

    Examples:
        Create a schema and inspect it:

        >>> from sqlearn.core.schema import Schema
        >>> schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        >>> schema.numeric()
        ['price']
        >>> schema.categorical()
        ['city']
        >>> schema.column_category("price")
        'numeric'

        Immutable mutations return new schemas:

        >>> new = schema.add({"quantity": "INTEGER"})
        >>> len(new)  # 3 columns
        3
        >>> len(schema)  # original unchanged
        2

        Drop and rename columns:

        >>> trimmed = schema.drop(["city"])
        >>> renamed = schema.rename({"price": "cost"})

    See Also:
        :func:`~sqlearn.core.schema.resolve_columns`: Resolve column specs.
        :func:`~sqlearn.core.schema.numeric`: Select numeric columns.
        :func:`~sqlearn.core.schema.matching`: Select columns by pattern.
    """

    columns: dict[str, str]

    def __post_init__(self) -> None:
        """Defensive copy and reserved prefix validation."""
        object.__setattr__(self, "columns", dict(self.columns))
        reserved = [c for c in self.columns if c.startswith("__sq_") and c.endswith("__")]
        if reserved:
            msg = (
                f"Column(s) {sorted(reserved)} use reserved sqlearn prefix "
                "'__sq_*__'. Please rename these columns."
            )
            raise SchemaError(msg)

    def __hash__(self) -> int:
        """Hash based on column names and types for cache key use."""
        return hash(frozenset(self.columns.items()))

    # --- Mutation (returns new Schema) ---

    def add(self, new: dict[str, str]) -> Schema:
        """Add columns. Raises ValueError if any already exist.

        Args:
            new: Mapping of new column names to SQL type strings.

        Returns:
            New Schema with additional columns appended.

        Raises:
            SchemaError: If any column name already exists.
        """
        overlap = set(new) & set(self.columns)
        if overlap:
            msg = f"Columns already exist: {sorted(overlap)}"
            raise SchemaError(msg)
        return Schema({**self.columns, **new})

    def drop(self, cols: list[str]) -> Schema:
        """Remove columns.

        Args:
            cols: Column names to remove.

        Returns:
            New Schema without the specified columns.

        Raises:
            SchemaError: If any column name doesn't exist.
        """
        missing = set(cols) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise SchemaError(msg)
        return Schema({k: v for k, v in self.columns.items() if k not in cols})

    def rename(self, mapping: dict[str, str]) -> Schema:
        """Rename columns.

        Args:
            mapping: Old name to new name mapping.

        Returns:
            New Schema with columns renamed, preserving order.

        Raises:
            SchemaError: If any old name doesn't exist.
        """
        missing = set(mapping) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise SchemaError(msg)
        return Schema({mapping.get(k, k): v for k, v in self.columns.items()})

    def cast(
        self,
        col: str | dict[str, str],
        new_type: str | None = None,
    ) -> Schema:
        """Change column type(s).

        Single: ``schema.cast("price", "FLOAT")``
        Batch: ``schema.cast({"price": "FLOAT", "qty": "BIGINT"})``

        Args:
            col: Column name (single) or mapping of names to new types (batch).
            new_type: New SQL type string (required for single, ignored for batch).

        Returns:
            New Schema with updated types.

        Raises:
            SchemaError: If any column name doesn't exist.
            TypeError: If single form used without new_type.
        """
        if isinstance(col, dict):
            mapping = col
        else:
            if new_type is None:
                msg = "new_type is required when col is a string"
                raise TypeError(msg)
            mapping = {col: new_type}
        missing = set(mapping) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise SchemaError(msg)
        return Schema({k: mapping.get(k, v) for k, v in self.columns.items()})

    def select(self, cols: list[str]) -> Schema:
        """Keep only these columns, preserving their original order.

        Args:
            cols: Column names to keep.

        Returns:
            New Schema with only the specified columns.

        Raises:
            SchemaError: If any column name doesn't exist.
        """
        missing = set(cols) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise SchemaError(msg)
        keep = set(cols)
        return Schema({k: v for k, v in self.columns.items() if k in keep})

    # --- Query ---

    def column_category(self, col: str) -> str:
        """Return the type category for a column.

        Args:
            col: Column name.

        Returns:
            One of: ``'numeric'``, ``'categorical'``, ``'temporal'``,
            ``'boolean'``, or ``'other'``.

        Raises:
            MissingColumnError: If column doesn't exist.
        """
        if col not in self.columns:
            msg = f"Column not found: {col!r}"
            raise MissingColumnError(msg, column=col, available=list(self.columns))
        return _classify_type(self.columns[col])

    def numeric(self) -> list[str]:
        """Return all numeric column names, preserving order."""
        return [c for c in self.columns if _classify_type(self.columns[c]) == "numeric"]

    def categorical(self) -> list[str]:
        """Return all categorical column names, preserving order."""
        return [c for c in self.columns if _classify_type(self.columns[c]) == "categorical"]

    def temporal(self) -> list[str]:
        """Return all temporal column names, preserving order."""
        return [c for c in self.columns if _classify_type(self.columns[c]) == "temporal"]

    def boolean(self) -> list[str]:
        """Return all boolean column names, preserving order."""
        return [c for c in self.columns if _classify_type(self.columns[c]) == "boolean"]

    # --- Dunder ---

    def __len__(self) -> int:
        """Return number of columns."""
        return len(self.columns)

    def __contains__(self, col: object) -> bool:
        """Check if column name exists."""
        return col in self.columns

    def __getitem__(self, col: str) -> str:
        """Get SQL type for a column. Raises KeyError if missing."""
        return self.columns[col]

    def __iter__(self) -> Iterator[str]:
        """Iterate column names in insertion order."""
        return iter(self.columns)

    def __repr__(self) -> str:
        """Show column=type pairs: Schema(price=DOUBLE, city=VARCHAR)."""
        pairs = ", ".join(f"{k}={v}" for k, v in self.columns.items())
        return f"Schema({pairs})"


# ---------------------------------------------------------------------------
# Column selectors
# ---------------------------------------------------------------------------


class ColumnSelector:
    """Base class for column selection criteria.

    Subclasses implement :meth:`resolve` to produce a list of column names
    from a :class:`Schema`.

    Selectors support set-like composition operators:

    - ``selector1 | selector2`` — union (columns matching either)
    - ``selector1 & selector2`` — intersection (columns matching both)
    - ``~selector`` — negation (columns NOT matching)
    - ``selector1 - selector2`` — difference (first but not second)

    Examples:
        Compose selectors for flexible column routing:

        >>> from sqlearn.core.schema import Schema, numeric, boolean, matching
        >>> schema = Schema({"price": "DOUBLE", "active": "BOOLEAN", "city": "VARCHAR"})
        >>> (numeric() | boolean()).resolve(schema)
        ['price', 'active']
        >>> (~numeric()).resolve(schema)
        ['active', 'city']
    """

    def resolve(self, schema: Schema) -> list[str]:
        """Resolve to concrete column names from schema.

        Args:
            schema: The schema to resolve against.

        Returns:
            Ordered list of matching column names.
        """
        raise NotImplementedError

    def __or__(self, other: ColumnSelector) -> _UnionSelector:
        """Return a selector matching columns in either selector (union).

        Args:
            other: Another selector to union with.

        Returns:
            A new selector that resolves to the union of both, preserving
            order from the left selector first, then unseen from the right.
        """
        return _UnionSelector(self, other)

    def __and__(self, other: ColumnSelector) -> _IntersectionSelector:
        """Return a selector matching columns in both selectors (intersection).

        Args:
            other: Another selector to intersect with.

        Returns:
            A new selector that resolves to columns present in both,
            preserving order from the right selector.
        """
        return _IntersectionSelector(self, other)

    def __invert__(self) -> _NegationSelector:
        """Return a selector matching columns NOT in this selector.

        Returns:
            A new selector that resolves to all schema columns except
            those matched by this selector.
        """
        return _NegationSelector(self)

    def __sub__(self, other: ColumnSelector) -> _DifferenceSelector:
        """Return a selector matching columns in this but not the other.

        Args:
            other: Selector whose matches are excluded.

        Returns:
            A new selector that resolves to columns in this selector
            minus those in the other, preserving order.
        """
        return _DifferenceSelector(self, other)


class TypeSelector(ColumnSelector):
    """Selects columns matching a type category.

    Args:
        types: Frozenset of SQL type strings to match against.
        name: Category name for repr (e.g. ``'numeric'``).
    """

    def __init__(self, types: frozenset[str], name: str) -> None:
        self._types = types
        self._name = name

    def resolve(self, schema: Schema) -> list[str]:
        """Return columns whose normalized type is in the type set."""
        return [col for col in schema if _normalize_type(schema[col]) in self._types]

    def __repr__(self) -> str:
        """Show factory name: numeric()."""
        return f"{self._name}()"


class PatternSelector(ColumnSelector):
    """Selects columns matching a glob pattern (fnmatch, not regex).

    Regex matching (e.g. ``Drop(pattern="^id_")``) is handled by individual
    transformers, not by the selector system.

    Args:
        pattern: Glob pattern string (e.g. ``'price_*'``).
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    def resolve(self, schema: Schema) -> list[str]:
        """Return columns whose names match the glob pattern."""
        return [col for col in schema if fnmatch.fnmatch(col, self._pattern)]

    def __repr__(self) -> str:
        """Show factory call: matching('price_*')."""
        return f"matching({self._pattern!r})"


class DTypeSelector(ColumnSelector):
    """Selects columns matching a SQL type (both sides normalized).

    ``sq.dtype("DECIMAL")`` matches ``DECIMAL``, ``DECIMAL(18,3)``, etc.
    ``sq.dtype("DECIMAL(18,3)")`` also normalizes to ``DECIMAL``.

    Args:
        sql_type: SQL type string to match against (normalized before comparison).
    """

    def __init__(self, sql_type: str) -> None:
        self._sql_type = _normalize_type(sql_type)

    def resolve(self, schema: Schema) -> list[str]:
        """Return columns whose normalized type matches."""
        return [col for col in schema if _normalize_type(schema[col]) == self._sql_type]

    def __repr__(self) -> str:
        """Show factory call: dtype('DOUBLE')."""
        return f"dtype({self._sql_type!r})"


class _UnionSelector(ColumnSelector):
    """Selects columns matching either of two selectors (set union).

    Order: columns from left selector first, then unseen from right selector.
    Duplicates are removed.

    Args:
        left: First selector.
        right: Second selector.
    """

    def __init__(self, left: ColumnSelector, right: ColumnSelector) -> None:
        self._left = left
        self._right = right

    def resolve(self, schema: Schema) -> list[str]:
        """Return union of both selectors, preserving order without duplicates."""
        left_cols = self._left.resolve(schema)
        seen = set(left_cols)
        return left_cols + [c for c in self._right.resolve(schema) if c not in seen]

    def __repr__(self) -> str:
        """Show composition: (numeric() | boolean())."""
        return f"({self._left!r} | {self._right!r})"


class _IntersectionSelector(ColumnSelector):
    """Selects columns matching both selectors (set intersection).

    Order: preserves order from the right selector, filtered by left.

    Args:
        left: First selector.
        right: Second selector.
    """

    def __init__(self, left: ColumnSelector, right: ColumnSelector) -> None:
        self._left = left
        self._right = right

    def resolve(self, schema: Schema) -> list[str]:
        """Return intersection of both selectors."""
        left_set = set(self._left.resolve(schema))
        return [c for c in self._right.resolve(schema) if c in left_set]

    def __repr__(self) -> str:
        """Show composition: (numeric() & matching('price_*'))."""
        return f"({self._left!r} & {self._right!r})"


class _NegationSelector(ColumnSelector):
    """Selects all columns NOT matched by the inner selector.

    Order: preserves schema column order.

    Args:
        inner: Selector whose matches are excluded.
    """

    def __init__(self, inner: ColumnSelector) -> None:
        self._inner = inner

    def resolve(self, schema: Schema) -> list[str]:
        """Return all schema columns not in the inner selector."""
        excluded = set(self._inner.resolve(schema))
        return [c for c in schema if c not in excluded]

    def __repr__(self) -> str:
        """Show negation: ~numeric()."""
        return f"~{self._inner!r}"


class _DifferenceSelector(ColumnSelector):
    """Selects columns in the left selector but not the right (set difference).

    Order: preserves order from the left selector.

    Args:
        left: Selector to keep.
        right: Selector to exclude.
    """

    def __init__(self, left: ColumnSelector, right: ColumnSelector) -> None:
        self._left = left
        self._right = right

    def resolve(self, schema: Schema) -> list[str]:
        """Return columns in left selector minus those in right selector."""
        excluded = set(self._right.resolve(schema))
        return [c for c in self._left.resolve(schema) if c not in excluded]

    def __repr__(self) -> str:
        """Show difference: (numeric() - matching('id_*'))."""
        return f"({self._left!r} - {self._right!r})"


class _AllSelector(ColumnSelector):
    """Selects all columns in the schema.

    Useful as the starting point for difference or negation expressions:
    ``all_columns() - categorical()`` selects everything except categoricals.
    """

    def resolve(self, schema: Schema) -> list[str]:
        """Return all column names in schema order."""
        return list(schema)

    def __repr__(self) -> str:
        """Show factory call: all_columns()."""
        return "all_columns()"


class _ColumnsSelector(ColumnSelector):
    """Selects explicitly named columns.

    Only columns that exist in the schema are returned. Non-existent names
    are silently skipped (use :func:`resolve_columns` with a list for strict
    validation).

    Args:
        names: Column names to select.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def resolve(self, schema: Schema) -> list[str]:
        """Return the named columns that exist in the schema."""
        schema_cols = set(schema.columns)
        return [c for c in self._names if c in schema_cols]

    def __repr__(self) -> str:
        """Show factory call: columns('price', 'qty')."""
        args = ", ".join(repr(n) for n in self._names)
        return f"columns({args})"


# ---------------------------------------------------------------------------
# Selector factory functions
# ---------------------------------------------------------------------------


def numeric() -> TypeSelector:
    """Select all numeric columns.

    Returns:
        A selector that resolves to columns with numeric SQL types
        (INTEGER, DOUBLE, FLOAT, DECIMAL, etc.).
    """
    return TypeSelector(NUMERIC_TYPES, "numeric")


def categorical() -> TypeSelector:
    """Select all categorical columns.

    Returns:
        A selector that resolves to columns with categorical SQL types
        (VARCHAR, TEXT, STRING, etc.).
    """
    return TypeSelector(CATEGORICAL_TYPES, "categorical")


def temporal() -> TypeSelector:
    """Select all temporal columns.

    Returns:
        A selector that resolves to columns with temporal SQL types
        (DATE, TIME, TIMESTAMP, etc.).
    """
    return TypeSelector(TEMPORAL_TYPES, "temporal")


def boolean() -> TypeSelector:
    """Select all boolean columns.

    Returns:
        A selector that resolves to columns with boolean SQL types
        (BOOLEAN, BOOL, LOGICAL).
    """
    return TypeSelector(BOOLEAN_TYPES, "boolean")


def matching(pattern: str) -> PatternSelector:
    """Select columns matching a glob pattern.

    Uses :func:`fnmatch.fnmatch` for matching. Supports ``*``, ``?``,
    ``[seq]``, and ``[!seq]`` wildcards.

    Args:
        pattern: Glob pattern string (e.g. ``'price_*'``).

    Returns:
        A selector that resolves to columns whose names match the pattern.
    """
    return PatternSelector(pattern)


def dtype(sql_type: str) -> DTypeSelector:
    """Select columns matching a specific SQL type.

    Both the selector type and column types are normalized before comparison
    (parameters stripped, uppercased). So ``dtype("DECIMAL")`` matches
    ``DECIMAL(18,3)``.

    Args:
        sql_type: SQL type string to match.

    Returns:
        A selector that resolves to columns with matching normalized type.
    """
    return DTypeSelector(sql_type)


def all_columns() -> _AllSelector:
    """Select all columns in the schema.

    Useful as the starting point for difference or negation expressions.

    Returns:
        A selector that resolves to every column in the schema.

    Examples:
        Select everything except categoricals:

        >>> from sqlearn.core.schema import Schema, all_columns, categorical
        >>> schema = Schema({"price": "DOUBLE", "city": "VARCHAR", "active": "BOOLEAN"})
        >>> (all_columns() - categorical()).resolve(schema)
        ['price', 'active']
    """
    return _AllSelector()


def columns(*names: str) -> _ColumnsSelector:
    """Select explicitly named columns.

    Creates a selector for a fixed set of column names. Only columns that
    exist in the schema at resolution time are returned; non-existent names
    are silently skipped.

    Args:
        *names: Column names to select.

    Returns:
        A selector that resolves to the named columns present in the schema.

    Examples:
        Select specific columns:

        >>> from sqlearn.core.schema import Schema, columns
        >>> schema = Schema({"price": "DOUBLE", "qty": "INTEGER", "city": "VARCHAR"})
        >>> columns("price", "qty").resolve(schema)
        ['price', 'qty']

        Compose with other selectors:

        >>> from sqlearn.core.schema import numeric
        >>> (numeric() - columns("qty")).resolve(schema)
        ['price']
    """
    return _ColumnsSelector(names)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def resolve_columns(
    schema: Schema,
    columns: str | list[str] | ColumnSelector | None,
) -> list[str]:
    """Resolve column specification to concrete column names.

    Handles all the ways columns can be specified in sqlearn:

    - String literal (``"numeric"``, ``"categorical"``, ``"temporal"``,
      ``"boolean"``): type-based filtering, used by ``_default_columns``.
    - ``"all"``: returns every column regardless of type.
    - ``list[str]``: explicit column names, validated against schema.
    - :class:`ColumnSelector`: calls ``.resolve(schema)``.
    - ``None``: raises ValueError. Transformers with ``_default_columns = None``
      require explicit columns from the user.

    Args:
        schema: Current table schema.
        columns: Column specification.

    Returns:
        Ordered list of column names.

    Raises:
        ValueError: If columns is None or string is not a recognized category.
        SchemaError: If explicit column names don't exist in schema.
    """
    if columns is None:
        msg = "columns=None requires explicit column specification"
        raise ValueError(msg)

    if isinstance(columns, ColumnSelector):
        return columns.resolve(schema)

    if isinstance(columns, list):
        missing = set(columns) - set(schema.columns)
        if missing:
            msg = f"Columns not found in schema: {sorted(missing)}"
            raise SchemaError(msg)
        return columns

    # String literal
    if columns == "all":
        return list(schema)

    if columns in _CATEGORY_MAP:
        return TypeSelector(_CATEGORY_MAP[columns], columns).resolve(schema)

    msg = f"Unknown column specification: {columns!r}"
    raise ValueError(msg)
