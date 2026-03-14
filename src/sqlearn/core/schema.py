"""Schema dataclass and column type system.

Provides the Schema class for representing table structure, type category
constants for column classification, column selectors for type-aware routing,
and resolve_columns() for unified column resolution.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """

    columns: dict[str, str]

    def __post_init__(self) -> None:
        """Defensive copy to prevent external mutation."""
        object.__setattr__(self, "columns", dict(self.columns))

    # --- Mutation (returns new Schema) ---

    def add(self, new: dict[str, str]) -> Schema:
        """Add columns. Raises ValueError if any already exist.

        Args:
            new: Mapping of new column names to SQL type strings.

        Returns:
            New Schema with additional columns appended.

        Raises:
            ValueError: If any column name already exists.
        """
        overlap = set(new) & set(self.columns)
        if overlap:
            msg = f"Columns already exist: {sorted(overlap)}"
            raise ValueError(msg)
        return Schema({**self.columns, **new})

    def drop(self, cols: list[str]) -> Schema:
        """Remove columns.

        Args:
            cols: Column names to remove.

        Returns:
            New Schema without the specified columns.

        Raises:
            KeyError: If any column name doesn't exist.
        """
        missing = set(cols) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise KeyError(msg)
        return Schema({k: v for k, v in self.columns.items() if k not in cols})

    def rename(self, mapping: dict[str, str]) -> Schema:
        """Rename columns.

        Args:
            mapping: Old name to new name mapping.

        Returns:
            New Schema with columns renamed, preserving order.

        Raises:
            KeyError: If any old name doesn't exist.
        """
        missing = set(mapping) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise KeyError(msg)
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
            KeyError: If any column name doesn't exist.
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
            raise KeyError(msg)
        return Schema({k: mapping.get(k, v) for k, v in self.columns.items()})

    def select(self, cols: list[str]) -> Schema:
        """Keep only these columns, preserving their original order.

        Args:
            cols: Column names to keep.

        Returns:
            New Schema with only the specified columns.

        Raises:
            KeyError: If any column name doesn't exist.
        """
        missing = set(cols) - set(self.columns)
        if missing:
            msg = f"Columns not found: {sorted(missing)}"
            raise KeyError(msg)
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
            KeyError: If column doesn't exist.
        """
        if col not in self.columns:
            msg = f"Column not found: {col!r}"
            raise KeyError(msg)
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
    """

    def resolve(self, schema: Schema) -> list[str]:
        """Resolve to concrete column names from schema.

        Args:
            schema: The schema to resolve against.

        Returns:
            Ordered list of matching column names.
        """
        raise NotImplementedError


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
