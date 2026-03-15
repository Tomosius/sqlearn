"""String transforms -- Length, Lower, Upper, Trim, Replace, Substring.

All transformers in this module are **static** (no ``discover()`` needed)
and default to ``categorical`` columns (string types). They compile to
inline SQL functions via ``expressions()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class StringLength(Transformer):
    """Compute the length of string columns via ``LENGTH(col)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in a ``LENGTH()`` call, producing an
    integer column. The output schema is updated to reflect the type
    change from ``VARCHAR`` to ``INTEGER``.

    Generated SQL::

        SELECT
          LENGTH(name) AS name,
          LENGTH(city) AS city
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all
            categorical (string) columns.

    Examples:
        Compute string lengths:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import StringLength
        >>> pipe = sq.Pipeline([StringLength()])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # LENGTH(name) AS name, LENGTH(city) AS city

        Target specific columns:

        >>> pipe = sq.Pipeline([StringLength(columns=["name"])])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # LENGTH(name) AS name

    See Also:
        :class:`Lower`: Convert strings to lowercase.
        :class:`Upper`: Convert strings to uppercase.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize StringLength transformer.

        Args:
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline LENGTH expressions for each target column.

        Wraps each column's current expression in ``LENGTH(expr)``
        using the sqlglot AST.

        Args:
            columns: Target columns to compute length for.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in LENGTH.
        """
        return {col: exp.Length(this=exprs[col]) for col in columns}

    def output_schema(self, schema: Schema) -> Schema:
        """Update schema types: string columns become INTEGER after LENGTH.

        Args:
            schema: Input schema.

        Returns:
            New schema with target columns cast to ``INTEGER``.
        """
        if self.columns_ is None:
            return schema
        cast_updates = {col: "INTEGER" for col in self.columns_ if col in schema.columns}
        if not cast_updates:
            return schema
        return schema.cast(cast_updates)


class Lower(Transformer):
    """Convert string columns to lowercase via ``LOWER(col)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in a ``LOWER()`` call. Output type
    remains ``VARCHAR``.

    Generated SQL::

        SELECT
          LOWER(name) AS name,
          LOWER(city) AS city
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all
            categorical (string) columns.

    Examples:
        Lowercase all string columns:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import Lower
        >>> pipe = sq.Pipeline([Lower()])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # LOWER(name) AS name, LOWER(city) AS city

        Target specific columns:

        >>> pipe = sq.Pipeline([Lower(columns=["name"])])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # LOWER(name) AS name

    See Also:
        :class:`Upper`: Convert strings to uppercase.
        :class:`Trim`: Remove leading/trailing whitespace.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Lower transformer.

        Args:
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline LOWER expressions for each target column.

        Args:
            columns: Target columns to convert to lowercase.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in LOWER.
        """
        return {col: exp.Lower(this=exprs[col]) for col in columns}


class Upper(Transformer):
    """Convert string columns to uppercase via ``UPPER(col)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in an ``UPPER()`` call. Output type
    remains ``VARCHAR``.

    Generated SQL::

        SELECT
          UPPER(name) AS name,
          UPPER(city) AS city
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all
            categorical (string) columns.

    Examples:
        Uppercase all string columns:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import Upper
        >>> pipe = sq.Pipeline([Upper()])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # UPPER(name) AS name, UPPER(city) AS city

        Target specific columns:

        >>> pipe = sq.Pipeline([Upper(columns=["city"])])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # UPPER(city) AS city

    See Also:
        :class:`Lower`: Convert strings to lowercase.
        :class:`Trim`: Remove leading/trailing whitespace.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Upper transformer.

        Args:
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline UPPER expressions for each target column.

        Args:
            columns: Target columns to convert to uppercase.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in UPPER.
        """
        return {col: exp.Upper(this=exprs[col]) for col in columns}


class Trim(Transformer):
    """Remove leading and trailing whitespace (or specified characters) via ``TRIM(col)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in a ``TRIM()`` call. When ``characters``
    is provided, the generated SQL uses ``TRIM(col, characters)`` to remove
    those specific characters instead of whitespace.

    Generated SQL (default)::

        SELECT
          TRIM(name) AS name
        FROM __input__

    Generated SQL (with characters)::

        SELECT
          TRIM(name, '#') AS name
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all
            categorical (string) columns.
        characters: Specific characters to trim. If ``None``, trims
            whitespace (SQL default).

    Examples:
        Trim whitespace from all string columns:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import Trim
        >>> pipe = sq.Pipeline([Trim()])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # TRIM(name) AS name

        Trim specific characters:

        >>> pipe = sq.Pipeline([Trim(characters="#")])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # TRIM(name, '#') AS name

    See Also:
        :class:`Lower`: Convert strings to lowercase.
        :class:`Replace`: Replace substrings.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
        characters: str | None = None,
    ) -> None:
        """Initialize Trim transformer.

        Args:
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.
            characters: Specific characters to trim. If ``None``, trims
                whitespace.
        """
        super().__init__(columns=columns)
        self.characters = characters

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline TRIM expressions for each target column.

        Args:
            columns: Target columns to trim.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in TRIM.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            if self.characters is not None:
                result[col] = exp.Anonymous(
                    this="TRIM",
                    expressions=[exprs[col], exp.Literal.string(self.characters)],
                )
            else:
                result[col] = exp.Trim(this=exprs[col])
        return result


class Replace(Transformer):
    """Replace occurrences of a substring via ``REPLACE(col, old, new)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in a ``REPLACE()`` call that substitutes
    all occurrences of ``old`` with ``new``.

    Generated SQL::

        SELECT
          REPLACE(name, 'foo', 'bar') AS name
        FROM __input__

    Args:
        old: Substring to search for. Must be a non-empty string.
        new: Replacement string.
        columns: Column specification override. Defaults to all
            categorical (string) columns.

    Raises:
        TypeError: If ``old`` or ``new`` is not a string.
        ValueError: If ``old`` is empty.

    Examples:
        Replace a substring in all string columns:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import Replace
        >>> pipe = sq.Pipeline([Replace(old="foo", new="bar")])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # REPLACE(name, 'foo', 'bar') AS name

        Delete a substring (replace with empty string):

        >>> pipe = sq.Pipeline([Replace(old="-", new="")])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # REPLACE(name, '-', '') AS name

    See Also:
        :class:`Trim`: Remove leading/trailing characters.
        :class:`Lower`: Convert strings to lowercase.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        old: str,
        new: str,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Replace transformer.

        Args:
            old: Substring to search for. Must be a non-empty string.
            new: Replacement string.
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.

        Raises:
            TypeError: If ``old`` or ``new`` is not a string.
            ValueError: If ``old`` is empty.
        """
        if not isinstance(old, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"old must be a string, got {type(old).__name__}"
            raise TypeError(msg)
        if not isinstance(new, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"new must be a string, got {type(new).__name__}"
            raise TypeError(msg)
        if len(old) == 0:
            msg = "old must be non-empty"
            raise ValueError(msg)
        super().__init__(columns=columns)
        self.old = old
        self.new = new

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline REPLACE expressions for each target column.

        Args:
            columns: Target columns to apply replacement on.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in REPLACE.
        """
        return {
            col: exp.Anonymous(
                this="REPLACE",
                expressions=[
                    exprs[col],
                    exp.Literal.string(self.old),
                    exp.Literal.string(self.new),
                ],
            )
            for col in columns
        }


class Substring(Transformer):
    """Extract a substring via ``SUBSTRING(col, start, length)``.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each target column is wrapped in a ``SUBSTRING()`` call. The ``start``
    position is 1-based (SQL convention). When ``length`` is omitted, the
    substring extends to the end of the string.

    Generated SQL (with length)::

        SELECT
          SUBSTRING(name, 1, 3) AS name
        FROM __input__

    Generated SQL (without length)::

        SELECT
          SUBSTRING(name, 2) AS name
        FROM __input__

    Args:
        start: 1-based start position. Must be a positive integer.
        length: Number of characters to extract. If ``None``, extracts
            to the end of the string.
        columns: Column specification override. Defaults to all
            categorical (string) columns.

    Raises:
        TypeError: If ``start`` or ``length`` is not an integer.
        ValueError: If ``start`` is less than 1 or ``length`` is less than 0.

    Examples:
        Extract first 3 characters:

        >>> import sqlearn as sq
        >>> from sqlearn.features.string import Substring
        >>> pipe = sq.Pipeline([Substring(start=1, length=3)])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # SUBSTRING(name, 1, 3) AS name

        Extract from position 2 to end:

        >>> pipe = sq.Pipeline([Substring(start=2)])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # SUBSTRING(name, 2) AS name

    See Also:
        :class:`Replace`: Replace substrings.
        :class:`StringLength`: Compute string length.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        start: int,
        length: int | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Substring transformer.

        Args:
            start: 1-based start position. Must be >= 1.
            length: Number of characters to extract. If ``None``,
                extracts to the end of the string. Must be >= 0 if provided.
            columns: Column specification override. If ``None``, defaults
                to all categorical (string) columns.

        Raises:
            TypeError: If ``start`` is not an int, or ``length`` is not
                an int when provided.
            ValueError: If ``start`` < 1 or ``length`` < 0.
        """
        if not isinstance(start, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"start must be an int, got {type(start).__name__}"
            raise TypeError(msg)
        if start < 1:
            msg = f"start must be >= 1, got {start}"
            raise ValueError(msg)
        if length is not None:
            if not isinstance(length, int):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"length must be an int, got {type(length).__name__}"
                raise TypeError(msg)
            if length < 0:
                msg = f"length must be >= 0, got {length}"
                raise ValueError(msg)
        super().__init__(columns=columns)
        self.start = start
        self.length = length

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SUBSTRING expressions for each target column.

        Args:
            columns: Target columns to extract substrings from.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions, each wrapped in SUBSTRING.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            kwargs: dict[str, exp.Expression] = {
                "this": exprs[col],
                "start": exp.Literal.number(self.start),
            }
            if self.length is not None:
                kwargs["length"] = exp.Literal.number(self.length)
            result[col] = exp.Substring(**kwargs)
        return result
