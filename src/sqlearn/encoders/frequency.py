"""FrequencyEncoder — encode categorical columns as their observed frequency.

Compiles to inline SQL: ``CASE WHEN col = 'cat' THEN 0.3 ... ELSE 0.0 END``
where each category is replaced by its proportion (or raw count) from training data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class FrequencyEncoder(Transformer):
    """Encode categorical columns as their observed frequency via SQL.

    Each category value is replaced with its frequency (proportion) in the
    training data. A category that appeared in 30% of rows maps to 0.3.
    When ``normalize=False``, raw counts are used instead.

    Unknown categories encountered at transform time are mapped to
    ``fill_value`` (default 0.0) when ``handle_unknown="value"``.

    Generated SQL (normalize=True)::

        SELECT
          CASE
            WHEN city = 'London' THEN 0.50
            WHEN city = 'Paris' THEN 0.25
            WHEN city = 'Tokyo' THEN 0.25
            ELSE 0.0
          END AS city
        FROM __input__

    Generated SQL (normalize=False)::

        SELECT
          CASE
            WHEN city = 'London' THEN 4
            WHEN city = 'Paris' THEN 2
            WHEN city = 'Tokyo' THEN 2
            ELSE 0.0
          END AS city
        FROM __input__

    Args:
        normalize: If True (default), output proportions in [0, 1].
            If False, output raw counts.
        handle_unknown: Strategy for unseen categories at transform time.
            ``"value"`` maps unknowns to ``fill_value`` (default).
            ``"error"`` raises at transform time (not yet implemented).
        fill_value: Value assigned to unknown categories when
            ``handle_unknown="value"``. Defaults to 0.0.
        columns: Column specification override. Defaults to all categorical
            columns via ``_default_columns = "categorical"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        FitError: If the input data has issues that prevent fitting
            (e.g. empty table).
        ValueError: If ``handle_unknown`` is not ``"value"`` or ``"error"``.

    Examples:
        Basic usage -- encode all categorical columns by frequency:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.FrequencyEncoder()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN city = 'London' THEN 0.50 ... END AS city

        Use raw counts instead of proportions:

        >>> encoder = sq.FrequencyEncoder(normalize=False)
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # CASE WHEN city = 'London' THEN 4 ... END AS city

        Encode specific columns only:

        >>> encoder = sq.FrequencyEncoder(columns=["city", "color"])
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only city and color are frequency-encoded

        Custom fill value for unknown categories:

        >>> encoder = sq.FrequencyEncoder(fill_value=-1.0)
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # unknowns map to -1.0 in ELSE clause

    See Also:
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Binary indicator encoding.
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before encoding.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        normalize: bool = True,
        handle_unknown: str = "value",
        fill_value: float = 0.0,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if handle_unknown not in ("value", "error"):
            msg = f"handle_unknown must be 'value' or 'error', got {handle_unknown!r}"
            raise ValueError(msg)
        self.normalize = normalize
        self.handle_unknown = handle_unknown
        self.fill_value = fill_value

    def discover(
        self,
        columns: list[str],  # noqa: ARG002
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn total row count via COUNT(*).

        Returns a sqlglot aggregate that the compiler executes during
        fit(). The total count is needed to compute proportions.

        Args:
            columns: Target columns (unused, total count is global).
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'__total_count'`` to ``COUNT(*)``.
        """
        return {"__total_count": exp.Count(this=exp.Star())}

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn distinct category values and their counts per column.

        Returns sqlglot SELECT queries that the compiler executes during
        fit(). The compiler adds a FROM clause automatically.

        Args:
            columns: Target columns to discover frequencies for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__freq'`` to ``SELECT col, COUNT(*)``
            grouped by column value, for each column.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            result[f"{col}__freq"] = exp.Select(
                expressions=[
                    exp.Column(this=col),
                    exp.Count(this=exp.Star()).as_("_count"),  # pyright: ignore[reportUnknownMemberType]
                ]
            ).group_by(exp.Column(this=col), copy=False)
        return result

    def _get_frequencies(self, col: str) -> list[tuple[str, float]]:
        """Get sorted (category, frequency) pairs for a column.

        Computes proportions or raw counts depending on ``normalize``.
        Categories are sorted alphabetically for deterministic SQL output.
        NULL categories are excluded.

        Args:
            col: Column name to get frequencies for.

        Returns:
            Sorted list of (category, frequency) tuples.
        """
        raw: list[dict[str, Any]] = self.sets_[f"{col}__freq"]  # type: ignore[index]
        total: float = float(self.params_["__total_count"])  # type: ignore[index]
        pairs: list[tuple[str, float]] = []
        for row in raw:  # pyright: ignore[reportUnknownVariableType]
            val: Any = row[col]  # pyright: ignore[reportUnknownVariableType]
            count: float = float(row["_count"])  # pyright: ignore[reportUnknownArgumentType]
            if val is not None:
                freq = (count / total if total > 0 else 0.0) if self.normalize else count
                pairs.append((str(val), freq))  # pyright: ignore[reportUnknownArgumentType]
        pairs.sort(key=lambda x: x[0])
        return pairs

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate CASE WHEN expressions for frequency encoding.

        Creates one ``CASE WHEN col = 'cat' THEN freq ... ELSE fill_value END``
        expression per column. Categories in each CASE are sorted alphabetically
        for deterministic SQL output.

        Args:
            columns: Target columns to encode.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of column expressions mapping categories to their frequencies.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            pairs = self._get_frequencies(col)
            if not pairs:
                # No known categories (e.g. all NULLs) — return fill_value directly
                result[col] = exp.Literal.number(self.fill_value)  # pyright: ignore[reportUnknownMemberType]
                continue
            ifs: list[exp.If] = []
            for cat, freq in pairs:
                ifs.append(
                    exp.If(
                        this=exp.EQ(
                            this=exprs[col],
                            expression=exp.Literal.string(cat),  # pyright: ignore[reportUnknownMemberType]
                        ),
                        true=exp.Literal.number(freq),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )
            result[col] = exp.Case(
                ifs=ifs,
                default=exp.Literal.number(self.fill_value),  # pyright: ignore[reportUnknownMemberType]
            )
        return result
