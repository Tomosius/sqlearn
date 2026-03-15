"""DropCorrelated — drop one of each pair of highly correlated features.

Computes pairwise Pearson correlations during ``discover()`` and identifies
columns to drop using a greedy strategy: for each correlated pair, drop the
column that appears in more correlated pairs.

This is a **dynamic** transformer — it learns correlation statistics during
``fit()``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class DropCorrelated(Transformer):
    """Drop one of each pair of highly correlated features via SQL.

    Computes pairwise Pearson correlations for all numeric columns during
    ``fit()``. For each pair with absolute correlation above ``threshold``,
    one column is dropped using a greedy strategy: the column that appears
    in more correlated pairs is dropped first, breaking ties alphabetically.

    Generated SQL (after dropping ``col_b``)::

        SELECT
          col_a,
          col_c
        FROM (
          SELECT * FROM __input__
        ) AS __input__

    Args:
        threshold: Absolute correlation threshold above which one column
            in a pair is dropped. Default is ``0.95``.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``threshold`` is not between 0 and 1 (exclusive).

    Examples:
        Drop highly correlated features:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.DropCorrelated(threshold=0.9)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT only non-correlated columns

        Use a stricter threshold:

        >>> pipe = sq.Pipeline([sq.DropCorrelated(threshold=0.99)])
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()  # fewer columns dropped

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.feature_selection.drop.Drop`: Drop explicit columns.
        :class:`~sqlearn.feature_selection.variance.VarianceThreshold`:
            Drop low-variance features.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize DropCorrelated transformer.

        Args:
            threshold: Absolute correlation threshold (0 < threshold < 1).
            columns: Column specification override.

        Raises:
            ValueError: If threshold is not between 0 and 1 (exclusive).
        """
        if not (0 < threshold < 1):
            msg = f"threshold must be between 0 and 1 (exclusive), got {threshold}"
            raise ValueError(msg)

        super().__init__(columns=columns)
        self.threshold = threshold
        self._dropped: list[str] = []

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Compute pairwise Pearson correlations for all column pairs.

        Returns sqlglot ``CORR(col_a, col_b)`` aggregates for each unique
        pair of target columns. Results are stored in ``params_`` as
        ``'{col_a}__{col_b}__corr'`` keys.

        Args:
            columns: Target columns to compute correlations for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping pair keys to sqlglot CORR expressions.
        """
        result: dict[str, exp.Expression] = {}
        for i, col_a in enumerate(columns):
            for col_b in columns[i + 1 :]:
                key = f"{col_a}__{col_b}__corr"
                result[key] = exp.Corr(
                    this=exp.Column(this=exp.to_identifier(col_a)),
                    expression=exp.Column(this=exp.to_identifier(col_b)),
                )
        return result

    def _determine_drops(self, columns: list[str]) -> list[str]:
        """Determine which columns to drop based on learned correlations.

        Uses a greedy strategy: count how many correlated pairs each column
        appears in, then drop columns with the highest count first. Ties
        are broken alphabetically.

        Args:
            columns: Target columns that were analyzed.

        Returns:
            List of column names to drop.
        """
        params = self.params_ or {}

        # Build list of correlated pairs
        correlated_pairs: list[tuple[str, str]] = []
        for i, col_a in enumerate(columns):
            for col_b in columns[i + 1 :]:
                key = f"{col_a}__{col_b}__corr"
                corr_value = params.get(key)
                if corr_value is not None and abs(float(corr_value)) > self.threshold:
                    correlated_pairs.append((col_a, col_b))

        if not correlated_pairs:
            return []

        # Count appearances in correlated pairs
        dropped: set[str] = set()
        # Greedy: process pairs, drop the column appearing in more pairs
        while correlated_pairs:
            # Count remaining appearances
            count: dict[str, int] = {}
            for a, b in correlated_pairs:
                if a not in dropped:
                    count[a] = count.get(a, 0) + 1
                if b not in dropped:
                    count[b] = count.get(b, 0) + 1

            if not count:
                break

            # Drop the column with the highest count (alphabetical tie-break)
            to_drop = max(count, key=lambda c: (count[c], c))
            dropped.add(to_drop)

            # Remove pairs that include the dropped column
            correlated_pairs = [
                (a, b) for a, b in correlated_pairs if a not in dropped and b not in dropped
            ]

        return sorted(dropped)

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a SELECT wrapping the input, excluding correlated columns.

        After ``fit()``, the dropped columns are determined by
        ``_determine_drops()``. This method wraps the input in a SELECT
        that excludes those columns.

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression excluding correlated columns.
        """
        if self.input_schema_ is None:
            return input_query  # pragma: no cover

        # Determine which columns to drop (lazy, computed once)
        if not self._dropped and self.columns_ is not None:
            self._dropped = self._determine_drops(self.columns_)
            # Store in params_ for inspection
            if self.params_ is None:
                self.params_ = {}
            self.params_["__dropped__"] = self._dropped

        drop_set = set(self._dropped)
        selections: list[exp.Expression] = [
            exp.Column(this=exp.to_identifier(col_name))
            for col_name in self.input_schema_.columns
            if col_name not in drop_set
        ]

        return exp.Select(expressions=selections).from_(  # pyright: ignore[reportUnknownMemberType]
            exp.Subquery(
                this=input_query,
                alias=exp.to_identifier("__input__"),
            )
        )

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after dropping correlated columns.

        Removes the identified correlated columns from the input schema.

        Args:
            schema: Input schema.

        Returns:
            Output schema with correlated columns removed.
        """
        if not self._dropped and self.columns_ is not None:
            self._dropped = self._determine_drops(self.columns_)

        if not self._dropped:
            return schema

        existing = [c for c in self._dropped if c in schema.columns]
        if not existing:
            return schema
        return schema.drop(existing)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Not used --- DropCorrelated uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}

    def clone(self) -> DropCorrelated:
        """Create independent copy with cloned state.

        Returns:
            New DropCorrelated with same params and fitted state.
        """
        new = DropCorrelated(threshold=self.threshold, columns=self.columns)
        new._fitted = self._fitted
        new.params_ = copy.deepcopy(self.params_)
        new.sets_ = copy.deepcopy(self.sets_)
        new.columns_ = copy.deepcopy(self.columns_)
        new.input_schema_ = self.input_schema_
        new.output_schema_ = self.output_schema_
        new._y_column = self._y_column
        new._dropped = list(self._dropped)
        new._owner_thread = None
        new._owner_pid = None
        new._connection = None
        return new

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state including _dropped list.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state
