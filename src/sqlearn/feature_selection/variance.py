"""VarianceThreshold — drop low-variance features.

Computes population variance for each numeric column during ``discover()``
and drops columns with variance at or below the threshold.

This is a **dynamic** transformer — it learns variance statistics during
``fit()``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class VarianceThreshold(Transformer):
    """Drop low-variance features via SQL.

    Computes the population variance (``VAR_POP``) for each numeric column
    during ``fit()``. Columns with variance at or below ``threshold`` are
    dropped from the output.

    With the default ``threshold=0.0``, only constant columns (zero
    variance) are removed.

    Generated SQL (after dropping ``constant_col``)::

        SELECT
          price,
          quantity
        FROM (
          SELECT * FROM __input__
        ) AS __input__

    Args:
        threshold: Minimum variance to keep a column. Columns with
            variance <= threshold are dropped. Default is ``0.0``.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``threshold`` is negative.

    Examples:
        Remove constant columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.VarianceThreshold()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT only non-constant columns

        Remove low-variance features:

        >>> pipe = sq.Pipeline([sq.VarianceThreshold(threshold=0.1)])
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.feature_selection.drop.Drop`: Drop explicit columns.
        :class:`~sqlearn.feature_selection.kbest.SelectKBest`:
            Select top K features by score.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize VarianceThreshold transformer.

        Args:
            threshold: Minimum variance to keep a column.
            columns: Column specification override.

        Raises:
            ValueError: If threshold is negative.
        """
        if threshold < 0:
            msg = f"threshold must be non-negative, got {threshold}"
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
        """Compute population variance for each target column.

        Returns sqlglot ``VAR_POP(col)`` aggregates for each column.
        Results are stored in ``params_`` as ``'{col}__var'`` keys.

        Args:
            columns: Target columns to compute variance for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping variance keys to sqlglot VAR_POP expressions.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            key = f"{col}__var"
            result[key] = exp.VariancePop(
                this=exp.Column(this=exp.to_identifier(col)),
            )
        return result

    def _determine_drops(self, columns: list[str]) -> list[str]:
        """Determine which columns to drop based on learned variances.

        Columns with variance at or below ``self.threshold`` are marked
        for dropping.

        Args:
            columns: Target columns that were analyzed.

        Returns:
            Sorted list of column names to drop.
        """
        params = self.params_ or {}
        dropped: list[str] = []
        for col in columns:
            key = f"{col}__var"
            variance = params.get(key)
            if variance is not None and float(variance) <= self.threshold:
                dropped.append(col)
        return sorted(dropped)

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a SELECT wrapping the input, excluding low-variance columns.

        After ``fit()``, the dropped columns are determined by
        ``_determine_drops()``. This method wraps the input in a SELECT
        that excludes those columns.

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression excluding low-variance columns.
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
        """Declare output schema after dropping low-variance columns.

        Removes the identified low-variance columns from the input schema.

        Args:
            schema: Input schema.

        Returns:
            Output schema with low-variance columns removed.
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
        """Not used --- VarianceThreshold uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}

    def clone(self) -> VarianceThreshold:
        """Create independent copy with cloned state.

        Returns:
            New VarianceThreshold with same params and fitted state.
        """
        new = VarianceThreshold(threshold=self.threshold, columns=self.columns)
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
