"""SelectKBest — select top K features by score.

Computes a score for each feature against the target column during
``discover()`` and keeps only the top K features.

This is a **dynamic** transformer — it learns feature scores during
``fit()`` and requires ``y`` to be provided.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class SelectKBest(Transformer):
    """Select top K features by score via SQL.

    Computes a score for each numeric feature against the target column
    during ``fit()`` and keeps only the K highest-scoring features. The
    ``score_func`` parameter controls which scoring method is used.

    Scoring methods:

    - ``"f_regression"``: Uses absolute Pearson correlation with the target
      (``|CORR(col, y)|``). Default.
    - ``"mutual_info"``: Uses absolute correlation as a proxy for mutual
      information. True MI would require binning which is deferred.
    - ``"f_classif"``: Uses absolute correlation as a proxy for ANOVA
      F-statistic. True F-test for classification is deferred.

    Generated SQL (keeping top 2 features)::

        SELECT
          price,
          quantity
        FROM (
          SELECT * FROM __input__
        ) AS __input__

    Args:
        k: Number of top features to keep.
        score_func: Scoring function name. One of ``"f_regression"``,
            ``"mutual_info"``, ``"f_classif"``. Default is ``"f_regression"``.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``k`` < 1 or ``score_func`` is not recognized.

    Examples:
        Select top 3 features:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.SelectKBest(k=3)])
        >>> pipe.fit("data.parquet", y="target")
        >>> pipe.to_sql()
        ... # SELECT top_3_cols FROM ...

        Use a different scoring function:

        >>> pipe = sq.Pipeline([sq.SelectKBest(k=5, score_func="mutual_info")])
        >>> pipe.fit("data.parquet", y="target")
        >>> pipe.get_feature_names_out()

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.feature_selection.variance.VarianceThreshold`:
            Drop low-variance features.
        :class:`~sqlearn.feature_selection.drop.Drop`: Drop explicit columns.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    _VALID_SCORE_FUNCS: frozenset[str] = frozenset({"f_regression", "mutual_info", "f_classif"})

    def __init__(
        self,
        *,
        k: int,
        score_func: str = "f_regression",
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize SelectKBest transformer.

        Args:
            k: Number of top features to keep. Must be >= 1.
            score_func: Scoring function name.
            columns: Column specification override.

        Raises:
            ValueError: If k < 1 or score_func is not recognized.
        """
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)

        if score_func not in self._VALID_SCORE_FUNCS:
            valid = sorted(self._VALID_SCORE_FUNCS)
            msg = f"score_func must be one of {valid}, got {score_func!r}"
            raise ValueError(msg)

        super().__init__(columns=columns)
        self.k = k
        self.score_func = score_func
        self._selected: list[str] = []

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Compute feature scores against the target column.

        Uses ``CORR(col, y_column)`` for all scoring functions. The
        absolute correlation value is used as the score.

        Args:
            columns: Target columns to score.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name. Required for scoring.

        Returns:
            Dict mapping score keys to sqlglot CORR expressions.

        Raises:
            ValueError: If y_column is None (target required for scoring).
        """
        if y_column is None:
            msg = "SelectKBest requires y (target column) to be specified during fit()"
            raise ValueError(msg)

        result: dict[str, exp.Expression] = {}
        for col in columns:
            if col == y_column:
                continue
            key = f"{col}__score"
            result[key] = exp.Corr(
                this=exp.Column(this=exp.to_identifier(col)),
                expression=exp.Column(this=exp.to_identifier(y_column)),
            )
        return result

    def _determine_selected(self, columns: list[str]) -> list[str]:
        """Determine which columns to keep based on learned scores.

        Selects the top ``self.k`` columns by absolute score. If k is
        greater than the number of scored columns, all scored columns
        are kept.

        Args:
            columns: Target columns that were scored.

        Returns:
            List of column names to keep, in original order.
        """
        params = self.params_ or {}

        # Collect (column, abs_score) pairs
        scored: list[tuple[str, float]] = []
        for col in columns:
            key = f"{col}__score"
            score = params.get(key)
            if score is not None:
                scored.append((col, abs(float(score))))

        # Sort by score descending, then alphabetically for ties
        scored.sort(key=lambda x: (-x[1], x[0]))

        # Take top k
        top_k = scored[: self.k]
        selected_set = {col for col, _ in top_k}

        # Return in original column order
        return [col for col in columns if col in selected_set]

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a SELECT wrapping the input, keeping only selected columns.

        After ``fit()``, the selected columns are determined by
        ``_determine_selected()``. This method wraps the input in a SELECT
        that includes only the selected columns plus any non-targeted columns.

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression keeping only selected columns.
        """
        if self.input_schema_ is None:
            return input_query  # pragma: no cover

        # Determine which columns to keep (lazy, computed once)
        if not self._selected and self.columns_ is not None:
            self._selected = self._determine_selected(self.columns_)
            # Store in params_ for inspection
            if self.params_ is None:
                self.params_ = {}
            self.params_["__selected__"] = self._selected

        selected_set = set(self._selected)
        columns_set = set(self.columns_ or [])
        selections: list[exp.Expression] = [
            exp.Column(this=exp.to_identifier(col_name))
            for col_name in self.input_schema_.columns
            if col_name in selected_set or col_name not in columns_set
        ]

        return exp.Select(expressions=selections).from_(  # pyright: ignore[reportUnknownMemberType]
            exp.Subquery(
                this=input_query,
                alias=exp.to_identifier("__input__"),
            )
        )

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema keeping only selected columns.

        Removes non-selected target columns from the input schema.
        Non-targeted columns (e.g., categoricals when targeting numerics)
        are preserved.

        Args:
            schema: Input schema.

        Returns:
            Output schema with only selected target columns.
        """
        if not self._selected and self.columns_ is not None:
            self._selected = self._determine_selected(self.columns_)

        if not self._selected:
            return schema

        selected_set = set(self._selected)
        columns_set = set(self.columns_ or [])
        to_drop = [c for c in columns_set if c not in selected_set and c in schema.columns]
        if not to_drop:
            return schema
        return schema.drop(to_drop)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Not used --- SelectKBest uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}

    def clone(self) -> SelectKBest:
        """Create independent copy with cloned state.

        Returns:
            New SelectKBest with same params and fitted state.
        """
        new = SelectKBest(
            k=self.k,
            score_func=self.score_func,
            columns=self.columns,
        )
        new._fitted = self._fitted
        new.params_ = copy.deepcopy(self.params_)
        new.sets_ = copy.deepcopy(self.sets_)
        new.columns_ = copy.deepcopy(self.columns_)
        new.input_schema_ = self.input_schema_
        new.output_schema_ = self.output_schema_
        new._y_column = self._y_column
        new._selected = list(self._selected)
        new._owner_thread = None
        new._owner_pid = None
        new._connection = None
        return new

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state including _selected list.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state
