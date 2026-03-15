"""Columns — apply different transformers to different column subsets.

Replaces sklearn's ColumnTransformer. Routes column subsets to dedicated
transformers and combines their SQL outputs into a single SELECT.

sqlearn's auto column routing (``_default_columns``) means most pipelines
don't need Columns at all. Use Columns when you need explicit control over
which transformer gets which columns.
"""

from __future__ import annotations

import copy
from typing import Any

import sqlglot.expressions as exp  # noqa: TC002

from sqlearn.core.errors import InvalidStepError, NotFittedError, SchemaError
from sqlearn.core.schema import ColumnSelector, Schema, resolve_columns
from sqlearn.core.transformer import Transformer

# Type alias for a transformer group: (name, transformer, column_spec)
_TransformerGroup = tuple[str, Transformer, str | list[str] | ColumnSelector]


class Columns(Transformer):
    """Apply different transformers to different column subsets.

    Routes each column subset to its dedicated transformer, then combines
    all outputs into a single SQL SELECT. Replaces sklearn's
    ``ColumnTransformer`` with a simpler, SQL-native API.

    Each group is a ``(name, transformer, columns)`` triple where columns
    can be a list of names, a type string (``"numeric"``), or a
    :class:`~sqlearn.core.schema.ColumnSelector`.

    Unmatched columns (not assigned to any group) are handled by the
    ``remainder`` parameter: ``"drop"`` excludes them, ``"passthrough"``
    includes them unchanged.

    Generated SQL (scale + encode)::

        SELECT
          (price - 3.0) / NULLIF(1.41, 0) AS price,
          (quantity - 30.0) / NULLIF(14.14, 0) AS quantity,
          CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
          CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS city_paris
        FROM __input__

    Args:
        transformers: List of ``(name, transformer, columns)`` triples.
            Each name must be unique. Each transformer is a
            :class:`~sqlearn.core.transformer.Transformer` instance.
        remainder: What to do with columns not assigned to any group.
            ``"drop"`` (default) excludes them from the output.
            ``"passthrough"`` includes them unchanged.

    Raises:
        InvalidStepError: If transformers list is empty, contains
            non-Transformer objects, or has duplicate group names.
        SchemaError: If column groups overlap (same column assigned to
            multiple groups).
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.

    Examples:
        Scale numerics and encode categoricals:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Columns(
        ...             [
        ...                 ("scale", sq.StandardScaler(), sq.numeric()),
        ...                 ("encode", sq.OneHotEncoder(), sq.categorical()),
        ...             ]
        ...         ),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()

        Keep unmatched columns with passthrough:

        >>> cols = sq.Columns(
        ...     [("scale", sq.StandardScaler(), ["price"])],
        ...     remainder="passthrough",
        ... )
        >>> pipe = sq.Pipeline([cols])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # price scaled, other columns passed through

    See Also:
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
        :class:`~sqlearn.core.transformer.Transformer`: Base class for steps.
        :func:`~sqlearn.core.schema.numeric`: Select numeric columns.
        :func:`~sqlearn.core.schema.categorical`: Select categorical columns.
    """

    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]
    _default_columns: str = "all"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        transformers: list[_TransformerGroup],
        *,
        remainder: str = "drop",
    ) -> None:
        super().__init__()
        self.transformers = transformers
        self.remainder = remainder

        self._validate_init()

        # Fitted state: resolved column lists per group
        self._resolved_groups: list[tuple[str, Transformer, list[str]]] | None = None
        self._remainder_cols: list[str] | None = None

    def _validate_init(self) -> None:
        """Validate constructor arguments.

        Raises:
            InvalidStepError: If transformers is empty, contains
                non-Transformer objects, or has duplicate names.
            ValueError: If remainder is not ``"drop"`` or ``"passthrough"``.
        """
        if not self.transformers:
            msg = "Columns requires at least one transformer group"
            raise InvalidStepError(msg)

        seen_names: set[str] = set()
        for item in self.transformers:
            if len(item) != 3:  # noqa: PLR2004
                msg = (
                    f"Each transformer group must be a (name, transformer, columns) "
                    f"triple, got {len(item)} elements"
                )
                raise InvalidStepError(msg)
            name, transformer, _cols = item
            if not isinstance(transformer, Transformer):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"Group {name!r} transformer is not a Transformer: {type(transformer)}"
                raise InvalidStepError(msg)
            if name in seen_names:
                msg = f"Duplicate group name: {name!r}"
                raise InvalidStepError(msg)
            seen_names.add(name)

        if self.remainder not in ("drop", "passthrough"):
            msg = f"remainder must be 'drop' or 'passthrough', got {self.remainder!r}"
            raise ValueError(msg)

    # --- Column resolution ---

    def _resolve_columns_spec(self) -> str | list[str] | ColumnSelector | None:
        """Return 'all' since Columns manages its own column routing.

        Returns:
            ``"all"`` to receive the full schema during fitting.
        """
        return "all"

    def _resolve_groups(
        self,
        schema: Schema,
    ) -> tuple[list[tuple[str, Transformer, list[str]]], list[str]]:
        """Resolve column selectors for each group against the schema.

        Validates that no column appears in more than one group.

        Args:
            schema: Current table schema.

        Returns:
            Tuple of (resolved groups, remainder columns).

        Raises:
            SchemaError: If column groups overlap.
        """
        resolved: list[tuple[str, Transformer, list[str]]] = []
        all_assigned: set[str] = set()

        for name, transformer, col_spec in self.transformers:
            cols = resolve_columns(schema, col_spec)

            # Check for overlap
            overlap = all_assigned & set(cols)
            if overlap:
                msg = (
                    f"Column(s) {sorted(overlap)} assigned to group {name!r} "
                    f"but already assigned to another group"
                )
                raise SchemaError(msg)

            all_assigned.update(cols)
            resolved.append((name, transformer, cols))

        # Remainder: columns not assigned to any group
        remainder_cols = [c for c in schema.columns if c not in all_assigned]
        return resolved, remainder_cols

    # --- Subclass overrides ---

    def discover(
        self,
        columns: list[str],  # noqa: ARG002
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Collect aggregate expressions from all sub-transformers.

        Resolves column groups, then collects aggregate expressions from
        each sub-transformer's discover() call. Prefixes parameter names
        with the group name to avoid collisions.

        Args:
            columns: All columns (unused, groups handle their own).
            schema: Current table schema.
            y_column: Target column name, if provided.

        Returns:
            Combined dict of aggregate expressions from all groups.
        """
        resolved, remainder_cols = self._resolve_groups(schema)
        self._resolved_groups = resolved
        self._remainder_cols = remainder_cols

        result: dict[str, exp.Expression] = {}
        for name, transformer, cols in resolved:
            if not cols:
                continue
            sub_discover = transformer.discover(cols, schema, y_column)
            for param_name, agg_expr in sub_discover.items():
                # Prefix to avoid collisions between groups
                prefixed = f"__{name}__{param_name}"
                result[prefixed] = agg_expr
        return result

    def discover_sets(
        self,
        columns: list[str],  # noqa: ARG002
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Collect set-valued queries from all sub-transformers.

        Args:
            columns: All columns (unused, groups handle their own).
            schema: Current table schema.
            y_column: Target column name, if provided.

        Returns:
            Combined dict of set queries from all groups.
        """
        if self._resolved_groups is None:
            resolved, remainder_cols = self._resolve_groups(schema)
            self._resolved_groups = resolved
            self._remainder_cols = remainder_cols

        result: dict[str, exp.Expression] = {}
        for name, transformer, cols in self._resolved_groups:
            if not cols:
                continue
            sub_sets = transformer.discover_sets(cols, schema, y_column)
            for set_name, set_query in sub_sets.items():
                prefixed = f"__{name}__{set_name}"
                result[prefixed] = set_query
        return result

    def _distribute_params(self) -> None:
        """Distribute learned params and sets to sub-transformers.

        After the compiler populates self.params_ and self.sets_,
        this method unpacks prefixed keys and distributes the values
        to each sub-transformer's params_ and sets_.
        """
        if self._resolved_groups is None:
            return

        for name, transformer, cols in self._resolved_groups:
            if not cols:
                continue

            self._distribute_scalar_params(name, transformer)
            self._distribute_set_params(name, transformer)

            # Mark sub-transformer as fitted
            transformer.columns_ = cols
            transformer.input_schema_ = self.input_schema_
            transformer._fitted = True  # noqa: SLF001

    def _distribute_scalar_params(self, name: str, transformer: Transformer) -> None:
        """Distribute scalar params to a single sub-transformer.

        Args:
            name: Group name (used as prefix).
            transformer: Sub-transformer to receive params.
        """
        if not self.params_:
            return
        prefix = f"__{name}__"
        sub_params: dict[str, Any] = {}
        for key, value in self.params_.items():
            if key.startswith(prefix):
                sub_params[key[len(prefix) :]] = value
        if sub_params:
            transformer.params_ = sub_params

    def _distribute_set_params(self, name: str, transformer: Transformer) -> None:
        """Distribute set params to a single sub-transformer.

        Args:
            name: Group name (used as prefix).
            transformer: Sub-transformer to receive sets.
        """
        if not self.sets_:
            return
        prefix = f"__{name}__"
        sub_sets: dict[str, list[dict[str, Any]]] = {}
        for key, value in self.sets_.items():
            if key.startswith(prefix):
                sub_sets[key[len(prefix) :]] = value
        if sub_sets:
            transformer.sets_ = sub_sets

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Combine expressions from all sub-transformers.

        Each sub-transformer produces expressions for its column subset.
        Results are merged into a single dict. Remainder columns are
        included if ``remainder="passthrough"``.

        Args:
            columns: All columns (unused, groups handle their own).
            exprs: Current expression dict for all columns.

        Returns:
            Combined expression dict from all groups plus remainder.
        """
        # Ensure params are distributed before generating expressions
        self._distribute_params()

        result: dict[str, exp.Expression] = {}

        if self._resolved_groups is None:
            return result

        for _name, transformer, cols in self._resolved_groups:
            if not cols:
                continue

            # Update sub-transformer output_schema_ now that params/sets are set
            if transformer.input_schema_ is not None:
                transformer.output_schema_ = transformer.output_schema(transformer.input_schema_)

            sub_exprs = transformer.expressions(cols, exprs)
            result.update(sub_exprs)

            # For transformers that remove original columns (e.g. OneHotEncoder),
            # detect which columns were dropped and remove them from the result.
            if transformer.output_schema_ is not None and transformer.input_schema_ is not None:
                out_cols = set(transformer.output_schema_.columns.keys())
                in_cols = set(transformer.input_schema_.columns.keys())
                dropped = (in_cols - out_cols) & set(cols)
                for col in dropped:
                    result.pop(col, None)

        # Passthrough remainder columns
        if self.remainder == "passthrough" and self._remainder_cols:
            for col in self._remainder_cols:
                if col in exprs and col not in result:
                    result[col] = exprs[col]

        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema combining all group outputs.

        Merges output schemas from each sub-transformer. Adds remainder
        columns if ``remainder="passthrough"``.

        When called post-fit (params_/sets_ available), distributes
        learned parameters to sub-transformers so their output_schema()
        can reflect learned state (e.g. OneHotEncoder expanding columns).

        Args:
            schema: Input schema.

        Returns:
            Combined output schema.
        """
        # Use cached resolved groups if available, otherwise resolve fresh
        if self._resolved_groups is not None:
            resolved = self._resolved_groups
            remainder_cols = self._remainder_cols or []
        else:
            try:
                resolved, remainder_cols = self._resolve_groups(schema)
            except (ValueError, SchemaError):
                return schema

        # Ensure sub-transformers have their params/sets before output_schema
        if self.params_ is not None or self.sets_ is not None:
            self._distribute_params()

        out_columns: dict[str, str] = {}

        for _name, transformer, cols in resolved:
            if not cols:
                continue
            # Build sub-schema with only this group's columns
            sub_schema = Schema({c: schema.columns[c] for c in cols if c in schema.columns})
            sub_output = transformer.output_schema(sub_schema)
            out_columns.update(sub_output.columns)

        if self.remainder == "passthrough":
            for col in remainder_cols:
                if col in schema.columns:
                    out_columns[col] = schema.columns[col]

        return Schema(out_columns) if out_columns else schema

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names after fitting.

        Returns:
            List of column names from the combined output schema.

        Raises:
            NotFittedError: If Columns is not fitted.
        """
        if not self._fitted:
            msg = "Columns is not fitted"
            raise NotFittedError(msg)
        if self.output_schema_ is None:
            msg = "output_schema_ is not set"
            raise NotFittedError(msg)
        return list(self.output_schema_.columns.keys())

    # --- Display ---

    def __repr__(self) -> str:
        """Show group names and transformer types.

        Returns:
            Format like ``Columns(scale=StandardScaler, encode=OneHotEncoder)``.
        """
        parts = [f"{name}={type(t).__name__}" for name, t, _ in self.transformers]
        remainder_str = f", remainder={self.remainder!r}" if self.remainder != "drop" else ""
        return f"Columns({', '.join(parts)}{remainder_str})"

    # --- sklearn introspection ---

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters as dict. sklearn-compatible.

        Args:
            deep: If True, includes sub-transformer params with
                ``'groupname__param'`` separator.

        Returns:
            Dict of parameter names to current values.
        """
        params: dict[str, Any] = {
            "transformers": self.transformers,
            "remainder": self.remainder,
        }
        if deep:
            for name, transformer, _cols in self.transformers:
                sub_params = transformer.get_params(deep=True)
                for key, value in sub_params.items():
                    params[f"{name}__{key}"] = value
        return params

    # --- Copying ---

    def clone(self) -> Columns:
        """Create independent copy with cloned sub-transformers.

        Returns:
            New Columns with independently cloned sub-transformers.
        """
        cloned_groups: list[_TransformerGroup] = [
            (name, transformer.clone(), col_spec)
            for name, transformer, col_spec in self.transformers
        ]
        new = Columns(cloned_groups, remainder=self.remainder)
        new._fitted = self._fitted
        new.params_ = copy.deepcopy(self.params_)
        new.sets_ = copy.deepcopy(self.sets_)
        new.columns_ = copy.deepcopy(self.columns_)
        new.input_schema_ = self.input_schema_
        new.output_schema_ = self.output_schema_
        if self._resolved_groups is not None:
            new._resolved_groups = [
                (name, new_t, list(cols))
                for (name, _old_t, cols), (_, new_t, _) in zip(
                    self._resolved_groups, cloned_groups, strict=True
                )
            ]
        new._remainder_cols = (
            list(self._remainder_cols) if self._remainder_cols is not None else None
        )
        new._owner_thread = None
        new._owner_pid = None
        new._connection = None
        return new

    # --- Serialization ---

    def __getstate__(self) -> dict[str, Any]:
        """Null out DuckDB connection before pickling.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state
