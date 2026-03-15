"""Union — combine outputs from multiple transformer branches horizontally.

Replaces sklearn's FeatureUnion. Each branch processes the same input
independently and their column outputs are merged into a single SELECT.
No SQL UNION is used — this is column-wise (horizontal) combination.

Generated SQL example (StandardScaler + OneHotEncoder)::

    SELECT
      (price - 42.5) / NULLIF(10.2, 0) AS scaled_price,
      (qty - 5.0) / NULLIF(2.1, 0) AS scaled_qty,
      CASE WHEN city = 'London' THEN 1 ELSE 0 END AS encoded_city_london,
      CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS encoded_city_paris
    FROM __input__
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from sqlearn.core.errors import InvalidStepError, NotFittedError
from sqlearn.core.schema import Schema, resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    import sqlglot.expressions as exp


class Union(Transformer):
    """Combine outputs from multiple transformer branches horizontally.

    Each branch receives the same input schema and processes it independently.
    The outputs from all branches are merged column-wise into a single SELECT
    statement. Column names are prefixed with the branch name to avoid
    collisions (e.g. ``scaled_price``, ``encoded_city_london``).

    This replaces sklearn's ``FeatureUnion``. Unlike SQL UNION (which
    concatenates rows), this combines columns horizontally.

    Generated SQL (StandardScaler + OneHotEncoder union)::

        SELECT
          (price - 42.5) / NULLIF(10.2, 0) AS scaled_price,
          (qty - 5.0) / NULLIF(2.1, 0) AS scaled_qty,
          CASE WHEN city = 'London' THEN 1 ELSE 0 END AS encoded_city_london,
          CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS encoded_city_paris
        FROM __input__

    Args:
        transformers: List of ``(name, transformer)`` pairs. Each
            transformer processes the full input independently. Names
            are used as column prefixes in the output.

    Raises:
        InvalidStepError: If transformers list is empty, contains
            non-Transformers, has duplicate names, or has invalid format.

    Examples:
        Combine scaled numerics with encoded categoricals:

        >>> import sqlearn as sq
        >>> union = sq.Union(
        ...     [
        ...         ("scaled", sq.StandardScaler()),
        ...         ("encoded", sq.OneHotEncoder()),
        ...     ]
        ... )
        >>> pipe = sq.Pipeline([union])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT scaled_price, ..., encoded_city_london, ... FROM __input__

        Single branch (passthrough, trivial case):

        >>> union = sq.Union([("only", sq.StandardScaler())])
        >>> pipe = sq.Pipeline([union])
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()
        ['only_price', 'only_qty']

    See Also:
        :class:`~sqlearn.core.pipeline.Pipeline`: Sequential composition.
        :class:`~sqlearn.core.transformer.Transformer`: Base class for steps.
        :class:`~sqlearn.scalers.standard.StandardScaler`: Scale numeric columns.
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Encode categoricals.
    """

    _default_columns: str = "all"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str | None = None  # auto-detect from branches

    def __init__(
        self,
        transformers: list[tuple[str, Transformer]],
    ) -> None:
        # Do NOT pass transformers to super().__init__ — it only accepts columns=
        super().__init__()
        self.transformers = _validate_transformers(transformers)

    # --- Classification ---

    def _classify(self) -> str:
        """Classify Union as dynamic if any branch is dynamic.

        Returns:
            ``'static'`` if all branches are static, ``'dynamic'`` otherwise.
        """
        for _, step in self.transformers:
            if step._classify() == "dynamic":  # noqa: SLF001
                return "dynamic"
        return "static"

    # --- Discovery (fit phase) ---

    def discover(
        self,
        columns: list[str],  # noqa: ARG002
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Collect scalar aggregations from all branches.

        Delegates to each branch's ``discover()``, prefixing parameter
        names with the branch name to avoid collisions between branches.

        Args:
            columns: All input columns (unused — each branch resolves its own).
            schema: Current input schema.
            y_column: Target column name, if any.

        Returns:
            Merged dict of ``'{branch}__{param}'`` to sqlglot aggregates
            from all branches.
        """
        result: dict[str, exp.Expression] = {}
        for name, step in self.transformers:
            branch_cols = _resolve_branch_columns(step, schema)
            branch_discover = step.discover(branch_cols, schema, y_column)
            for param_name, agg_expr in branch_discover.items():
                result[f"{name}__{param_name}"] = agg_expr
        return result

    def discover_sets(
        self,
        columns: list[str],  # noqa: ARG002
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Collect set-valued queries from all branches.

        Delegates to each branch's ``discover_sets()``, prefixing query
        names with the branch name to avoid collisions.

        Args:
            columns: All input columns (unused — each branch resolves its own).
            schema: Current input schema.
            y_column: Target column name, if any.

        Returns:
            Merged dict of ``'{branch}__{set_name}'`` to sqlglot SELECT
            queries from all branches.
        """
        result: dict[str, exp.Expression] = {}
        for name, step in self.transformers:
            branch_cols = _resolve_branch_columns(step, schema)
            branch_sets = step.discover_sets(branch_cols, schema, y_column)
            for set_name, set_query in branch_sets.items():
                result[f"{name}__{set_name}"] = set_query
        return result

    # --- Expression generation (transform phase) ---

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Merge expressions from all branches with branch-name prefixes.

        Each branch generates its own column expressions independently.
        All outputs are prefixed with the branch name to avoid collisions
        (e.g. ``scaled_price``, ``encoded_city_london``).

        Args:
            columns: All input columns (unused — each branch resolves its own).
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Merged dict of prefixed column expressions from all branches.
        """
        result: dict[str, exp.Expression] = {}

        for name, step in self.transformers:
            branch_exprs = _get_branch_expressions(step, exprs)
            for col_name, col_expr in branch_exprs.items():
                prefixed = f"{name}_{col_name}"
                result[prefixed] = col_expr

        return result

    # --- Schema management ---

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: union of all branch output schemas with prefixes.

        Drops all input columns and replaces them with prefixed outputs
        from each branch. Each branch's output columns are prefixed with
        the branch name (e.g. ``scaled_price``, ``encoded_city_london``).

        Args:
            schema: Input schema.

        Returns:
            Output schema with all branch outputs combined, all prefixed.
        """
        new_cols: dict[str, str] = {}

        for name, step in self.transformers:
            branch_schema = step.output_schema(schema)
            for col_name, col_type in branch_schema.columns.items():
                prefixed = f"{name}_{col_name}"
                new_cols[prefixed] = col_type

        return Schema(new_cols)

    # --- Fitted state distribution ---

    def _distribute_fitted_params(self) -> None:
        """Distribute learned params and sets back to branch transformers.

        After the compiler fits Union as a single step, this method splits
        the prefixed params_ and sets_ back to individual branches so their
        expressions() methods work correctly.

        Called internally during the fit lifecycle, after params_ and sets_
        are populated by the compiler.
        """
        if self.params_ is not None:
            _distribute_params(self.params_, self.transformers)
        if self.sets_ is not None:
            _distribute_sets(self.sets_, self.transformers)

    def _mark_branches_fitted(self, schema: Schema) -> None:
        """Mark all branch transformers as fitted with resolved columns.

        After the compiler fits Union, each branch transformer needs its own
        fitted state (columns_, input_schema_, output_schema_, _fitted)
        so that its expressions() method works correctly.

        Args:
            schema: Input schema that each branch receives.
        """
        for _, step in self.transformers:
            branch_cols = _resolve_branch_columns(step, schema)
            step.columns_ = branch_cols
            step.input_schema_ = schema
            step.output_schema_ = step.output_schema(schema)
            step._fitted = True  # noqa: SLF001

    # --- Sklearn introspection ---

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return init parameters. Supports deep nested params.

        Args:
            deep: If True, includes nested transformer params using
                ``'__'`` separator (e.g. ``scaled__with_mean``).

        Returns:
            Dict of parameter names to values.
        """
        params: dict[str, Any] = {"transformers": self.transformers}
        if deep:
            for name, step in self.transformers:
                step_params = step.get_params(deep=True)
                for key, value in step_params.items():
                    params[f"{name}__{key}"] = value
        return params

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names after fitting.

        Returns prefixed column names from all branches.

        Returns:
            List of prefixed column names.

        Raises:
            NotFittedError: If Union is not fitted.
        """
        if not self._fitted:
            msg = "Union is not fitted"
            raise NotFittedError(msg)
        if self.output_schema_ is None:
            msg = "output_schema_ is not set"
            raise NotFittedError(msg)
        return list(self.output_schema_.columns.keys())

    # --- Display ---

    def __repr__(self) -> str:
        """Show branch names and transformer types.

        Returns:
            Format: ``Union(scaled=StandardScaler, encoded=OneHotEncoder)``.
        """
        parts = [f"{name}={type(step).__name__}" for name, step in self.transformers]
        return f"Union({', '.join(parts)})"

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Returns:
            HTML string with Union details and branch info.
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        branches = ", ".join(f"{name}={type(step).__name__}" for name, step in self.transformers)
        return f"<div><strong>Union</strong>({branches}) [{fitted_str}]</div>"

    # --- Copying ---

    def clone(self) -> Union:
        """Create independent copy with cloned branches.

        Returns:
            New Union with cloned branch transformers.
        """
        cloned_transformers = [(name, step.clone()) for name, step in self.transformers]
        new = Union(cloned_transformers)
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

    # --- Serialization ---

    def __getstate__(self) -> dict[str, Any]:
        """Null out connections before pickling.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state


# ── Module-level helpers ──────────────────────────────────────────

_TUPLE_PAIR_LEN = 2


def _distribute_params(
    params: dict[str, Any],
    transformers: list[tuple[str, Transformer]],
) -> None:
    """Distribute prefixed params to individual branch transformers.

    Args:
        params: Union's merged params_ dict with branch-prefixed keys.
        transformers: List of (name, transformer) branch pairs.
    """
    for name, step in transformers:
        prefix = f"{name}__"
        branch_params: dict[str, Any] = {}
        for key, value in params.items():
            if key.startswith(prefix):
                branch_params[key[len(prefix) :]] = value
        if branch_params:
            step.params_ = branch_params


def _distribute_sets(
    sets: dict[str, list[dict[str, Any]]],
    transformers: list[tuple[str, Transformer]],
) -> None:
    """Distribute prefixed sets to individual branch transformers.

    Args:
        sets: Union's merged sets_ dict with branch-prefixed keys.
        transformers: List of (name, transformer) branch pairs.
    """
    for name, step in transformers:
        prefix = f"{name}__"
        branch_sets: dict[str, list[dict[str, Any]]] = {}
        for key, value in sets.items():
            if key.startswith(prefix):
                branch_sets[key[len(prefix) :]] = value
        if branch_sets:
            step.sets_ = branch_sets


def _validate_transformers(
    transformers: list[tuple[str, Transformer]],
) -> list[tuple[str, Transformer]]:
    """Validate and normalize the transformers list.

    Args:
        transformers: List of (name, transformer) pairs.

    Returns:
        Validated list of (name, transformer) pairs.

    Raises:
        InvalidStepError: If list is empty, contains non-Transformers,
            non-tuples, or has duplicate names.
    """
    if not transformers:
        msg = "Union requires at least one transformer branch"
        raise InvalidStepError(msg)

    seen_names: set[str] = set()
    result: list[tuple[str, Transformer]] = []

    for item in transformers:
        if not isinstance(item, tuple) or len(item) != _TUPLE_PAIR_LEN:  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                f"Union transformers must be (name, transformer) tuples, got {type(item).__name__}"
            )
            raise InvalidStepError(msg)

        name, step = item
        if not isinstance(name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Branch name must be a string, got {type(name).__name__}"
            raise InvalidStepError(msg)

        if not isinstance(step, Transformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Branch {name!r} is not a Transformer: {type(step)}"
            raise InvalidStepError(msg)

        if name in seen_names:
            msg = f"Duplicate branch name: {name!r}"
            raise InvalidStepError(msg)

        seen_names.add(name)
        result.append((name, step))

    return result


def _resolve_branch_columns(step: Transformer, schema: Schema) -> list[str]:
    """Resolve columns for a branch transformer against the input schema.

    Args:
        step: Branch transformer.
        schema: Input schema.

    Returns:
        Resolved list of column names for this branch.
    """
    col_spec = step._resolve_columns_spec()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if col_spec is None:
        return list(schema.columns.keys())
    return resolve_columns(schema, col_spec)


def _get_branch_expressions(
    step: Transformer,
    exprs: dict[str, exp.Expression],
) -> dict[str, exp.Expression]:
    """Get output expressions from a branch transformer.

    Calls the branch's expressions() method to get its column
    transformations, then determines which columns appear in its
    output schema to return only those.

    Args:
        step: Fitted branch transformer.
        exprs: Current expression dict for ALL input columns.

    Returns:
        Dict of output column expressions for this branch.
    """
    if step.columns_ is None:
        return {}

    try:
        modified = step.expressions(step.columns_, exprs)
    except NotImplementedError:
        modified = {}

    # Start with passthrough of all input columns
    branch_result = dict(exprs)
    branch_result.update(modified)

    # Filter to output schema columns
    if step.output_schema_ is not None:
        output_cols = set(step.output_schema_.columns.keys())
        branch_result = {k: v for k, v in branch_result.items() if k in output_cols}

    return branch_result
