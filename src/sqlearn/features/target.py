"""TargetTransform -- apply transformations to the target column for regression.

Supports log, sqrt, and Box-Cox transformations via SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema

# Threshold for auto lambda: if abs(log_mean) < this, use lambda=0 (log)
_BOXCOX_NEAR_ZERO_THRESHOLD = 0.1


class TargetTransform(Transformer):
    """Apply mathematical transformations to target columns via SQL.

    Useful for transforming skewed regression targets to approximate
    normality, improving model performance. Supports:

    - **log**: ``LN(col + 1)`` -- natural log with +1 offset for zero safety.
    - **sqrt**: ``SQRT(col)`` -- square root transformation.
    - **boxcox**: ``(POW(col, lambda) - 1) / lambda`` -- Box-Cox power
      transform with learned or fixed lambda.

    **Log** generated SQL::

        SELECT LN(price + 1) AS price FROM __input__

    **Box-Cox** generated SQL::

        SELECT (POW(price, 0.5) - 1) / 0.5 AS price FROM __input__

    Args:
        method: Transformation method. ``"log"`` (default), ``"sqrt"``,
            or ``"boxcox"``.
        lambda_: Box-Cox lambda parameter. Only used when ``method="boxcox"``.
            If ``"auto"`` (default), the optimal lambda is learned from data
            during ``fit()``. If a number, that fixed value is used.
        columns: Column specification. Required -- there is no default
            column routing since this transformer targets specific columns
            (typically the ``y`` column).

    Raises:
        ValueError: If ``method`` is not ``"log"``, ``"sqrt"``, or ``"boxcox"``.
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()`` (only for ``method="boxcox"`` with ``lambda_="auto"``).

    Examples:
        Log-transform a target column:

        >>> import sqlearn as sq
        >>> transform = sq.TargetTransform(columns=["price"])
        >>> pipe = sq.Pipeline([transform])
        >>> pipe.fit("data.parquet", y="price")
        >>> sql = pipe.to_sql()  # LN(price + 1) AS price

        Square root transform:

        >>> transform = sq.TargetTransform(method="sqrt", columns=["price"])
        >>> pipe = sq.Pipeline([transform])

        Box-Cox with auto lambda:

        >>> transform = sq.TargetTransform(method="boxcox", columns=["price"])
        >>> pipe = sq.Pipeline([transform])
        >>> pipe.fit("data.parquet")

        Box-Cox with fixed lambda:

        >>> transform = sq.TargetTransform(method="boxcox", lambda_=0.5, columns=["price"])

    See Also:
        :class:`~sqlearn.scalers.standard.StandardScaler`: Standardize features.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def _classification(self) -> str:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        """Classify as static or dynamic based on method.

        Returns:
            ``'static'`` for log and sqrt (no data learning needed),
            ``'dynamic'`` for boxcox with auto lambda (needs to learn
            lambda from data).
        """
        if self.method == "boxcox" and self.lambda_ == "auto":
            return "dynamic"
        return "static"

    def __init__(
        self,
        *,
        method: str = "log",
        lambda_: str | float = "auto",
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize TargetTransform.

        Args:
            method: Transformation method (``"log"``, ``"sqrt"``, or ``"boxcox"``).
            lambda_: Box-Cox lambda (``"auto"`` or a float). Only used
                when ``method="boxcox"``.
            columns: Column specification. Required.

        Raises:
            ValueError: If method is not recognized.
        """
        super().__init__(columns=columns)
        if method not in ("log", "sqrt", "boxcox"):
            msg = f"method must be 'log', 'sqrt', or 'boxcox', got {method!r}"
            raise ValueError(msg)
        self.method = method
        self.lambda_ = lambda_

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn Box-Cox lambda from data if method is boxcox with auto lambda.

        Approximates optimal lambda using the mean of log values. This is a
        simplified heuristic -- for production use, specify ``lambda_``
        explicitly after offline optimization.

        For log and sqrt methods, returns empty dict (no learning needed).

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__log_mean'`` to ``AVG(LN(col))`` for
            Box-Cox auto lambda. Empty dict for log and sqrt methods.
        """
        if self.method != "boxcox" or self.lambda_ != "auto":
            return {}

        result: dict[str, exp.Expression] = {}
        for col in columns:
            # Learn mean of log values for lambda approximation
            result[f"{col}__log_mean"] = exp.Avg(
                this=exp.Ln(this=exp.Column(this=col)),
            )
        return result

    def _get_lambda(self, col: str) -> float:
        """Get the Box-Cox lambda for a column.

        Args:
            col: Column name.

        Returns:
            Lambda value (learned or fixed).
        """
        if isinstance(self.lambda_, int | float):
            return float(self.lambda_)

        # Auto: approximate lambda from learned log mean
        params = self.params_ or {}
        log_mean = float(params.get(f"{col}__log_mean", 0.0))
        # Simple heuristic: if log_mean is near 0, data is already
        # roughly log-normal, so lambda ~ 0 (log transform).
        # Otherwise use a basic approximation.
        if abs(log_mean) < _BOXCOX_NEAR_ZERO_THRESHOLD:
            return 0.0
        return round(1.0 / (1.0 + abs(log_mean)), 4)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for target transformation.

        - log: ``LN(col + 1)``
        - sqrt: ``SQRT(col)``
        - boxcox: ``(POW(col, lambda) - 1) / lambda`` when lambda != 0,
          ``LN(col)`` when lambda == 0

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            col_expr = exprs[col]

            if self.method == "log":
                # LN(col + 1)
                result[col] = exp.Ln(
                    this=exp.Paren(
                        this=exp.Add(
                            this=col_expr,
                            expression=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                        ),
                    ),
                )
            elif self.method == "sqrt":
                # SQRT(col)
                result[col] = exp.Sqrt(this=col_expr)
            else:  # boxcox
                lam = self._get_lambda(col)
                if lam == 0.0:
                    # LN(col)
                    result[col] = exp.Ln(this=col_expr)
                else:
                    # (POW(col, lambda) - 1) / lambda
                    result[col] = exp.Div(
                        this=exp.Paren(
                            this=exp.Sub(
                                this=exp.Pow(
                                    this=col_expr,
                                    expression=exp.Literal.number(lam),  # pyright: ignore[reportUnknownMemberType]
                                ),
                                expression=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                            ),
                        ),
                        expression=exp.Literal.number(lam),  # pyright: ignore[reportUnknownMemberType]
                    )
        return result
