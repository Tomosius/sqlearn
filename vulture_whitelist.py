"""Vulture whitelist — false positives for base class interface parameters."""

# Transformer base class interface methods have parameters that are only
# used by subclass overrides, not by the default implementation.
y_column  # noqa
exprs  # noqa
input_query  # noqa
