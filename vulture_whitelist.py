"""Vulture whitelist — false positives for base class interface parameters."""

# Transformer base class interface methods have parameters that are only
# used by subclass overrides, not by the default implementation.
y_column  # noqa
exprs  # noqa
input_query  # noqa
deep  # noqa — get_params(deep=True) used when Pipeline supports nested params

# Stub method parameters — used when Pipeline/Compiler/Backend land.
data  # noqa
y  # noqa
backend  # noqa
out  # noqa
batch_size  # noqa
dtype  # noqa — not yet whitelisted above since it's a new param name
exclude_target  # noqa
kwargs  # noqa
dialect  # noqa
table  # noqa
