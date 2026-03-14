"""Input resolution for sqlearn.

Normalizes user input (file paths, table names, DataFrames) to a
source name string that a Backend can query.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlearn.core.backend import Backend

_input_counter: int = 0
_input_lock = threading.Lock()


def _next_input_name() -> str:
    """Generate the next auto-incremented input table name.

    Thread-safe via module-level lock.

    Returns:
        Name like ``'__sqlearn_input_0'``, ``'__sqlearn_input_1'``, etc.
    """
    global _input_counter  # noqa: PLW0603
    with _input_lock:
        name = f"__sqlearn_input_{_input_counter}"
        _input_counter += 1
    return name


def resolve_input(
    data: object,
    backend: Backend,
    *,
    table_name: str | None = None,
) -> str:
    """Resolve user input to a queryable source name.

    Takes whatever the user passes to fit()/transform() and returns
    a source name string that the Backend can query.

    Args:
        data: Input data — file path (str), table name (str),
            or pandas DataFrame.
        backend: Backend instance for registering DataFrames.
        table_name: Override auto-generated name for DataFrames.

    Returns:
        Source name string usable by the backend.

    Raises:
        TypeError: If data type is not supported.
    """
    # String: file path or table name
    if isinstance(data, str):
        return data

    # pandas DataFrame
    type_name = type(data).__module__ + "." + type(data).__qualname__
    if type_name.startswith("pandas.") and "DataFrame" in type(data).__name__:
        name = table_name if table_name is not None else _next_input_name()
        backend.register(data, name)
        return name

    msg = (
        f"Unsupported input type: {type(data).__name__}. "
        "Expected str (file path or table name) or pandas DataFrame."
    )
    raise TypeError(msg)
