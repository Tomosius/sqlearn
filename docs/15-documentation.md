> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Decisions](14-decisions.md)

## 15. Documentation Strategy

### 15.1 Principles

sqlearn documentation follows the same standard as scikit-learn, NumPy, and other
professional Python libraries:

- **Every public API is documented** — classes, functions, parameters, return types, exceptions
- **Examples on everything** — not just descriptions, but runnable code showing how and why
- **Design decisions explained** — why each approach was chosen, what alternatives were considered
- **Two audiences** — expert ML engineers (concise API reference) and newcomers (guided tutorials)
- **Documentation grows with code** — new modules ship with docs, not as a separate "docs sprint"

### 15.2 Documentation Layers

| Layer | What it contains | Audience |
|---|---|---|
| **API Reference** | Auto-generated from docstrings. Every class, method, parameter. Type signatures, defaults, exceptions, examples. | Expert users looking up specifics |
| **User Guide** | Conceptual explanations. How the compiler works. How expressions compose. How layers resolve. Why SQL instead of numpy. | Users wanting to understand the system |
| **Tutorials** | Step-by-step walkthroughs. "Build your first pipeline." "Migrate from sklearn." "Create a custom transformer." | New users getting started |
| **Examples Gallery** | Runnable scripts with real datasets. One per transformer, one per pattern (impute+scale, encode+select, etc.) | Users looking for copy-paste starting points |
| **Design Decisions** | Why we chose X over Y. Trade-offs considered. Links to architecture docs. | Contributors and advanced users |
| **Architecture Deep-Dive** | Compiler internals, expression composition, CTE promotion, layer resolution, fit batching. | Contributors and people evaluating the library |

### 15.3 Docstring Standard

All docstrings follow Google style. Enforced by `interrogate` at 95%+ coverage.

Every public class/function docstring MUST include:

```python
def resolve_input(
    data: Any,
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

    Examples:
        >>> backend = DuckDBBackend()
        >>> resolve_input("train.parquet", backend)
        'train.parquet'
        >>> resolve_input("my_table", backend)
        'my_table'
        >>> import pandas as pd
        >>> df = pd.DataFrame({"x": [1, 2, 3]})
        >>> name = resolve_input(df, backend)
        >>> name.startswith("__sqlearn_input_")
        True
    """
```

**Required sections:**
- One-line summary (imperative mood)
- Extended description (when non-obvious)
- `Args:` — every parameter, with type and meaning
- `Returns:` — what and why
- `Raises:` — every exception that can be raised
- `Examples:` — at least one runnable doctest

### 15.4 Phased Rollout

#### End of M2 (v0.1.0) — Infrastructure + API Reference

- Set up mkdocs with mkdocstrings (Material for MkDocs theme)
- Auto-generate API reference from docstrings for all shipped modules:
  schema, transformer, backend, io, errors, compiler, pipeline
- Add doctest examples to all existing public APIs
- Host locally during development (`mkdocs serve`)

**Why now:** Docstrings already exist at 100% coverage. The infrastructure is a one-time
setup (~2 hours). Every future module automatically gets API docs.

#### M3 (v0.2.0) — User Guide + Tutorials

- **User guide chapters:**
  - How expression composition works (with SQL examples at each step)
  - How the fit/transform lifecycle works (layers, batching, classification)
  - How column routing works (defaults, selectors, explicit)
  - How custom transformers work (three levels, with full examples)
  - Why SQL over numpy (performance model, memory model, bigger-than-RAM)
- **Tutorials:**
  - "Your first pipeline" — Imputer + StandardScaler + OneHotEncoder
  - "Migrating from sklearn" — side-by-side comparison
  - "Creating custom transformers" — all three levels
  - "Working with different data sources" — parquet, CSV, DataFrames, tables
- **Design decisions guide:**
  - Why one base class instead of sklearn's eight
  - Why sqlglot ASTs instead of raw SQL strings
  - Why expressions() returns only modified columns
  - Why discover() vs sklearn's fit() pattern
  - Why DuckDB as default engine

#### M4 (v0.3.0) — Examples Gallery

- Runnable examples per transformer category (scalers, encoders, features, etc.)
- Real dataset examples (Kaggle-style workflows)
- Performance comparison examples (sqlearn vs sklearn on same data)
- Each example: problem statement, code, output, explanation of what the SQL does

#### M7 (v1.0.0) — Full Documentation Site

- Hosted documentation (Read the Docs or GitHub Pages)
- Versioned docs (one per release)
- Search functionality
- Complete API reference with examples on every public class/function
- Architecture deep-dive: compiler internals with diagrams
- Migration guide: sklearn → sqlearn with side-by-side code
- Performance benchmarks with methodology
- Contributing guide for the open-source community

### 15.5 Documentation as Tests

Doctest examples serve double duty — they're documentation AND tests:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--doctest-modules"
```

This means every `Examples:` section in a docstring is a test that runs in CI.
If the API changes and the example breaks, CI catches it. Documentation can never
go stale.

### 15.6 What NOT to Document

- Internal/private methods (single underscore prefix) — docstrings for maintainers, not in public docs
- Implementation details that change frequently — link to source instead
- Anything already in the architecture docs (`docs/01-14`) — those are design docs, not user docs
- Third-party library internals (DuckDB, sqlglot) — link to their docs
