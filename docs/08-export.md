> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Analysis & Recommendations](07-analysis.md) | Next: [Error Handling](09-error-handling.md)

## 9. Export & Deployment

### 9.1 Export Formats

```python
# SQL — any dialect via sqlglot
pipe.to_sql()                        # DuckDB (default)
pipe.to_sql(dialect="snowflake")     # Snowflake
pipe.to_sql(dialect="bigquery")      # BigQuery
pipe.to_sql(dialect="postgres")      # PostgreSQL

# dbt model
pipe.to_dbt("feature_pipeline", output_dir="./dbt/models/")

# Config — human-readable, version-controllable
pipe.to_config("pipeline.yaml")
pipe = sq.Pipeline.from_config("pipeline.yaml")

# Binary — fast save/load of fitted pipeline
pipe.save("pipeline.sqlearn")
pipe = sq.load("pipeline.sqlearn")

# Standalone Python — no sqlearn dependency needed
pipe.export("pipeline.py")   # contains baked SQL + minimal duckdb script
```

### 9.2 Pipeline Versioning

Every saved pipeline embeds version metadata for reproducibility and compatibility:

```python
pipe.save("pipeline.sqlearn")
# Embeds:
#   sqlearn_version: "0.4.2"
#   format_version: 1
#   python_version: "3.11.5"
#   created_at: "2025-01-15T10:30:00Z"
#   sql_hash: "a1b2c3d4..."          # hash of compiled SQL — detects silent changes
#   schema_in: {columns: {price: DOUBLE, city: VARCHAR, ...}}
#   schema_out: {columns: {price: DOUBLE, city_london: INTEGER, ...}}
```

**On `sq.load()`, compatibility is checked:**

```python
pipe = sq.load("pipeline.sqlearn")
# If sqlearn version differs:
#   Same major version → load with info log
#   Different major version → load with UserWarning
#   Format version incompatible → raise LoadError with migration instructions

# Verify SQL hasn't silently changed:
pipe = sq.load("pipeline.sqlearn", verify=True)
# Recompiles pipeline from params, compares SQL hash to saved hash.
# If different: UserWarning — "Pipeline SQL differs from saved version.
#   This may be due to a sqlglot or sqlearn update. Review with pipe.to_sql()."
```

**Config format (YAML) is the human-readable archive:**

```yaml
# pipeline.yaml — version-controlled, human-readable
sqlearn_version: "0.4.2"
format_version: 1

steps:
  - type: Imputer
    params:
      strategy: median
    fitted:
      price__median: 42.5
      city__mode: "London"

  - type: StandardScaler
    params: {}
    fitted:
      price__mean: 42.5
      price__std: 12.3

  - type: OneHotEncoder
    params:
      max_categories: 30
    fitted:
      city__categories: ["London", "Paris", "Berlin"]

schema_in:
  price: DOUBLE
  city: VARCHAR
  age: INTEGER

schema_out:
  price: DOUBLE
  city_london: INTEGER
  city_paris: INTEGER
  city_berlin: INTEGER
  age: INTEGER
```

### 9.3 Frozen Pipelines — Immutable Deployment Artifacts

`pipe.freeze()` returns a `FrozenPipeline`: an immutable, pre-compiled, deployment-ready
artifact. This is the bridge from experimentation to production.

**Why freeze exists:**

| Problem | Without freeze | With freeze |
|---|---|---|
| Accidental refit | `pipe.fit(new_data)` silently changes learned params | `FrozenPipeline.fit()` raises `FrozenError` |
| Version drift | sqlearn update changes SQL generation | Frozen SQL is pre-compiled, never regenerated |
| Schema mismatch | New data has unexpected columns → cryptic SQL error | Frozen pipeline validates input schema before execution |
| Reproducibility | Same pipeline + different sqlearn version = different output? | Frozen SQL hash guarantees identical computation |
| Cross-fold schema corruption | See Section 4.7b | Freeze bakes schema from full training data |

**How to freeze:**

```python
pipe.fit("train.parquet", y="target")

# Freeze: pre-compile SQL, lock params, validate schema
frozen = pipe.freeze()

# Frozen pipeline transforms data but refuses to learn:
X = frozen.transform("test.parquet")     # ✓ works
frozen.fit("new_data.parquet")            # ✗ raises FrozenError

# Schema validation on transform:
frozen.transform("wrong_schema.parquet")
# SchemaError: Expected columns {price: DOUBLE, city: VARCHAR, age: INTEGER}
# but got {amount: DOUBLE, location: VARCHAR}. Missing: price, city. Extra: amount, location.

# Pre-compiled SQL — no compilation overhead at transform time:
frozen.sql_                              # the exact SQL string (pre-compiled)
frozen.sql_hash_                         # SHA256 of the SQL — reproducibility fingerprint

# Save frozen pipeline — smallest, fastest format:
frozen.save("pipeline_v1.frozen.sqlearn")
frozen = sq.load_frozen("pipeline_v1.frozen.sqlearn")
```


**What `freeze()` bakes in:**

| Component | Status | Notes |
|---|---|---|
| Learned parameters (means, stds, categories) | Baked as literals in SQL | No `self.params_` lookup at transform time |
| Output schema | Fixed | Input schema validated before execution |
| SQL string | Pre-compiled for target dialect | No sqlglot compilation at transform time |
| SQL hash | Stored | Reproducibility fingerprint |
| Step classifications | Frozen | No re-inspection |
| Column order | Fixed | Deterministic output ordering |

**FrozenPipeline is the deployment artifact.** Use `Pipeline` during development
(fit, iterate, experiment). Use `FrozenPipeline` in production (transform only,
validated, immutable, fast).

```python
# Development workflow:
pipe = sq.Pipeline([...])
pipe.fit("train.parquet", y="target")
pipe.describe()           # inspect, iterate

# Production workflow:
frozen = pipe.freeze()
frozen.save("prod_pipeline.frozen.sqlearn")

# In production code:
frozen = sq.load_frozen("prod_pipeline.frozen.sqlearn")
X = frozen.transform("incoming_data.parquet")   # validated, pre-compiled, fast
model = joblib.load("model.pkl")
predictions = model.predict(X)
```

**Freeze and cross-validation schema safety:**

`freeze()` bakes the FULL training data's schema discovery (Phase 1 from Section 4.7b).
This means the frozen pipeline's OneHotEncoder always produces ALL categories seen in
training — even if a specific production batch doesn't contain all categories. This is
exactly the right behavior: the model was trained expecting those columns, and the
frozen pipeline guarantees they're always present (with value 0 for absent categories).

### 9.4 Security Model

**SQL injection:** `sq.Expression()` and `sq.Filter()` accept user-provided SQL strings.
These are parsed through sqlglot (which validates syntax) but are NOT sanitized for
malicious content. sqlearn trusts all SQL inputs.

**Rule: Never pass untrusted user input to `Expression()`, `Filter()`, or any sqlearn
function that accepts SQL strings.** sqlearn is a data science library, not a web
framework. If you build a web service on top of sqlearn, sanitize inputs yourself.

```python
# SAFE: hardcoded SQL from your own code
sq.Filter("price > 0 AND city != 'Unknown'")

# DANGEROUS: user-submitted SQL from a web form
user_input = request.form["filter"]       # could be "1=1; DROP TABLE data"
sq.Filter(user_input)                      # ← NEVER DO THIS

# If you must accept user filters, use parameterized column/value:
sq.Filter(column="price", op=">", value=0)  # safe — no raw SQL
```

`sq.studio()` runs on localhost only (`127.0.0.1`) and is not designed for multi-user
or public-facing deployment. The Studio API endpoints accept SQL-like filter expressions
from the frontend — this is safe because the frontend is local and trusted.

---
