> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Compiler](04-compiler.md) | Next: [Model Integration](06-model-integration.md)

## 6. Built-in Transformers

### 6.1 Preprocessing

| Transformer | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `Imputer(strategy=)` | all | Yes | No | `COALESCE(col, learned_value)` |

**Imputer — Four Calling Conventions:**

```python
# 1. No args — auto-detect: numeric→median, categorical→most_frequent
sq.Imputer()

# 2. Single strategy — applies to all columns
sq.Imputer(strategy="mean")

# 3. Columns + strategy — explicit targets
sq.Imputer(columns=["price", "qty"], strategy="median")

# 4. Dict — per-column strategies (the power move)
sq.Imputer({
    "price": "mean",        # string = strategy name
    "quantity": "median",
    "category": "most_frequent",
    "status": "active",     # non-numeric string that isn't a strategy name = constant fill
    "score": 0,             # number = constant fill
})
```

The dict API is auto-detected: if the constructor receives a dict as the first
positional argument, it becomes per-column mode. String values are checked against
known strategies (`"mean"`, `"median"`, `"most_frequent"`, `"zero"`). If not a known
strategy, it's treated as a constant fill value. Numbers are always constant fill.

**SQL output (dict mode):**

```sql
SELECT
    COALESCE(price, 42.5) AS price,           -- mean of price
    COALESCE(quantity, 10.0) AS quantity,      -- median of quantity
    COALESCE(category, 'Electronics') AS category,  -- most_frequent
    COALESCE(status, 'active') AS status,     -- constant
    COALESCE(score, 0) AS score               -- constant
FROM data
```
| `StandardScaler()` | numeric | Yes | No | `(col - mean) / std` |
| `MinMaxScaler()` | numeric | Yes | No | `(col - min) / (max - min)` |
| `RobustScaler()` | numeric | Yes | No | `(col - median) / iqr` |
| `MaxAbsScaler()` | numeric | Yes | No | `col / max_abs` |
| `Normalizer(norm=)` | numeric | No | No | `col / sqrt(sum(col^2))` per row — uses `query()` because L2 norm is cross-column (requires referencing multiple columns in one expression, which needs its own CTE even though no data is learned). Document this in docstring so users expecting "one SELECT, zero CTEs" understand why Normalizer adds a CTE. |
| `Binarizer(threshold=)` | numeric | No | No | `CASE WHEN col > t THEN 1 ELSE 0 END` |
| `KBinsDiscretizer(n=)` | numeric | Yes | No | `NTILE(n) OVER (ORDER BY col)` or equal-width bins |

### 6.2 Encoders

| Encoder | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `OneHotEncoder(max_categories=30, sparse=False, encode_nan=False)` | categorical | Yes | Yes | N `CASE WHEN col = 'val' THEN 1 ELSE 0 END` columns (dense by default — sklearn changed to dense in v1.2 because sparse caused too many downstream issues with models, numpy ops, and pandas). When `encode_nan=True`, NaN gets its own column: `CASE WHEN col IS NULL THEN 1 ELSE 0 END AS col_nan` (instead of all-zeros row). Default `False` matches sklearn behavior. |
| `OrdinalEncoder(order=)` | categorical | Yes | No | `CASE WHEN col = 'S' THEN 0 WHEN 'M' THEN 1 ... END` |
| `TargetEncoder(smooth=)` | categorical | Yes | No | Smoothed target mean per category (needs `y`) |
| `HashEncoder(n_bins=)` | categorical | No | No | `HASH(col) % n_bins` (zero-fit!) |
| `FrequencyEncoder()` | categorical | Yes | No | `COUNT(*) OVER (PARTITION BY col) / COUNT(*) OVER ()` |
| `BinaryEncoder()` | categorical | Yes | Yes | Binary decomposition of ordinal index |
| `AutoEncoder(thresholds=)` | categorical | Yes | Varies | Auto-selects strategy per column by cardinality |

**AutoEncoder — the killer feature sklearn doesn't have:**

```python
sq.AutoEncoder()
# During fit, for each categorical column:
#   < 20 unique  → OneHotEncoder
#   20-500       → TargetEncoder (if y provided) or FrequencyEncoder
#   > 500        → HashEncoder
```

User just says "encode my categoricals." sqlearn picks the optimal strategy per column.

**TargetEncoder — `y` column usage:**

TargetEncoder is the only built-in encoder that needs the target column. The `y` column
name flows from `pipeline.fit("data", y="price")` into `discover(columns, schema, y_column)`:

```python
class TargetEncoder(Transformer):
    _default_columns = "categorical"

    def discover(self, columns, schema, y_column=None):
        if y_column is None:
            raise FitError(
                "TargetEncoder requires a target column. "
                "Pass y='column_name' to pipeline.fit()."
            )
        aggs = {}
        for col in columns:
            # Smoothed target mean per category — computed via SQL
            aggs[f"{col}__target_mean"] = ...  # AVG(y) GROUP BY col, smoothed
            aggs[f"{col}__global_mean"] = exp.Avg(this=exp.Column(this=y_column))
        return aggs
```

The target column is automatically excluded from `transform()` output — no manual
`df.drop("target")` needed.

### 6.3 Feature Engineering — Arithmetic & Math

| Transformer | Default Columns | Dynamic? | SQL Pattern |
|---|---|---|---|
| `Log(base=e)` | explicit | No | `LN(col + 1)` |
| `Sqrt()` | explicit | No | `SQRT(col)` |
| `Power(n=)` | explicit | No | `POW(col, n)` |
| `Clip(lower=, upper=)` | explicit | No | `GREATEST(LEAST(col, upper), lower)` |
| `Abs()` | explicit | No | `ABS(col)` |
| `Round(decimals=)` | explicit | No | `ROUND(col, n)` |
| `Sign()` | explicit | No | `SIGN(col)` |
| `Add(a, b, name=)` | explicit | No | `a + b AS name` |
| `Multiply(a, b, name=)` | explicit | No | `a * b AS name` |
| `Ratio(a, b, name=)` | explicit | No | `a / NULLIF(b, 0) AS name` |
| `Diff(a, b, name=)` | explicit | No | `a - b AS name` |
| `Modulo(a, b, name=)` | explicit | No | `a % b AS name` |
| `Expression(sql)` | explicit | No | User-provided SQL, parsed through sqlglot |
| `PolynomialFeatures(degree=)` | numeric | No | All degree-N interaction terms |

### 6.4 Feature Engineering — String & Structured Data

| Transformer | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `StringLength()` | explicit | No | Yes (adds) | `LENGTH(col) AS col_length` |
| `StringLower()` | explicit | No | No | `LOWER(col)` |
| `StringUpper()` | explicit | No | No | `UPPER(col)` |
| `StringTrim()` | explicit | No | No | `TRIM(col)` |
| `StringContains(pattern=)` | explicit | No | Yes (adds) | `CASE WHEN col LIKE '%pat%' THEN 1 ELSE 0 END` |
| `StringReplace(old, new)` | explicit | No | No | `REPLACE(col, old, new)` |
| `StringExtract(regex=)` | explicit | No | Yes (adds) | `REGEXP_EXTRACT(col, pattern, group)` |
| `StringSplit(by=, max_parts=)` | explicit | No | Yes (adds N) | `SPLIT_PART(col, by, n)` for each part |

**`StringSplit` — split one column into N columns:**

```python
# Split a comma-separated column into individual feature columns
sq.StringSplit(columns=["tags"], by=",", max_parts=3)
# "electronics,sale,new" → tags_1="electronics", tags_2="sale", tags_3="new"

# Auto-detect max_parts from data (uses discover())
sq.StringSplit(columns=["tags"], by=",", max_parts="auto")
# Runs: SELECT MAX(LENGTH(tags) - LENGTH(REPLACE(tags, ',', '')) + 1) FROM data
# Learns max_parts=4 → creates 4 columns

# Drop original column after splitting
sq.StringSplit(columns=["tags"], by=",", max_parts=3, keep_original=False)

# Custom output names
sq.StringSplit(columns=["full_name"], by=" ", max_parts=2, names=["first_name", "last_name"])
```

SQL output:
```sql
SELECT
    TRIM(SPLIT_PART(tags, ',', 1)) AS tags_1,
    TRIM(SPLIT_PART(tags, ',', 2)) AS tags_2,
    TRIM(SPLIT_PART(tags, ',', 3)) AS tags_3,
    -- NULL if fewer than N parts (SPLIT_PART returns '' → NULLIF)
    NULLIF(TRIM(SPLIT_PART(tags, ',', 3)), '') AS tags_3
FROM data
```

**Structured data extractors:**

| Transformer | Input | Output Columns | SQL Pattern |
|---|---|---|---|
| `JsonExtract(fields=)` | JSON string | One column per field | `json_col->>'field'` |
| `URLParts()` | URL string | `domain`, `path`, `scheme`, `query` | DuckDB URL functions / regex |
| `EmailParts()` | Email string | `local_part`, `domain` | `SPLIT_PART(col, '@', 1/2)` |
| `IPParts()` | IP address | `octet_1..4`, `is_private` | `SPLIT_PART(col, '.', n)` |

```python
# JSON extraction — common for API data
sq.JsonExtract(columns=["metadata"], fields=["source", "version", "score"])
# metadata: '{"source":"web","version":2,"score":0.8}'
# → metadata_source="web", metadata_version=2, metadata_score=0.8

# URL breakdown
sq.URLParts(columns=["referrer_url"])
# "https://www.google.com/search?q=test"
# → referrer_url_scheme="https", referrer_url_domain="google.com",
#   referrer_url_path="/search", referrer_url_query="q=test"

# Email splitting
sq.EmailParts(columns=["email"])
# "user@company.com" → email_local="user", email_domain="company.com"

# IP address breakdown
sq.IPParts(columns=["ip_address"])
# "192.168.1.100" → ip_octet_1=192, ..., ip_is_private=1
```

All extractors have `keep_original=True` by default. Set `keep_original=False` to
drop the source column after extraction.

### 6.5 Feature Engineering — Datetime & Temporal

| Transformer | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `DateParts(parts=)` | temporal | No | Yes (adds) | `EXTRACT(part FROM col)` |
| `DateDiff(start, end, unit=)` | explicit | No | Yes (adds) | `DATEDIFF(unit, start, end)` |
| `IsWeekend()` | temporal | No | Yes (adds) | `EXTRACT(DOW FROM col) IN (0, 6)` |
| `IsHoliday(country=)` | temporal | No | Yes (adds) | Lookup against holiday table |
| `TimeSinceEvent(reference=)` | temporal | No | Yes (adds) | `DATEDIFF('day', reference, col)` |
| `CyclicEncode(period=)` | explicit | No | Yes (adds 2) | `SIN(2π × col / period)`, `COS(...)` |
| `AutoDatetime(granularity=)` | temporal | No | Yes (adds N) | Auto-extract all useful parts |

**`AutoDatetime` — type-aware automatic datetime expansion:**

The key feature you asked about. Detects temporal columns and auto-expands based on
the actual data range and granularity:

```python
# Auto-detect and expand ALL datetime columns
sq.AutoDatetime()

# Explicit columns
sq.AutoDatetime(columns=["created_at", "updated_at"])

# Control granularity
sq.AutoDatetime(granularity="day")     # year, month, day, dayofweek, is_weekend
sq.AutoDatetime(granularity="hour")    # + hour, is_business_hours
sq.AutoDatetime(granularity="minute")  # + minute
sq.AutoDatetime(granularity="auto")    # detect from data (default)
```

**Auto-detection logic (during `discover()`):**

```python
# AutoDatetime inspects data range + distinct values to pick granularity:
#
#   Range > 365 days AND < 100 distinct dates  → "month" granularity
#     → year, quarter, month, dayofweek, is_weekend
#
#   Range > 365 days AND ≥ 100 distinct dates  → "day" granularity
#     → year, quarter, month, day, dayofweek, is_weekend, day_of_year
#
#   Has time component (hours vary)             → "hour" granularity
#     → year, month, day, hour, dayofweek, is_weekend, is_business_hours
#
#   Has sub-hour precision (minutes vary)       → "minute" granularity
#     → year, month, day, hour, minute, dayofweek, is_weekend
#
#   TIMESTAMP type                              → all of the above
#   DATE type                                   → up to day granularity
```

SQL output for a TIMESTAMP column with `granularity="hour"`:

```sql
SELECT
    EXTRACT(YEAR FROM created_at)                AS created_at_year,
    EXTRACT(QUARTER FROM created_at)             AS created_at_quarter,
    EXTRACT(MONTH FROM created_at)               AS created_at_month,
    EXTRACT(DAY FROM created_at)                 AS created_at_day,
    EXTRACT(HOUR FROM created_at)                AS created_at_hour,
    EXTRACT(DOW FROM created_at)                 AS created_at_dayofweek,
    CASE WHEN EXTRACT(DOW FROM created_at) IN (0, 6) THEN 1 ELSE 0 END
                                                 AS created_at_is_weekend,
    CASE WHEN EXTRACT(HOUR FROM created_at) BETWEEN 9 AND 17
         AND EXTRACT(DOW FROM created_at) NOT IN (0, 6) THEN 1 ELSE 0 END
                                                 AS created_at_is_business_hours
FROM data
```

**Override per-column:** If different datetime columns need different treatment:

```python
sq.AutoDatetime({
    "created_at": "hour",       # granularity per column
    "birth_date": "month",      # only need year + month for age
    "event_time": "minute",     # high-frequency events
})
```

**`CyclicEncode` — sin/cos for periodic features:**

Hour of day, day of week, month of year — these are CYCLICAL. Hour 23 is close to
hour 0, but ordinal encoding says they're far apart. Cyclic encoding fixes this:

```python
# After DateParts or AutoDatetime, encode cyclical features
sq.CyclicEncode(columns=["hour"], period=24)       # hour: 0-23, period=24
sq.CyclicEncode(columns=["dayofweek"], period=7)   # day: 0-6, period=7
sq.CyclicEncode(columns=["month"], period=12)       # month: 1-12, period=12

# Auto-detect period from column name (convenience)
sq.CyclicEncode(columns=["hour", "dayofweek", "month"], period="auto")
```

SQL:
```sql
SELECT
    SIN(2 * PI() * hour / 24) AS hour_sin,
    COS(2 * PI() * hour / 24) AS hour_cos
FROM data
```

Two output columns per input (sin + cos). Drops the original integer column by default.
Set `keep_original=True` to keep both representations.

### 6.6 Feature Engineering — Window & Aggregation

| Transformer | Default Columns | Dynamic? | SQL Pattern |
|---|---|---|---|
| `Lag(n=, by=, order=)` | explicit | No | `LAG(col, n) OVER (PARTITION BY ... ORDER BY ...)` |
| `Lead(n=)` | explicit | No | `LEAD(col, n) OVER (...)` |
| `RollingMean(window=)` | explicit | No | `AVG(col) OVER (... ROWS w PRECEDING)` |
| `RollingStd(window=)` | explicit | No | `STDDEV(col) OVER (...)` |
| `RollingMin(window=)` | explicit | No | `MIN(col) OVER (...)` |
| `RollingMax(window=)` | explicit | No | `MAX(col) OVER (...)` |
| `RollingSum(window=)` | explicit | No | `SUM(col) OVER (...)` |
| `EWM(alpha=, span=)` | explicit | No | Exponentially weighted mean via recursive CTE |
| `Rank(by=, order=)` | explicit | No | `RANK() OVER (PARTITION BY ... ORDER BY ...)` |
| `PercentRank()` | explicit | No | `PERCENT_RANK() OVER (ORDER BY col)` |
| `CumSum(order=)` | explicit | No | `SUM(col) OVER (ORDER BY ... ROWS UNBOUNDED PRECEDING)` |
| `CumMax(order=)` | explicit | No | `MAX(col) OVER (ORDER BY ... ROWS UNBOUNDED PRECEDING)` |
| `GroupFeatures(by=, aggs=)` | explicit | No | Window-based by default (preserves rows) |

Note: Window/aggregation transforms use `query()` not `expressions()`. They
automatically get CTE treatment from the compiler.

### 6.7 Feature Selection & Dropping

**Dropping features explicitly or by rule — sklearn has `sklearn.feature_selection`,
sqlearn makes it SQL-native and adds auto-detection.**

| Transformer | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `Drop(columns=)` | explicit | No | Yes (removes) | Omit from SELECT |
| `DropConstant()` | all | Yes | Yes (removes) | `COUNT(DISTINCT col) = 1` → drop |
| `DropCorrelated(threshold=)` | numeric | Yes | Yes (removes) | `CORR(a, b) > threshold` → drop one |
| `DropLowVariance(threshold=)` | numeric | Yes | Yes (removes) | `VAR_POP(col) < threshold` → drop |
| `DropHighNull(threshold=)` | all | Yes | Yes (removes) | `null_pct > threshold` → drop |
| `DropHighCardinality(threshold=)` | categorical | Yes | Yes (removes) | `COUNT(DISTINCT) > threshold` → drop |
| `SelectKBest(k=, method=)` | numeric | Yes | Yes (removes) | Keep top K by target correlation/MI |
| `SelectByName(pattern=)` | all | No | Yes (removes) | Keep columns matching glob/regex |
| `VarianceThreshold(threshold=)` | numeric | Yes | Yes (removes) | `VAR_POP(col) >= threshold` → keep |

**`Drop` — explicit column removal:**

```python
# Drop specific columns
sq.Drop(columns=["customer_id", "row_number", "index"])

# Drop by pattern (regex)
sq.Drop(pattern="^id_|_raw$")   # drop columns starting with id_ or ending with _raw

# Drop by type
sq.Drop(dtype="VARCHAR")         # drop all string columns
```

**`DropConstant` — auto-detect and remove useless columns:**

```python
sq.DropConstant()
# During fit: SELECT COUNT(DISTINCT col) FROM data → for each column
# Drops any column with 1 unique value (e.g., country="US" in a US-only dataset)
# Also drops all-NULL columns (0 distinct non-null values)
```

**`DropCorrelated` — remove redundant features:**

```python
sq.DropCorrelated(threshold=0.95)
# During fit: computes pairwise CORR(a, b) for all numeric pairs
# When |correlation| > threshold, drops the column with lower target correlation
# (or the second one alphabetically if no target)

# With target awareness — keeps the more predictive column
pipe.fit("data", y="price")
# bedrooms ↔ bathrooms: r=0.95 → keeps whichever correlates more with price
```

SQL for pairwise correlation:
```sql
SELECT
    CORR(bedrooms, bathrooms) AS bedrooms__bathrooms,
    CORR(bedrooms, sqft) AS bedrooms__sqft,
    ...
FROM data
```

**`SelectKBest` — keep top K features by target relationship:**

```python
sq.SelectKBest(k=10, method="correlation")    # top 10 by |CORR(col, target)|
sq.SelectKBest(k=10, method="mutual_info")    # top 10 by mutual information
sq.SelectKBest(k=10, method="anova")          # top 10 by ANOVA F-statistic
sq.SelectKBest(k="auto", method="correlation", threshold=0.05)  # keep all with |r| > 0.05
```

Requires `y=` in `pipeline.fit()`. The `discover()` method uses `y_column` to compute
feature-target relationships via SQL aggregation.

**Rule: selection transforms go AFTER encoding** (so they can see the expanded features).

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.OneHotEncoder(),
    sq.DropConstant(),            # remove any constant columns (e.g., one-hot with rare categories)
    sq.DropCorrelated(0.95),      # remove redundant features
    sq.SelectKBest(k=20),         # keep top 20 features by target correlation
    sq.StandardScaler(),
])
pipe.fit("data.parquet", y="target")
```

### 6.8 Auto Feature Engineering

**The auto-* family: type-aware transformers that inspect data and generate features
automatically. Each can be overridden per column.**

| Transformer | Detects | Generates | SQL Pattern |
|---|---|---|---|
| `AutoDatetime()` | Datetime columns | Year, month, day, hour, cyclical, etc. | See Section 6.5 |
| `AutoSplit()` | Delimited strings | Split into N columns | `SPLIT_PART` per delimiter |
| `AutoNumeric()` | Skewed/high-range numerics | Log, bins, or clipped version | `LN`, `NTILE`, `GREATEST/LEAST` |
| `AutoCategorical()` | Categorical by cardinality | Best encoder per column | Same as `AutoEncoder` |
| `AutoFeatures()` | All of the above | All of the above, combined | One step does everything |

**`AutoSplit` — auto-detect and split delimited strings:**

```python
# Auto-detect delimiter from data
sq.AutoSplit()
# During discover(): samples 1000 rows, checks for common delimiters (,;|/\t)
# If >50% of values contain the same delimiter → split
# Learns max_parts from data

# Explicit delimiter
sq.AutoSplit(columns=["tags"], by=",")

# Per-column config
sq.AutoSplit({
    "tags": ",",                # comma-separated
    "categories": "|",          # pipe-separated
    "full_name": " ",           # space-separated
})
```

Discovery SQL:
```sql
-- Detect most common delimiter per column
SELECT
    col,
    -- Count occurrences of each delimiter candidate
    AVG(LENGTH(tags) - LENGTH(REPLACE(tags, ',', ''))) AS comma_avg,
    AVG(LENGTH(tags) - LENGTH(REPLACE(tags, '|', ''))) AS pipe_avg,
    AVG(LENGTH(tags) - LENGTH(REPLACE(tags, ';', ''))) AS semi_avg,
    -- Max parts for the winning delimiter
    MAX(LENGTH(tags) - LENGTH(REPLACE(tags, ',', '')) + 1) AS max_parts
FROM data USING SAMPLE 1000
```

**`AutoNumeric` — auto-process numeric columns based on distribution:**

```python
sq.AutoNumeric()
# During discover(), for each numeric column:
#   skewness > 2         → apply Log(col + 1)
#   skewness > 1         → apply Sqrt(col)
#   range > 1000 × IQR   → apply Clip(lower=p01, upper=p99)
#   bimodal distribution  → apply KBinsDiscretizer(n=5)

# Override defaults
sq.AutoNumeric(skew_threshold=1.5, clip_percentile=0.02)

# Per-column override
sq.AutoNumeric({
    "income": "log",        # force log regardless of skewness
    "age": "bins:10",       # force 10 bins
    "score": "clip:0.01",   # clip at 1st and 99th percentile
    "price": "none",        # leave untouched
})
```

**`AutoFeatures` — the one-liner that does everything:**

```python
# Analyze data, generate all reasonable features automatically
pipe = sq.Pipeline([
    sq.Imputer(),             # handle nulls
    sq.AutoFeatures(),        # ← does ALL of the below automatically
    sq.StandardScaler(),
])

# AutoFeatures() internally creates and chains:
#   AutoDatetime()           — expand datetime columns
#   AutoSplit()              — split delimited strings
#   AutoNumeric()            — transform skewed/high-range numerics
#   AutoEncoder()            — encode categoricals (from Section 6.2)
#   CyclicEncode()           — encode cyclical datetime parts
#   DropConstant()           — remove constant columns generated above
```

Fine-grained control — enable/disable individual auto-transforms:

```python
sq.AutoFeatures(
    datetime=True,            # default: True
    split_strings=True,       # default: True
    numeric=True,             # default: True
    encode=True,              # default: True — auto-encode categoricals
    cyclic=True,              # default: True — sin/cos for hour/day/month
    drop_constant=True,       # default: True — clean up after expansion
    drop_original=False,      # default: False — keep original columns
)

# Or disable specific ones:
sq.AutoFeatures(split_strings=False, cyclic=False)
```

**`AutoFeatures.to_explicit()` — freeze auto-decisions into reproducible pipeline:**

Auto-detection is great for exploration but risky for production — if thresholds change
between sqlearn versions, your pipeline changes silently. `to_explicit()` converts
auto-decisions into a concrete pipeline with all parameters baked in:

```python
auto = sq.AutoFeatures()
auto.fit("data.parquet")

# See what AutoFeatures decided:
explicit = auto.to_explicit()
print(explicit)
# Pipeline([
#     sq.AutoDatetime({"created_at": "hour", "updated_at": "day"}),
#     sq.AutoSplit({"tags": ",", "categories": "|"}, max_parts={"tags": 4, "categories": 3}),
#     sq.AutoNumeric({"income": "log", "price": "clip:0.01"}),
#     sq.AutoEncoder({"city": "onehot", "zip": "hash:64", "state": "target"}),
#     sq.CyclicEncode(columns=["created_at_hour", "created_at_dayofweek"]),
#     sq.DropConstant(),
# ])

# Use explicit version in production — deterministic, version-safe:
prod_pipe = sq.Pipeline([sq.Imputer(), explicit, sq.StandardScaler()])
```

`to_explicit()` returns a Pipeline with all auto-detected parameters frozen as explicit
constructor arguments. The returned pipeline produces identical output but doesn't
depend on auto-detection logic. This is the bridge from exploration to production.

`sq.autopipeline()` also uses this internally: `autopipeline(as_code=True)` outputs
the explicit form, not `AutoFeatures()`.

**How `AutoFeatures` decides what to do (decision tree during `discover()`):**

```
For each column:
  ├── TIMESTAMP / DATE / TIME
  │     → AutoDatetime(granularity="auto")
  │     → CyclicEncode(columns=[hour, dayofweek, month]) if applicable
  │
  ├── VARCHAR / TEXT
  │     ├── Contains common delimiter (,;|) in >50% of values AND stddev of
  │     │   delimiter count < 1 (consistent splitting)
  │     │     → AutoSplit(by=detected_delimiter)
  │     ├── Valid JSON (parsed successfully via TRY_CAST or json_valid()) in >80% of rows
  │     │     → JsonExtract(fields=auto-detected top-level keys)
  │     │     (NOTE: "starts with {" is NOT sufficient — CSS, Python dicts, error messages
  │     │      also start with {. Must actually validate JSON parsing.)
  │     ├── Matches URL regex (scheme://domain pattern) in >80% of rows
  │     │     → URLParts()
  │     ├── Matches email regex (local@domain) in >80% of rows
  │     │     → EmailParts()
  │     ├── Cardinality < 20
  │     │     → OneHotEncoder
  │     ├── Cardinality 20-500
  │     │     → TargetEncoder (if y) or FrequencyEncoder
  │     └── Cardinality > 500
  │           → HashEncoder(n_bins=64)
  │
  ├── INTEGER / DOUBLE / FLOAT / DECIMAL
  │     ├── Constant (1 unique value)
  │     │     → DropConstant (remove)
  │     ├── Looks like ID (unique per row, monotonic, AND column name matches
  │     │   common ID patterns: *_id, id_*, rownum, index)
  │     │     → WARN + passthrough (NOT auto-drop — zip codes, sensor readings,
  │     │       and sequential customer IDs may be monotonic but are real features.
  │     │       User must explicitly Drop() if they agree.)
  │     ├── Skewness > 2
  │     │     → Log(col + 1)
  │     ├── Skewness 1-2
  │     │     → Sqrt(col)
  │     ├── Extreme outliers (max > 100 × p99)
  │     │     → Clip(lower=p01, upper=p99)
  │     └── Otherwise
  │           → passthrough
  │
  └── BOOLEAN
        → Cast to INTEGER (0/1) — passthrough
```

### 6.9 Outlier Handling

| Transformer | Default Columns | Dynamic? | Schema Change? | SQL Pattern |
|---|---|---|---|---|
| `OutlierHandler(method=, action=)` | numeric | Yes | Varies | IQR/zscore → clip/null/flag/drop |

```python
# Clip outliers to IQR bounds (default — safest)
sq.OutlierHandler()
# Default: method="iqr", action="clip", factor=1.5
# Learn: Q1, Q3, IQR per column
# Apply: GREATEST(LEAST(col, Q3 + 1.5*IQR), Q1 - 1.5*IQR)

# Z-score method
sq.OutlierHandler(method="zscore", threshold=3.0, action="clip")
# Learn: mean, std per column
# Apply: GREATEST(LEAST(col, mean + 3*std), mean - 3*std)

# Percentile method
sq.OutlierHandler(method="percentile", lower=0.01, upper=0.99, action="clip")
# Learn: p01, p99 per column
# Apply: GREATEST(LEAST(col, p99), p01)

# Actions — what to do with outliers
sq.OutlierHandler(action="clip")      # default: clip to bounds
sq.OutlierHandler(action="null")      # replace outliers with NULL (then impute later)
sq.OutlierHandler(action="flag")      # add col_outlier binary column (keep original)
sq.OutlierHandler(action="drop_rows") # filter out rows with outliers (uses query())

# Per-column config
sq.OutlierHandler({
    "income": {"method": "iqr", "factor": 3.0},      # lenient for income
    "age": {"method": "percentile", "lower": 0, "upper": 0.99},
    "score": "skip",                                   # leave untouched
})
```

SQL for IQR clipping:
```sql
-- Learned: price_q1=25000, price_q3=75000, price_iqr=50000
SELECT
    GREATEST(LEAST(price, 75000 + 1.5 * 50000), 25000 - 1.5 * 50000) AS price
    -- = GREATEST(LEAST(price, 150000), -50000) = clips to [-50000, 150000]
FROM data
```

### 6.10 Target Transforms

| Transformer | Applies To | Dynamic? | Invertible? | SQL Pattern |
|---|---|---|---|---|
| `TargetTransform(func=)` | target column only | Varies | Yes | Transform `y` before model, inverse after predict |

**For regression targets that are skewed or have large range:**

```python
# Log transform — most common for right-skewed targets (price, income, count)
sq.TargetTransform(func="log")
# fit: learns nothing (static)
# transform: LN(y + 1)
# inverse:   EXP(y_pred) - 1

# Sqrt — moderate skew correction
sq.TargetTransform(func="sqrt")

# Standard scaling — normalize target range
sq.TargetTransform(func="standard")
# fit: learns mean, std of y
# transform: (y - mean) / std
# inverse:   y_pred * std + mean

# MinMax — bound target to [0, 1]
sq.TargetTransform(func="minmax")
# fit: learns min, max of y
# transform: (y - min) / (max - min)
# inverse:   y_pred * (max - min) + min

# Quantile — uniform or normal distribution
sq.TargetTransform(func="quantile", output_distribution="normal")
# fit: learns quantile mapping
# transform: quantile → normal
```

**Usage in pipeline:**

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])

full = sq.ModelPipeline(
    preprocessor=pipe,
    model=XGBRegressor(),
    target_transform=sq.TargetTransform(func="log"),  # transform y before fit
)
full.fit("data", y="price")
predictions = full.predict("test")   # auto-inverts: EXP(raw_pred) - 1
```

`TargetTransform` only makes sense via `ModelPipeline` or `sq.Search` — it transforms
the target column for model training and auto-inverts predictions. It does NOT go inside
a regular `sq.Pipeline` (which transforms features, not targets).

### 6.11 Data Operations

Non-ML transforms for data cleanup. These are pipeline steps that modify columns
without statistical learning.

| Transformer | Purpose | Schema Change? | SQL Pattern |
|---|---|---|---|
| `Rename(mapping=)` | Rename columns | Yes | `col AS new_name` |
| `Cast(mapping=)` | Change column types | No | `CAST(col AS type)` |
| `Reorder(columns=)` | Change column order | No | Reorder in SELECT |
| `Filter(condition=)` | Filter rows | No | `WHERE condition` (uses `query()`) |
| `Sample(n=, frac=)` | Sample rows | No | `USING SAMPLE n` or `TABLESAMPLE(pct)` |
| `Deduplicate(subset=)` | Remove duplicate rows | No | `DISTINCT ON (subset)` or `ROW_NUMBER()` |

```python
# Rename columns
sq.Rename({"old_name": "new_name", "amt": "amount"})

# Cast types
sq.Cast({"zip_code": "VARCHAR", "flag": "INTEGER", "price": "DOUBLE"})
sq.Cast(columns=sq.numeric(), dtype="FLOAT")   # cast all numeric to float32

# Reorder — useful before model input
sq.Reorder(columns=["feature_1", "feature_2", ...])  # explicit order
sq.Reorder(sort="alphabetical")                        # sort A-Z

# Filter rows
sq.Filter("price > 0 AND price < 1000000")             # SQL string
sq.Filter(columns=["age"], condition="NOT NULL")         # shorthand: drop rows where age is NULL

# Sample
sq.Sample(n=10000)                  # exactly 10K rows
sq.Sample(frac=0.1, seed=42)       # 10% sample, reproducible

# Deduplicate
sq.Deduplicate()                                  # exact duplicate rows
sq.Deduplicate(subset=["user_id"], keep="first")  # keep first per user
```

**Typical cleanup pipeline:**

```python
cleanup = sq.Pipeline([
    sq.Filter("price > 0"),                        # remove invalid rows
    sq.Drop(columns=["row_id", "created_by"]),     # remove non-features
    sq.DropConstant(),                              # remove constant columns
    sq.Rename({"amt": "amount", "qty": "quantity"}),
    sq.Cast({"zip_code": "VARCHAR"}),              # zip is categorical, not numeric
])

features = sq.Pipeline([
    sq.Imputer(),
    sq.AutoFeatures(),
    sq.StandardScaler(),
])

full = cleanup + features   # → flat Pipeline with all steps
```

### 6.12 Composition

| Class | Replaces | Purpose |
|---|---|---|
| `Pipeline` | `sklearn.pipeline.Pipeline` | Sequential chain of transforms. Supports `+` and `+=`. |
| `Union` | `sklearn.pipeline.FeatureUnion` | Parallel feature combination |
| `Columns` | `sklearn.compose.ColumnTransformer` | Column routing (rarely needed with auto routing) |
| `Lookup` | N/A (sklearn has no equivalent) | Join features from another table mid-pipeline |

---
