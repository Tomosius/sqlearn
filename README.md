# sqlearn

Compile ML preprocessing pipelines to SQL.

Write Python pipelines (sklearn-style), get valid SQL. Every pipeline becomes one query.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")
X = pipe.transform("test.parquet")    # numpy array
sql = pipe.to_sql()                    # valid DuckDB SQL
```

## Status

Pre-implementation. See `docs/12-milestones.md` for the roadmap.

## License

MIT
