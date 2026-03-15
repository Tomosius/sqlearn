"""Tests for sqlearn.encoders.hash — HashEncoder."""

from __future__ import annotations

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.encoders.hash import HashEncoder

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test HashEncoder constructor and parameter storage."""

    def test_default_n_features(self) -> None:
        """Default n_features is 8."""
        enc = HashEncoder()
        assert enc.n_features == 8

    def test_custom_n_features(self) -> None:
        """Custom n_features is stored."""
        enc = HashEncoder(n_features=16)
        assert enc.n_features == 16

    def test_default_columns_is_none(self) -> None:
        """Default columns is None (falls back to _default_columns)."""
        enc = HashEncoder()
        assert enc.columns is None

    def test_custom_columns(self) -> None:
        """Explicit columns override."""
        enc = HashEncoder(columns=["city", "color"])
        assert enc.columns == ["city", "color"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        enc = HashEncoder(n_features=4, columns=["city"])
        params = enc.get_params()
        assert params == {"n_features": 4, "columns": ["city"]}

    def test_n_features_less_than_one_raises(self) -> None:
        """n_features < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be >= 1"):
            HashEncoder(n_features=0)

    def test_n_features_negative_raises(self) -> None:
        """Negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be >= 1"):
            HashEncoder(n_features=-3)


# ── Static classification tests ──────────────────────────────────


class TestClassification:
    """Test HashEncoder is classified as static (no discover needed)."""

    def test_classification_is_static(self) -> None:
        """Class attribute marks it as static."""
        assert HashEncoder._classification == "static"

    def test_classify_method_returns_static(self) -> None:
        """_classify() confirms static classification."""
        enc = HashEncoder()
        assert enc._classify() == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Class default routes to categorical columns."""
        assert HashEncoder._default_columns == "categorical"


# ── expressions() tests ──────────────────────────────────────────


class TestExpressions:
    """Test HashEncoder.expressions() generates correct CASE WHEN HASH ASTs."""

    def test_generates_n_features_columns(self) -> None:
        """Each input column produces n_features output expressions."""
        enc = HashEncoder(n_features=4)
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert len(result) == 4
        assert all(f"city_hash_{i}" in result for i in range(4))

    def test_each_expression_is_case(self) -> None:
        """Each generated expression is a CASE node."""
        enc = HashEncoder(n_features=3)
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        for i in range(3):
            assert isinstance(result[f"city_hash_{i}"], exp.Case)

    def test_multiple_columns(self) -> None:
        """Multiple columns produce all hash combinations."""
        enc = HashEncoder(n_features=2)
        exprs = {"city": exp.Column(this="city"), "color": exp.Column(this="color")}
        result = enc.expressions(["city", "color"], exprs)
        assert len(result) == 4
        assert "city_hash_0" in result
        assert "city_hash_1" in result
        assert "color_hash_0" in result
        assert "color_hash_1" in result

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column — preserves composition."""
        enc = HashEncoder(n_features=1)
        # Simulate a prior transform wrapping col in UPPER
        prior = exp.Upper(this=exp.Column(this="city"))
        exprs = {"city": prior}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city_hash_0"]
        assert isinstance(case_expr, exp.Case)
        # Walk into the AST: Case -> If -> EQ -> Mod -> Abs -> Anonymous -> Upper
        if_expr = case_expr.args["ifs"][0]
        eq_expr = if_expr.this
        assert isinstance(eq_expr, exp.EQ)
        mod_expr = eq_expr.this
        assert isinstance(mod_expr, exp.Mod)
        abs_expr = mod_expr.this
        assert isinstance(abs_expr, exp.Abs)
        hash_expr = abs_expr.this
        assert isinstance(hash_expr, exp.Anonymous)
        assert hash_expr.this == "HASH"
        # The argument to HASH should be Upper (the prior transform)
        assert isinstance(hash_expr.expressions[0], exp.Upper)

    def test_n_features_one(self) -> None:
        """n_features=1 produces a single column (always 1 for non-NULL)."""
        enc = HashEncoder(n_features=1)
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert len(result) == 1
        assert "city_hash_0" in result

    def test_ast_node_independence(self) -> None:
        """Each bucket expression uses an independent copy of the mod AST."""
        enc = HashEncoder(n_features=3)
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        # Extract the Mod nodes from each CASE expression
        mods = []
        for i in range(3):
            case_expr = result[f"city_hash_{i}"]
            if_expr = case_expr.args["ifs"][0]
            eq_expr = if_expr.this
            mods.append(eq_expr.this)
        # Each Mod should be an independent object (no shared nodes)
        assert mods[0] is not mods[1]
        assert mods[1] is not mods[2]


# ── output_schema() tests ────────────────────────────────────────


class TestOutputSchema:
    """Test HashEncoder.output_schema() in pre-fit and post-fit modes."""

    def test_drops_target_adds_hash_columns(self) -> None:
        """Target categorical columns replaced by hash bucket columns."""
        enc = HashEncoder(n_features=4)
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        result = enc.output_schema(schema)
        assert "city" not in result.columns
        assert "price" in result.columns
        for i in range(4):
            assert f"city_hash_{i}" in result.columns
            assert result.columns[f"city_hash_{i}"] == "INTEGER"

    def test_preserves_non_target_columns(self) -> None:
        """Non-target columns pass through unchanged."""
        enc = HashEncoder(n_features=2)
        schema = Schema({"price": "DOUBLE", "qty": "INTEGER", "city": "VARCHAR"})
        result = enc.output_schema(schema)
        assert result.columns["price"] == "DOUBLE"
        assert result.columns["qty"] == "INTEGER"

    def test_multiple_categorical_columns(self) -> None:
        """Multiple categorical columns each get n_features hash columns."""
        enc = HashEncoder(n_features=2)
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR", "color": "VARCHAR"})
        result = enc.output_schema(schema)
        assert "city" not in result.columns
        assert "color" not in result.columns
        assert "city_hash_0" in result.columns
        assert "city_hash_1" in result.columns
        assert "color_hash_0" in result.columns
        assert "color_hash_1" in result.columns

    def test_no_target_columns_returns_schema(self) -> None:
        """Schema with no categorical columns returns unchanged."""
        enc = HashEncoder()
        schema = Schema({"price": "DOUBLE", "qty": "INTEGER"})
        result = enc.output_schema(schema)
        assert result.columns == schema.columns

    def test_output_column_count(self) -> None:
        """Output schema has correct total column count."""
        enc = HashEncoder(n_features=8)
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR", "color": "VARCHAR"})
        result = enc.output_schema(schema)
        # price + 8 city hashes + 8 color hashes = 17
        assert len(result) == 17

    def test_output_schema_no_columns_spec(self) -> None:
        """output_schema() with _resolve_columns_spec() -> None returns schema.

        When both user columns and _default_columns resolve to None, output_schema
        cannot determine which columns to encode and returns the schema unchanged.
        """
        encoder = HashEncoder()
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        encoder._default_columns = None  # type: ignore[assignment]
        encoder.columns = None
        result = encoder.output_schema(schema)
        assert result is schema

    def test_output_schema_resolve_columns_raises(self) -> None:
        """output_schema() with invalid column spec returns schema unchanged."""
        encoder = HashEncoder(columns=["nonexistent"])
        schema = Schema({"price": "DOUBLE"})
        result = encoder.output_schema(schema)
        assert result is schema


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test HashEncoder integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with categorical test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), "
            "(3.0, 'London'), (4.0, 'Tokyo') t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape: 4 rows x 9 cols (price + 8 city hash cols)."""
        pipe = Pipeline([HashEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 9)

    def test_custom_n_features_shape(self, backend: DuckDBBackend) -> None:
        """n_features=4 produces price + 4 hash cols = 5 columns."""
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 5)

    def test_to_sql_contains_hash(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains HASH function."""
        pipe = Pipeline([HashEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "HASH" in sql.upper()

    def test_to_sql_contains_case_when(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains CASE WHEN."""
        pipe = Pipeline([HashEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "CASE" in sql
        assert "WHEN" in sql

    def test_get_feature_names_out(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out returns correct names."""
        pipe = Pipeline([HashEncoder(n_features=3)], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "price" in names
        for i in range(3):
            assert f"city_hash_{i}" in names
        assert len(names) == 4  # price + 3 hash cols

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([HashEncoder()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([HashEncoder()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_sql_custom_table(self, backend: DuckDBBackend) -> None:
        """to_sql(table=...) uses custom table name."""
        pipe = Pipeline([HashEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql(table="raw_data")
        assert "raw_data" in sql


# ── Determinism tests ────────────────────────────────────────────


class TestDeterminism:
    """Same input always produces same hash output."""

    def test_same_value_same_bucket(self) -> None:
        """Same category value hashes to same bucket across rows."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('London'), ('Paris'), ('London'), ('London') t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=8)], backend=backend)
        result = pipe.fit_transform("t")
        # Rows 0, 2, 3 are all 'London' — should be identical
        np.testing.assert_array_equal(result[0], result[2])
        np.testing.assert_array_equal(result[0], result[3])

    def test_repeated_transform_identical(self) -> None:
        """Calling transform() twice yields identical results."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES ('London'), ('Paris'), ('Tokyo') t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder()], backend=backend)
        pipe.fit("t")
        r1 = pipe.transform("t")
        r2 = pipe.transform("t")
        np.testing.assert_array_equal(r1, r2)


# ── Binary output tests ─────────────────────────────────────────


class TestBinaryOutput:
    """Verify output is strictly binary (0 or 1)."""

    @pytest.fixture
    def result(self) -> np.ndarray:
        """Fit/transform test data and return raw numpy array."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('London'), ('Paris'), ('Tokyo'), "
            "('Berlin'), ('Madrid') t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=8)], backend=backend)
        return np.asarray(pipe.fit_transform("t"))

    def test_only_zeros_and_ones(self, result: np.ndarray) -> None:
        """All values in hash columns are 0 or 1."""
        unique_values = set(np.unique(result))
        assert unique_values <= {0, 1}

    def test_exactly_one_bucket_per_row(self, result: np.ndarray) -> None:
        """Each row has exactly one 1 across its hash columns."""
        row_sums = result.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(len(row_sums)))


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_null_values_produce_zeros_or_one(self) -> None:
        """NULL values hash consistently (HASH(NULL) is deterministic in DuckDB)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (CAST(NULL AS VARCHAR)), ('London'), (CAST(NULL AS VARCHAR)) t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        result = np.asarray(pipe.fit_transform("t"))
        # NULL rows should be deterministic (same bucket each time)
        np.testing.assert_array_equal(result[0], result[2])

    def test_empty_string(self) -> None:
        """Empty string hashes to some bucket (not an error)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (''), ('hello'), ('') t(city)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        result = np.asarray(pipe.fit_transform("t"))
        # Should produce valid binary output
        unique_values = set(np.unique(result))
        assert unique_values <= {0, 1}
        # Empty strings should hash consistently
        np.testing.assert_array_equal(result[0], result[2])

    def test_single_column(self) -> None:
        """Single categorical column transforms correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), ('C') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 4)

    def test_n_features_one(self) -> None:
        """n_features=1 produces all-ones column (everything maps to bucket 0)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES ('London'), ('Paris'), ('Tokyo') t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=1)], backend=backend)
        result = np.asarray(pipe.fit_transform("t"))
        # With n_features=1, ABS(HASH(x)) % 1 = 0 always, so all rows = [1]
        np.testing.assert_array_equal(result, np.ones((3, 1)))

    def test_n_features_two(self) -> None:
        """n_features=2 produces valid binary output."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), ('C'), ('D') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=2)], backend=backend)
        result = np.asarray(pipe.fit_transform("t"))
        assert result.shape == (4, 2)
        # Each row sums to exactly 1
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(4))

    def test_large_n_features(self) -> None:
        """n_features=32 produces correct shape."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=32)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 32)

    def test_numeric_passthrough(self) -> None:
        """Numeric columns are not hashed (pass through unchanged)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 100, 'A'), (2.0, 200, 'B') t(price, qty, color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=3)], backend=backend)
        result = pipe.fit_transform("t")
        # price, qty pass through + 3 color hash columns = 5
        assert result.shape == (2, 5)
        # price and qty should be unchanged
        np.testing.assert_allclose(result[:, 0].astype(float), [1.0, 2.0])
        np.testing.assert_allclose(result[:, 1].astype(float), [100.0, 200.0])

    def test_multiple_categorical_columns(self) -> None:
        """Multiple categorical columns each get hashed independently."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('London', 'Red'), ('Paris', 'Blue') t(city, color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        result = pipe.fit_transform("t")
        # 4 city hashes + 4 color hashes = 8
        assert result.shape == (2, 8)


# ── Not-fitted guard tests ───────────────────────────────────────


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([HashEncoder()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([HashEncoder()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()


# ── Clone and pickle tests ──────────────────────────────────────


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([HashEncoder(n_features=4)], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_is_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned encoder does not affect original."""
        cloned = fitted_pipe.clone()
        original_enc = fitted_pipe.steps[0][1]
        cloned_enc = cloned.steps[0][1]
        assert isinstance(cloned_enc, HashEncoder)
        cloned_enc.n_features = 99
        assert isinstance(original_enc, HashEncoder)
        assert original_enc.n_features == 4

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual encoder preserves parameters."""
        import pickle

        enc = HashEncoder(n_features=16)
        enc._fitted = True
        enc.columns_ = ["city"]
        enc.input_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        enc.output_schema_ = enc.output_schema(enc.input_schema_)
        data = pickle.dumps(enc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.n_features == 16
        assert restored._fitted is True
        assert restored.columns_ == ["city"]


# ── Composition tests ────────────────────────────────────────────


class TestComposition:
    """HashEncoder composing with other transformers."""

    def test_imputer_then_hash_encoder(self) -> None:
        """Imputer + HashEncoder: NULLs filled before hashing."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), HashEncoder(n_features=4)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 4)

    def test_full_pipeline(self) -> None:
        """Imputer + StandardScaler + HashEncoder: full pipeline."""
        from sqlearn.imputers.imputer import Imputer
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'A'), (NULL, 'B'), (3.0, NULL), (4.0, 'A') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler(), HashEncoder(n_features=3)], backend=backend)
        result = pipe.fit_transform("t")
        # num (scaled) + 3 cat hash columns = 4
        assert result.shape == (4, 4)

    def test_hash_sql_composes_with_imputer(self) -> None:
        """SQL shows COALESCE nested inside HASH from composition."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL) t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), HashEncoder(n_features=2)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "HASH" in sql


# ── repr tests ───────────────────────────────────────────────────


class TestRepr:
    """Test HashEncoder __repr__ follows sklearn convention."""

    def test_default_repr(self) -> None:
        """Default params: HashEncoder()."""
        enc = HashEncoder()
        assert repr(enc) == "HashEncoder()"

    def test_custom_repr(self) -> None:
        """Custom params shown in repr."""
        enc = HashEncoder(n_features=16, columns=["city"])
        r = repr(enc)
        assert "n_features=16" in r
        assert "columns=['city']" in r
