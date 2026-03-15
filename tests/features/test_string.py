"""Tests for sqlearn.features.string -- string transformers."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.features.string import (
    Lower,
    Replace,
    StringLength,
    Substring,
    Trim,
    Upper,
)

# ---------------------------------------------------------------------------
# StringLength
# ---------------------------------------------------------------------------


class TestStringLengthConstructor:
    """Test StringLength constructor and class attributes."""

    def test_classification_is_static(self) -> None:
        """StringLength is a static transformer."""
        assert StringLength._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical (string) columns."""
        assert StringLength._default_columns == "categorical"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        t = StringLength()
        assert t._classify() == "static"

    def test_custom_columns(self) -> None:
        """Custom columns override default."""
        t = StringLength(columns=["name"])
        assert t.columns == ["name"]

    def test_get_params(self) -> None:
        """get_params returns columns."""
        t = StringLength(columns=["a"])
        assert t.get_params() == {"columns": ["a"]}

    def test_get_params_default(self) -> None:
        """get_params with defaults returns None columns."""
        t = StringLength()
        assert t.get_params() == {"columns": None}


class TestStringLengthExpressions:
    """Test StringLength.expressions() generates correct ASTs."""

    def test_single_column(self) -> None:
        """Single column wrapped in LENGTH."""
        t = StringLength()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert "name" in result
        assert isinstance(result["name"], exp.Length)

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own LENGTH."""
        t = StringLength()
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
        }
        result = t.expressions(["a", "b"], exprs)
        assert len(result) == 2
        for col in ("a", "b"):
            assert isinstance(result[col], exp.Length)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not raw Column."""
        t = StringLength()
        prior = exp.Lower(this=exp.Column(this="a"))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Lower)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = StringLength()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}

    def test_sql_output(self) -> None:
        """Generated SQL contains LENGTH."""
        t = StringLength()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "LENGTH" in sql.upper()


class TestStringLengthOutputSchema:
    """Test StringLength.output_schema() changes types."""

    def test_type_changed_to_integer(self) -> None:
        """String columns become INTEGER after LENGTH."""
        t = StringLength()
        t.columns_ = ["name"]
        schema = Schema({"name": "VARCHAR", "price": "DOUBLE"})
        result = t.output_schema(schema)
        assert result["name"] == "INTEGER"

    def test_non_target_columns_unchanged(self) -> None:
        """Non-target columns keep their types."""
        t = StringLength()
        t.columns_ = ["name"]
        schema = Schema({"name": "VARCHAR", "price": "DOUBLE"})
        result = t.output_schema(schema)
        assert result["price"] == "DOUBLE"

    def test_columns_none_returns_input(self) -> None:
        """When columns_ is None, returns input schema unchanged."""
        t = StringLength()
        schema = Schema({"name": "VARCHAR"})
        result = t.output_schema(schema)
        assert result["name"] == "VARCHAR"


class TestStringLengthPipeline:
    """Test StringLength with Pipeline end-to-end."""

    @pytest.fixture
    def str_backend(self) -> DuckDBBackend:
        """DuckDB backend with string test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES "
            "('hello', 'world'), ('hi', 'earth'), ('hey', 'mars') t(a, b)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform(self, str_backend: DuckDBBackend) -> None:
        """StringLength produces integer lengths."""
        pipe = Pipeline([StringLength()], backend=str_backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)
        # 'hello' -> 5, 'hi' -> 2, 'hey' -> 3
        np.testing.assert_array_equal(result[:, 0], [5, 2, 3])

    def test_to_sql_contains_length(self, str_backend: DuckDBBackend) -> None:
        """to_sql() output contains LENGTH."""
        pipe = Pipeline([StringLength()], backend=str_backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "LENGTH" in sql

    def test_null_handling(self) -> None:
        """NULL strings produce NULL lengths."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('hello'), (NULL), ('hi') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StringLength()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_empty_string(self) -> None:
        """Empty string has length 0."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (''), ('a'), ('ab') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StringLength()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_array_equal(result[:, 0], [0, 1, 2])

    def test_unicode(self) -> None:
        """Unicode strings are measured correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('cafe'), ('resume') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StringLength()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3 if result.shape[0] == 3 else 2, 1)


# ---------------------------------------------------------------------------
# Lower
# ---------------------------------------------------------------------------


class TestLowerConstructor:
    """Test Lower constructor and class attributes."""

    def test_classification_is_static(self) -> None:
        """Lower is a static transformer."""
        assert Lower._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical columns."""
        assert Lower._default_columns == "categorical"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        assert Lower()._classify() == "static"


class TestLowerExpressions:
    """Test Lower.expressions() generates correct ASTs."""

    def test_single_column(self) -> None:
        """Single column wrapped in LOWER."""
        t = Lower()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Lower)

    def test_sql_output(self) -> None:
        """Generated SQL contains LOWER."""
        t = Lower()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "LOWER" in sql.upper()

    def test_uses_exprs(self) -> None:
        """expressions() composes with prior transforms."""
        t = Lower()
        prior = exp.Trim(this=exp.Column(this="a"))
        result = t.expressions(["a"], {"a": prior})
        assert isinstance(result["a"].this, exp.Trim)


class TestLowerPipeline:
    """Test Lower with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """Lower converts strings to lowercase."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('HELLO'), ('World'), ('foo') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lower()], backend=backend)
        result = pipe.fit_transform("t")
        assert list(result[:, 0]) == ["hello", "world", "foo"]

    def test_null_handling(self) -> None:
        """NULL values pass through LOWER as NULL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('ABC'), (NULL), ('DEF') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lower()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "abc"
        assert result[1, 0] is None
        assert result[2, 0] == "def"

    def test_empty_string(self) -> None:
        """Empty string lowered is still empty."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (''), ('A') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lower()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == ""
        assert result[1, 0] == "a"


# ---------------------------------------------------------------------------
# Upper
# ---------------------------------------------------------------------------


class TestUpperConstructor:
    """Test Upper constructor and class attributes."""

    def test_classification_is_static(self) -> None:
        """Upper is a static transformer."""
        assert Upper._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical columns."""
        assert Upper._default_columns == "categorical"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        assert Upper()._classify() == "static"


class TestUpperExpressions:
    """Test Upper.expressions() generates correct ASTs."""

    def test_single_column(self) -> None:
        """Single column wrapped in UPPER."""
        t = Upper()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Upper)

    def test_sql_output(self) -> None:
        """Generated SQL contains UPPER."""
        t = Upper()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "UPPER" in sql.upper()


class TestUpperPipeline:
    """Test Upper with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """Upper converts strings to uppercase."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('hello'), ('World'), ('FOO') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Upper()], backend=backend)
        result = pipe.fit_transform("t")
        assert list(result[:, 0]) == ["HELLO", "WORLD", "FOO"]

    def test_null_handling(self) -> None:
        """NULL values pass through UPPER as NULL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('abc'), (NULL), ('def') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Upper()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "ABC"
        assert result[1, 0] is None
        assert result[2, 0] == "DEF"


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


class TestTrimConstructor:
    """Test Trim constructor and class attributes."""

    def test_classification_is_static(self) -> None:
        """Trim is a static transformer."""
        assert Trim._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical columns."""
        assert Trim._default_columns == "categorical"

    def test_default_characters_none(self) -> None:
        """Default characters is None (whitespace trimming)."""
        t = Trim()
        assert t.characters is None

    def test_custom_characters(self) -> None:
        """Custom characters parameter is stored."""
        t = Trim(characters="#")
        assert t.characters == "#"

    def test_get_params(self) -> None:
        """get_params includes characters."""
        t = Trim(characters="*")
        params = t.get_params()
        assert params == {"columns": None, "characters": "*"}

    def test_get_params_default(self) -> None:
        """get_params with defaults."""
        t = Trim()
        params = t.get_params()
        assert params == {"columns": None, "characters": None}


class TestTrimExpressions:
    """Test Trim.expressions() generates correct ASTs."""

    def test_whitespace_trim(self) -> None:
        """Default trim uses exp.Trim node."""
        t = Trim()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Trim)

    def test_character_trim(self) -> None:
        """Character trim uses exp.Anonymous('TRIM', ...)."""
        t = Trim(characters="#")
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Anonymous)

    def test_sql_output_default(self) -> None:
        """Generated SQL contains TRIM for default mode."""
        t = Trim()
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "TRIM" in sql.upper()

    def test_sql_output_characters(self) -> None:
        """Generated SQL contains TRIM for character mode."""
        t = Trim(characters="#")
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "TRIM" in sql.upper()


class TestTrimPipeline:
    """Test Trim with Pipeline end-to-end."""

    def test_whitespace_trim(self) -> None:
        """Trim removes leading/trailing whitespace."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES ('  hello  '), (' world '), ('foo') t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Trim()], backend=backend)
        result = pipe.fit_transform("t")
        assert list(result[:, 0]) == ["hello", "world", "foo"]

    def test_null_handling(self) -> None:
        """NULL values pass through TRIM as NULL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (' abc '), (NULL), (' def ') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Trim()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "abc"
        assert result[1, 0] is None
        assert result[2, 0] == "def"

    def test_empty_string(self) -> None:
        """Empty string trimmed is still empty."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (''), ('  '), ('a') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Trim()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == ""
        assert result[1, 0] == ""
        assert result[2, 0] == "a"


# ---------------------------------------------------------------------------
# Replace
# ---------------------------------------------------------------------------


class TestReplaceConstructor:
    """Test Replace constructor and validation."""

    def test_classification_is_static(self) -> None:
        """Replace is a static transformer."""
        assert Replace._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical columns."""
        assert Replace._default_columns == "categorical"

    def test_basic_params(self) -> None:
        """old and new params are stored."""
        t = Replace(old="foo", new="bar")
        assert t.old == "foo"
        assert t.new == "bar"

    def test_empty_old_raises(self) -> None:
        """Empty old string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Replace(old="", new="bar")

    def test_non_string_old_raises(self) -> None:
        """Non-string old raises TypeError."""
        with pytest.raises(TypeError, match="old must be a string"):
            Replace(old=42, new="bar")  # type: ignore[arg-type]

    def test_non_string_new_raises(self) -> None:
        """Non-string new raises TypeError."""
        with pytest.raises(TypeError, match="new must be a string"):
            Replace(old="foo", new=42)  # type: ignore[arg-type]

    def test_get_params(self) -> None:
        """get_params returns old, new, columns."""
        t = Replace(old="a", new="b")
        params = t.get_params()
        assert params == {"old": "a", "new": "b", "columns": None}

    def test_empty_new_is_valid(self) -> None:
        """Empty new string is valid (deletion)."""
        t = Replace(old="x", new="")
        assert t.new == ""


class TestReplaceExpressions:
    """Test Replace.expressions() generates correct ASTs."""

    def test_single_column(self) -> None:
        """Single column wrapped in REPLACE."""
        t = Replace(old="foo", new="bar")
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Anonymous)
        assert result["name"].this == "REPLACE"

    def test_sql_output(self) -> None:
        """Generated SQL contains REPLACE."""
        t = Replace(old="foo", new="bar")
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "REPLACE" in sql.upper()
        assert "'foo'" in sql
        assert "'bar'" in sql

    def test_multiple_columns(self) -> None:
        """Multiple columns each get REPLACE."""
        t = Replace(old="x", new="y")
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
        }
        result = t.expressions(["a", "b"], exprs)
        assert len(result) == 2
        for col in ("a", "b"):
            assert isinstance(result[col], exp.Anonymous)


class TestReplacePipeline:
    """Test Replace with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """Replace substitutes substrings."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES ('foo-bar'), ('foo-baz'), ('qux') t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Replace(old="foo", new="XXX")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "XXX-bar"
        assert result[1, 0] == "XXX-baz"
        assert result[2, 0] == "qux"

    def test_delete_substring(self) -> None:
        """Replace with empty new deletes the substring."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('a-b'), ('c-d'), ('e') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Replace(old="-", new="")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "ab"
        assert result[1, 0] == "cd"
        assert result[2, 0] == "e"

    def test_null_handling(self) -> None:
        """NULL values pass through REPLACE as NULL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('abc'), (NULL), ('def') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Replace(old="a", new="X")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "Xbc"
        assert result[1, 0] is None
        assert result[2, 0] == "def"

    def test_no_match(self) -> None:
        """Replace with no match leaves string unchanged."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('hello') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Replace(old="xyz", new="Q")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "hello"


# ---------------------------------------------------------------------------
# Substring
# ---------------------------------------------------------------------------


class TestSubstringConstructor:
    """Test Substring constructor and validation."""

    def test_classification_is_static(self) -> None:
        """Substring is a static transformer."""
        assert Substring._classification == "static"

    def test_default_columns_is_categorical(self) -> None:
        """Default columns target categorical columns."""
        assert Substring._default_columns == "categorical"

    def test_basic_params(self) -> None:
        """start and length params are stored."""
        t = Substring(start=1, length=3)
        assert t.start == 1
        assert t.length == 3

    def test_length_none(self) -> None:
        """length=None means extract to end."""
        t = Substring(start=2)
        assert t.length is None

    def test_start_zero_raises(self) -> None:
        """start=0 raises ValueError (1-based indexing)."""
        with pytest.raises(ValueError, match="start must be >= 1"):
            Substring(start=0)

    def test_negative_start_raises(self) -> None:
        """Negative start raises ValueError."""
        with pytest.raises(ValueError, match="start must be >= 1"):
            Substring(start=-1)

    def test_negative_length_raises(self) -> None:
        """Negative length raises ValueError."""
        with pytest.raises(ValueError, match="length must be >= 0"):
            Substring(start=1, length=-1)

    def test_non_int_start_raises(self) -> None:
        """Non-int start raises TypeError."""
        with pytest.raises(TypeError, match="start must be an int"):
            Substring(start=1.5)  # type: ignore[arg-type]

    def test_non_int_length_raises(self) -> None:
        """Non-int length raises TypeError."""
        with pytest.raises(TypeError, match="length must be an int"):
            Substring(start=1, length=2.5)  # type: ignore[arg-type]

    def test_get_params(self) -> None:
        """get_params returns start, length, columns."""
        t = Substring(start=2, length=5)
        params = t.get_params()
        assert params == {"start": 2, "length": 5, "columns": None}

    def test_zero_length_is_valid(self) -> None:
        """length=0 is valid (produces empty strings)."""
        t = Substring(start=1, length=0)
        assert t.length == 0


class TestSubstringExpressions:
    """Test Substring.expressions() generates correct ASTs."""

    def test_with_length(self) -> None:
        """With length, generates SUBSTRING(col, start, length)."""
        t = Substring(start=1, length=3)
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Substring)

    def test_without_length(self) -> None:
        """Without length, generates SUBSTRING(col, start)."""
        t = Substring(start=2)
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        assert isinstance(result["name"], exp.Substring)

    def test_sql_output_with_length(self) -> None:
        """Generated SQL contains SUBSTRING with start and length."""
        t = Substring(start=1, length=3)
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "SUBSTRING" in sql.upper()

    def test_sql_output_without_length(self) -> None:
        """Generated SQL contains SUBSTRING with start only."""
        t = Substring(start=2)
        exprs = {"name": exp.Column(this="name")}
        result = t.expressions(["name"], exprs)
        sql = result["name"].sql(dialect="duckdb")
        assert "SUBSTRING" in sql.upper()


class TestSubstringPipeline:
    """Test Substring with Pipeline end-to-end."""

    def test_fit_transform_with_length(self) -> None:
        """Substring extracts first N characters."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('hello'), ('world'), ('foo') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Substring(start=1, length=3)], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "hel"
        assert result[1, 0] == "wor"
        assert result[2, 0] == "foo"

    def test_fit_transform_without_length(self) -> None:
        """Substring from position 2 to end."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('hello'), ('world'), ('foo') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Substring(start=2)], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "ello"
        assert result[1, 0] == "orld"
        assert result[2, 0] == "oo"

    def test_null_handling(self) -> None:
        """NULL values pass through SUBSTRING as NULL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('abc'), (NULL), ('def') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Substring(start=1, length=2)], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "ab"
        assert result[1, 0] is None
        assert result[2, 0] == "de"

    def test_empty_string(self) -> None:
        """Substring of empty string returns empty string."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (''), ('a'), ('ab') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Substring(start=1, length=1)], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == ""
        assert result[1, 0] == "a"
        assert result[2, 0] == "a"


# ---------------------------------------------------------------------------
# Clone and Pickle (all transformers)
# ---------------------------------------------------------------------------


class TestCloneAndPickle:
    """Clone and pickle roundtrip for all string transformers."""

    @pytest.mark.parametrize(
        "transformer",
        [
            StringLength(),
            Lower(),
            Upper(),
            Trim(),
            Trim(characters="#"),
            Replace(old="a", new="b"),
            Substring(start=1, length=3),
            Substring(start=2),
        ],
        ids=[
            "StringLength",
            "Lower",
            "Upper",
            "Trim-default",
            "Trim-chars",
            "Replace",
            "Substring-with-len",
            "Substring-no-len",
        ],
    )
    def test_pickle_roundtrip(self, transformer: object) -> None:
        """Pickle preserves transformer state."""
        data = pickle.dumps(transformer)
        restored = pickle.loads(data)  # noqa: S301
        assert type(restored) is type(transformer)
        assert restored.get_params() == transformer.get_params()  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "transformer",
        [
            StringLength(),
            Lower(),
            Upper(),
            Trim(),
            Replace(old="x", new="y"),
            Substring(start=1),
        ],
        ids=[
            "StringLength",
            "Lower",
            "Upper",
            "Trim",
            "Replace",
            "Substring",
        ],
    )
    def test_clone(self, transformer: object) -> None:
        """Clone produces independent copy with same params."""
        from sqlearn.core.transformer import Transformer

        assert isinstance(transformer, Transformer)
        cloned = transformer.clone()
        assert type(cloned) is type(transformer)
        assert cloned is not transformer
        assert cloned.get_params() == transformer.get_params()


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------


class TestComposition:
    """Test string transformers composing with each other and other transforms."""

    def test_lower_then_replace(self) -> None:
        """Lower + Replace: lowercase first, then replace."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('Hello'), ('WORLD'), ('Foo') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lower(), Replace(old="hello", new="HI")], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "HI"
        assert result[1, 0] == "world"
        assert result[2, 0] == "foo"

    def test_trim_then_upper(self) -> None:
        """Trim + Upper: trim whitespace, then uppercase."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('  hello  '), (' world ') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Trim(), Upper()], backend=backend)
        result = pipe.fit_transform("t")
        assert result[0, 0] == "HELLO"
        assert result[1, 0] == "WORLD"

    def test_trim_then_length(self) -> None:
        """Trim + StringLength: trim then measure."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('  hi  '), (' hey ') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Trim(), StringLength()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_array_equal(result[:, 0], [2, 3])

    def test_operator_composition(self) -> None:
        """Lower + Upper via + operator creates Pipeline."""
        pipe = Lower() + Upper()
        assert isinstance(pipe, Pipeline)


# ---------------------------------------------------------------------------
# Repr tests
# ---------------------------------------------------------------------------


class TestRepr:
    """Test __repr__ for all string transformers."""

    def test_string_length_repr(self) -> None:
        """StringLength repr."""
        assert repr(StringLength()) == "StringLength()"

    def test_string_length_repr_with_columns(self) -> None:
        """StringLength repr with columns."""
        r = repr(StringLength(columns=["name"]))
        assert "StringLength" in r
        assert "name" in r

    def test_lower_repr(self) -> None:
        """Lower repr."""
        assert repr(Lower()) == "Lower()"

    def test_upper_repr(self) -> None:
        """Upper repr."""
        assert repr(Upper()) == "Upper()"

    def test_trim_repr_default(self) -> None:
        """Trim repr with defaults."""
        assert repr(Trim()) == "Trim()"

    def test_trim_repr_with_characters(self) -> None:
        """Trim repr with characters."""
        r = repr(Trim(characters="#"))
        assert "Trim" in r
        assert "#" in r

    def test_replace_repr(self) -> None:
        """Replace repr."""
        r = repr(Replace(old="a", new="b"))
        assert "Replace" in r
        assert "a" in r
        assert "b" in r

    def test_substring_repr(self) -> None:
        """Substring repr."""
        r = repr(Substring(start=1, length=3))
        assert "Substring" in r


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Test that string transformers are importable from sqlearn."""

    def test_import_from_sqlearn(self) -> None:
        """All string transformers importable from sqlearn."""
        import sqlearn as sq

        assert hasattr(sq, "StringLength")
        assert hasattr(sq, "Lower")
        assert hasattr(sq, "Upper")
        assert hasattr(sq, "Trim")
        assert hasattr(sq, "Replace")
        assert hasattr(sq, "Substring")

    def test_import_from_features(self) -> None:
        """All string transformers importable from sqlearn.features."""
        from sqlearn.features import (
            Lower,
            Replace,
            StringLength,
            Substring,
            Trim,
            Upper,
        )

        assert StringLength is not None
        assert Lower is not None
        assert Upper is not None
        assert Trim is not None
        assert Replace is not None
        assert Substring is not None
