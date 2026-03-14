"""Tests for sqlearn.core.schema."""

from __future__ import annotations

import pytest

from sqlearn.core.schema import (
    BOOLEAN_TYPES,
    CATEGORICAL_TYPES,
    NUMERIC_TYPES,
    TEMPORAL_TYPES,
    Schema,
    _classify_type,
    _normalize_type,
)


class TestSchemaConstruction:
    """Test Schema creation and basic properties."""

    def test_create_schema(self) -> None:
        """Schema stores column name to SQL type mapping."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert s.columns == {"price": "DOUBLE", "city": "VARCHAR"}

    def test_empty_schema(self) -> None:
        """Empty schema is valid."""
        s = Schema({})
        assert len(s) == 0

    def test_columns_are_copied(self) -> None:
        """External dict mutation does not affect Schema."""
        original = {"price": "DOUBLE"}
        s = Schema(original)
        original["city"] = "VARCHAR"
        assert "city" not in s.columns

    def test_frozen(self) -> None:
        """Schema attributes cannot be reassigned."""
        s = Schema({"price": "DOUBLE"})
        with pytest.raises(AttributeError):
            s.columns = {"city": "VARCHAR"}  # type: ignore[misc]


class TestSchemaDunder:
    """Test Schema dunder methods."""

    def test_len(self) -> None:
        """len() returns number of columns."""
        assert len(Schema({"a": "INT", "b": "INT"})) == 2

    def test_contains(self) -> None:
        """'in' checks column name existence."""
        s = Schema({"price": "DOUBLE"})
        assert "price" in s
        assert "city" not in s

    def test_getitem(self) -> None:
        """Bracket access returns SQL type string."""
        s = Schema({"price": "DOUBLE"})
        assert s["price"] == "DOUBLE"

    def test_getitem_missing_raises(self) -> None:
        """Bracket access for missing column raises KeyError."""
        s = Schema({"price": "DOUBLE"})
        with pytest.raises(KeyError):
            s["missing"]

    def test_iter(self) -> None:
        """Iterating yields column names in insertion order."""
        s = Schema({"b": "INT", "a": "VARCHAR", "c": "DOUBLE"})
        assert list(s) == ["b", "a", "c"]

    def test_repr(self) -> None:
        """repr shows column=type pairs."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        r = repr(s)
        assert "price=DOUBLE" in r
        assert "city=VARCHAR" in r

    def test_eq(self) -> None:
        """Two schemas with same columns are equal."""
        s1 = Schema({"a": "INT", "b": "VARCHAR"})
        s2 = Schema({"a": "INT", "b": "VARCHAR"})
        assert s1 == s2

    def test_neq_different_types(self) -> None:
        """Schemas with different types are not equal."""
        s1 = Schema({"a": "INT"})
        s2 = Schema({"a": "DOUBLE"})
        assert s1 != s2

    def test_neq_different_columns(self) -> None:
        """Schemas with different column names are not equal."""
        s1 = Schema({"a": "INT"})
        s2 = Schema({"b": "INT"})
        assert s1 != s2


class TestSchemaAdd:
    """Test Schema.add() method."""

    def test_add_columns(self) -> None:
        """add() appends new columns."""
        s = Schema({"a": "INT"})
        s2 = s.add({"b": "VARCHAR", "c": "DOUBLE"})
        assert s2.columns == {"a": "INT", "b": "VARCHAR", "c": "DOUBLE"}

    def test_add_returns_new_schema(self) -> None:
        """add() does not mutate the original."""
        s = Schema({"a": "INT"})
        s2 = s.add({"b": "VARCHAR"})
        assert "b" not in s
        assert "b" in s2

    def test_add_duplicate_raises(self) -> None:
        """add() raises ValueError if column already exists."""
        s = Schema({"a": "INT"})
        with pytest.raises(ValueError, match="already exist"):
            s.add({"a": "DOUBLE"})

    def test_add_empty(self) -> None:
        """add() with empty dict returns equal schema."""
        s = Schema({"a": "INT"})
        assert s.add({}) == s


class TestSchemaDrop:
    """Test Schema.drop() method."""

    def test_drop_columns(self) -> None:
        """drop() removes specified columns."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.drop(["b"])
        assert s2.columns == {"a": "INT", "c": "DOUBLE"}

    def test_drop_returns_new_schema(self) -> None:
        """drop() does not mutate the original."""
        s = Schema({"a": "INT", "b": "VARCHAR"})
        s2 = s.drop(["b"])
        assert "b" in s
        assert "b" not in s2

    def test_drop_missing_raises(self) -> None:
        """drop() raises KeyError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(KeyError, match="not found"):
            s.drop(["missing"])

    def test_drop_multiple(self) -> None:
        """drop() can remove multiple columns at once."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.drop(["a", "c"])
        assert s2.columns == {"b": "VARCHAR"}


class TestSchemaRename:
    """Test Schema.rename() method."""

    def test_rename_column(self) -> None:
        """rename() changes column names preserving order."""
        s = Schema({"old": "INT", "keep": "VARCHAR"})
        s2 = s.rename({"old": "new"})
        assert list(s2) == ["new", "keep"]
        assert s2["new"] == "INT"

    def test_rename_returns_new_schema(self) -> None:
        """rename() does not mutate the original."""
        s = Schema({"a": "INT"})
        s2 = s.rename({"a": "b"})
        assert "a" in s
        assert "a" not in s2

    def test_rename_missing_raises(self) -> None:
        """rename() raises KeyError if old name doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(KeyError, match="not found"):
            s.rename({"missing": "new"})

    def test_rename_preserves_order(self) -> None:
        """rename() preserves column position."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.rename({"b": "bb"})
        assert list(s2) == ["a", "bb", "c"]


class TestSchemaCast:
    """Test Schema.cast() method."""

    def test_cast_single(self) -> None:
        """cast() changes one column's type."""
        s = Schema({"a": "INT", "b": "VARCHAR"})
        s2 = s.cast("a", "DOUBLE")
        assert s2["a"] == "DOUBLE"
        assert s2["b"] == "VARCHAR"

    def test_cast_batch(self) -> None:
        """cast() with dict changes multiple types."""
        s = Schema({"a": "INT", "b": "VARCHAR"})
        s2 = s.cast({"a": "DOUBLE", "b": "TEXT"})
        assert s2["a"] == "DOUBLE"
        assert s2["b"] == "TEXT"

    def test_cast_missing_raises(self) -> None:
        """cast() raises KeyError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(KeyError, match="not found"):
            s.cast("missing", "DOUBLE")

    def test_cast_single_no_type_raises(self) -> None:
        """cast() with string col but no new_type raises TypeError."""
        s = Schema({"a": "INT"})
        with pytest.raises(TypeError, match="new_type is required"):
            s.cast("a")  # type: ignore[call-overload]

    def test_cast_preserves_order(self) -> None:
        """cast() preserves column position."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.cast("b", "TEXT")
        assert list(s2) == ["a", "b", "c"]


class TestSchemaSelect:
    """Test Schema.select() method."""

    def test_select_columns(self) -> None:
        """select() keeps only specified columns."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.select(["a", "c"])
        assert s2.columns == {"a": "INT", "c": "DOUBLE"}

    def test_select_preserves_original_order(self) -> None:
        """select() preserves original column order, not argument order."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        s2 = s.select(["c", "a"])
        assert list(s2) == ["a", "c"]

    def test_select_missing_raises(self) -> None:
        """select() raises KeyError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(KeyError, match="not found"):
            s.select(["missing"])


class TestNormalizeType:
    """Test _normalize_type() helper."""

    def test_simple_type(self) -> None:
        """Simple type passes through uppercased."""
        assert _normalize_type("double") == "DOUBLE"

    def test_parameterized_type(self) -> None:
        """Parameters are stripped."""
        assert _normalize_type("DECIMAL(18,3)") == "DECIMAL"

    def test_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _normalize_type("  varchar  ") == "VARCHAR"

    def test_parameterized_with_spaces(self) -> None:
        """Parameters with spaces are stripped."""
        assert _normalize_type("DECIMAL( 18, 3 )") == "DECIMAL"


class TestClassifyType:
    """Test _classify_type() helper."""

    @pytest.mark.parametrize(
        "sql_type",
        [
            "INTEGER",
            "INT",
            "BIGINT",
            "DOUBLE",
            "FLOAT",
            "DECIMAL",
            "DECIMAL(18,3)",
            "REAL",
            "SMALLINT",
            "HUGEINT",
        ],
    )
    def test_numeric_types(self, sql_type: str) -> None:
        """Numeric SQL types classify as 'numeric'."""
        assert _classify_type(sql_type) == "numeric"

    @pytest.mark.parametrize(
        "sql_type",
        [
            "VARCHAR",
            "TEXT",
            "STRING",
            "CHAR",
            "ENUM",
        ],
    )
    def test_categorical_types(self, sql_type: str) -> None:
        """Categorical SQL types classify as 'categorical'."""
        assert _classify_type(sql_type) == "categorical"

    @pytest.mark.parametrize(
        "sql_type",
        [
            "DATE",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMPTZ",
            "INTERVAL",
            "TIMESTAMP WITH TIME ZONE",
        ],
    )
    def test_temporal_types(self, sql_type: str) -> None:
        """Temporal SQL types classify as 'temporal'."""
        assert _classify_type(sql_type) == "temporal"

    @pytest.mark.parametrize("sql_type", ["BOOLEAN", "BOOL", "LOGICAL"])
    def test_boolean_types(self, sql_type: str) -> None:
        """Boolean SQL types classify as 'boolean'."""
        assert _classify_type(sql_type) == "boolean"

    @pytest.mark.parametrize("sql_type", ["BLOB", "JSON", "UUID", "STRUCT"])
    def test_other_types(self, sql_type: str) -> None:
        """Unknown SQL types classify as 'other'."""
        assert _classify_type(sql_type) == "other"

    def test_case_insensitive(self) -> None:
        """Classification is case-insensitive."""
        assert _classify_type("double") == "numeric"
        assert _classify_type("Varchar") == "categorical"


class TestSchemaColumnCategory:
    """Test Schema.column_category() method."""

    def test_numeric(self) -> None:
        """Numeric column returns 'numeric'."""
        s = Schema({"price": "DOUBLE"})
        assert s.column_category("price") == "numeric"

    def test_categorical(self) -> None:
        """VARCHAR column returns 'categorical'."""
        s = Schema({"city": "VARCHAR"})
        assert s.column_category("city") == "categorical"

    def test_temporal(self) -> None:
        """TIMESTAMP column returns 'temporal'."""
        s = Schema({"created": "TIMESTAMP"})
        assert s.column_category("created") == "temporal"

    def test_boolean(self) -> None:
        """BOOLEAN column returns 'boolean'."""
        s = Schema({"active": "BOOLEAN"})
        assert s.column_category("active") == "boolean"

    def test_other(self) -> None:
        """Unknown type returns 'other'."""
        s = Schema({"data": "BLOB"})
        assert s.column_category("data") == "other"

    def test_parameterized(self) -> None:
        """Parameterized type is normalized before classification."""
        s = Schema({"amount": "DECIMAL(18,3)"})
        assert s.column_category("amount") == "numeric"

    def test_missing_raises(self) -> None:
        """Missing column raises KeyError."""
        s = Schema({"a": "INT"})
        with pytest.raises(KeyError, match="not found"):
            s.column_category("missing")


class TestSchemaFilterMethods:
    """Test Schema.numeric(), .categorical(), .temporal(), .boolean()."""

    @pytest.fixture
    def mixed_schema(self) -> Schema:
        """Schema with all type categories."""
        return Schema(
            {
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
                "name": "TEXT",
                "created": "TIMESTAMP",
                "active": "BOOLEAN",
                "data": "BLOB",
            }
        )

    def test_numeric(self, mixed_schema: Schema) -> None:
        """numeric() returns numeric columns in order."""
        assert mixed_schema.numeric() == ["price", "qty"]

    def test_categorical(self, mixed_schema: Schema) -> None:
        """categorical() returns categorical columns in order."""
        assert mixed_schema.categorical() == ["city", "name"]

    def test_temporal(self, mixed_schema: Schema) -> None:
        """temporal() returns temporal columns in order."""
        assert mixed_schema.temporal() == ["created"]

    def test_boolean(self, mixed_schema: Schema) -> None:
        """boolean() returns boolean columns in order."""
        assert mixed_schema.boolean() == ["active"]

    def test_empty_category(self) -> None:
        """Filter method returns empty list when no columns match."""
        s = Schema({"price": "DOUBLE"})
        assert s.categorical() == []


class TestTypeCategorySets:
    """Test that type category frozensets are complete and non-overlapping."""

    def test_no_overlap(self) -> None:
        """No type string belongs to multiple categories."""
        all_sets = [NUMERIC_TYPES, CATEGORICAL_TYPES, TEMPORAL_TYPES, BOOLEAN_TYPES]
        for i, a in enumerate(all_sets):
            for b in all_sets[i + 1 :]:
                overlap = a & b
                assert not overlap, f"Overlap: {overlap}"

    def test_core_numeric_types_present(self) -> None:
        """Common numeric types are in NUMERIC_TYPES."""
        for t in ("INTEGER", "INT", "BIGINT", "FLOAT", "DOUBLE", "DECIMAL"):
            assert t in NUMERIC_TYPES

    def test_core_categorical_types_present(self) -> None:
        """Common categorical types are in CATEGORICAL_TYPES."""
        for t in ("VARCHAR", "TEXT", "STRING"):
            assert t in CATEGORICAL_TYPES

    def test_core_temporal_types_present(self) -> None:
        """Common temporal types are in TEMPORAL_TYPES."""
        for t in ("DATE", "TIME", "TIMESTAMP"):
            assert t in TEMPORAL_TYPES
