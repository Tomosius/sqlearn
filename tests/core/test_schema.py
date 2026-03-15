"""Tests for sqlearn.core.schema."""

from __future__ import annotations

import pytest

from sqlearn.core.errors import MissingColumnError, SchemaError
from sqlearn.core.schema import (
    BOOLEAN_TYPES,
    CATEGORICAL_TYPES,
    NUMERIC_TYPES,
    TEMPORAL_TYPES,
    ColumnSelector,
    Schema,
    _classify_type,
    _normalize_type,
    all_columns,
    boolean,
    categorical,
    columns,
    dtype,
    matching,
    numeric,
    resolve_columns,
    temporal,
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

    def test_reserved_prefix_raises(self) -> None:
        """Columns with __sq_*__ prefix are rejected."""
        with pytest.raises(SchemaError, match="reserved sqlearn prefix"):
            Schema({"__sq_fold__": "INTEGER", "price": "DOUBLE"})

    def test_reserved_prefix_multiple(self) -> None:
        """Multiple reserved columns are all reported."""
        with pytest.raises(SchemaError, match="reserved sqlearn prefix"):
            Schema({"__sq_fold__": "INT", "__sq_idx__": "INT"})

    def test_reserved_prefix_partial_ok(self) -> None:
        """Columns that don't match both prefix and suffix are fine."""
        s = Schema({"__sq_not_reserved": "INT", "sq_fold__": "INT"})
        assert len(s) == 2


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

    def test_hash_equal_schemas(self) -> None:
        """Equal schemas have the same hash."""
        s1 = Schema({"a": "INT", "b": "VARCHAR"})
        s2 = Schema({"a": "INT", "b": "VARCHAR"})
        assert hash(s1) == hash(s2)

    def test_hash_different_schemas(self) -> None:
        """Different schemas have different hashes (high probability)."""
        s1 = Schema({"a": "INT"})
        s2 = Schema({"a": "DOUBLE"})
        assert hash(s1) != hash(s2)

    def test_hash_usable_as_dict_key(self) -> None:
        """Schema can be used as a dictionary key."""
        s = Schema({"a": "INT"})
        d = {s: "cached_plan"}
        assert d[Schema({"a": "INT"})] == "cached_plan"

    def test_hash_usable_in_set(self) -> None:
        """Schema can be added to a set."""
        s1 = Schema({"a": "INT"})
        s2 = Schema({"a": "INT"})
        assert len({s1, s2}) == 1


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
        """add() raises SchemaError if column already exists."""
        s = Schema({"a": "INT"})
        with pytest.raises(SchemaError, match="already exist"):
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
        """drop() raises SchemaError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(SchemaError, match="not found"):
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
        """rename() raises SchemaError if old name doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(SchemaError, match="not found"):
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
        """cast() raises SchemaError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(SchemaError, match="not found"):
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
        """select() raises SchemaError if column doesn't exist."""
        s = Schema({"a": "INT"})
        with pytest.raises(SchemaError, match="not found"):
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
        """Missing column raises MissingColumnError."""
        s = Schema({"a": "INT"})
        with pytest.raises(MissingColumnError, match="not found"):
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


class TestTypeSelector:
    """Test TypeSelector via factory functions."""

    def test_numeric_selector(self) -> None:
        """numeric() selects numeric columns."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR", "qty": "INT"})
        result = numeric().resolve(s)
        assert result == ["price", "qty"]

    def test_categorical_selector(self) -> None:
        """categorical() selects categorical columns."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR", "name": "TEXT"})
        result = categorical().resolve(s)
        assert result == ["city", "name"]

    def test_temporal_selector(self) -> None:
        """temporal() selects temporal columns."""
        s = Schema({"ts": "TIMESTAMP", "d": "DATE", "x": "INT"})
        result = temporal().resolve(s)
        assert result == ["ts", "d"]

    def test_boolean_selector(self) -> None:
        """boolean() selects boolean columns."""
        s = Schema({"active": "BOOLEAN", "x": "INT"})
        result = boolean().resolve(s)
        assert result == ["active"]

    def test_no_matches(self) -> None:
        """Selector returns empty list when no columns match."""
        s = Schema({"price": "DOUBLE"})
        assert categorical().resolve(s) == []

    def test_preserves_order(self) -> None:
        """Selector preserves column insertion order."""
        s = Schema({"b": "DOUBLE", "a": "INT", "c": "FLOAT"})
        assert numeric().resolve(s) == ["b", "a", "c"]

    def test_parameterized_types(self) -> None:
        """Selector handles parameterized types like DECIMAL(18,3)."""
        s = Schema({"amount": "DECIMAL(18,3)", "name": "VARCHAR"})
        assert numeric().resolve(s) == ["amount"]


class TestPatternSelector:
    """Test PatternSelector via matching() factory."""

    def test_glob_star(self) -> None:
        """matching() with * wildcard."""
        s = Schema({"price_usd": "DOUBLE", "price_eur": "DOUBLE", "city": "VARCHAR"})
        result = matching("price_*").resolve(s)
        assert result == ["price_usd", "price_eur"]

    def test_glob_question(self) -> None:
        """matching() with ? wildcard."""
        s = Schema({"col1": "INT", "col2": "INT", "col10": "INT"})
        result = matching("col?").resolve(s)
        assert result == ["col1", "col2"]

    def test_no_matches(self) -> None:
        """matching() returns empty list when nothing matches."""
        s = Schema({"price": "DOUBLE"})
        assert matching("city_*").resolve(s) == []

    def test_exact_match(self) -> None:
        """matching() with exact name (no wildcards)."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        assert matching("price").resolve(s) == ["price"]


class TestDTypeSelector:
    """Test DTypeSelector via dtype() factory."""

    def test_exact_type(self) -> None:
        """dtype() selects columns with matching type."""
        s = Schema({"a": "DOUBLE", "b": "INT", "c": "DOUBLE"})
        result = dtype("DOUBLE").resolve(s)
        assert result == ["a", "c"]

    def test_normalized_matching(self) -> None:
        """dtype() normalizes both sides for comparison."""
        s = Schema({"a": "DECIMAL(18,3)", "b": "DECIMAL(10,2)", "c": "INT"})
        result = dtype("DECIMAL").resolve(s)
        assert result == ["a", "b"]

    def test_parameterized_selector_normalizes(self) -> None:
        """dtype('DECIMAL(18,3)') normalizes to DECIMAL, matches all variants."""
        s = Schema({"a": "DECIMAL(18,3)", "b": "DECIMAL(10,2)"})
        result = dtype("DECIMAL(18,3)").resolve(s)
        assert result == ["a", "b"]

    def test_case_insensitive(self) -> None:
        """dtype() is case-insensitive."""
        s = Schema({"a": "double"})
        result = dtype("DOUBLE").resolve(s)
        assert result == ["a"]

    def test_no_matches(self) -> None:
        """dtype() returns empty list when nothing matches."""
        s = Schema({"a": "INT"})
        assert dtype("VARCHAR").resolve(s) == []


class TestSelectorRepr:
    """Test selector repr for debugging."""

    def test_numeric_repr(self) -> None:
        """numeric() has readable repr."""
        assert repr(numeric()) == "numeric()"

    def test_categorical_repr(self) -> None:
        """categorical() has readable repr."""
        assert repr(categorical()) == "categorical()"

    def test_matching_repr(self) -> None:
        """matching() shows the pattern."""
        assert repr(matching("price_*")) == "matching('price_*')"

    def test_dtype_repr(self) -> None:
        """dtype() shows the type."""
        assert repr(dtype("DOUBLE")) == "dtype('DOUBLE')"


class TestSelectorIsColumnSelector:
    """Test that all selectors are ColumnSelector subclasses."""

    def test_type_selector(self) -> None:
        """TypeSelector is a ColumnSelector."""
        assert isinstance(numeric(), ColumnSelector)

    def test_pattern_selector(self) -> None:
        """PatternSelector is a ColumnSelector."""
        assert isinstance(matching("*"), ColumnSelector)

    def test_dtype_selector(self) -> None:
        """DTypeSelector is a ColumnSelector."""
        assert isinstance(dtype("INT"), ColumnSelector)


class TestResolveColumns:
    """Test resolve_columns() unified resolution."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Mixed-type schema for testing."""
        return Schema(
            {
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
                "created": "TIMESTAMP",
                "active": "BOOLEAN",
            }
        )

    # --- String literals ---

    def test_numeric_string(self, schema: Schema) -> None:
        """'numeric' string resolves to numeric columns."""
        assert resolve_columns(schema, "numeric") == ["price", "qty"]

    def test_categorical_string(self, schema: Schema) -> None:
        """'categorical' string resolves to categorical columns."""
        assert resolve_columns(schema, "categorical") == ["city"]

    def test_temporal_string(self, schema: Schema) -> None:
        """'temporal' string resolves to temporal columns."""
        assert resolve_columns(schema, "temporal") == ["created"]

    def test_boolean_string(self, schema: Schema) -> None:
        """'boolean' string resolves to boolean columns."""
        assert resolve_columns(schema, "boolean") == ["active"]

    def test_all_string(self, schema: Schema) -> None:
        """'all' string resolves to every column."""
        result = resolve_columns(schema, "all")
        assert result == ["price", "qty", "city", "created", "active"]

    def test_unknown_string_raises(self, schema: Schema) -> None:
        """Unknown string literal raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column specification"):
            resolve_columns(schema, "unknown")

    # --- Explicit list ---

    def test_explicit_list(self, schema: Schema) -> None:
        """Explicit column list is validated and returned."""
        assert resolve_columns(schema, ["price", "city"]) == ["price", "city"]

    def test_explicit_list_missing_raises(self, schema: Schema) -> None:
        """Explicit list with missing column raises SchemaError."""
        with pytest.raises(SchemaError, match="not found"):
            resolve_columns(schema, ["price", "nonexistent"])

    def test_explicit_empty_list(self, schema: Schema) -> None:
        """Empty explicit list returns empty list."""
        assert resolve_columns(schema, []) == []

    # --- ColumnSelector ---

    def test_column_selector(self, schema: Schema) -> None:
        """ColumnSelector object is resolved."""
        assert resolve_columns(schema, numeric()) == ["price", "qty"]

    def test_pattern_selector(self) -> None:
        """PatternSelector works through resolve_columns."""
        s = Schema({"price_usd": "DOUBLE", "price_eur": "DOUBLE", "city": "VARCHAR"})
        assert resolve_columns(s, matching("price_*")) == ["price_usd", "price_eur"]

    # --- None ---

    def test_none_raises(self, schema: Schema) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="None"):
            resolve_columns(schema, None)


class TestExports:
    """Test that public API is exported correctly."""

    def test_schema_from_core(self) -> None:
        """Schema is importable from sqlearn.core."""
        import sqlearn.core

        assert sqlearn.core.Schema is Schema

    def test_schema_from_package(self) -> None:
        """Schema is importable from sqlearn."""
        import sqlearn

        assert sqlearn.Schema is Schema

    def test_selectors_from_package(self) -> None:
        """Selector factories are importable from sqlearn."""
        import sqlearn

        assert sqlearn.numeric is numeric
        assert sqlearn.categorical is categorical
        assert sqlearn.temporal is temporal
        assert sqlearn.boolean is boolean
        assert sqlearn.matching is matching
        assert sqlearn.dtype is dtype

    def test_column_selector_from_package(self) -> None:
        """ColumnSelector base class is importable from sqlearn."""
        import sqlearn

        assert sqlearn.ColumnSelector is ColumnSelector

    def test_resolve_columns_not_in_package(self) -> None:
        """resolve_columns is internal, not in sqlearn namespace."""
        import sqlearn

        assert not hasattr(sqlearn, "resolve_columns")

    def test_all_columns_from_package(self) -> None:
        """all_columns is importable from sqlearn."""
        import sqlearn

        assert sqlearn.all_columns is all_columns

    def test_columns_from_package(self) -> None:
        """columns is importable from sqlearn."""
        import sqlearn

        assert sqlearn.columns is columns


# ---------------------------------------------------------------------------
# Selector composition operators
# ---------------------------------------------------------------------------


class TestUnionOperator:
    """Test the | (union) operator on ColumnSelector."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with diverse types for composition tests."""
        return Schema(
            {
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
                "created": "TIMESTAMP",
                "active": "BOOLEAN",
            }
        )

    def test_union_two_types(self, schema: Schema) -> None:
        """Union of numeric and boolean selects columns from both."""
        result = (numeric() | boolean()).resolve(schema)
        assert result == ["price", "qty", "active"]

    def test_union_preserves_order(self, schema: Schema) -> None:
        """Union preserves left-first order and deduplicates."""
        result = (boolean() | numeric()).resolve(schema)
        assert result == ["active", "price", "qty"]

    def test_union_no_duplicates(self, schema: Schema) -> None:
        """Union of overlapping selectors has no duplicates."""
        result = (numeric() | numeric()).resolve(schema)
        assert result == ["price", "qty"]

    def test_union_with_pattern(self, schema: Schema) -> None:
        """Union with a pattern selector works."""
        result = (numeric() | matching("ci*")).resolve(schema)
        assert result == ["price", "qty", "city"]

    def test_union_empty_left(self) -> None:
        """Union where left matches nothing returns right only."""
        s = Schema({"city": "VARCHAR"})
        result = (numeric() | categorical()).resolve(s)
        assert result == ["city"]

    def test_union_empty_right(self) -> None:
        """Union where right matches nothing returns left only."""
        s = Schema({"price": "DOUBLE"})
        result = (numeric() | categorical()).resolve(s)
        assert result == ["price"]

    def test_union_both_empty(self) -> None:
        """Union where both match nothing returns empty list."""
        s = Schema({"data": "BLOB"})
        result = (numeric() | categorical()).resolve(s)
        assert result == []

    def test_union_is_column_selector(self) -> None:
        """Union result is a ColumnSelector."""
        assert isinstance(numeric() | boolean(), ColumnSelector)

    def test_union_repr(self) -> None:
        """Union selector has readable repr."""
        sel = numeric() | boolean()
        assert repr(sel) == "(numeric() | boolean())"


class TestIntersectionOperator:
    """Test the & (intersection) operator on ColumnSelector."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with columns sharing name and type patterns."""
        return Schema(
            {
                "price_usd": "DOUBLE",
                "price_eur": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
            }
        )

    def test_intersection_type_and_pattern(self, schema: Schema) -> None:
        """Intersection of numeric and pattern keeps only matches."""
        result = (numeric() & matching("price_*")).resolve(schema)
        assert result == ["price_usd", "price_eur"]

    def test_intersection_no_overlap(self, schema: Schema) -> None:
        """Intersection with no overlap returns empty list."""
        result = (categorical() & numeric()).resolve(schema)
        assert result == []

    def test_intersection_full_overlap(self, schema: Schema) -> None:
        """Intersection where both match same columns returns those columns."""
        result = (numeric() & numeric()).resolve(schema)
        assert result == ["price_usd", "price_eur", "qty"]

    def test_intersection_preserves_right_order(self, schema: Schema) -> None:
        """Intersection preserves order from the right selector."""
        # matching("*") yields all columns, numeric() filters
        result = (matching("*") & numeric()).resolve(schema)
        assert result == ["price_usd", "price_eur", "qty"]

    def test_intersection_is_column_selector(self) -> None:
        """Intersection result is a ColumnSelector."""
        assert isinstance(numeric() & matching("*"), ColumnSelector)

    def test_intersection_repr(self) -> None:
        """Intersection selector has readable repr."""
        sel = numeric() & matching("price_*")
        assert repr(sel) == "(numeric() & matching('price_*'))"


class TestNegationOperator:
    """Test the ~ (negation) operator on ColumnSelector."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with diverse types for negation tests."""
        return Schema(
            {
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
                "active": "BOOLEAN",
            }
        )

    def test_negate_numeric(self, schema: Schema) -> None:
        """Negating numeric selects non-numeric columns."""
        result = (~numeric()).resolve(schema)
        assert result == ["city", "active"]

    def test_negate_categorical(self, schema: Schema) -> None:
        """Negating categorical selects non-categorical columns."""
        result = (~categorical()).resolve(schema)
        assert result == ["price", "qty", "active"]

    def test_negate_all_types(self) -> None:
        """Negating a selector that matches everything returns empty list."""
        s = Schema({"a": "INT", "b": "DOUBLE"})
        result = (~numeric()).resolve(s)
        assert result == []

    def test_negate_no_types(self) -> None:
        """Negating a selector that matches nothing returns all columns."""
        s = Schema({"a": "INT"})
        result = (~categorical()).resolve(s)
        assert result == ["a"]

    def test_double_negation(self, schema: Schema) -> None:
        """Double negation returns original selection."""
        result = (~~numeric()).resolve(schema)
        assert result == ["price", "qty"]

    def test_negation_preserves_schema_order(self, schema: Schema) -> None:
        """Negation returns columns in schema insertion order."""
        result = (~boolean()).resolve(schema)
        assert result == ["price", "qty", "city"]

    def test_negation_is_column_selector(self) -> None:
        """Negation result is a ColumnSelector."""
        assert isinstance(~numeric(), ColumnSelector)

    def test_negation_repr(self) -> None:
        """Negation selector has readable repr."""
        sel = ~numeric()
        assert repr(sel) == "~numeric()"


class TestDifferenceOperator:
    """Test the - (difference) operator on ColumnSelector."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with varied types and name patterns."""
        return Schema(
            {
                "id_user": "INTEGER",
                "id_order": "INTEGER",
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
            }
        )

    def test_difference_removes_pattern(self, schema: Schema) -> None:
        """Difference removes columns matching the right selector."""
        result = (numeric() - matching("id_*")).resolve(schema)
        assert result == ["price", "qty"]

    def test_difference_no_overlap(self, schema: Schema) -> None:
        """Difference with no overlap returns left unchanged."""
        result = (numeric() - categorical()).resolve(schema)
        assert result == ["id_user", "id_order", "price", "qty"]

    def test_difference_full_overlap(self, schema: Schema) -> None:
        """Difference with full overlap returns empty list."""
        result = (numeric() - numeric()).resolve(schema)
        assert result == []

    def test_difference_preserves_left_order(self, schema: Schema) -> None:
        """Difference preserves order from the left selector."""
        result = (numeric() - matching("id_*")).resolve(schema)
        assert result == ["price", "qty"]

    def test_difference_is_column_selector(self) -> None:
        """Difference result is a ColumnSelector."""
        assert isinstance(numeric() - matching("*"), ColumnSelector)

    def test_difference_repr(self) -> None:
        """Difference selector has readable repr."""
        sel = numeric() - matching("id_*")
        assert repr(sel) == "(numeric() - matching('id_*'))"


class TestAllColumnsSelector:
    """Test all_columns() selector."""

    def test_selects_all(self) -> None:
        """all_columns() returns every column."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "BOOLEAN"})
        assert all_columns().resolve(s) == ["a", "b", "c"]

    def test_empty_schema(self) -> None:
        """all_columns() on empty schema returns empty list."""
        s = Schema({})
        assert all_columns().resolve(s) == []

    def test_preserves_order(self) -> None:
        """all_columns() preserves schema insertion order."""
        s = Schema({"z": "INT", "a": "INT", "m": "INT"})
        assert all_columns().resolve(s) == ["z", "a", "m"]

    def test_is_column_selector(self) -> None:
        """all_columns() returns a ColumnSelector."""
        assert isinstance(all_columns(), ColumnSelector)

    def test_repr(self) -> None:
        """all_columns() has readable repr."""
        assert repr(all_columns()) == "all_columns()"

    def test_minus_categorical(self) -> None:
        """all_columns() - categorical() excludes categorical columns."""
        s = Schema({"price": "DOUBLE", "city": "VARCHAR", "active": "BOOLEAN"})
        result = (all_columns() - categorical()).resolve(s)
        assert result == ["price", "active"]

    def test_with_resolve_columns(self) -> None:
        """all_columns() works through resolve_columns."""
        s = Schema({"a": "INT", "b": "VARCHAR"})
        assert resolve_columns(s, all_columns()) == ["a", "b"]


class TestColumnsSelector:
    """Test columns() selector."""

    def test_selects_named(self) -> None:
        """columns() selects exactly the named columns."""
        s = Schema({"price": "DOUBLE", "qty": "INTEGER", "city": "VARCHAR"})
        result = columns("price", "qty").resolve(s)
        assert result == ["price", "qty"]

    def test_preserves_argument_order(self) -> None:
        """columns() preserves the order of the arguments."""
        s = Schema({"a": "INT", "b": "INT", "c": "INT"})
        result = columns("c", "a").resolve(s)
        assert result == ["c", "a"]

    def test_skips_nonexistent(self) -> None:
        """columns() silently skips names not in the schema."""
        s = Schema({"price": "DOUBLE"})
        result = columns("price", "nonexistent").resolve(s)
        assert result == ["price"]

    def test_all_nonexistent(self) -> None:
        """columns() returns empty list when no names match."""
        s = Schema({"price": "DOUBLE"})
        result = columns("missing", "absent").resolve(s)
        assert result == []

    def test_empty_args(self) -> None:
        """columns() with no args returns empty list."""
        s = Schema({"price": "DOUBLE"})
        result = columns().resolve(s)
        assert result == []

    def test_no_duplicates(self) -> None:
        """columns() with duplicate names returns each once."""
        s = Schema({"a": "INT", "b": "INT"})
        result = columns("a", "a", "b").resolve(s)
        assert result == ["a", "a", "b"]  # mirrors user intent

    def test_is_column_selector(self) -> None:
        """columns() returns a ColumnSelector."""
        assert isinstance(columns("a"), ColumnSelector)

    def test_repr_single(self) -> None:
        """columns() with one name has readable repr."""
        assert repr(columns("price")) == "columns('price')"

    def test_repr_multiple(self) -> None:
        """columns() with multiple names has readable repr."""
        assert repr(columns("price", "qty")) == "columns('price', 'qty')"

    def test_repr_empty(self) -> None:
        """columns() with no names has readable repr."""
        assert repr(columns()) == "columns()"

    def test_with_resolve_columns(self) -> None:
        """columns() works through resolve_columns."""
        s = Schema({"a": "INT", "b": "VARCHAR", "c": "DOUBLE"})
        assert resolve_columns(s, columns("a", "c")) == ["a", "c"]

    def test_composed_with_difference(self) -> None:
        """columns() can be used as the right side of difference."""
        s = Schema({"price": "DOUBLE", "qty": "INTEGER", "city": "VARCHAR"})
        result = (numeric() - columns("qty")).resolve(s)
        assert result == ["price"]


# ---------------------------------------------------------------------------
# Composition chains
# ---------------------------------------------------------------------------


class TestCompositionChains:
    """Test chaining multiple composition operators."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with varied types and name patterns for chain tests."""
        return Schema(
            {
                "id_user": "INTEGER",
                "id_order": "BIGINT",
                "price": "DOUBLE",
                "qty": "INTEGER",
                "city": "VARCHAR",
                "name": "TEXT",
                "created": "TIMESTAMP",
                "active": "BOOLEAN",
            }
        )

    def test_union_then_difference(self, schema: Schema) -> None:
        """(numeric() | boolean()) - matching('id_*') chains correctly."""
        result = ((numeric() | boolean()) - matching("id_*")).resolve(schema)
        assert result == ["price", "qty", "active"]

    def test_negation_then_intersection(self, schema: Schema) -> None:
        """(~categorical()) & matching('pr*') chains correctly."""
        result = ((~categorical()) & matching("pr*")).resolve(schema)
        assert result == ["price"]

    def test_triple_union(self, schema: Schema) -> None:
        """Three-way union selects from all three."""
        result = (numeric() | categorical() | boolean()).resolve(schema)
        assert result == ["id_user", "id_order", "price", "qty", "city", "name", "active"]

    def test_difference_chain(self, schema: Schema) -> None:
        """numeric() - matching('id_*') - columns('qty') removes both."""
        result = (numeric() - matching("id_*") - columns("qty")).resolve(schema)
        assert result == ["price"]

    def test_all_minus_union(self, schema: Schema) -> None:
        """all_columns() - (categorical() | temporal()) works."""
        result = (all_columns() - (categorical() | temporal())).resolve(schema)
        assert result == ["id_user", "id_order", "price", "qty", "active"]

    def test_intersection_preserves_subset(self, schema: Schema) -> None:
        """Intersection of all_columns() with numeric() equals numeric()."""
        result = (all_columns() & numeric()).resolve(schema)
        assert result == numeric().resolve(schema)

    def test_complex_chain_repr(self) -> None:
        """Complex composition has readable repr."""
        sel = (numeric() | boolean()) - matching("id_*")
        assert "numeric()" in repr(sel)
        assert "boolean()" in repr(sel)
        assert "matching('id_*')" in repr(sel)

    def test_composed_selectors_work_with_resolve_columns(self, schema: Schema) -> None:
        """Composed selectors work through resolve_columns."""
        sel = numeric() - matching("id_*")
        result = resolve_columns(schema, sel)
        assert result == ["price", "qty"]
