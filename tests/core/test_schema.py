"""Tests for sqlearn.core.schema."""

from __future__ import annotations

import pytest

from sqlearn.core.schema import Schema


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
