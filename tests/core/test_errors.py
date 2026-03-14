"""Tests for sqlearn.core.errors."""

from __future__ import annotations

import pytest

from sqlearn.core.errors import (
    ClassificationError,
    CompilationError,
    FitError,
    FrozenError,
    InvalidStepError,
    MissingColumnError,
    NotFittedError,
    ProFeatureError,
    SchemaError,
    SQLearnError,
    StaticViolationError,
    UnseenCategoryError,
)


class TestHierarchy:
    """Every error is a SQLearnError."""

    @pytest.mark.parametrize(
        "cls",
        [
            NotFittedError,
            SchemaError,
            MissingColumnError,
            FitError,
            CompilationError,
            UnseenCategoryError,
            ClassificationError,
            StaticViolationError,
            FrozenError,
            InvalidStepError,
            ProFeatureError,
        ],
    )
    def test_subclass_of_sqlearn_error(self, cls: type[SQLearnError]) -> None:
        """All errors inherit from SQLearnError."""
        assert issubclass(cls, SQLearnError)
        assert issubclass(cls, Exception)

    def test_missing_column_is_schema_error(self) -> None:
        """MissingColumnError is a SchemaError subclass."""
        assert issubclass(MissingColumnError, SchemaError)

    def test_catch_all(self) -> None:
        """except SQLearnError catches every subclass."""
        with pytest.raises(SQLearnError):
            raise NotFittedError("msg")
        with pytest.raises(SQLearnError):
            raise SchemaError("msg")
        with pytest.raises(SQLearnError):
            raise MissingColumnError("msg", column="x", available=[])
        with pytest.raises(SQLearnError):
            raise FitError("msg")
        with pytest.raises(SQLearnError):
            raise CompilationError("msg")
        with pytest.raises(SQLearnError):
            raise UnseenCategoryError("msg", column="x", categories=["a"])
        with pytest.raises(SQLearnError):
            raise ClassificationError("msg")
        with pytest.raises(SQLearnError):
            raise StaticViolationError("msg")
        with pytest.raises(SQLearnError):
            raise FrozenError("msg")
        with pytest.raises(SQLearnError):
            raise InvalidStepError("msg")
        with pytest.raises(SQLearnError):
            raise ProFeatureError("msg", feature="studio")


class TestMessages:
    """Errors store and display messages."""

    def test_simple_message(self) -> None:
        """Plain errors store message string."""
        err = NotFittedError("call fit() first")
        assert str(err) == "call fit() first"

    def test_schema_error_message(self) -> None:
        """SchemaError with custom message."""
        err = SchemaError("Columns already exist: ['price']")
        assert "already exist" in str(err)


class TestMissingColumnError:
    """MissingColumnError stores column + available attributes."""

    def test_attributes(self) -> None:
        """Stores column name and available columns."""
        err = MissingColumnError(
            "Column 'prce' not found",
            column="prce",
            available=["price", "city", "age"],
        )
        assert err.column == "prce"
        assert err.available == ["price", "city", "age"]
        assert "prce" in str(err)

    def test_is_schema_error(self) -> None:
        """Can be caught as SchemaError."""
        with pytest.raises(SchemaError):
            raise MissingColumnError("x", column="x", available=[])


class TestUnseenCategoryError:
    """UnseenCategoryError stores column + categories attributes."""

    def test_attributes(self) -> None:
        """Stores column and unseen categories."""
        err = UnseenCategoryError(
            "Unseen category 'Berlin' in column 'city'",
            column="city",
            categories=["Berlin", "Tokyo"],
        )
        assert err.column == "city"
        assert err.categories == ["Berlin", "Tokyo"]
        assert "Berlin" in str(err)


class TestProFeatureError:
    """ProFeatureError stores feature attribute."""

    def test_attributes(self) -> None:
        """Stores feature name."""
        err = ProFeatureError(
            "Pipeline builder requires Studio Pro",
            feature="pipeline_builder",
        )
        assert err.feature == "pipeline_builder"
        assert "Studio Pro" in str(err)
