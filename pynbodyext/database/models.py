"""User-facing metadata; values stay in the existing ResultStore tables."""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pynbodyext.orm.models import Catalogue, CatalogueObject, DataObject, Run


class Base(DeclarativeBase):
    """Database-only tables; importing the facade does not change bare ORM schemas."""


class InputInfo(Base):
    __tablename__ = "pynbodyext_database_inputs"
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id), primary_key=True)
    description: Mapped[str] = mapped_column(Text)
    default_catalogue_id: Mapped[int | None] = mapped_column(ForeignKey(Catalogue.id))


class ResultAlias(Base):
    __tablename__ = "pynbodyext_database_aliases"
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id), primary_key=True)
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    calculator_hash: Mapped[str] = mapped_column(String(64))


class SaveOperation(Base):
    __tablename__ = "pynbodyext_database_saves"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id))
    signature_text: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    name: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(ForeignKey(Run.id))
    ref_json: Mapped[str | None] = mapped_column(Text)
    reused: Mapped[bool] = mapped_column(Boolean, default=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class CatalogueBinding(Base):
    __tablename__ = "pynbodyext_database_catalogues"
    catalogue_id: Mapped[int] = mapped_column(ForeignKey(Catalogue.id), primary_key=True)
    adapter_json: Mapped[str] = mapped_column(Text)


class ObjectSelection(Base):
    __tablename__ = "pynbodyext_database_object_selections"
    object_id: Mapped[int] = mapped_column(ForeignKey(CatalogueObject.id), primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id))


class ResultSummary(Base):
    __tablename__ = "pynbodyext_database_result_summaries"
    version_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id), index=True)
    calculator_hash: Mapped[str] = mapped_column(String(64))
    ref_json: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text)
