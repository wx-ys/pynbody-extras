"""Metadata for confirmed selection origins and exact, version-pinned uses."""

from __future__ import annotations

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from pynbodyext.orm.models import DataObject

from .models import Base, ResultSummary, SaveOperation


class StagedObservations(Base):
    """A save's unpublished observations. Particle sequences refer to immutable Selections."""

    __tablename__ = "pynbodyext_database_staged_observations"
    operation_id: Mapped[str] = mapped_column(ForeignKey(SaveOperation.id), primary_key=True)
    observations_json: Mapped[str] = mapped_column(Text)


class StagedMemberInput(Base):
    """An exact view interned solely for a save that has not yet been confirmed."""

    __tablename__ = "pynbodyext_database_staged_member_inputs"
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id), primary_key=True)


class SelectionOrigin(Base):
    __tablename__ = "pynbodyext_database_selection_origins"
    __table_args__ = (
        Index("ix_selection_source_parent_filter", "source_locator_json", "parent_origin_id", "filter_hash"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id))
    source_locator_json: Mapped[str] = mapped_column(Text)
    parent_origin_id: Mapped[str | None] = mapped_column(ForeignKey("pynbodyext_database_selection_origins.id"))
    member_input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id), index=True)
    particle_count: Mapped[int]
    filter_hash: Mapped[str] = mapped_column(String(64))
    filter_signature: Mapped[str] = mapped_column(Text)
    parameters_json: Mapped[str] = mapped_column(Text)
    context_json: Mapped[str] = mapped_column(Text)


class OutputObservation(Base):
    __tablename__ = "pynbodyext_database_output_observations"

    version_id: Mapped[str] = mapped_column(ForeignKey(ResultSummary.version_id), primary_key=True)
    output_path_json: Mapped[str] = mapped_column(Text, primary_key=True)
    source_locator_json: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[str] = mapped_column(Text)


class SelectionUse(Base):
    __tablename__ = "pynbodyext_database_selection_uses"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "output_path_json"], [OutputObservation.version_id, OutputObservation.output_path_json]
        ),
    )

    version_id: Mapped[str] = mapped_column(ForeignKey(ResultSummary.version_id), primary_key=True)
    output_path_json: Mapped[str] = mapped_column(Text, primary_key=True)
    occurrence_path_json: Mapped[str] = mapped_column(Text, primary_key=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True)
    origin_id: Mapped[str] = mapped_column(ForeignKey(SelectionOrigin.id), primary_key=True, index=True)
