"""Relational metadata only; scientific values remain in ResultStore."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Simulation(Base):
    __tablename__ = "pynbodyext_orm_simulations"
    __table_args__ = (UniqueConstraint("project_id", "logical_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(Text)
    logical_id: Mapped[str] = mapped_column(Text)
    timesteps: Mapped[list[Timestep]] = relationship(back_populates="simulation")


class Timestep(Base):
    __tablename__ = "pynbodyext_orm_timesteps"
    __table_args__ = (UniqueConstraint("simulation_id", "logical_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[int] = mapped_column(ForeignKey(Simulation.id, ondelete="RESTRICT"), index=True)
    logical_id: Mapped[str] = mapped_column(Text)
    simulation: Mapped[Simulation] = relationship(back_populates="timesteps")
    inputs: Mapped[list[DataObject]] = relationship(back_populates="timestep")


class DataObject(Base):
    __tablename__ = "pynbodyext_orm_data_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestep_id: Mapped[int] = mapped_column(ForeignKey(Timestep.id, ondelete="RESTRICT"), index=True)
    identity_key: Mapped[str] = mapped_column(Text, unique=True)
    identity_json: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("pynbodyext_orm_data_objects.id", ondelete="RESTRICT"))
    timestep: Mapped[Timestep] = relationship(back_populates="inputs")
    parent: Mapped[DataObject | None] = relationship(back_populates="children", remote_side=lambda: DataObject.id)
    children: Mapped[list[DataObject]] = relationship(back_populates="parent")
    locations: Mapped[list[DataLocation]] = relationship(back_populates="data_object")
    runs: Mapped[list[Run]] = relationship(back_populates="input")

    @property
    def signature(self) -> tuple[Any, ...]:
        """The logical snapshot/view signature, independent of ORM primary keys."""
        return tuple(json.loads(self.identity_json))


class DataLocation(Base):
    __tablename__ = "pynbodyext_orm_data_locations"
    __table_args__ = (UniqueConstraint("data_object_id", "uri"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data_object_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id, ondelete="RESTRICT"), index=True)
    uri: Mapped[str] = mapped_column(Text)
    data_object: Mapped[DataObject] = relationship(back_populates="locations")


class Catalogue(Base):
    __tablename__ = "pynbodyext_orm_catalogues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    root_input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id, ondelete="RESTRICT"), index=True)
    identity_key: Mapped[str] = mapped_column(Text, unique=True)
    catalogue_id: Mapped[str] = mapped_column(Text)
    finder: Mapped[str] = mapped_column(Text)
    revision: Mapped[str | None] = mapped_column(Text)
    loader_id: Mapped[str] = mapped_column(Text)
    root_input: Mapped[DataObject] = relationship()
    objects: Mapped[list[CatalogueObject]] = relationship(back_populates="catalogue")


class CatalogueObject(Base):
    __tablename__ = "pynbodyext_orm_catalogue_objects"
    __table_args__ = (
        UniqueConstraint("catalogue_fk", "kind", "typed_object_key"),
        CheckConstraint("kind IN ('halo', 'subhalo')"),
        CheckConstraint("host_id IS NULL OR kind = 'subhalo'"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    catalogue_fk: Mapped[int] = mapped_column(ForeignKey(Catalogue.id, ondelete="RESTRICT"), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    typed_object_key: Mapped[str] = mapped_column(Text)
    host_id: Mapped[int | None] = mapped_column(
        ForeignKey("pynbodyext_orm_catalogue_objects.id", ondelete="RESTRICT"), index=True
    )
    catalogue: Mapped[Catalogue] = relationship(back_populates="objects")
    host: Mapped[CatalogueObject | None] = relationship(
        back_populates="subhalos", remote_side=lambda: CatalogueObject.id
    )
    subhalos: Mapped[list[CatalogueObject]] = relationship(back_populates="host")


class Selection(Base):
    __tablename__ = "pynbodyext_orm_selections"
    __table_args__ = (CheckConstraint("expected_count >= 0"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    root_input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id, ondelete="RESTRICT"), index=True)
    view_input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id, ondelete="RESTRICT"), index=True)
    schema: Mapped[str] = mapped_column(Text)
    definition_json: Mapped[str] = mapped_column(Text)
    expected_count: Mapped[int] = mapped_column(Integer)
    descriptor_key: Mapped[str] = mapped_column(String(64), unique=True)
    root_input: Mapped[DataObject] = relationship(foreign_keys=[root_input_id])
    view_input: Mapped[DataObject] = relationship(foreign_keys=[view_input_id])


class Recipe(Base):
    __tablename__ = "pynbodyext_orm_recipes"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    signature_text: Mapped[str] = mapped_column(Text)
    runs: Mapped[list[Run]] = relationship(back_populates="recipe")


class Run(Base):
    __tablename__ = "pynbodyext_orm_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'succeeded', 'failed')"),
        CheckConstraint(
            "(status = 'running' AND finished_at IS NULL) OR (status != 'running' AND finished_at IS NOT NULL)"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey(DataObject.id, ondelete="RESTRICT"), index=True)
    input_location_id: Mapped[int | None] = mapped_column(ForeignKey(DataLocation.id, ondelete="RESTRICT"), index=True)
    selection_id: Mapped[int | None] = mapped_column(ForeignKey(Selection.id, ondelete="RESTRICT"), index=True)
    recipe_hash: Mapped[str] = mapped_column(ForeignKey(Recipe.hash, ondelete="RESTRICT"), index=True)
    store_name: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)
    started_at: Mapped[float] = mapped_column(Float)
    finished_at: Mapped[float | None] = mapped_column(Float)
    reused: Mapped[bool | None] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    _execution_context_json: Mapped[str] = mapped_column("execution_context", Text)
    input: Mapped[DataObject] = relationship(back_populates="runs")
    input_location: Mapped[DataLocation | None] = relationship()
    selection: Mapped[Selection | None] = relationship()
    recipe: Mapped[Recipe] = relationship(back_populates="runs")
    result_link: Mapped[ResultLink | None] = relationship(back_populates="run", uselist=False)

    @property
    def execution_context(self) -> dict[str, Any]:
        """A fresh copy of the immutable invocation snapshot committed at begin."""
        return dict(json.loads(self._execution_context_json))


class ResultLink(Base):
    __tablename__ = "pynbodyext_orm_result_links"

    run_id: Mapped[str] = mapped_column(ForeignKey(Run.id, ondelete="RESTRICT"), primary_key=True)
    ref_json: Mapped[str] = mapped_column(Text)
    store_schema: Mapped[str] = mapped_column(Text)
    run: Mapped[Run] = relationship(back_populates="result_link")
