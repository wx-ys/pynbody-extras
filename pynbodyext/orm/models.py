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
    recipe_hash: Mapped[str] = mapped_column(ForeignKey(Recipe.hash, ondelete="RESTRICT"), index=True)
    store_name: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)
    started_at: Mapped[float] = mapped_column(Float)
    finished_at: Mapped[float | None] = mapped_column(Float)
    reused: Mapped[bool | None] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    input: Mapped[DataObject] = relationship(back_populates="runs")
    input_location: Mapped[DataLocation | None] = relationship()
    recipe: Mapped[Recipe] = relationship(back_populates="runs")
    result_link: Mapped[ResultLink | None] = relationship(back_populates="run", uselist=False)


class ResultLink(Base):
    __tablename__ = "pynbodyext_orm_result_links"

    run_id: Mapped[str] = mapped_column(ForeignKey(Run.id, ondelete="RESTRICT"), primary_key=True)
    ref_json: Mapped[str] = mapped_column(Text)
    store_schema: Mapped[str] = mapped_column(Text)
    run: Mapped[Run] = relationship(back_populates="result_link")
