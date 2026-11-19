"""Publish observed origins only with the producer's confirmed immutable receipt."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import event, func, select

import pynbodyext as pe
from pynbodyext.calculate import Pipeline
from pynbodyext.database.database import DatabasePersistenceError
from pynbodyext.database.models import ResultSummary, SaveOperation
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

from test_database import write_snapshot


def pipeline():
    return Pipeline({"one": ParamSum("mass"), "two": ParamSum("mass")}).filter(FamilyFilter("dm"))


def counts(db):
    from pynbodyext.database.member_models import OutputObservation, SelectionOrigin, SelectionUse
    with db._repo.session() as session:
        return tuple(session.scalar(select(func.count()).select_from(model))
                     for model in (SelectionOrigin, SelectionUse, OutputObservation, ResultSummary))


def test_no_receipt_cannot_be_repaired_from_current_cache(tmp_path, monkeypatch):
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        first = pipeline().run(sim, database=db)
        snap = next(iter(db.snapshots))
        assert hasattr(snap, "members"), "membership must be pinned to a confirmed producing version"
        former = counts(db)
        monkeypatch.setattr(db._store, "_put", lambda _: (_ for _ in ()).throw(RuntimeError("before receipt")))
        with pytest.raises(RuntimeError, match="before receipt"):
            pipeline().run(sim).save(db)
        with db._repo.session() as session:
            pending = session.scalars(select(SaveOperation).where(SaveOperation.ref_json.is_(None))).one()
        with pytest.raises(ValueError, match="exact receipt"):
            db._finish_save(pending.id)
        db.recover(first.store_ref)
        assert counts(db) == former
        assert len(db.versions(first.store_ref)) == 1


@pytest.mark.parametrize("boundary", ["before_finalize", "first_use"])
def test_failed_publication_is_invisible_and_exact_recovery_idempotent(tmp_path, monkeypatch, boundary):
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db, pe.Database(tmp_path / "db", read_only=True) as reader:
        snap = db.register(sim, simulation="run", snapshot="s")
        assert hasattr(snap, "members"), "member publication is part of the final save transaction"
        old_finish = db._finish_save
        if boundary == "before_finalize":
            monkeypatch.setattr(db, "_finish_save", lambda _: (_ for _ in ()).throw(RuntimeError("before finalize")))
        else:
            from pynbodyext.database.member_models import SelectionUse
            def fail_after_use(connection, cursor, statement, *args):
                if statement.lstrip().upper().startswith("INSERT") and SelectionUse.__tablename__ in statement:
                    assert not list(reader.snapshots["s"].members.where())
                    raise RuntimeError("interrupted after a use insert")
            event.listen(db._engine, "after_cursor_execute", fail_after_use)
        with pytest.raises(DatabasePersistenceError) as failure:
            pipeline().run(snap)
        ref = failure.value.result_ref
        assert ref.version_id is not None
        assert counts(reader) == (0, 0, 0, 0)
        monkeypatch.setattr(db, "_finish_save", old_finish)
        if boundary == "first_use":
            event.remove(db._engine, "after_cursor_execute", fail_after_use)
        db.recover(ref)
        db.recover(ref)
        assert counts(reader) == (1, 2, 2, 1)
        selected = reader.snapshots["s"].members["dm"]
        from pynbodyext.database.member_models import SelectionUse
        with reader._repo.session() as session:
            by_origin = session.scalars(select(SelectionUse).where(SelectionUse.origin_id == selected.origin.id)).all()
            assert {(u.version_id, tuple(json.loads(u.output_path_json)), u.role) for u in by_origin} == {
                (ref.version_id, ("one",), "output_filter"), (ref.version_id, ("two",), "output_filter")
            }
        assert db.versions(ref) == [ref]


def test_two_connections_share_an_origin_and_pin_independent_uses(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    gate = Barrier(2)
    with pe.Database(path) as setup:
        root = setup.register(sim, simulation="run", snapshot="s")
        assert hasattr(root, "members"), "independent writers must reuse the same archived origin"
    def produce(_):
        with pe.Database(path) as db:
            gate.wait(timeout=10)
            return pipeline().run(db.snapshots["s"], mode="refresh").store_ref
    with ThreadPoolExecutor(max_workers=2) as pool:
        refs = list(pool.map(produce, range(2)))
    with pe.Database(path, read_only=True) as reader:
        assert len({ref.version_id for ref in refs}) == 2
        assert counts(reader) == (1, 4, 4, 2)
        assert len(reader.snapshots["s"].members["dm"].origins) == 1


def test_historical_observations_survive_partial_refresh_and_cache_reuse(tmp_path, monkeypatch):
    from pynbodyext.calculate import PropertyBase

    @PropertyBase.dataclass
    class Failable(PropertyBase):
        explode = False
        def calculate(self, sim, params=None):
            if self.explode:
                raise ArithmeticError("no measurement")
            return len(sim)

    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        assert hasattr(snap, "members"), "failed scientific refreshes cannot publish observations"
        cc = Pipeline({"good": ParamSum("mass"), "bad": Failable()}).filter(FamilyFilter("dm"))
        first = cc.run(snap)
        before = counts(db)
        assert cc.run(snap).from_store
        assert counts(db) == before
        monkeypatch.setattr(Failable, "explode", True)
        partial = cc.run(snap, mode="refresh", errors="collect_partial")
        assert partial.errors
        assert counts(db) == before
        assert db.fetch(first.store_ref).value["bad"] == 4
        assert len(db.runs) == 3


def test_legacy_read_only_browses_without_schema_writes_then_writable_upgrades(tmp_path):
    from sqlalchemy import inspect
    from pynbodyext.database.models import Base

    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    with pe.Database(path) as db:
        calc = ParamSum("mass")
        ref = calc.run(sim, database=db).store_ref
        snap = next(iter(db.snapshots))
        assert hasattr(snap, "members"), "legacy readers must expose missing observation evidence"
        from pynbodyext.database.member_models import OutputObservation, SelectionOrigin, SelectionUse, StagedMemberInput, StagedObservations
        added = {item.__table__ for item in (SelectionOrigin, SelectionUse, OutputObservation, StagedMemberInput, StagedObservations)}
        with db._engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                if table in added:
                    table.drop(conn)
    with pe.Database(path, read_only=True) as db:
        sql = []
        event.listen(db._engine, "before_cursor_execute", lambda c, cur, stmt, *args: sql.append(stmt))
        assert float(db.fetch(ref).value) == 10.
        assert not list(next(iter(db.snapshots)).members.where())
        assert all(s.lstrip().split(None, 1)[0].upper() in {"SELECT", "PRAGMA", "BEGIN"} for s in sql)
        assert not inspect(db._engine).has_table(SelectionOrigin.__tablename__)
    with pe.Database(path) as db:
        assert inspect(db._engine).has_table(SelectionOrigin.__tablename__)
        stars = next(iter(db.snapshots)).members.register("star")
        assert len(stars.load()) == 0


def test_concurrent_first_runs_on_same_catalogue_halo_share_binding_and_origin(tmp_path, monkeypatch):
    import inspect
    import numpy as np
    from pynbody.halo.number_array import HaloNumberCatalogue
    from pynbodyext.database import catalogues

    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, -1])
    path = tmp_path / "db"
    with pe.Database(path) as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp", ignore=-1), name="groups")
    at_binding = Barrier(2)
    at_unlocked_read = Barrier(2)
    original = catalogues._bind_object

    def bind(db, object_id, input_id):
        at_binding.wait(timeout=10)
        return original(db, object_id, input_id)
    monkeypatch.setattr(catalogues, "_bind_object", bind)

    def writer(_):
        with pe.Database(path) as db:
            def on_statement(connection, cursor, stmt, *args):
                unlocked = not connection.connection.driver_connection.in_transaction
                if unlocked and stmt.startswith("SELECT") and "database_object_selections" in stmt:
                    if any(frame.frame.f_code is original.__code__ for frame in inspect.stack()):
                        # A read-then-insert implementation lets both writers decide that the same binding is absent.
                        at_unlocked_read.wait(timeout=10)
            event.listen(db._engine, "after_cursor_execute", on_statement)
            return pipeline().run(db.snapshots["s"].halos[5], mode="refresh").store_ref
    with ThreadPoolExecutor(max_workers=2) as pool:
        refs = list(pool.map(writer, range(2)))
    with pe.Database(path, read_only=True) as db:
        assert len({ref.version_id for ref in refs}) == 2
        assert counts(db) == (1, 4, 4, 2)
        assert len(db.snapshots["s"].halos[5].members["dm"].origins) == 1


def test_use_foreign_keys_and_unique_occurrence_are_enforced(tmp_path):
    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError
    from pynbodyext.database.member_models import SelectionUse
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        ref = pipeline().run(sim, database=db).store_ref
        with db._repo.session() as session:
            use = session.scalars(select(SelectionUse).where(
                SelectionUse.version_id == ref.version_id, SelectionUse.output_path_json == '["one"]'
            )).one()
            original = {column.name: getattr(use, column.name) for column in SelectionUse.__table__.columns}
        for changes in ({}, {"version_id": "missing"}, {"origin_id": "unknown"}, {"output_path_json": '["no-such-output"]'}):
            with pytest.raises(IntegrityError), db._engine.begin() as conn:
                conn.execute(insert(SelectionUse).values(original | changes))
        assert counts(db) == (1, 2, 2, 1)


@pytest.mark.parametrize("release", ["recover", "ordinary_save", "explicit_registration"])
def test_pending_blocks_are_private_until_an_independent_confirmation(tmp_path, monkeypatch, release):
    from test_database_members import MassBelow
    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    with pe.Database(path) as db, pe.Database(path, read_only=True) as reader:
        plain = ParamSum("mass")
        old = plain.run(sim[[0, 1]], database=db)
        exact = db._registered_input(old.store_ref.sim_signature)
        remote = next(iter(reader.snapshots))
        assert {view.key for view in remote.selections} == {exact.key}
        calc = Pipeline({"old": plain.filter(MassBelow()), "new": plain.filter(FamilyFilter("dm"))})
        finish = db._finish_save
        monkeypatch.setattr(db, "_finish_save", lambda _: (_ for _ in ()).throw(RuntimeError("no confirmation")))
        with pytest.raises(DatabasePersistenceError) as err:
            calc.run(sim, database=db)
        assert {view.key for view in remote.selections} == {exact.key}
        assert len(remote.members.where()) == 0
        monkeypatch.setattr(db, "_finish_save", finish)
        if release == "recover":
            db.recover(err.value.result_ref)
        elif release == "ordinary_save":
            plain.run(sim[[0, 1, 2, 3]], database=db)
        else:
            next(iter(db.snapshots)).members.register("dm")
        assert len(remote.selections) == 2
        db.recover(err.value.result_ref)
        assert len(remote.selections) == 2


def test_recovery_revalidates_staged_index_blocks_before_releasing_views(tmp_path, monkeypatch):
    from pynbodyext.orm.models import Selection
    from pynbodyext.database.member_models import StagedMemberInput
    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    with pe.Database(path) as db, pe.Database(path, read_only=True) as reader:
        finish = db._finish_save
        monkeypatch.setattr(db, "_finish_save", lambda _: (_ for _ in ()).throw(RuntimeError("before finalize")))
        with pytest.raises(DatabasePersistenceError) as err:
            pipeline().run(sim, database=db)
        ref = err.value.result_ref
        monkeypatch.setattr(db, "_finish_save", finish)
        with db._repo.session() as session, session.begin():
            pk = session.scalars(select(StagedMemberInput.input_id)).one()
            block = session.scalars(select(Selection).where(Selection.view_input_id == pk)).one()
            old = block.definition_json
            block.definition_json = old.replace("[0,1,2,3]", "[1,0,2,3]")
        with pytest.raises(ValueError, match="selection|digest"):
            db.recover(ref)
        assert len(next(iter(reader.snapshots)).selections) == 0
        assert counts(reader) == (0, 0, 0, 0)
        with db._repo.session() as session, session.begin():
            session.get(Selection, block.id).definition_json = old
        db.recover(ref)
        assert len(next(iter(reader.snapshots)).selections) == 1
        assert counts(reader) == (1, 2, 2, 1)


@pytest.mark.parametrize("corruption", ["other_root", "wrong_count", "invalid_descriptor"])
def test_metadata_and_replay_reject_foreign_or_damaged_coordinate_target(tmp_path, corruption):
    import numpy as np
    from pynbodyext.database.member_models import OutputObservation, SelectionOrigin
    from pynbodyext.orm.models import Selection
    from pynbodyext.transforms import ShiftPosTo

    calc = ParamSum("mass").transform(ShiftPosTo(np.array([1., 0., 0.]), move_all=False)).filter(FamilyFilter("dm"))
    path = tmp_path / "db"
    with pe.Database(path) as db:
        a = write_snapshot(tmp_path / "a" / "particles")
        b = write_snapshot(tmp_path / "b" / "particles")
        first = calc.run(a[[0, 2]], database=db)
        second = calc.run(b[[1, 3]], database=db)
        source_a = db._registered_input(first.store_ref.sim_signature)
        source_b = db._registered_input(second.store_ref.sim_signature)
        assert source_a.members["dm"].origin.context.steps[0].target_indices == (0, 2)
        assert source_b.members["dm"].origin.context.steps[0].target_indices == (1, 3)
        with db._repo.session() as session, session.begin():
            origin = session.get(SelectionOrigin, source_a.members["dm"].origin.id)
            output = session.get(OutputObservation, (first.store_ref.version_id, "[]"))
            recorded = json.loads(origin.context_json)
            target_id = recorded["steps"][0]["target_selection_id"]
            if corruption == "other_root":
                other = session.get(OutputObservation, (second.store_ref.version_id, "[]"))
                recorded["steps"][0]["target_selection_id"] = json.loads(other.context_json)["steps"][0]["target_selection_id"]
            elif corruption == "wrong_count":
                recorded["root_count"] = 5
            else:
                session.get(Selection, target_id).descriptor_key = "0" * 64
            origin.context_json = output.context_json = json.dumps(recorded)
    with pe.Database(path, read_only=True) as db:
        db.bind_loader("default", lambda _: pytest.fail("corrupt coordinate evidence reached particle IO"))
        source = db._registered_input(first.store_ref.sim_signature)
        member = source.members["dm"]
        with pytest.raises(ValueError, match="root|count|descriptor"):
            _ = member.origin.context
        with pytest.raises(ValueError, match="root|count|descriptor"):
            member.load(frame="selection")
        with pytest.raises(ValueError, match="root|count|descriptor"):
            _ = source.results.where(calculator=calc).one().context
        if corruption != "invalid_descriptor":
            import pynbody
            db.bind_loader("default", pynbody.load)
            original = member.load()
            assert tuple(original.get_index_list(original.ancestor)) == (0, 2)
