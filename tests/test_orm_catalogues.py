"""Catalogue scientific identity is separate from snapshot views and physical hosts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import LogicalSnapshotIdentity
from pynbodyext.core.calculate.store import InMemoryResultStore


@pytest.fixture
def repo(tmp_path):
    repository = orm.AnalysisRepository(f"sqlite:///{tmp_path / 'catalogues.sqlite'}")
    yield repository
    repository.engine.dispose()


def root(repo, **kwargs):
    return repo.register_input(replace(LogicalSnapshotIdentity("project", "simulation", "099"), **kwargs))


def catalogue(repo, **kwargs):
    return repo.register_catalogue(root(repo).id, "main", finder="finder", **kwargs)


def counts(repo):
    with repo.session() as session:
        return tuple(
            session.scalar(sa.select(sa.func.count()).select_from(model))
            for model in (orm.Catalogue, orm.CatalogueObject)
        )


def test_catalogue_registration_is_idempotent_with_null_revision(repo):
    data = root(repo)
    first = repo.register_catalogue(data.id, "main", finder="finder")
    second = repo.register_catalogue(data.id, "main", finder="finder")
    assert first.id == second.id
    assert first.revision is None and first.loader_id == "default"
    for found in (first, second, repo.get_catalogue(first.id), *repo.catalogues(data.id)):
        assert sa.inspect(found).detached
        assert found.root_input.id == data.id and found.root_input.parent is None
        assert found.root_input.timestep.simulation.project_id == "project"
    assert [row.id for row in repo.catalogues(data.id)] == [first.id]
    assert repo.catalogues(99999) == []
    with pytest.raises(KeyError):
        repo.get_catalogue(99999)


@pytest.mark.parametrize("field", ["project_id", "simulation_id", "timestep_id", "revision", "loader_id"])
def test_root_identity_namespaces_catalogues(repo, field):
    first = catalogue(repo)
    other = repo.register_catalogue(root(repo, **{field: "other"}).id, "main", finder="finder")
    assert first.id != other.id and first.identity_key != other.identity_key


@pytest.mark.parametrize("field", ["catalogue_id", "finder", "revision", "loader_id"])
def test_all_catalogue_labels_namespace_identity(repo, field):
    first = catalogue(repo)
    kwargs = {"catalogue_id": "main", "finder": "finder", "revision": None, "loader_id": "default"}
    kwargs[field] = "other"
    second = repo.register_catalogue(first.root_input_id, **kwargs)
    assert second.id != first.id and second.identity_key != first.identity_key
    assert repo.register_catalogue(first.root_input_id, **kwargs).id == second.id
    assert [row.id for row in repo.catalogues(first.root_input_id)] == [first.id, second.id]


def test_null_revision_is_distinct_from_literal_none(repo):
    assert catalogue(repo).id != catalogue(repo, revision="None").id


@pytest.mark.parametrize("field", ["catalogue_id", "finder", "revision", "loader_id"])
@pytest.mark.parametrize("bad", ["", "  ", 1, False])
def test_invalid_catalogue_labels_fail_without_writes(repo, field, bad):
    kwargs = {"catalogue_id": "main", "finder": "finder", "revision": None, "loader_id": "default"}
    kwargs[field] = bad
    with pytest.raises(ValueError):
        repo.register_catalogue(root(repo).id, **kwargs)
    assert counts(repo) == (0, 0)


def test_typed_keys_and_host_hierarchy(repo):
    cat = catalogue(repo)
    keys = (1, "1", True, 1.0)
    objects = [repo.register_catalogue_object(cat.id, key) for key in keys]
    assert len({row.id for row in objects}) == 4
    assert len({row.typed_object_key for row in objects}) == 4
    assert [repo.register_catalogue_object(cat.id, key).id for key in keys] == [row.id for row in objects]
    child = repo.register_catalogue_object(cat.id, 1, kind="subhalo", host_id=objects[0].id)
    assert child.id != objects[0].id
    assert repo.register_catalogue_object(cat.id, 1, kind="subhalo", host_id=objects[0].id).id == child.id
    for found in (child, repo.get_catalogue_object(child.id), *repo.catalogue_objects(cat.id, kind="subhalo")):
        assert sa.inspect(found).detached
        assert found.catalogue.id == cat.id and found.host.id == objects[0].id
        assert found.host.kind == "halo"
        assert found.catalogue.root_input.timestep.simulation.project_id == "project"
    assert [row.id for row in repo.catalogue_objects(cat.id, host_id=objects[0].id)] == [child.id]
    assert [row.id for row in repo.catalogue_objects(cat.id)] == [row.id for row in objects] + [child.id]
    assert repo.catalogue_objects(99999) == []
    with pytest.raises(KeyError):
        repo.get_catalogue_object(99999)


def test_catalogue_namespaces_and_unknown_host_are_explicit(repo):
    first, second = catalogue(repo), catalogue(repo, loader_id="other")
    a = repo.register_catalogue_object(first.id, 7)
    b = repo.register_catalogue_object(second.id, 7)
    assert a.id != b.id
    child = repo.register_catalogue_object(first.id, 7, kind="subhalo")
    assert child.host is None and repo.get_catalogue_object(child.id).host is None
    assert len(repo.catalogue_objects(first.id, host_id=None)) == 2
    before = counts(repo)
    with pytest.raises(ValueError, match="conflict"):
        repo.register_catalogue_object(first.id, 7, kind="subhalo", host_id=a.id)
    assert counts(repo) == before and repo.get_catalogue_object(child.id).host is None


def test_signed_float_zero_and_empty_string_object_keys(repo):
    cat = catalogue(repo)
    assert repo.register_catalogue_object(cat.id, -0.0).id == repo.register_catalogue_object(cat.id, 0.0).id
    assert repo.register_catalogue_object(cat.id, "").id != repo.register_catalogue_object(cat.id, 0).id


@pytest.mark.parametrize(
    "key", [None, (), [], b"1", np.int64(1), np.float64(1.0), object(), float("nan"), float("inf"), -float("inf")]
)
def test_invalid_object_keys_fail_without_writes(repo, key):
    cat = catalogue(repo)
    with pytest.raises((TypeError, ValueError)):
        repo.register_catalogue_object(cat.id, key)
    assert counts(repo) == (1, 0)


@pytest.mark.parametrize(
    "problem", ["missing", "subset", "halo", "hierarchy", "kind", "identity_key", "json", "parent"]
)
def test_catalogue_roots_require_validated_snapshot_metadata(repo, problem):
    data = root(repo)
    input_id = data.id
    if problem == "missing":
        input_id = 99999
    elif problem in ("subset", "halo"):
        suffix = ("subset", "a" * 16) if problem == "subset" else ("halo", 1, "a" * 16)
        input_id = repo.register_input(data.signature + suffix).id
    else:
        with repo.session() as session, session.begin():
            row = session.get(orm.DataObject, data.id)
            if problem == "hierarchy":
                row.timestep.simulation.logical_id = "different"
            elif problem == "parent":
                row.parent_id = data.id
            elif problem == "json":
                row.identity_json = "invalid"
            else:
                setattr(row, problem, "corrupt")
    with pytest.raises((KeyError, ValueError)):
        repo.register_catalogue(input_id, "main", finder="finder")
    assert counts(repo) == (0, 0)


@pytest.mark.parametrize(
    "problem", ["catalogue", "host", "catalogue_host", "other_root", "subhalo_host", "halo_with_host", "kind"]
)
def test_invalid_hosts_fail_atomically_and_repository_remains_usable(repo, problem):
    cat = catalogue(repo)
    host = repo.register_catalogue_object(cat.id, 1)
    kwargs = {"catalogue_pk": cat.id, "object_key": 2, "kind": "subhalo", "host_id": host.id}
    if problem == "catalogue":
        kwargs["catalogue_pk"] = 99999
    elif problem == "host":
        kwargs["host_id"] = 99999
    elif problem in ("catalogue_host", "other_root"):
        data = root(repo, timestep_id="100") if problem == "other_root" else root(repo)
        other = repo.register_catalogue(data.id, "other", finder="finder")
        kwargs["host_id"] = repo.register_catalogue_object(other.id, 1).id
    elif problem == "subhalo_host":
        kwargs["host_id"] = repo.register_catalogue_object(cat.id, 3, kind="subhalo").id
    elif problem == "halo_with_host":
        kwargs["kind"] = "halo"
    else:
        kwargs["kind"] = "unknown"
    before = counts(repo)
    with pytest.raises((KeyError, ValueError)):
        repo.register_catalogue_object(**kwargs)
    assert counts(repo) == before
    assert repo.register_catalogue_object(cat.id, 2, kind="subhalo", host_id=host.id).host_id == host.id


def test_conflicting_host_registration_never_reparents(repo):
    cat = catalogue(repo)
    hosts = [repo.register_catalogue_object(cat.id, key) for key in (1, 2)]
    child = repo.register_catalogue_object(cat.id, 3, kind="subhalo", host_id=hosts[0].id)
    for host_id in (None, hosts[1].id):
        with pytest.raises(ValueError, match="conflict"):
            repo.register_catalogue_object(cat.id, 3, kind="subhalo", host_id=host_id)
    assert counts(repo) == (1, 3) and repo.get_catalogue_object(child.id).host_id == hosts[0].id
    with pytest.raises(ValueError):
        repo.catalogue_objects(cat.id, kind="bad")


@pytest.mark.parametrize("table", ["catalogues", "catalogue_objects"])
def test_metadata_insert_failure_rolls_back_and_retry_succeeds(repo, table):
    data = root(repo)
    cat = catalogue(repo) if table == "catalogue_objects" else None
    before = counts(repo)

    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith(f"INSERT INTO pynbodyext_orm_{table}"):
            raise RuntimeError("injected metadata failure")

    sa.event.listen(repo.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            if cat is None:
                repo.register_catalogue(data.id, "main", finder="finder")
            else:
                repo.register_catalogue_object(cat.id, 1)
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", fail)
    assert counts(repo) == before
    if cat is None:
        assert repo.register_catalogue(data.id, "main", finder="finder").root_input_id == data.id
    else:
        assert repo.register_catalogue_object(cat.id, 1).catalogue_fk == cat.id


def test_sqlite_reopen_joins_and_metadata_only_boundary(tmp_path):
    calls = []

    def loader(uri):
        calls.append(uri)
        pytest.fail("catalogue metadata must not load snapshots")

    store = InMemoryResultStore()
    url = f"sqlite:///{tmp_path / 'reopen.sqlite'}"
    original = orm.AnalysisRepository(url, stores={"results": store}, loaders={"default": loader})
    try:
        cat = catalogue(original)
        host = original.register_catalogue_object(cat.id, 1)
        child = original.register_catalogue_object(cat.id, "1", kind="subhalo", host_id=host.id)
        assert calls == [] and original.runs() == [] and store.list_refs() == []
    finally:
        original.engine.dispose()
    reopened = orm.AnalysisRepository(url)
    try:
        assert catalogue(reopened).id == cat.id
        assert reopened.get_catalogue_object(child.id).host.id == host.id
        assert [row.id for row in reopened.catalogue_objects(cat.id)] == [host.id, child.id]
        with reopened.session() as session:
            host_alias = sa.orm.aliased(orm.CatalogueObject)
            found = session.scalars(
                sa.select(orm.CatalogueObject)
                .join(orm.CatalogueObject.catalogue)
                .join(orm.Catalogue.root_input)
                .join(orm.DataObject.timestep)
                .join(orm.Timestep.simulation)
                .join(host_alias, orm.CatalogueObject.host)
                .where(orm.Simulation.project_id == "project", host_alias.kind == "halo")
            ).one()
            assert found.id == child.id
            assert found.host.subhalos[0].id == child.id
            assert found.catalogue.objects[0].id == host.id
            assert session.scalar(sa.select(sa.func.count()).select_from(orm.Recipe)) == 0
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize("problem", ["catalogue_key", "catalogue_label", "host_kind", "host_catalogue"])
def test_existing_metadata_conflicts_are_not_silently_repaired(repo, problem):
    cat = catalogue(repo)
    host = repo.register_catalogue_object(cat.id, 1)
    child = repo.register_catalogue_object(cat.id, 2, kind="subhalo", host_id=host.id)
    other = catalogue(repo, revision="other")
    with repo.session() as session, session.begin():
        if problem == "catalogue_key":
            session.get(orm.Catalogue, cat.id).identity_key = "corrupt"
        elif problem == "catalogue_label":
            session.get(orm.Catalogue, cat.id).finder = "different"
        elif problem == "host_kind":
            session.get(orm.CatalogueObject, host.id).kind = "subhalo"
        else:
            session.get(orm.CatalogueObject, host.id).catalogue_fk = other.id
    with pytest.raises(ValueError):
        repo.get_catalogue_object(child.id)
    with pytest.raises(ValueError):
        repo.register_catalogue_object(cat.id, 2, kind="subhalo", host_id=host.id)


@pytest.mark.parametrize(
    "problem", ["catalogue_duplicate", "object_duplicate", "catalogue_fk", "host_fk", "kind", "halo_host"]
)
def test_database_uniqueness_foreign_keys_and_checks(repo, problem):
    cat = catalogue(repo)
    host = repo.register_catalogue_object(cat.id, 1)
    with pytest.raises(sa.exc.IntegrityError):
        with repo.session() as session, session.begin():
            if problem == "catalogue_duplicate":
                session.add(
                    orm.Catalogue(
                        root_input_id=cat.root_input_id,
                        catalogue_id=cat.catalogue_id,
                        finder=cat.finder,
                        revision=None,
                        loader_id=cat.loader_id,
                        identity_key=cat.identity_key,
                    )
                )
            else:
                kwargs = {"catalogue_fk": cat.id, "kind": "halo", "typed_object_key": host.typed_object_key}
                if problem == "catalogue_fk":
                    kwargs["catalogue_fk"] = 99999
                elif problem == "host_fk":
                    kwargs.update(kind="subhalo", host_id=99999)
                elif problem == "kind":
                    kwargs["kind"] = "bad"
                elif problem == "halo_host":
                    kwargs.update(typed_object_key="other", host_id=host.id)
                session.add(orm.CatalogueObject(**kwargs))
    assert counts(repo) == (1, 1)
