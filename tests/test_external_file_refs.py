"""Caller-owned local file descriptors remain explicit across persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import Pipeline, PropertyBase
from pynbodyext.core.calculate import store as store_api


def test_external_file_ref_public_api():
    assert hasattr(store_api, "ExternalFileRef"), "missing ExternalFileRef public API"
    assert hasattr(store_api, "ExternalFileInspection"), "missing ExternalFileInspection public API"


@pytest.fixture
def ref_type():
    assert hasattr(store_api, "ExternalFileRef"), "missing ExternalFileRef public API"
    return store_api.ExternalFileRef


@pytest.mark.parametrize("size", [None, 0, 7])
@pytest.mark.parametrize("digest", [None, "a" * 64])
def test_descriptor_json_without_existing_file(ref_type, tmp_path, size, digest):
    ref = ref_type(str(tmp_path / "absent.bin"), " application/octet-stream ", size, digest)
    restored = ref_type.from_json(ref.to_json())
    assert restored == ref and hash(restored) == hash(ref)
    assert restored.uri == ref.uri and restored.media_type == " application/octet-stream "
    assert restored.size_bytes == size and restored.sha256 == digest
    with pytest.raises(FrozenInstanceError):
        ref.uri = "/tmp/replaced.bin"


@pytest.mark.parametrize("uri", ["/tmp/map.png", "/tmp/a#b?c", "file:///tmp/map%20one.png",
                                  "file:/tmp/map.png", "file://localhost/tmp/map.png", "/tmp/图像%20.png"])
def test_local_path_descriptor_preserves_text(ref_type, uri):
    ref = ref_type(uri, "image/png")
    assert ref_type.from_json(ref.to_json()).uri == uri


@pytest.mark.parametrize("uri", ["map.png", "~/map.png", "file:map.png", "https://host/map.png",
                                  "s3://bucket/map.png", "file://other-host/tmp/map.png", "file:///tmp/map.png?x=1",
                                  "file:///tmp/map.png#part", "file:///tmp/%00map.png", "file:///tmp/%xx",
                                  "file:///tmp/%ff", "file://user@localhost/tmp/map.png", "file://localhost:80/tmp/map.png",
                                  "", " ", None, 1, "/tmp/a\0b", "file://", "file:///tmp/%", "file:///tmp/%2",
                                  " file:///tmp/map.png", "\x01file:///tmp/map.png", "file:///tmp/map\n.png",
                                  "file:///tmp/map\t.png"])
def test_descriptor_rejects_nonlocal_or_invalid_uri(ref_type, uri):
    with pytest.raises(ValueError):
        ref_type(uri, "image/png")


@pytest.mark.parametrize("field,value", [("media_type", ""), ("media_type", " \t"), ("media_type", 1),
                                          ("media_type", None), ("size_bytes", True), ("size_bytes", False),
                                          ("size_bytes", -1), ("size_bytes", 1.0), ("size_bytes", "1"),
                                          ("size_bytes", np.int64(1)), ("sha256", "A" * 64), ("sha256", "a" * 63),
                                          ("sha256", "a" * 65), ("sha256", "g" * 64), ("sha256", ""), ("sha256", 1)])
def test_descriptor_rejects_invalid_fields(ref_type, field, value):
    kwargs = {"uri": "/tmp/map.png", "media_type": "image/png", field: value}
    with pytest.raises(ValueError):
        ref_type(**kwargs)


@pytest.mark.parametrize("change", ["schema", "missing_schema", "extra", "missing_uri", "missing_size",
                                    "nonmapping", "invalid_size", "invalid_uri"])
def test_json_rejects_invalid_descriptor(ref_type, change):
    payload = json.loads(ref_type("/tmp/map.png", "image/png").to_json())
    if change == "schema":
        payload["schema"] = "unknown"
    elif change == "extra":
        payload["extra"] = 1
    elif change.startswith("missing_"):
        del payload[{"missing_schema": "schema", "missing_uri": "uri", "missing_size": "size_bytes"}[change]]
    elif change == "nonmapping":
        payload = []
    elif change == "invalid_size":
        payload["size_bytes"] = True
    else:
        payload["uri"] = "relative.png"
    with pytest.raises(ValueError):
        ref_type.from_json(json.dumps(payload))


def test_json_rejects_duplicates_and_malformed_text(ref_type):
    valid = ref_type("/tmp/map.png", "image/png").to_json()
    duplicate = valid[:-1] + ',"uri":"/tmp/replaced.png"}'
    for text in (duplicate, "{", "null", '"text"'):
        with pytest.raises(ValueError):
            ref_type.from_json(text)


def test_inspect_missing_directory_and_broken_symlink(ref_type, tmp_path):
    absent = ref_type(str(tmp_path / "absent"), "application/octet-stream")
    assert absent.inspect() == store_api.ExternalFileInspection("missing", "not_found")
    directory = ref_type(str(tmp_path), "application/octet-stream")
    assert directory.inspect() == store_api.ExternalFileInspection("changed", "not_regular_file")
    link = tmp_path / "broken"
    link.symlink_to(tmp_path / "absent")
    assert ref_type(str(link), "application/octet-stream").inspect().status == "missing"


@pytest.mark.parametrize("location", ["path", "file_uri", "localhost_uri"])
def test_inspect_checksum_is_explicit_and_equal_size_change_is_detected(ref_type, tmp_path, location):
    path = tmp_path / "图像 one.bin"
    path.write_bytes(b"first")
    uri = {"path": str(path), "file_uri": path.as_uri(),
           "localhost_uri": path.as_uri().replace("file:///", "file://localhost/")}[location]
    ref = ref_type(uri, "application/octet-stream", 5, hashlib.sha256(b"first").hexdigest())
    assert ref.inspect() == store_api.ExternalFileInspection("available", "checksum_not_checked")
    assert ref.inspect(verify_checksum=True) == store_api.ExternalFileInspection("available", "checksum_match", True)
    path.write_bytes(b"other")
    assert ref.inspect() == store_api.ExternalFileInspection("available", "checksum_not_checked")
    assert ref.inspect(True) == store_api.ExternalFileInspection("changed", "checksum_mismatch")
    path.write_bytes(b"longer")
    assert ref.inspect(True) == store_api.ExternalFileInspection("changed", "size_mismatch")


def test_inspect_never_reads_without_requested_supplied_checksum(ref_type, tmp_path, monkeypatch):
    path = tmp_path / "map.bin"
    path.write_bytes(b"first")
    with_digest = ref_type(str(path), "application/octet-stream", sha256=hashlib.sha256(b"first").hexdigest())
    no_digest = ref_type(str(path), "application/octet-stream")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("unexpected checksum read"))
        assert with_digest.inspect().reason == "checksum_not_checked"
        assert no_digest.inspect(True) == store_api.ExternalFileInspection("available", "checksum_not_supplied")


@pytest.mark.parametrize("operation", ["stat", "open"])
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_inspect_access_errors_propagate(ref_type, tmp_path, monkeypatch, operation, error_type):
    path = tmp_path / "map.bin"
    path.write_bytes(b"first")
    ref = ref_type(str(path), "application/octet-stream", sha256=hashlib.sha256(b"first").hexdigest())

    def denied(*args, **kwargs):
        raise error_type("injected access error")

    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, denied)
        with pytest.raises(error_type, match="injected access error"):
            ref.inspect(verify_checksum=True)


def test_inspect_disappearance_during_checksum_is_missing(ref_type, tmp_path, monkeypatch):
    path = tmp_path / "map.bin"
    path.write_bytes(b"first")
    ref = ref_type(str(path), "application/octet-stream", sha256=hashlib.sha256(b"first").hexdigest())

    def disappeared(*args, **kwargs):
        raise FileNotFoundError("disappeared after stat")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", disappeared)
        assert ref.inspect(True) == store_api.ExternalFileInspection("missing", "not_found")


@pytest.mark.parametrize("flag", [1, "true", None])
def test_inspect_rejects_nonboolean_flag(ref_type, tmp_path, flag):
    with pytest.raises(ValueError, match="verify_checksum"):
        ref_type(str(tmp_path / "absent"), "application/octet-stream").inspect(flag)


def test_codec_uses_explicit_versioned_descriptor(ref_type, tmp_path, monkeypatch):
    import pickle

    ref = ref_type(str(tmp_path / "absent.bin"), "application/octet-stream")
    codec = store_api.ValueCodec()
    expected = {"__kind": "external_file_ref_v1", "descriptor": json.loads(ref.to_json())}
    assert codec.encode(ref) == expected
    with monkeypatch.context() as patch:
        patch.setattr(pickle, "dumps", lambda *args, **kwargs: pytest.fail("descriptor must not pickle"))
        patch.setattr(pickle, "loads", lambda *args, **kwargs: pytest.fail("descriptor must not unpickle"))
        assert codec.decode(json.loads(json.dumps(codec.encode(ref)))) == ref
    assert codec.encode(ref.uri) == {"__kind": "str", "value": ref.uri}


def test_codec_nested_descriptors_and_shaped_object_array(ref_type, tmp_path):
    ref = ref_type(str(tmp_path / "absent.bin"), "application/octet-stream", 0, "a" * 64)
    value = {"files": [ref, (ref, {"inner": ref})]}
    array = np.empty((2, 2), dtype=object)
    for index in range(array.size):
        array.flat[index] = value
    codec = store_api.ValueCodec()
    assert codec.decode(json.loads(json.dumps(codec.encode(value)))) == value
    restored = codec.decode(json.loads(json.dumps(codec.encode(array))))
    assert restored.shape == array.shape
    assert restored.tolist() == array.tolist()
    assert codec.encode(value)["items"][0][1]["items"][0]["__kind"] == "external_file_ref_v1"


@pytest.mark.parametrize("change", ["extra_envelope", "absent_descriptor", "extra_descriptor", "schema", "missing_size",
                                    "boolean_size", "relative_uri", "nonmapping", "new_tag"])
def test_codec_rejects_invalid_descriptor_payload(ref_type, change):
    data = {"__kind": "external_file_ref_v1", "descriptor": json.loads(ref_type("/tmp/map", "image/png").to_json())}
    if change == "extra_envelope":
        data["extra"] = 1
    elif change == "absent_descriptor":
        del data["descriptor"]
    elif change == "extra_descriptor":
        data["descriptor"]["extra"] = 1
    elif change == "schema":
        data["descriptor"]["schema"] = "unknown"
    elif change == "missing_size":
        del data["descriptor"]["size_bytes"]
    elif change == "boolean_size":
        data["descriptor"]["size_bytes"] = True
    elif change == "relative_uri":
        data["descriptor"]["uri"] = "map.png"
    elif change == "nonmapping":
        data["descriptor"] = []
    else:
        data["__kind"] = "external_file_ref_v2"
    with pytest.raises(ValueError):
        store_api.ValueCodec().decode(data)


def test_codec_json_construction_and_hash_do_no_filesystem_io(ref_type, tmp_path, monkeypatch):
    import builtins

    uri = str(tmp_path / "absent.bin")
    codec = store_api.ValueCodec()

    def forbidden(*args, **kwargs):
        pytest.fail("descriptor operation performed filesystem IO or inspection")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(ref_type, "inspect", forbidden)
        ref = ref_type(uri, "application/octet-stream")
        restored = ref_type.from_json(ref.to_json())
        assert restored == ref
        assert codec.decode(codec.encode(ref)) == ref
        assert codec.kind(ref) == "external_file_ref_v1"
        digest = codec.content_hash(ref)
        assert digest == codec.content_hash(restored)
        assert codec.size_bytes(ref) > 0


@PropertyBase.dataclass
class FileDescriptor(PropertyBase):
    uri: str
    observe: bool = False

    def calculate(self, sim, params=None):
        ref_type = store_api.ExternalFileRef
        if self.observe:
            data = Path(self.uri).read_bytes()
            descriptor = ref_type(self.uri, "application/octet-stream", len(data), hashlib.sha256(data).hexdigest())
        else:
            descriptor = ref_type(self.uri, "application/octet-stream")
        return {"files": [descriptor, (descriptor,)]}


IDENTITY = ("sim", "external-files")


@pytest.fixture(params=["memory", "sqlite"])
def result_store(request, tmp_path):
    if request.param == "memory":
        yield store_api.InMemoryResultStore()
    else:
        pytest.importorskip("sqlalchemy")
        adapter = store_api.SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'external.sqlite'}")
        try:
            yield adapter
        finally:
            adapter.engine.dispose()


def test_store_nested_absent_descriptor_in_value_named_and_record_json(ref_type, tmp_path, result_store):
    uri = str(tmp_path / "absent.bin")
    calc = Pipeline({"files": FileDescriptor(uri).keep("files")})
    result = calc.run(pynbody.new(1), store=result_store, sim_identity=lambda sim: IDENTITY)
    fetched = result_store.fetch(result.store_ref)
    expected = {"files": [ref_type(uri, "application/octet-stream"), (ref_type(uri, "application/octet-stream"),)]}
    assert dict(fetched.value)["files"] == expected
    assert fetched.named["files"].record.value == expected
    assert fetched.named["files"].record.value["files"][0].inspect().status == "missing"
    record = store_api.ResultRecord.from_result(result, sim_signature=IDENTITY,
                                               calculator_signature=calc.to_signature(), codec=result_store.codec)
    restored_record = store_api.ResultRecord.from_json(record.to_json())
    assert result_store.codec.decode(restored_record.named["files"]) == expected
    assert restored_record.named["files"]["items"][0][1]["items"][0]["__kind"] == "external_file_ref_v1"


def test_sqlite_reopen_preserves_descriptor_and_named_output(ref_type, tmp_path):
    pytest.importorskip("sqlalchemy")
    url = f"sqlite:///{tmp_path / 'reopen.sqlite'}"
    uri = str(tmp_path / "absent.bin")
    calc = Pipeline({"files": FileDescriptor(uri).keep("files")})
    adapter = store_api.SQLAlchemyResultStore(url)
    try:
        result = calc.run(pynbody.new(1), store=adapter, sim_identity=lambda sim: IDENTITY)
        receipt = result.store_ref.to_json()
    finally:
        adapter.engine.dispose()
    reopened = store_api.SQLAlchemyResultStore(url)
    try:
        fetched = reopened.fetch(store_api.RecordRef.from_json(receipt))
        descriptor = fetched.named["files"].record.value["files"][0]
        assert type(descriptor) is ref_type
        assert descriptor == ref_type(uri, "application/octet-stream")
        assert dict(fetched.value)["files"] == fetched.named["files"].record.value
        assert descriptor.inspect().status == "missing"
    finally:
        reopened.engine.dispose()


def test_store_refresh_preserves_descriptor_but_cannot_restore_overwritten_file(ref_type, tmp_path, result_store):
    path = tmp_path / "map.bin"
    path.write_bytes(b"first")
    calc = FileDescriptor(str(path), observe=True)
    sim = pynbody.new(1)
    opts = {"store": result_store, "sim_identity": lambda sim: IDENTITY}
    first = calc.run(sim, **opts)
    old = first.value["files"][0]
    path.write_bytes(b"other")
    second = calc.run(sim, reuse=False, **opts)
    assert first.store_ref != second.store_ref
    assert first.store_ref.cache_key == second.store_ref.cache_key
    assert result_store.list_versions(first.store_ref) == [first.store_ref, second.store_ref]
    saved_receipt = first.store_ref.to_json()
    adapter = result_store
    if isinstance(result_store, store_api.SQLAlchemyResultStore):
        url = str(result_store.engine.url)
        result_store.engine.dispose()
        adapter = store_api.SQLAlchemyResultStore(url)
    before = set(tmp_path.iterdir())
    try:
        restored = adapter.fetch(store_api.RecordRef.from_json(saved_receipt)).value["files"][0]
        assert restored == old
        assert restored.sha256 == hashlib.sha256(b"first").hexdigest()
        assert adapter.fetch(second.store_ref).value["files"][0].sha256 == hashlib.sha256(b"other").hexdigest()
        assert restored.inspect(True) == store_api.ExternalFileInspection("changed", "checksum_mismatch")
        assert path.read_bytes() == b"other"
        assert set(tmp_path.iterdir()) == before
    finally:
        if adapter is not result_store:
            adapter.engine.dispose()
