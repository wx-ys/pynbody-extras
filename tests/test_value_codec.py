"""Regression contracts for portable scalar values and dictionary keys."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pynbodyext.core.calculate.store import ValueCodec


@pytest.mark.parametrize(
    "value",
    [
        np.int8(-3),
        np.int64(3),
        np.uint64(2**64 - 1),
        np.float32(1.5),
        np.float64(1.5),
        np.bool_(False),
        np.str_("stars"),
        np.bytes_(b"stars"),
        np.str_("stars\0"),
        np.bytes_(b"stars\0"),
        np.str_("\0"),
        np.bytes_(b"\0"),
        np.float32(np.nan),
        np.float64(np.inf),
        np.float64(-np.inf),
    ],
)
def test_numpy_scalar_survives_json_with_its_dtype(value):
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))

    assert type(restored) is type(value)
    assert restored.dtype == value.dtype
    np.testing.assert_equal(restored, value)
    if value.dtype.kind in ("U", "S"):
        assert restored.tobytes() == value.tobytes()


@pytest.mark.parametrize(
    "key",
    [None, (), (1,), (1, 2), ("1", "2"), ("a,b", "quote'", "paren)"), ((1, "star"), None, False, 2.5), (b"star",)],
)
def test_dictionary_key_survives_json_without_changing_identity(key):
    codec = ValueCodec()
    value = {key: "selected"}
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))

    assert restored == value
    assert type(next(iter(restored))) is type(key)


def test_none_and_tuple_keys_do_not_collapse_into_string_keys():
    codec = ValueCodec()
    value = {None: 1, "None": 2, (1, 2): 3, ("1", "2"): 4}
    assert codec.decode(codec.encode(value)) == value


@pytest.mark.parametrize(
    ("encoded_key", "expected"),
    [
        ({"kind": "NoneType", "text": "None"}, None),
        ({"kind": "none", "text": "None"}, None),
        ({"kind": "tuple", "text": "(1, 'star', (None, False))"}, (1, "star", (None, False))),
        ({"kind": "tuple", "text": "('a,b',)"}, ("a,b",)),
    ],
)
def test_existing_dictionary_key_payloads_remain_readable(encoded_key, expected):
    stored = {"__kind": "dict", "items": [[encoded_key, {"__kind": "int", "value": 7}]]}
    assert ValueCodec().decode(stored) == {expected: 7}


@pytest.mark.parametrize("text", ["(unknown_name,)", "tuple([1, 2])", "[1, 2]", "1", "([1],)"])
def test_invalid_tuple_key_is_rejected_instead_of_silently_changed(text):
    stored = {"__kind": "dict", "items": [[{"kind": "tuple", "text": text}, {"__kind": "int", "value": 7}]]}
    with pytest.raises(ValueError, match="tuple key"):
        ValueCodec().decode(stored)


def test_tuple_key_text_cannot_execute_code(tmp_path):
    marker = tmp_path / "should-not-exist"
    text = f"(__import__('pathlib').Path({str(marker)!r}).touch(),)"
    stored = {"__kind": "dict", "items": [[{"kind": "tuple", "text": text}, {"__kind": "none"}]]}
    with pytest.raises(ValueError, match="tuple key"):
        ValueCodec().decode(stored)
    assert not marker.exists()


def test_supported_key_encoding_and_content_hash_remain_unchanged():
    # Pin the existing wire representation: this fix does not migrate records.
    codec = ValueCodec()
    value = {(1, "star"): 2, None: 3}
    encoded = {
        "__kind": "dict",
        "items": [
            ({"kind": "NoneType", "text": "None"}, {"__kind": "int", "value": 3}),
            ({"kind": "tuple", "text": "(1, 'star')"}, {"__kind": "int", "value": 2}),
        ],
    }
    assert codec.encode(value) == encoded
    assert codec.content_hash(value) == codec.content_hash(dict(reversed(list(value.items()))))


def _object_array(shape):
    value = np.empty(shape, dtype=object)
    for index in range(value.size):
        value.flat[index] = [index, index + 1]
    return value


@pytest.mark.parametrize("shape", [(2,), (), (0, 2), (2, 0, 3), (2, 3)])
def test_object_array_preserves_shape_and_element_boundaries(shape):
    value = _object_array(shape)
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.tolist() == value.tolist()


def test_object_array_nested_values_noncontiguous_view_and_units():
    from pynbody.array import SimArray

    value = np.empty((2, 3), dtype=object)
    for index in range(value.size):
        value.flat[index] = {"entry": (index, [index + 1, index + 2])}
    value = value[:, ::2]
    assert not value.flags.c_contiguous
    codec = ValueCodec()
    for original in (value, SimArray(value, units="kpc")):
        restored = codec.decode(json.loads(json.dumps(codec.encode(original))))
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert restored.tolist() == original.tolist()
        if isinstance(original, SimArray):
            assert isinstance(restored, SimArray)
            assert str(restored.units) == str(original.units)


@pytest.mark.parametrize(
    ("shape", "items"),
    [([-1], []), ([True], []), ([1.0], []), (["1"], []), ("1", []),
     ([2], []), ([], []), ([0, 2], [{"__kind": "none"}]), ([1], {})],
)
def test_invalid_object_array_shape_or_element_count_is_rejected(shape, items):
    with pytest.raises(ValueError, match="object-array shape or element count"):
        ValueCodec().decode({"__kind": "array_objects_v2", "shape": shape, "items": items})


def test_legacy_object_array_tag_remains_readable():
    restored = ValueCodec().decode({"__kind": "array_objects", "items": [{"__kind": "int", "value": 3}]})
    assert restored.shape == (1,)
    assert restored.dtype == object
    assert restored.tolist() == [3]


STRUCTURED_DTYPES = [
    np.dtype([("id", ">i4"), ("mass", "<f8")]),
    np.dtype([("nested", [("id", "i2"), ("mass", "f8")])]),
    np.dtype([("velocity", ">f8", (2, 3)), ("id", "i4")]),
    np.dtype([("id", "i1"), ("mass", "f8")], align=True),
    np.dtype({"names": ["id", "mass"], "formats": ["i4", "f8"], "offsets": [0, 16], "itemsize": 32}),
]


@pytest.mark.parametrize("dtype", STRUCTURED_DTYPES)
@pytest.mark.parametrize("shape", [(2,), (), (0, 2)])
def test_structured_array_dtype_shape_and_values_survive_json(dtype, shape):
    value = np.ones(shape, dtype=dtype)
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.dtype.itemsize == value.dtype.itemsize
    np.testing.assert_array_equal(restored, value)


def test_noncontiguous_structured_simarray_preserves_units():
    from pynbody.array import SimArray

    value = SimArray(np.ones((3, 4), dtype=STRUCTURED_DTYPES[0]), units="kpc")[:, ::2]
    assert not value.flags.c_contiguous
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert isinstance(restored, SimArray)
    assert restored.dtype == value.dtype
    assert restored.shape == value.shape
    assert str(restored.units) == str(value.units)
    np.testing.assert_array_equal(restored, value)


@pytest.mark.parametrize("dtype", [np.dtype([("object", object)]), np.dtype([("nested", [("object", object)])])])
def test_structured_object_fields_are_rejected_before_encoding(dtype):
    with pytest.raises(TypeError, match="object fields"):
        ValueCodec().encode(np.zeros(2, dtype=dtype))


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype([("id", "i4")], metadata={"unit": "count"}),
        np.dtype([("id", np.dtype("i4", metadata={"unit": "count"}))]),
        np.dtype([("nested", [("id", np.dtype("i4", metadata={"unit": "count"}))])]),
        np.dtype([("vector", np.dtype("f8", metadata={"unit": "kpc"}), (2,))]),
    ],
)
def test_structured_dtype_metadata_is_rejected_without_silent_loss(dtype):
    with pytest.raises(TypeError, match="metadata"):
        ValueCodec().encode(np.zeros(2, dtype=dtype))


@pytest.mark.parametrize("offsets", [[0, 0], [8, 0]])
def test_unrepresentable_structured_layout_is_rejected_during_encoding(offsets):
    dtype = np.dtype({"names": ["a", "b"], "formats": ["i8", "i8"], "offsets": offsets})
    with pytest.raises(TypeError, match="losslessly"):
        ValueCodec().encode(np.zeros(2, dtype=dtype))


@pytest.mark.parametrize("payload_kind", ["truncated", "object", "numeric", "npz"])
def test_invalid_structured_payload_is_rejected_without_pickle(payload_kind, monkeypatch):
    import base64
    import io
    import pickle

    def fail_pickle(*args, **kwargs):
        pytest.fail("structured-array decoding must not invoke pickle")

    buffer = io.BytesIO()
    if payload_kind == "npz":
        np.savez(buffer, data=np.zeros(1, dtype=STRUCTURED_DTYPES[0]))
    else:
        dtype = [("object", object)] if payload_kind == "object" else STRUCTURED_DTYPES[0]
        value = np.zeros(1) if payload_kind == "numeric" else np.zeros(1, dtype=dtype)
        np.save(buffer, value, allow_pickle=payload_kind == "object")
    raw = buffer.getvalue()
    if payload_kind == "truncated":
        raw = raw[:-3]
    monkeypatch.setattr(pickle, "load", fail_pickle)
    monkeypatch.setattr(pickle, "loads", fail_pickle)
    with pytest.raises((ValueError, EOFError)):
        ValueCodec().decode({"__kind": "structured_array_v1", "data_b64": base64.b64encode(raw).decode("ascii")})


@pytest.mark.parametrize("title", [frozenset({1}), np.int64(2)])
def test_unreadable_structured_field_titles_are_rejected_during_encoding(title):
    dtype = np.dtype([((title, "field"), "i4")])
    with pytest.raises(TypeError, match="losslessly"):
        ValueCodec().encode(np.zeros(1, dtype=dtype))


def test_large_structured_header_roundtrips_with_pickle_disabled():
    value = np.ones(1, dtype=[(f"field_{index}", "f8") for index in range(500)])
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)


def test_literal_field_titles_survive_structured_encoding():
    value = np.ones(1, dtype=[(("mass label", "mass"), "f8"), (((1, "id"), "id"), "i4")])
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)
