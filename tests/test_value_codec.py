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
