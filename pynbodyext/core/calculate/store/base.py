"""Result-store interface, record model, and in-memory adapter.

This is the persistence seam for the future database-backed store.  A
:class:`ResultStore` persists a calculator :class:`Result` under a composite
identity: the simulation signature (which *which* snapshot/object the value
belongs to) plus the content-addressed calculator signature (which *what* was
computed).

The signature system is the source of the identity:

- ``sim_signature`` — a pluggable address tuple (see
  :mod:`~pynbodyext.core.calculate.runtime.sim_identity`), e.g. ``("sim",
  "/path/snap_103", "halo_0")``.
- ``calculator_signature`` — a :class:`CalculatorSignature` whose short hash
  identifies the calculator independently of its textual representation, and
  whose JSON text allows reconstruction.

:class:`ResultStore` is deliberately an abstract template: it implements the
high-level :meth:`ResultStore.store`, :meth:`ResultStore.fetch`,
:meth:`ResultStore.get`, :meth:`ResultStore.has`, and :meth:`ResultStore.delete`
in terms of four small abstract primitives (``_put``, ``_get_by_key``,
``_delete``, ``_list``).  :class:`InMemoryResultStore` is the reference adapter;
a SQLAlchemy adapter in the same package implements the same primitives for a
relational database.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.result.signature import CalculatorSignature, calculator_from_signature

from .codec import ValueCodec

if TYPE_CHECKING:
    from pynbodyext.core.calculate.result.result import Result, ValueSummary

__all__ = [
    "RESULT_STORE_SCHEMA",
    "RecordRef",
    "ResultRecord",
    "ResultStore",
    "InMemoryResultStore",
    "compute_calculator_key",
]

#: Discriminator for records produced by this store version.
RESULT_STORE_SCHEMA = "pynbodyext.result.store/v1"


def _sig_text(sig: tuple[Any, ...]) -> str:
    """Serialize a signature tuple to a stable string (for composite keys)."""
    return json.dumps([str(s) for s in sig], sort_keys=True, separators=(",", ":"))


def _record_key(sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> str:
    return f"{_sig_text(sim_signature)}::{calculator_signature_hash}"


def _tag_tuples(value: Any) -> Any:
    """Replace every ``tuple`` with a tagged dict so JSON preserves it.

    The stock JSON encoder serialises tuples as arrays *before* its ``default``
    hook runs, so ``default`` cannot rescue them.  Pre-walking the record and
    rewriting each tuple as ``{"__type": "tuple", "items": [...]}`` lets
    ``_json_object_hook`` restore them exactly.  Without this the nested codec
    payloads (which pair keys with values as tuples) and the provenance
    ``cache_key()`` structure silently degrade tuples to lists on round-trip.
    """
    if isinstance(value, tuple):
        return {"__type": "tuple", "items": [_tag_tuples(v) for v in value]}
    if isinstance(value, list):
        return [_tag_tuples(v) for v in value]
    if isinstance(value, dict):
        return {k: _tag_tuples(v) for k, v in value.items()}
    return value


def _json_object_hook(data: dict[str, Any]) -> Any:
    """``json.loads`` hook that restores tuples tagged by :func:`_tag_tuples`."""
    if data.get("__type") == "tuple":
        return tuple(data["items"])
    return data


def compute_calculator_key(signature: CalculatorSignature) -> tuple[str, str]:
    """Return ``(short_hash, json_text)`` for a structured signature.

    The short hash is the content-addressed identity used as the lookup key; the
    full JSON is retained so a loaded store entry can reconstruct the calculator.
    """
    return (signature.short_hash(), signature.to_json())


@dataclass(frozen=True, slots=True)
class RecordRef:
    """Opaque handle to a stored result, returned by :meth:`ResultStore.store`.

    Carries the composite identity so an adapter can resolve it; a
    database-backed adapter may additionally hold a primary key in
    :attr:`id`.
    """

    sim_signature: tuple[Any, ...]
    calculator_signature_hash: str
    id: Any = None

    @property
    def key(self) -> str:
        return _record_key(self.sim_signature, self.calculator_signature_hash)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, RecordRef) and other.key == self.key


@dataclass(slots=True)
class ResultRecord:
    """Self-describing, codec-encoded row for one stored result."""

    schema: str = RESULT_STORE_SCHEMA
    sim_signature: tuple[Any, ...] = ()
    calculator_signature_hash: str = ""
    calculator_signature_text: str = ""
    pretty_key: str | None = None
    value: dict[str, Any] = field(default_factory=dict)
    named: dict[str, dict[str, Any]] | None = None
    provenance: dict[str, Any] | None = None
    created_at: float = 0.0

    @classmethod
    def from_result(
        cls,
        result: Result,
        *,
        sim_signature: tuple[Any, ...],
        calculator_signature: CalculatorSignature,
        codec: ValueCodec,
        pretty_key: str | None = None,
        created_at: float | None = None,
    ) -> ResultRecord:
        """Encode a live :class:`Result` into a storeable record."""
        hash_, text = compute_calculator_key(calculator_signature)
        if pretty_key is None:
            pretty_key = calculator_signature.pretty()

        provenance: dict[str, Any] | None = None
        if result.provenance is not None:
            provenance = {
                "calculator_signature": list(result.provenance.calculator_signature),
                "sim_signature": list(result.provenance.sim_signature),
                "started_at": float(result.provenance.started_at),
                "finished_at": result.provenance.finished_at,
            }

        named: dict[str, dict[str, Any]] = {}
        for name, node in result.named.items():
            if node.stored_value and node.value is not None:
                named[name] = codec.encode(node.value)

        return cls(
            schema=RESULT_STORE_SCHEMA,
            sim_signature=tuple(sim_signature),
            calculator_signature_hash=hash_,
            calculator_signature_text=text,
            pretty_key=pretty_key,
            value=codec.encode(result.value),
            named=named or None,
            provenance=provenance,
            created_at=created_at if created_at is not None else time.time(),
        )

    def to_result(self, codec: ValueCodec) -> Result:
        """Reconstruct a :class:`Result` from a stored record."""
        from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeStatus
        from pynbodyext.core.calculate.result.result import PerfSummary, ProvenanceInfo, Result, ResultNode

        value = codec.decode(self.value)

        named: dict[str, ResultNode] = {}
        named_values: dict[str, Any] = {}
        for name, encoded in (self.named or {}).items():
            decoded = codec.decode(encoded)
            named_values[name] = decoded
            named[name] = ResultNode(
                node_id=f"stored:{name}",
                kind=BuiltinKinds.CALCULATOR,
                signature=(self.calculator_signature_hash,),
                status=NodeStatus.OK,
                name=name,
                display_name=name,
                value=decoded,
                stored_value=True,
                value_summary=_value_summary(decoded),
            )

        root = ResultNode(
            node_id="stored:1",
            kind=BuiltinKinds.CALCULATOR,
            signature=(self.calculator_signature_hash,),
            status=NodeStatus.OK,
            name=self.pretty_key,
            display_name=self.pretty_key,
            value=value,
            stored_value=True,
            value_summary=_value_summary(value),
        )

        nodes: dict[str, ResultNode] = {"stored:1": root}
        nodes.update(named)

        provenance: ProvenanceInfo | None = None
        if self.provenance is not None:
            provenance = ProvenanceInfo(
                calculator_signature=tuple(self.provenance.get("calculator_signature", ()) or ()),
                sim_signature=tuple(self.provenance.get("sim_signature", self.sim_signature) or self.sim_signature),
                started_at=float(self.provenance.get("started_at", 0.0)),
                finished_at=self.provenance.get("finished_at"),
                calculator_signature_text=self.calculator_signature_text,
                calculator_signature_hash=self.calculator_signature_hash,
            )

        calculator = None
        if self.calculator_signature_text:
            try:
                calculator = calculator_from_signature(self.calculator_signature_text)
            except Exception:
                # Not all signature texts are reconstructible (e.g. non-constructible
                # or external-dependency calculators); degrade to a missing ref.
                calculator = None

        return Result(
            value=value,
            root=root,
            nodes=nodes,
            named=named,
            calculator=calculator,
            provenance=provenance,
            perf_summary=PerfSummary(),
        )

    def to_json(self) -> str:
        """Serialize the whole record to a JSON string.

        The ``value``/``named``/``provenance`` fields are already
        codec-encoded and therefore JSON-safe; only the signature tuple needs a
        scalar conversion.
        """
        return json.dumps(
            _tag_tuples(
                {
                    "schema": self.schema,
                    "sim_signature": list(self.sim_signature),
                    "calculator_signature_hash": self.calculator_signature_hash,
                    "calculator_signature_text": self.calculator_signature_text,
                    "pretty_key": self.pretty_key,
                    "value": self.value,
                    "named": self.named,
                    "provenance": self.provenance,
                    "created_at": self.created_at,
                }
            ),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_json(cls, text: str) -> ResultRecord:
        """Reconstruct a record from the output of :meth:`to_json`."""
        data = json.loads(text, object_hook=_json_object_hook)
        return cls(
            schema=data.get("schema", RESULT_STORE_SCHEMA),
            sim_signature=tuple(data.get("sim_signature", ())),
            calculator_signature_hash=data.get("calculator_signature_hash", ""),
            calculator_signature_text=data.get("calculator_signature_text", ""),
            pretty_key=data.get("pretty_key"),
            value=data.get("value") or {},
            named=data.get("named"),
            provenance=data.get("provenance"),
            created_at=float(data.get("created_at", 0.0)),
        )


def _value_summary(value: Any) -> ValueSummary:
    """Build a compact :class:`ValueSummary` for a decoded value."""
    import numpy as np

    from pynbodyext.core.calculate.result.result import ValueSummary

    py_type = type(value).__name__
    if isinstance(value, np.ndarray):
        units = getattr(value, "units", None)
        return ValueSummary(
            python_type=py_type,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            units=None if units is None else str(units),
        )
    return ValueSummary(python_type=py_type)


class ResultStore(ABC):
    """Abstract persistence seam keyed on simulation + calculator identity.

    Concrete adapters only implement ``_put``, ``_get_by_key``, ``_delete``, and
    ``_list``; everything else (encoding a :class:`Result`, decoding it back,
    deriving the content-addressed key) is shared here.
    """

    def __init__(self, codec: ValueCodec | None = None) -> None:
        self.codec = codec or ValueCodec()

    # ------------------------------------------------------------------
    # Public, shared API
    # ------------------------------------------------------------------

    def store(
        self,
        result: Result,
        *,
        sim_signature: tuple[Any, ...],
        calculator_signature: CalculatorSignature,
        pretty_key: str | None = None,
    ) -> RecordRef:
        """Persist a result under the given simulation + calculator identity."""
        record = ResultRecord.from_result(
            result,
            sim_signature=sim_signature,
            calculator_signature=calculator_signature,
            codec=self.codec,
            pretty_key=pretty_key,
        )
        return self._put(record)

    def fetch(self, ref: RecordRef) -> Result:
        """Load a result previously stored under a :class:`RecordRef`."""
        record = self._get_by_key(ref.sim_signature, ref.calculator_signature_hash)
        if record is None:
            raise KeyError(f"no stored result for {ref.key!r}")
        return record.to_result(self.codec)

    def get(self, *, sim_signature: tuple[Any, ...], calculator_signature: CalculatorSignature) -> Result | None:
        """Return the stored result for an identity, or ``None`` on a miss."""
        hash_, _text = compute_calculator_key(calculator_signature)
        record = self._get_by_key(tuple(sim_signature), hash_)
        return None if record is None else record.to_result(self.codec)

    def has(self, *, sim_signature: tuple[Any, ...], calculator_signature: CalculatorSignature) -> bool:
        """Whether a result already exists for an identity."""
        hash_, _text = compute_calculator_key(calculator_signature)
        return self._get_by_key(tuple(sim_signature), hash_) is not None

    def delete(self, ref: RecordRef) -> bool:
        """Delete a stored result; return whether it existed."""
        return self._delete(ref)

    def list_refs(self, *, sim_signature: tuple[Any, ...] | None = None) -> list[RecordRef]:
        """List stored records, optionally filtered by simulation identity."""
        return self._list(sim_signature)

    def __contains__(self, ref: RecordRef) -> bool:
        return self._get_by_key(ref.sim_signature, ref.calculator_signature_hash) is not None

    def load(self, *, sim_signature: tuple[Any, ...], calculator_signature_text: str) -> Result | None:
        """Load a stored result by simulation identity + calculator signature text.

        This is the persistence counterpart to :meth:`store` built on the
        two-field ``(pretty_key, signature_text)`` pattern: given the signature
        text, the stored :class:`Result` is returned with a reconstructed,
        runnable ``result.calculator`` attached (the recipe and its outcome in a
        single object), or ``None`` on a miss.

        The text is deserialized into a :class:`CalculatorSignature` so the
        content-addressed hash can be derived (matching the one recorded at
        :meth:`store` time) to locate the record; the calculator is then rebuilt
        from the record's own ``calculator_signature_text`` via
        :func:`calculator_from_signature`.

        Returns
        -------
        Result | None
            The stored result, with ``result.calculator`` set, or ``None`` when
            no record exists for the identity.
        """
        signature = CalculatorSignature.from_json(calculator_signature_text)
        record = self._get_by_key(tuple(sim_signature), signature.short_hash())
        return None if record is None else record.to_result(self.codec)

    # ------------------------------------------------------------------
    # Adapter primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def _put(self, record: ResultRecord) -> RecordRef: ...

    @abstractmethod
    def _get_by_key(self, sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> ResultRecord | None: ...

    @abstractmethod
    def _delete(self, ref: RecordRef) -> bool: ...

    @abstractmethod
    def _list(self, sim_signature: tuple[Any, ...] | None) -> list[RecordRef]: ...


class InMemoryResultStore(ResultStore):
    """Reference :class:`ResultStore` adapter backed by ``dict``."""

    def __init__(self, codec: ValueCodec | None = None) -> None:
        super().__init__(codec)
        self._records: dict[str, ResultRecord] = {}

    def _put(self, record: ResultRecord) -> RecordRef:
        key = _record_key(record.sim_signature, record.calculator_signature_hash)
        self._records[key] = record
        return RecordRef(record.sim_signature, record.calculator_signature_hash)

    def _get_by_key(self, sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> ResultRecord | None:
        return self._records.get(_record_key(tuple(sim_signature), calculator_signature_hash))

    def _delete(self, ref: RecordRef) -> bool:
        return self._records.pop(ref.key, None) is not None

    def _list(self, sim_signature: tuple[Any, ...] | None) -> list[RecordRef]:
        refs: list[RecordRef] = []
        for record in self._records.values():
            if sim_signature is not None and record.sim_signature != tuple(sim_signature):
                continue
            refs.append(RecordRef(record.sim_signature, record.calculator_signature_hash))
        return refs
