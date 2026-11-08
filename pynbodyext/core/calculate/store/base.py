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
- ``calculator_signature`` — a :class:`CalculatorSignature` whose full sha256
  :attr:`~CalculatorSignature.hash` identifies the calculator independently of
  its textual representation, and whose JSON text allows reconstruction.

A record found under a hash is checked against the requested signature before it
is returned.  A persistent store refuses either half of a key that would mean
nothing in the next session: a calculator signature that is not
:attr:`~CalculatorSignature.stable` (identified by ``id()`` or a function repr),
and a session-only simulation identity (see
:mod:`~pynbodyext.core.calculate.runtime.sim_identity`).

:class:`ResultStore` is deliberately an abstract template: it implements the
high-level :meth:`ResultStore.store`, :meth:`ResultStore.fetch`,
:meth:`ResultStore.get`, :meth:`ResultStore.has`, and :meth:`ResultStore.delete`
in terms of six small abstract primitives (``_put``, ``_get_by_key``,
``_delete``, ``_list``, ``_get_version``, ``_list_versions``).  :class:`InMemoryResultStore` is the reference adapter;
a SQLAlchemy adapter in the same package implements the same primitives for a
relational database.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from pynbodyext.core.calculate.interfaces import is_stable_sim_identity
from pynbodyext.core.calculate.result.signature import CalculatorSignature, calculator_from_signature

from .codec import ValueCodec
from .identity import sim_identity_key

if TYPE_CHECKING:
    from pynbodyext.core.calculate.result.result import Result, ValueSummary

    from .catalog import SignatureCatalog

__all__ = [
    "RESULT_STORE_SCHEMA",
    "UnstableSignatureError",
    "UnstableSimIdentityError",
    "SignatureMismatchError",
    "RecordRef",
    "ResultRecord",
    "ResultStore",
    "InMemoryResultStore",
    "compute_calculator_key",
]

#: v4 adds immutable save versions; calculator keys remain full sha256.
RESULT_STORE_SCHEMA = "pynbodyext.result.store/v4"


class UnstableSignatureError(ValueError):
    """A persistent store was asked to key a record on a session-specific signature."""


class UnstableSimIdentityError(ValueError):
    """A persistent store was asked to key a record on a session-only simulation identity."""


class SignatureMismatchError(RuntimeError):
    """Stored identity, calculator signature or version differs from the request."""


def _sig_text(sig: tuple[Any, ...]) -> str:
    """Serialize a simulation identity using the shared typed-key contract."""
    return sim_identity_key(sig)


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
    """Return ``(hash, json_text)`` for a structured signature.

    The full sha256 hash is the content-addressed identity used as the lookup key;
    the JSON is retained so a loaded store entry can reconstruct the calculator,
    and so a hit can be checked against the signature that asked for it.
    """
    return (signature.hash, signature.to_json())


def _check_record(record: ResultRecord | None, signature: CalculatorSignature) -> ResultRecord | None:
    """Return *record* if it was stored for *signature*; raise if it was not.

    Keys are content hashes, so a different payload under the same key means a
    collision or a corrupted row.  Either way the stored value answers another
    question, and returning it would be a silent wrong result.
    """
    if record is None or not record.calculator_signature_text:
        return record
    stored = CalculatorSignature.from_json(record.calculator_signature_text)
    if stored.identity() != signature.identity():
        raise SignatureMismatchError(
            f"record {record.calculator_signature_hash[:12]}... was stored for {stored.pretty()!r}, "
            f"not for {signature.pretty()!r}"
        )
    return record


@dataclass(frozen=True, slots=True)
class RecordRef:
    """A cache address, or an immutable saved version when version_id is set.

    ``store()`` returns pinned versions; ``list_refs()`` returns current cache
    addresses. Pair either reference with its owning backend. ``id`` is optional
    backend-local metadata and is excluded from identity and JSON serialization.
    """

    sim_signature: tuple[Any, ...]
    calculator_signature_hash: str
    id: Any = None
    version_id: str | None = None

    def __post_init__(self) -> None:
        if self.version_id is not None and (not isinstance(self.version_id, str) or not self.version_id):
            raise ValueError("version_id must be a non-empty string or None")

    @property
    def cache_key(self) -> str:
        """The computation address, independent of any saved version."""
        return _record_key(self.sim_signature, self.calculator_signature_hash)

    @property
    def key(self) -> str:
        """The address including the saved version, when pinned."""
        if self.version_id is None:
            return self.cache_key
        return f"{self.cache_key}::version:{json.dumps(self.version_id)}"

    def to_json(self) -> str:
        """Serialize a reference for later use with the same backend."""
        _sig_text(self.sim_signature)  # validate types before serializing
        return json.dumps(
            _tag_tuples(
                {
                    "schema": "pynbodyext.result.ref/v1",
                    "sim_signature": self.sim_signature,
                    "calculator_signature_hash": self.calculator_signature_hash,
                    "version_id": self.version_id,
                }
            ),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> RecordRef:
        """Restore a typed cache or version address; unknown schemas are rejected."""
        data = json.loads(text, object_hook=_json_object_hook)
        if not isinstance(data, dict) or data.get("schema") != "pynbodyext.result.ref/v1":
            raise ValueError("unsupported result-reference schema")
        ref = cls(data["sim_signature"], data["calculator_signature_hash"], version_id=data["version_id"])
        _sig_text(ref.sim_signature)
        if not isinstance(ref.calculator_signature_hash, str) or not ref.calculator_signature_hash:
            raise ValueError("calculator_signature_hash must be a non-empty string")
        return ref

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
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def ref(self) -> RecordRef:
        """The immutable reference to this saved record."""
        return RecordRef(self.sim_signature, self.calculator_signature_hash, version_id=self.version_id)

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
            if node.record.stored_value and node.record.value is not None:
                named[name] = codec.encode(node.record.value)

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

    def to_result(self, codec: ValueCodec, *, calculator: Any = None) -> Result:
        """Reconstruct a :class:`Result` from a stored record.

        *calculator* is attached as ``result.calculator`` when the caller already
        holds the live node (a cache hit during a run); otherwise it is rebuilt
        from the stored signature text when possible.
        """
        from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeStatus
        from pynbodyext.core.calculate.result.result import NodeRecord, PerfSummary, ProvenanceInfo, Result, ResultNode

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
                record=NodeRecord(value=decoded, stored_value=True, value_summary=_value_summary(decoded)),
            )

        root = ResultNode(
            node_id="stored:1",
            kind=BuiltinKinds.CALCULATOR,
            signature=(self.calculator_signature_hash,),
            status=NodeStatus.OK,
            name=self.pretty_key,
            display_name=self.pretty_key,
            record=NodeRecord(value=value, stored_value=True, value_summary=_value_summary(value)),
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

        if calculator is None and self.calculator_signature_text:
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
            from_store=True,
            store_ref=self.ref,
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
                    "version_id": self.version_id,
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
        if not isinstance(data, dict) or data.get("schema") != RESULT_STORE_SCHEMA:
            raise ValueError("unsupported result-store schema; expected " + RESULT_STORE_SCHEMA)
        if not isinstance(data.get("version_id"), str) or not data["version_id"]:
            raise ValueError("result-store record requires a non-empty version_id")
        return cls(
            schema=data["schema"],
            sim_signature=tuple(data.get("sim_signature", ())),
            calculator_signature_hash=data.get("calculator_signature_hash", ""),
            calculator_signature_text=data.get("calculator_signature_text", ""),
            pretty_key=data.get("pretty_key"),
            value=data.get("value") or {},
            named=data.get("named"),
            provenance=data.get("provenance"),
            created_at=float(data.get("created_at", 0.0)),
            version_id=data["version_id"],
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

    Concrete adapters implement ``_put``, ``_get_by_key``, ``_delete``,
    ``_list``, ``_get_version`` and ``_list_versions``; everything else (encoding a :class:`Result`, decoding it back,
    deriving the content-addressed key, checking a hit) is shared here.
    """

    #: Whether records outlive the Python process.  A persistent store refuses
    #: calculator signatures that are not stable across sessions.
    persistent: ClassVar[bool] = True

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
        """Append an immutable version and update the current cache atomically.

        Returns a pinned reference. Each save creates a new version, even when
        values are identical. Historical versions are retained after deletion.

        Raises
        ------
        UnstableSignatureError
            If the store is :attr:`persistent` and the signature is not
            :attr:`~CalculatorSignature.stable`.
        """
        if self.persistent and not is_stable_sim_identity(tuple(sim_signature)):
            raise UnstableSimIdentityError(
                f"cannot persist a result keyed on a session-only simulation identity {tuple(sim_signature)!r}. "
                "Use a snapshot loaded from a file with no transformation applied outside the calculator "
                "(see runtime.sim_identity.snapshot_identity), or pass a stable sim_identity provider."
            )
        if self.persistent and not calculator_signature.stable:
            paths = ", ".join(calculator_signature.unstable_paths) or "unknown"
            raise UnstableSignatureError(
                f"cannot persist a result keyed on a session-specific signature ({calculator_signature.pretty()}); "
                f"identified only within this session at: {paths}. Give the calculator a stable identity "
                "(dataclass parameters or signature_payload()) instead of a lambda or an opaque object."
            )
        record = ResultRecord.from_result(
            result,
            sim_signature=sim_signature,
            calculator_signature=calculator_signature,
            codec=self.codec,
            pretty_key=pretty_key,
        )
        return self._put(record)

    def fetch(self, ref: RecordRef) -> Result:
        """Read an exact saved version, or the current result for an unpinned ref."""
        record = self._read_ref(ref)
        if record is None:
            raise KeyError(f"no stored result for {ref.key!r}")
        return record.to_result(self.codec)

    def _read_ref(self, ref: RecordRef) -> ResultRecord | None:
        record = (
            self._get_by_key(ref.sim_signature, ref.calculator_signature_hash)
            if ref.version_id is None
            else self._get_version(ref.version_id)
        )
        if record is None:
            return None
        if record.ref.cache_key != ref.cache_key or (
            ref.version_id is not None and record.version_id != ref.version_id
        ):
            raise SignatureMismatchError("stored record does not match the requested identity/version")
        if record.calculator_signature_text:
            signature = CalculatorSignature.from_json(record.calculator_signature_text)
            if signature.hash != ref.calculator_signature_hash:
                raise SignatureMismatchError("stored signature text does not match the requested hash")
        return record

    def get(
        self, *, sim_signature: tuple[Any, ...], calculator_signature: CalculatorSignature, calculator: Any = None
    ) -> Result | None:
        """Return the stored result for an identity, or ``None`` on a miss.

        *calculator*, when given, becomes ``result.calculator`` instead of a node
        rebuilt from the stored signature.
        """
        record = _check_record(
            self._read_ref(RecordRef(tuple(sim_signature), calculator_signature.hash)), calculator_signature
        )
        return None if record is None else record.to_result(self.codec, calculator=calculator)

    def has(self, *, sim_signature: tuple[Any, ...], calculator_signature: CalculatorSignature) -> bool:
        """Whether a result already exists for an identity."""
        record = self._read_ref(RecordRef(tuple(sim_signature), calculator_signature.hash))
        return _check_record(record, calculator_signature) is not None

    def delete(self, ref: RecordRef) -> bool:
        """Invalidate current cache, retaining history; pinned refs must be current."""
        return self._delete(ref)

    def list_refs(self, *, sim_signature: tuple[Any, ...] | None = None) -> list[RecordRef]:
        """List current cache addresses, optionally filtered by simulation identity."""
        return self._list(sim_signature)

    def list_versions(self, ref: RecordRef) -> list[RecordRef]:
        """List pinned historical versions of an address in insertion order."""
        return self._list_versions(ref)

    def catalog(self) -> SignatureCatalog:
        """A :class:`~pynbodyext.core.calculate.store.catalog.SignatureCatalog` of current cached calculators.

        Each record carries the signature it was stored under, so the store can
        say which calculators it holds results for, rebuild them, and describe a
        key it returns.  Entries carry no label; a sample table, or the caller,
        labels the ones it uses.
        """
        from .catalog import SignatureCatalog

        catalog = SignatureCatalog()
        seen: set[str] = set()
        for ref in self.list_refs():
            if ref.calculator_signature_hash in seen:
                continue
            seen.add(ref.calculator_signature_hash)
            record = self._read_ref(ref)
            if record is None or not record.calculator_signature_text:
                continue
            catalog.add(CalculatorSignature.from_json(record.calculator_signature_text))
        return catalog

    def sim_signatures(self) -> list[tuple[Any, ...]]:
        """Simulation identities in the current cache, in backend listing order."""
        out: dict[str, tuple[Any, ...]] = {}
        for ref in self.list_refs():
            out.setdefault(_sig_text(ref.sim_signature), ref.sim_signature)
        return list(out.values())

    def __contains__(self, ref: RecordRef) -> bool:
        return self._read_ref(ref) is not None

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
        record = _check_record(self._read_ref(RecordRef(tuple(sim_signature), signature.hash)), signature)
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

    @abstractmethod
    def _get_version(self, version_id: str) -> ResultRecord | None: ...

    @abstractmethod
    def _list_versions(self, ref: RecordRef) -> list[RecordRef]: ...


class InMemoryResultStore(ResultStore):
    """Reference :class:`ResultStore` adapter backed by ``dict``.

    Lives as long as the process, so it accepts session-specific signatures too.
    """

    persistent: ClassVar[bool] = False

    def __init__(self, codec: ValueCodec | None = None) -> None:
        super().__init__(codec)
        self._records: dict[str, ResultRecord] = {}
        self._versions: dict[str, str] = {}
        self._history: dict[str, list[RecordRef]] = {}

    def _put(self, record: ResultRecord) -> RecordRef:
        key = _record_key(record.sim_signature, record.calculator_signature_hash)
        encoded = record.to_json()
        current = ResultRecord.from_json(encoded)
        if record.version_id in self._versions:
            raise ValueError("a saved result version cannot be overwritten")
        self._versions[record.version_id] = encoded
        self._history.setdefault(key, []).append(record.ref)
        self._records[key] = current
        return record.ref

    def _get_by_key(self, sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> ResultRecord | None:
        current = self._records.get(_record_key(tuple(sim_signature), calculator_signature_hash))
        # The cache selects a version; its metadata is not the saved payload.
        # Resolve through history just as the SQL adapter joins its version table.
        return None if current is None else self._get_version(current.version_id)

    def _delete(self, ref: RecordRef) -> bool:
        current = self._records.get(ref.cache_key)
        if current is None or (ref.version_id is not None and current.version_id != ref.version_id):
            return False
        del self._records[ref.cache_key]
        return True

    def _get_version(self, version_id: str) -> ResultRecord | None:
        encoded = self._versions.get(version_id)
        return None if encoded is None else ResultRecord.from_json(encoded)

    def _list_versions(self, ref: RecordRef) -> list[RecordRef]:
        return list(self._history.get(ref.cache_key, ()))

    def _list(self, sim_signature: tuple[Any, ...] | None) -> list[RecordRef]:
        refs: list[RecordRef] = []
        requested = None if sim_signature is None else _sig_text(tuple(sim_signature))
        for record in self._records.values():
            if requested is not None and _sig_text(record.sim_signature) != requested:
                continue
            refs.append(RecordRef(record.sim_signature, record.calculator_signature_hash))
        return refs
