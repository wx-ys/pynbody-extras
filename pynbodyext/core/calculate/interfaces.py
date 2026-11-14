"""The seams between the calculator layers.

Import direction in ``core.calculate`` is a rule, not a habit: a module may only
import from the layers *below* it, so each layer can be imported, tested or replaced
without the ones above it.  The order, lowest first — the same tuple
``tests/test_layering_contract.py`` checks every run-time import against:

``interfaces`` · ``display`` · ``params`` · ``diagnostics`` · ``result`` · ``store`` ·
``runtime`` · ``nodes`` · ``bins`` · ``sample``

Two things break that order on their own, and both are handled here rather than by
an import:

- A lower layer needs to *recognise* an object the upper layer defines (is this a
  calculator node?) — that is a :class:`Protocol`, checked structurally.
- A lower layer needs the upper layer to *build* or *serialize* something (rebuild a
  calculator from a payload; wrap transforms in a chain) — that is a registration
  hook, filled in by the layer that owns the classes.

It also holds the **signature type names**: the stable name a calculator class is
identified by in a signature payload (see :func:`signature_type_name`), registered
by ``CalculatorBase.__init_subclass__`` and resolved back to the class by every
layer that reads payloads.

Nothing here imports a sibling module, which is the property that makes it usable
from every layer.  ``tests/test_layering_contract.py`` holds the rule.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pynbodyext.util._type import class_path, import_object

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "CalculatorNode",
    "ScopeLike",
    "is_calculator_node",
    "register_calculator_layer",
    "encode_calculator_payload",
    "decode_calculator_payload",
    "register_node_factory",
    "make_node",
    "signature_type_name",
    "signature_type_version",
    "register_signature_type",
    "resolve_signature_type",
    "SESSION_TAG",
    "STABLE_TAG",
    "is_stable_sim_identity",
]


@runtime_checkable
class CalculatorNode(Protocol):
    """What a lower layer needs from a calculator node, structurally.

    The parameter machinery asks "is this value a calculator I should resolve, or a
    dependency I should walk?" without importing the node classes; the node layer
    satisfies this by being a ``CalculatorBase``, and nothing else does.

    Three members, because a runtime protocol check is a ``hasattr`` walk: an object
    that happens to answer ``signature()`` alone (a stored record, a snapshot) must
    not be mistaken for a calculator.
    """

    kind: Any

    def signature(self) -> tuple[Any, ...]: ...

    def children(self) -> list[Any]: ...


@runtime_checkable
class ScopeLike(Protocol):
    """What a lower layer needs from a calculator's scope."""

    @property
    def is_empty(self) -> bool: ...


# ---------------------------------------------------------------------------
# Hooks the node layer installs
# ---------------------------------------------------------------------------

_CALCULATOR_LAYER: dict[str, Callable[..., Any]] = {}
_NODE_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_calculator_layer(
    *,
    is_node: Callable[[Any], bool],
    encode: Callable[..., tuple[Any, bool, tuple[str, ...], tuple[str, ...]]],
    decode: Callable[[dict[str, Any]], Any],
) -> None:
    """Install the node layer's three answers about calculators.

    ``is_node`` recognises a calculator object, ``encode`` turns one into a
    ``(payload, constructible, non_constructible_paths, unstable_paths)`` tuple,
    and ``decode`` rebuilds one from a payload.  ``nodes/serde.py`` calls this at import.
    """
    _CALCULATOR_LAYER.update(is_node=is_node, encode=encode, decode=decode)


def is_calculator_node(value: Any) -> bool:
    """Whether *value* is a calculator node, per the installed hook."""
    is_node = _CALCULATOR_LAYER.get("is_node")
    return bool(is_node(value)) if is_node is not None else False


def encode_calculator_payload(
    calculator: Any, *, path: str = "calculator", inline_array_bytes: int = 128
) -> tuple[Any, bool, tuple[str, ...], tuple[str, ...]]:
    """Encode *calculator* into its signature payload, via the node layer.

    Returns ``(payload, constructible, non_constructible_paths, unstable_paths)``;
    see :class:`~pynbodyext.core.calculate.result.signature.CalculatorSignature`.
    """
    encode = _CALCULATOR_LAYER.get("encode")
    if encode is None:
        raise RuntimeError(
            "no calculator encoder is registered; import pynbodyext.core.calculate.nodes first"
        )
    return encode(calculator, path=path, inline_array_bytes=inline_array_bytes)


def decode_calculator_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild a calculator from *payload*, via the node layer."""
    decode = _CALCULATOR_LAYER.get("decode")
    if decode is None:
        raise RuntimeError(
            "no calculator decoder is registered; import pynbodyext.core.calculate.nodes first"
        )
    return decode(dict(payload))


def register_node_factory(kind: str, factory: Callable[..., Any]) -> None:
    """Register a node-building factory under *kind*, for a lower layer to call.

    ``runtime.scopes`` builds a transform chain and a pipeline — both node classes —
    so it asks for them by name instead of importing the node layer.
    """
    _NODE_FACTORIES[kind] = factory


def make_node(kind: str, *args: Any, **kwargs: Any) -> Any:
    """Build the node registered under *kind*.

    The first parameter is positional-only in spirit (it names the registry slot), so
    it is spelled ``kind`` to leave ``name`` free for the node being built.
    """
    factory = _NODE_FACTORIES.get(kind)
    if factory is None:
        raise RuntimeError(f"no node factory registered as {kind!r}")
    return factory(*args, **kwargs)


# ---------------------------------------------------------------------------
# Signature type names
# ---------------------------------------------------------------------------

#: Declared signature names and aliases -> the class they identify.
_SIGNATURE_TYPES: dict[str, type] = {}


def _own(cls: type, attribute: str, default: Any) -> Any:
    """Return *attribute* as declared on *cls* itself, ignoring its bases.

    Type identity is not inherited: a subclass of ``Sphere`` is a different
    calculator and must not be stored, or rebuilt, as a ``Sphere``.
    """
    return cls.__dict__.get(attribute, default)


def signature_type_name(obj: Any) -> str:
    """The name a calculator class (or instance) has in signature payloads.

    The class's own ``signature_name`` when it declares one — conventionally its
    public import path, such as ``"pynbodyext.filters.Sphere"``, which stays put
    when the defining module is reorganised — and otherwise its full
    :func:`~pynbodyext.util._type.class_path`.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    declared = _own(cls, "signature_name", None)
    return declared if isinstance(declared, str) and declared else class_path(cls)


def signature_type_version(obj: Any) -> int:
    """The class's own ``signature_version`` (0 when it declares none).

    Bump it when what a calculator computes changes for the same parameters, so
    results stored under the old signature stop matching the new code.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    version = _own(cls, "signature_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise TypeError(f"{cls.__qualname__}.signature_version must be a non-negative int, got {version!r}")
    return version


def register_signature_type(cls: type) -> None:
    """Register the ``signature_name`` and ``signature_aliases`` *cls* declares.

    Called for every calculator class by ``CalculatorBase.__init_subclass__``; a
    class that declares neither is identified by its class path and needs no
    entry.  A name is unique: claiming one another class already holds raises
    :class:`TypeError`, because two classes behind one name would make a stored
    signature ambiguous.  Re-creating the same class (a module reload, or the
    copy ``dataclasses`` makes for ``slots=True``) replaces the entry.
    """
    names = []
    declared = _own(cls, "signature_name", None)
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise TypeError(f"{cls.__qualname__}.signature_name must be a non-empty string, got {declared!r}")
        names.append(declared)
    aliases = _own(cls, "signature_aliases", ())
    if isinstance(aliases, str) or not all(isinstance(alias, str) and alias for alias in aliases):
        raise TypeError(f"{cls.__qualname__}.signature_aliases must be a tuple of non-empty strings")
    names.extend(aliases)
    signature_type_version(cls)  # validate early, at class creation

    for name in names:
        current = _SIGNATURE_TYPES.get(name)
        if current is not None and current is not cls and class_path(current) != class_path(cls):
            raise TypeError(
                f"signature name {name!r} is already taken by {class_path(current)}; {class_path(cls)} cannot claim it"
            )
    for name in names:
        _SIGNATURE_TYPES[name] = cls


def resolve_signature_type(name: str) -> Any:
    """Return the class a payload names: a registered name or alias, else an import path."""
    registered = _SIGNATURE_TYPES.get(name)
    if registered is not None:
        return registered
    return import_object(name)


# ---------------------------------------------------------------------------
# Simulation identity tags
# ---------------------------------------------------------------------------

#: First element of a simulation identity that is valid in this process only.
SESSION_TAG = "session"
#: First element of a simulation identity that addresses the same data in every session.
STABLE_TAG = "sim"


def is_stable_sim_identity(identity: tuple[Any, ...]) -> bool:
    """Whether a simulation identity addresses the same data in another session.

    The convention is shared by the runtime (which makes identities, see
    ``runtime.sim_identity``) and the store (which refuses session-only ones for
    persistence), so it lives here, below both.
    """
    return bool(identity) and identity[0] != SESSION_TAG


# Persistence is an optional upper-layer capability. These definitions must stay
# free of runtime/store/ORM imports so Results can delegate saving down this seam.


class DatabaseSession(Protocol):
    def run_calculator(self, calculator: Any, source: Any, *, options: Any, mode: str) -> Any: ...

    def save_result(self, result: Any, *, name: str | None = None) -> Any: ...


@dataclass(frozen=True)
class SavedInput:
    """Input facts captured before execution, with no live particle references."""

    signature: tuple[Any, ...]
    root_signature: tuple[Any, ...]
    location: str | None
    root_count: int
    indices: tuple[int, ...] | None = None
    unit_basis: tuple[str, ...] = ()


DATABASE_CONTEXT: ContextVar[tuple[DatabaseSession, str] | None] = ContextVar("pynbodyext_database", default=None)
