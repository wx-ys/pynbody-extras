"""A calculator's identity in a signature does not depend on where its class lives.

A payload names a calculator class by its *signature name*: the class's own
``signature_name`` (conventionally its public import path) or, failing that, its
class path.  Built-in calculators declare their public path, so reorganising the
modules that define them leaves every stored key untouched; ``signature_aliases``
keep old names readable after a rename, and ``signature_version`` changes the key
when the algorithm changes.  Hand-written (non-dataclass) nodes become rebuildable
by defining ``from_signature_payload``.
"""

from __future__ import annotations

import inspect

import pytest
from calculate_helpers import make_sim

import pynbodyext.filters
import pynbodyext.properties
import pynbodyext.transforms
from pynbodyext.core.calculate import CalculatorBase, Pipeline, PropertyBase, interfaces
from pynbodyext.core.calculate.interfaces import (
    resolve_signature_type,
    signature_type_name,
    signature_type_version,
)
from pynbodyext.core.calculate.result.signature import calculator_from_signature, calculator_to_signature
from pynbodyext.filters import Sphere
from pynbodyext.util._type import class_path, import_object


@pytest.fixture
def registry():
    """Restore the signature-name registry after a test that declares classes."""
    saved = dict(interfaces._SIGNATURE_TYPES)
    yield interfaces._SIGNATURE_TYPES
    interfaces._SIGNATURE_TYPES.clear()
    interfaces._SIGNATURE_TYPES.update(saved)


# ---------------------------------------------------------------------------
# Built-in calculators
# ---------------------------------------------------------------------------


def _exported_calculators() -> list[type]:
    out = []
    for module in (pynbodyext.filters, pynbodyext.properties, pynbodyext.transforms):
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, CalculatorBase) and obj is not PropertyBase:
                out.append(obj)
    return out


def test_every_exported_calculator_declares_its_public_path() -> None:
    """A built-in calculator is named by the path users import it from."""
    calculators = _exported_calculators()
    assert len(calculators) >= 20
    for cls in calculators:
        name = cls.__dict__.get("signature_name")
        assert name is not None, f"{cls.__qualname__} declares no signature_name"
        assert name.startswith(("pynbodyext.filters.", "pynbodyext.properties.", "pynbodyext.transforms."))


def test_every_registered_builtin_name_imports_to_its_class() -> None:
    """A declared name must stay importable, so a payload rebuilds without the registry."""
    builtin = {name: cls for name, cls in interfaces._SIGNATURE_TYPES.items() if name.startswith("pynbodyext.")}
    assert "pynbodyext.filters.Sphere" in builtin
    assert "pynbodyext.core.calculate.Pipeline" in builtin
    for name, cls in builtin.items():
        assert import_object(name) is cls, name
        assert resolve_signature_type(name) is cls, name


def test_a_payload_names_the_public_path_not_the_defining_module() -> None:
    payload = calculator_to_signature(Sphere("30 kpc")).payload
    assert payload["class"] == "pynbodyext.filters.Sphere"
    assert class_path(Sphere) == "pynbodyext.filters.filt.Sphere"  # where it actually lives


def test_a_subclass_does_not_inherit_the_name() -> None:
    """A subclass of Sphere is another calculator: it must not be stored or rebuilt as one."""

    @CalculatorBase.dataclass
    class BigSphere(Sphere):
        pass

    assert signature_type_name(BigSphere) == class_path(BigSphere)
    assert calculator_to_signature(BigSphere("30 kpc")).hash != calculator_to_signature(Sphere("30 kpc")).hash


# ---------------------------------------------------------------------------
# Moving and renaming a class
# ---------------------------------------------------------------------------


def _mass_class(qualname: str, **attributes):
    """A dataclass property built under *qualname*, as if defined in another module."""

    def calculate(self, sim, params=None) -> float:
        return float(sim["mass"].sum())

    namespace = {"__qualname__": qualname, "__module__": f"relocated.{qualname.lower()}", "calculate": calculate}
    namespace.update(attributes)
    return PropertyBase.dataclass(type(qualname, (PropertyBase,), namespace))


def test_moving_a_named_class_keeps_its_key(registry) -> None:
    before = _mass_class("TotalMass", signature_name="labkit.TotalMass")
    key = calculator_to_signature(before()).hash

    registry.pop("labkit.TotalMass")  # the old module is gone ...
    after = _mass_class("TotalMassMoved", signature_name="labkit.TotalMass")  # ... the class lives elsewhere
    assert calculator_to_signature(after()).hash == key


def test_an_alias_reads_payloads_written_under_the_old_name(registry) -> None:
    old = _mass_class("OldMass", signature_name="labkit.OldMass")
    text = calculator_to_signature(old()).to_json()
    registry.pop("labkit.OldMass")

    renamed = _mass_class("NewMass", signature_name="labkit.NewMass", signature_aliases=("labkit.OldMass",))
    rebuilt = calculator_from_signature(text)
    assert type(rebuilt) is renamed
    assert rebuilt(make_sim()) == 15.0


def test_a_name_belongs_to_one_class(registry) -> None:
    _mass_class("First", signature_name="labkit.Taken")
    with pytest.raises(TypeError, match="already taken"):
        _mass_class("Second", signature_name="labkit.Taken")


def test_re_creating_the_same_class_replaces_its_entry(registry) -> None:
    """A module reload (or a slotted dataclass copy) re-registers the same class path."""
    first = _mass_class("Reloaded", signature_name="labkit.Reloaded")
    second = _mass_class("Reloaded", signature_name="labkit.Reloaded")
    assert first is not second
    assert resolve_signature_type("labkit.Reloaded") is second


@pytest.mark.parametrize(
    "attributes",
    [{"signature_name": ""}, {"signature_aliases": "labkit.x"}, {"signature_version": -1}, {"signature_version": 1.5}],
    ids=["empty-name", "alias-string", "negative-version", "float-version"],
)
def test_invalid_declarations_fail_at_class_creation(registry, attributes) -> None:
    with pytest.raises(TypeError):
        _mass_class("Invalid", **attributes)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_a_version_bump_changes_the_key(registry) -> None:
    v0 = _mass_class("Versioned", signature_name="labkit.Versioned")
    key0 = calculator_to_signature(v0()).hash
    assert "version" not in calculator_to_signature(v0()).payload  # version 0 leaves keys alone
    registry.pop("labkit.Versioned")

    v1 = _mass_class("Versioned", signature_name="labkit.Versioned", signature_version=1)
    sig1 = calculator_to_signature(v1())
    assert sig1.payload["version"] == 1
    assert signature_type_version(v1) == 1
    assert sig1.hash != key0


def test_rebuilding_an_older_version_warns(registry) -> None:
    v0 = _mass_class("Evolving", signature_name="labkit.Evolving")
    text = calculator_to_signature(v0()).to_json()
    registry.pop("labkit.Evolving")
    v2 = _mass_class("Evolving", signature_name="labkit.Evolving", signature_version=2)

    with pytest.warns(UserWarning, match="version 0"):
        rebuilt = calculator_from_signature(text)
    assert type(rebuilt) is v2
    assert calculator_to_signature(rebuilt).hash != calculator_to_signature(v0()).hash


# ---------------------------------------------------------------------------
# Hand-written nodes
# ---------------------------------------------------------------------------


class HotFraction(PropertyBase[float]):
    """A non-dataclass node that declares its identity and how to rebuild from it."""

    signature_name = "tests.HotFraction"

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def signature_payload(self):
        return {"threshold": self.threshold}

    @classmethod
    def from_signature_payload(cls, payload):
        return cls(payload["threshold"])

    def calculate(self, sim, params=None) -> float:
        return float((sim["temp"] > self.threshold).mean())


class Unrebuildable(PropertyBase[float]):
    """Declares an identity but not how to rebuild from it."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def signature_payload(self):
        return {"threshold": self.threshold}

    def calculate(self, sim, params=None) -> float:
        return 0.0


def test_a_hand_written_node_with_from_signature_payload_round_trips() -> None:
    node = HotFraction(25.0)
    sig = node.to_signature()
    assert sig.constructible and sig.stable
    rebuilt = calculator_from_signature(sig.to_json())
    assert isinstance(rebuilt, HotFraction)
    assert rebuilt.threshold == 25.0
    assert rebuilt.to_signature().hash == sig.hash
    assert rebuilt(make_sim()) == node(make_sim())


def test_a_hand_written_node_without_it_is_stable_but_not_rebuildable() -> None:
    sig = Unrebuildable(1.0).to_signature()
    assert sig.stable
    assert not sig.constructible
    with pytest.raises(ValueError, match="not constructible"):
        calculator_from_signature(sig.to_json())


def test_a_rebuildable_hand_written_node_inside_a_pipeline_round_trips() -> None:
    pipeline = Pipeline({"hot": HotFraction(25.0)})
    rebuilt = calculator_from_signature(pipeline.to_signature().to_json())
    assert rebuilt(make_sim()) == pipeline(make_sim())


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_result_find_matches_a_named_class() -> None:
    """Nodes record the payload's class name; a query by class still finds them."""
    result = Pipeline({"hot": HotFraction(25.0)}).run(make_sim())
    found = result.find(HotFraction)
    assert [node.class_info.semantic_calculator_class_path for node in found] == ["tests.HotFraction"]


def test_rendering_and_packing_do_not_run_default_factories_again(registry):
    from dataclasses import field

    from pynbodyext.core.calculate.result.signature import calculator_pack, calculator_parameter_rows

    calls = []

    def next_sample():
        calls.append(1)
        return len(calls)

    @PropertyBase.dataclass
    class Sample(PropertyBase):
        signature_name = "tests.FactorySample"
        sample: int = field(default_factory=next_sample)

        def calculate(self, sim, params=None):
            return self.sample

    first, second = Sample(), Sample()
    assert (first.sample, second.sample) == (1, 2)
    assert repr(first) == "Sample(1)"
    assert calculator_parameter_rows(first)[0][:2] == ("sample", "1")
    assert first._repr_html_()
    text, signature = calculator_pack(second)
    assert text == "FactorySample(2)"
    assert calculator_from_signature(signature).sample == 2
    assert len(calls) == 2


@pytest.mark.parametrize("style", ["rich", "simple"])
def test_constant_and_handwritten_identity_values_appear_in_parameter_rows(style):
    from pynbodyext.core.calculate.display import repr_style
    from pynbodyext.core.calculate.nodes.expr import ConstantProperty
    from pynbodyext.core.calculate.result.signature import calculator_parameter_rows

    for node, name, value in [(ConstantProperty(3), "value", "3"), (HotFraction(25), "threshold", "25")]:
        assert calculator_parameter_rows(node) == [(name, value, False)]
        with repr_style(style):
            html = node._repr_html_()
        assert ">Parameters</" in html and name in html
