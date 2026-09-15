"""Declarative parameter machinery behind ``@X.dataclass``.

Two contracts live here.  A role base declares a constructor parameter **once**
(``TransformBase.move_all``) and the decorator materialises it on every subclass,
so a subclass keeps its own fields first and the parameter stays keyword-only.
And the parameter-spec cache is per class: a decorated base must not shadow the
parameters its subclasses declare.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap

import pytest

from pynbodyext.core.calculate.nodes.transforms import TransformBase
from pynbodyext.properties import CenPos
from pynbodyext.core.calculate.params.fields import collect_param_specs
from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox

TRANSFORMS = (WrapBox, ShiftPosTo, ShiftVelTo, AlignVec)


def _call_shape(label: str) -> tuple[int, tuple[str | None, ...]]:
    """``(positional count, keyword names)`` of a rendered label."""
    call = ast.parse(label, mode="eval").body
    assert isinstance(call, ast.Call)
    return len(call.args), tuple(keyword.arg for keyword in call.keywords)


def _declared_field_names(cls: type) -> set[str]:
    """Annotated names written in *cls*'s own class body (not inherited)."""
    node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    assert isinstance(node, ast.ClassDef)
    return {
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    }


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_role_base_param_is_materialised_as_a_keyword_only_field(cls: type) -> None:
    """``move_all`` is declared on ``TransformBase``, not on each transform."""
    field = {item.name: item for item in dataclasses.fields(cls)}["move_all"]
    assert field.kw_only is True
    assert field.default is True
    assert "move_all" not in _declared_field_names(cls)

    parameter = inspect.signature(cls).parameters["move_all"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_transform_constructor_keeps_positional_parameters() -> None:
    """The inherited parameter does not disturb a subclass's positional fields."""
    assert WrapBox(None, "minirange").move_all is True
    assert ShiftPosTo("ssc").move_all is True
    assert WrapBox(move_all=False).move_all is False
    assert ShiftPosTo("ssc", move_all=False).move_all is False


def test_keyword_only_parameter_is_never_rendered_positionally() -> None:
    """Rendered labels stay callable: a keyword-only parameter stays ``name=``.

    It is also left out while it holds its declared default, so a tree does not
    repeat ``move_all=True`` on every transform line.
    """
    assert repr(WrapBox()) == 'WrapBox(None, "minirange")'
    assert repr(WrapBox(move_all=False)) == 'WrapBox(None, "minirange", move_all=False)'
    assert repr(ShiftPosTo("com", move_all=False)) == 'ShiftPosTo("com", move_all=False)'
    assert ShiftPosTo("com", move_all=False).to_signature().pretty() == 'ShiftPosTo("com", move_all=False)'
    assert _call_shape(repr(WrapBox(move_all=False)))[1] == ("move_all",)


def test_label_lists_positional_parameters_as_argument_or_declared_default() -> None:
    """A position parameter always shows, as the argument or the declared default.

    A parameter shows the value the constructor was given (``"com"``), or the declared
    default when it was not passed.  ``ShiftPosTo.__post_init__`` rewrites a mode
    string into a ``CenPos`` node, so the two labels differ only in that value — never
    in which parameters are listed, and never in showing a normalised object in place
    of the argument.
    """
    assert repr(ShiftPosTo()) == 'ShiftPosTo("ssc")'
    assert repr(ShiftPosTo("ssc")) == 'ShiftPosTo("ssc")'
    assert repr(ShiftPosTo("com")) == 'ShiftPosTo("com")'
    assert repr(WrapBox(convention="minirange")) == 'WrapBox(None, "minirange")'
    assert repr(WrapBox(convention="center")) == 'WrapBox(None, "center")'
    # a node passed explicitly is shown as the node (that is what the caller wrote)
    assert repr(ShiftPosTo(CenPos("com"))) == 'ShiftPosTo(CenPos("com"))'
    # the same effective state written as "the default" keeps a single identity
    assert ShiftPosTo().signature_hash() == ShiftPosTo("ssc").signature_hash()
    assert WrapBox().signature_hash() == WrapBox(convention="minirange").signature_hash()


def test_labels_of_one_class_list_the_same_parameters() -> None:
    """Same class, same positional list; keyword-only extras appear only when set."""
    shift = [ShiftPosTo("ssc"), ShiftPosTo("com"), ShiftPosTo(), ShiftPosTo("ssc", move_all=False)]
    assert {_call_shape(repr(node))[0] for node in shift} == {1}
    assert {_call_shape(repr(node))[1] for node in shift} == {(), ("move_all",)}

    wrap = [WrapBox(), WrapBox(convention="center"), WrapBox(move_all=False)]
    assert {_call_shape(repr(node))[0] for node in wrap} == {2}
    assert {_call_shape(repr(node))[1] for node in wrap} == {(), ("move_all",)}


def test_live_label_and_stored_payload_use_the_same_rule() -> None:
    """A signature payload renders exactly like the live object it came from."""
    from pynbodyext.core.calculate.result.render import TreePrinter
    from pynbodyext.core.calculate.result.signature import calculator_pretty_init_args

    for node in (WrapBox(), WrapBox(move_all=False), ShiftPosTo(), ShiftPosTo("com")):
        rendered = TreePrinter.dataclass_args(node.to_signature().payload)
        assert rendered == calculator_pretty_init_args(node), node


def test_field_changed_after_construction_is_no_longer_a_default() -> None:
    """A clone records the field it changed, without disturbing the original."""
    original = ShiftPosTo("ssc")
    clone = original._clone(move_all=False)
    assert repr(clone) == 'ShiftPosTo("ssc", move_all=False)'
    assert clone.signature_hash() != original.signature_hash()
    # ``_clone`` shares state by copy, so the record must not be shared with it
    assert repr(original) == 'ShiftPosTo("ssc")'
    assert clone._clone(move_all=True).signature_hash() == original.signature_hash()


def test_parameter_specs_are_cached_per_class() -> None:
    """A decorated base must not shadow the parameters of its subclasses."""

    @TransformBase.dataclass
    class Mid(TransformBase):
        k: int = 1

        def build_handle(self, sim, target, params=None):  # pragma: no cover - never run
            return None

    @Mid.dataclass
    class Leaf(Mid):
        j: int = 2

    names = [spec.name for spec in collect_param_specs(Leaf)]
    assert "j" in names, f"subclass parameter lost to the base cache: {names}"
    assert "k" in names
    assert "move_all" in names
