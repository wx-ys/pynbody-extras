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
from pynbodyext.core.calculate.params.fields import collect_param_specs
from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox

TRANSFORMS = (WrapBox, ShiftPosTo, ShiftVelTo, AlignVec)


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
    """Rendered labels stay callable: a keyword-only parameter stays ``name=``."""
    assert repr(WrapBox()) == 'WrapBox(None, "minirange", move_all=True)'
    assert repr(WrapBox(move_all=False)) == "WrapBox(move_all=False)"
    assert repr(ShiftPosTo("ssc", move_all=False)) == 'ShiftPosTo(CenPos("ssc"), move_all=False)'
    assert WrapBox(move_all=False).to_signature().pretty() == "WrapBox(move_all=False)"


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
