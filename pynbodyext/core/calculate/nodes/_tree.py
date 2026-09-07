"""Tree-labeling and rendering helpers for calculator dependency graphs.

These are pure functions that translate a calculator node (or a group of nodes)
into the multi-line tree strings shown by ``CalculatorBase.format_tree`` and the
node-tree report.  They operate on the duck-typed node surface — ``kind``,
``tree_label``, and ``children()`` — so they stay independent of the concrete
class hierarchy in :mod:`pynbodyext.core.calculate.nodes.base`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase


def kind_label(kind: str, *, compact: bool) -> str:
    """Return a human label for *kind*, optionally shortened in compact mode."""
    if not compact:
        return kind
    return {"property": "prop", "filter": "filt", "transform": "trans", "calculator": "calc", "combined": "comb"}.get(
        kind, kind
    )


def input_node(node: CalculatorBase[Any, Any]) -> CalculatorBase[Any, Any]:
    """Return the node whose init args should label the tree entry."""
    return node


def label_for(node: CalculatorBase[Any, Any], *, show_inputs: bool, compact_kinds: bool) -> str:
    """Build the ``name<kind>`` label for a tree entry, optionally with init args."""
    from pynbodyext.core.calculate.result.signature import calculator_pretty_init_args

    label = node.tree_label
    target = input_node(node)

    if show_inputs:
        init_text = calculator_pretty_init_args(target)
        if init_text:
            label = f"{label}({init_text})"

    kind = kind_label(node.kind, compact=compact_kinds)
    return f"{label}<{kind}>"


def hidden_label(children: list[CalculatorBase[Any, Any]]) -> str:
    """Describe *children* that are elided from a truncated tree."""
    if not children:
        return "..."

    descendants = 0
    stack = list(children)
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        key = id(current)
        if key in seen:
            continue
        seen.add(key)
        descendants += 1
        stack.extend(current.children())

    suffix = "node" if descendants == 1 else "nodes"
    return f"... {descendants} {suffix} hidden"


def render_children(
    node: CalculatorBase[Any, Any],
    *,
    prefix: str,
    depth: int,
    max_depth: int | None,
    max_children: int | None,
    show_inputs: bool,
    compact_kinds: bool,
) -> list[str]:
    """Render the subtree rooted at *node* as a list of tree lines."""
    children = node.children()
    if not children:
        return []

    if max_depth is not None and depth >= max_depth:
        return [f"{prefix}└─ {hidden_label(children)}"]

    visible_children = children if max_children is None else children[:max_children]
    hidden_children = [] if max_children is None else children[max_children:]

    lines: list[str] = []
    for index, child in enumerate(visible_children):
        is_last = index == len(visible_children) - 1 and not hidden_children
        branch = "└─" if is_last else "├─"
        lines.append(f"{prefix}{branch} {label_for(child, show_inputs=show_inputs, compact_kinds=compact_kinds)}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        lines.extend(
            render_children(
                child,
                prefix=child_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                show_inputs=show_inputs,
                compact_kinds=compact_kinds,
            )
        )

    if hidden_children:
        lines.append(f"{prefix}└─ {hidden_label(hidden_children)}")

    return lines
