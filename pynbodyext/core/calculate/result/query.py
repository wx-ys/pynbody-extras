"""Graph traversal, node lookup, and tree rendering for :class:`Result`.

:class:`ResultQuery` is a stateless helper that operates on :class:`Result`
and :class:`ResultNode` objects produced by the evaluation engine.

All methods are static so they can be used without a class instance, but
:class:`Result` delegates to them via regular instance methods for convenience.

Responsibilities
----------------
- Node lookup by id, name, type, or predicate
- Parent/child/ancestor/descendant traversal
- Node tree and execution tree text rendering
- Visible-node limits and hidden-count helpers
- Performance and cache suffix generation
"""

from __future__ import annotations

from typing import Any

from pynbodyext.core.calculate.diagnostics.observer import format_observation_access
from pynbodyext.core.calculate.display import display_value, format_time

from .enums import NodeStatus
from .result import PhaseRecord, Result, ResultNode


class ResultQuery:
    @staticmethod
    def _class_path(value: Any) -> str:
        cls = value if isinstance(value, type) else type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    @staticmethod
    def _short_class_name(path: str | None) -> str | None:
        if path is None:
            return None
        return path.rsplit(".", 1)[-1]

    @staticmethod
    def _reverse_parents(result: Result[Any], node_id: str) -> list[ResultNode]:
        return [candidate for candidate in result.nodes.values() if node_id in candidate.children]

    @staticmethod
    def find(result: Result[Any], query: Any) -> list[ResultNode]:
        if isinstance(query, ResultNode):
            resolved = result.nodes.get(query.node_id)
            return [resolved] if resolved is not None else []

        if isinstance(query, str):
            return [
                node
                for node in result.nodes.values()
                if query in {
                    node.name,
                    node.display_name,
                    node.calculator_type,
                    node.calculator_class_path,
                    node.semantic_calculator_class_path,
                    ResultQuery._short_class_name(node.calculator_class_path),
                    ResultQuery._short_class_name(node.semantic_calculator_class_path),
                }
            ]

        if isinstance(query, type):
            class_path = ResultQuery._class_path(query)
            class_name = query.__name__
            return [
                node
                for node in result.nodes.values()
                if class_path in {node.semantic_calculator_class_path, node.calculator_class_path}
                or node.calculator_type == class_name
            ]

        cache_key_factory = getattr(query, "cache_key", None)
        if callable(cache_key_factory):
            try:
                cache_key = cache_key_factory()
            except TypeError:
                cache_key = None
            if isinstance(cache_key, tuple):
                return [node for node in result.nodes.values() if node.signature == cache_key]

        signature_factory = getattr(query, "signature", None)
        if callable(signature_factory):
            try:
                signature = signature_factory()
            except TypeError:
                signature = None
            if isinstance(signature, tuple):
                return [node for node in result.nodes.values() if node.signature == signature]

        if callable(query):
            return [node for node in result.nodes.values() if bool(query(node))]

        raise TypeError(
            "query must be a ResultNode, string, calculator class, "
            "calculator instance, CalculatorSignature, or predicate"
        )

    @staticmethod
    def resolve_node(result: Result[Any], node: str | ResultNode) -> ResultNode:
        if isinstance(node, ResultNode):
            return node
        if node in result.nodes:
            return result.nodes[node]
        if node in result.named:
            return result.named[node]
        raise KeyError(node)

    @staticmethod
    def parents_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        if resolved.parent_ids:
            return [result.nodes[parent_id] for parent_id in resolved.parent_ids if parent_id in result.nodes]
        return ResultQuery._reverse_parents(result, resolved.node_id)

    @staticmethod
    def parent_of(result: Result[Any], node: str | ResultNode) -> ResultNode | None:
        resolved = ResultQuery.resolve_node(result, node)
        parents = ResultQuery.parents_of(result, resolved)
        if not parents:
            return None
        if len(parents) != 1:
            raise ValueError(
                f"Node {resolved.node_id!r} has {len(parents)} parents; "
                "use parents_of() for shared dependencies."
            )
        return parents[0]

    @staticmethod
    def children_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        return [result.nodes[node_id] for node_id in resolved.children if node_id in result.nodes]

    @staticmethod
    def ancestors_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        out: list[ResultNode] = []
        seen: set[str] = set()
        stack = list(reversed(ResultQuery.parents_of(result, resolved)))

        while stack:
            current = stack.pop()
            if current.node_id in seen:
                continue
            seen.add(current.node_id)
            out.append(current)
            stack.extend(reversed(ResultQuery.parents_of(result, current)))

        return out

    @staticmethod
    def descendants_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        out: list[ResultNode] = []
        seen: set[str] = set()
        stack = list(reversed(ResultQuery.children_of(result, resolved)))

        while stack:
            current = stack.pop()
            if current.node_id in seen:
                continue
            seen.add(current.node_id)
            out.append(current)
            stack.extend(reversed(ResultQuery.children_of(result, current)))

        return out

    @staticmethod
    def display_children_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        children = ResultQuery.children_of(result, resolved)

        if (
            resolved.calculator_type == "BoundCalculator"
            and children
            and children[0].label == resolved.label
            and children[0].kind == resolved.kind
        ):
            base_node = children[0]
            return ResultQuery.children_of(result, base_node) + children[1:]

        return children

    @staticmethod
    def phases_of(result: Result[Any], node: str | ResultNode) -> list[PhaseRecord]:
        return list(ResultQuery.resolve_node(result, node).phases)

    @staticmethod
    def walk_depth_first(result: Result[Any]) -> list[ResultNode]:
        out: list[ResultNode] = []

        def visit(current: ResultNode) -> None:
            out.append(current)
            for child in ResultQuery.children_of(result, current):
                visit(child)

        visit(result.root)
        return out

    @staticmethod
    def find_by_kind(result: Result[Any], kind: str) -> list[ResultNode]:
        return [node for node in result.nodes.values() if str(node.kind) == kind]

    @staticmethod
    def find_error_nodes(result: Result[Any]) -> list[ResultNode]:
        return [node for node in result.nodes.values() if node.error is not None]

    @staticmethod
    def describe_node(result: Result[Any], node: str | ResultNode) -> str:
        resolved = ResultQuery.resolve_node(result, node)
        lines = [
            f"node_id: {resolved.node_id}",
            f"name: {resolved.name}",
            f"kind: {resolved.kind}",
            f"status: {resolved.status}",
            f"calculator_type: {resolved.calculator_type}",
            f"record_policy: {resolved.record_policy}",
            f"stored_value: {resolved.stored_value}",
            f"stored_raw: {resolved.stored_raw}",
            f"parents: {len(ResultQuery.parents_of(result, resolved))}",
            f"children: {len(resolved.children)}",
            f"phases: {len(resolved.phases)}",
        ]
        if resolved.semantic_calculator_class_path is not None:
            lines.append(f"semantic_class: {resolved.semantic_calculator_class_path}")
        if resolved.value_summary is not None:
            lines.append(f"value_type: {resolved.value_summary.python_type}")
            if resolved.value_summary.preview:
                lines.append(f"preview: {resolved.value_summary.preview}")
        if resolved.error is not None:
            lines.append(f"error: {resolved.error.error_type}: {resolved.error.message}")
        if resolved.observation is not None:
            lines.append(f"observer_events: {resolved.observation.event_count}")
            lines.append(f"observer_reads: {len(resolved.observation.reads)}")
            lines.append(f"observer_dirty: {len(resolved.observation.dirty_fields)}")
            lines.append(f"observer_deletes: {len(resolved.observation.deletes)}")
        return "\n".join(lines)

    @staticmethod
    def node_label(
        node: ResultNode,
        *,
        show_ids: bool = False,
        show_ref: bool = False,
        show_kind: bool = True,
        max_width: int | None = None,
    ) -> str:
        label = node.label
        if show_kind:
            label = f"{label}<{node.kind}>"
        if show_ref:
            label = f"[{node.ref}] {label}"
        if show_ids:
            label = f"{label} [{node.node_id}]"
        if max_width is not None and len(label) > max_width:
            return label[: max_width - 3] + "..."
        return label

    @staticmethod
    def _validate_tree_limits(max_depth: int | None, max_children: int | None) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if max_children is not None and max_children < 0:
            raise ValueError("max_children must be non-negative or None")

    @staticmethod
    def _subtree_nodes(result: Result[Any], roots: list[ResultNode]) -> list[ResultNode]:
        out: list[ResultNode] = []
        seen: set[str] = set()
        stack = list(reversed(roots))
        while stack:
            node = stack.pop()
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            out.append(node)
            stack.extend(reversed(ResultQuery.display_children_of(result, node)))
        return out

    @staticmethod
    def _hidden_count_label(result: Result[Any], roots: list[ResultNode]) -> str:
        count = len(ResultQuery._subtree_nodes(result, roots))
        suffix = "node" if count == 1 else "nodes"
        return f"... {count} {suffix} hidden"

    @staticmethod
    def _visible_tree_nodes(
        result: Result[Any],
        start: ResultNode,
        *,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> tuple[list[ResultNode], int]:
        visible: list[ResultNode] = []
        hidden_ids: set[str] = set()
        visible_ids: set[str] = set()

        def hide(roots: list[ResultNode]) -> None:
            for node in ResultQuery._subtree_nodes(result, roots):
                if node.node_id not in visible_ids:
                    hidden_ids.add(node.node_id)

        def visit(node: ResultNode, depth: int) -> None:
            visible.append(node)
            visible_ids.add(node.node_id)
            children = ResultQuery.display_children_of(result, node)
            if max_depth is not None and depth >= max_depth and children:
                hide(children)
                return

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for child in visible_children:
                visit(child, depth + 1)
            hide(hidden_children)

        visit(start, 0)
        return visible, len(hidden_ids)

    @staticmethod
    def _node_elapsed_s(node: ResultNode) -> float | None:
        values = [phase.elapsed_s for phase in node.phases if phase.elapsed_s is not None]
        if not values:
            return None
        return sum(values)

    @staticmethod
    def _cache_events_by_node(result: Result[Any]) -> dict[str, dict[str, int]]:
        grouped: dict[str, dict[str, int]] = {}
        for event in result.cache_events():
            node_id = getattr(event, "node_id", None)
            if not node_id:
                continue
            node_events = grouped.setdefault(node_id, {})
            event_name = str(getattr(event, "event", ""))
            node_events[event_name] = node_events.get(event_name, 0) + 1
        return grouped

    @staticmethod
    def _cache_suffix(
        node: ResultNode,
        cache_events: dict[str, dict[str, int]],
        *,
        hit_occurrence: bool = False,
    ) -> str:
        if hit_occurrence:
            return "hit"

        events = cache_events.get(node.node_id, {})
        parts: list[str] = []
        hits = events.get("hit", 0)
        stores = events.get("store", 0)
        if stores:
            parts.append("store" if stores == 1 else f"{stores} stores")
            if hits:
                parts.append(f"nhit={hits}")
        elif hits:
            parts.append("hit" if hits == 1 else f"{hits} hits")
        return "; ".join(parts)

    @staticmethod
    def _format_field_set(values: set[str], *, max_items: int = 4) -> str:
        if not values:
            return ""
        items = sorted(values)
        if len(items) <= max_items:
            return ",".join(items)
        hidden = len(items) - max_items
        return f"{','.join(items[:max_items])},+{hidden}"

    @staticmethod
    def _value_suffix(node: ResultNode) -> str:
        if node.value_summary is None:
            return ""
        summary = node.value_summary
        parts = [summary.python_type]
        if summary.shape is not None:
            parts.append(f"shape={summary.shape!r}")
        if summary.units is not None:
            parts.append(f"units={summary.units}")
        return " ".join(parts)

    @staticmethod
    def _execution_label(
        result: Result[Any],
        node: ResultNode,
        *,
        show_ids: bool,
        include_perf: bool,
        include_cache: bool,
        include_observer: bool,
        include_values: bool,
        cache_events: dict[str, dict[str, int]],
        hit_occurrence: bool = False,
    ) -> str:
        label = ResultQuery.node_label(node, show_ids=show_ids)
        parts: list[str] = []

        if hit_occurrence:
            return f"{label}  [cache=hit]"

        if node.error is not None:
            parts.append(f"error={node.error.error_type}")
        elif node.status != NodeStatus.OK:
            parts.append(f"status={display_value(node.status)}")

        if include_perf:
            elapsed = ResultQuery._node_elapsed_s(node)
            if elapsed is not None:
                parts.append(format_time(elapsed))

        if include_cache:
            cache_text = ResultQuery._cache_suffix(node, cache_events)
            if cache_text:
                parts.append(f"cache={cache_text}")

        if include_observer:
            access_text = format_observation_access(
                result.observation_of(node),
                read_items=3,
                dirty_items=3,
                delete_items=2,
            )
            if access_text:
                parts.append(access_text)

        if include_values:
            value_text = ResultQuery._value_suffix(node)
            if value_text:
                parts.append(f"value={value_text}")

        if not parts:
            return label
        return f"{label}  [{'; '.join(parts)}]"

    @staticmethod
    def _hidden_execution_summary(
        result: Result[Any],
        roots: list[ResultNode],
        *,
        include_perf: bool,
        include_cache: bool,
        include_observer: bool,
        cache_events: dict[str, dict[str, int]],
    ) -> str:
        nodes = ResultQuery._subtree_nodes(result, roots)
        suffix = "node" if len(nodes) == 1 else "nodes"
        parts = [f"... {len(nodes)} {suffix} hidden"]

        if include_perf:
            elapsed_values = [ResultQuery._node_elapsed_s(node) for node in nodes]
            elapsed = sum(value for value in elapsed_values if value is not None)
            if elapsed:
                parts.append(format_time(elapsed))

        if include_cache:
            hits = sum(cache_events.get(node.node_id, {}).get("hit", 0) for node in nodes)
            stores = sum(cache_events.get(node.node_id, {}).get("store", 0) for node in nodes)
            cache_parts: list[str] = []
            if stores:
                cache_parts.append(f"{stores} store")
            if hits:
                cache_parts.append(f"nhit={hits}")
            if cache_parts:
                parts.append(f"cache={','.join(cache_parts)}")

        if include_observer:
            reads: set[str] = set()
            dirty_fields: set[str] = set()
            deletes: set[str] = set()
            for node in nodes:
                observation = result.observation_of(node)
                if observation is None:
                    continue
                reads.update(observation.reads)
                dirty_fields.update(observation.dirty_fields)
                deletes.update(observation.deletes)
            read_text = ResultQuery._format_field_set(reads, max_items=3)
            dirty_text = ResultQuery._format_field_set(dirty_fields, max_items=3)
            delete_text = ResultQuery._format_field_set(deletes, max_items=2)
            if read_text:
                parts.append(f"read={read_text}")
            if dirty_text:
                parts.append(f"dirty={dirty_text}")
            if delete_text:
                parts.append(f"del={delete_text}")

        return "  ".join(parts)

    @staticmethod
    def node_tree(
        result: Result[Any],
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        ResultQuery._validate_tree_limits(max_depth, max_children)
        start = result.root if node is None else ResultQuery.resolve_node(result, node)

        def render(current: ResultNode, prefix: str, is_last: bool, depth: int) -> list[str]:
            branch = "└─" if is_last else "├─"
            lines = [f"{prefix}{branch} {ResultQuery.node_label(current, show_ids=show_ids)}"]
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = ResultQuery.display_children_of(result, current)
            if max_depth is not None and depth >= max_depth and children:
                lines.append(f"{child_prefix}└─ {ResultQuery._hidden_count_label(result, children)}")
                return lines

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, child_prefix, is_child_last, depth + 1))
            if hidden_children:
                lines.append(f"{child_prefix}└─ {ResultQuery._hidden_count_label(result, hidden_children)}")
            return lines

        lines = [ResultQuery.node_label(start, show_ids=show_ids)]
        children = ResultQuery.display_children_of(result, start)
        if max_depth == 0 and children:
            lines.append(f"└─ {ResultQuery._hidden_count_label(result, children)}")
        else:
            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, "", is_child_last, 1))
            if hidden_children:
                lines.append(f"└─ {ResultQuery._hidden_count_label(result, hidden_children)}")
        return "\n".join(lines)

    @staticmethod
    def execution_tree(
        result: Result[Any],
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
        include_perf: bool = True,
        include_cache: bool = True,
        include_observer: bool = True,
        include_values: bool = False,
    ) -> str:
        ResultQuery._validate_tree_limits(max_depth, max_children)
        start = result.root if node is None else ResultQuery.resolve_node(result, node)
        cache_events = ResultQuery._cache_events_by_node(result)
        rendered_nodes: set[str] = set()

        def label(current: ResultNode, *, hit_occurrence: bool = False) -> str:
            return ResultQuery._execution_label(
                result,
                current,
                show_ids=show_ids,
                include_perf=include_perf,
                include_cache=include_cache,
                include_observer=include_observer,
                include_values=include_values,
                cache_events=cache_events,
                hit_occurrence=hit_occurrence,
            )

        def hidden_summary(children: list[ResultNode]) -> str:
            return ResultQuery._hidden_execution_summary(
                result,
                children,
                include_perf=include_perf,
                include_cache=include_cache,
                include_observer=include_observer,
                cache_events=cache_events,
            )

        def render(current: ResultNode, prefix: str, is_last: bool, depth: int) -> list[str]:
            branch = "└─" if is_last else "├─"
            hit_occurrence = current.node_id in rendered_nodes
            lines = [f"{prefix}{branch} {label(current, hit_occurrence=hit_occurrence)}"]
            if hit_occurrence:
                return lines
            rendered_nodes.add(current.node_id)
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = ResultQuery.display_children_of(result, current)
            if max_depth is not None and depth >= max_depth and children:
                lines.append(f"{child_prefix}└─ {hidden_summary(children)}")
                return lines

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, child_prefix, is_child_last, depth + 1))
            if hidden_children:
                lines.append(f"{child_prefix}└─ {hidden_summary(hidden_children)}")
            return lines

        lines = [label(start)]
        rendered_nodes.add(start.node_id)
        children = ResultQuery.display_children_of(result, start)
        if max_depth == 0 and children:
            lines.append(f"└─ {hidden_summary(children)}")
        else:
            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, "", is_child_last, 1))
            if hidden_children:
                lines.append(f"└─ {hidden_summary(hidden_children)}")
        return "\n".join(lines)
