"""Field-level UPSERT semantics for capability and function entities.

When the iterative discovery loop re-encounters an entity it has already seen
(matched by ``node_id``), we union information from both records rather than
overwriting or duplicating. The interrogator can therefore call these as
``MERGE``-style helpers analogous to ``MERGE (n {node_id: $id}) SET n += $props``
in Cypher, which lays the groundwork for graph-backed storage.
"""

from typing import List, Optional

from .models import Capability, Function, Parameter, normalize_name


def _prefer_longer(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Pick the more informative of two optional strings.

    Falls back to whichever is non-empty, breaks ties by length.
    """
    if a and b:
        return a if len(a) >= len(b) else b
    return a or b


def merge_parameters(
    existing: List[Parameter], incoming: List[Parameter]
) -> List[Parameter]:
    """Union two parameter lists keyed by (normalized name, normalized type).

    On collision: prefer the richer description, mark required if either side
    requires it, and keep the first-seen default.
    """
    by_key: "dict[tuple[str, str], Parameter]" = {}
    order: List["tuple[str, str]"] = []
    for p in list(existing) + list(incoming):
        key = (normalize_name(p.name), normalize_name(p.type))
        if key in by_key:
            prior = by_key[key]
            by_key[key] = Parameter(
                name=prior.name,
                type=prior.type,
                description=_prefer_longer(prior.description, p.description),
                required=prior.required or p.required,
                default=prior.default if prior.default is not None else p.default,
            )
        else:
            by_key[key] = p
            order.append(key)
    return [by_key[k] for k in order]


def merge_function(existing: Function, incoming: Function) -> Function:
    """Field-level merge of two ``Function`` records that share a ``node_id``."""
    return Function(
        node_id=existing.node_id,
        name=existing.name,
        description=_prefer_longer(existing.description, incoming.description),
        parameters=merge_parameters(existing.parameters, incoming.parameters),
        return_type=_prefer_longer(existing.return_type, incoming.return_type),
    )


def merge_capability(existing: Capability, incoming: Capability) -> Capability:
    """Field-level merge of two ``Capability`` records that share a ``node_id``.

    Inner functions are themselves merged by ``node_id``; metadata is unioned
    with first-seen-wins on conflicts.
    """
    by_id: "dict[str, Function]" = {f.node_id: f for f in existing.functions}
    order: List[str] = [f.node_id for f in existing.functions]
    for f in incoming.functions:
        if f.node_id in by_id:
            by_id[f.node_id] = merge_function(by_id[f.node_id], f)
        else:
            by_id[f.node_id] = f
            order.append(f.node_id)

    merged_metadata = dict(existing.metadata)
    for k, v in incoming.metadata.items():
        if merged_metadata.get(k) in (None, "", [], {}):
            merged_metadata[k] = v

    return Capability(
        node_id=existing.node_id,
        name=existing.name,
        description=_prefer_longer(existing.description, incoming.description),
        functions=[by_id[k] for k in order],
        metadata=merged_metadata,
    )
