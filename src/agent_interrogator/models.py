"""Data models for representing agent capabilities and functions."""

import hashlib
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(value: Optional[str]) -> str:
    """Canonicalize a free-text name into a stable form for identity hashing.

    Lowercases, replaces runs of non-alphanumerics with a single underscore,
    and strips leading/trailing underscores. ``"Web Search!"`` and
    ``"web-search"`` both normalize to ``"web_search"``.
    """
    if not value:
        return ""
    return _NORMALIZE_RE.sub("_", value.lower().strip()).strip("_")


def function_signature(
    name: str, parameters: List["Parameter"], return_type: Optional[str]
) -> str:
    """Stable signature string used to derive a function's content-addressed id."""
    norm_name = normalize_name(name)
    param_sig = ",".join(
        sorted(f"{normalize_name(p.name)}:{normalize_name(p.type)}" for p in parameters)
    )
    return f"{norm_name}|{param_sig}|{normalize_name(return_type or '')}"


def _hash_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


class Parameter(BaseModel):
    """Represents a parameter for a function."""

    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None


class Function(BaseModel):
    """Represents a function within a capability."""

    node_id: str = Field(
        default="",
        description="Stable content-addressed id (sha1 of normalized name+signature). "
        "Computed automatically from name/parameters/return_type if left empty.",
    )
    name: str
    description: Optional[str] = None
    parameters: List[Parameter] = Field(default_factory=list)
    return_type: Optional[str] = None

    @model_validator(mode="after")
    def _ensure_node_id(self) -> "Function":
        if not self.node_id:
            self.node_id = _hash_id(
                "fn", function_signature(self.name, self.parameters, self.return_type)
            )
        return self


class Capability(BaseModel):
    """Represents a capability of the agent."""

    node_id: str = Field(
        default="",
        description="Stable content-addressed id (sha1 of normalized name). "
        "Computed automatically from name if left empty.",
    )
    name: str
    description: Optional[str] = None
    functions: List[Function] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_node_id(self) -> "Capability":
        if not self.node_id:
            self.node_id = _hash_id("cap", normalize_name(self.name))
        return self


class AgentProfile(BaseModel):
    """Complete profile of an agent's capabilities."""

    capabilities: List[Capability] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
