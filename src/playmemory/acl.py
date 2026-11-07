"""Tenant isolation and role/group access control, enforced in Memory's single
visibility gate so every read path (search, spans, neighbours, get, seek, window)
honours it.

Reserved frame meta keys:
  tenant      str       frames belong to this tenant (required for isolation)
  visibility  str       "public" (default) or "restricted"
  roles       [str]     restricted frames: readable by principals holding any of these roles
  groups      [str]     ... or any of these groups
A Principal with no tenant sees only untenanted frames if allow_untenanted is true.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TENANT, VISIBILITY, ROLES, GROUPS = "tenant", "visibility", "roles", "groups"


@dataclass(frozen=True)
class Principal:
    tenant: str | None = None
    roles: frozenset = field(default_factory=frozenset)
    groups: frozenset = field(default_factory=frozenset)
    allow_untenanted: bool = False  # may this principal read frames that carry no tenant?

    def __post_init__(self):
        object.__setattr__(self, "roles", frozenset(self.roles))
        object.__setattr__(self, "groups", frozenset(self.groups))


def allowed(p: Principal, meta: dict) -> bool:
    """Single-frame check (the Memory mask is a vectorised form of the same rule)."""
    tenant = meta.get(TENANT)
    if tenant is None:
        if not p.allow_untenanted:
            return False
    elif tenant != p.tenant:
        return False
    if meta.get(VISIBILITY, "public") == "restricted":
        return bool(p.roles & set(meta.get(ROLES) or ())) or bool(p.groups & set(meta.get(GROUPS) or ()))
    return True
