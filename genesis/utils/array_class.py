"""Compatibility names for task kernels written before declarative solver data.

These are aliases of the upstream data types, not separate legacy layouts.
New code should import the types from their solver's data module. Allocation
is handled by the declarative Data API; the former get_* allocation functions
are intentionally not reproduced here.
"""

from genesis.engine.bvh import BVHState, BVHTreeState
from genesis.engine.solvers.rigid.collider.data import ColliderInfo, ColliderState
from genesis.engine.solvers.rigid.constraint.data import ConstraintState
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidConfig, RigidErrorCode, RigidInfo

ErrorCode = RigidErrorCode

__all__ = [
    "BVHState",
    "BVHTreeState",
    "ColliderInfo",
    "ColliderState",
    "ConstraintState",
    "DynInfo",
    "DynState",
    "ErrorCode",
    "RigidConfig",
    "RigidInfo",
]
