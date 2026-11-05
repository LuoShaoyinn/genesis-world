"""Registered suction cups backed by the rigid solver's device kernels."""

import math

import torch

import genesis as gs

from .constraint.weld import kernel_get_suction_wrench, kernel_request_suction_weld, kernel_release_suction, kernel_suction_normal


class SuctionCup:
    """A suction frame shared by all environments of one scene.

    Instances are created by ``rigid_solver.add_suction_cup`` before build.
    All returned tensors keep an environment dimension, including when the scene
    is not parallelized. Runtime methods accept the usual Genesis ``envs_idx``.
    """

    def __init__(self, solver, link_idx, stiffness, damping, max_wrench, max_cast_distance):
        self._solver = solver
        self._link_idx = link_idx
        self._stiffness, self._damping, self._max_wrench = stiffness, damping, max_wrench
        self._max_cast_distance = max_cast_distance

    def _inputs(self, envs_idx, *, mutate=False):
        if not self._solver._scene.is_built:
            raise RuntimeError("Build the scene before using suction cups")
        envs = self._solver._scene._sanitize_envs_idx(envs_idx).to(dtype=gs.tc_int).contiguous()
        # A duplicate would make multiple kernel threads mutate the same weld list.
        if mutate and envs.numel() > 1:
            ordered = torch.sort(envs).values
            torch._assert_async((ordered[1:] != ordered[:-1]).all(), "Duplicate environment indices")
        links = torch.full_like(envs, self._link_idx)
        return envs, links

    def _trees(self):
        from genesis.engine.sensors.raycaster import RaycastContext

        solver = self._solver
        solver.update_forward_pos()
        contexts = solver.sim._sensor_manager._shared_contexts
        context = contexts.get(RaycastContext)
        if context is None:
            context = contexts[RaycastContext] = RaycastContext(solver.sim)
        if context.is_active:
            context.update()
        else:
            context.activate()
        trees = [entry for entry in context.collision_bvh_contexts if entry.solver is solver]
        if len(trees) > 2:
            raise RuntimeError("Unexpected number of rigid collision BVHs")
        return trees

    def request_suction_cup_welding(self, envs_idx=None):
        """Raycast and create or re-arm a weld; an existing active weld is unchanged.

        The ray starts half the configured distance behind the cup origin and
        follows cup-local +Z over ``max_cast_distance``. It ignores the cup's
        own entity and attaches to the nearest collision surface. A miss does
        nothing. Wrench is available after a physics substep.
        """
        envs, links = self._inputs(envs_idx, mutate=True)
        if not envs.numel():
            return
        trees = self._trees()
        if not trees:
            return
        first, second = trees[0], trees[-1]
        count = envs.numel()
        arrays = [value.expand(count, 6).contiguous() for value in (self._stiffness, self._damping, self._max_wrench)]
        self._solver.constraint_solver._eq_const_info_cache.clear()
        kernel_request_suction_weld(
            envs, links, *arrays, self._max_cast_distance,
            first.bvh_state.tree, first.env_bvh_idx, second.bvh_state.tree, second.env_bvh_idx, len(trees) == 2,
            self._solver.dyn_state, self._solver.constraint_solver.constraint_state,
            self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config, self._solver._errno,
        )

    def release_suction_cup(self, envs_idx=None):
        """Release active or broken welds for the selected environments."""
        envs, links = self._inputs(envs_idx, mutate=True)
        if not envs.numel():
            return
        self._solver.constraint_solver._eq_const_info_cache.clear()
        kernel_release_suction(envs, links, self._solver.constraint_solver.constraint_state,
                               self._solver.dyn_info, self._solver.rigid_info)

    def get_wrench(self, envs_idx=None):
        """World-frame force (N) and direct torque (N m), shape [N, 6].

        Returns +inf for absent, broken, or not-yet-solved welds. Values are
        the most recently solved substep's wrench on this suction frame.
        """
        envs, links = self._inputs(envs_idx)
        out = torch.full((envs.numel(), 6), float("inf"), device=gs.device, dtype=gs.tc_float)
        if envs.numel():
            kernel_get_suction_wrench(envs, links, out, self._solver.constraint_solver.constraint_state,
                                      self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config)
        return out

    def get_normal_proj(self, envs_idx=None):
        """Fresh raycast normal in the cup's local frame, shape [N, 3].

        The surface normal points against the cast direction. A miss returns
        [0, 0, 0]. The ray and range are identical to the welding request.
        """
        envs, links = self._inputs(envs_idx)
        out = torch.zeros((envs.numel(), 3), device=gs.device, dtype=gs.tc_float)
        if not envs.numel():
            return out
        trees = self._trees()
        if trees:
            first, second = trees[0], trees[-1]
            kernel_suction_normal(
                envs, links, self._max_cast_distance, out,
                first.bvh_state.tree, first.env_bvh_idx, second.bvh_state.tree, second.env_bvh_idx,
                len(trees) == 2, self._solver.dyn_state, self._solver.dyn_info, self._solver.rigid_config,
            )
        return out


class SuctionWeldMixin:
    @gs.assert_unbuilt
    def add_suction_cup(self, link_idx, stiffness, damping, max_wrench, max_cast_distance):
        """Register a collision-free frame and return its batched suction handle.

        Stiffness, damping and maximum wrench are scalar or six-component
        vectors in force XYZ / torque XYZ order. Distances are in meters.
        Stiffness/damping use physical linear and angular units; max_wrench
        uses per-world-axis N and N m limits. Registration must precede build.
        """
        if self._requires_grad or self._disable_constraint or not self._enable_collision:
            raise RuntimeError("Suction requires non-differentiable rigid collision solving")
        from genesis.engine.couplers import IPCCoupler, SAPCoupler
        if isinstance(self.sim.coupler, (SAPCoupler, IPCCoupler)):
            raise RuntimeError("Suction is not supported by SAP or IPC coupling")
        link_idx = int(link_idx)
        if link_idx < 0 or link_idx >= self.n_links:
            raise ValueError("Invalid global suction link index")
        if self.links[link_idx].n_geoms:
            raise ValueError("Suction frame must not have collision geometry")
        if not math.isfinite(max_cast_distance) or max_cast_distance <= 0:
            raise ValueError("max_cast_distance must be positive and finite")
        cups = getattr(self, "_suction_cups", None)
        if cups is None:
            cups = self._suction_cups = []
        if any(cup._link_idx == link_idx for cup in cups):
            raise ValueError("A suction cup is already registered on this link")
        if len(cups) >= self._options.max_dynamic_constraints:
            raise ValueError("Increase max_dynamic_constraints to register more suction cups")
        values = []
        for name, value in (("stiffness", stiffness), ("damping", damping), ("max_wrench", max_wrench)):
            value = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)
            if value.ndim == 0:
                value = value.expand(6)
            if value.shape != (6,):
                raise ValueError(f"{name} must be scalar or have shape (6,)")
            valid = ~torch.isnan(value) & (value >= 0)
            if name != "max_wrench":
                valid = valid & torch.isfinite(value)
            torch._assert_async(valid.all(), f"Invalid {name}")
            values.append(value.contiguous().clone())
        cup = SuctionCup(self, link_idx, *values, float(max_cast_distance))
        cups.append(cup)
        return cup
