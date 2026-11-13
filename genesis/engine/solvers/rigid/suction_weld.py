"""Registered suction cups backed by the rigid solver's device kernels."""

import math

import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.state_change import mutates
from genesis.grad.bridge import custom_backward

from .constraint.weld import (
    kernel_apply_suction_ghost_spring,
    kernel_cast_suction,
    kernel_capture_suction,
    kernel_set_suction_coefficients,
    kernel_set_attached_suction_coefficients,
    kernel_get_suction_welded,
    kernel_release_suction,
    kernel_request_suction_weld,
)


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

    def _build(self, batch_size):
        kernel_set_suction_coefficients(self._link_idx, self._solver._scene._envs_idx,
            self._stiffness, self._damping, self._max_wrench, self._solver.dyn_state)

    def _apply_ghost_spring(self, trees):
        if not trees:
            return
        envs = self._solver._scene._envs_idx
        self._cast(envs, trees)
        kernel_apply_suction_ghost_spring(
            self._link_idx, self._max_cast_distance, envs,
            self._solver.dyn_state, self._solver.constraint_solver.constraint_state,
            self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config,
        )

    def _cast(self, envs, trees):
        first, second = trees[0], trees[-1]
        kernel_cast_suction(envs, self._link_idx, self._max_cast_distance,
            first.bvh_state.tree, first.env_bvh_idx, second.bvh_state.tree, second.env_bvh_idx,
            len(trees) == 2, self._solver.dyn_state, self._solver.dyn_info,
            self._solver.constraint_solver.constraint_state, self._solver.rigid_config)

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
        # BVH construction and target selection are discrete topology operations.
        with custom_backward(None):
            if context.is_active:
                context.update()
            else:
                context.activate()
        trees = [entry for entry in context.collision_bvh_contexts if entry.solver is solver]
        if len(trees) > 2:
            raise RuntimeError("Unexpected number of rigid collision BVHs")
        return trees

    def set_parameters(self, stiffness=None, damping=None, max_wrench=None, envs_idx=None):
        """Set physical coefficients, including active welds; tensors retain gradients.

        Call inside each recorded rollout to optimize coefficients. Break limits
        have no derivative through the discrete break decision.
        """
        values = []
        for name, value, previous in (("stiffness", stiffness, self._stiffness),
                                       ("damping", damping, self._damping),
                                       ("max_wrench", max_wrench, self._max_wrench)):
            value = previous if value is None else torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)
            if value.ndim == 0:
                value = value.expand(6)
            if value.shape != (6,):
                raise ValueError(f"{name} must be scalar or have shape (6,)")
            valid = (value >= 0) & (~torch.isnan(value) if name == "max_wrench" else torch.isfinite(value))
            torch._assert_async(valid.all(), f"Invalid {name}")
            values.append(value.contiguous())
        self._solver._set_suction_parameters(self, *values, envs_idx)

    def request_suction_cup_welding(self, envs_idx=None):
        """Attach to the nearest ray hit; keep its discrete identity fixed in backward."""
        return self._solver._request_suction(self, envs_idx)

    def _request(self, envs_idx=None):
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
        self._cast(envs, trees)
        self._solver.constraint_solver._eq_const_info_cache.clear()
        kernel_request_suction_weld(
            envs, links,
            self._solver.dyn_state, self._solver.constraint_solver.constraint_state,
            self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config, self._solver._errno,
        )

        kernel_capture_suction(envs, self._link_idx, self._max_cast_distance,
                               self._solver.dyn_state, self._solver.dyn_info)

    def release_suction_cup(self, envs_idx=None):
        """Release selected cups, preserving gradients through surviving records."""
        return self._solver._release_suction(self, envs_idx)

    def _release(self, envs_idx=None):
        """Release active or broken welds for the selected environments."""
        envs, links = self._inputs(envs_idx, mutate=True)
        if not envs.numel():
            return
        self._solver.constraint_solver._eq_const_info_cache.clear()
        kernel_release_suction(envs, links, self._solver.constraint_solver.constraint_state,
                               self._solver.dyn_info, self._solver.rigid_info)

    def get_welded(self, envs_idx=None):
        """Return active attachment flags [N], including newly created welds."""
        envs, links = self._inputs(envs_idx)
        out = torch.zeros(envs.numel(), device=gs.device, dtype=torch.bool)
        if envs.numel():
            kernel_get_suction_welded(envs, links, out, self._solver.constraint_solver.constraint_state,
                                      self._solver.dyn_info, self._solver.rigid_info)
        return out

    def get_wrench(self, envs_idx=None):
        """Latest cup-frame force (N) and direct torque (N m), shape [N, 6].

        An unwelded cup reports its last ghost-spring force and zero torque;
        a welded cup reports its last solved weld wrench. An absent or unsolved
        force is zero. Wrenches are computed during a physics substep.
        """
        solver = self._solver
        envs, _ = self._inputs(envs_idx)
        info = solver.dyn_info.equalities
        # Fixed-size reduction, without packing or CPU readback.
        kinds, first, second, forces, had_rows = solver._read_data(
            (info.eq_type, info.eq_obj1id, info.eq_obj2id, info.soft_weld_force, info.soft_weld_had_rows),
            envs, None, is_framed=False)
        counts = solver._read_data(solver.constraint_solver.constraint_state.qd_n_equalities,
                                   None, None, is_framed=False, is_batched=False)[envs]
        slots = torch.arange(kinds.shape[1], device=gs.device)
        valid = slots[None, :] < counts[:, None]
        active = valid & (kinds == int(gs.EQUALITY_TYPE.SOFT_WELD)) & ((first == self._link_idx) | (second == self._link_idx))
        sign = torch.where(first == self._link_idx, 1.0, -1.0)
        wrench = (forces * (active * had_rows * sign)[..., None]).sum(dim=1)
        ghost = solver._read_data(solver.dyn_state.links.suction_ghost_force, envs, self._link_idx, is_framed=False)[:, 0]
        force = torch.where(active.any(dim=1)[:, None], wrench[:, :3], ghost)
        quat = solver._read_data(solver.dyn_state.links.quat, envs, self._link_idx, is_framed=False)[:, 0]
        return torch.cat((gu.inv_transform_by_quat(force, quat), gu.inv_transform_by_quat(wrench[:, 3:], quat)), dim=-1)



class SuctionWeldMixin:
    @mutates()
    def _set_suction_parameters(self, cup, stiffness, damping, max_wrench, envs_idx=None):
        envs, _ = cup._inputs(envs_idx, mutate=True)
        self.constraint_solver._eq_const_info_cache.clear()
        kernel_set_suction_coefficients(cup._link_idx, envs, stiffness, damping, max_wrench, self.dyn_state)
        kernel_set_attached_suction_coefficients(cup._link_idx, envs, stiffness, damping, max_wrench,
                                                self.dyn_info, self.constraint_solver.constraint_state)

    @mutates()
    def _request_suction(self, cup, envs_idx=None):
        cup._request(envs_idx)

    @mutates()
    def _release_suction(self, cup, envs_idx=None):
        cup._release(envs_idx)

    @gs.assert_unbuilt
    def add_suction_cup(self, link_idx, stiffness, damping, max_wrench, max_cast_distance):
        """Register a collision-free frame and return its batched suction handle.

        Stiffness, damping and maximum wrench are scalar or six-component
        vectors in force XYZ / torque XYZ order. Distances are in meters.
        Stiffness/damping use physical linear and angular units; max_wrench
        uses per-world-axis N and N m limits. Registration must precede build.
        """
        if self._disable_constraint or not self._enable_collision:
            raise RuntimeError("Suction requires rigid collision solving")
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
