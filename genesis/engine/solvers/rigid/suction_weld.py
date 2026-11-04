"""Rigid-solver APIs for suction cups and breakable welds."""

import numpy as np
import torch

import genesis as gs

from .constraint.weld import kernel_get_suction_status, kernel_get_suction_wrench, kernel_request_suction_weld


class SuctionWeldMixin:
    def configure_contact_soft_welds(self, welds):
        """Configure (cup link, ground link, tip link, local anchor, gains) for substep valves."""
        from genesis.engine.couplers import SAPCoupler
        if self._requires_grad or isinstance(self.sim.coupler, SAPCoupler) or not self._enable_collision or self._disable_constraint:
            raise RuntimeError('Contact-gated soft welds require non-differentiable rigid collision solving')
        self._contact_soft_welds = tuple(welds)
        shape = (self._scene.n_envs, len(welds))
        self._contact_soft_weld_valves = torch.zeros(shape, device=gs.device, dtype=gs.tc_float)
        self._contact_soft_weld_contacts = torch.zeros(shape, device=gs.device, dtype=torch.bool)
        self._contact_soft_weld_engages = torch.zeros(shape, device=gs.device, dtype=torch.bool)

    def set_contact_soft_weld_valves(self, valves):
        """Set per-environment valve requests for the next control interval."""
        if valves.shape != self._contact_soft_weld_valves.shape:
            raise ValueError('Wrong contact-gated valve shape')
        self._contact_soft_weld_valves.copy_(valves)
        self._contact_soft_weld_contacts.zero_()
        self._contact_soft_weld_engages.zero_()

    def get_contact_soft_weld_diagnostics(self):
        return self._contact_soft_weld_contacts, self._contact_soft_weld_engages

    def add_soft_weld_constraint(
        self,
        link1_idx,
        link2_idx,
        *,
        linear_stiffness=1000.0,
        linear_damping=40.0,
        angular_stiffness=1.0,
        angular_damping=0.02,
        max_force=float("inf"),
        max_torque=float("inf"),
        anchor_pos=None,
        anchor_local=None,
        project_other_anchor_z=None,
        align_to_world_plane=False,
        anchor_quat=(1.0, 0.0, 0.0, 0.0),
        envs_idx=None,
    ):
        """Add or re-arm a breakable six-DoF weld at the current relative pose.

        ``linear_stiffness`` is in N/m and ``linear_damping`` in N s/m.
        ``angular_stiffness`` is in N m/rad and ``angular_damping`` in
        N m s/rad. The linear coefficients use an implicit spring-damper
        discretization; an isolated static load gives extension
        ``force / linear_stiffness``. Angular coefficients use the same
        implicit compliance with quaternion-row scaling corrected for the
        small-angle physical stiffness/damping. Finite-angle restoring torque
        follows the quaternion error (``stiffness * sin(angle)`` for pure tilt).
        ``max_force`` (N) and ``max_torque`` (N m) are per-world-axis limits.
        ``anchor_pos`` is the shared attachment point in world coordinates at
        creation time. If omitted, the lower-index link's origin is used.
        ``anchor_local`` gives the attachment point in ``link1_idx``'s local
        coordinates, allowing a different world anchor per environment.
        ``project_other_anchor_z`` places the other link's anchor at the same
        world x/y and the specified world z, e.g. on a ground plane.
        ``align_to_world_plane=True`` saves the nearest yaw-only orientation
        of the cup frame with +Z along world +Z, rather than its initial tilt.
        ``anchor_quat`` is the cup frame's local WXYZ orientation in
        ``link1_idx``. Both anchor and target orientation are captured once
        and then follow the other link; use a fixed link for a world plane.
        This requires ``anchor_local`` and ``project_other_anchor_z``.
        An exceeded limit breaks that environment's weld for the next physics
        substep. The same call re-arms a broken pair and captures its new pose.
        """
        return self.constraint_solver.add_soft_weld_constraint(
            link1_idx,
            link2_idx,
            linear_stiffness=linear_stiffness,
            linear_damping=linear_damping,
            angular_stiffness=angular_stiffness,
            angular_damping=angular_damping,
            max_force=max_force,
            max_torque=max_torque,
            anchor_pos=anchor_pos,
            anchor_local=anchor_local,
            project_other_anchor_z=project_other_anchor_z,
            align_to_world_plane=align_to_world_plane,
            anchor_quat=anchor_quat,
            envs_idx=envs_idx,
        )

    def _suction_request_inputs(self, envs_idx, link_idx):
        envs = torch.as_tensor(envs_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        links = torch.as_tensor(link_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        if envs.ndim != 1 or links.shape != envs.shape:
            raise ValueError('envs_idx and link_idx must be one-dimensional and have equal length')
        torch._assert_async(((envs >= 0) & (envs < self._scene.n_envs)).all(), 'Invalid environment index')
        torch._assert_async(((links >= 0) & (links < self.n_links)).all(), 'Invalid link index')
        return envs, links

    def request_suction_cup_welding(self, envs_idx, link_idx, stiffness, damping, max_wrench, max_cast_distance=0.01):
        """Request scene-surface welds; each [N,6] row is force XYZ then torque XYZ.

        Cast from suction-frame local (0, 0, -max_cast_distance / 2) along local +Z.
        ``link_idx`` must identify the collision-free suction frame, not its parent.
        The nearest collision surface outside the frame's own entity is used.
        A ray miss leaves the weld absent. The distance is in meters.

        Wrenches become available after a physics substep. Existing active pairs
        remain unchanged; broken pairs are re-armed at their current poses.
        """
        if self._requires_grad:
            raise RuntimeError('Dynamic suction welds do not support differentiable simulation')
        if not np.isfinite(max_cast_distance) or max_cast_distance <= 0:
            raise ValueError('max_cast_distance must be positive and finite')
        envs, links = self._suction_request_inputs(envs_idx, link_idx)
        if not hasattr(self, '_suction_collision_free_links'):
            self._suction_collision_free_links = torch.tensor(
                [link.n_geoms == 0 for link in self.links], dtype=torch.bool, device=gs.device
            )
        torch._assert_async(
            self._suction_collision_free_links[links.long()].all(),
            'Suction frame must not have collision geometry',
        )
        arrays = []
        for name, value in (('stiffness', stiffness), ('damping', damping), ('max_wrench', max_wrench)):
            value = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device).contiguous()
            if value.shape != (envs.numel(), 6):
                raise ValueError(f'{name} must have shape (N, 6)')
            torch._assert_async((~torch.isnan(value) & (value >= 0)).all(), f'Invalid {name}')
            arrays.append(value)
        # Reuse Genesis's shared sensor raycast BVH where available. If no
        # raycasting sensor exists, lazily create the same context on demand;
        # its dynamic geometry is refreshed when requests arrive, not per step.
        from genesis.engine.sensors.raycaster import RaycastContext
        self.update_forward_pos()
        context = self.sim._sensor_manager._shared_contexts.get(RaycastContext)
        if context is None:
            context = getattr(self, '_suction_raycast_context', None)
            if context is None:
                context = self._suction_raycast_context = RaycastContext(self.sim)
        if context.is_active:
            context.update()
        else:
            context.activate()
        trees = [entry for entry in context.collision_bvh_contexts if entry.solver is self]
        if not trees:
            return
        if len(trees) > 2:
            raise RuntimeError('Unexpected number of rigid collision BVHs')
        first = trees[0]
        second = trees[1] if len(trees) == 2 else first
        # Group by environment on-device; each solver thread owns its env's
        # dynamic constraint list and processes its requests in original order.
        order = torch.argsort(envs, stable=True)
        sorted_envs = envs[order].contiguous()
        boundaries = torch.arange(self._scene.n_envs + 1, device=envs.device, dtype=envs.dtype)
        offsets = torch.searchsorted(sorted_envs, boundaries, out_int32=True)
        self.constraint_solver._eq_const_info_cache.clear()
        control = getattr(self, '_suction_valve_call', None)
        valves, previous = control if control is not None else (links, links)
        if control is not None:
            hits, engages, points, normals = self._suction_valve_diagnostics
        else:
            hits = engages = links
            points = normals = arrays[0]
        kernel_request_suction_weld(envs, links, order, offsets, *arrays, float(max_cast_distance),
            valves, previous, control is not None,
            hits, engages, points, normals,
            first.bvh_state.tree, first.env_bvh_idx, second.bvh_state.tree, second.env_bvh_idx, len(trees) == 2, self.dyn_state,
            self.constraint_solver.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config, self._errno)

    def set_suction_cup_valves(self, envs_idx, link_idx, valves, stiffness, damping, max_wrench, max_cast_distance=0.01):
        """Apply valve changes to the same scene-ray request operator; no setup is required."""
        envs, links = self._suction_request_inputs(envs_idx, link_idx)
        valves = torch.as_tensor(valves, dtype=gs.tc_float, device=gs.device).reshape(-1).contiguous()
        if valves.shape != envs.shape:
            raise ValueError('valves must match the one-dimensional request batch')
        if not hasattr(self, '_suction_valve_previous'):
            self._suction_valve_previous = torch.zeros_like(valves, dtype=torch.bool)
        if self._suction_valve_previous.shape != valves.shape:
            raise ValueError('Suction valve batch shape cannot change between calls')
        self.get_suction_cup_diagnostics(envs, links)
        self._suction_valve_diagnostics[0].zero_()
        self._suction_valve_diagnostics[1].zero_()
        self._suction_valve_arguments = (envs, links, stiffness, damping, max_wrench, max_cast_distance)
        self._suction_valve_call = (valves, self._suction_valve_previous)
        try:
            self.request_suction_cup_welding(envs, links, stiffness, damping, max_wrench, max_cast_distance)
        finally:
            self._suction_valve_call = None

    def reset_suction_cup_valves(self, envs_idx):
        """Release suction and clear valve latches for selected environments."""
        if not hasattr(self, '_suction_valve_arguments'):
            return
        envs, links, stiffness, damping, max_wrench, distance = self._suction_valve_arguments
        valves = self._suction_valve_previous.to(dtype=gs.tc_float)
        valves[torch.isin(envs, envs_idx)] = 0
        self.set_suction_cup_valves(envs, links, valves, stiffness, damping, max_wrench, distance)

    def get_suction_cup_status(self, envs_idx, link_idx):
        """Return active and broken flags for each suction frame request."""
        envs, links = self._suction_request_inputs(envs_idx, link_idx)
        out = torch.zeros((envs.numel(), 2), dtype=gs.tc_int, device=gs.device)
        kernel_get_suction_status(envs, links, out, self.constraint_solver.constraint_state,
                                  self.dyn_info, self.rigid_info, self.rigid_config)
        return {'active': out[:, 0].bool(), 'broken': out[:, 1].bool()}

    def get_suction_cup_measurements(self, envs_idx, link_idx):
        """Return status and finite wrenches for observation consumers."""
        envs, links = self._suction_request_inputs(envs_idx, link_idx)
        status = self.get_suction_cup_status(envs, links)
        wrench = torch.zeros((envs.numel(), 6), device=gs.device, dtype=gs.tc_float)
        kernel_get_suction_wrench(envs, links, wrench, self.constraint_solver.constraint_state,
                                  self.dyn_info, self.rigid_info, self.rigid_config)
        return status, wrench

    def get_suction_cup_diagnostics(self, envs_idx, link_idx):
        """Return per-request hit and engagement flags and the last hit geometry."""
        envs, _ = self._suction_request_inputs(envs_idx, link_idx)
        if not hasattr(self, '_suction_valve_diagnostics'):
            count = envs.numel()
            self._suction_valve_diagnostics = (
                torch.zeros(count, dtype=torch.bool, device=gs.device),
                torch.zeros(count, dtype=torch.bool, device=gs.device),
                torch.zeros((count, 3), dtype=gs.tc_float, device=gs.device),
                torch.zeros((count, 3), dtype=gs.tc_float, device=gs.device),
            )
        if self._suction_valve_diagnostics[0].numel() != envs.numel():
            raise ValueError('Suction diagnostic batch shape cannot change between calls')
        return self._suction_valve_diagnostics

    def get_wrench(self, envs_idx, link_idx):
        """Latest solved world-frame [N,6] wrench; +inf for absent, broken, or unsolved welds."""
        envs, links = self._suction_request_inputs(envs_idx, link_idx)
        out = torch.full((envs.numel(), 6), float('inf'), device=gs.device, dtype=gs.tc_float)
        kernel_get_suction_wrench(envs, links, out, self.constraint_solver.constraint_state,
                                  self.dyn_info, self.rigid_info, self.rigid_config)
        return out

    def request_soft_weld_constraint(self, link1_idx, link2_idx, **kwargs):
        """Request a trial weld; call the returned handle's result after stepping."""
        return self.constraint_solver.request_soft_weld_constraint(link1_idx, link2_idx, **kwargs)

    def get_soft_weld_pair_status(self, link1_idx, link2_idx, envs_idx=None):
        """Read active/broken/solved device tensors for scalar or batched soft-weld pairs."""
        return self.constraint_solver.get_soft_weld_pair_status(link1_idx, link2_idx, envs_idx)

    def get_soft_weld_pair_wrench(self, link1_idx, link2_idx, envs_idx=None):
        """Read last world-frame force (N) and direct torque (N m) on the first link.

        Scalar pairs return (environments, 6); broadcastable 1-D link batches
        return (environments, pairs, 6). Missing pairs return zeros; broken
        pairs retain their breaking wrench. Runs without CPU readback.
        """
        return self.constraint_solver.get_soft_weld_pair_wrench(link1_idx, link2_idx, envs_idx)

    def apply_soft_weld_masked(self, link1_idx, link2_idx, engage_mask, release_mask, **kwargs):
        """Apply batched valve masks without extracting environment indices on the CPU."""
        return self.constraint_solver.apply_soft_weld_masked(
            link1_idx, link2_idx, engage_mask, release_mask, **kwargs
        )

    def delete_soft_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        """Delete an active or broken soft weld from selected environments."""
        return self.constraint_solver.delete_soft_weld_constraint(link1_idx, link2_idx, envs_idx)

    def get_soft_weld_constraints(self):
        """Return padded batched link pairs, break flags and last wrenches.

        ``force[..., :3]`` is force in N and ``force[..., 3:]`` is direct
        torque in N m on ``link_a``. Padded entries use ``link_a == -1``.
        """
        return self.constraint_solver.get_soft_weld_constraints()
