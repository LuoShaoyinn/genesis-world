"""Breakable weld storage, request handling, and wrench bookkeeping."""

import math

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_torch
from genesis.utils.raycast_qd import bvh_ray_cast


def _soft_weld_anchor_quat(anchor_quat):
    """Normalize the attachment frame's WXYZ orientation in the cup link."""
    quat = np.asarray(anchor_quat, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1.0e-12:
        raise ValueError("anchor_quat must be a finite nonzero WXYZ quaternion")
    return tuple(map(float, quat / np.linalg.norm(quat)))


class SoftWeldRequest:
    """A weld request whose success is known after the next physics step."""

    def __init__(self, solver, link_a, link_b, envs_idx):
        self._solver = solver
        self._link_a = min(link_a, link_b)
        self._link_b = max(link_a, link_b)
        self.envs_idx = envs_idx.clone()
        self._result = None

    def result(self):
        """Return device-side success flags after the next physics step."""
        if self._result is not None:
            return self._result.clone()
        status = self._solver.get_soft_weld_pair_status(self._link_a, self._link_b, self.envs_idx)
        self._result = status["active"] & status["solved"]
        return self._result.clone()


class SoftWeldConstraintMixin:
    def get_soft_weld_constraints(self):
        """Return active and broken soft welds, padded across environments.

        ``force`` is the last six-component wrench on ``link_a`` (N, N m),
        including the wrench that caused a break. Broken welds retain their
        link pair and report ``broken=True`` until reused/deleted.
        """
        records = self.get_equality_constraints(as_tensor=True, to_torch=True)
        kinds = records["type"]
        mask = (kinds == gs.EQUALITY_TYPE.SOFT_WELD) | (kinds == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
        n_envs = kinds.shape[0]
        max_count = int(mask.sum(dim=1).max().item()) if n_envs else 0
        link_a = torch.full((n_envs, max_count), -1, dtype=gs.tc_int, device=gs.device)
        link_b = torch.full_like(link_a, -1)
        broken = torch.zeros((n_envs, max_count), dtype=torch.bool, device=gs.device)
        solved = torch.zeros_like(broken)
        force = torch.zeros((n_envs, max_count, 6), dtype=gs.tc_float, device=gs.device)
        last_force = qd_to_torch(self._solver.dyn_info.equalities.soft_weld_force, transpose=True)
        had_rows = qd_to_torch(self._solver.dyn_info.equalities.soft_weld_had_rows, transpose=True)
        env_idx, eq_idx = mask.nonzero(as_tuple=True)
        slot_idx = mask.to(torch.int32).cumsum(dim=1)[mask] - 1
        link_a[env_idx, slot_idx] = records["obj_a"][env_idx, eq_idx]
        link_b[env_idx, slot_idx] = records["obj_b"][env_idx, eq_idx]
        broken[env_idx, slot_idx] = kinds[env_idx, eq_idx] == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD
        solved[env_idx, slot_idx] = had_rows[env_idx, eq_idx]
        force[env_idx, slot_idx] = last_force[env_idx, eq_idx]
        return {"link_a": link_a, "link_b": link_b, "broken": broken, "solved": solved, "force": force}

    def request_soft_weld_constraint(self, link1_idx, link2_idx, **kwargs):
        """Register a trial weld and return its deferred first-step result."""
        envs_idx = self._solver._scene._sanitize_envs_idx(kwargs.get("envs_idx"))
        self.add_soft_weld_constraint(link1_idx, link2_idx, **kwargs)
        return SoftWeldRequest(self, int(link1_idx), int(link2_idx), envs_idx)

    def _soft_weld_query_inputs(self, link1_idx, link2_idx, envs_idx):
        # Shape checks use metadata only: device tensors never undergo a host readback.
        if (isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool) or (
            isinstance(envs_idx, np.ndarray) and envs_idx.dtype == np.bool_
        ):
            raise ValueError("Use integer environment indices, not a variable-length boolean mask.")
        first = torch.as_tensor(link1_idx, dtype=gs.tc_int, device=gs.device)
        second = torch.as_tensor(link2_idx, dtype=gs.tc_int, device=gs.device)
        if first.ndim > 1 or second.ndim > 1:
            raise ValueError("Link indices must be scalars or one-dimensional batches.")
        first, second = torch.broadcast_tensors(first, second)
        scalar = first.ndim == 0
        pairs = torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1).contiguous()
        return pairs, self._solver._scene._sanitize_envs_idx(envs_idx), scalar

    def get_soft_weld_pair_status(self, link1_idx, link2_idx, envs_idx=None):
        """Return device-side active, broken and solved boolean tensors.

        Scalar links return (environments,); broadcastable 1-D link batches
        return (environments, pairs), preserving the requested order.
        Missing pairs return false. Solved means an active weld had rows in the last substep.
        No device values are read back to the CPU.
        """
        pairs, envs_idx, scalar = self._soft_weld_query_inputs(link1_idx, link2_idx, envs_idx)
        status = torch.zeros((envs_idx.numel(), pairs.shape[0], 3), dtype=gs.tc_int, device=gs.device)
        kernel_query_soft_weld_pairs(
            pairs, envs_idx, status, False, self.constraint_state,
            self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config,
        )
        if scalar:
            status = status[:, 0]
        return {"active": status[..., 0].bool(), "broken": status[..., 1].bool(),
                "solved": status[..., 2].bool()}

    def get_soft_weld_pair_wrench(self, link1_idx, link2_idx, envs_idx=None):
        """Return the latest solved world-frame wrench on the requested first link.

        Components are force XYZ (N), then direct torque XYZ (N m), excluding
        the moment of the attachment force. Scalar links return (environments, 6);
        broadcastable 1-D batches return (environments, pairs, 6).
        Missing/unsolved pairs return zeros; broken pairs retain their breaking
        wrench. Reversing a pair negates the wrench. No CPU readback is used.
        """
        pairs, envs_idx, scalar = self._soft_weld_query_inputs(link1_idx, link2_idx, envs_idx)
        wrench = torch.zeros((envs_idx.numel(), pairs.shape[0], 6), dtype=gs.tc_float, device=gs.device)
        kernel_query_soft_weld_pairs(
            pairs, envs_idx, wrench, True, self.constraint_state,
            self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config,
        )
        return wrench[:, 0] if scalar else wrench

    def apply_soft_weld_masked(
        self, link1_idx, link2_idx, engage_mask, release_mask, *,
        linear_stiffness, linear_damping, angular_stiffness, angular_damping,
        max_force, max_torque, anchor_local, project_other_anchor_z, gain_scale=None,
        align_to_world_plane=False, anchor_quat=(1.0, 0.0, 0.0, 0.0),
    ):
        """Apply valve changes on device, optionally capturing a horizontal attachment frame.

        Projection height is scalar or one GPU value per environment. With
        ``align_to_world_plane``, the cup frame is link orientation times
        ``anchor_quat`` (local WXYZ). Its nearest yaw-only orientation is saved
        on engagement, so existing tilt immediately creates a restoring torque.
        Both the projected anchor and orientation remain fixed on the other link.
        """
        anchor_quat = _soft_weld_anchor_quat(anchor_quat)
        link1_idx, link2_idx = int(link1_idx), int(link2_idx)
        if link1_idx == link2_idx or min(link1_idx, link2_idx) < 0 or max(link1_idx, link2_idx) >= self._solver.n_links:
            raise ValueError("Soft-weld links must be two distinct valid indices")
        if self._solver._requires_grad:
            raise RuntimeError("Dynamic soft-weld breakage is not supported with differentiable simulation")
        values = (linear_stiffness, linear_damping, angular_stiffness, angular_damping,
                  max_force, max_torque, *anchor_local)
        if not all(math.isfinite(value) for value in values) or min(max_force, max_torque) <= 0:
            raise ValueError("Soft-weld parameters must be finite with positive limits")
        engage_mask = torch.as_tensor(engage_mask, dtype=torch.bool, device=gs.device)
        release_mask = torch.as_tensor(release_mask, dtype=torch.bool, device=gs.device)
        if engage_mask.shape != (self._solver._scene.n_envs,) or release_mask.shape != engage_mask.shape:
            raise ValueError("Soft-weld masks must have one value per environment")
        if gain_scale is None:
            gain_scale = torch.ones_like(engage_mask, dtype=gs.tc_float)
        else:
            gain_scale = torch.as_tensor(gain_scale, dtype=gs.tc_float, device=gs.device)
            if gain_scale.shape != engage_mask.shape:
                raise ValueError("Soft-weld gain_scale must have one value per environment")
            gain_scale = gain_scale.contiguous()
        if isinstance(project_other_anchor_z, (float, int)) and not math.isfinite(project_other_anchor_z):
            raise ValueError("Projection height must be finite")
        project_other_anchor_z = torch.as_tensor(project_other_anchor_z, dtype=gs.tc_float, device=gs.device)
        if project_other_anchor_z.ndim == 0:
            project_other_anchor_z = project_other_anchor_z.expand(engage_mask.shape)
        if project_other_anchor_z.shape != engage_mask.shape:
            raise ValueError("Projection height must be scalar or one value per environment")
        project_other_anchor_z = project_other_anchor_z.contiguous()
        local_anchor_on_second = int(link1_idx > link2_idx)
        link1_idx, link2_idx = sorted((link1_idx, link2_idx))
        self._eq_const_info_cache.clear()
        kernel_apply_soft_weld_masked(
            link1_idx, link2_idx, engage_mask, release_mask, gain_scale,
            linear_stiffness, linear_damping, angular_stiffness, angular_damping,
            max_force, max_torque, local_anchor_on_second,
            *map(float, anchor_local), project_other_anchor_z,
            int(align_to_world_plane), *anchor_quat,
            self._solver.dyn_state, self.constraint_state, self._solver.dyn_info,
            self._solver.rigid_info, self._solver.rigid_config, self._solver._errno,
        )

    def add_soft_weld_constraint(
        self,
        link1_idx,
        link2_idx,
        *,
        linear_stiffness=1000.0,
        linear_damping=40.0,
        angular_stiffness=1.0,
        angular_damping=0.02,
        max_force=math.inf,
        max_torque=math.inf,
        anchor_pos=None,
        anchor_local=None,
        project_other_anchor_z=None,
        align_to_world_plane=False,
        anchor_quat=(1.0, 0.0, 0.0, 0.0),
        envs_idx=None,
    ):
        """Attach two links with a compliant six-row weld.

        Linear stiffness/damping are in N/m and N s/m. Angular coefficients
        are the small-angle stiffness/damping in N m/rad and N m s/rad.
        The limits apply per world axis to the force and direct torque at the
        attachment, in N and N m respectively.
        Exceeding either limit breaks the weld for the next physics substep.
        """
        if self._solver._requires_grad:
            raise RuntimeError("Dynamic soft-weld breakage is not supported with differentiable simulation")
        anchor_quat = _soft_weld_anchor_quat(anchor_quat)
        if align_to_world_plane and (anchor_local is None or project_other_anchor_z is None):
            raise ValueError("Plane alignment requires anchor_local and project_other_anchor_z")
        local_anchor_on_second = int(anchor_local is not None and int(link1_idx) > int(link2_idx))
        link1_idx, link2_idx = sorted((int(link1_idx), int(link2_idx)))
        if link1_idx < 0 or link2_idx >= self._solver.n_links or link1_idx == link2_idx:
            raise ValueError("Soft-weld links must be two distinct valid global link indices")
        values = (linear_stiffness, linear_damping, angular_stiffness, angular_damping)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Soft-weld stiffness and damping must be finite and nonnegative")
        if any(math.isnan(value) or value <= 0 for value in (max_force, max_torque)):
            raise ValueError("Soft-weld force and torque limits must be positive")
        if anchor_pos is not None and anchor_local is not None:
            raise ValueError("Specify either anchor_pos or anchor_local")
        if project_other_anchor_z is not None:
            if anchor_local is None or not math.isfinite(project_other_anchor_z):
                raise ValueError("Ground projection requires a local anchor and finite z")
        if anchor_pos is not None:
            anchor_pos = np.asarray(anchor_pos, dtype=float)
            if anchor_pos.shape != (3,) or not np.isfinite(anchor_pos).all():
                raise ValueError("anchor_pos must be a finite world-space 3-vector")
            anchor_pos = tuple(map(float, anchor_pos))
        if anchor_local is not None:
            anchor_local = np.asarray(anchor_local, dtype=float)
            if anchor_local.shape != (3,) or not np.isfinite(anchor_local).all():
                raise ValueError("anchor_local must be a finite local-space 3-vector")
            anchor_local = tuple(map(float, anchor_local))
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        if torch.unique(envs_idx).numel() != envs_idx.numel():
            raise ValueError("envs_idx must not contain duplicates")
        self._eq_const_info_cache.clear()
        status = kernel_add_soft_weld_constraint(
            link1_idx,
            link2_idx,
            linear_stiffness,
            linear_damping,
            angular_stiffness,
            angular_damping,
            max_force,
            max_torque,
            int(anchor_pos is not None),
            *(anchor_pos if anchor_pos is not None else (0.0, 0.0, 0.0)),
            int(anchor_local is not None),
            local_anchor_on_second,
            *(anchor_local if anchor_local is not None else (0.0, 0.0, 0.0)),
            int(project_other_anchor_z is not None),
            float(project_other_anchor_z if project_other_anchor_z is not None else 0.0),
            int(align_to_world_plane), *anchor_quat,
            envs_idx,
            self._solver.dyn_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )
        if status & 1:
            raise RuntimeError("No dynamic equality slots remain; increase max_dynamic_constraints")
        if status & 2:
            raise ValueError("These links already have an active weld in a selected environment")

    def delete_soft_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        link1_idx, link2_idx = sorted((int(link1_idx), int(link2_idx)))
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        self._eq_const_info_cache.clear()
        kernel_delete_soft_weld_constraint(
            link1_idx,
            link2_idx,
            envs_idx,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )


@qd.kernel(fastcache=True)
def kernel_query_soft_weld_pairs(
    pairs: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    output: qd.types.ndarray(),
    read_wrench: qd.template(),
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_, i_p in qd.ndrange(envs_idx.shape[0], pairs.shape[0]):
        i_b = envs_idx[i_b_]
        first, second = pairs[i_p, 0], pairs[i_p, 1]
        link_a, link_b = qd.min(first, second), qd.max(first, second)
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if (
                dyn_info.equalities.eq_obj1id[i_e, i_b] == link_a
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link_b
            ):
                eq_type = dyn_info.equalities.eq_type[i_e, i_b]
                if eq_type == gs.EQUALITY_TYPE.SOFT_WELD or eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                    if qd.static(read_wrench):
                        sign = 1.0
                        if first != link_a:
                            sign = -1.0
                        for axis in qd.static(range(6)):
                            output[i_b_, i_p, axis] = sign * dyn_info.equalities.soft_weld_force[i_e, i_b][axis]
                    else:
                        output[i_b_, i_p, 0] = gs.qd_int(eq_type == gs.EQUALITY_TYPE.SOFT_WELD)
                        output[i_b_, i_p, 1] = gs.qd_int(eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
                        output[i_b_, i_p, 2] = gs.qd_int(
                            eq_type == gs.EQUALITY_TYPE.SOFT_WELD
                            and dyn_info.equalities.soft_weld_had_rows[i_e, i_b]
                        )


@qd.func
def func_remove_dynamic_equality(
    i_e, i_b, constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo,
):
    """Swap-remove a dynamic equality, including all soft-weld state."""
    last = constraint_state.qd_n_equalities[i_b] - 1
    if i_e < last:
        dyn_info.equalities.eq_type[i_e, i_b] = dyn_info.equalities.eq_type[last, i_b]
        dyn_info.equalities.eq_obj1id[i_e, i_b] = dyn_info.equalities.eq_obj1id[last, i_b]
        dyn_info.equalities.eq_obj2id[i_e, i_b] = dyn_info.equalities.eq_obj2id[last, i_b]
        dyn_info.equalities.eq_data[i_e, i_b] = dyn_info.equalities.eq_data[last, i_b]
        dyn_info.equalities.sol_params[i_e, i_b] = dyn_info.equalities.sol_params[last, i_b]
        dyn_info.equalities.soft_weld_params[i_e, i_b] = dyn_info.equalities.soft_weld_params[last, i_b]
        dyn_info.equalities.soft_weld_stiffness[i_e, i_b] = dyn_info.equalities.soft_weld_stiffness[last, i_b]
        dyn_info.equalities.soft_weld_damping[i_e, i_b] = dyn_info.equalities.soft_weld_damping[last, i_b]
        dyn_info.equalities.soft_weld_max_wrench[i_e, i_b] = dyn_info.equalities.soft_weld_max_wrench[last, i_b]
        dyn_info.equalities.soft_weld_force[i_e, i_b] = dyn_info.equalities.soft_weld_force[last, i_b]
        dyn_info.equalities.soft_weld_had_rows[i_e, i_b] = dyn_info.equalities.soft_weld_had_rows[last, i_b]
    constraint_state.qd_n_equalities[i_b] = last


@qd.func
def func_set_weld_anchors(
    i_e, i_b, link1_idx, link2_idx, anchor1, anchor2,
    dyn_state: array_class.DynState, dyn_info: array_class.DynInfo,
):
    """Store world-space attachment points in their respective link frames."""
    pos1 = gu.qd_inv_transform_by_trans_quat(
        anchor1, dyn_state.links.pos[link1_idx, i_b], dyn_state.links.quat[link1_idx, i_b]
    )
    pos2 = gu.qd_inv_transform_by_trans_quat(
        anchor2, dyn_state.links.pos[link2_idx, i_b], dyn_state.links.quat[link2_idx, i_b]
    )
    for axis in qd.static(range(3)):
        dyn_info.equalities.eq_data[i_e, i_b][axis + 3] = pos1[axis]
        dyn_info.equalities.eq_data[i_e, i_b][axis] = pos2[axis]


@qd.func
def func_set_soft_weld_params(i_e, i_b, params, dyn_info: array_class.DynInfo):
    """Initialize isotropic soft-weld gains and clear the previous solution."""
    dyn_info.equalities.soft_weld_params[i_e, i_b] = params
    p = dyn_info.equalities.soft_weld_params[i_e, i_b]
    dyn_info.equalities.soft_weld_stiffness[i_e, i_b] = gs.qd_vec6([p[0], p[0], p[0], p[2], p[2], p[2]])
    dyn_info.equalities.soft_weld_damping[i_e, i_b] = gs.qd_vec6([p[1], p[1], p[1], p[3], p[3], p[3]])
    dyn_info.equalities.soft_weld_max_wrench[i_e, i_b] = gs.qd_vec6([p[4], p[4], p[4], p[5], p[5], p[5]])
    dyn_info.equalities.soft_weld_force[i_e, i_b] = qd.Vector.zero(gs.qd_float, 6)
    dyn_info.equalities.soft_weld_had_rows[i_e, i_b] = False


@qd.func
def func_init_weld_record(
    i_e,
    i_b,
    link1_idx,
    link2_idx,
    eq_type,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Capture the links' current relative pose for either dynamic weld type."""
    shared_pos = dyn_state.links.pos[link1_idx, i_b]
    dyn_info.equalities.eq_type[i_e, i_b] = gs.qd_int(eq_type)
    dyn_info.equalities.eq_obj1id[i_e, i_b] = link1_idx
    dyn_info.equalities.eq_obj2id[i_e, i_b] = link2_idx
    func_set_weld_anchors(i_e, i_b, link1_idx, link2_idx, shared_pos, shared_pos, dyn_state, dyn_info)

    relpose = gu.qd_quat_mul(gu.qd_inv_quat(dyn_state.links.quat[link1_idx, i_b]), dyn_state.links.quat[link2_idx, i_b])
    for i_4 in qd.static(range(4)):
        dyn_info.equalities.eq_data[i_e, i_b][i_4 + 6] = relpose[i_4]
    dyn_info.equalities.eq_data[i_e, i_b][10] = 1.0
    dyn_info.equalities.sol_params[i_e, i_b] = qd.Vector(
        [2 * rigid_info.substep_dt[None], 1.0, 0.9, 0.95, 0.001, 0.5, 2.0]
    )


@qd.func
def func_align_weld_to_world_plane(
    i_e, i_b, cup_link, anchor_quat,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Save the nearest horizontal cup frame, including its local frame offset."""
    cup_quat = gu.qd_quat_mul(dyn_state.links.quat[cup_link, i_b], anchor_quat)
    rotation = gu.qd_quat_to_R(cup_quat, rigid_info.EPS[None])
    # Minimize ||Rz(yaw) - Rcup||_F: fit both projected horizontal axes.
    yaw_x = rotation[0, 0] + rotation[1, 1]
    yaw_y = rotation[1, 0] - rotation[0, 1]
    # At an exactly upside-down cup the closest yaw is nonunique. Choose
    # projected +X deterministically instead of relying on atan2(0, 0).
    if yaw_x * yaw_x + yaw_y * yaw_y < 1.0e-12:
        yaw_x = rotation[0, 0]
        yaw_y = rotation[1, 0]
    half_yaw = 0.5 * qd.atan2(yaw_y, yaw_x)
    plane_quat = gs.qd_vec4([qd.cos(half_yaw), 0.0, 0.0, qd.sin(half_yaw)])
    target_link_quat = gu.qd_quat_mul(plane_quat, gu.qd_inv_quat(anchor_quat))
    link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
    link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
    relpose = gu.qd_quat_mul(gu.qd_inv_quat(target_link_quat), dyn_state.links.quat[link2_idx, i_b])
    if cup_link == link2_idx:
        relpose = gu.qd_quat_mul(gu.qd_inv_quat(dyn_state.links.quat[link1_idx, i_b]), target_link_quat)
    for axis in qd.static(range(4)):
        dyn_info.equalities.eq_data[i_e, i_b][axis + 6] = relpose[axis]


@qd.kernel(fastcache=True)
def kernel_apply_soft_weld_masked(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    engage_mask: qd.types.ndarray(),
    release_mask: qd.types.ndarray(),
    gain_scale: qd.types.ndarray(),
    linear_stiffness: qd.f32,
    linear_damping: qd.f32,
    angular_stiffness: qd.f32,
    angular_damping: qd.f32,
    max_force: qd.f32,
    max_torque: qd.f32,
    local_anchor_on_second: qd.i32,
    local_x: qd.f32,
    local_y: qd.f32,
    local_z: qd.f32,
    other_anchor_z: qd.types.ndarray(),
    align_to_world_plane: qd.i32,
    anchor_qw: qd.f32,
    anchor_qx: qd.f32,
    anchor_qy: qd.f32,
    anchor_qz: qd.f32,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(engage_mask.shape[0]):
        if release_mask[i_b]:
            for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
                eq_type = dyn_info.equalities.eq_type[i_e, i_b]
                if (
                    (eq_type == gs.EQUALITY_TYPE.SOFT_WELD or eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
                    and dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                    and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
                ):
                    func_remove_dynamic_equality(i_e, i_b, constraint_state, dyn_info)
        if engage_mask[i_b]:
            i_broken = gs.qd_int(-1)
            i_reusable = gs.qd_int(-1)
            is_duplicate = False
            for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
                eq_type = dyn_info.equalities.eq_type[i_e, i_b]
                same_pair = (
                    dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                    and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
                )
                if same_pair and (eq_type == gs.EQUALITY_TYPE.WELD or eq_type == gs.EQUALITY_TYPE.SOFT_WELD):
                    is_duplicate = True
                if eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                    if i_reusable < 0:
                        i_reusable = i_e
                    if same_pair:
                        i_broken = i_e
            if is_duplicate:
                errno[i_b] = errno[i_b] | array_class.ErrorCode.DUPLICATE_DYNAMIC_WELD
            else:
                i_e = i_broken
                if i_e < 0:
                    i_e = i_reusable
                if i_e < 0:
                    i_e = constraint_state.qd_n_equalities[i_b]
                    if i_e == rigid_info.n_candidate_equalities[None]:
                        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_DYNAMIC_EQUALITIES
                    else:
                        constraint_state.qd_n_equalities[i_b] = i_e + 1
                if i_e < rigid_info.n_candidate_equalities[None]:
                    func_init_weld_record(
                        i_e, i_b, link1_idx, link2_idx, gs.EQUALITY_TYPE.SOFT_WELD,
                        dyn_state, dyn_info, rigid_info,
                    )
                    cup_link = link1_idx
                    if local_anchor_on_second:
                        cup_link = link2_idx
                    cup_anchor = gu.qd_transform_by_trans_quat(
                        gs.qd_vec3([local_x, local_y, local_z]),
                        dyn_state.links.pos[cup_link, i_b], dyn_state.links.quat[cup_link, i_b],
                    )
                    ground_anchor = gs.qd_vec3([cup_anchor[0], cup_anchor[1], other_anchor_z[i_b]])
                    anchor1 = cup_anchor
                    anchor2 = ground_anchor
                    if local_anchor_on_second:
                        anchor1 = ground_anchor
                        anchor2 = cup_anchor
                    func_set_weld_anchors(i_e, i_b, link1_idx, link2_idx, anchor1, anchor2, dyn_state, dyn_info)
                    if align_to_world_plane:
                        func_align_weld_to_world_plane(
                            i_e, i_b, cup_link, gs.qd_vec4([anchor_qw, anchor_qx, anchor_qy, anchor_qz]),
                            dyn_state, dyn_info, rigid_info,
                        )
                    scale = gain_scale[i_b]
                    damping_scale = qd.sqrt(scale)
                    func_set_soft_weld_params(i_e, i_b, qd.Vector(
                        [linear_stiffness * scale, linear_damping * damping_scale,
                         angular_stiffness * scale, angular_damping * damping_scale,
                         max_force, max_torque]
                    ), dyn_info)


@qd.kernel(fastcache=True)
def kernel_contact_gated_soft_weld(
    link1_idx: qd.i32, link2_idx: qd.i32, tip_link_idx: qd.i32, pair_index: qd.i32,
    valves: qd.types.ndarray(), contact_seen: qd.types.ndarray(), engage_seen: qd.types.ndarray(),
    linear_stiffness: qd.f32, linear_damping: qd.f32, angular_stiffness: qd.f32,
    angular_damping: qd.f32, max_force: qd.f32, max_torque: qd.f32,
    local_anchor_on_second: qd.i32, local_x: qd.f32, local_y: qd.f32, local_z: qd.f32,
    align_to_world_plane: qd.i32, anchor_qw: qd.f32, anchor_qx: qd.f32,
    anchor_qy: qd.f32, anchor_qz: qd.f32,
    dyn_state: array_class.DynState, collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo, rigid_config: qd.template(), errno: qd.Tensor,
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(valves.shape[0]):
        requested = valves[i_b, pair_index] > 0.5
        hit = False
        for i_c in range(collider_state.n_contacts[i_b]):
            a = collider_state.contact_data.link_a[i_c, i_b]
            b = collider_state.contact_data.link_b[i_c, i_b]
            if (a == link1_idx and b == link2_idx) or (a == link2_idx and b == link1_idx):
                hit = True
        active = False
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if (dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
                and dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.SOFT_WELD):
                active = True
        engage = requested and hit and not active
        if hit:
            contact_seen[i_b, pair_index] = True
        if engage:
            engage_seen[i_b, pair_index] = True
        if not requested:
            for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
                eq_type = dyn_info.equalities.eq_type[i_e, i_b]
                if (
                    (eq_type == gs.EQUALITY_TYPE.SOFT_WELD or eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
                    and dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                    and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
                ):
                    func_remove_dynamic_equality(i_e, i_b, constraint_state, dyn_info)
        if engage:
            i_broken = gs.qd_int(-1)
            i_reusable = gs.qd_int(-1)
            is_duplicate = False
            for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
                eq_type = dyn_info.equalities.eq_type[i_e, i_b]
                same_pair = (
                    dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                    and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
                )
                if same_pair and (eq_type == gs.EQUALITY_TYPE.WELD or eq_type == gs.EQUALITY_TYPE.SOFT_WELD):
                    is_duplicate = True
                if eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                    if i_reusable < 0:
                        i_reusable = i_e
                    if same_pair:
                        i_broken = i_e
            if is_duplicate:
                errno[i_b] = errno[i_b] | array_class.ErrorCode.DUPLICATE_DYNAMIC_WELD
            else:
                i_e = i_broken
                if i_e < 0:
                    i_e = i_reusable
                if i_e < 0:
                    i_e = constraint_state.qd_n_equalities[i_b]
                    if i_e == rigid_info.n_candidate_equalities[None]:
                        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_DYNAMIC_EQUALITIES
                    else:
                        constraint_state.qd_n_equalities[i_b] = i_e + 1
                if i_e < rigid_info.n_candidate_equalities[None]:
                    func_init_weld_record(
                        i_e, i_b, link1_idx, link2_idx, gs.EQUALITY_TYPE.SOFT_WELD,
                        dyn_state, dyn_info, rigid_info,
                    )
                    cup_link = link1_idx
                    if local_anchor_on_second:
                        cup_link = link2_idx
                    cup_anchor = gu.qd_transform_by_trans_quat(
                        gs.qd_vec3([local_x, local_y, local_z]),
                        dyn_state.links.pos[cup_link, i_b], dyn_state.links.quat[cup_link, i_b],
                    )
                    ground_anchor = gs.qd_vec3([cup_anchor[0], cup_anchor[1], dyn_state.links.pos[tip_link_idx, i_b][2]])
                    anchor1 = cup_anchor
                    anchor2 = ground_anchor
                    if local_anchor_on_second:
                        anchor1 = ground_anchor
                        anchor2 = cup_anchor
                    func_set_weld_anchors(i_e, i_b, link1_idx, link2_idx, anchor1, anchor2, dyn_state, dyn_info)
                    if align_to_world_plane:
                        func_align_weld_to_world_plane(
                            i_e, i_b, cup_link, gs.qd_vec4([anchor_qw, anchor_qx, anchor_qy, anchor_qz]),
                            dyn_state, dyn_info, rigid_info,
                        )
                    scale = 1.0
                    damping_scale = qd.sqrt(scale)
                    func_set_soft_weld_params(i_e, i_b, qd.Vector(
                        [linear_stiffness * scale, linear_damping * damping_scale,
                         angular_stiffness * scale, angular_damping * damping_scale,
                         max_force, max_torque]
                    ), dyn_info)


@qd.kernel(fastcache=True)
def kernel_add_soft_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    linear_stiffness: qd.f32,
    linear_damping: qd.f32,
    angular_stiffness: qd.f32,
    angular_damping: qd.f32,
    max_force: qd.f32,
    max_torque: qd.f32,
    has_anchor_pos: qd.i32,
    anchor_x: qd.f32,
    anchor_y: qd.f32,
    anchor_z: qd.f32,
    has_anchor_local: qd.i32,
    local_anchor_on_second: qd.i32,
    local_x: qd.f32,
    local_y: qd.f32,
    local_z: qd.f32,
    project_other_anchor: qd.i32,
    other_anchor_z: qd.f32,
    align_to_world_plane: qd.i32,
    anchor_qw: qd.f32,
    anchor_qx: qd.f32,
    anchor_qy: qd.f32,
    anchor_qz: qd.f32,
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> qd.i32:
    status = gs.qd_int(0)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_broken = gs.qd_int(-1)
        i_reusable = gs.qd_int(-1)
        is_duplicate = False
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            eq_type = dyn_info.equalities.eq_type[i_e, i_b]
            same_pair = (
                dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
            ) or (
                dyn_info.equalities.eq_obj1id[i_e, i_b] == link2_idx
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link1_idx
            )
            if same_pair and (eq_type == gs.EQUALITY_TYPE.WELD or eq_type == gs.EQUALITY_TYPE.SOFT_WELD):
                is_duplicate = True
            if eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                if i_reusable < 0:
                    i_reusable = i_e
                if same_pair:
                    i_broken = i_e
        if is_duplicate:
            status = status | gs.qd_int(2)
        else:
            i_e = i_broken
            if i_e < 0:
                i_e = i_reusable
            if i_e < 0:
                i_e = constraint_state.qd_n_equalities[i_b]
                if i_e == rigid_info.n_candidate_equalities[None]:
                    status = status | gs.qd_int(1)
                else:
                    constraint_state.qd_n_equalities[i_b] = i_e + 1
            if i_e < rigid_info.n_candidate_equalities[None]:
                func_init_weld_record(
                    i_e, i_b, link1_idx, link2_idx, gs.EQUALITY_TYPE.SOFT_WELD, dyn_state, dyn_info, rigid_info
                )
                if has_anchor_pos or has_anchor_local:
                    anchor = gs.qd_vec3([anchor_x, anchor_y, anchor_z])
                    anchor_link = link1_idx
                    if has_anchor_local:
                        if local_anchor_on_second:
                            anchor_link = link2_idx
                        anchor = gu.qd_transform_by_trans_quat(
                            gs.qd_vec3([local_x, local_y, local_z]),
                            dyn_state.links.pos[anchor_link, i_b], dyn_state.links.quat[anchor_link, i_b],
                        )
                    anchor1 = anchor
                    anchor2 = anchor
                    if project_other_anchor:
                        projected = gs.qd_vec3([anchor[0], anchor[1], other_anchor_z])
                        if local_anchor_on_second:
                            anchor1 = projected
                        else:
                            anchor2 = projected
                    func_set_weld_anchors(i_e, i_b, link1_idx, link2_idx, anchor1, anchor2, dyn_state, dyn_info)
                    if align_to_world_plane:
                        func_align_weld_to_world_plane(
                            i_e, i_b, anchor_link, gs.qd_vec4([anchor_qw, anchor_qx, anchor_qy, anchor_qz]),
                            dyn_state, dyn_info, rigid_info,
                        )
                func_set_soft_weld_params(i_e, i_b, qd.Vector(
                    [linear_stiffness, linear_damping, angular_stiffness, angular_damping, max_force, max_torque]
                ), dyn_info)
    return status


@qd.kernel(fastcache=True)
def kernel_request_suction_weld(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    order: qd.types.ndarray(), offsets: qd.types.ndarray(),
    stiffness: qd.types.ndarray(), damping: qd.types.ndarray(), max_wrench: qd.types.ndarray(),
    max_cast_distance: qd.f32,
    valves: qd.types.ndarray(), previous: qd.types.ndarray(), use_valves: qd.template(),
    hits: qd.types.ndarray(), engages: qd.types.ndarray(), points: qd.types.ndarray(), normals: qd.types.ndarray(),
    tree_a: array_class.BVHTreeState, tree_env_a: qd.types.ndarray(),
    tree_b: array_class.BVHTreeState, tree_env_b: qd.types.ndarray(), has_tree_b: qd.template(),
    dyn_state: array_class.DynState, constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(), errno: qd.Tensor,
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(offsets.shape[0] - 1):
        for pos in range(offsets[i_b], offsets[i_b + 1]):
            i_req = order[pos]
            frame = links_idx[i_req]
            request_now = True
            release = False
            if qd.static(use_valves):
                requested = valves[i_req] > 0.5
                request_now = requested and not previous[i_req]
                release = not requested
                previous[i_req] = requested
            frame_info = [frame, i_b] if qd.static(rigid_config.batch_links_info) else frame
            owner = dyn_info.links.entity_idx[frame_info]
            active = False
            reusable = gs.qd_int(-1)
            frame_record = gs.qd_int(-1)
            for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
                is_frame = dyn_info.equalities.eq_obj1id[i_e, i_b] == frame or dyn_info.equalities.eq_obj2id[i_e, i_b] == frame
                if is_frame and dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.SOFT_WELD:
                    active = True
                    frame_record = i_e
                if is_frame and dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                    reusable = i_e
                    frame_record = i_e
            if release and frame_record >= 0:
                func_remove_dynamic_equality(frame_record, i_b, constraint_state, dyn_info)
            if request_now and not active:
                cup_pos = dyn_state.links.pos[frame, i_b]
                cup_quat = dyn_state.links.quat[frame, i_b]
                origin = cup_pos + gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, -0.5 * max_cast_distance]), cup_quat)
                direction = gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, 1.0]), cup_quat)
                face, distance, normal = bvh_ray_cast(
                    tree_env_a[i_b], i_b, origin, direction, max_cast_distance, dyn_state,
                    tree_a, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
                )
                if qd.static(has_tree_b):
                    other_face, other_distance, other_normal = bvh_ray_cast(
                        tree_env_b[i_b], i_b, origin, direction, distance, dyn_state,
                        tree_b, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
                    )
                    if other_face >= 0:
                        face, distance, normal = other_face, other_distance, other_normal
                if face >= 0:
                    if qd.static(use_valves):
                        hits[i_req] = True
                        for axis in qd.static(range(3)):
                            points[i_req, axis] = origin[axis] + distance * direction[axis]
                            normals[i_req, axis] = normal[axis] if normal.dot(direction) <= 0.0 else -normal[axis]
                    target = dyn_info.geoms.link_idx[dyn_info.faces.geom_idx[face]]
                    link1_idx, link2_idx = qd.min(frame, target), qd.max(frame, target)
                    i_e = reusable
                    if i_e < 0:
                        i_e = constraint_state.qd_n_equalities[i_b]
                        if i_e == rigid_info.n_candidate_equalities[None]:
                            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_DYNAMIC_EQUALITIES
                        else:
                            constraint_state.qd_n_equalities[i_b] = i_e + 1
                    if i_e < rigid_info.n_candidate_equalities[None]:
                        if qd.static(use_valves):
                            engages[i_req] = True
                        func_init_weld_record(i_e, i_b, link1_idx, link2_idx, gs.EQUALITY_TYPE.SOFT_WELD,
                                              dyn_state, dyn_info, rigid_info)
                        point = origin + distance * direction
                        func_set_weld_anchors(i_e, i_b, link1_idx, link2_idx, point, point, dyn_state, dyn_info)
                        for axis in qd.static(range(6)):
                            dyn_info.equalities.soft_weld_stiffness[i_e, i_b][axis] = stiffness[i_req, axis]
                            dyn_info.equalities.soft_weld_damping[i_e, i_b][axis] = damping[i_req, axis]
                            dyn_info.equalities.soft_weld_max_wrench[i_e, i_b][axis] = max_wrench[i_req, axis]
                        dyn_info.equalities.soft_weld_force[i_e, i_b] = qd.Vector.zero(gs.qd_float, 6)
                        dyn_info.equalities.soft_weld_had_rows[i_e, i_b] = False


@qd.kernel(fastcache=True)
def kernel_get_suction_wrench(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    out: qd.types.ndarray(), constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo, rigid_config: qd.template(),
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        for e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[b]):
            first = dyn_info.equalities.eq_obj1id[e, b]
            second = dyn_info.equalities.eq_obj2id[e, b]
            if ((first == frame or second == frame)
                and dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD
                and dyn_info.equalities.soft_weld_had_rows[e, b]):
                sign = 1.0 if frame == first else -1.0
                for axis in qd.static(range(6)):
                    out[i, axis] = sign * dyn_info.equalities.soft_weld_force[e, b][axis]


@qd.kernel(fastcache=True)
def kernel_get_suction_status(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(), out: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo, rigid_config: qd.template(),
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        for e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[b]):
            first = dyn_info.equalities.eq_obj1id[e, b]
            second = dyn_info.equalities.eq_obj2id[e, b]
            if first == frame or second == frame:
                kind = dyn_info.equalities.eq_type[e, b]
                if kind == gs.EQUALITY_TYPE.SOFT_WELD:
                    out[i, 0] = 1
                elif kind == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                    out[i, 1] = 1


@qd.kernel(fastcache=True)
def kernel_delete_soft_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            eq_type = dyn_info.equalities.eq_type[i_e, i_b]
            if (
                (eq_type == gs.EQUALITY_TYPE.SOFT_WELD or eq_type == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
                and dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
            ):
                func_remove_dynamic_equality(i_e, i_b, constraint_state, dyn_info)


@qd.func
def func_update_soft_weld_wrenches(
    i_b, dyn_state: array_class.DynState, constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo
):
    # Equality rows are first in the solve. Keep a broken record (rather
    # than moving records here) so the current substep's solved forces and
    # integration remain consistent; it contributes no rows next substep.
    i_row = gs.qd_int(0)
    for i_e in range(constraint_state.qd_n_equalities[i_b]):
        eq_type = dyn_info.equalities.eq_type[i_e, i_b]
        if eq_type == gs.EQUALITY_TYPE.CONNECT:
            i_row = i_row + 3
        elif eq_type == gs.EQUALITY_TYPE.WELD:
            i_row = i_row + 6
        elif eq_type == gs.EQUALITY_TYPE.JOINT:
            i_row = i_row + 1
        elif eq_type == gs.EQUALITY_TYPE.SCREW:
            i_row = i_row + 5
        elif eq_type == gs.EQUALITY_TYPE.SOFT_WELD:
            params = dyn_info.equalities.soft_weld_params[i_e, i_b]
            broken = False
            for i_axis in qd.static(range(3)):
                row_force = constraint_state.efc_force[i_row + i_axis, i_b]
                dyn_info.equalities.soft_weld_force[i_e, i_b][i_axis] = row_force
                if qd.abs(row_force) > dyn_info.equalities.soft_weld_max_wrench[i_e, i_b][i_axis]:
                    broken = True

            # The orientation rows use a quaternion Jacobian. Their raw
            # multipliers are not torques; map them through the rotational
            # Jacobian before comparing with the limit in N m.
            link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
            link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
            quat1 = dyn_state.links.quat[link1_idx, i_b]
            inv_quat2 = gu.qd_inv_quat(dyn_state.links.quat[link2_idx, i_b])
            relpose = gs.qd_vec4(
                [
                    dyn_info.equalities.eq_data[i_e, i_b][6],
                    dyn_info.equalities.eq_data[i_e, i_b][7],
                    dyn_info.equalities.eq_data[i_e, i_b][8],
                    dyn_info.equalities.eq_data[i_e, i_b][9],
                ]
            )
            q = gu.qd_quat_mul(quat1, relpose)
            torquescale = dyn_info.equalities.eq_data[i_e, i_b][10]
            for i_axis in qd.static(range(3)):
                axis = qd.Vector.zero(gs.qd_float, 3)
                axis[i_axis] = 1.0
                quat_jac = gu.qd_quat_mul(gu.qd_quat_mul_axis(inv_quat2, axis), q)
                torque = gs.qd_float(0.0)
                for i_rot in qd.static(range(3)):
                    torque = (
                        torque
                        + 0.5
                        * quat_jac[i_rot + 1]
                        * torquescale
                        * constraint_state.efc_force[i_row + 3 + i_rot, i_b]
                    )
                dyn_info.equalities.soft_weld_force[i_e, i_b][i_axis + 3] = torque
                if qd.abs(torque) > dyn_info.equalities.soft_weld_max_wrench[i_e, i_b][i_axis + 3]:
                    broken = True
            if broken:
                dyn_info.equalities.eq_type[i_e, i_b] = gs.qd_int(gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
            i_row = i_row + 6
