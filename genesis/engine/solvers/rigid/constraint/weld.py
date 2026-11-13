"""Breakable weld storage, request handling, and wrench bookkeeping."""

import quadrants as qd

import genesis as gs
from genesis.grad.bridge import no_grad, register_custom_backward
from genesis.engine.bvh import BVHTreeState
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo, RigidErrorCode
from .data import ConstraintState
import genesis.utils.geom as gu
# Raycasting also imports the rigid solver. Resolve its functions at kernel
# compilation time, after both modules have finished initializing.
import genesis.utils.raycast_qd as raycast_qd
from ..abd.forward_dynamics import func_vel_at_point
from ..abd.misc import func_apply_coupling_force


@qd.func
def func_remove_dynamic_equality(
    i_e, i_b, constraint_state: ConstraintState, dyn_info: DynInfo,
):
    """Swap-remove a dynamic equality, including all soft-weld state."""
    last = constraint_state.qd_n_equalities[i_b] - 1
    if i_e < last:
        dyn_info.equalities.release_source[i_e, i_b] = dyn_info.equalities.release_source[last, i_b]
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
    dyn_state: DynState, dyn_info: DynInfo,
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
def func_init_weld_record(
    i_e,
    i_b,
    link1_idx,
    link2_idx,
    eq_type,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
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


@no_grad
@qd.kernel(fastcache=True)
def kernel_request_suction_weld(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    dyn_state: DynState, constraint_state: ConstraintState,
    dyn_info: DynInfo, rigid_info: RigidInfo,
    rigid_config: qd.template(), errno: qd.Tensor,
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_req in range(envs_idx.shape[0]):
        i_b = envs_idx[i_req]
        frame = links_idx[i_req]
        dyn_state.links.suction_new_record[frame, i_b] = -1
        frame_info = [frame, i_b] if qd.static(rigid_config.batch_links_info) else frame
        owner = dyn_info.links.entity_idx[frame_info]
        active = False
        reusable = gs.qd_int(-1)
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            is_frame = dyn_info.equalities.eq_obj1id[i_e, i_b] == frame or dyn_info.equalities.eq_obj2id[i_e, i_b] == frame
            if is_frame and dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.SOFT_WELD:
                active = True
            if is_frame and dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD:
                reusable = i_e
        if not active:
            target = dyn_state.links.suction_target[frame, i_b]
            if target >= 0:
                link1_idx, link2_idx = qd.min(frame, target), qd.max(frame, target)
                i_e = reusable
                if i_e < 0:
                    i_e = constraint_state.qd_n_equalities[i_b]
                    if i_e == rigid_info.n_candidate_equalities[None]:
                        errno[i_b] = errno[i_b] | RigidErrorCode.OVERFLOW_DYNAMIC_EQUALITIES
                    else:
                        constraint_state.qd_n_equalities[i_b] = i_e + 1
                if i_e < rigid_info.n_candidate_equalities[None]:
                    dyn_state.links.suction_new_record[frame, i_b] = i_e
                    # Continuous pose capture is a separate AD-safe kernel.
                    dyn_info.equalities.eq_type[i_e, i_b] = gs.qd_int(gs.EQUALITY_TYPE.SOFT_WELD)
                    dyn_info.equalities.eq_obj1id[i_e, i_b] = link1_idx
                    dyn_info.equalities.eq_obj2id[i_e, i_b] = link2_idx
                    dyn_info.equalities.eq_data[i_e, i_b][10] = 1.0
                    dyn_info.equalities.sol_params[i_e, i_b] = qd.Vector([2*rigid_info.substep_dt[None], 1.0, 0.9, 0.95, 0.001, 0.5, 2.0])
                    for axis in qd.static(range(6)):
                        dyn_info.equalities.soft_weld_max_wrench[i_e, i_b][axis] = dyn_state.links.suction_max_wrench[frame, i_b][axis]
                    dyn_info.equalities.soft_weld_force[i_e, i_b] = qd.Vector.zero(gs.qd_float, 6)
                    dyn_info.equalities.soft_weld_had_rows[i_e, i_b] = False


@qd.kernel(fastcache=True)
def kernel_get_suction_welded(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(), out: qd.types.ndarray(),
    constraint_state: ConstraintState, dyn_info: DynInfo, rigid_info: RigidInfo,
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        for e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[b]):
            first, second = dyn_info.equalities.eq_obj1id[e, b], dyn_info.equalities.eq_obj2id[e, b]
            if ((first == frame or second == frame)
                and dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD):
                out[i] = True


@qd.kernel(fastcache=True)
def kernel_get_suction_wrench(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(), ghost_force: qd.types.ndarray(),
    out: qd.types.ndarray(), constraint_state: ConstraintState,
    dyn_state: DynState, dyn_info: DynInfo, rigid_info: RigidInfo,
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        welded = False
        force = qd.Vector.zero(gs.qd_float, 3)
        torque = qd.Vector.zero(gs.qd_float, 3)
        for e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[b]):
            first, second = dyn_info.equalities.eq_obj1id[e, b], dyn_info.equalities.eq_obj2id[e, b]
            if ((first == frame or second == frame)
                and dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD):
                welded = True
                if dyn_info.equalities.soft_weld_had_rows[e, b]:
                    sign = 1.0 if frame == first else -1.0
                    for axis in qd.static(range(3)):
                        force[axis] = sign * dyn_info.equalities.soft_weld_force[e, b][axis]
                        torque[axis] = sign * dyn_info.equalities.soft_weld_force[e, b][axis + 3]
        if not welded:
            for axis in qd.static(range(3)):
                force[axis] = ghost_force[b, axis]
        quat = dyn_state.links.quat[frame, b]
        force_local = gu.qd_inv_transform_by_quat(force, quat)
        torque_local = gu.qd_inv_transform_by_quat(torque, quat)
        for axis in qd.static(range(3)):
            out[i, axis] = force_local[axis]
            out[i, axis + 3] = torque_local[axis]


@qd.func
def func_update_soft_weld_wrenches(
    i_b, dyn_state: DynState, constraint_state: ConstraintState, dyn_info: DynInfo,
    break_weld: qd.template() = True,
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
            if broken and break_weld:
                dyn_info.equalities.eq_type[i_e, i_b] = gs.qd_int(gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
            i_row = i_row + 6


@qd.kernel(fastcache=True)
def kernel_release_suction_bw(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    constraint_state: ConstraintState, dyn_info: DynInfo, rigid_info: RigidInfo,
):
    # Reverse the swap-removal permutation. A serial inner walk avoids in-place adjoint aliasing.
    for i in range(envs_idx.shape[0]):
        b = envs_idx[i]
        for e in range(rigid_info.n_candidate_equalities[None]):
            dyn_info.equalities.release_data_grad[e, b] = qd.Vector.zero(gs.qd_float, 18)
            dyn_info.equalities.release_stiffness_grad[e, b] = qd.Vector.zero(gs.qd_float, 6)
            dyn_info.equalities.release_damping_grad[e, b] = qd.Vector.zero(gs.qd_float, 6)
            dyn_info.equalities.release_force_grad[e, b] = qd.Vector.zero(gs.qd_float, 6)
        for e in range(constraint_state.qd_n_equalities[b]):
            source = dyn_info.equalities.release_source[e, b]
            dyn_info.equalities.release_data_grad[source, b] = dyn_info.equalities.eq_data.grad[e, b]
            dyn_info.equalities.release_stiffness_grad[source, b] = dyn_info.equalities.soft_weld_stiffness.grad[e, b]
            dyn_info.equalities.release_damping_grad[source, b] = dyn_info.equalities.soft_weld_damping.grad[e, b]
            dyn_info.equalities.release_force_grad[source, b] = dyn_info.equalities.soft_weld_force.grad[e, b]
        for e in range(rigid_info.n_candidate_equalities[None]):
            dyn_info.equalities.eq_data.grad[e, b] = dyn_info.equalities.release_data_grad[e, b]
            dyn_info.equalities.soft_weld_stiffness.grad[e, b] = dyn_info.equalities.release_stiffness_grad[e, b]
            dyn_info.equalities.soft_weld_damping.grad[e, b] = dyn_info.equalities.release_damping_grad[e, b]
            dyn_info.equalities.soft_weld_force.grad[e, b] = dyn_info.equalities.release_force_grad[e, b]


@register_custom_backward(kernel_release_suction_bw)
@qd.kernel(fastcache=True)
def kernel_release_suction(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    constraint_state: ConstraintState, dyn_info: DynInfo,
    rigid_info: RigidInfo,
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        for slot in range(rigid_info.n_candidate_equalities[None]):
            dyn_info.equalities.release_source[slot, b] = slot
        e = rigid_info.n_equalities[None]
        while e < constraint_state.qd_n_equalities[b]:
            kind = dyn_info.equalities.eq_type[e, b]
            first = dyn_info.equalities.eq_obj1id[e, b]
            second = dyn_info.equalities.eq_obj2id[e, b]
            if ((kind == gs.EQUALITY_TYPE.SOFT_WELD or kind == gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)
                and (first == frame or second == frame)):
                func_remove_dynamic_equality(e, b, constraint_state, dyn_info)
            else:
                e += 1


@qd.kernel(fastcache=True)
def kernel_apply_suction_ghost_spring(
    frame: qd.i32, max_cast_distance: qd.f32,
    envs_idx: qd.types.ndarray(),
    dyn_state: DynState, constraint_state: ConstraintState,
    dyn_info: DynInfo, rigid_info: RigidInfo, rigid_config: qd.template(),
):
    for i_req in range(envs_idx.shape[0]):
        b = envs_idx[i_req]
        for axis in qd.static(range(3)):
            dyn_state.links.suction_ghost_force[frame, b][axis] = 0.0
        welded = False
        for e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[b]):
            first, second = dyn_info.equalities.eq_obj1id[e, b], dyn_info.equalities.eq_obj2id[e, b]
            if ((first == frame or second == frame)
                and dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD):
                welded = True
        if not welded:
            frame_info = [frame, b] if qd.static(rigid_config.batch_links_info) else frame
            owner = dyn_info.links.entity_idx[frame_info]
            quat = dyn_state.links.quat[frame, b]
            cup_pos = dyn_state.links.pos[frame, b]
            origin = cup_pos + gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, -0.5 * max_cast_distance]), quat)
            direction = gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, 1.0]), quat)
            target = dyn_state.links.suction_target[frame, b]
            distance = gs.qd_float(max_cast_distance)
            normal = gs.qd_vec3(0.0, 0.0, 0.0)
            if target >= 0:
                distance, normal = func_suction_intersection(frame, b, origin, direction, dyn_state)
            if target >= 0 and distance < 0.5 * max_cast_distance:
                if normal.dot(direction) > 0.0:
                    normal = -normal
                # Normal depth, rather than ray depth, makes F = -grad(1/2 k x^2)
                # on a fixed plane even when the cup ray meets it obliquely.
                penetration = (0.5 * max_cast_distance - distance) * qd.max(-normal.dot(direction), 0.0)
                point = cup_pos + penetration * normal
                relative_velocity = (func_vel_at_point(frame, b, point, dyn_state.links)
                                     - func_vel_at_point(target, b, point, dyn_state.links))
                k, c = gs.qd_float(0.0), gs.qd_float(0.0)
                for axis in qd.static(range(3)):
                    k += normal[axis] * normal[axis] * dyn_state.links.suction_stiffness[frame, b][axis]
                    c += normal[axis] * normal[axis] * dyn_state.links.suction_damping[frame, b][axis]
                magnitude = qd.max(0.0, k * penetration - c * relative_velocity.dot(normal))
                force = normal * magnitude
                # Same world point ensures no artificial net couple between bodies.
                func_apply_coupling_force(frame, b, point, force, dyn_state.links)
                func_apply_coupling_force(target, b, point, -force, dyn_state.links)
                for axis in qd.static(range(3)):
                    dyn_state.links.suction_ghost_force[frame, b][axis] = force[axis]

@no_grad
@qd.kernel(fastcache=True)
def kernel_cast_suction(
    envs_idx: qd.types.ndarray(), frame: qd.i32, max_cast_distance: qd.f32,
    tree_a: BVHTreeState, tree_env_a: qd.types.ndarray(),
    tree_b: BVHTreeState, tree_env_b: qd.types.ndarray(), has_tree_b: qd.template(),
    dyn_state: DynState, dyn_info: DynInfo, constraint_state: ConstraintState, rigid_config: qd.template(),
):
    for i_req in range(envs_idx.shape[0]):
        i_b = envs_idx[i_req]
        welded = False
        for e in range(constraint_state.qd_n_equalities[i_b]):
            if dyn_info.equalities.eq_type[e, i_b] == gs.EQUALITY_TYPE.SOFT_WELD:
                if dyn_info.equalities.eq_obj1id[e, i_b] == frame or dyn_info.equalities.eq_obj2id[e, i_b] == frame:
                    welded = True
        if not welded:
            frame_info = [frame, i_b] if qd.static(rigid_config.batch_links_info) else frame
            owner = dyn_info.links.entity_idx[frame_info]
            cup_pos = dyn_state.links.pos[frame, i_b]
            cup_quat = dyn_state.links.quat[frame, i_b]
            origin = cup_pos + gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, -0.5 * max_cast_distance]), cup_quat)
            direction = gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, 1.0]), cup_quat)
            face, distance, normal = raycast_qd.bvh_ray_cast(
                tree_env_a[i_b], i_b, origin, direction, max_cast_distance, dyn_state,
                tree_a, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
            )
            if qd.static(has_tree_b):
                other_face, other_distance, other_normal = raycast_qd.bvh_ray_cast(
                    tree_env_b[i_b], i_b, origin, direction, distance, dyn_state,
                    tree_b, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
                )
                if other_face >= 0:
                    face, distance, normal = other_face, other_distance, other_normal

            dyn_state.links.suction_target[frame, i_b] = -1
            if face >= 0:
                target = dyn_info.geoms.link_idx[dyn_info.faces.geom_idx[face]]
                dyn_state.links.suction_target[frame, i_b] = target
                dyn_state.links.suction_plane_pos[frame, i_b] = gu.qd_inv_transform_by_trans_quat(
                    origin + distance * direction, dyn_state.links.pos[target, i_b], dyn_state.links.quat[target, i_b])
                dyn_state.links.suction_plane_normal[frame, i_b] = gu.qd_inv_transform_by_quat(
                    normal, dyn_state.links.quat[target, i_b])



@qd.func
def func_suction_intersection(frame, b, origin, direction, dyn_state: DynState):
    target = dyn_state.links.suction_target[frame, b]
    point = gu.qd_transform_by_trans_quat(dyn_state.links.suction_plane_pos[frame, b],
        dyn_state.links.pos[target, b], dyn_state.links.quat[target, b])
    normal = gu.qd_transform_by_quat(dyn_state.links.suction_plane_normal[frame, b], dyn_state.links.quat[target, b])
    distance = (point - origin).dot(normal) / direction.dot(normal)
    return distance, normal


@qd.kernel(fastcache=True)
def kernel_update_suction_wrenches(dyn_state: DynState, constraint_state: ConstraintState, dyn_info: DynInfo):
    for b in range(constraint_state.qd_n_equalities.shape[0]):
        func_update_soft_weld_wrenches(b, dyn_state, constraint_state, dyn_info, False)


@no_grad
@qd.kernel(fastcache=True)
def kernel_break_suction(constraint_state: ConstraintState, dyn_info: DynInfo):
    # Keep the pre-break topology available while reversing the solved wrench.
    for b in range(constraint_state.qd_n_equalities.shape[0]):
        for e in range(constraint_state.qd_n_equalities[b]):
            if dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD:
                broken = False
                for axis in qd.static(range(6)):
                    if qd.abs(dyn_info.equalities.soft_weld_force[e, b][axis]) > dyn_info.equalities.soft_weld_max_wrench[e, b][axis]:
                        broken = True
                if broken:
                    dyn_info.equalities.eq_type[e, b] = gs.qd_int(gs.EQUALITY_TYPE.BROKEN_SOFT_WELD)


@qd.kernel(fastcache=True)
def kernel_set_suction_coefficients(frame: qd.i32, envs: qd.types.ndarray(),
    stiffness: qd.types.ndarray(), damping: qd.types.ndarray(), max_wrench: qd.types.ndarray(), dyn_state: DynState):
    for i in range(envs.shape[0]):
        for axis in qd.static(range(6)):
            dyn_state.links.suction_max_wrench[frame, envs[i]][axis] = max_wrench[axis]
            dyn_state.links.suction_stiffness[frame, envs[i]][axis] = stiffness[axis]
            dyn_state.links.suction_damping[frame, envs[i]][axis] = damping[axis]


@qd.kernel(fastcache=True)
def kernel_set_attached_suction_coefficients(frame: qd.i32, envs: qd.types.ndarray(),
    stiffness: qd.types.ndarray(), damping: qd.types.ndarray(), max_wrench: qd.types.ndarray(),
    dyn_info: DynInfo, constraint_state: ConstraintState):
    for i in range(envs.shape[0]):
        b = envs[i]
        for e in range(constraint_state.qd_n_equalities[b]):
            if dyn_info.equalities.eq_type[e, b] == gs.EQUALITY_TYPE.SOFT_WELD:
                if dyn_info.equalities.eq_obj1id[e, b] == frame or dyn_info.equalities.eq_obj2id[e, b] == frame:
                    for axis in qd.static(range(6)):
                        dyn_info.equalities.soft_weld_stiffness[e, b][axis] = stiffness[axis]
                        dyn_info.equalities.soft_weld_damping[e, b][axis] = damping[axis]
                        dyn_info.equalities.soft_weld_max_wrench[e, b][axis] = max_wrench[axis]


@qd.kernel(fastcache=True)
def kernel_capture_suction(envs: qd.types.ndarray(), frame: qd.i32, max_cast_distance: qd.f32,
                           dyn_state: DynState, dyn_info: DynInfo):
    # One write per continuous attachment output: reverse AD must not see the
    # initializer overwrite the very anchors this kernel captures.
    for i in range(envs.shape[0]):
        b = envs[i]
        e = dyn_state.links.suction_new_record[frame, b]
        if e >= 0:
            first = dyn_info.equalities.eq_obj1id[e, b]
            second = dyn_info.equalities.eq_obj2id[e, b]
            quat = dyn_state.links.quat[frame, b]
            origin = dyn_state.links.pos[frame, b] + gu.qd_transform_by_quat(gs.qd_vec3([0.0,0.0,-0.5*max_cast_distance]), quat)
            direction = gu.qd_transform_by_quat(gs.qd_vec3([0.0,0.0,1.0]), quat)
            distance, normal = func_suction_intersection(frame, b, origin, direction, dyn_state)
            point = origin + distance*direction
            func_set_weld_anchors(e, b, first, second, point, point, dyn_state, dyn_info)
            relative = gu.qd_quat_mul(gu.qd_inv_quat(dyn_state.links.quat[first,b]), dyn_state.links.quat[second,b])
            for axis in qd.static(range(4)):
                dyn_info.equalities.eq_data[e,b][axis+6] = relative[axis]
            for axis in qd.static(range(6)):
                dyn_info.equalities.soft_weld_stiffness[e,b][axis] = dyn_state.links.suction_stiffness[frame,b][axis]
                dyn_info.equalities.soft_weld_damping[e,b][axis] = dyn_state.links.suction_damping[frame,b][axis]
