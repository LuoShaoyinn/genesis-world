"""Breakable weld storage, request handling, and wrench bookkeeping."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.utils.raycast_qd import bvh_ray_cast


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


@qd.kernel(fastcache=True)
def kernel_request_suction_weld(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    stiffness: qd.types.ndarray(), damping: qd.types.ndarray(), max_wrench: qd.types.ndarray(),
    max_cast_distance: qd.f32,
    tree_a: array_class.BVHTreeState, tree_env_a: qd.types.ndarray(),
    tree_b: array_class.BVHTreeState, tree_env_b: qd.types.ndarray(), has_tree_b: qd.template(),
    dyn_state: array_class.DynState, constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(), errno: qd.Tensor,
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_req in range(envs_idx.shape[0]):
        i_b = envs_idx[i_req]
        frame = links_idx[i_req]
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


@qd.kernel(fastcache=True)
def kernel_release_suction(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
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
def kernel_suction_normal(
    envs_idx: qd.types.ndarray(), links_idx: qd.types.ndarray(), max_cast_distance: qd.f32,
    out: qd.types.ndarray(),
    tree_a: array_class.BVHTreeState, tree_env_a: qd.types.ndarray(),
    tree_b: array_class.BVHTreeState, tree_env_b: qd.types.ndarray(), has_tree_b: qd.template(),
    dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template(),
):
    for i in range(envs_idx.shape[0]):
        b, frame = envs_idx[i], links_idx[i]
        frame_info = [frame, b] if qd.static(rigid_config.batch_links_info) else frame
        owner = dyn_info.links.entity_idx[frame_info]
        quat = dyn_state.links.quat[frame, b]
        origin = dyn_state.links.pos[frame, b] + gu.qd_transform_by_quat(
            gs.qd_vec3([0.0, 0.0, -0.5 * max_cast_distance]), quat
        )
        direction = gu.qd_transform_by_quat(gs.qd_vec3([0.0, 0.0, 1.0]), quat)
        face, distance, normal = bvh_ray_cast(
            tree_env_a[b], b, origin, direction, max_cast_distance, dyn_state,
            tree_a, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
        )
        if qd.static(has_tree_b):
            other_face, other_distance, other_normal = bvh_ray_cast(
                tree_env_b[b], b, origin, direction, distance, dyn_state,
                tree_b, dyn_info, gs.EPS, 1, owner, rigid_config.batch_links_info, True,
            )
            if other_face >= 0:
                face, distance, normal = other_face, other_distance, other_normal
        if face >= 0:
            if normal.dot(direction) > 0.0:
                normal = -normal
            local_normal = gu.qd_inv_transform_by_quat(normal, quat)
            for axis in qd.static(range(3)):
                out[i, axis] = local_normal[axis]
