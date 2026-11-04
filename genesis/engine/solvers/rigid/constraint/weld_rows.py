"""Jacobian and compliance rows for ordinary and breakable welds."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def func_equality_weld(
    i_b: int,
    i_e: int,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    func_equality_jdotv: qd.template(),
    _append_relevant_dof: qd.template(),
    _sort_relevant_dofs_descending: qd.template(),
):
    EPS = rigid_info.EPS[None]

    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    # Get equality info for this constraint
    link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
    link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if qd.static(rigid_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if qd.static(rigid_config.batch_links_info) else link2_idx

    # For weld, eq_data layout:
    # [0:3]  : anchor2 (local pos in body2)
    # [3:6]  : anchor1 (local pos in body1)
    # [6:10] : relative pose (quat) of body 2 related to body 1 to match orientations
    # [10]   : torquescale
    anchor1_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][3],
            dyn_info.equalities.eq_data[i_e, i_b][4],
            dyn_info.equalities.eq_data[i_e, i_b][5],
        ]
    )
    anchor2_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][0],
            dyn_info.equalities.eq_data[i_e, i_b][1],
            dyn_info.equalities.eq_data[i_e, i_b][2],
        ]
    )
    relpose = gs.qd_vec4(
        [
            dyn_info.equalities.eq_data[i_e, i_b][6],
            dyn_info.equalities.eq_data[i_e, i_b][7],
            dyn_info.equalities.eq_data[i_e, i_b][8],
            dyn_info.equalities.eq_data[i_e, i_b][9],
        ]
    )
    torquescale = dyn_info.equalities.eq_data[i_e, i_b][10]
    sol_params = dyn_info.equalities.sol_params[i_e, i_b]
    is_soft = dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.SOFT_WELD
    soft_params = qd.Vector.zero(gs.qd_float, 6)
    if is_soft:
        soft_params = dyn_info.equalities.soft_weld_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.qd_transform_by_trans_quat(
        pos=anchor1_pos, trans=dyn_state.links.pos[link1_idx, i_b], quat=dyn_state.links.quat[link1_idx, i_b]
    )
    global_anchor2 = gu.qd_transform_by_trans_quat(
        pos=anchor2_pos, trans=dyn_state.links.pos[link2_idx, i_b], quat=dyn_state.links.quat[link2_idx, i_b]
    )

    pos_error = global_anchor1 - global_anchor2

    # Compute orientation error.
    # For weld: compute q = body1_quat * relpose, then error = (inv(body2_quat) * q)
    quat_body1 = dyn_state.links.quat[link1_idx, i_b]
    quat_body2 = dyn_state.links.quat[link2_idx, i_b]
    q = gu.qd_quat_mul(quat_body1, relpose)
    inv_quat_body2 = gu.qd_inv_quat(quat_body2)
    error_quat = gu.qd_quat_mul(inv_quat_body2, q)
    # Take the vector (axis) part and scale by torquescale.
    rot_error = gs.qd_vec3([error_quat[1], error_quat[2], error_quat[3]]) * torquescale

    all_error = gs.qd_vec6([pos_error[0], pos_error[1], pos_error[2], rot_error[0], rot_error[1], rot_error[2]])
    pos_imp = all_error.norm()

    # Compute inverse weight from both bodies.
    invweight = dyn_info.links.invweight[link_a_maybe_batch] + dyn_info.links.invweight[link_b_maybe_batch]

    # Centripetal/Coriolis bias of the anchor pair: see func_equality_connect for the rationale of subtracting
    # Jdot @ qvel from aref. The chains' angular parts feed the rotation rows' bias below.
    jdotv1, cddb1_ang = func_equality_jdotv(i_b, link1_idx, global_anchor1, dyn_state, dyn_info, rigid_config)
    jdotv2, cddb2_ang = func_equality_jdotv(i_b, link2_idx, global_anchor2, dyn_state, dyn_info, rigid_config)
    jdotv = jdotv1 - jdotv2

    dyn_state.links.is_constrained[link1_idx, i_b] = True
    dyn_state.links.is_constrained[link2_idx, i_b] = True

    # --- Position part (first 3 constraints) ---
    same_root = (
        link2_idx > -1 and dyn_info.links.root_idx[link_a_maybe_batch] == dyn_info.links.root_idx[link_b_maybe_batch]
    )
    for i in range(3):
        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
        qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_dofs = 0

        if qd.static(rigid_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

        jac_qvel = gs.qd_float(0.0)
        for i_ab in range(2):
            sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
            link = link1_idx if i_ab == 0 else link2_idx
            pos_anchor = global_anchor1 if i_ab == 0 else global_anchor2

            # Accumulate jacobian contributions along the kinematic chain.
            # (Assuming similar structure to equality_connect.)
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

                for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                    i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_
                    cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                    cdot_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                    t_pos = pos_anchor - dyn_state.links.root_COM[link, i_b]
                    # t_quat = gu.qd_identity_quat()
                    # _ang, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)
                    vel = cdot_vel - t_pos.cross(cdof_ang)
                    diff = sign * vel
                    jac = diff[i]
                    jac_qvel = jac_qvel + jac * dyn_state.dofs.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    con_n_dofs = _append_relevant_dof(
                        n_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                    )
                link = dyn_info.links.parent_idx[link_maybe_batch]

        constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)

        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
        diag = qd.max(invweight[0] * (1 - imp) / imp, EPS)
        if is_soft:
            dt = rigid_info.substep_dt[None]
            # Implicit Euler for F = -k*x_next - c*v_next, with
            # x_next = x + dt*v_next and v_next = v + dt*x_ddot:
            # x_ddot + F / (dt*c + dt^2*k)
            #     = -(k*x + (c + dt*k)*v) / (dt*c + dt^2*k).
            # J*M^-1*J^T already contains the articulated inverse mass.
            # Scaling aref by invweight again makes static extension depend
            # on mass, damping, and substep size instead of just F/k.
            stiffness = dyn_info.equalities.soft_weld_stiffness[i_e, i_b][i]
            damping = dyn_info.equalities.soft_weld_damping[i_e, i_b][i]
            compliance_denom = qd.max(dt * (damping + dt * stiffness), EPS)
            diag = 1.0 / compliance_denom
            aref = -(stiffness * pos_error[i] + (damping + dt * stiffness) * jac_qvel) / compliance_denom

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref - jdotv[i]
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # --- Orientation part (next 3 constraints) ---
    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 3)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
    con_n_dofs = 0
    for i_con in range(n_con, n_con + 3):
        for i_d in range(n_dofs):
            constraint_state.jac[i_con, i_d, i_b] = gs.qd_float(0.0)

    for i_ab in range(2):
        sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
        link = link1_idx if i_ab == 0 else link2_idx
        # For rotation, we use the body's orientation (here we use its quaternion)
        # and a suitable reference frame. (You may need a more detailed implementation.)
        while link > -1:
            link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

            for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_
                jac = sign * dyn_state.dofs.cdof_ang[i_d, i_b]

                for i_con in range(n_con, n_con + 3):
                    constraint_state.jac[i_con, i_d, i_b] = constraint_state.jac[i_con, i_d, i_b] + jac[i_con - n_con]

                # The 3 orientation constraints share the same support (the DOFs along both kinematic chains); record
                # it so sparse assembly does not drop them. (The position part above does the same per constraint.)
                n_dofs_new = con_n_dofs
                for i_con in range(n_con, n_con + 3):
                    n_dofs_new = _append_relevant_dof(
                        i_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                    )
                con_n_dofs = n_dofs_new
            link = dyn_info.links.parent_idx[link_maybe_batch]

    jac_qvel = qd.Vector([0.0, 0.0, 0.0])
    for i_d in range(n_dofs):
        # quat2 = neg(q1)*(jac0-jac1)
        # quat3 = neg(q1)*(jac0-jac1)*q0*relpose
        jac_diff_r = qd.Vector(
            [
                constraint_state.jac[n_con, i_d, i_b],
                constraint_state.jac[n_con + 1, i_d, i_b],
                constraint_state.jac[n_con + 2, i_d, i_b],
            ]
        )
        quat2 = gu.qd_quat_mul_axis(inv_quat_body2, jac_diff_r)
        quat3 = gu.qd_quat_mul(quat2, q)

        for i_con in range(n_con, n_con + 3):
            constraint_state.jac[i_con, i_d, i_b] = 0.5 * quat3[i_con - n_con + 1] * torquescale
            jac_qvel[i_con - n_con] = (
                jac_qvel[i_con - n_con] + constraint_state.jac[i_con, i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
            )

    for i_con in range(n_con, n_con + 3):
        constraint_state.jac_n_dofs[i_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(i_con, i_b, con_n_dofs, constraint_state, rigid_config)

    # Rotational Jdot @ qvel: differentiate the rotation rows 0.5 * neg(q1) * (Jr0 - Jr1) @ qvel * q0 * relpose in
    # time; the product rule yields three terms, with quaternion derivatives qdot = 0.5 * (0, omega) * quat.
    omega1 = dyn_state.links.cd_ang[link1_idx, i_b]
    omega2 = dyn_state.links.cd_ang[link2_idx, i_b]
    domega = omega1 - omega2
    qdot_body1 = 0.5 * gu.qd_quat_mul(gs.qd_vec4([0.0, omega1[0], omega1[1], omega1[2]]), quat_body1)
    qdot0r = gu.qd_quat_mul(qdot_body1, relpose)
    qdot_body2 = 0.5 * gu.qd_quat_mul(gs.qd_vec4([0.0, omega2[0], omega2[1], omega2[2]]), quat_body2)
    djrdv = cddb1_ang - cddb2_ang
    t1 = gu.qd_quat_mul(gu.qd_quat_mul_axis(gu.qd_inv_quat(qdot_body2), domega), q)
    t2 = gu.qd_quat_mul(gu.qd_quat_mul_axis(inv_quat_body2, djrdv), q)
    t3 = gu.qd_quat_mul(gu.qd_quat_mul_axis(inv_quat_body2, domega), qdot0r)

    for i_con_ in qd.static(range(3)):
        i_con = i_con_ + n_con
        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel[i_con_], rot_error[i_con_])
        diag = qd.max(invweight[1] * (1.0 - imp) / imp, EPS)
        if is_soft:
            dt = rigid_info.substep_dt[None]
            # Quaternion-vector error and its Jacobian are both half the
            # rotation vector near rest. Four times the physical coefficients
            # gives torque = -k*angle - c*angular_velocity at small angles.
            # Use the same implicit compliance as translation; multiplying by
            # invweight and then softening again makes torque vanish with dt.
            stiffness = 4.0 * dyn_info.equalities.soft_weld_stiffness[i_e, i_b][i_con_ + 3]
            damping = 4.0 * dyn_info.equalities.soft_weld_damping[i_e, i_b][i_con_ + 3]
            compliance_denom = qd.max(dt * (damping + dt * stiffness), EPS)
            diag = 1.0 / compliance_denom
            aref = -(stiffness * rot_error[i_con_] + (damping + dt * stiffness) * jac_qvel[i_con_]) / compliance_denom
        constraint_state.diag[i_con, i_b] = diag
        constraint_state.aref[i_con, i_b] = (
            aref - 0.5 * (t1[i_con_ + 1] + t2[i_con_ + 1] + t3[i_con_ + 1]) * torquescale
        )
        constraint_state.efc_D[i_con, i_b] = 1.0 / diag
