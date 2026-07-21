# SPDX-License-Identifier: Apache-2.0
"""Simulate the Damiao MIT control law + real physics, closed-loop with MPC, in Viser.

Pure software "digital twin" of the real MPC+hardware loop, with NO hardware
involved at all — for investigating whether the real-hardware "MPC goes
crazy" instability reproduces given:
  1. cuRobo's MPC, exactly as used live (same config, same goal-drag loop).
  2. The Damiao motor's own MIT control law, emulated exactly:
     tau = kp*(p_des - p_actual) + kd*(v_des - v_actual) + tau_ff
     using this project's real MOTOR_MAP kp/kd and the real
     compute_gravity_torque() feedforward (tau_ff = tau_g).
  3. Real forward dynamics (Pinocchio ABA: qddot = M^-1(q)(tau - nle(q,qdot)))
     using the same mass-corrected URDF as everything else, so the
     simulated arm actually has to fight its own real inertia and gravity —
     not an idealized kinematic tracker.

If the instability reproduces here, it's a genuine MPC/MIT-control-loop
interaction problem (bad gains, model mismatch, insufficient damping margin,
etc.) — fixable and testable entirely in software. If it does NOT reproduce
here, the bug is elsewhere (CAN bus, motorbridge, timing, hardware-specific).

Usage:

.. code-block:: bash

   python rebot_mit_sim.py --viz-port 8080
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pinocchio as pin
import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

from rebot_mpc_control import MOTOR_MAP, REBOT_URDF_PATH, ROBOT_YML, compute_gravity_torque

CSPACE_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1"]

# gripper_joint1 doesn't affect the tracked end-effector pose (gripper_link
# is the parent frame, before the finger joints), so MPC's pose-tracking
# cost is blind to it -- it's a free/unconstrained DOF from MPC's
# perspective and has no reason to move sensibly. Held fixed rather than
# actively "controlled" by MPC's otherwise-ungoverned output for this index.
GRIPPER_JOINT_IDX = CSPACE_JOINT_NAMES.index("gripper_joint1")

# Outer position->velocity and inner velocity->torque proportional gains for
# the approximated FORCE_POS ("CSP") servo (see step_csp()). Saturation
# (vlim / force_ratio*effort) only bounds the FAR-from-target regime; once
# |K_POS * position_error| < vlim, v_cmd stops saturating and the cascade
# collapses to a plain linear PD law with effective gains
# Kp_eff = CSP_K_POS * CSP_K_VEL, Kd_eff = CSP_K_VEL. An earlier choice of
# 20/5 gave Kp_eff=100 -- stiffer than even the original raw MIT kp=18 that
# caused joint6 (very low effective inertia) to diverge, so the "safe by
# saturation" design was actually reintroducing that same problem right at
# the point saturation stops helping (settling near the target). Chosen so
# Kp_eff/Kd_eff exactly match the gravity-comp gains (GRAVITY_SOFT_KP=2.0,
# GRAVITY_SOFT_KD=1.0) already confirmed numerically stable for joint6 --
# far-from-target motion still saturates at vlim/force_ratio*effort (e.g.
# any error > vlim/CSP_K_POS = 0.5 rad at vlim=1.0), only the final
# settling phase uses this proven-stable, gentle regulation.
CSP_K_POS = 2.0
CSP_K_VEL = 1.0


class MitDynamicsSimulator:
    """Simulates the Damiao MIT law + real forward dynamics for the rebot arm.

    Internal state is the FULL Pinocchio model's (q, v) — 8 DOF (cspace's 7
    plus the mechanically-mirrored gripper_joint2, which is not an
    independently actuated DOF: after every integration step its position
    and velocity are forced to match gripper_joint1's, approximating the
    real mimic linkage's rigid coupling).
    """

    def __init__(self, initial_cspace_position: np.ndarray, lock_gripper_at: float = None):
        """lock_gripper_at: if set, gripper_joint1/2 are hard-locked to this
        position (q forced, v forced to 0) at the end of every integration
        step -- not just a soft PD hold. A soft hold (fixed p_des, normal
        kp/kd) still lets the joint move: cross-joint dynamic coupling
        through the mass matrix/Coriolis terms from the ARM's own motion can
        perturb it even when its own target never changes, and gripper's
        kp=8 isn't stiff enough to fully suppress that. A hard lock
        guarantees zero motion regardless of coupling -- true "no motor
        control" (nothing to actuate it, and nothing lets it drift either).
        """
        self.model = pin.buildModelFromUrdf(REBOT_URDF_PATH)
        self.data = self.model.createData()
        self.names = self.model.names.tolist()
        self._cspace_idx = [self.names.index(n) - 1 for n in CSPACE_JOINT_NAMES]
        self._g2_idx = self.names.index("gripper_joint2") - 1
        self._g1_idx = self.names.index("gripper_joint1") - 1
        self._gripper_lock = lock_gripper_at

        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)
        for i, idx in enumerate(self._cspace_idx):
            self.q[idx] = initial_cspace_position[i]
        self.q[self._g2_idx] = self.q[self._g1_idx]

        # Position and effort (torque/force) limits, parsed straight from the
        # URDF's <limit> tags by Pinocchio -- covers both the revolute joints
        # (rad, N*m) and gripper_joint1/2's prismatic slide (m, N). Nothing
        # in step() enforced these before, so a transient tracking error had
        # no mechanical stop or torque ceiling to arrest it -- e.g.
        # gripper_joint1 (real range 0-0.0715m) reaching +47 in an earlier
        # MPC trace, which is not physically possible on the real hardware.
        self._lower = self.model.lowerPositionLimit.copy()
        self._upper = self.model.upperPositionLimit.copy()
        self._effort_limit = self.model.effortLimit.copy()

    def cspace_position(self) -> np.ndarray:
        return np.array([self.q[i] for i in self._cspace_idx])

    def cspace_velocity(self) -> np.ndarray:
        return np.array([self.v[i] for i in self._cspace_idx])

    def step(
        self, p_des: np.ndarray, v_des: np.ndarray, dt: float,
        kp: np.ndarray = None, kd: np.ndarray = None, tau_g: np.ndarray = None,
    ) -> None:
        """Advance the simulation by dt, given (p_des, v_des) for cspace joints.

        kp/kd default to each joint's real MOTOR_MAP gains (the stiff MPC
        tracking gains); pass explicit arrays to use different gains (e.g.
        the soft gravity-comp gains) with the same dynamics/MIT-law code.

        tau_g defaults to being recomputed from the current position every
        call; callers doing many fine substeps per outer control tick should
        instead compute it ONCE per tick and pass it in — gravity load barely
        changes over a single tick, and recomputing it at microsecond-scale
        substep rates is pure wasted cost.
        """
        if tau_g is None:
            tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, self.cspace_position().tolist())

        tau_full = np.zeros(self.model.nv)
        for i, (name, idx) in enumerate(zip(CSPACE_JOINT_NAMES, self._cspace_idx)):
            spec = MOTOR_MAP[name]
            kp_i = spec.kp if kp is None else kp[i]
            kd_i = spec.kd if kd is None else kd[i]
            p_actual = self.q[idx]
            v_actual = self.v[idx]
            tau_full[idx] = kp_i * (p_des[i] - p_actual) + kd_i * (v_des[i] - v_actual) + tau_g[i]
        # gripper_joint2 has no independent actuator; its dynamics are driven
        # entirely by the mimic constraint enforced below, not by MIT torque.
        self._integrate(tau_full, dt)

    def step_csp(
        self, p_des: np.ndarray, dt: float,
        vlim: np.ndarray = None, force_ratio: np.ndarray = None, tau_g: np.ndarray = None,
    ) -> None:
        """Approximates Damiao FORCE_POS ("CSP"-style) mode for cspace joints.

        Real FORCE_POS firmware is a proprietary, undocumented position
        servo — this models it as the standard cascaded structure that kind
        of drive mode implies: an outer position->velocity loop saturated at
        vlim, feeding an inner velocity->torque loop saturated at
        force_ratio * rated effort. Unlike the raw MIT PD law in step()
        (tau = kp*(p_des-p) + kd*(v_des-v), unbounded for a large error),
        both stages here are hard-saturated by construction — a big or
        discontinuous p_des cannot produce more than vlim of commanded
        velocity or more than force_ratio*effort of applied torque, no
        after-the-fact clamping required. vlim/force_ratio default to each
        joint's real MOTOR_MAP.max_velocity / force_pos_ratio.
        """
        if tau_g is None:
            tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, self.cspace_position().tolist())
        if vlim is None:
            vlim = np.array([MOTOR_MAP[n].max_velocity for n in CSPACE_JOINT_NAMES])
        if force_ratio is None:
            force_ratio = np.array([MOTOR_MAP[n].force_pos_ratio for n in CSPACE_JOINT_NAMES])

        tau_full = np.zeros(self.model.nv)
        for i, (name, idx) in enumerate(zip(CSPACE_JOINT_NAMES, self._cspace_idx)):
            p_actual = self.q[idx]
            v_actual = self.v[idx]
            v_cmd = np.clip(CSP_K_POS * (p_des[i] - p_actual), -vlim[i], vlim[i])
            max_tau = force_ratio[i] * self._effort_limit[idx]
            tau_full[idx] = np.clip(CSP_K_VEL * (v_cmd - v_actual), -max_tau, max_tau) + tau_g[i]

        self._integrate(tau_full, dt)

    def _integrate(self, tau_full: np.ndarray, dt: float) -> None:
        # Torque/force saturation: a real motor cannot exceed its rated
        # effort (27 N*m for the 4340P shoulder/elbow joints, 7 N*m for the
        # 4310 wrist joints, 100 N for the gripper slide) no matter how large
        # the commanded torque is. Clamp before integrating so an oversized
        # tracking error can't turn into unphysical acceleration.
        tau_full = np.clip(tau_full, -self._effort_limit, self._effort_limit)

        ddq = pin.aba(self.model, self.data, self.q, self.v, tau_full)

        self.v = self.v + ddq * dt
        self.q = pin.integrate(self.model, self.q, self.v * dt)

        # Hard mechanical stop: clamp position to the URDF's declared travel
        # range, and zero out any velocity still pushing further past the
        # limit (inelastic stop, not a bounce) -- otherwise nothing prevents
        # the integrator from running straight through a joint's physical
        # limit.
        at_lower = self.q <= self._lower
        at_upper = self.q >= self._upper
        self.q = np.clip(self.q, self._lower, self._upper)
        self.v = np.where(at_lower & (self.v < 0), 0.0, self.v)
        self.v = np.where(at_upper & (self.v > 0), 0.0, self.v)

        # Enforce the mimic constraint (real mechanical coupling).
        self.q[self._g2_idx] = self.q[self._g1_idx]
        self.v[self._g2_idx] = self.v[self._g1_idx]

        if self._gripper_lock is not None:
            self.q[self._g1_idx] = self._gripper_lock
            self.v[self._g1_idx] = 0.0
            self.q[self._g2_idx] = self._gripper_lock
            self.v[self._g2_idx] = 0.0

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.q)) and np.all(np.isfinite(self.v)))


# Soft gains matching rebot_gravity_comp.py / the reference's
# gravity_compensation_controller (kp=2.0, kd=1.0 uniformly) — confirmed
# stable on real hardware. Used as the validation baseline for this
# simulator: if THIS diverges with the same integrator, the simulator
# itself is broken; if it stays stable while MPC's stiffer gains don't,
# that's real signal about the MPC/MIT interaction specifically.
GRAVITY_SOFT_KP = 2.0
GRAVITY_SOFT_KD = 1.0

# joint6 (wrist roll) has ~10-100x lower effective inertia than every other
# joint (mass matrix diagonal ~0.0004 vs 0.02-0.2 for the rest), making its
# closed-loop dynamics numerically stiff under simple semi-implicit Euler
# integration. dt=0.002s (the old default's effective sim_dt) diverges to
# NaN within 2-3 steps; dt=5e-5s (40x finer) was confirmed stable for 2000
# steps (0.1s simulated) in isolation. This is a simulator/integrator
# accuracy issue, not a real control-loop instability — the same soft-gain
# control law is independently confirmed stable on real hardware.
DEFAULT_SIM_DT = 5e-5


def run_gravity_sim(viz_port: int = 8080, control_dt: float = 0.02, sim_dt: float = DEFAULT_SIM_DT):
    """Simulate the KNOWN-GOOD gravity-compensation control law, no MPC at all.

    Validation baseline: rebot_gravity_comp.py's control law (soft kp=2/kd=1
    + Pinocchio gravity feedforward, hold-at-current-position) is confirmed
    stable on real hardware. If this simulation is ALSO stable, the
    simulator's dynamics/integration are trustworthy; if this ALSO diverges,
    the bug is in the simulator, not in MPC.
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    initial_position = np.zeros(len(CSPACE_JOINT_NAMES))
    sim = MitDynamicsSimulator(initial_position)
    kp = np.full(len(CSPACE_JOINT_NAMES), GRAVITY_SOFT_KP)
    kd = np.full(len(CSPACE_JOINT_NAMES), GRAVITY_SOFT_KD)
    zero_vel = np.zeros(len(CSPACE_JOINT_NAMES))

    def cspace_state() -> JointState:
        pos = torch.tensor([sim.cspace_position()], device="cuda", dtype=torch.float32)
        st = JointState.from_position(pos, joint_names=CSPACE_JOINT_NAMES)
        return st

    viser_viz.set_joint_state(cspace_state().squeeze(0))

    print(f"\nGravity-comp-only simulator (validation baseline) at http://localhost:{viz_port}")
    print(f"Soft gains kp={GRAVITY_SOFT_KP}, kd={GRAVITY_SOFT_KD} + gravity feedforward — should hold ~still, no drift.")
    print("Press Ctrl+C to exit.\n")

    tick = 0
    report_every = max(1, int(round(1.0 / control_dt)))
    sim_substeps = max(1, round(control_dt / sim_dt))
    print(f"Integrating at sim_dt={sim_dt:g}s ({sim_substeps} substeps per {control_dt:g}s tick).")
    while True:
        p_des = sim.cspace_position()  # hold at wherever it currently is
        tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, p_des.tolist())
        for _ in range(sim_substeps):
            sim.step(p_des, zero_vel, sim_dt, kp=kp, kd=kd, tau_g=tau_g)
            if not sim.is_finite():
                print(f"  [FATAL] NaN/Inf at tick {tick} — simulator diverged. This would mean the "
                      "simulator itself is broken (dynamics/integration bug), since this is the "
                      "known-stable-on-real-hardware gravity-comp control law.")
                return

        if tick % report_every == 0:
            tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, sim.cspace_position().tolist())
            print(f"  [tick {tick}] q: " + "  ".join(f"{v:+.4f}" for v in sim.cspace_position())
                  + "  tau_g: " + "  ".join(f"{t:+.2f}" for t in tau_g))

        viser_viz.set_joint_state(cspace_state().squeeze(0))
        tick += 1
        time.sleep(control_dt)


def run_mit_sim(
    viz_port: int = 8080, control_dt: float = 0.03, sim_dt: float = DEFAULT_SIM_DT,
    kp_scale: float = 1.0, kd_scale: float = 1.0, ignore_mpc_velocity: bool = False,
    no_gravity_comp: bool = False, replan_from_belief: bool = False,
):
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML,
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        optimization_dt=control_dt,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    mpc = ModelPredictiveControl(config)

    initial_position = mpc.default_joint_position.clone().cpu().numpy()
    gripper_hold_position = initial_position[GRIPPER_JOINT_IDX]
    sim = MitDynamicsSimulator(initial_position, lock_gripper_at=gripper_hold_position)

    # None => MitDynamicsSimulator.step() uses each joint's real MOTOR_MAP
    # tracking gains (kp=120/18) unmodified. kp_scale/kd_scale let this
    # diagnostic test whether MPC's much stiffer gains (vs. gravity-comp's
    # uniform kp=2/kd=1, which is confirmed stable) are what's driving the
    # tick-23 divergence via cross-joint coupling into joint6's low inertia.
    track_kp = track_kd = None
    if kp_scale != 1.0 or kd_scale != 1.0:
        track_kp = np.array([MOTOR_MAP[n].kp for n in CSPACE_JOINT_NAMES]) * kp_scale
        track_kd = np.array([MOTOR_MAP[n].kd for n in CSPACE_JOINT_NAMES]) * kd_scale
        print(f"Tracking gains scaled: kp_scale={kp_scale} kd_scale={kd_scale} -> kp={track_kp} kd={track_kd}")

    def cspace_state() -> JointState:
        pos = torch.tensor([sim.cspace_position()], device="cuda", dtype=torch.float32)
        vel = torch.tensor([sim.cspace_velocity()], device="cuda", dtype=torch.float32)
        st = JointState.from_position(pos, joint_names=CSPACE_JOINT_NAMES)
        st.velocity = vel
        st.acceleration = torch.zeros_like(pos)
        return st

    def make_state(pos_np, vel_np) -> JointState:
        pos = torch.tensor([pos_np], device="cuda", dtype=torch.float32)
        vel = torch.tensor([vel_np], device="cuda", dtype=torch.float32)
        st = JointState.from_position(pos, joint_names=CSPACE_JOINT_NAMES)
        st.velocity = vel
        st.acceleration = torch.zeros_like(pos)
        return st

    # replan_from_belief mirrors rebot_ik_control.py's loop exactly: IK never
    # reads back real/simulated feedback for its next solve's current_state
    # -- it reseeds from ITS OWN previous solution (result.js_solution),
    # effectively assuming "I commanded X, so I'm at X now." MPC's loop
    # instead calls cspace_state() every tick, feeding back the REAL
    # (dynamics-lagged) simulated position/velocity -- with zero dynamics
    # model, MPC can't distinguish normal tracking lag from a real
    # disturbance, so it replans off of it regardless. This tests whether
    # that is itself the destabilizing mechanism: when True, MPC's next
    # optimize_action_sequence() call is seeded from its own last-commanded
    # (believed) state instead of the real simulated one, while the physical
    # simulation underneath still runs exactly as before.
    believed_position = sim.cspace_position().copy()
    believed_velocity = np.zeros(len(CSPACE_JOINT_NAMES))

    current_state = cspace_state()
    mpc.setup(current_state)

    kin_result = mpc.compute_kinematics(current_state)
    goal_tool_poses = kin_result.tool_poses.to_dict()
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(goal_tool_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=False,
    )

    viser_viz.set_joint_state(current_state.squeeze(0))
    for frame_name, pose in goal_tool_poses.items():
        if frame_name in viser_viz._control_frames:
            handle = viser_viz._control_frames[frame_name]
            handle.position = pose.position.cpu().squeeze().numpy()
            handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

    print(f"\nMIT+dynamics simulator running at http://localhost:{viz_port}")
    print("Drag the end-effector gizmo. This is PURE SIMULATION (Pinocchio dynamics + Damiao MIT law) — no hardware.")
    print("If instability reproduces here, it's a real MPC/MIT-loop problem, not a hardware issue.")
    if replan_from_belief:
        print("replan_from_belief=True: MPC replans from its OWN last-commanded state (IK-style), not real feedback.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False
    tick = 0
    report_every = max(1, int(round(1.0 / control_dt)))
    sim_substeps = max(1, round(control_dt / sim_dt))
    print(f"Integrating at sim_dt={sim_dt:g}s ({sim_substeps} substeps per {control_dt:g}s tick).")

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }
        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                mpc.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                pose_changed = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = target_poses
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            target_link_poses = {k.replace("target_", ""): v for k, v in target_poses.items()}
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(
                    target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                ),
                run_ik=False,
            )
            pose_changed = False

        current_state = make_state(believed_position, believed_velocity) if replan_from_belief else cspace_state()
        result = mpc.optimize_action_sequence(current_state)
        has_action = result.action_sequence is not None and result.action_sequence.position.shape[1] > 0

        report = (tick % report_every == 0)
        if report:
            pos_err = result.position_error
            err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
            print(
                f"  [tick {tick}] position error: {err_str}  action_valid: {has_action}  "
                f"q: " + "  ".join(f"{v:+.3f}" for v in sim.cspace_position())
            )
            if has_action:
                # Distinguishes "MPC itself is commanding an insane target"
                # (planning-side bug, e.g. FK periodicity letting it wander
                # off in joint space unpunished) from "MPC's target stays
                # sane but the simulated/tracked state runs away from it"
                # (tracking/dynamics-side bug, e.g. missing torque
                # saturation letting a large position error apply unphysical
                # torque).
                last_p_des = result.action_sequence.position[:, -1, :].squeeze(0).cpu().numpy()
                print("           p_des:    " + "  ".join(f"{v:+.3f}" for v in last_p_des))

        if has_action:
            # Use the FULL interpolated waypoint sequence (interpolation_steps
            # waypoints spanning this tick's optimization_dt) as a progressive
            # feedforward trajectory, instead of jumping straight to the last
            # waypoint and holding it static for all sim_substeps. Holding a
            # single distant setpoint is a much bigger step input (and thus a
            # bigger torque transient) than what MPC actually planned — MPC
            # planned a smooth path through all these waypoints, not a jump.
            # (The real hardware bridge currently also only uses [:, -1, :]
            # via its 200Hz resend thread — this is a candidate improvement
            # for both, being validated here in sim first.)
            seq_pos = result.action_sequence.position.squeeze(0).cpu().numpy()  # [n_wp, dof]
            seq_vel = result.action_sequence.velocity.squeeze(0).cpu().numpy()
            # gripper_joint1 is unconstrained by MPC's pose-tracking cost
            # (see GRIPPER_JOINT_IDX) -- override its planned trajectory to
            # hold fixed instead of following whatever ungoverned value MPC
            # happened to solve for it.
            seq_pos[:, GRIPPER_JOINT_IDX] = gripper_hold_position
            seq_vel[:, GRIPPER_JOINT_IDX] = 0.0
            n_wp = seq_pos.shape[0]
            substeps_per_wp = max(1, sim_substeps // n_wp)
            # For replan_from_belief: the state MPC believes it'll be in by
            # the end of this tick (its own last commanded waypoint) --
            # exactly IK's "I commanded X, so I'm at X now" assumption.
            believed_position = seq_pos[-1]
            believed_velocity = seq_vel[-1]
            # tau_g computed once per tick, reused for every waypoint's
            # substeps: gravity load barely changes over one tick, and
            # recomputing it at sim_dt (microsecond) rate is pure wasted cost.
            # no_gravity_comp tests whether compute_gravity_torque() itself
            # (e.g. producing a bad/large value for an unusual joint config)
            # is contributing to the instability -- IK's real-hardware
            # control doesn't lean on it for stability (its kp/kd are already
            # stiff enough to dominate gravity's contribution), so this
            # isolates whether MPC's crash depends on it too.
            tau_g = (
                np.zeros(len(CSPACE_JOINT_NAMES)) if no_gravity_comp
                else compute_gravity_torque(CSPACE_JOINT_NAMES, sim.cspace_position().tolist())
            )
            diverged = False
            for wp in range(n_wp):
                p_des = seq_pos[wp]
                # Vanilla curobo's own IK/reactive_control examples never run
                # a torque-based tracking law at all -- they set positions
                # directly, no velocity feedforward. MPC's v_des is optimized
                # under a purely kinematic model with zero torque/dynamics
                # awareness, so feeding it into kd*(v_des-v_actual) can inject
                # torque that has nothing to do with the real dynamics,
                # especially into joint6's low inertia. Test: ignore it,
                # track position only (kd damps against actual velocity) --
                # structurally identical to the proven-stable gravity-comp
                # law, just with a moving target.
                v_des = np.zeros_like(seq_vel[wp]) if ignore_mpc_velocity else seq_vel[wp]
                for _ in range(substeps_per_wp):
                    sim.step(p_des, v_des, sim_dt, kp=track_kp, kd=track_kd, tau_g=tau_g)
                    if not sim.is_finite():
                        print(
                            f"  [FATAL] NaN/Inf at tick {tick} (waypoint {wp}/{n_wp}) — simulator "
                            "diverged. Before blaming MPC: run --mode gravity first. If that stays "
                            "stable (it should, it's the known-good real-hardware control law) while "
                            "this doesn't, it's more likely numerical stiffness (kp=120 vs a "
                            "too-coarse integration dt) than a genuine MPC/MIT control problem — try "
                            "a smaller --sim-dt or --kp-scale/--kd-scale before concluding "
                            "instability is real."
                        )
                        diverged = True
                        break
                if diverged:
                    break
            if diverged:
                return
        else:
            print("  [warn] MPC returned no valid action_sequence this tick.")

        viser_viz.set_joint_state(cspace_state().squeeze(0))
        tick += 1
        time.sleep(control_dt)


# force_pos_ratio in MOTOR_MAP defaults to 0.07 (7% of rated effort) for
# every joint, but that value was only ever chosen/tuned for the GRIPPER's
# crush-avoidance behavior (see MotorSpec's docstring) -- it was never
# intended to cap the ARM joints' own torque authority, and 7% of an
# already modest rated effort (27 N*m shoulder/elbow, 7 N*m wrist) is
# nowhere near enough to accelerate the arm's own mass at a reasonable
# speed. Full rated torque is the right default for the arm; keep the
# gentle ratio only where crush-avoidance is the actual intent.
CSP_DEFAULT_FORCE_RATIO = {name: 1.0 for name in CSPACE_JOINT_NAMES}
CSP_DEFAULT_FORCE_RATIO["gripper_joint1"] = MOTOR_MAP["gripper_joint1"].force_pos_ratio


def run_csp_sim(
    viz_port: int = 8080, control_dt: float = 0.03, sim_dt: float = DEFAULT_SIM_DT,
    force_ratio_override: float = None, no_gravity_comp: bool = False,
):
    """Same MPC-driven closed loop as run_mit_sim(), but tracked with the
    approximated FORCE_POS ("CSP") servo (step_csp()) instead of the raw MIT
    PD law. This mode is doubly-saturated by construction (velocity capped
    at vlim, torque capped at force_ratio*effort) — a large or discontinuous
    p_des cannot itself produce unbounded torque, unlike step()'s MIT law
    which needed after-the-fact clamping added to stay bounded.
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML,
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        optimization_dt=control_dt,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    mpc = ModelPredictiveControl(config)

    initial_position = mpc.default_joint_position.clone().cpu().numpy()
    gripper_hold_position = initial_position[GRIPPER_JOINT_IDX]
    sim = MitDynamicsSimulator(initial_position, lock_gripper_at=gripper_hold_position)

    max_velocity = np.array([MOTOR_MAP[n].max_velocity for n in CSPACE_JOINT_NAMES])
    if force_ratio_override is not None:
        force_ratio = np.full(len(CSPACE_JOINT_NAMES), force_ratio_override)
    else:
        force_ratio = np.array([CSP_DEFAULT_FORCE_RATIO[n] for n in CSPACE_JOINT_NAMES])
    print(f"force_ratio: " + "  ".join(f"{n}={r:.2f}" for n, r in zip(CSPACE_JOINT_NAMES, force_ratio)))

    def cspace_state() -> JointState:
        pos = torch.tensor([sim.cspace_position()], device="cuda", dtype=torch.float32)
        vel = torch.tensor([sim.cspace_velocity()], device="cuda", dtype=torch.float32)
        st = JointState.from_position(pos, joint_names=CSPACE_JOINT_NAMES)
        st.velocity = vel
        st.acceleration = torch.zeros_like(pos)
        return st

    current_state = cspace_state()
    mpc.setup(current_state)

    kin_result = mpc.compute_kinematics(current_state)
    goal_tool_poses = kin_result.tool_poses.to_dict()
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(goal_tool_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=False,
    )

    viser_viz.set_joint_state(current_state.squeeze(0))
    for frame_name, pose in goal_tool_poses.items():
        if frame_name in viser_viz._control_frames:
            handle = viser_viz._control_frames[frame_name]
            handle.position = pose.position.cpu().squeeze().numpy()
            handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

    print(f"\nCSP (FORCE_POS-style)+dynamics simulator running at http://localhost:{viz_port}")
    print("Drag the end-effector gizmo. This is PURE SIMULATION (Pinocchio dynamics + approximated FORCE_POS law) — no hardware.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False
    tick = 0
    report_every = max(1, int(round(1.0 / control_dt)))
    sim_substeps = max(1, round(control_dt / sim_dt))
    print(f"Integrating at sim_dt={sim_dt:g}s ({sim_substeps} substeps per {control_dt:g}s tick).")

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }
        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                mpc.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                pose_changed = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = target_poses
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            target_link_poses = {k.replace("target_", ""): v for k, v in target_poses.items()}
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(
                    target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                ),
                run_ik=False,
            )
            pose_changed = False

        current_state = cspace_state()
        result = mpc.optimize_action_sequence(current_state)
        has_action = result.action_sequence is not None and result.action_sequence.position.shape[1] > 0

        report = (tick % report_every == 0)
        if report:
            pos_err = result.position_error
            err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
            print(
                f"  [tick {tick}] position error: {err_str}  action_valid: {has_action}  "
                f"q: " + "  ".join(f"{v:+.3f}" for v in sim.cspace_position())
            )
            if has_action:
                last_p_des = result.action_sequence.position[:, -1, :].squeeze(0).cpu().numpy()
                print("           p_des:    " + "  ".join(f"{v:+.3f}" for v in last_p_des))

        if has_action:
            # Same progressive-waypoint feedforward as run_mit_sim(), but
            # vlim per waypoint is derived from that waypoint's own v_des
            # magnitude (clamped to each joint's real max_velocity safety
            # ceiling) — mirroring exactly how the real bridge's
            # write_joint_command() computes vlim for send_force_pos()
            # (rebot_mpc_control.py's `vlim = abs(motor_vel) if motor_vel
            # != 0 else 1.0`, with motor_vel pre-clamped to spec.max_velocity).
            seq_pos = result.action_sequence.position.squeeze(0).cpu().numpy()  # [n_wp, dof]
            seq_vel = result.action_sequence.velocity.squeeze(0).cpu().numpy()
            # gripper_joint1 is unconstrained by MPC's pose-tracking cost
            # (see GRIPPER_JOINT_IDX) -- hold it fixed instead of following
            # whatever ungoverned value MPC happened to solve for it.
            seq_pos[:, GRIPPER_JOINT_IDX] = gripper_hold_position
            seq_vel[:, GRIPPER_JOINT_IDX] = 0.0
            n_wp = seq_pos.shape[0]
            substeps_per_wp = max(1, sim_substeps // n_wp)
            tau_g = (
                np.zeros(len(CSPACE_JOINT_NAMES)) if no_gravity_comp
                else compute_gravity_torque(CSPACE_JOINT_NAMES, sim.cspace_position().tolist())
            )
            diverged = False
            for wp in range(n_wp):
                p_des = seq_pos[wp]
                v_clamped = np.clip(seq_vel[wp], -max_velocity, max_velocity)
                vlim = np.where(v_clamped != 0.0, np.abs(v_clamped), 1.0)
                for _ in range(substeps_per_wp):
                    sim.step_csp(p_des, sim_dt, vlim=vlim, force_ratio=force_ratio, tau_g=tau_g)
                    if not sim.is_finite():
                        print(f"  [FATAL] NaN/Inf at tick {tick} (waypoint {wp}/{n_wp}) — simulator diverged.")
                        diverged = True
                        break
                if diverged:
                    break
            if diverged:
                return
        else:
            print("  [warn] MPC returned no valid action_sequence this tick.")

        viser_viz.set_joint_state(cspace_state().squeeze(0))
        tick += 1
        time.sleep(control_dt)


def main():
    parser = argparse.ArgumentParser(description="Simulate Damiao MIT control + real dynamics, closed-loop with MPC")
    parser.add_argument(
        "--mode", choices=["mpc", "gravity", "csp"], default="mpc",
        help="'gravity': validation baseline, known-good gravity-comp-only control law, no MPC. "
        "'mpc': full MPC-driven simulation tracked via the raw MIT PD law (the one that reproduced "
        "the real-hardware runaway). 'csp': same MPC-driven loop, but tracked via the approximated "
        "FORCE_POS/CSP-style servo (step_csp()), which is doubly-saturated (velocity- and "
        "torque-limited) by construction instead of needing after-the-fact clamping. Run 'gravity' "
        "first to confirm the simulator itself is stable/trustworthy before trusting 'mpc'/'csp'.",
    )
    parser.add_argument("--viz-port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--control-dt", type=float, default=0.03, help="MPC solve/goal-check period (s)")
    parser.add_argument(
        "--sim-dt", type=float, default=DEFAULT_SIM_DT,
        help="Fixed physics integration timestep (s). joint6 (wrist roll) has ~10-100x lower "
        "effective inertia than the other joints, which makes semi-implicit Euler integration "
        "numerically stiff at coarse dt — dt=2e-3 diverges to NaN within a few steps; the "
        "default 5e-5 was confirmed stable for 2000 steps in isolation. Substeps per tick are "
        "derived as control_dt/sim_dt, so a smaller value means more substeps (slower to run, "
        "but not more 'real'-looking — this only affects integration accuracy, not the control "
        "law being tested).",
    )
    parser.add_argument(
        "--kp-scale", type=float, default=1.0,
        help="--mode mpc only: scale factor on MOTOR_MAP's real tracking kp gains (120/18). "
        "MPC uses much stiffer gains than gravity-comp's uniform kp=2 (confirmed stable); try "
        "e.g. 0.25 to test whether reducing tracking stiffness removes the tick-23 NaN "
        "divergence, which would point at cross-joint coupling (via the mass matrix) from the "
        "stiffer gains exciting joint6's low inertia, not a genuine planning bug.",
    )
    parser.add_argument(
        "--kd-scale", type=float, default=1.0,
        help="--mode mpc only: scale factor on MOTOR_MAP's real tracking kd gains (8/2). See --kp-scale.",
    )
    parser.add_argument(
        "--force-ratio", type=float, default=None,
        help="--mode csp only: override force_ratio (fraction of rated effort) for ALL cspace joints, "
        "including the gripper. Default (omit this flag) is 1.0 (full torque) for the arm joints and "
        "MOTOR_MAP's own 0.07 for the gripper (crush-avoidance). Lower this to test how much torque "
        "authority is actually needed for stable tracking.",
    )
    parser.add_argument(
        "--ignore-mpc-velocity", action="store_true",
        help="--mode mpc only: always track v_des=0 (position only; kd damps against actual velocity, "
        "not MPC's planned velocity). MPC's velocity output is optimized under a purely kinematic "
        "model with no torque/dynamics awareness -- this tests whether feeding it into the MIT law's "
        "kd*(v_des-v_actual) term is itself injecting bad torque, vs. tracking position only like the "
        "proven-stable gravity-comp law (but with a moving target).",
    )
    parser.add_argument(
        "--no-gravity-comp", action="store_true",
        help="--mode mpc/csp only: zero out the gravity feedforward torque entirely. IK's real-hardware "
        "control doesn't lean on it for stability (kp/kd there are already stiff enough to dominate "
        "gravity's contribution) -- this isolates whether compute_gravity_torque() itself (e.g. a bad "
        "value for an unusual joint config) is contributing to the instability.",
    )
    parser.add_argument(
        "--replan-from-belief", action="store_true",
        help="--mode mpc only: MPC replans from its OWN last-commanded state instead of real/simulated "
        "feedback -- mirroring rebot_ik_control.py exactly, which reseeds its next solve from its own "
        "previous js_solution and never reads back real state. MPC's loop normally calls cspace_state() "
        "every tick, feeding back the REAL (dynamics-lagged) position/velocity; with zero dynamics "
        "model MPC can't tell normal tracking lag from a real disturbance, so it replans off of it "
        "regardless. This isolates whether that feedback-from-reality is itself the destabilizing "
        "mechanism. The physical simulation underneath is unchanged either way.",
    )
    args = parser.parse_args()
    if args.mode == "gravity":
        run_gravity_sim(viz_port=args.viz_port, control_dt=args.control_dt, sim_dt=args.sim_dt)
    elif args.mode == "csp":
        run_csp_sim(
            viz_port=args.viz_port, control_dt=args.control_dt, sim_dt=args.sim_dt,
            force_ratio_override=args.force_ratio, no_gravity_comp=args.no_gravity_comp,
        )
    else:
        run_mit_sim(
            viz_port=args.viz_port, control_dt=args.control_dt, sim_dt=args.sim_dt,
            kp_scale=args.kp_scale, kd_scale=args.kd_scale,
            ignore_mpc_velocity=args.ignore_mpc_velocity, no_gravity_comp=args.no_gravity_comp,
            replan_from_belief=args.replan_from_belief,
        )


if __name__ == "__main__":
    main()
