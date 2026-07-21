# SPDX-License-Identifier: Apache-2.0
"""Drive the reBot B601-DM arm with cuRobo's GPU motion planner, with a Viser viewer.

Mirrors ``curobo/examples/getting_started/motion_planning.py``'s
``interactive_motion_planning()`` (same solver config, same Move/Grasp
buttons, same trajectory-plot GUI panel), but drives the REAL arm's joints
through each planned waypoint via ``RebotArmBridge``'s proven-safe open-loop
execution (background command thread resending the latest target, same as
``rebot_ik_control.py``) instead of only updating the Viser display.

Unlike ``rebot_mpc_control.py``'s MPC, ``MotionPlanner`` solves a COMPLETE
trajectory ONCE per goal (a smooth, collision-checked path from the current
joint state to the goal pose), which is then executed open-loop -- there is
no continuous re-planning from real feedback every tick. That distinction
matters: the MPC/MIT closed-loop instability investigated at length in
``rebot_mit_sim.py`` came from MPC (which has no dynamics model) replanning
off real, tracking-lagged feedback every tick and being unable to tell
normal lag from a real disturbance. A one-shot planned trajectory, walked
through open-loop like IK's solution, doesn't have that failure mode.

``gripper_joint1`` is unconstrained by the pose-tracking cost (``gripper_link``,
the tracked tool frame, is upstream of the finger joints) so its planned
trajectory is overridden to hold at its current position throughout Move;
Grasp explicitly closes/opens it around the grasp phase via a direct
commanded move instead of trusting whatever ungoverned value the planner
solved for it.

Usage:

.. code-block:: bash

   # Safe: no hardware touched.
   python rebot_motion_planning.py

   # Real hardware, once rebot_mpc_control.MOTOR_MAP is calibrated (shared).
   python rebot_motion_planning.py --live --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time

import numpy as np
import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

from rebot_mpc_control import (
    MOTOR_MAP,
    ROBOT_YML,
    RebotArmBridge,
    _tool_position,
    cap_cartesian_step,
    compute_gravity_torque,
    home_to_pose,
    send_smoothed_segment,
)
from rebot_mit_sim import CSPACE_JOINT_NAMES, DEFAULT_SIM_DT, GRIPPER_JOINT_IDX, MitDynamicsSimulator

GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 0.06  # inside the URDF's 0-0.0715m range, leaves a small grip margin


def _create_trajectory_image(trajectory, joint_names, title=""):
    """Render a joint trajectory as a PNG image array for the Viser GUI panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = trajectory.squeeze(0)
    pos = np.atleast_2d(traj.position[0].cpu().numpy())
    dt_val = traj.dt.item() if traj.dt is not None else 0.02
    t = np.arange(pos.shape[0]) * dt_val

    vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None

    n_plots = 1 + (vel is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(5, 2 * n_plots), dpi=100, sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_data = [(pos, "Position (rad)")]
    if vel is not None:
        plot_data.append((vel, "Velocity (rad/s)"))

    for ax, (data, ylabel) in zip(axes, plot_data):
        for j in range(data.shape[1]):
            label = joint_names[j] if j < len(joint_names) else f"J{j}"
            if len(label) > 8:
                label = label[:6] + ".."
            ax.plot(t, data[:, j], linewidth=1.0, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    from PIL import Image
    img_array = np.array(Image.open(buf))
    plt.close(fig)
    buf.close()
    return img_array


def _hold_gripper(trajectory, hold_position: float):
    """Overwrite gripper_joint1's planned position/velocity to hold fixed.

    gripper_joint1 doesn't affect the tracked tool pose, so the planner has
    no reason to move it sensibly -- the same "unconstrained DOF" issue
    diagnosed at length for MPC in rebot_mit_sim.py. Held fixed here rather
    than trusting its output.
    """
    trajectory.position[..., GRIPPER_JOINT_IDX] = hold_position
    if trajectory.velocity is not None:
        trajectory.velocity[..., GRIPPER_JOINT_IDX] = 0.0
    return trajectory


def run_interactive_motion_planning(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    viz_port: int = 8080,
    observe_only: bool = False,
    scene_model: str = "collision_test.yml",
    sim_mode: bool = False,
    enable_mpc_fallback: bool = True,
):
    """Drag the EE gizmo, click Move to plan+execute a collision-free trajectory.

    Click Grasp to plan+execute an approach/grasp/lift sequence, closing the
    gripper (a direct commanded move, not part of the planned trajectory --
    see _hold_gripper()) between the grasp and lift phases.

    If Move can't find a full plan (target out of a one-shot reachable
    range, or in collision), it falls back to a continuous MPC tracking
    loop targeting the same pose -- MPC is a local optimizer that minimizes
    pose error even when it can't fully satisfy it, so it extends toward
    the goal as far as actually reachable instead of just giving up. Uses
    the replan-from-belief fix confirmed stable in rebot_mit_sim.py (MPC
    replans from its own last-commanded state, not real feedback, since it
    has no dynamics model and can't tell normal tracking lag from a real
    disturbance).

    sim_mode (dry-run only) executes every planned trajectory through
    rebot_mit_sim.MitDynamicsSimulator -- the same Damiao MIT PD law +
    Pinocchio forward dynamics + gravity feedforward + torque/joint limits
    used to investigate the MPC/MIT instability -- instead of teleporting
    the display straight to each waypoint. This shows whether the real
    motor control loop can actually track a motion-planned trajectory
    before ever touching hardware.
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = MotionPlannerCfg.create(robot=ROBOT_YML, scene_model=scene_model, max_goalset=10)
    planner = MotionPlanner(config)

    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    bridge = RebotArmBridge(
        joint_names=planner.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=planner.default_joint_state.position.clone().unsqueeze(0),
        observe_only=observe_only,
    )

    is_moving = False
    # Defined before try/finally so `finally` can always safely reference them,
    # even if setup fails before the MPC-fallback machinery below is reached.
    mpc_stop_event = threading.Event()
    auto_track_stop_event = threading.Event()

    try:
        do_home = not observe_only
        if do_home:
            home_to_pose(bridge, planner.default_joint_state.position.clone().unsqueeze(0))
        real_state = bridge.read_joint_state()
        bridge.set_target(real_state.position, torch.zeros_like(real_state.position))

        # Batched [1, dof] throughout, matching curobo's own
        # interactive_motion_planning() convention for MotionPlanner calls
        # (unlike InverseKinematics, get_active_js()/plan_pose() here take
        # batched state directly -- no squeeze/unsqueeze dance needed).
        current_state = JointState.from_position(real_state.position.clone(), joint_names=bridge.joint_names)

        kin_state = planner.compute_kinematics(real_state)
        goal_tool_poses = kin_state.tool_poses.to_dict()

        print("Warming up motion planner...")
        planner.warmup(enable_graph=True, num_warmup_iterations=5)
        bridge.start_command_thread()

        # Snap the draggable gizmo (placed at FK(default_joint_state) at
        # construction time) and rendered robot to the real starting pose --
        # same fix as every other rebot_*.py interactive script.
        viser_viz.set_joint_state(real_state.squeeze(0))
        for frame_name, pose in goal_tool_poses.items():
            if frame_name in viser_viz._control_frames:
                handle = viser_viz._control_frames[frame_name]
                handle.position = pose.position.cpu().squeeze().numpy()
                handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

        gripper_hold_position = real_state.position.squeeze(0)[GRIPPER_JOINT_IDX].item()

        # sim_mode: run every planned trajectory through the real MIT
        # dynamics simulator instead of teleporting the display straight to
        # each waypoint (dry-run only -- real motors are never touched
        # either way, this only changes what the Viser display shows).
        mit_sim = None
        control_dt = 0.02
        if sim_mode:
            mit_sim = MitDynamicsSimulator(real_state.position.squeeze(0).cpu().numpy())

        # MPC fallback: used by on_move() when plan_pose can't find a full
        # solution. Set up and warmed up (its CUDA graph captured) here,
        # BEFORE the background command thread starts below, matching the
        # established rule that CUDA-graph-capturing calls must happen
        # before any concurrent CUDA-free background thread claims the
        # stream -- see RebotArmBridge's own note on this.
        mpc_config = ModelPredictiveControlCfg.create(
            robot=ROBOT_YML,
            scene_model=scene_model,
            use_cuda_graph=True,
            optimization_dt=0.03,
            interpolation_steps=4,
            optimizer_collision_activation_distance=0.03,
        )
        mpc = ModelPredictiveControl(mpc_config)
        mpc_warmup_state = JointState.from_position(
            real_state.position.clone(), joint_names=mpc.joint_names,
        )
        mpc_warmup_state.velocity = torch.zeros_like(mpc_warmup_state.position)
        mpc_warmup_state.acceleration = torch.zeros_like(mpc_warmup_state.position)
        mpc.setup(mpc_warmup_state)
        mpc_kin = mpc.compute_kinematics(mpc_warmup_state)
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                mpc_kin.tool_poses.to_dict(), ordered_tool_frames=mpc.tool_frames, num_goalset=1,
            ),
            run_ik=False,
        )
        mpc.optimize_action_sequence(mpc_warmup_state)  # captures the CUDA graph now, not mid-fallback
        mpc_thread_holder = [None]

        server = viser_viz._server
        traj_plot = server.gui.add_image(
            _create_trajectory_image(current_state, planner.joint_names, title="No trajectory yet"),
            label="Joint Trajectory",
            format="png",
        )

        def update_obstacles():
            for k in obstacle_frames.keys():
                new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
                if new_pose != old_obstacle_poses[k]:
                    planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                    old_obstacle_poses[k] = new_pose.clone()

        def execute_trajectory(trajectory, interrupt_check=None):
            """interrupt_check: optional no-arg callable polled before each
            waypoint; if it returns True, execution stops early (used by
            auto_track_loop to break off mid-trajectory the instant the live
            gizmo target changes again, rather than waiting for the full
            planned trajectory -- which can be a second or more -- to
            finish). current_state is resynced to wherever execution
            actually stopped, not the trajectory's intended endpoint.
            """
            nonlocal current_state
            traj = trajectory.squeeze(0)
            n_steps = traj.position.shape[-2]
            interp_dt = traj.dt.item() if traj.dt is not None else 0.02
            sim_substeps = max(1, round(interp_dt / DEFAULT_SIM_DT))

            # Real-hardware path: cap each waypoint's implied Cartesian tool
            # speed and send it as an acceleration/jerk-limited ramp, same
            # as rebot_mpc_control.py/rebot_hybrid_control.py. TrajOpt's own
            # trajectory is already smoothness-costed (unlike MPC's raw
            # per-tick output), so this should rarely need to stretch --
            # it's a safety net for this being the first real-hardware use
            # of the motion planner, not an expected steady-state behavior.
            prev_position = current_state.position
            prev_velocity = (
                current_state.velocity if current_state.velocity is not None
                else torch.zeros_like(prev_position)
            )
            prev_accel = torch.zeros_like(prev_position)
            prev_tool_position = None if mit_sim is not None else _tool_position(planner, prev_position)

            for i in range(n_steps):
                if interrupt_check is not None and interrupt_check():
                    break
                pos = traj.position[0, i, :].unsqueeze(0)
                vel = (
                    traj.velocity[0, i, :].unsqueeze(0) if traj.velocity is not None
                    else torch.zeros_like(pos)
                )
                if mit_sim is not None:
                    # Step the REAL MIT dynamics toward this waypoint instead
                    # of teleporting the display straight to it -- shows
                    # whether the control loop actually tracks the planned
                    # trajectory (position/velocity PD + gravity feedforward
                    # + torque/joint limits), not just an idealized path.
                    p_des = pos.squeeze(0).cpu().numpy()
                    v_des = vel.squeeze(0).cpu().numpy()
                    tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, mit_sim.cspace_position().tolist())
                    for _ in range(sim_substeps):
                        mit_sim.step(p_des, v_des, DEFAULT_SIM_DT, tau_g=tau_g)
                    display_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                    viser_viz.set_joint_state(
                        JointState.from_position(display_pos, joint_names=traj.joint_names).squeeze(0)
                    )
                    time.sleep(interp_dt)
                else:
                    wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
                        planner, prev_position, prev_tool_position, pos, vel, interp_dt,
                    )
                    prev_accel = send_smoothed_segment(
                        bridge, prev_position, prev_velocity, prev_accel,
                        wp_position, wp_velocity, interp_dt,
                    )
                    prev_position = wp_position
                    prev_velocity = wp_velocity
                    viser_viz.set_joint_state(
                        JointState.from_position(wp_position, joint_names=traj.joint_names).squeeze(0)
                    )
            if mit_sim is not None:
                final_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                current_state = JointState.from_position(final_pos, joint_names=traj.joint_names)
            else:
                # prev_position is wherever execution actually ended up --
                # the trajectory's final index unless interrupt_check cut it
                # short, and possibly adjusted from traj's own values by
                # cap_cartesian_step along the way.
                current_state = JointState.from_position(prev_position, joint_names=traj.joint_names)
                current_state.velocity = prev_velocity

        def move_gripper(target_position: float, duration: float = 1.0):
            """Direct open-loop gripper move, holding the arm at its current pose."""
            nonlocal current_state
            arm_position = current_state.position.clone()
            steps = max(1, int(duration / 0.02))
            start = arm_position[0, GRIPPER_JOINT_IDX].item()
            pos = arm_position.clone()
            sim_substeps = max(1, round(0.02 / DEFAULT_SIM_DT))
            for i in range(steps + 1):
                frac = i / steps
                pos = arm_position.clone()
                pos[0, GRIPPER_JOINT_IDX] = start + (target_position - start) * frac
                if mit_sim is not None:
                    p_des = pos.squeeze(0).cpu().numpy()
                    v_des = np.zeros(len(CSPACE_JOINT_NAMES))
                    tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, mit_sim.cspace_position().tolist())
                    for _ in range(sim_substeps):
                        mit_sim.step(p_des, v_des, DEFAULT_SIM_DT, tau_g=tau_g)
                    display_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                    viser_viz.set_joint_state(
                        JointState.from_position(display_pos, joint_names=bridge.joint_names).squeeze(0)
                    )
                else:
                    bridge.set_target(pos, torch.zeros_like(pos))
                    viser_viz.set_joint_state(
                        JointState.from_position(pos, joint_names=bridge.joint_names).squeeze(0)
                    )
                time.sleep(0.02)
            if mit_sim is not None:
                final_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                current_state = JointState.from_position(final_pos, joint_names=bridge.joint_names)
            else:
                current_state = JointState.from_position(pos, joint_names=bridge.joint_names)

        def go_home_sync():
            """Open-loop move to the default/home joint configuration.

            Manual recovery for when the arm gets stuck at a near-singular
            pose during MPC tracking (a known failure mode of gradient-based
            local optimizers -- the Jacobian becomes ill-conditioned and
            progress stalls), and also used internally whenever a target
            turns out to only be solvable by re-seeding the plan from home.
            """
            nonlocal current_state
            home_target = planner.default_joint_state.position.clone().unsqueeze(0)
            if mit_sim is not None:
                start_pos = current_state.position.clone()
                duration = 3.0
                steps = max(1, int(duration / 0.02))
                sim_substeps = max(1, round(0.02 / DEFAULT_SIM_DT))
                for i in range(1, steps + 1):
                    frac = i / steps
                    pos = start_pos + frac * (home_target - start_pos)
                    p_des = pos.squeeze(0).cpu().numpy()
                    v_des = np.zeros(len(CSPACE_JOINT_NAMES))
                    tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, mit_sim.cspace_position().tolist())
                    for _ in range(sim_substeps):
                        mit_sim.step(p_des, v_des, DEFAULT_SIM_DT, tau_g=tau_g)
                    display_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                    viser_viz.set_joint_state(
                        JointState.from_position(display_pos, joint_names=bridge.joint_names).squeeze(0)
                    )
                    time.sleep(0.02)
                final_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                current_state = JointState.from_position(final_pos, joint_names=bridge.joint_names)
            else:
                home_to_pose(bridge, home_target)
                display_position = home_target if dry_run else bridge.read_joint_state().position.clone()
                viser_viz.set_joint_state(
                    JointState.from_position(display_position, joint_names=bridge.joint_names).squeeze(0)
                )
                current_state = JointState.from_position(display_position, joint_names=bridge.joint_names)

        def try_plan_pose(target_poses, start_state=None, max_attempts=3):
            """Try plan_pose seeded from the current state; if that fails,
            retry seeded from the home/default pose instead. A pose can be
            genuinely reachable while still failing to plan from the
            arm's CURRENT configuration if that configuration is awkward or
            near a singularity for this particular target -- re-seeding from
            a known-good configuration (home) resolves that class of failure
            without meaning the target itself is actually out of range.

            Returns (result, needs_home_first). needs_home_first is True
            when the caller must physically move to home (go_home_sync())
            before executing the returned trajectory, since it was planned
            assuming that starting configuration.

            start_state defaults to the tracked current_state; pass one
            explicitly when checking from a different belief (e.g. the MPC
            fallback loop's own continuously-updated believed position,
            which current_state does NOT track while the loop is running).
            """
            if start_state is None:
                start_state = current_state.clone()
            active_js = planner.kinematics.get_active_js(start_state)
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js, use_implicit_goal=True, max_attempts=max_attempts,
            )
            if result is not None and result.success.any():
                return result, False

            home_js = JointState.from_position(
                planner.default_joint_state.position.clone().unsqueeze(0), joint_names=planner.joint_names,
            )
            home_active_js = planner.kinematics.get_active_js(home_js)
            home_result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                home_active_js, use_implicit_goal=True, max_attempts=max_attempts,
            )
            if home_result is not None and home_result.success.any():
                return home_result, True

            return None, False

        def mpc_fallback_loop(control_dt=0.03, replan_check_every=10):
            """Continuously track the live gizmo pose via MPC as a fallback
            for Move: MPC minimizes pose error even when it can't fully
            satisfy it, so it extends toward the goal as far as actually
            reachable, instead of Move's all-or-nothing plan_pose failure.

            Replans from its own last-commanded state (believed_position/
            velocity), never from real/simulated feedback -- the fix
            confirmed stable in rebot_mit_sim.py, since MPC has no dynamics
            model and can't distinguish real tracking lag from a genuine
            disturbance. gripper_joint1 is held fixed for the same
            unconstrained-DOF reason as everywhere else in this project.

            Every replan_check_every ticks, opportunistically checks whether
            the live target has become solvable by a full one-shot plan
            (e.g. the user dragged the gizmo somewhere reachable while MPC
            was extending toward an out-of-range one, or the arm's own
            tracking progress carried it into a configuration the target IS
            solvable from). A full plan is preferable to continuous
            best-effort tracking whenever one is actually available, so this
            switches over to it and ends the fallback loop.
            """
            nonlocal is_moving
            believed_position = current_state.position.clone()
            believed_velocity = torch.zeros_like(believed_position)
            prev_accel = torch.zeros_like(believed_position)
            sim_substeps = max(1, round(control_dt / DEFAULT_SIM_DT))
            tick = 0

            while not mpc_stop_event.is_set():
                update_obstacles()
                target_poses = viser_viz.get_control_frame_pose()

                if tick % replan_check_every == 0:
                    belief_js = JointState.from_position(believed_position, joint_names=planner.joint_names)
                    plan_result, needs_home_first = try_plan_pose(target_poses, start_state=belief_js, max_attempts=1)
                    if plan_result is not None:
                        print(
                            "  [motion-planning] target now solvable -- switching from MPC tracking "
                            + ("(via home pose) " if needs_home_first else "")
                            + "to a generated plan."
                        )
                        is_moving = True
                        try:
                            if needs_home_first:
                                go_home_sync()
                            interp = plan_result.get_interpolated_plan()
                            _hold_gripper(interp, gripper_hold_position)
                            traj_plot.image = _create_trajectory_image(
                                interp, planner.joint_names,
                                title=f"Pose Plan  |  {plan_result.total_time:.3f}s",
                            )
                            execute_trajectory(interp)
                        finally:
                            is_moving = False
                        return
                tick += 1

                mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(
                        target_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                    ),
                    run_ik=False,
                )

                belief_state = JointState.from_position(believed_position, joint_names=mpc.joint_names)
                belief_state.velocity = believed_velocity
                belief_state.acceleration = torch.zeros_like(believed_position)
                result = mpc.optimize_action_sequence(belief_state)
                has_action = (
                    result.action_sequence is not None and result.action_sequence.position.shape[1] > 0
                )
                if has_action:
                    # Use the FULL interpolation_steps waypoint sequence
                    # progressively (not just the last one held statically
                    # for the whole tick), cap each waypoint's implied
                    # Cartesian tool speed, and send as an acceleration/
                    # jerk-limited ramp -- same layered fix as
                    # rebot_mpc_control.py/rebot_hybrid_control.py, needed
                    # here for the same reason (a fixed one-shot jump per
                    # tick produces jerky/high-acceleration bursts live).
                    seq_pos = result.action_sequence.position.clone()  # [1, n_wp, dof]
                    seq_vel = result.action_sequence.velocity.clone()
                    seq_pos[:, :, GRIPPER_JOINT_IDX] = gripper_hold_position
                    seq_vel[:, :, GRIPPER_JOINT_IDX] = 0.0
                    n_wp = seq_pos.shape[1]
                    wp_dt = control_dt / n_wp

                    prev_position = believed_position
                    prev_velocity = believed_velocity
                    prev_tool_position = None if mit_sim is not None else _tool_position(mpc, prev_position)
                    for wp in range(n_wp):
                        wp_position, wp_velocity = seq_pos[:, wp, :], seq_vel[:, wp, :]
                        if mit_sim is None:
                            wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
                                mpc, prev_position, prev_tool_position, wp_position, wp_velocity, wp_dt,
                            )

                        if mit_sim is not None:
                            p_des = wp_position.squeeze(0).cpu().numpy()
                            v_des = wp_velocity.squeeze(0).cpu().numpy()
                            tau_g = compute_gravity_torque(
                                CSPACE_JOINT_NAMES, mit_sim.cspace_position().tolist(),
                            )
                            for _ in range(sim_substeps):
                                mit_sim.step(p_des, v_des, DEFAULT_SIM_DT, tau_g=tau_g)
                            display_pos = torch.tensor(
                                [mit_sim.cspace_position()], device="cuda", dtype=torch.float32,
                            )
                            viser_viz.set_joint_state(
                                JointState.from_position(display_pos, joint_names=mpc.joint_names).squeeze(0)
                            )
                            time.sleep(wp_dt)
                        else:
                            prev_accel = send_smoothed_segment(
                                bridge, prev_position, prev_velocity, prev_accel,
                                wp_position, wp_velocity, wp_dt,
                            )
                            viser_viz.set_joint_state(
                                JointState.from_position(wp_position, joint_names=mpc.joint_names).squeeze(0)
                            )
                        prev_position = wp_position
                        prev_velocity = wp_velocity

                    believed_position = prev_position
                    believed_velocity = prev_velocity
                else:
                    time.sleep(control_dt)

        def stop_mpc_fallback():
            nonlocal current_state
            mpc_stop_event.set()
            t = mpc_thread_holder[0]
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
            mpc_thread_holder[0] = None
            mpc_stop_event.clear()
            # Resync current_state to wherever the fallback actually left the
            # arm (real feedback if live, the simulator's state in sim_mode)
            # -- mpc_fallback_loop only tracked its own believed state, not
            # this nonlocal, while it was running.
            if mit_sim is not None:
                final_pos = torch.tensor([mit_sim.cspace_position()], device="cuda", dtype=torch.float32)
                current_state = JointState.from_position(final_pos, joint_names=bridge.joint_names)
            elif not dry_run:
                real = bridge.read_joint_state()
                current_state = JointState.from_position(real.position.clone(), joint_names=bridge.joint_names)

        def start_mpc_fallback():
            mpc_stop_event.clear()
            t = threading.Thread(target=mpc_fallback_loop, daemon=True)
            mpc_thread_holder[0] = t
            t.start()

        def auto_track_loop(poll_dt=0.03):
            """Continuously replan and execute via motion planning whenever
            the live gizmo target changes -- plan_pose only takes ~30ms, so
            there's no need to gate it behind a manual Move click. Falls
            back to MPC tracking (mpc_fallback_loop) when the live target
            isn't (yet) solvable; that loop's own periodic check switches
            back to a generated plan automatically once it becomes solvable
            -- same switch logic as a manual Move, just continuous instead
            of button-triggered.

            Executes each plan via execute_trajectory's interrupt_check, so
            if the gizmo moves again mid-execution (rather than waiting for
            a potentially multi-second trajectory to finish), this breaks
            off immediately and replans from wherever it actually got to --
            matching the responsiveness MPC-only tracking used to have.
            """
            nonlocal is_moving
            last_target_poses = None
            while not auto_track_stop_event.is_set():
                if is_moving or (mpc_thread_holder[0] is not None and mpc_thread_holder[0].is_alive()):
                    time.sleep(poll_dt)
                    continue

                target_poses = viser_viz.get_control_frame_pose()
                changed = last_target_poses is None or any(
                    target_poses[k] != last_target_poses[k] for k in target_poses
                )
                if not changed:
                    time.sleep(poll_dt)
                    continue
                last_target_poses = {k: v.clone() for k, v in target_poses.items()}

                is_moving = True
                try:
                    update_obstacles()
                    result, needs_home_first = try_plan_pose(target_poses)
                    if result is not None:
                        if needs_home_first:
                            print(
                                "  [motion-planning] current pose can't reach target -- "
                                "returning to home pose first..."
                            )
                            go_home_sync()
                        interp = result.get_interpolated_plan()
                        _hold_gripper(interp, gripper_hold_position)
                        traj_plot.image = _create_trajectory_image(
                            interp, planner.joint_names, title=f"Pose Plan  |  {result.total_time:.3f}s",
                        )
                        execute_trajectory(
                            interp,
                            interrupt_check=lambda: any(
                                viser_viz.get_control_frame_pose()[k] != last_target_poses[k]
                                for k in last_target_poses
                            ),
                        )
                    elif enable_mpc_fallback:
                        print(
                            "  [motion-planning] target unreachable or in collision -- "
                            "falling back to MPC (best-effort tracking, extends as far as reachable)..."
                        )
                        start_mpc_fallback()
                    else:
                        print(
                            "  [motion-planning] target unreachable or in collision, and MPC fallback "
                            "is disabled (--no-mpc-fallback) -- waiting for a reachable target."
                        )
                finally:
                    is_moving = False

        def on_move(_):
            nonlocal is_moving
            if is_moving:
                return

            def plan_and_execute():
                nonlocal is_moving
                is_moving = True
                try:
                    stop_mpc_fallback()
                    update_obstacles()
                    target_poses = viser_viz.get_control_frame_pose()
                    result, needs_home_first = try_plan_pose(target_poses)
                    if result is not None:
                        if needs_home_first:
                            print(
                                "  [motion-planning] current pose can't reach target -- "
                                "returning to home pose first..."
                            )
                            go_home_sync()
                        interp = result.get_interpolated_plan()
                        _hold_gripper(interp, gripper_hold_position)
                        traj_plot.image = _create_trajectory_image(
                            interp, planner.joint_names, title=f"Pose Plan  |  {result.total_time:.3f}s",
                        )
                        execute_trajectory(interp)
                    elif enable_mpc_fallback:
                        print(
                            "  [motion-planning] planning failed -- target unreachable or in collision. "
                            "Falling back to MPC (best-effort tracking, extends as far as reachable)..."
                        )
                        start_mpc_fallback()
                    else:
                        print(
                            "  [motion-planning] planning failed -- target unreachable or in collision, "
                            "and MPC fallback is disabled (--no-mpc-fallback)."
                        )
                finally:
                    is_moving = False

            threading.Thread(target=plan_and_execute, daemon=True).start()

        def on_grasp(_):
            nonlocal is_moving
            if is_moving:
                return

            def plan_grasp_and_execute():
                nonlocal is_moving
                is_moving = True
                try:
                    stop_mpc_fallback()
                    update_obstacles()
                    target_poses = viser_viz.get_control_frame_pose()
                    active_js = planner.kinematics.get_active_js(current_state.clone())
                    grasp_poses = GoalToolPose.from_poses(target_poses, num_goalset=1)
                    result = planner.plan_grasp(
                        grasp_poses,
                        active_js,
                        plan_approach_to_grasp=True,
                        plan_grasp_to_lift=True,
                        grasp_lift_in_tool_frame=True,
                    )
                    if result is None or result.success is None or not result.success.any():
                        print("  [motion-planning] grasp planning failed.")
                        return

                    if result.approach_interpolated_trajectory is not None:
                        _hold_gripper(result.approach_interpolated_trajectory, gripper_hold_position)
                        traj_plot.image = _create_trajectory_image(
                            result.approach_interpolated_trajectory, planner.joint_names, title="Approach",
                        )
                        execute_trajectory(result.approach_interpolated_trajectory)

                    if result.grasp_interpolated_trajectory is not None:
                        _hold_gripper(result.grasp_interpolated_trajectory, gripper_hold_position)
                        traj_plot.image = _create_trajectory_image(
                            result.grasp_interpolated_trajectory, planner.joint_names, title="Grasp approach",
                        )
                        execute_trajectory(result.grasp_interpolated_trajectory)

                    print("  [motion-planning] closing gripper...")
                    move_gripper(GRIPPER_CLOSED_POSITION)

                    if result.lift_interpolated_trajectory is not None:
                        _hold_gripper(result.lift_interpolated_trajectory, GRIPPER_CLOSED_POSITION)
                        traj_plot.image = _create_trajectory_image(
                            result.lift_interpolated_trajectory, planner.joint_names, title="Lift",
                        )
                        execute_trajectory(result.lift_interpolated_trajectory)
                finally:
                    is_moving = False

            threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

        def on_open_gripper(_):
            if is_moving:
                return
            threading.Thread(target=move_gripper, args=(GRIPPER_OPEN_POSITION,), daemon=True).start()

        def on_stop_tracking(_):
            threading.Thread(target=stop_mpc_fallback, daemon=True).start()

        def on_home(_):
            nonlocal is_moving
            if is_moving:
                print("  [motion-planning] busy executing a trajectory -- ignoring Go Home click.")
                return

            def go_home_and_reset():
                nonlocal is_moving
                is_moving = True
                try:
                    stop_mpc_fallback()
                    print("  [motion-planning] moving to home pose...")
                    go_home_sync()
                    print("  [motion-planning] home reached.")
                finally:
                    is_moving = False

            threading.Thread(target=go_home_and_reset, daemon=True).start()

        move_btn = server.gui.add_button("Move", color="green")
        move_btn.on_click(on_move)
        grasp_btn = server.gui.add_button("Grasp", color="blue")
        grasp_btn.on_click(on_grasp)
        open_btn = server.gui.add_button("Open Gripper", color="orange")
        open_btn.on_click(on_open_gripper)
        stop_btn = server.gui.add_button("Stop Tracking", color="red")
        stop_btn.on_click(on_stop_tracking)
        home_btn = server.gui.add_button("Go Home", color="purple")
        home_btn.on_click(on_home)

        threading.Thread(target=auto_track_loop, daemon=True).start()

        print(f"\nInteractive Motion Planner running at http://localhost:{viz_port}")
        print(f"Target links: {planner.tool_frames}  (dry_run={dry_run}, observe_only={observe_only})")
        if sim_mode:
            print("sim_mode=True: trajectories are executed through rebot_mit_sim's MIT dynamics simulator.")
        print(f"MPC fallback: {'enabled' if enable_mpc_fallback else 'DISABLED (--no-mpc-fallback)'}")
        print("  - Drag the end-effector gizmo -- planning and execution now happen automatically")
        if enable_mpc_fallback:
            print("    (falls back to continuous MPC tracking if the target isn't fully reachable,")
            print("     and switches back to a generated plan automatically once it is)")
        else:
            print("    (an unreachable target just waits -- MPC fallback is disabled)")
        print("  - Drag obstacles to reposition them")
        print("  - Click 'Move' to force an immediate (re-)plan attempt")
        print("  - Click 'Grasp' to plan and execute approach/grasp(+close)/lift")
        print("  - Click 'Open Gripper' to release")
        print("  - Click 'Stop Tracking' to halt an active MPC fallback")
        print("  - Click 'Go Home' to recover if the arm gets stuck near a singular pose")
        print("Press Ctrl+C to exit.\n")

        while True:
            time.sleep(0.1)
    finally:
        auto_track_stop_event.set()
        mpc_stop_event.set()
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="cuRobo motion-planning-based control for the rebot arm")
    parser.add_argument(
        "--live", action="store_true",
        help="Open the real Damiao serial bridge and command motors. Default is --dry-run.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Damiao serial bridge port")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--viz-port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--observe-only", action="store_true",
        help="With --live: open the real serial port and read state, but never enable torque or "
        "send a motor command (fully safe, arm cannot move).",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Dry-run only: execute every planned trajectory through rebot_mit_sim's "
        "MitDynamicsSimulator (real Damiao MIT PD law + Pinocchio forward dynamics + gravity "
        "feedforward + torque/joint limits) instead of teleporting the display straight to each "
        "waypoint. Shows whether the real motor control loop can actually track a "
        "motion-planned trajectory before ever touching hardware. Ignored with --live.",
    )
    parser.add_argument(
        "--no-mpc-fallback", action="store_true",
        help="Disable the continuous MPC tracking fallback used when Move/auto-track can't find a "
        "full plan (target unreachable or in collision, even after the home-reseed retry). With "
        "this set, an unreachable target just waits (or you can drag it somewhere reachable / click "
        "'Go Home') instead of switching to best-effort MPC tracking. MPC fallback is the least "
        "hardware-tested part of this script -- consider starting with this flag for a first live run.",
    )
    args = parser.parse_args()

    if args.live and not args.observe_only:
        confirm = input(
            "About to command REAL motors on /dev of your choosing. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    run_interactive_motion_planning(
        dry_run=not args.live,
        port=args.port,
        baud=args.baud,
        viz_port=args.viz_port,
        observe_only=args.observe_only,
        sim_mode=args.sim and not args.live,
        enable_mpc_fallback=not args.no_mpc_fallback,
    )


if __name__ == "__main__":
    main()
