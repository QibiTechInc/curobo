# SPDX-License-Identifier: Apache-2.0
"""Drive the reBot B601-DM arm with a hybrid IK-jump / MPC-track strategy.

rebot_mpc_control.py's continuous MPC loop kept dropping torque ("falls
off") even after fixing the CUDA graph capture race shared with
rebot_ik_control.py — something about *sustained* continuous MPC solving
still causes it, not yet root-caused. rebot_ik_control.py (one-shot IK solve
per goal update, no continuous solve loop) works reliably.

Hybrid strategy, in three steps:

1. FK: read the real arm's current joint state (proven-working bridge code).
2. IK for the initial pose: when a new drag *starts* (target was static,
   now changed), solve one-shot IK and jump there — reliable, matches
   rebot_ik_control.py's already-confirmed-working path.
3. MPC only while actively moving: if the target keeps changing on
   consecutive ticks (the user is actively dragging), hand off to MPC's
   continuous solve for smooth tracking. Once the target stops changing for
   a few ticks, drop back out of MPC tracking to the simple held position
   (background thread just resends the last target, no continuous solving)
   — this bounds how long MPC's continuous loop ever runs for, in case the
   still-unexplained fall-off is a function of sustained solve duration.

Usage:

.. code-block:: bash

   python rebot_hybrid_control.py --visualize
   python rebot_hybrid_control.py --live --visualize --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

from rebot_mpc_control import (
    MOTOR_MAP,
    ROBOT_YML,
    RebotArmBridge,
    _tool_position,
    cap_cartesian_step,
    home_to_pose,
    send_smoothed_segment,
)

# Consecutive idle ticks (no target change) before dropping out of MPC
# tracking back to a simple held position.
IDLE_TICKS_BEFORE_HOLD = 5


def run_hybrid_control(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    viz_port: int = 8080,
    observe_only: bool = False,
    scene_model: str = "collision_test.yml",
    control_dt: float = 0.03,
):
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    ik_config = InverseKinematicsCfg.create(
        robot=ROBOT_YML,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=scene_model,
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
    )
    ik_config.scene_collision_cfg.use_warp_collision = True
    scene_cfg = ik_config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    ik_solver = InverseKinematics(ik_config)
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    mpc_config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML,
        scene_model=scene_model,
        use_cuda_graph=not (dry_run or observe_only),
        optimization_dt=control_dt,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )
    mpc = ModelPredictiveControl(mpc_config)
    gripper_joint_idx = mpc.joint_names.index("gripper_joint1")

    bridge = RebotArmBridge(
        joint_names=mpc.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=mpc.default_joint_position.clone().unsqueeze(0),
        observe_only=observe_only,
    )

    try:
        do_home = not observe_only
        if do_home:
            home_to_pose(bridge, mpc.default_joint_position.clone().unsqueeze(0))
        real_state = bridge.read_joint_state()
        # gripper_joint1 doesn't affect the tracked tool pose (gripper_link is
        # upstream of the finger joints), so neither IK's nor MPC's cost has
        # any reason to move it sensibly -- held fixed throughout rather than
        # trusting either solver's ungoverned output (see rebot_mit_sim.py).
        gripper_hold_position = real_state.position[:, gripper_joint_idx].clone()
        bridge.set_target(real_state.position, torch.zeros_like(real_state.position))

        # Warm up BOTH solvers' CUDA graph capture (if any) before the
        # background command thread starts — see rebot_mpc_control.py's
        # __init__ note on why the thread must be CUDA-free and started
        # after any capturing call, not concurrently with one.
        mpc.setup(real_state)
        unbatched_state = JointState.from_position(
            real_state.position.squeeze(0), joint_names=bridge.joint_names
        )
        ik_current_state = ik_solver.get_active_js(unbatched_state).unsqueeze(0)
        kin_result = mpc.compute_kinematics(real_state)
        goal_tool_poses = kin_result.tool_poses.to_dict()
        target_link = mpc.tool_frames[0]

        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(goal_tool_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
            run_ik=False,
        )
        ik_solver.solve_pose(
            goal_tool_poses=GoalToolPose.from_poses(
                goal_tool_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1,
            ),
            current_state=ik_current_state.clone(),
            return_seeds=1,
        )

        bridge.start_command_thread()

        # Snap the gizmo to the real starting pose (same fix as the other
        # two scripts — ViserVisualizer places it at FK(default) otherwise).
        viser_viz.set_joint_state(real_state.squeeze(0))
        for frame_name, pose in goal_tool_poses.items():
            if frame_name in viser_viz._control_frames:
                handle = viser_viz._control_frames[frame_name]
                handle.position = pose.position.cpu().squeeze().numpy()
                handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

        print(f"\nHybrid IK/MPC control running at http://localhost:{viz_port}")
        print(f"Target link: {target_link}  (dry_run={dry_run}, observe_only={observe_only})")
        print("Drag the end-effector gizmo: first move jumps via IK, continued dragging tracks via MPC.")
        print("Press Ctrl+C to exit.\n")

        previous_target_poses = None
        idle_ticks = 0
        tracking = False
        current_state = real_state
        prev_accel = torch.zeros_like(real_state.position)
        tick = 0

        while True:
            obstacle_poses = {
                k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
                for k in obstacle_frames.keys()
            }
            obstacle_changed = False
            for k in obstacle_poses.keys():
                if obstacle_poses[k] != old_obstacle_poses[k]:
                    ik_solver.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                    mpc.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                    obstacle_changed = True
            old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

            target_poses = viser_viz.get_control_frame_pose()
            if previous_target_poses is None:
                previous_target_poses = target_poses
                target_changed = False
            else:
                target_changed = False
                for frame_name in target_poses.keys():
                    if target_poses[frame_name] != previous_target_poses[frame_name]:
                        previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                        target_changed = True
                        break

            target_link_poses = {k.replace("target_", ""): v for k, v in target_poses.items()}

            if target_changed and not tracking:
                # A new drag just started from a held position: jump there
                # via one-shot IK (reliable, proven), then switch to MPC for
                # any further continuous movement of this same drag.
                active_js = ik_solver.get_active_js(current_state.squeeze(0)).unsqueeze(0)
                result = ik_solver.solve_pose(
                    goal_tool_poses=GoalToolPose.from_poses(
                        target_link_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1,
                    ),
                    current_state=active_js.squeeze(1).clone(),
                    return_seeds=1,
                )
                if result.success.any():
                    js = result.js_solution.squeeze(0).squeeze(0)
                    js.position[gripper_joint_idx] = gripper_hold_position
                    bridge.set_target(js.position.unsqueeze(0), torch.zeros_like(js.position.unsqueeze(0)))
                    viser_viz.set_joint_state(js)
                    current_state = JointState.from_position(
                        js.position.unsqueeze(0), joint_names=bridge.joint_names,
                    )
                    current_state.velocity = torch.zeros_like(current_state.position)
                    current_state.acceleration = torch.zeros_like(current_state.position)
                    mpc.update_goal_tool_poses(
                        GoalToolPose.from_poses(
                            target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                        ),
                        run_ik=False,
                    )
                    tracking = True
                    idle_ticks = 0
                    print("  [ik] jumped to new target, switching to MPC tracking")
                else:
                    print("  [ik] target unreachable — staying put")
            elif target_changed and tracking:
                mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(
                        target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                    ),
                    run_ik=False,
                )
                idle_ticks = 0
            elif obstacle_changed and tracking:
                idle_ticks = 0
            elif tracking:
                idle_ticks += 1
                if idle_ticks >= IDLE_TICKS_BEFORE_HOLD:
                    tracking = False
                    print("  [mpc] target settled, dropping out of MPC tracking (holding position)")

            if tracking:
                # Replan from MPC's OWN last-commanded state (current_state,
                # updated below from the solver's own output), never real
                # hardware feedback. MPC has no dynamics model, so it can't
                # distinguish ordinary tracking lag from a genuine
                # disturbance -- replanning off real feedback every tick let
                # that lag compound into the runaway that caused this
                # script's "falls off" symptom in the first place. Confirmed
                # via rebot_mit_sim.py: seeding MPC's next solve from its own
                # last-commanded state ("I commanded X, so I'm at X now" --
                # the same assumption rebot_ik_control.py already made
                # successfully) is what finally made MPC track stably.
                result = mpc.optimize_action_sequence(current_state)
                has_action = (
                    result.action_sequence is not None and result.action_sequence.position.shape[1] > 0
                )
                if has_action:
                    # Walk through the FULL interpolation_steps waypoint
                    # sequence progressively instead of jumping straight to
                    # the last waypoint and holding it statically for the
                    # whole tick -- see rebot_mpc_control.py's comment for
                    # why (confirmed live: jerky/high-acceleration bursts
                    # from a coarse staircase reference, tracking was never
                    # the problem).
                    seq_pos = result.action_sequence.position.clone()  # [1, n_wp, dof]
                    seq_vel = result.action_sequence.velocity.clone()
                    seq_pos[:, :, gripper_joint_idx] = gripper_hold_position
                    seq_vel[:, :, gripper_joint_idx] = 0.0
                    n_wp = seq_pos.shape[1]
                    wp_dt = control_dt / n_wp
                    # Cap each waypoint's implied Cartesian tool speed, then
                    # send it as an acceleration/jerk-limited ramp -- see
                    # rebot_mpc_control.py's cap_cartesian_step() and
                    # send_smoothed_segment() for why.
                    prev_position = current_state.position
                    prev_velocity = current_state.velocity
                    prev_tool_position = _tool_position(mpc, prev_position)
                    for wp in range(n_wp):
                        wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
                            mpc, prev_position, prev_tool_position,
                            seq_pos[:, wp, :], seq_vel[:, wp, :], wp_dt,
                        )
                        prev_accel = send_smoothed_segment(
                            bridge, prev_position, prev_velocity, prev_accel,
                            wp_position, wp_velocity, wp_dt,
                        )
                        prev_position = wp_position
                        prev_velocity = wp_velocity

                    next_position = prev_position
                    next_velocity = prev_velocity
                    current_state = JointState.from_position(next_position, joint_names=bridge.joint_names)
                    current_state.velocity = next_velocity
                    current_state.acceleration = result.action_sequence.acceleration[:, -1, :]
                    # Real state read here is for the Viser display /
                    # operator monitoring only -- never fed back into MPC.
                    real_state = bridge.read_joint_state()
                    viser_viz.set_joint_state(real_state.squeeze(0))
                else:
                    print("  [warn] MPC returned no valid action_sequence this tick.")
                    time.sleep(control_dt)
            else:
                time.sleep(control_dt)

            tick += 1
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid IK-jump / MPC-track control for the rebot arm")
    parser.add_argument("--live", action="store_true", help="Command real motors. Default is --dry-run.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Damiao serial bridge port")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--viz-port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--observe-only", action="store_true",
        help="With --live: read real state but never enable torque or send commands (fully safe).",
    )
    args = parser.parse_args()

    if args.live and not args.observe_only:
        confirm = input(
            "About to command REAL motors on /dev of your choosing. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    run_hybrid_control(
        dry_run=not args.live,
        port=args.port,
        baud=args.baud,
        viz_port=args.viz_port,
        observe_only=args.observe_only,
    )


if __name__ == "__main__":
    main()
