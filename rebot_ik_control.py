# SPDX-License-Identifier: Apache-2.0
"""Drive the reBot B601-DM arm with cuRobo IK + simple open-loop joint moves.

Simpler alternative to rebot_mpc_control.py: instead of a continuous MPC
control loop (which requires the arm to be actively, continuously commanded
at a fast fixed rate or Damiao's MIT-mode motors time out and drop torque —
see rebot_mpc_control.py's history for the debugging trail there), this
solves IK once per goal update (matching
``curobo/examples/getting_started/inverse_kinematics.py``'s
``interactive_ik_example``) and moves the real arm to the solved joint
configuration via the same simple, already-proven-working open-loop
joint-space move used for homing (``home_to_pose``/the background command
thread in rebot_mpc_control.py). No continuous closed-loop MPC tracking, no
solve-time-vs-motor-timeout race.

Usage:

.. code-block:: bash

   # Safe: no hardware touched.
   python rebot_ik_control.py --visualize

   # Real hardware, once rebot_mpc_control.MOTOR_MAP is calibrated (shared).
   python rebot_ik_control.py --live --visualize --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

from rebot_mpc_control import MOTOR_MAP, ROBOT_YML, RebotArmBridge, home_to_pose


def run_interactive_ik_control(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    viz_port: int = 8080,
    observe_only: bool = False,
    scene_model: str = "collision_test.yml",
):
    """Drag the EE gizmo in cuRobo's Viser GUI; each successful IK solve moves the real arm.

    Mirrors curobo's own interactive_ik_example() almost exactly (same solver
    config, same gizmo-drag-to-solve loop), but additionally drives the real
    arm's joints to each solution via RebotArmBridge/home_to_pose's simple
    open-loop control, instead of only updating the Viser display.
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = InverseKinematicsCfg.create(
        robot=ROBOT_YML,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=scene_model,
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
    )
    config.scene_collision_cfg.use_warp_collision = True
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    ik_solver = InverseKinematics(config)
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    bridge = RebotArmBridge(
        joint_names=ik_solver.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=ik_solver.default_joint_state.position.clone().unsqueeze(0),
        observe_only=observe_only,
    )

    try:
        do_home = not observe_only
        if do_home:
            home_to_pose(bridge, ik_solver.default_joint_state.position.clone().unsqueeze(0))
        real_state = bridge.read_joint_state()
        bridge.set_target(real_state.position, torch.zeros_like(real_state.position))

        # real_state.position is already batched (1, 7) from read_joint_state();
        # get_active_js()+solve_pose() expect the unbatched-in/unsqueeze(0)-out
        # pattern used by curobo's own interactive_ik_example, so squeeze first.
        unbatched_state = JointState.from_position(
            real_state.position.squeeze(0), joint_names=bridge.joint_names
        )
        current_state = ik_solver.get_active_js(unbatched_state).unsqueeze(0)

        kin_state = ik_solver.compute_kinematics(real_state)
        goal_tool_poses = kin_state.tool_poses.to_dict()

        # Warm-up solve BEFORE starting the command thread: with
        # use_cuda_graph=True this first call captures a CUDA graph, which
        # claims the CUDA stream exclusively — starting the (now CUDA-free)
        # background thread only after this completes is extra safety margin.
        ik_solver.solve_pose(
            goal_tool_poses=GoalToolPose.from_poses(
                goal_tool_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1,
            ),
            current_state=current_state.clone(),
            return_seeds=1,
        )
        bridge.start_command_thread()

        # Snap the draggable gizmo (placed by ViserVisualizer at
        # FK(default_joint_state) at construction time) to the real arm's
        # actual starting pose, and the rendered robot too — same fix as
        # rebot_mpc_control.py's interactive loop, same underlying cause.
        viser_viz.set_joint_state(real_state.squeeze(0))
        for frame_name, pose in goal_tool_poses.items():
            if frame_name in viser_viz._control_frames:
                handle = viser_viz._control_frames[frame_name]
                handle.position = pose.position.cpu().squeeze().numpy()
                handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

        print(f"\nInteractive IK running at http://localhost:{viz_port}")
        print(f"Target links: {ik_solver.tool_frames}  (dry_run={dry_run}, observe_only={observe_only})")
        print("Drag the end-effector gizmo to solve IK and move the arm.")
        print("Press Ctrl+C to exit.\n")

        previous_target_poses = None
        pose_changed = False

        while True:
            obstacle_poses = {
                k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
                for k in obstacle_frames.keys()
            }
            for k in obstacle_poses.keys():
                if obstacle_poses[k] != old_obstacle_poses[k]:
                    ik_solver.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
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
                active_js = ik_solver.get_active_js(current_state.squeeze(0)).unsqueeze(0)
                target_link_poses = {
                    k.replace("target_", ""): v for k, v in target_poses.items()
                }
                result = ik_solver.solve_pose(
                    goal_tool_poses=GoalToolPose.from_poses(
                        target_link_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1,
                    ),
                    current_state=active_js.squeeze(1).clone(),
                    return_seeds=1,
                )
                if result.success.any():
                    pose_changed = False
                    current_state = result.js_solution.clone()
                    solved_position = result.js_solution.squeeze(0).squeeze(0).position.unsqueeze(0)
                    bridge.set_target(solved_position, torch.zeros_like(solved_position))
                    viser_viz.set_joint_state(result.js_solution.squeeze(0).squeeze(0))
                else:
                    print("  [ik] target unreachable — holding last solution.")

            time.sleep(0.02)
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="cuRobo IK-based control for the rebot arm")
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
    args = parser.parse_args()

    if args.live and not args.observe_only:
        confirm = input(
            "About to command REAL motors on /dev of your choosing. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    run_interactive_ik_control(
        dry_run=not args.live,
        port=args.port,
        baud=args.baud,
        viz_port=args.viz_port,
        observe_only=args.observe_only,
    )


if __name__ == "__main__":
    main()
