# SPDX-License-Identifier: Apache-2.0
"""Pure gravity-compensation mode for the reBot B601-DM arm, with a Viser viewer.

Mirrors ~/reBotArm_control_py/example/9_gravity_compensation.py's control law
(MIT mode, soft position gains + Pinocchio-computed gravity feedforward
torque, so the arm holds its pose but can be freely pushed by hand to any
configuration) but adds a live Viser 3D view of the real arm's pose, and
reuses this project's proven RebotArmBridge/MOTOR_MAP/CAN-handling code
instead of reBotArm_control_py's own actuator layer.

Expected behavior: the arm holds its current position but is easily
backdrivable — push it and it should stay wherever you leave it, without
sagging under gravity or fighting your hand. This is a good direct visual/
physical check that compute_gravity_torque() (in rebot_mpc_control.py) is
correct, independent of any MPC/IK tracking logic.

Usage:

.. code-block:: bash

   python rebot_gravity_comp.py --visualize
   python rebot_gravity_comp.py --live --visualize --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import torch

from curobo.types import ContentPath, JointState
from curobo.viewer import ViserVisualizer

from rebot_mpc_control import (
    MOTOR_MAP,
    ROBOT_YML,
    RebotArmBridge,
    compute_gravity_torque,
)

# Soft gains for backdrivable gravity-compensation mode — much softer than
# MOTOR_MAP's tracking gains (kp=120/18), matching reBotArm_control_py's own
# gravity_compensation_controller (kp=2.0, kd=1.0 uniformly). The gravity
# feedforward does the heavy lifting; kp/kd only need to provide a little
# restoring pull, not hold position against an external push.
SOFT_KP = 2.0
SOFT_KD = 1.0
GRAVITY_COMP_MOTOR_MAP = {
    name: dataclasses.replace(spec, kp=SOFT_KP, kd=SOFT_KD)
    for name, spec in MOTOR_MAP.items()
}


def run_gravity_compensation(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    viz_port: int = 8080,
    observe_only: bool = False,
    control_dt: float = 0.02,
):
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0",
        connect_port=viz_port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1"]
    default_position = torch.zeros(1, len(joint_names), device="cuda", dtype=torch.float32)

    bridge = RebotArmBridge(
        joint_names=joint_names,
        motor_map=GRAVITY_COMP_MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=default_position,
        observe_only=observe_only,
    )

    try:
        # Deliberately NOT using start_command_thread()/set_target() here.
        # That decoupled background-thread architecture exists to survive
        # MPC's slow, variable-latency solves — it always resends the last
        # target so a slow solve doesn't cause a gap. For gravity comp there
        # is no slow solve: this is a trivial per-tick computation, and the
        # right architecture is the reference's tight, single-threaded
        # read-compute-send loop (matching
        # ~/reBotArm_control_py/example/9_gravity_compensation.py exactly).
        # Decoupling it added a full loop-cycle of staleness to a
        # self-referential "hold at wherever you currently are" feedback
        # loop, which is exactly the kind of delay that turns even soft
        # gains into an oscillating/diverging instability.
        current_state = bridge.read_joint_state(verbose=True)
        viser_viz.set_joint_state(current_state.squeeze(0))

        print(f"\nGravity compensation running at http://localhost:{viz_port}")
        print(f"Soft gains: kp={SOFT_KP}, kd={SOFT_KD} + Pinocchio gravity feedforward")
        print("Expected: arm holds position but is easily pushed by hand to any pose.")
        print("Press Ctrl+C to exit.\n")

        tick = 0
        report_every = max(1, int(round(1.0 / control_dt)))  # ~once per second
        while True:
            current_state = bridge.read_joint_state(verbose=(tick % report_every == 0))
            bridge.write_joint_command(current_state.position, torch.zeros_like(current_state.position))
            viser_viz.set_joint_state(current_state.squeeze(0))

            if tick % report_every == 0:
                tau_g = compute_gravity_torque(joint_names, current_state.position.squeeze(0).tolist())
                print(f"  [tick {tick}] tau_g: " + "  ".join(f"{t:+.3f}" for t in tau_g) + " N*m")

            tick += 1
            time.sleep(control_dt)
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Gravity-compensation-only control for the rebot arm")
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
            "About to command REAL motors on /dev of your choosing (soft gravity-comp gains). "
            "Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    run_gravity_compensation(
        dry_run=not args.live,
        port=args.port,
        baud=args.baud,
        viz_port=args.viz_port,
        observe_only=args.observe_only,
    )


if __name__ == "__main__":
    main()
