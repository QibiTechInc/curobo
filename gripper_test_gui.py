# SPDX-License-Identifier: Apache-2.0
"""Standalone tkinter GUI to position ONLY the reBot gripper motor (CAN id
0x07) -- no arm joints, no cuRobo solve, no MPC/IK. A hardware sanity check
for the gripper motor/linkage in isolation, ahead of wiring pinch-gesture
gripper control into the Quest 3 teleop (quest3_ego_teleop.py).

Reuses the gripper's MotorSpec (motor id, model, sign, scale, offset, kp/kd,
control mode) straight from MOTOR_MAP in rebot_mpc_control.py -- that's the
one calibrated source of truth for this hardware; duplicating it here would
risk drift.

Usage:

.. code-block:: bash

   # Safe: no serial port opened, commands only printed.
   python gripper_test_gui.py --dry-run

   # Real hardware.
   python gripper_test_gui.py --live --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

from rebot_mpc_control import MOTOR_MAP

GRIPPER_JOINT_NAME = "gripper_joint1"
# Confirmed via pinocchio FK on the URDF (gripper_left/gripper_right finger
# frames ~0mm apart at gripper_joint1=0.0, ~143mm apart at 0.0715) and
# cross-checked against the gripper's measured ~157mm length minus
# rack-and-pinion deadzone (~71.5mm useful travel, matching the URDF range):
# 0.0 = fully CLOSED, 0.0715 = fully OPEN -- the reverse of this script's
# original (unverified) assumption.
GRIPPER_MIN_POSITION = 0.0      # fully closed (URDF lower limit)
GRIPPER_MAX_POSITION = 0.0715   # fully open (URDF upper limit)
GRIPPER_OPEN_PRESET = 0.07      # short of the hard 0.0715 limit -- avoids
                                 # driving the linkage into its mechanical stop
GRIPPER_CLOSED_PRESET = 0.005   # short of the hard 0.0 limit, same reason
# Max commandable holding force (closing bias), in Newtons at the fingers.
# Rough starting ceiling, not a rated mechanical limit -- lower this if the
# gripper/object can't take it, raise it once verified safe on real hardware.
GRIPPER_MAX_HOLD_FORCE_N = 15.0
COMMAND_RATE_HZ = 50.0
READOUT_RATE_HZ = 10.0
# Per-tick position-step safety clamp, same idea as MAX_STEP_DELTA_RAD in
# rebot_mpc_control.py: refuses to jump the target further than this in one
# command tick, regardless of how far the slider moved.
MAX_STEP_PER_TICK = 0.003


class GripperBridge:
    """Reads/writes the single gripper motor over motorbridge's Damiao transport.

    In dry-run mode no serial port is opened: reads return the last commanded
    position and writes are only printed.
    """

    def __init__(self, port: str, baud: int, dry_run: bool):
        self.dry_run = dry_run
        self.spec = MOTOR_MAP[GRIPPER_JOINT_NAME]
        self._ctrl = None
        self._motor = None
        self._can_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._target_position = GRIPPER_OPEN_PRESET
        self._last_sent_position = GRIPPER_OPEN_PRESET
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._torque_enabled = True
        self._hold_force_n = 0.0

        if not dry_run:
            from motorbridge import Controller, Mode

            mode_by_name = {"mit": Mode.MIT, "force_pos": Mode.FORCE_POS}
            self._ctrl = Controller.from_dm_serial(serial_port=port, baud=baud)
            self._motor = self._ctrl.add_damiao_motor(
                self.spec.motor_id, self.spec.feedback_id, self.spec.model,
            )

            target_mode = mode_by_name[self.spec.control_mode]
            last_exc = None
            for _attempt in range(10):
                try:
                    self._motor.ensure_mode(target_mode, timeout_ms=1000)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(0.05)
            if last_exc is not None:
                raise RuntimeError(f"Failed to set gripper motor mode after 10 attempts: {last_exc}")
            print(f"  [mode] {GRIPPER_JOINT_NAME}: confirmed {self.spec.control_mode}")

            self._ctrl.disable_all()
            time.sleep(0.1)
            self._ctrl.enable_all()
            time.sleep(0.3)

            # Bumpless start: hold wherever the gripper actually is right now
            # instead of snapping to GRIPPER_OPEN_PRESET -- the physical
            # position at connect time won't exactly match any hardcoded
            # preset, and a stiff MIT hold at a mismatched target is exactly
            # what caused an unwanted jump on startup before this fix.
            actual = self.read_position()
            if actual is not None:
                actual = max(GRIPPER_MIN_POSITION, min(GRIPPER_MAX_POSITION, actual))
                self._target_position = actual
                self._last_sent_position = actual
                print(f"  [gripper] bumpless start at actual position {actual:+.4f} m")

    def set_target(self, position: float) -> None:
        position = max(GRIPPER_MIN_POSITION, min(GRIPPER_MAX_POSITION, position))
        with self._target_lock:
            self._target_position = position

    def disable_torque(self) -> None:
        """Cuts torque so the gripper can be moved by hand. No-op in
        dry-run (there's no real motor holding position anyway).
        """
        if self.dry_run:
            self._torque_enabled = False
            return
        with self._can_lock:
            self._ctrl.disable_all()
        self._torque_enabled = False

    def enable_torque(self) -> None:
        """Re-enables torque. Resyncs the internal target/last-sent position
        to the gripper's actual current position first -- otherwise, since
        the gripper was likely moved by hand while free, the motor would
        snap back toward whatever the target was left at before disabling.
        """
        actual = self.read_position()
        if actual is not None:
            with self._target_lock:
                self._target_position = actual
            self._last_sent_position = actual
        if not self.dry_run:
            with self._can_lock:
                self._ctrl.enable_all()
        self._torque_enabled = True

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_hold_force(self, newtons: float) -> None:
        """Constant closing-direction feedforward torque, on top of the
        position PD term -- NOT a hard force limit (MIT mode has no such
        cap; the PD term still adds its own torque on top of this if there's
        a position error). Use control_mode="force_pos" instead of raising
        this if what's actually needed is a true crush-safe force ceiling.
        """
        self._hold_force_n = max(0.0, min(GRIPPER_MAX_HOLD_FORCE_N, newtons))

    def read_position(self) -> Optional[float]:
        if self.dry_run:
            return self._last_sent_position
        with self._can_lock:
            self._motor.request_feedback()
            self._ctrl.poll_feedback_once()
            state = self._motor.get_state()
        if state is None:
            return None
        return self.spec.sign * (state.pos - self.spec.offset) / self.spec.scale

    def read_raw_position(self) -> Optional[float]:
        """Motor feedback before sign/offset are applied -- i.e. exactly
        what MOTOR_MAP['gripper_joint1'].offset needs to be re-measured
        against whenever the motor is re-zeroed (e.g. via lerobot-calibrate).
        """
        if self.dry_run:
            return None
        with self._can_lock:
            self._motor.request_feedback()
            self._ctrl.poll_feedback_once()
            state = self._motor.get_state()
        return None if state is None else state.pos

    def start(self, rate_hz: float = COMMAND_RATE_HZ) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._command_loop, args=(rate_hz,), daemon=True)
        self._thread.start()

    def _command_loop(self, rate_hz: float) -> None:
        dt = 1.0 / rate_hz
        while self._running:
            if not self._torque_enabled:
                # Free-move: don't ramp _last_sent_position while the
                # gripper is being moved by hand -- it no longer reflects
                # reality, and enable_torque() resyncs it before resuming.
                time.sleep(dt)
                continue
            with self._target_lock:
                target = self._target_position
            delta = target - self._last_sent_position
            delta = max(-MAX_STEP_PER_TICK, min(MAX_STEP_PER_TICK, delta))
            command_position = self._last_sent_position + delta
            self._last_sent_position = command_position
            command_velocity = delta * rate_hz

            spec = self.spec
            motor_pos = spec.sign * command_position * spec.scale + spec.offset
            motor_vel = spec.sign * command_velocity * spec.scale
            if abs(motor_vel) > spec.max_velocity:
                motor_vel = spec.max_velocity if motor_vel > 0 else -spec.max_velocity
            # Closing-direction feedforward torque: virtual work (tau*dtheta
            # = F*dx, dtheta = scale*dx) gives motor_tau = F/scale, and the
            # sign works out to push toward CLOSED (this joint's sign=-1
            # already makes increasing raw = closing, see MOTOR_MAP comment).
            motor_tau = self._hold_force_n / spec.scale

            if self.dry_run:
                print(
                    f"\r  [dry-run] send_mit(pos={motor_pos:+.4f}, vel={motor_vel:+.4f}, "
                    f"tau={motor_tau:+.4f})     ",
                    end="", flush=True,
                )
            else:
                with self._can_lock:
                    self._motor.send_mit(motor_pos, motor_vel, spec.kp, spec.kd, motor_tau)
            time.sleep(dt)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if not self.dry_run and self._ctrl is not None:
            self._ctrl.disable_all()


class GripperTestApp:
    def __init__(self, root: tk.Tk, bridge: GripperBridge):
        self.bridge = bridge
        root.title("reBot Gripper Test")

        frame = ttk.Frame(root, padding=16)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="Gripper position (m)").grid(row=0, column=0, columnspan=3, sticky="w")

        self.slider_var = tk.DoubleVar(value=GRIPPER_OPEN_PRESET)
        self.slider = ttk.Scale(
            frame, from_=GRIPPER_MIN_POSITION, to=GRIPPER_MAX_POSITION,
            orient="horizontal", length=320, variable=self.slider_var, command=self._on_slider_move,
        )
        self.slider.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        self.open_btn = ttk.Button(frame, text="Open", command=self._go_open)
        self.open_btn.grid(row=2, column=0, sticky="ew", padx=2)
        self.close_btn = ttk.Button(frame, text="Close", command=self._go_closed)
        self.close_btn.grid(row=2, column=1, sticky="ew", padx=2)
        self.stop_btn = ttk.Button(frame, text="STOP", command=self._stop)
        self.stop_btn.grid(row=2, column=2, sticky="ew", padx=2)

        self.torque_btn = ttk.Button(frame, text="Disable Torque (free move)", command=self._toggle_torque)
        self.torque_btn.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        ttk.Label(frame, text="Holding force, closing direction (N)").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 0),
        )
        self.hold_force_var = tk.DoubleVar(value=0.0)
        self.hold_force_slider = ttk.Scale(
            frame, from_=0.0, to=GRIPPER_MAX_HOLD_FORCE_N,
            orient="horizontal", length=320, variable=self.hold_force_var, command=self._on_hold_force_move,
        )
        self.hold_force_slider.grid(row=5, column=0, columnspan=3, sticky="ew")

        self.status_var = tk.StringVar(value="target=0.0000 m   actual=--")
        ttk.Label(frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        mode = "DRY-RUN (no hardware)" if bridge.dry_run else "LIVE"
        ttk.Label(frame, text=f"Mode: {mode}").grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self._stopped = False
        self._poll_readout(root)
        root.protocol("WM_DELETE_WINDOW", lambda: self._on_close(root))

    def _on_slider_move(self, _value: str) -> None:
        if self._stopped or not self.bridge.torque_enabled:
            return
        self.bridge.set_target(self.slider_var.get())

    def _on_hold_force_move(self, _value: str) -> None:
        if not self.bridge.torque_enabled:
            return
        self.bridge.set_hold_force(self.hold_force_var.get())

    def _go_open(self) -> None:
        if not self.bridge.torque_enabled:
            return
        self._stopped = False
        self.slider_var.set(GRIPPER_OPEN_PRESET)
        self.bridge.set_target(GRIPPER_OPEN_PRESET)

    def _go_closed(self) -> None:
        if not self.bridge.torque_enabled:
            return
        self._stopped = False
        self.slider_var.set(GRIPPER_CLOSED_PRESET)
        self.bridge.set_target(GRIPPER_CLOSED_PRESET)

    def _stop(self) -> None:
        """Freezes the target at the gripper's current commanded position --
        does not disable torque (use "Disable Torque" for that).
        """
        if not self.bridge.torque_enabled:
            return
        self._stopped = True
        self.bridge.set_target(self.bridge._last_sent_position)
        self.slider_var.set(self.bridge._last_sent_position)

    def _toggle_torque(self) -> None:
        if self.bridge.torque_enabled:
            self.bridge.disable_torque()
            self.bridge.set_hold_force(0.0)
            self.hold_force_var.set(0.0)
            self.torque_btn.config(text="Enable Torque")
            self.slider.state(["disabled"])
            self.open_btn.state(["disabled"])
            self.close_btn.state(["disabled"])
            self.stop_btn.state(["disabled"])
            self.hold_force_slider.state(["disabled"])
        else:
            self.bridge.enable_torque()
            self._stopped = False
            self.slider_var.set(self.bridge._last_sent_position)
            self.torque_btn.config(text="Disable Torque (free move)")
            self.slider.state(["!disabled"])
            self.open_btn.state(["!disabled"])
            self.close_btn.state(["!disabled"])
            self.stop_btn.state(["!disabled"])
            self.hold_force_slider.state(["!disabled"])

    def _poll_readout(self, root: tk.Tk) -> None:
        actual = self.bridge.read_position()
        actual_str = f"{actual:+.4f} m" if actual is not None else "no feedback"
        raw = self.bridge.read_raw_position()
        raw_str = f"{raw:+.4f}" if raw is not None else "--"
        self.status_var.set(
            f"target={self.bridge._target_position:+.4f} m   actual={actual_str}   raw_motor_pos={raw_str}   "
            f"hold_force={self.bridge._hold_force_n:.2f} N"
        )
        if not self.bridge.torque_enabled and actual is not None:
            # Free-move: the gripper is being moved by hand -- follow it
            # visually so the slider isn't stuck showing a stale position.
            self.slider_var.set(actual)
        root.after(int(1000 / READOUT_RATE_HZ), self._poll_readout, root)

    def _on_close(self, root: tk.Tk) -> None:
        self.bridge.stop()
        root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--live", action="store_true", help="Open the real serial port and enable torque.")
    parser.add_argument("--dry-run", dest="live", action="store_false", help="Default: no hardware touched.")
    parser.set_defaults(live=False)
    args = parser.parse_args()

    if args.live:
        confirm = input(
            "About to enable torque on the REAL gripper motor. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    bridge = GripperBridge(port=args.port, baud=args.baud, dry_run=not args.live)
    bridge.start()

    root = tk.Tk()
    GripperTestApp(root, bridge)
    try:
        root.mainloop()
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
