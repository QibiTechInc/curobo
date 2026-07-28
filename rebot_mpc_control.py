# SPDX-License-Identifier: Apache-2.0
"""Drive the reBot B601-DM arm with cuRobo's reactive MPC over a Damiao CAN bridge.

This mirrors the control loop in
``curobo/examples/getting_started/reactive_control.py`` (see that file for the
tutorial version running against cuRobo's internal simulated state) but reads
current joint state from, and writes commanded joint state to, the physical
motors via the ``motorbridge`` SDK (Damiao serial-bridge transport, e.g.
Seeed's reBot B601-DM follower arm).

Before running against real hardware you MUST fill in ``MOTOR_MAP`` below with
the actual CAN motor_id / feedback_id / model for every joint, and verify the
``sign``/``offset`` calibration (motor zero position vs. URDF joint zero) with
the arm in a known, safe pose. Until then, run with ``--dry-run`` (the
default): the bridge never opens a serial port, joint state is read from
cuRobo's own kinematics defaults, and commanded actions are only printed.

Usage:

.. code-block:: bash

   # Safe: no hardware touched, just exercises the MPC + bridge plumbing.
   python rebot_mpc_control.py --dry-run

   # Real hardware, once MOTOR_MAP is filled in and calibrated.
   python rebot_mpc_control.py --live --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import torch
from scipy.signal import butter, lfilter

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

ROBOT_YML = "/home/kendemu/curobo/rebot_b601_gripper.yml"
REBOT_URDF_PATH = (
    "/home/kendemu/ros2_ws/src/reBotArmController_ROS2/src/rebotarm_bringup/"
    "description/urdf/reBot_B601_DM_with_gripper.urdf"
)

# Gravity computation uses REBOT_URDF_PATH directly — ONE URDF for
# everything (collision model, MPC kinematics, gravity comp), not two.
# Originally this used a separate reference URDF
# (~/reBotArm_control_py/.../reBot-DevArm_fixend.urdf) after discovering our
# own URDF's per-link masses were badly under-estimated (base_link=0.80kg,
# link6=0.019kg vs their proven-correct 0.84kg/0.366kg — every arm link
# 1.6-2.7x too light, link6 almost 19x) — that mismatch was the actual cause
# of "some gravity feedforward torque, just not enough to prevent sag." Now
# fixed at the source instead: REBOT_URDF_PATH's <inertial> blocks for
# base_link/link1-6 were updated to match those proven values directly (see
# rebot_arm_curobo_mpc memory for the full investigation), and
# rebot_b601_gripper.yml was regenerated from the corrected URDF.
#
# cuRobo's cspace order: [joint1..joint6, gripper_joint1] (7 DOF). Pinocchio,
# built directly from the URDF, does NOT collapse the <mimic> tag — it sees
# 8 independent DOF in URDF declaration order: [joint1..joint6,
# gripper_joint1, gripper_joint2]. gripper_joint2 = gripper_joint1 (mimic
# multiplier=1, offset=0), so its gravity contribution folds onto
# gripper_joint1's feedforward via the chain rule (effective_tau =
# tau[gripper_joint1] + multiplier * tau[gripper_joint2]).
ENABLE_GRAVITY_COMPENSATION = True
_PIN_MODEL: Optional[pin.Model] = None
_PIN_DATA: Optional[pin.Data] = None


def _get_pin_model():
    global _PIN_MODEL, _PIN_DATA
    if _PIN_MODEL is None:
        _PIN_MODEL = pin.buildModelFromUrdf(REBOT_URDF_PATH)
        _PIN_DATA = _PIN_MODEL.createData()
    return _PIN_MODEL, _PIN_DATA


def compute_gravity_torque(cspace_joint_names: list, position: list) -> list:
    """Generalized gravity torque g(q) (N*m) for cuRobo's 7-DOF cspace.

    ``position`` is in cspace_joint_names order (joint1..joint6,
    gripper_joint1). Returns a list in the same order/length.
    """
    model, data = _get_pin_model()
    names = model.names.tolist()
    q = np.zeros(model.nq)
    for i, name in enumerate(cspace_joint_names):
        idx = names.index(name) - 1  # -1: 'universe' has no q entry
        q[idx] = position[i]
    # gripper_joint2 mirrors gripper_joint1 (mimic multiplier=1, offset=0)
    if "gripper_joint2" in names:
        g2_idx = names.index("gripper_joint2") - 1
        g1_idx = cspace_joint_names.index("gripper_joint1")
        q[g2_idx] = position[g1_idx]

    tau_full = pin.computeGeneralizedGravity(model, data, q)

    tau = []
    for name in cspace_joint_names:
        idx = names.index(name) - 1
        tau.append(float(tau_full[idx]))
    if "gripper_joint1" in cspace_joint_names and "gripper_joint2" in names:
        g2_idx = names.index("gripper_joint2") - 1
        g1_pos = cspace_joint_names.index("gripper_joint1")
        tau[g1_pos] += float(tau_full[g2_idx])
    return tau


# Cartesian (end-effector) speed cap for the progressive-waypoint sends in
# the MPC tracking loops. Even with the per-tick waypoint interpolation fix,
# occasional ticks still produce a fast tool-frame move (e.g. MPC solving a
# larger-than-usual correction at some points) -- this bounds the actual
# tool speed directly, rather than bounding each joint independently (which
# doesn't guarantee a Cartesian speed limit near a less-favorable Jacobian
# configuration).
MAX_CARTESIAN_VELOCITY = 0.2  # m/s (200 mm/s)


def _tool_position(kin_solver, position: torch.Tensor) -> torch.Tensor:
    """FK position (batched, [1, 3]) of kin_solver's first tool frame at the given joint position."""
    state = JointState.from_position(position, joint_names=kin_solver.joint_names)
    kin_result = kin_solver.compute_kinematics(state)
    tool_frame = kin_solver.tool_frames[0]
    return kin_result.tool_poses.to_dict()[tool_frame].position


def cap_cartesian_step(
    kin_solver,
    prev_position: torch.Tensor,
    prev_tool_position: torch.Tensor,
    next_position: torch.Tensor,
    next_velocity: torch.Tensor,
    dt: float,
    max_speed: float = MAX_CARTESIAN_VELOCITY,
):
    """Scale a commanded joint-space step down (toward prev_position) if the
    implied Cartesian velocity of the tracked tool frame would exceed
    max_speed (m/s).

    Estimated by finite-differencing FK between prev_position and
    next_position over dt -- a linear joint-space scale-down is only an
    approximation of a true Cartesian-space cap (FK is nonlinear), but each
    waypoint step is already small (a fraction of one MPC tick), so it's
    accurate enough for a safety cap, not a precision requirement.

    Returns (capped_position, capped_velocity, next_tool_position) -- the
    last so the caller can reuse it as prev_tool_position for the following
    waypoint instead of recomputing FK twice per step.
    """
    next_tool_position = _tool_position(kin_solver, next_position)
    delta = next_tool_position - prev_tool_position
    implied_speed = (delta.norm(dim=-1) / dt).clamp_min(1e-9)
    scale = (max_speed / implied_speed).clamp(max=1.0).view(-1, 1)
    if float(scale.min()) >= 1.0:
        return next_position, next_velocity, next_tool_position

    capped_position = prev_position + scale * (next_position - prev_position)
    capped_velocity = next_velocity * scale
    # Recompute exactly for the actual (capped) position rather than
    # interpolating the FK result, since FK is nonlinear.
    capped_tool_position = _tool_position(kin_solver, capped_position)
    return capped_position, capped_velocity, capped_tool_position


# Per-joint acceleration/jerk limits for send_smoothed_segment() below.
# Starting points for live tuning, not proven values.
MAX_JOINT_ACCEL = 2.0  # rad/s^2
MAX_JOINT_JERK = 20.0  # rad/s^3
MIN_SUBSTEP_DT = 0.005  # s -- finest spacing between sent targets regardless of stretching


def send_smoothed_segment(
    bridge,
    prev_position: torch.Tensor,
    prev_velocity: torch.Tensor,
    prev_accel: torch.Tensor,
    next_position: torch.Tensor,
    next_velocity: torch.Tensor,
    nominal_dt: float,
    max_accel: float = MAX_JOINT_ACCEL,
    max_jerk: float = MAX_JOINT_JERK,
    min_substep_dt: float = MIN_SUBSTEP_DT,
) -> torch.Tensor:
    """Send one MPC waypoint-to-waypoint transition as an acceleration/jerk
    capped ramp instead of one linear jump over nominal_dt.

    Key point: subdividing the SAME nominal_dt into more updates does NOT
    reduce acceleration -- a given velocity change over a fixed duration is
    a fixed average acceleration, no matter how many intermediate points you
    send. The only way to actually reduce it is to STRETCH the segment's
    duration (slow down through it), then subdivide that longer duration
    into fine substeps so the reference update rate stays high. This also
    checks jerk -- the rate of change of acceleration from the previous
    segment to this one -- and stretches further if that would be too
    abrupt, which is what a fixed 4-waypoint linear interpolation cannot
    account for (it has no notion of the PREVIOUS segment's acceleration).

    Position/velocity within the (possibly stretched) segment are linearly
    interpolated between the two endpoints -- this gives a constant
    acceleration within the segment (a trapezoidal velocity ramp), which is
    what the accel cap above actually bounds. It does not by itself
    guarantee zero jerk at the interpolation points at min_substep_dt
    resolution, but combined with the cross-segment jerk check, this
    removes the large jerk/accel bursts a naive fixed 4-point breakdown
    can't see coming.

    Returns the ACHIEVED acceleration of this segment, to pass as
    prev_accel for the next call (maintains jerk continuity across
    waypoints and across ticks).

    Duration math: stretching the segment by a factor s multiplies the
    ACHIEVED acceleration by 1/s (accel = delta_v / (nominal_dt * s)), but
    the ACHIEVED jerk by 1/s^2 (jerk = accel_change / (nominal_dt * s), and
    accel_change itself already carries one 1/s) -- so solving for the
    minimum required duration from each constraint independently gives
    t_accel = |delta_v| / max_accel (linear in delta_v) and
    t_jerk = sqrt(|delta_v| / max_jerk) (SQUARE ROOT, not linear) for the
    delta_v-driven component, plus |prev_accel| / max_jerk for bleeding off
    whatever acceleration the previous segment ended with. Treating both
    like a simple linear ratio against nominal_dt (an earlier version of
    this function did) double-penalizes jerk through nominal_dt and
    produces wildly excessive stretching (confirmed: 6.7s for what should
    need ~0.5s in one test case).
    """
    delta_v = next_velocity - prev_velocity
    delta_p = next_position - prev_position

    t_accel = delta_v.abs() / max_accel
    t_jerk = torch.maximum(
        (delta_v.abs() / max_jerk).sqrt(),
        prev_accel.abs() / max_jerk,
    )
    required_dt = torch.maximum(torch.maximum(t_accel, t_jerk), torch.full_like(t_accel, nominal_dt))
    scaled_dt = float(required_dt.max())
    achieved_accel = delta_v / scaled_dt

    n_sub = max(1, math.ceil(scaled_dt / min_substep_dt))
    sub_dt = scaled_dt / n_sub

    for i in range(1, n_sub + 1):
        frac = i / n_sub
        bridge.set_target(prev_position + frac * delta_p, prev_velocity + frac * delta_v)
        time.sleep(sub_dt)

    return achieved_accel


# Butterworth low-pass filter for real measured velocity feedback: raw
# encoder-derived velocity is noisy, and feeding that noise into MPC's
# optimizer (which uses it as part of "current state" for the next solve)
# can inject noise-driven instability that compounds with kd damping.
VELOCITY_FILTER_ORDER = 2
VELOCITY_FILTER_CUTOFF_HZ = 5.0
VELOCITY_FILTER_SAMPLE_HZ = 30.0  # nominal control loop rate
_VEL_FILTER_B, _VEL_FILTER_A = butter(
    VELOCITY_FILTER_ORDER, VELOCITY_FILTER_CUTOFF_HZ / (VELOCITY_FILTER_SAMPLE_HZ / 2), btype="low"
)

# Observed real hardware starting pose (all joints near zero — fully extended),
# read from the arm right after the lerobot-calibrate zero-set. Used with
# --debug-start-pose=real to reproduce, in pure simulation (no hardware), the
# "MPC produces zero motion toward an easily-reachable goal" symptom seen live
# — confirmed reproducible: 60 sim steps toward a +0.1m Y offset from this
# exact pose produced *bit-for-bit zero* joint movement, while the same code
# converges normally from cuRobo's own default retract pose.
REAL_HARDWARE_START_POSE = [-0.0021, 0.0006, -0.0010, -0.0109, 0.0750, -0.1040, 0.0189]

# cuRobo's cspace.joint_names order for rebot_b601_gripper.yml (checked via
# `yaml.safe_load(...)["kinematics"]["cspace"]["joint_names"]`):
#   ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_joint1']
#
# gripper_joint2 is declared as a <mimic joint="gripper_joint1" multiplier="1"
# offset="0"/> of gripper_joint1 in the URDF (both fingers are driven by the
# single physical motor found at CAN id 0x07), so cuRobo excludes it from
# cspace automatically and derives its position via FK. Nothing to command
# for it here.


@dataclasses.dataclass
class MotorSpec:
    """Maps one cuRobo joint to one physical Damiao motor.

    Args:
        motor_id: CAN arbitration ID used to command the motor.
        feedback_id: CAN ID the motor replies on.
        model: Damiao model string (e.g. "4310", "4340") — must match
            ``DAMIAO_MODEL_LIMITS`` in ``motorbridge.cli.damiao`` so
            ``ensure_mode``/``send_mit`` scale MIT-mode floats correctly.
        sign: +1.0 or -1.0. Flips motor rotation direction to match the
            URDF joint's positive-rotation convention.
        scale: Raw motor units per URDF unit (rad/rad for the six direct-
            drive arm joints, hence 1.0; rad of motor shaft rotation per
            metre of travel for the gripper's rack-and-pinion, hence >>1 —
            the motor shaft doesn't turn 1:1 with a prismatic joint's linear
            position). Applied as ``raw = sign * scale * urdf + offset`` /
            ``urdf = sign * (raw - offset) / scale``. Also applied to
            commanded velocity (chain rule) and inversely to feedforward
            torque (virtual work: motor_tau = urdf_tau / scale).
        offset: Raw units added to ``scale * urdf_position`` to get the
            motor's raw command (and subtracted before dividing by scale to
            get URDF position back from raw feedback). Set by reading
            ``get_state().pos`` with the joint held at the URDF's zero pose.
        control_mode: "mit" (position+velocity+gains) or "force_pos"
            (force-limited position — safer for the gripper, avoids
            crushing on grasp).
        kp, kd: MIT-mode gains. Unused when control_mode="force_pos".
        force_pos_ratio: FORCE_POS-mode max force ratio in [0, 1]. Unused
            when control_mode="mit".
        max_velocity: Hard safety clamp (rad/s, raw/post-scale) on the
            commanded velocity sent to this motor, independent of what any
            solver outputs. Defaults to the proven POS_VEL vlim from
            ~/reBotArm_control_py/config/rebotarm_dm.yaml.
    """

    motor_id: Optional[int]
    feedback_id: Optional[int]
    model: str
    sign: float = 1.0
    scale: float = 1.0
    offset: float = 0.0
    control_mode: str = "mit"
    kp: float = 8.0
    kd: float = 0.8
    force_pos_ratio: float = 0.07
    max_velocity: float = 3.0


# Confirmed via `motorbridge-cli scan --vendor damiao --transport dm-serial
# --serial-port /dev/ttyACM0 --serial-baud 921600`: 7 motors respond, CAN ids
# 0x01-0x07, feedback ids 0x11-0x17 (motor_id + 0x10), assumed in assembly
# order base->wrist->gripper — later confirmed exactly by Seeed's own
# lerobot driver config (motor_can_ids in config_rebot_b601_follower.py).
#
# Model confirmed two ways: (1) `motorbridge-cli id-dump --rids 21,22,23`
# (PMAX/VMAX/TMAX) matched against DAMIAO_MODEL_LIMITS, and (2) cross-checked
# against Seeed's own lerobot driver (lerobot/robots/rebot_b601_follower/
# rebot_b601_follower.py MOTOR_MODELS dict), which is authoritative for the
# exact variant string and also confirms this motor_id/feedback_id mapping
# (its `motor_can_ids` config matches exactly):
#   joint1-3 (shoulder_pan/lift, elbow_flex): DM4340P (12.5/10.0/28.0, high torque)
#   joint4-6, gripper (wrist_flex/yaw/roll, gripper): DM4310 (12.5/30.0/10.0)
# 4340 and 4340P share identical PMAX/VMAX/TMAX, so id-dump alone can't tell
# them apart — the driver source is what confirms the "P" variant.
#
# Calibrated on this hardware via lerobot's own calibration flow
# (`motor.set_zero_position()` per motor while the arm was held at its zero
# pose, then `lerobot-calibrate` saved
# ~/.cache/huggingface/lerobot/calibration/robots/rebot_b601_follower/follower1.json).
# All 7 joints show drive_mode=0, homing_offset=0 — the zero lives in each
# motor's own firmware, not a software offset, so offset=0.0 below matches.
#
# Cross-checked lerobot's joint_limits (degrees) against the URDF's radian
# limits to infer `sign`: joint1-6 agree in polarity (e.g. wrist_yaw:
# lerobot -90/90 vs URDF -1.57/1.57 rad — exact match), so sign=+1 for all
# six. The gripper does NOT agree: lerobot's motor range is -270/0 (rotates
# negative) but the URDF's gripper_joint1 range is 0/+0.0715 (prismatic,
# positive = opening) — the linkage inverts direction, so sign=-1 here.
#
# Still worth a supervised low-power sanity check before trusting this
# blindly (e.g. small MIT commands with low kp per joint, one at a time,
# watching it move the expected way) — this is inferred from range
# comparison, not a direct confirmation.
#
# kp/kd taken from ~/reBotArm_control_py/config/rebotarm_dm.yaml — a proven,
# working control stack for this exact arm (Damiao + motorbridge, same CAN
# ids/models). These are ~2.5-3x stiffer than the Seeed lerobot-derived
# values used earlier (kp=45/kd=12 for joints1-3, kp=8-9/kd=1 for 4-6),
# which were too soft to hold position without gravity-feedforward torque —
# the likely cause of the arm sagging/falling after the hold period. This
# proven config relies on stiffness alone (tau=0), no feedforward needed.
# Gripper also switched from FORCE_POS to MIT (kp=8/kd=1), matching this
# reference exactly, since force_pos was our own choice, not proven here.
_MODEL_UNCONFIRMED = "UNCONFIRMED"
# max_velocity: hard safety clamp on commanded velocity (rad/s), enforced in
# write_joint_command()/_write_joint_command_raw() independent of what any
# solver outputs. Set conservatively (well below the proven POS_VEL vlim of
# 3.0-5.0 in rebotarm_dm.yaml) after a real incident: feeding real velocity
# feedback into MPC caused it to diverge and drive the arm into the ground
# hard and fast — there was previously NO limit on commanded velocity at
# all, only on position step size. Raise later once tracking is re-verified
# stable at low speed.
# kp/kd for joints1-6 reduced from the originally-proven-elsewhere values
# (120/8, 18/2) after live MPC testing (2026-07-21): with replan_from_belief
# fixed and MPC finally tracking stably, the remaining complaint was a fixed
# linear PD law having no cap on speed or torque -- the SAME gains that felt
# "too fast" on a 1-3cm move produced overshoot-driving torque on a longer
# move, causing oscillation. kp lowered to ~40% (cuts peak torque/speed for
# any given error) while kd:kp ratio was INCREASED, not scaled down with it
# (relatively more damping specifically to fight the large-move oscillation,
# not just uniformly softer everywhere). Starting point for further live
# tuning, not a final answer -- adjust from here based on real behavior.
# max_velocity graduated by joint position in the chain (2026-07-21, after
# the Cartesian cap alone wasn't enough): proximal joints (base/shoulder/
# elbow) move more mass over a longer lever arm and have more visible/
# consequential impact when fast, so they get the tightest cap; distal
# joints (wrist) are lighter and get progressively more headroom -- also
# matches the motors' own rated speeds (4340P joints1-3: 50 rad/s rated;
# 4310 joints4-6: 200 rad/s rated). Starting point for further live tuning,
# not a final answer.
MOTOR_MAP: Dict[str, MotorSpec] = {
    "joint1": MotorSpec(motor_id=0x01, feedback_id=0x11, model="4340P", sign=1.0, offset=0.0, kp=50.0, kd=6.0, max_velocity=0.3),
    "joint2": MotorSpec(motor_id=0x02, feedback_id=0x12, model="4340P", sign=1.0, offset=0.0, kp=50.0, kd=6.0, max_velocity=0.4),
    "joint3": MotorSpec(motor_id=0x03, feedback_id=0x13, model="4340P", sign=1.0, offset=0.0, kp=50.0, kd=6.0, max_velocity=0.5),
    "joint4": MotorSpec(motor_id=0x04, feedback_id=0x14, model="4310", sign=1.0, offset=0.0, kp=8.0, kd=1.5, max_velocity=0.8),
    "joint5": MotorSpec(motor_id=0x05, feedback_id=0x15, model="4310", sign=1.0, offset=0.0, kp=8.0, kd=1.5, max_velocity=1.0),
    "joint6": MotorSpec(motor_id=0x06, feedback_id=0x16, model="4310", sign=1.0, offset=0.0, kp=8.0, kd=1.5, max_velocity=1.2),
    # scale/offset re-derived 2026-07-22 after rerunning lerobot-calibrate
    # (new gripper zero reference = default/open pose). Measured live via
    # gripper_test_gui.py's raw_motor_pos readout at both physical extremes
    # (URDF's own gripper_joint1=0.0/0.0715 confirmed closed/open via
    # pinocchio FK -- fingers ~0mm apart at 0.0, ~143mm apart at 0.0715 --
    # and cross-checked against the gripper's measured ~157mm length minus
    # rack-and-pinion deadzone, matching the ~71.5mm URDF range):
    #   physically open  (urdf=0.0715): raw = -0.0189
    #   physically closed (urdf=0.0):   raw = +5.8082
    # Solving raw = sign*scale*urdf + offset from those two points gives
    # sign*scale = -81.498 -- far from +-1, confirming the gripper's motor
    # does NOT turn 1:1 with the joint's linear travel like the arm's direct
    # -drive joints do (a rack-and-pinion needs a large motor rotation per
    # metre of slide) -- this is what the `scale` field above exists for.
    # max_velocity raised from the pre-`scale` value of 1.0: that cap was
    # tuned back when a full open<->close swing was (wrongly) only a 0.0715
    # rad raw error; with scale=81.498 applied it's really a ~5.83 rad raw
    # error, so the old cap made a full traverse take ~5.8s. 8.0 rad/s gives
    # ~0.7s full-range while staying well under the DM4310's 30 rad/s rated
    # VMAX -- starting point for further live tuning, not a final answer.
    "gripper_joint1": MotorSpec(
        motor_id=0x07, feedback_id=0x17, model="4310", sign=-1.0, scale=81.498, offset=5.8082,
        control_mode="mit", kp=8.0, kd=1.0, max_velocity=8.0,
    ),
}

# Per-tick safety clamp, independent of cuRobo's own joint-limit handling:
# refuses to send a position command that jumps further than this from the
# last known joint position. Guards against a bad MPC solve or a stale/torn
# CAN read commanding a violent motion.
MAX_STEP_DELTA_RAD = 0.15


class RebotArmBridge:
    """Reads/writes joint state for the rebot arm over motorbridge's Damiao transport.

    In dry-run mode no serial port is opened: reads return cuRobo's default
    joint position (zero velocity/acceleration) and writes are only printed.
    """

    def __init__(
        self,
        joint_names: list,
        motor_map: Dict[str, MotorSpec],
        port: str,
        baud: int,
        dry_run: bool,
        default_position: torch.Tensor,
        observe_only: bool = False,
    ):
        self.joint_names = joint_names
        self.motor_map = motor_map
        self.dry_run = dry_run
        self.observe_only = observe_only
        self._last_position = default_position.clone()
        self._default_position = default_position.clone()
        self._ctrl = None
        self._motors: Dict[str, object] = {}

        # Two locks: `_can_lock` serializes ALL access to the motorbridge
        # Controller/Motor objects (the underlying Rust ABI is not known to be
        # safe for concurrent calls from two Python threads) — both
        # read_joint_state() (main thread) and the background command thread's
        # writes take this. `_target_lock` protects only the small shared
        # target tensors the command thread resends every tick.
        self._can_lock = threading.Lock()
        self._target_lock = threading.Lock()
        # Plain Python floats, not torch tensors: the background command
        # thread must never touch CUDA. cuRobo's solvers use CUDA graph
        # capture (use_cuda_graph=True), which claims exclusive use of the
        # CUDA stream during capture — any CUDA op from another thread
        # (even something as innocuous as tensor.item()) during that window
        # corrupts the capture ("operation not permitted when stream is
        # capturing"). Converting to floats in set_target() (called from the
        # main thread, where CUDA is safe) keeps the background thread
        # 100% CUDA-free.
        self._target_position_raw = default_position.squeeze(0).detach().cpu().tolist()
        self._target_velocity_raw = [0.0] * len(self._target_position_raw)
        self._last_position_raw = list(self._target_position_raw)
        self._cmd_thread: Optional[threading.Thread] = None
        self._cmd_thread_running = False

        # Per-joint Butterworth filter state for real velocity feedback
        # (scipy.signal.lfilter's `zi`, one per joint, carried across calls).
        n = len(joint_names)
        filter_order = len(_VEL_FILTER_A) - 1
        self._velocity_filter_state = [np.zeros(filter_order) for _ in range(n)]

        if not dry_run:
            missing = [j for j in joint_names if motor_map[j].motor_id is None]
            if missing:
                raise RuntimeError(
                    "Refusing to open real motors: MOTOR_MAP is missing motor_id/"
                    f"feedback_id for joints {missing}. Fill in MOTOR_MAP and verify "
                    "sign/offset calibration before running with --live."
                )
            unconfirmed_model = [j for j in joint_names if motor_map[j].model == _MODEL_UNCONFIRMED]
            if unconfirmed_model:
                raise RuntimeError(
                    "Refusing to open real motors: MOTOR_MAP has an unconfirmed Damiao "
                    f"model for joints {unconfirmed_model}. MIT-mode command scaling "
                    "depends on the real PMAX/VMAX/TMAX — verify with "
                    "`motorbridge-cli id-dump ... --rids 21,22,23` before running with --live."
                )
            # Local import: only needed (and only installed) for live hardware runs.
            from motorbridge import Controller, Mode

            mode_by_name = {"mit": Mode.MIT, "force_pos": Mode.FORCE_POS}

            self._ctrl = Controller.from_dm_serial(serial_port=port, baud=baud)
            for name in joint_names:
                spec = motor_map[name]
                motor = self._ctrl.add_damiao_motor(spec.motor_id, spec.feedback_id, spec.model)
                self._motors[name] = motor

            if observe_only:
                print("  [observe-only] torque NOT enabled, no commands will be sent — reading state only.")
            else:
                # Sequence matches ~/reBotArm_control_py's proven RebotArm
                # setup exactly: set mode FIRST, then disable_all(), THEN
                # enable_all() last. We previously did enable_all() before
                # ensure_mode() (mode-set while already enabled) — that
                # produced zero torque on real hardware (mode confirmed by
                # ensure_mode() with no exception, but motors never actually
                # applied torque), while this order is the one confirmed
                # working via `python example/9_gravity_compensation.py`.
                #
                # Same CAN-latency race as get_state(): a single ensure_mode() call can
                # silently time out, leaving the motor in whatever mode it was already
                # in (possibly not MIT/FORCE_POS) while we go on to send MIT/FORCE_POS
                # frames it isn't listening for. Retry like Seeed's own driver does
                # (_ENSURE_MODE_RETRIES=9 there) instead of trusting one attempt.
                for name, motor in self._motors.items():
                    target_mode = mode_by_name[motor_map[name].control_mode]
                    last_exc = None
                    for attempt in range(10):
                        try:
                            motor.ensure_mode(target_mode, timeout_ms=1000)
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                            time.sleep(0.05)
                    if last_exc is not None:
                        raise RuntimeError(
                            f"Failed to set control mode for '{name}' after 10 attempts: {last_exc}"
                        )
                    print(f"  [mode] {name}: confirmed {motor_map[name].control_mode}")

                self._ctrl.disable_all()
                time.sleep(0.1)
                self._ctrl.enable_all()
                time.sleep(0.3)

    def read_joint_state(self, verbose: bool = False) -> JointState:
        velocity = None
        if self.dry_run:
            position = self._last_position.clone()
        else:
            # Matches the proven pattern in ~/reBotArm_control_py (JointGroup.
            # _request_feedback): request feedback from every motor, then a
            # SINGLE poll_feedback_once() call — not many polls with sleeps in
            # between. That reference runs its control loop at up to 500Hz
            # with this exact pattern, so multi-poll-with-sleep isn't needed
            # and was adding ~50ms+ of avoidable latency per read here.
            with self._can_lock:
                for motor in self._motors.values():
                    motor.request_feedback()
                self._ctrl.poll_feedback_once()

                positions = []
                velocities = []
                raw = []
                for name in self.joint_names:
                    spec = self.motor_map[name]
                    motor = self._motors[name]
                    state = motor.get_state()
                    if state is None:
                        raise RuntimeError(
                            f"No feedback for joint '{name}' — check CAN wiring/power."
                        )
                    positions.append(spec.sign * (state.pos - spec.offset) / spec.scale)
                    velocities.append(spec.sign * state.vel / spec.scale)
                    raw.append((name, state.pos, state.vel, state.status_code))

            # Butterworth low-pass filter: raw encoder-derived velocity is
            # noisy; feeding that noise into MPC's optimizer as "current
            # velocity" can inject noise-driven instability. Per-joint IIR
            # state carried across calls (see __init__).
            filtered_velocities = []
            for i, raw_vel in enumerate(velocities):
                out, self._velocity_filter_state[i] = lfilter(
                    _VEL_FILTER_B, _VEL_FILTER_A, [raw_vel], zi=self._velocity_filter_state[i]
                )
                filtered_velocities.append(float(out[0]))
            if verbose:
                for (name, raw_pos, raw_vel, status), filt_vel in zip(raw, filtered_velocities):
                    print(
                        f"    [raw] {name}: pos={raw_pos:+.4f} vel={raw_vel:+.4f} "
                        f"(filtered={filt_vel:+.4f}) status={status}"
                    )
            position = torch.tensor(
                positions, device=self._last_position.device, dtype=self._last_position.dtype
            ).unsqueeze(0)
            velocity = torch.tensor(
                filtered_velocities, device=self._last_position.device, dtype=self._last_position.dtype
            ).unsqueeze(0)
            self._last_position = position.clone()

        joint_state = JointState.from_position(position, joint_names=self.joint_names)
        # Real measured velocity (closed-loop) when live; zero in dry-run
        # (no real sensor to read there). Feeding MPC's own predicted
        # velocity back as "current state" every tick — matching cuRobo's
        # official reactive_control.py, which is pure simulation with no
        # real sensor — is open-loop on real hardware: if cuRobo's internal
        # dynamics model doesn't match the real arm, that belief can drift
        # from reality over sustained operation, and kd*(v_des - v_actual)
        # starts fighting the real motor instead of damping it.
        joint_state.velocity = velocity if velocity is not None else torch.zeros_like(joint_state.position)
        joint_state.acceleration = torch.zeros_like(joint_state.position)
        return joint_state

    def write_joint_command(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
        position = position.squeeze(0)
        velocity = velocity.squeeze(0)
        last = self._last_position.squeeze(0)

        clamped_position = position.clone()
        delta = clamped_position - last
        over = delta.abs() > MAX_STEP_DELTA_RAD
        if over.any():
            clamped_position = last + delta.clamp(-MAX_STEP_DELTA_RAD, MAX_STEP_DELTA_RAD)
            bad = [self.joint_names[i] for i in torch.nonzero(over).flatten().tolist()]
            print(f"  [safety] clamped oversized step for joints {bad}")

        tau_g = (
            compute_gravity_torque(self.joint_names, clamped_position.tolist())
            if ENABLE_GRAVITY_COMPENSATION else [0.0] * len(self.joint_names)
        )

        skip_send = self.dry_run or self.observe_only
        with self._can_lock:
            for i, name in enumerate(self.joint_names):
                spec = self.motor_map[name]
                motor_pos = spec.sign * clamped_position[i].item() * spec.scale + spec.offset
                motor_vel = spec.sign * velocity[i].item() * spec.scale
                motor_tau = spec.sign * tau_g[i] / spec.scale
                if abs(motor_vel) > spec.max_velocity:
                    print(
                        f"  [safety] clamped oversized velocity for {name}: "
                        f"{motor_vel:+.3f} -> {spec.max_velocity if motor_vel > 0 else -spec.max_velocity:+.3f} rad/s"
                    )
                    motor_vel = spec.max_velocity if motor_vel > 0 else -spec.max_velocity
                if spec.control_mode == "force_pos":
                    vlim = abs(motor_vel) if motor_vel != 0.0 else 1.0
                    if skip_send:
                        tag = "observe-only" if self.observe_only else "dry-run"
                        print(
                            f"  [{tag}] {name}: send_force_pos(pos={motor_pos:+.4f}, "
                            f"vlim={vlim:.4f}, ratio={spec.force_pos_ratio}) [NOT SENT]"
                        )
                    else:
                        self._motors[name].send_force_pos(motor_pos, vlim, spec.force_pos_ratio)
                else:
                    if skip_send:
                        tag = "observe-only" if self.observe_only else "dry-run"
                        print(
                            f"  [{tag}] {name}: send_mit(pos={motor_pos:+.4f}, vel={motor_vel:+.4f}, "
                            f"tau_g={motor_tau:+.3f}) [NOT SENT]"
                        )
                    else:
                        self._motors[name].send_mit(motor_pos, motor_vel, spec.kp, spec.kd, motor_tau)

        self._last_position = clamped_position.unsqueeze(0)

    def _write_joint_command_raw(self, position: list, velocity: list) -> None:
        """Pure-Python (no torch/CUDA) equivalent of write_joint_command().

        Only this method may run on the background command thread — it must
        stay 100% free of CUDA tensor ops (see the note in __init__).
        """
        last = self._last_position_raw
        clamped = []
        bad = []
        for i, (p, l) in enumerate(zip(position, last)):
            delta = p - l
            if abs(delta) > MAX_STEP_DELTA_RAD:
                delta = max(-MAX_STEP_DELTA_RAD, min(MAX_STEP_DELTA_RAD, delta))
                bad.append(self.joint_names[i])
            clamped.append(l + delta)
        if bad:
            print(f"  [safety] clamped oversized step for joints {bad}")

        tau_g = (
            compute_gravity_torque(self.joint_names, clamped)
            if ENABLE_GRAVITY_COMPENSATION else [0.0] * len(self.joint_names)
        )

        skip_send = self.dry_run or self.observe_only
        with self._can_lock:
            for i, name in enumerate(self.joint_names):
                spec = self.motor_map[name]
                motor_pos = spec.sign * clamped[i] * spec.scale + spec.offset
                motor_vel = spec.sign * velocity[i] * spec.scale
                motor_tau = spec.sign * tau_g[i] / spec.scale
                if abs(motor_vel) > spec.max_velocity:
                    print(
                        f"  [safety] clamped oversized velocity for {name}: "
                        f"{motor_vel:+.3f} -> {spec.max_velocity if motor_vel > 0 else -spec.max_velocity:+.3f} rad/s"
                    )
                    motor_vel = spec.max_velocity if motor_vel > 0 else -spec.max_velocity
                if spec.control_mode == "force_pos":
                    vlim = abs(motor_vel) if motor_vel != 0.0 else 1.0
                    if not skip_send:
                        self._motors[name].send_force_pos(motor_pos, vlim, spec.force_pos_ratio)
                else:
                    if not skip_send:
                        self._motors[name].send_mit(motor_pos, motor_vel, spec.kp, spec.kd, motor_tau)

        self._last_position_raw = clamped

    def set_target(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
        """Update the target the background command thread resends.

        Use this instead of calling write_joint_command() directly once the
        command thread is running (--live only): the MPC solve loop just
        updates the shared target whenever it has a new one, and the
        background thread keeps resending the latest target at a fixed high
        rate regardless of whether a given MPC tick produced anything new —
        so a tick with no valid solve just re-sends the last target instead
        of leaving a gap that lets MIT-mode motors time out and drop torque.

        Converts to plain Python floats here, in the caller's (main) thread,
        where touching CUDA is safe — see the CUDA graph capture note in
        __init__ for why the background thread itself must never do this.

        In dry-run/observe-only (no command thread running), falls back to
        a direct synchronous write so the existing print/_last_position
        tracking behavior is unchanged.
        """
        if self._cmd_thread is None:
            self.write_joint_command(position, velocity)
            return
        pos_list = position.squeeze(0).detach().cpu().tolist()
        vel_list = velocity.squeeze(0).detach().cpu().tolist()
        with self._target_lock:
            self._target_position_raw = pos_list
            self._target_velocity_raw = vel_list

    def start_command_thread(self, rate_hz: float = 200.0) -> None:
        """Start a dedicated thread resending the latest target at rate_hz.

        Matches ~/reBotArm_control_py's RebotArm.start_control_loop() design:
        command sending is decoupled from (and much faster than) the MPC/IK
        solve, which only asynchronously updates the shared target. Must be
        started AFTER any warm-up solve that triggers CUDA graph capture
        (e.g. the first solve_pose()/optimize_action_sequence() call with
        use_cuda_graph=True) — see the CUDA note in __init__.
        """
        if self.dry_run or self.observe_only or self._cmd_thread is not None:
            return
        self._cmd_thread_running = True

        def _loop():
            dt = 1.0 / rate_hz
            while self._cmd_thread_running:
                t0 = time.perf_counter()
                with self._target_lock:
                    pos = list(self._target_position_raw)
                    vel = list(self._target_velocity_raw)
                self._write_joint_command_raw(pos, vel)
                elapsed = time.perf_counter() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self._cmd_thread = threading.Thread(target=_loop, name="rebot-command-thread", daemon=True)
        self._cmd_thread.start()
        print(f"  [cmd-thread] started at {rate_hz:.0f} Hz")

    def stop_command_thread(self) -> None:
        if self._cmd_thread is not None:
            self._cmd_thread_running = False
            self._cmd_thread.join(timeout=2.0)
            self._cmd_thread = None

    def close(self) -> None:
        self.stop_command_thread()
        if self._ctrl is not None:
            try:
                self._ctrl.disable_all()
            finally:
                self._ctrl.close_bus()
                self._ctrl.close()


# cuRobo's MPC produces exactly zero corrective motion when the tracked joint
# state is precisely this arm's all-zero calibrated hardware pose. Rather than
# chasing the exact root cause further, home the real arm open-loop (simple
# linear interpolation, no MPC/collision-awareness) to cuRobo's own
# auto-computed default_joint_position before ever handing off to MPC — that
# pose is already confirmed (via the getting-started tutorial and our own
# testing) to converge normally.
HOMING_DURATION_SEC = 4.0


def home_to_pose(
    bridge: "RebotArmBridge", target_position: torch.Tensor, duration: float = HOMING_DURATION_SEC,
    control_dt: float = 0.05, settle_duration: float = 1.0,
) -> None:
    """Open-loop move the arm from wherever it is to target_position, then hold it there.

    Skipped by the caller for --debug-start-pose=real (which exists to
    reproduce the zero-pose MPC degeneracy, so must not be homed away from
    it) and for --observe-only (torque is off, nothing can move).

    The settle phase keeps sending position-hold commands after reaching the
    target: MIT-mode motors need commands to keep arriving periodically (they
    can time out/disable otherwise), and it gives the arm a moment to
    stabilize before MPC takes over the command stream.
    """
    current_state = bridge.read_joint_state()
    current = current_state.position.clone()
    target = target_position.to(device=current.device, dtype=current.dtype)

    if bridge.dry_run:
        bridge._last_position = target.clone()
        return

    delta = (target - current).abs().max().item()
    print(f"  [homing] moving to default pose over {duration:.1f}s (max joint delta {delta:.3f} rad)...")
    n_steps = max(1, int(duration / control_dt))
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp_pos = current + alpha * (target - current)
        interp_vel = (target - current) / duration
        bridge.write_joint_command(interp_pos, interp_vel)
        time.sleep(control_dt)
    print("  [homing] done.")

    if settle_duration > 0:
        print(f"  [settle] holding home pose for {settle_duration:.1f}s before handing off...")
        zero_velocity = torch.zeros_like(target)
        for _ in range(max(1, int(settle_duration / control_dt))):
            bridge.write_joint_command(target, zero_velocity)
            time.sleep(control_dt)
        print("  [settle] done.")


def run_home_only(
    dry_run: bool = True, port: str = "/dev/ttyACM0", baud: int = 921600,
    hold_duration: float = 5.0, hold_control_dt: float = 0.05,
) -> None:
    """Move the arm to cuRobo's default_joint_position, then hold it — no MPC at all.

    Isolated test of homing + sustained holding: confirms the arm can be
    commanded, via simple open-loop joint-space control, to the pose cuRobo's
    official getting_started tutorial already confirmed works for MPC, and
    can hold there stably (MIT-mode position commands must keep arriving
    periodically or Damiao motors may time out/disable — a single one-shot
    move-and-stop doesn't test that) — independent of whatever is or isn't
    wrong with the MPC control loop itself.
    """
    config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML, scene_model="collision_table.yml",
        use_cuda_graph=False, optimization_dt=0.025, interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(config)
    bridge = RebotArmBridge(
        joint_names=mpc.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=mpc.default_joint_position.clone().unsqueeze(0),
        observe_only=False,
    )
    try:
        target = mpc.default_joint_position.clone().unsqueeze(0)
        print(f"Target pose (cuRobo default_joint_position): {mpc.default_joint_position.tolist()}")
        home_to_pose(bridge, target, settle_duration=hold_duration, control_dt=hold_control_dt)
        final_state = bridge.read_joint_state(verbose=True)
        print(f"Reached position: {final_state.position.squeeze().tolist()}")
    finally:
        bridge.close()


def run_mpc_control(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    num_steps: int = 100,
    goal_offset_y: float = 0.2,
    optimization_dt: float = 0.025,
    observe_only: bool = False,
    debug_start_pose: str = "default",
):
    """Run cuRobo reactive MPC, driving the rebot arm over the Damiao bridge."""
    config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML,
        scene_model="collision_table.yml",
        use_cuda_graph=not (dry_run or observe_only),  # cuda graph assumes a stable call pattern
        optimization_dt=optimization_dt,
        interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(config)
    gripper_joint_idx = mpc.joint_names.index("gripper_joint1")

    if debug_start_pose == "real":
        default_position = torch.tensor(
            [REAL_HARDWARE_START_POSE], device=mpc.default_joint_position.device,
            dtype=mpc.default_joint_position.dtype,
        )
        print(f"  [debug] overriding start pose to REAL_HARDWARE_START_POSE: {REAL_HARDWARE_START_POSE}")
    else:
        default_position = mpc.default_joint_position.clone().unsqueeze(0)

    bridge = RebotArmBridge(
        joint_names=mpc.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=default_position,
        observe_only=observe_only,
    )

    try:
        do_home = debug_start_pose != "real" and not observe_only
        if do_home:
            # Do all the slow, one-time MPC setup (cuda-graph capture, IK solve
            # for the goal) BEFORE physically moving/holding the arm, using the
            # known home pose analytically — not a hardware read. Otherwise
            # this setup happens AFTER home_to_pose()'s settle loop stops
            # sending commands, and the Damiao motors (MIT mode requires
            # periodic commands or they time out and release torque) let the
            # arm sag under gravity during that gap before the first real MPC
            # command arrives.
            home_state = JointState.from_position(
                mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names,
            )
            home_state.velocity = torch.zeros_like(home_state.position)
            home_state.acceleration = torch.zeros_like(home_state.position)
            mpc.setup(home_state)
            kin_result = mpc.compute_kinematics(home_state)
        else:
            current_state = bridge.read_joint_state(verbose=True)
            mpc.setup(current_state)
            kin_result = mpc.compute_kinematics(current_state)

        goal_poses = kin_result.tool_poses.to_dict()
        target_link = mpc.tool_frames[0]
        goal_poses[target_link].position[..., 1] += goal_offset_y

        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(goal_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
            run_ik=True,
        )

        if do_home:
            home_to_pose(bridge, mpc.default_joint_position.clone().unsqueeze(0))
        current_state = bridge.read_joint_state(verbose=True)
        # gripper_joint1 doesn't affect the tracked tool pose (gripper_link is
        # upstream of the finger joints), so MPC's pose-tracking cost is blind
        # to it and has no reason to move it sensibly -- held fixed rather
        # than trusting its ungoverned output (see rebot_mit_sim.py).
        gripper_hold_position = current_state.position[:, gripper_joint_idx].clone()
        bridge.set_target(current_state.position, current_state.velocity)
        bridge.start_command_thread()
        prev_accel = torch.zeros_like(current_state.position)

        print(f"Target link: {target_link}")
        print(f"Running MPC for {num_steps} steps (dry_run={dry_run}, observe_only={observe_only})...")

        for step in range(num_steps):
            result = mpc.optimize_action_sequence(current_state)

            pos_err = result.position_error
            err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
            has_action = (
                result.action_sequence is not None and result.action_sequence.position.shape[1] > 0
            )
            print(f"  Step {step + 1}/{num_steps}  position error: {err_str}  action_sequence_valid: {has_action}")

            if has_action:
                # Walk through the FULL interpolation_steps waypoint sequence
                # progressively, instead of jumping straight to the last
                # waypoint and holding it statically for the whole tick.
                # Holding one static target means the background thread's
                # 200Hz resend just repeats a single point for ~30ms, then
                # jumps straight to the next tick's point -- a coarse
                # staircase reference, not a smooth time-interpolated
                # trajectory. Confirmed live (2026-07-21): the arm tracks
                # fine, but with jerky/high-acceleration bursts at ticks
                # where the target moved further -- exactly this symptom.
                # Sending each of MPC's own already-computed waypoints in
                # turn, spaced at the same dt MPC planned them at, gives the
                # background thread's target updates roughly 4x the
                # resolution (matching rebot_mit_sim.py's fix for the same
                # coarseness in simulation).
                seq_pos = result.action_sequence.position.clone()  # [1, n_wp, dof]
                seq_vel = result.action_sequence.velocity.clone()
                seq_pos[:, :, gripper_joint_idx] = gripper_hold_position
                seq_vel[:, :, gripper_joint_idx] = 0.0
                n_wp = seq_pos.shape[1]
                wp_dt = optimization_dt / n_wp
                # Cap each waypoint's implied Cartesian tool speed at
                # MAX_CARTESIAN_VELOCITY -- occasional ticks still produce a
                # fast tool-frame move even with the per-tick interpolation
                # above (e.g. MPC solving a larger-than-usual correction).
                # Then send it as an acceleration/jerk-limited ramp
                # (send_smoothed_segment) rather than one linear jump --
                # confirmed live (2026-07-21): the fixed 4-waypoint
                # interpolation alone still let occasional large
                # accel/jerk bursts through, since it has no notion of
                # acceleration/jerk limits, only position.
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
                next_acceleration = result.action_sequence.acceleration[:, -1, :]
                # Replan from MPC's OWN last-commanded state, never real
                # hardware feedback. MPC has no dynamics model, so it can't
                # distinguish ordinary tracking lag from a genuine
                # disturbance -- replanning off real feedback every tick let
                # that lag compound into a runaway that ultimately crashed
                # the arm. Confirmed via rebot_mit_sim.py: seeding MPC's next
                # solve from its own last-commanded state ("I commanded X, so
                # I'm at X now" -- the same assumption rebot_ik_control.py
                # already made successfully) was the fix that made MPC
                # finally track stably in simulation. Real state is still
                # read below, but only for logging/monitoring -- never fed
                # back into MPC.
                current_state = JointState.from_position(next_position, joint_names=bridge.joint_names)
                current_state.velocity = next_velocity
                current_state.acceleration = next_acceleration
                bridge.read_joint_state(verbose=True)
            else:
                print("  [warn] MPC returned no valid action_sequence this step — nothing commanded.")
                time.sleep(optimization_dt)
    finally:
        bridge.close()

    return True


def run_interactive_mpc_control(
    dry_run: bool = True,
    port: str = "/dev/ttyACM0",
    baud: int = 921600,
    viz_port: int = 8080,
    control_dt: float = 0.03,
    scene_model: str = "collision_test.yml",
    observe_only: bool = False,
    debug_start_pose: str = "default",
):
    """Drive the rebot arm with MPC, using cuRobo's Viser web GUI as the goal source.

    Open http://localhost:<viz_port> and drag the end-effector gizmo — the MPC
    goal updates live and the browser's robot model mirrors the arm's actual
    joint state, read back from the bridge every tick for display/monitoring
    purposes only.

    MPC itself replans from its OWN last-commanded state, not this real
    feedback: it has no dynamics model, so it can't distinguish ordinary
    tracking lag from a genuine disturbance, and replanning off real
    feedback every tick let that lag compound into a runaway that crashed
    the arm. Confirmed via rebot_mit_sim.py: seeding MPC's next solve from
    its own last-commanded state ("I commanded X, so I'm at X now" -- the
    same assumption rebot_ik_control.py already made successfully) is what
    finally made MPC track stably in simulation. gripper_joint1 is held
    fixed throughout (it doesn't affect the tracked tool pose, so MPC's cost
    is blind to it and has no reason to move it sensibly).
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
        scene_model=scene_model,
        use_cuda_graph=not (dry_run or observe_only),
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
    gripper_joint_idx = mpc.joint_names.index("gripper_joint1")

    if debug_start_pose == "real":
        default_position = torch.tensor(
            [REAL_HARDWARE_START_POSE], device=mpc.default_joint_position.device,
            dtype=mpc.default_joint_position.dtype,
        )
        print(f"  [debug] overriding start pose to REAL_HARDWARE_START_POSE: {REAL_HARDWARE_START_POSE}")
    else:
        default_position = mpc.default_joint_position.clone().unsqueeze(0)

    bridge = RebotArmBridge(
        joint_names=mpc.joint_names,
        motor_map=MOTOR_MAP,
        port=port,
        baud=baud,
        dry_run=dry_run,
        default_position=default_position,
        observe_only=observe_only,
    )

    try:
        do_home = debug_start_pose != "real" and not observe_only
        if do_home:
            # Same reasoning as run_mpc_control: do the slow one-time MPC setup
            # BEFORE physically moving/holding the arm, using the known home
            # pose analytically, so there's no dead gap (no commands sent) for
            # the Damiao MIT-mode motors to time out and drop torque during.
            home_state = JointState.from_position(
                mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names,
            )
            home_state.velocity = torch.zeros_like(home_state.position)
            home_state.acceleration = torch.zeros_like(home_state.position)
            mpc.setup(home_state)
            kin_result = mpc.compute_kinematics(home_state)
        else:
            current_state = bridge.read_joint_state()
            mpc.setup(current_state)
            kin_result = mpc.compute_kinematics(current_state)

        target_link_poses = kin_result.tool_poses.to_dict()
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
            ),
            run_ik=False,
        )

        if do_home:
            home_to_pose(bridge, mpc.default_joint_position.clone().unsqueeze(0))
        current_state = bridge.read_joint_state()
        gripper_hold_position = current_state.position[:, gripper_joint_idx].clone()
        bridge.set_target(current_state.position, current_state.velocity)
        bridge.start_command_thread()
        prev_accel = torch.zeros_like(current_state.position)

        # ViserVisualizer places the draggable gizmo at FK(default_joint_state) once,
        # at construction time — before we ever knew the real hardware pose. Left
        # alone, the gizmo and the real robot start out anchored to two different
        # poses, so a "small" drag in the browser can translate into a large/
        # unreachable jump in the goal actually sent to the real robot's frame.
        # Snap the gizmo(s) to the real starting pose now so they match from frame one.
        viser_viz.set_joint_state(current_state.squeeze(0))
        for frame_name, pose in target_link_poses.items():
            if frame_name in viser_viz._control_frames:
                handle = viser_viz._control_frames[frame_name]
                handle.position = pose.position.cpu().squeeze().numpy()
                handle.wxyz = pose.quaternion.cpu().squeeze().numpy()

        print(f"\nInteractive MPC running at http://localhost:{viz_port}")
        print(f"Target links: {mpc.tool_frames}  (dry_run={dry_run})")
        print("Drag the end-effector gizmo to update the goal pose.")
        print("Press Ctrl+C to exit.\n")

        previous_target_poses = None
        pose_changed = False
        tick = 0
        ticks_per_report = max(1, int(round(1.0 / control_dt)))  # ~once per second
        believed_state = current_state

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
                target_link_poses = {
                    k.replace("target_", ""): v for k, v in target_poses.items()
                }
                mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(
                        target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
                    ),
                    run_ik=False,
                )
                pose_changed = False

            # Replan from MPC's OWN last-commanded state (believed_state),
            # never real hardware feedback -- see the function docstring for
            # why (the confirmed root cause of the live-hardware runaway).
            report = (tick % ticks_per_report == 0)
            mpc_result = mpc.optimize_action_sequence(believed_state)

            has_action = (
                mpc_result.action_sequence is not None and mpc_result.action_sequence.position.shape[1] > 0
            )
            if report:
                pos_err = mpc_result.position_error
                err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
                print(f"  [tick {tick}] position error: {err_str}  action_sequence_valid: {has_action}")

            if has_action:
                # Walk through the FULL interpolation_steps waypoint sequence
                # progressively instead of jumping straight to the last
                # waypoint and holding it statically for the whole tick --
                # see run_mpc_control's comment for why (confirmed live:
                # jerky/high-acceleration bursts from a coarse staircase
                # reference, tracking was never the problem).
                seq_pos = mpc_result.action_sequence.position.clone()  # [1, n_wp, dof]
                seq_vel = mpc_result.action_sequence.velocity.clone()
                seq_pos[:, :, gripper_joint_idx] = gripper_hold_position
                seq_vel[:, :, gripper_joint_idx] = 0.0
                n_wp = seq_pos.shape[1]
                wp_dt = control_dt / n_wp
                # Cap each waypoint's implied Cartesian tool speed, then send
                # it as an acceleration/jerk-limited ramp -- see
                # run_mpc_control's comments for both.
                prev_position = believed_state.position
                prev_velocity = believed_state.velocity
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
                believed_state = JointState.from_position(next_position, joint_names=bridge.joint_names)
                believed_state.velocity = next_velocity
                believed_state.acceleration = torch.zeros_like(next_position)
            else:
                print("  [warn] MPC returned no valid action_sequence this tick — nothing commanded.")
                time.sleep(control_dt)

            # Real state read here is for the Viser display / operator
            # monitoring only -- never fed back into MPC's next solve.
            real_state = bridge.read_joint_state(verbose=report)
            viser_viz.set_joint_state(real_state.squeeze(0))
            tick += 1
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="cuRobo reactive MPC control for the rebot arm")
    parser.add_argument(
        "--live", action="store_true",
        help="Open the real Damiao serial bridge and command motors. "
        "Requires MOTOR_MAP to be filled in. Default is --dry-run.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Damiao serial bridge port")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of MPC steps")
    parser.add_argument(
        "--goal-offset-y", type=float, default=0.2,
        help="Meters to offset the initial end-effector pose by in Y, as the MPC goal "
        "(headless mode only; ignored with --visualize)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch cuRobo's Viser web GUI and use the dragged end-effector gizmo as "
        "the MPC goal, instead of the fixed --goal-offset-y test goal.",
    )
    parser.add_argument("--viz-port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--observe-only", action="store_true",
        help="With --live: open the real serial port and read real joint state/run the MPC "
        "solve, but never enable torque or send a single motor command (fully safe, arm "
        "cannot move). Prints raw motor feedback and whether the MPC solve produced a valid "
        "action each step, for diagnosing a 'nothing moved' run before trying again for real.",
    )
    parser.add_argument(
        "--debug-start-pose", choices=["default", "real"], default="default",
        help="'real' overrides the simulated/dry-run starting joint state to "
        "REAL_HARDWARE_START_POSE (the arm's observed all-near-zero calibration pose), to "
        "reproduce a pose-specific MPC behavior in pure simulation. Has no effect with --live "
        "(real hardware state always comes from the actual motors).",
    )
    parser.add_argument(
        "--home-only", action="store_true",
        help="Move the arm to cuRobo's default_joint_position via simple open-loop joint "
        "control and stop — no MPC at all. Isolated test of the homing move by itself.",
    )
    parser.add_argument(
        "--no-gravity-comp", action="store_true",
        help="Disable Pinocchio gravity-torque feedforward (on by default). Use to isolate "
        "whether a fix (e.g. the enable/mode-set ordering) helps MPC on its own, vs. only "
        "in combination with gravity compensation.",
    )
    args = parser.parse_args()

    if args.no_gravity_comp:
        global ENABLE_GRAVITY_COMPENSATION
        ENABLE_GRAVITY_COMPENSATION = False
        print("  [config] gravity compensation DISABLED for this run")

    if args.live and not args.observe_only:
        confirm = input(
            "About to command REAL motors on /dev of your choosing. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    if args.home_only:
        run_home_only(dry_run=not args.live, port=args.port, baud=args.baud)
    elif args.visualize:
        run_interactive_mpc_control(
            dry_run=not args.live,
            port=args.port,
            baud=args.baud,
            viz_port=args.viz_port,
            observe_only=args.observe_only,
            debug_start_pose=args.debug_start_pose,
        )
    else:
        run_mpc_control(
            dry_run=not args.live,
            port=args.port,
            baud=args.baud,
            num_steps=args.num_steps,
            goal_offset_y=args.goal_offset_y,
            debug_start_pose=args.debug_start_pose,
            observe_only=args.observe_only,
        )


if __name__ == "__main__":
    main()
