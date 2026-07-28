# SPDX-License-Identifier: Apache-2.0
"""Teleoperate the reBot B601-DM arm with Meta Quest 3 left-hand tracking.

Uses pyopenxr (the `xr` package) over an already-confirmed-working WiVRn
OpenXR runtime. The OpenXR session runs in its own background thread with a
headless EGL graphics binding (``XR_MNDX_egl_enable`` -- supported by
Monado-based runtimes like WiVRn) so no window/display is needed; this
thread does no CUDA/torch work at all (only OpenXR calls + numpy math),
matching the established rule that a background thread running concurrently
with CUDA-graph-capturing solver calls must stay 100% CUDA-free (see
``RebotArmBridge``'s background command thread in ``rebot_mpc_control.py``).

Pipeline:

1. Read the left hand's wrist pose from OpenXR hand tracking, recomputed
   relative to the head (egocentric). Uses head POSITION live (so the
   origin follows the operator if they lean/shift) but a FIXED reference
   orientation captured once, from the first valid head reading, for the
   rotation part -- turning your head to look around (the only way to look
   left/right/back in VR) must NOT itself inject hand motion into the
   mapped target, which it would if the live head orientation were used
   for the rotation too. Converted from OpenXR's coordinate convention
   (right-handed, +Y up, +X right, forward = -Z) to cuRobo/Viser/ROS's
   convention (right-handed, +Z up, +X forward, +Y left, REP-103-compatible).
2. A 6-direction calibration wizard records the operator's comfortable
   hand range of motion per axis, and a one-time FK-sampled estimate of
   the arm's own reachable workspace bounds; hand position is linearly
   mapped from the former range to the latter, clamped as a safety net.
   Orientation is relative/delta-tracked (hand's rotation away from its
   calibrated neutral, applied on top of the robot's own neutral EE
   orientation) rather than absolutely mapped.
3. The mapped target pose feeds into one of four control backends,
   selected via ``--mode``: one-shot IK (mirrors ``rebot_ik_control.py``),
   continuous MPC (mirrors ``rebot_mpc_control.py``'s
   ``run_interactive_mpc_control``, including the replan-from-belief fix,
   Cartesian speed cap, and jerk/accel-limited sending), or motion
   planning with an optional MPC fallback for otherwise-unreachable
   targets (mirrors ``rebot_motion_planning.py``). This is a standalone
   script -- it reuses those scripts' module-level utilities but does not
   modify or import their (non-reusable, closure-nested) control loops.

Left-hand pinch (thumb tip to index tip distance) continuously drives the
gripper open/close -- the one script in this project where the gripper is
actively teleoperated rather than held fixed.

Usage:

.. code-block:: bash

   # Safe: no hardware touched.
   python quest3_ego_teleop.py --mode ik

   # Real hardware.
   python quest3_ego_teleop.py --live --mode trajectory_mpc --port /dev/ttyACM0 --baud 921600
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import threading
import time
from ctypes import byref, cast
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import torch
import xr
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.spatial.transform import Rotation
from xr.utils.gl.context_object import ContextObject
from xr.utils.gl.egl_util import EGLOffscreenContextProvider

faulthandler.enable()


def _locate_hand_joints_ext(hand_tracker, locate_info):
    """xr.locate_hand_joints_ext() constructs a bare HandJointLocationsEXT()
    with a NULL joint_locations pointer / zero count and never allocates an
    output array before calling the C function -- xrLocateHandJointsEXT
    rejects that with XR_ERROR_VALIDATION_FAILURE
    (locations->jointLocations == NULL). Pre-allocate the array ourselves
    and call the raw function directly instead.
    """
    joint_locations_array = (xr.HandJointLocationEXT * xr.HAND_JOINT_COUNT_EXT)()
    locations = xr.HandJointLocationsEXT(joint_locations=joint_locations_array)
    fxn = cast(
        xr.get_instance_proc_addr(hand_tracker.instance, "xrLocateHandJointsEXT"),
        xr.PFN_xrLocateHandJointsEXT,
    )
    result = xr.check_result(fxn(hand_tracker, locate_info, byref(locations)))
    if result.is_exception():
        raise result
    return locations

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

from rebot_mit_sim import CSPACE_JOINT_NAMES, GRIPPER_JOINT_IDX
from rebot_motion_planning import _create_trajectory_image, _hold_gripper
from rebot_mpc_control import (
    MOTOR_MAP,
    REBOT_URDF_PATH,
    ROBOT_YML,
    RebotArmBridge,
    _tool_position,
    cap_cartesian_step,
    compute_gravity_torque,
    home_to_pose,
    send_smoothed_segment,
)

GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 0.06
# Calibrate these live if pinch feels off -- raw OpenXR thumb-tip/index-tip
# distance (meters) at which the gripper should be fully open/closed.
PINCH_OPEN_DISTANCE = 0.09
PINCH_CLOSED_DISTANCE = 0.015

# Butterworth low-pass filter for the raw hand-tracked EE pose, same
# style/purpose as rebot_mpc_control.py's VELOCITY_FILTER_* (real velocity
# feedback there is noisy for the same reason camera-based hand tracking is
# here: smooths jitter before it's used as a control target). Applied in
# OpenXrHandTracker itself (not per control-mode loop) so filtering happens
# at a consistent rate regardless of which mode's control loop is reading
# get_state() and at what rate.
EE_POSE_FILTER_ORDER = 2
EE_POSE_FILTER_CUTOFF_HZ = 3.0
EE_POSE_FILTER_SAMPLE_HZ = 72.0  # approximate Quest 3/WiVRn frame rate
_EE_POSE_FILTER_B, _EE_POSE_FILTER_A = butter(
    EE_POSE_FILTER_ORDER, EE_POSE_FILTER_CUTOFF_HZ / (EE_POSE_FILTER_SAMPLE_HZ / 2), btype="low"
)

# If the newly-mapped target hasn't moved more than this (meters) since the
# last update actually sent to a control backend, keep using the previous
# target -- avoids needless re-solving/replanning and residual jitter from
# hand-tracking noise the Butterworth filter above doesn't fully remove.
TARGET_POSE_DEADBAND_M = 0.01


# --- OpenXR -> ROS/cuRobo/Viser coordinate conversion -----------------------
# OpenXR: right-handed, +Y up, +X right, forward = -Z.
# ROS/cuRobo/Viser: right-handed, +Z up, +X forward, +Y left (REP-103).
# This is a proper rotation (both frames right-handed), not a reflection:
#   ros_x = -openxr_z   (openxr forward -> ros forward)
#   ros_y = -openxr_x   (openxr right -> ros left, negated)
#   ros_z =  openxr_y   (both up)
_OPENXR_TO_ROS = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def _openxr_pos_to_ros(p: np.ndarray) -> np.ndarray:
    return _OPENXR_TO_ROS @ p


def _openxr_rot_to_ros(rot: Rotation) -> Rotation:
    return Rotation.from_matrix(_OPENXR_TO_ROS @ rot.as_matrix() @ _OPENXR_TO_ROS.T)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def _cspace_joint_limits():
    """Lower/upper position limits for CSPACE_JOINT_NAMES, from the URDF
    (via Pinocchio) -- same source rebot_mit_sim.py's MitDynamicsSimulator
    already uses for its own joint-limit enforcement.
    """
    model = pin.buildModelFromUrdf(REBOT_URDF_PATH)
    names = model.names.tolist()
    lower = np.array([model.lowerPositionLimit[names.index(n) - 1] for n in CSPACE_JOINT_NAMES])
    upper = np.array([model.upperPositionLimit[names.index(n) - 1] for n in CSPACE_JOINT_NAMES])
    return lower, upper


# --- Hand tracking -----------------------------------------------------------
@dataclass
class HandState:
    position: np.ndarray  # (3,) ROS frame, relative to head
    orientation: np.ndarray  # (4,) wxyz, ROS frame, relative to head
    pinch_distance: float  # meters, raw OpenXR units (head-relative transform doesn't matter for a distance)
    timestamp: float = field(default_factory=time.time)


class OpenXrHandTracker:
    """Background-threaded OpenXR session polling left-hand tracking + head pose.

    Runs entirely in its own thread via a headless EGL graphics binding --
    see the module docstring for why this thread must and does stay
    CUDA-free.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Optional[HandState] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._error: Optional[BaseException] = None
        self._diag_last_print = 0.0
        # Captured once, from the first valid head reading -- see _poll()'s
        # comment on why the hand-relative-to-head transform uses head
        # POSITION live but this FIXED reference for rotation.
        self._head_orientation_ref: Optional[Rotation] = None
        # Butterworth filter state (scipy.signal.lfilter's `zi`), one per
        # position channel (x,y,z) and one per orientation channel (w,x,y,z
        # -- filtered in wxyz order, then renormalized; see _poll()).
        # None until the first sample initializes it flat (avoids a
        # start-up transient ramping from zero).
        self._pos_filter_state: Optional[list] = None
        self._quat_filter_state: Optional[list] = None
        self._prev_quat_wxyz: Optional[np.ndarray] = None

    def start(self, timeout: float = 20.0):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            raise RuntimeError(
                f"OpenXR session did not become ready within {timeout:.0f}s -- "
                "check the headset is on, WiVRn is connected, and hand tracking is enabled."
            )
        if self._error is not None:
            raise RuntimeError("OpenXR session failed to start") from self._error

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def get_state(self, max_staleness: float = 0.5) -> Optional[HandState]:
        """Returns the last hand pose seen, even if the most recent poll(s)
        reported tracking as momentarily inactive (hand tracking commonly
        drops out briefly during fast motion or partial occlusion) -- as
        long as it's no older than max_staleness seconds. None if tracking
        has been lost for longer than that, or never acquired at all.
        """
        with self._lock:
            state = self._state
        if state is None:
            return None
        if time.time() - state.timestamp > max_staleness:
            return None
        return state

    def _run(self):
        # Deliberately verbose, step-by-step (rather than relying on
        # ContextObject.__enter__()'s all-in-one setup): this is genuinely
        # new integration territory (no bundled pyopenxr example for hand
        # tracking, EGL headless graphics binding untested before this
        # project), so each OpenXR/EGL call is bracketed with a flushed
        # print -- if something crashes at the C level (a segfault, which
        # leaves no Python traceback), the last printed line pinpoints
        # exactly which call is responsible. Also skips swapchain/view
        # setup entirely -- genuinely unneeded for pure pose polling with
        # no rendering, not just simplified for debugging.
        instance = session = None
        try:
            print("[xr] creating EGL offscreen context...", flush=True)
            provider = EGLOffscreenContextProvider()
            print("[xr] EGL offscreen context created OK", flush=True)

            print("[xr] creating OpenXR instance...", flush=True)
            instance = xr.create_instance(
                create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        xr.MNDX_EGL_ENABLE_EXTENSION_NAME,
                        xr.EXT_HAND_TRACKING_EXTENSION_NAME,
                    ],
                ),
            )
            print("[xr] instance created OK", flush=True)

            print("[xr] getting system...", flush=True)
            system_id = xr.get_system(
                instance=instance, get_info=xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )
            print("[xr] system OK", flush=True)

            print("[xr] building OpenGL(EGL) graphics binding...", flush=True)
            from xr.utils.gl import OpenGLGraphics
            graphics = OpenGLGraphics(instance=instance, system=system_id, context_provider=provider)
            print("[xr] graphics binding OK", flush=True)

            print("[xr] creating session...", flush=True)
            session = xr.create_session(
                instance=instance,
                create_info=xr.SessionCreateInfo(system_id=system_id, next=graphics.graphics_binding.pointer),
            )
            print("[xr] session created OK", flush=True)

            print("[xr] creating LOCAL reference space...", flush=True)
            local_space = xr.create_reference_space(
                session=session,
                create_info=xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.LOCAL),
            )
            print("[xr] LOCAL space OK", flush=True)

            print("[xr] creating VIEW reference space...", flush=True)
            view_space = xr.create_reference_space(
                session=session,
                create_info=xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.VIEW),
            )
            print("[xr] VIEW space OK", flush=True)

            print("[xr] creating left-hand tracker...", flush=True)
            hand_tracker = xr.create_hand_tracker_ext(
                session, xr.HandTrackerCreateInfoEXT(hand=xr.HandEXT.LEFT, hand_joint_set=xr.HandJointSetEXT.DEFAULT),
            )
            print("[xr] hand tracker OK", flush=True)

            # frame_loop() unconditionally calls xrAttachSessionActionSets,
            # which requires at least one action set even though we don't
            # need any real input actions (hand tracking is a separate
            # extension, not tied to action sets) -- an empty, unused one
            # satisfies the validation requirement.
            print("[xr] creating placeholder action set...", flush=True)
            action_set = xr.create_action_set(
                instance=instance,
                create_info=xr.ActionSetCreateInfo(
                    action_set_name="quest3_teleop_actions",
                    localized_action_set_name="Quest 3 Teleop Actions",
                    priority=0,
                ),
            )
            print("[xr] action set OK -- entering frame loop", flush=True)

            context = ContextObject.__new__(ContextObject)
            context.instance = instance
            context.system_id = system_id
            context.graphics = graphics
            context.session = session
            context.space = local_space
            context.session_state = xr.SessionState.IDLE
            context.session_is_running = False
            context.action_sets = [action_set]
            context.render_layers = []
            context.view_configuration_type = xr.ViewConfigurationType.PRIMARY_STEREO
            context.environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
            context.form_factor = xr.FormFactor.HEAD_MOUNTED_DISPLAY
            context.exit_render_loop = False
            context.request_restart = False

            frame_count = 0
            try:
                for frame_state in context.frame_loop():
                    if frame_count < 3:
                        print(f"[xr] frame {frame_count} OK", flush=True)
                    frame_count += 1
                    if self._stop_event.is_set():
                        break
                    self._ready_event.set()
                    self._poll(local_space, hand_tracker, view_space, frame_state.predicted_display_time)
            finally:
                xr.destroy_action_set(action_set)
                xr.destroy_hand_tracker_ext(hand_tracker)
                xr.destroy_space(view_space)
                xr.destroy_space(local_space)
        except BaseException as exc:  # noqa: BLE001 -- surface any OpenXR failure to start()
            self._error = exc
            self._ready_event.set()
        finally:
            if session is not None:
                xr.destroy_session(session)
            if instance is not None:
                xr.destroy_instance(instance)

    def _poll(self, local_space, hand_tracker, view_space, display_time):
        # Deliberately does NOT clear self._state on a momentary tracking
        # dropout -- hand tracking commonly loses tracking briefly during
        # fast motion or partial occlusion, and get_state()'s staleness
        # check already handles expiry of genuinely stale data. Clearing
        # eagerly here caused every hand-not-tracked-at-this-exact-instant
        # read (e.g. right when a calibration prompt is answered) to report
        # None even though the headset's own UI showed the hand tracked
        # moments before/after.
        valid_flags = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT

        joints = _locate_hand_joints_ext(
            hand_tracker, xr.HandJointsLocateInfoEXT(base_space=local_space, time=display_time),
        )
        if not joints.is_active:
            self._log_diagnostic("hand tracker reports is_active=False")
            return

        wrist = joints.joint_locations[int(xr.HandJointEXT.WRIST)]
        thumb_tip = joints.joint_locations[int(xr.HandJointEXT.THUMB_TIP)]
        index_tip = joints.joint_locations[int(xr.HandJointEXT.INDEX_TIP)]
        if (wrist.location_flags & valid_flags) != valid_flags:
            self._log_diagnostic(f"wrist location_flags={wrist.location_flags:#x} (need {valid_flags:#x})")
            return

        head = xr.locate_space(view_space, local_space, display_time)
        if (head.location_flags & valid_flags) != valid_flags:
            self._log_diagnostic(f"head location_flags={head.location_flags:#x} (need {valid_flags:#x})")
            return

        self._log_diagnostic("tracking OK", ok=True)

        head_pos = np.array([head.pose.position.x, head.pose.position.y, head.pose.position.z])
        head_rot_live = Rotation.from_quat([
            head.pose.orientation.x, head.pose.orientation.y, head.pose.orientation.z, head.pose.orientation.w,
        ])
        wrist_pos = np.array([wrist.pose.position.x, wrist.pose.position.y, wrist.pose.position.z])
        wrist_rot = Rotation.from_quat([
            wrist.pose.orientation.x, wrist.pose.orientation.y, wrist.pose.orientation.z, wrist.pose.orientation.w,
        ])

        if self._head_orientation_ref is None:
            self._head_orientation_ref = head_rot_live
            print("[hand] head orientation reference captured (facing direction locked for this session)", flush=True)

        # Hand pose relative to head, still in OpenXR's native frame.
        # Deliberately uses head POSITION live (so the origin follows the
        # operator if they lean/shift) but the FIXED reference orientation
        # captured above (NOT head_rot_live) for the rotation -- otherwise
        # ordinary head-turning (the only way to look left/right/back in
        # VR) would rotate this vector and inject spurious hand motion even
        # with the hand held perfectly still.
        head_rot = self._head_orientation_ref
        rel_pos_openxr = head_rot.inv().apply(wrist_pos - head_pos)
        rel_rot_openxr = head_rot.inv() * wrist_rot

        ros_pos = _openxr_pos_to_ros(rel_pos_openxr)
        ros_rot = _openxr_rot_to_ros(rel_rot_openxr)
        ros_quat_wxyz = _xyzw_to_wxyz(ros_rot.as_quat())
        ros_pos, ros_quat_wxyz = self._filter_pose(ros_pos, ros_quat_wxyz)

        pinch_distance = float(np.linalg.norm(
            np.array([thumb_tip.pose.position.x, thumb_tip.pose.position.y, thumb_tip.pose.position.z])
            - np.array([index_tip.pose.position.x, index_tip.pose.position.y, index_tip.pose.position.z])
        ))

        with self._lock:
            self._state = HandState(position=ros_pos, orientation=ros_quat_wxyz, pinch_distance=pinch_distance)

    def _filter_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray):
        """Butterworth-filters position (3ch) and orientation (4ch, wxyz)
        with persistent per-channel state, flat-initialized on the first
        sample (via lfilter_zi) to avoid a start-up transient ramping from
        zero toward the real pose. Orientation sign is kept consistent with
        the previous sample first (q and -q represent the same rotation;
        filtering across an unhandled sign flip would average toward a
        wrong intermediate rotation), then renormalized after filtering
        (linear filtering of quaternion components isn't exactly valid on
        the rotation manifold, but is a standard, adequate approximation
        for smoothing small frame-to-frame jitter).
        """
        if self._prev_quat_wxyz is not None and np.dot(quat_wxyz, self._prev_quat_wxyz) < 0:
            quat_wxyz = -quat_wxyz
        self._prev_quat_wxyz = quat_wxyz

        if self._pos_filter_state is None:
            zi = lfilter_zi(_EE_POSE_FILTER_B, _EE_POSE_FILTER_A)
            self._pos_filter_state = [zi * pos[i] for i in range(3)]
            self._quat_filter_state = [zi * quat_wxyz[i] for i in range(4)]

        filtered_pos = np.zeros(3)
        for i in range(3):
            out, self._pos_filter_state[i] = lfilter(
                _EE_POSE_FILTER_B, _EE_POSE_FILTER_A, [pos[i]], zi=self._pos_filter_state[i],
            )
            filtered_pos[i] = out[0]

        filtered_quat = np.zeros(4)
        for i in range(4):
            out, self._quat_filter_state[i] = lfilter(
                _EE_POSE_FILTER_B, _EE_POSE_FILTER_A, [quat_wxyz[i]], zi=self._quat_filter_state[i],
            )
            filtered_quat[i] = out[0]
        filtered_quat = filtered_quat / max(float(np.linalg.norm(filtered_quat)), 1e-9)

        return filtered_pos, filtered_quat

    def _log_diagnostic(self, message: str, ok: bool = False, interval: float = 1.0):
        """Throttled diagnostic print -- shows in real time whether tracking
        is genuinely never active vs. just intermittently dropping out.
        """
        now = time.time()
        if now - self._diag_last_print < interval:
            return
        self._diag_last_print = now
        prefix = "[hand]" if ok else "[hand:lost]"
        print(f"{prefix} {message}", flush=True)


def pinch_to_gripper_position(pinch_distance: float) -> float:
    """Linearly maps raw pinch distance to the gripper's commanded position, clamped."""
    frac = (pinch_distance - PINCH_CLOSED_DISTANCE) / (PINCH_OPEN_DISTANCE - PINCH_CLOSED_DISTANCE)
    frac = max(0.0, min(1.0, frac))
    return GRIPPER_CLOSED_POSITION + frac * (GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION)


def _prompt_with_live_feedback(prompt: str, hand_tracker: OpenXrHandTracker):
    """input(), but prints the live hand pose (updating in place) while
    waiting -- input() blocks the main thread, so this runs a tiny
    background printer alongside it. Lets the operator see tracking is
    actually working (and get a feel for the reading) before committing a
    calibration point by pressing Enter.
    """
    stop = threading.Event()

    def _print_loop():
        while not stop.is_set():
            state = hand_tracker.get_state()
            if state is not None:
                print(
                    f"\r  live: pos=({state.position[0]:+.3f}, {state.position[1]:+.3f}, {state.position[2]:+.3f}) m"
                    f"  pinch={state.pinch_distance:.3f} m          ",
                    end="", flush=True,
                )
            else:
                print("\r  live: hand not tracked...                                              ", end="", flush=True)
            stop.wait(0.2)

    printer = threading.Thread(target=_print_loop, daemon=True)
    printer.start()
    try:
        input(prompt)
    finally:
        stop.set()
        printer.join(timeout=1.0)
        print()  # newline after the in-place-updating live line


# --- Workspace calibration ----------------------------------------------------
# Calibration only asks the operator to reach FORWARD, not up/down/left/right.
# The head-orientation reference used to decouple hand-relative-to-head pose
# from head movement (see OpenXrHandTracker._poll) is only trustworthy while
# the head stays roughly forward-facing -- there's no external observer to
# ground-truth the headset's own drift, so asking the operator to crane their
# head to look up/down/left/right at their hand during calibration would
# contaminate exactly the samples we rely on being clean. Forward reach keeps
# the head still and forward-facing, and the operator naturally sweeps their
# hand left/right/up/down a little while doing it (that range is expected to
# be small compared to the forward one) -- Y/Z bounds are sampled live during
# the same forward-reach window instead of via separate directional prompts.
FORWARD_SAMPLING_DURATION_S = 5.0


class WorkspaceCalibrator:
    """Maps the operator's hand range of motion to the arm's reachable workspace.

    robot_min/max/center come from one-time FK sampling (approximate -- see
    module docstring); hand_min/max/center come from a forward-reach-only
    calibration (see FORWARD_SAMPLING_DURATION_S above for why).
    """

    def __init__(self, kin_solver, joint_lower: np.ndarray, joint_upper: np.ndarray, num_samples: int = 3000):
        self.hand_min = np.zeros(3)
        self.hand_max = np.zeros(3)
        self.hand_center = np.zeros(3)
        self.hand_orientation_ref = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        print(f"Sampling {num_samples} random configurations to estimate the arm's reachable workspace...")
        self.robot_min, self.robot_max, self.robot_center = self._sample_robot_workspace(
            kin_solver, joint_lower, joint_upper, num_samples,
        )
        print(f"  robot workspace bounds (approx): min={self.robot_min}  max={self.robot_max}")

    @staticmethod
    def _sample_robot_workspace(kin_solver, lower: np.ndarray, upper: np.ndarray, num_samples: int):
        lower_t = torch.as_tensor(lower, device="cuda", dtype=torch.float32)
        upper_t = torch.as_tensor(upper, device="cuda", dtype=torch.float32)
        samples = lower_t + torch.rand((num_samples, lower_t.shape[0]), device="cuda") * (upper_t - lower_t)
        tool_frame = kin_solver.tool_frames[0]
        positions = []
        chunk = 256
        for i in range(0, num_samples, chunk):
            batch = samples[i:i + chunk]
            state = JointState.from_position(batch, joint_names=kin_solver.joint_names)
            kin_result = kin_solver.compute_kinematics(state)
            positions.append(kin_result.tool_poses.to_dict()[tool_frame].position.cpu().numpy())
        points = np.concatenate(positions, axis=0)
        robot_min = points.min(axis=0)
        robot_max = points.max(axis=0)
        robot_center = (robot_min + robot_max) / 2.0
        return robot_min, robot_max, robot_center

    def calibrate_hand_range(self, hand_tracker: OpenXrHandTracker):
        print("\n=== Hand range calibration ===")

        print("Hold your hand in a relaxed, natural resting position close to your body.")
        _prompt_with_live_feedback("Press Enter when ready...", hand_tracker)
        rest_state = hand_tracker.get_state()
        if rest_state is None:
            print("  [warn] hand not tracked right now -- resting position recorded as 0.0.")
            rest_pos = np.zeros(3)
        else:
            rest_pos = rest_state.position.copy()
            print(f"  recorded rest position: {rest_pos}")

        print(
            f"\nNow reach your hand forward to a comfortable maximum, then for the next "
            f"{FORWARD_SAMPLING_DURATION_S:.0f}s move it naturally -- left/right, up/down -- "
            f"within your comfortable reach while keeping it out in front of you.",
        )
        _prompt_with_live_feedback("Press Enter to begin sampling...", hand_tracker)
        sampled_min = np.full(3, np.inf)
        sampled_max = np.full(3, -np.inf)
        start_time = time.time()
        while time.time() - start_time < FORWARD_SAMPLING_DURATION_S:
            state = hand_tracker.get_state()
            if state is not None:
                sampled_min = np.minimum(sampled_min, state.position)
                sampled_max = np.maximum(sampled_max, state.position)
                print(
                    f"\r  sampling... pos=({state.position[0]:+.3f}, {state.position[1]:+.3f}, "
                    f"{state.position[2]:+.3f}) m          ",
                    end="", flush=True,
                )
            else:
                print("\r  sampling... hand not tracked...                                   ", end="", flush=True)
            time.sleep(0.05)
        print("\n  sampling complete.")

        if not np.all(np.isfinite(sampled_min)):
            print("  [warn] hand was never tracked during sampling -- recording 0.0 for all axes.")
            sampled_min = np.zeros(3)
            sampled_max = np.zeros(3)

        # Forward(X): near bound comes from the rest position, far bound from the
        # sampled forward reach. Y/Z: both bounds come straight from the natural
        # sweep recorded during the same forward-reach window.
        self.hand_min[0] = min(rest_pos[0], sampled_min[0])
        self.hand_max[0] = max(rest_pos[0], sampled_max[0])
        self.hand_min[1:] = sampled_min[1:]
        self.hand_max[1:] = sampled_max[1:]
        self.hand_center = (self.hand_min + self.hand_max) / 2.0

        print("\nHold your hand at a comfortable neutral position -- this becomes the orientation reference.")
        _prompt_with_live_feedback("Press Enter when ready...", hand_tracker)
        state = hand_tracker.get_state()
        if state is not None:
            self.hand_orientation_ref = state.orientation.copy()
        else:
            print("  [warn] hand not tracked -- orientation reference left at identity.")

        print("\nCalibration complete.")
        print(f"  hand_min:     {self.hand_min}")
        print(f"  hand_max:     {self.hand_max}")
        print(f"  hand_center:  {self.hand_center}")
        print(f"  robot_min:    {self.robot_min}")
        print(f"  robot_max:    {self.robot_max}")
        print(f"  robot_center: {self.robot_center}\n")

    def map_to_robot(
        self, hand_pos: np.ndarray, hand_orientation_wxyz: np.ndarray, robot_neutral_orientation_wxyz: np.ndarray,
    ):
        """Returns (target_position_xyz, target_orientation_wxyz)."""
        hand_range = np.maximum(self.hand_max - self.hand_min, 1e-6)
        scale = (self.robot_max - self.robot_min) / hand_range
        target_pos = self.robot_center + (hand_pos - self.hand_center) * scale
        target_pos = np.clip(target_pos, self.robot_min, self.robot_max)

        hand_ref_rot = Rotation.from_quat(_wxyz_to_xyzw(self.hand_orientation_ref))
        hand_cur_rot = Rotation.from_quat(_wxyz_to_xyzw(hand_orientation_wxyz))
        delta_rot = hand_ref_rot.inv() * hand_cur_rot
        robot_neutral_rot = Rotation.from_quat(_wxyz_to_xyzw(robot_neutral_orientation_wxyz))
        target_rot = robot_neutral_rot * delta_rot
        target_quat_wxyz = _xyzw_to_wxyz(target_rot.as_quat())
        return target_pos, target_quat_wxyz


def _pose_dict(tool_frame: str, position_xyz: np.ndarray, quaternion_wxyz: np.ndarray) -> Dict[str, Pose]:
    return {tool_frame: Pose.from_numpy(position_xyz.astype(np.float32), quaternion_wxyz.astype(np.float32))}


# --- Shared per-tick target-building helper ----------------------------------
class TargetPoseTracker:
    """Reads the current hand state, maps it to a target Pose dict, and
    applies a position deadband (TARGET_POSE_DEADBAND_M) before handing the
    result to a control backend: if the newly-mapped target hasn't moved
    more than the deadband since the last update actually used, the
    previous target is returned unchanged instead. This is on top of
    OpenXrHandTracker's own Butterworth filtering (which smooths the raw
    hand pose) -- the deadband specifically avoids needless re-solving/
    replanning from whatever residual jitter remains once it's this close
    to already being at the target.

    One instance per mode-run, shared across every call site that needs
    the current target within that run (so deadband state is consistent,
    not duplicated).
    """

    def __init__(
        self, hand_tracker: OpenXrHandTracker, calibrator: WorkspaceCalibrator, tool_frame: str,
        robot_neutral_orientation_wxyz: np.ndarray, deadband: float = TARGET_POSE_DEADBAND_M,
    ):
        self.hand_tracker = hand_tracker
        self.calibrator = calibrator
        self.tool_frame = tool_frame
        self.robot_neutral_orientation_wxyz = robot_neutral_orientation_wxyz
        self.deadband = deadband
        self._last_pos: Optional[np.ndarray] = None
        self._last_quat: Optional[np.ndarray] = None
        self._last_pinch_distance: Optional[float] = None

    def get_gripper_position(self) -> Optional[float]:
        """Gripper position mapped from the current pinch distance, or the
        last known one if the hand isn't tracked THIS instant -- same
        fallback reasoning as get_target(); avoids callers crashing on
        hand_tracker.get_state() being None while get_target() is still
        happily returning a cached target.
        """
        state = self.hand_tracker.get_state()
        if state is not None:
            self._last_pinch_distance = state.pinch_distance
        if self._last_pinch_distance is None:
            return None
        return pinch_to_gripper_position(self._last_pinch_distance)

    def get_target(self) -> Optional[Dict[str, Pose]]:
        """None only if the hand has never been tracked yet (caller should
        hold position) -- once a target has been established, transient
        tracking loss keeps returning the last target (matching
        OpenXrHandTracker.get_state()'s own staleness-tolerant caching).
        """
        state = self.hand_tracker.get_state()
        if state is None:
            if self._last_pos is None:
                return None
            target_pos, target_quat = self._last_pos, self._last_quat
        else:
            target_pos, target_quat = self.calibrator.map_to_robot(
                state.position, state.orientation, self.robot_neutral_orientation_wxyz,
            )
            if self._last_pos is not None and np.linalg.norm(target_pos - self._last_pos) < self.deadband:
                target_pos, target_quat = self._last_pos, self._last_quat
            else:
                self._last_pos, self._last_quat = target_pos, target_quat
        return _pose_dict(self.tool_frame, target_pos, target_quat)


# --- Mode: IK -----------------------------------------------------------------
def run_ik_teleop(
    hand_tracker: OpenXrHandTracker, calibrator: WorkspaceCalibrator, bridge: RebotArmBridge,
    viser_viz: ViserVisualizer, current_state: JointState, robot_neutral_orientation_wxyz: np.ndarray,
    control_dt: float = 0.05,
):
    config = InverseKinematicsCfg.create(
        robot=ROBOT_YML,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
    )
    ik_solver = InverseKinematics(config)
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False
    tool_frame = ik_solver.tool_frames[0]

    unbatched_state = JointState.from_position(current_state.position.squeeze(0), joint_names=bridge.joint_names)
    active_js = ik_solver.get_active_js(unbatched_state).unsqueeze(0)
    kin_state = ik_solver.compute_kinematics(current_state)
    goal_tool_poses = kin_state.tool_poses.to_dict()
    ik_solver.solve_pose(
        goal_tool_poses=GoalToolPose.from_poses(goal_tool_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1),
        current_state=active_js.clone(), return_seeds=1,
    )  # warm up CUDA graph before the background command thread starts

    bridge.start_command_thread()
    tracker = TargetPoseTracker(hand_tracker, calibrator, tool_frame, robot_neutral_orientation_wxyz)
    print(f"\n[ik mode] tracking left hand -> {tool_frame}. Press Ctrl+C to exit.\n")

    while True:
        target_poses = tracker.get_target()
        if target_poses is not None:
            active_js = ik_solver.get_active_js(current_state.squeeze(0)).unsqueeze(0)
            result = ik_solver.solve_pose(
                goal_tool_poses=GoalToolPose.from_poses(target_poses, ordered_tool_frames=ik_solver.tool_frames, num_goalset=1),
                current_state=active_js.squeeze(1).clone(), return_seeds=1,
            )
            if result.success.any():
                solved_position = result.js_solution.squeeze(0).squeeze(0).position.unsqueeze(0)
                gripper_pos = tracker.get_gripper_position()
                if gripper_pos is not None:
                    solved_position[:, GRIPPER_JOINT_IDX] = gripper_pos
                bridge.set_target(solved_position, torch.zeros_like(solved_position))
                current_state = JointState.from_position(solved_position, joint_names=bridge.joint_names)
                viser_viz.set_joint_state(current_state.squeeze(0))
        time.sleep(control_dt)


# --- Mode: MPC -----------------------------------------------------------------
def run_mpc_teleop(
    hand_tracker: OpenXrHandTracker, calibrator: WorkspaceCalibrator, bridge: RebotArmBridge,
    viser_viz: ViserVisualizer, current_state: JointState, robot_neutral_orientation_wxyz: np.ndarray,
    control_dt: float = 0.03,
):
    config = ModelPredictiveControlCfg.create(
        robot=ROBOT_YML, scene_model="collision_test.yml", use_cuda_graph=True,
        optimization_dt=control_dt, interpolation_steps=4, optimizer_collision_activation_distance=0.03,
    )
    mpc = ModelPredictiveControl(config)
    tool_frame = mpc.tool_frames[0]

    mpc.setup(current_state)
    kin_result = mpc.compute_kinematics(current_state)
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(kin_result.tool_poses.to_dict(), ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=False,
    )
    mpc.optimize_action_sequence(current_state)  # warm up CUDA graph before the background thread starts

    bridge.start_command_thread()
    believed_state = current_state
    prev_accel = torch.zeros_like(current_state.position)
    tracker = TargetPoseTracker(hand_tracker, calibrator, tool_frame, robot_neutral_orientation_wxyz)
    print(f"\n[mpc mode] tracking left hand -> {tool_frame}. Press Ctrl+C to exit.\n")

    while True:
        target_poses = tracker.get_target()
        if target_poses is not None:
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(target_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
                run_ik=False,
            )

        # Replan from MPC's OWN last-commanded state, never real hardware
        # feedback -- the confirmed fix for the live-hardware runaway (see
        # rebot_mpc_control.py's run_interactive_mpc_control docstring).
        result = mpc.optimize_action_sequence(believed_state)
        has_action = result.action_sequence is not None and result.action_sequence.position.shape[1] > 0
        if has_action:
            gripper_pos = tracker.get_gripper_position()
            seq_pos = result.action_sequence.position.clone()
            seq_vel = result.action_sequence.velocity.clone()
            if gripper_pos is not None:
                seq_pos[:, :, GRIPPER_JOINT_IDX] = gripper_pos
            seq_vel[:, :, GRIPPER_JOINT_IDX] = 0.0
            n_wp = seq_pos.shape[1]
            wp_dt = control_dt / n_wp

            prev_position = believed_state.position
            prev_velocity = believed_state.velocity
            prev_tool_position = _tool_position(mpc, prev_position)
            for wp in range(n_wp):
                wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
                    mpc, prev_position, prev_tool_position, seq_pos[:, wp, :], seq_vel[:, wp, :], wp_dt,
                )
                prev_accel = send_smoothed_segment(bridge, prev_position, prev_velocity, prev_accel, wp_position, wp_velocity, wp_dt)
                prev_position = wp_position
                prev_velocity = wp_velocity
                viser_viz.set_joint_state(JointState.from_position(wp_position, joint_names=mpc.joint_names).squeeze(0))

            believed_state = JointState.from_position(prev_position, joint_names=mpc.joint_names)
            believed_state.velocity = prev_velocity
            believed_state.acceleration = torch.zeros_like(prev_position)
        else:
            time.sleep(control_dt)


# --- Mode: Trajectory / Trajectory+MPC -----------------------------------------
def _execute_trajectory(bridge, planner, trajectory, gripper_pinch_fn, interrupt_check=None):
    """Plays back a MotionPlanner trajectory with the Cartesian-cap +
    jerk/accel-limited safety chain, gripper driven live by pinch instead of
    held fixed. Returns the JointState execution actually ended at.
    """
    traj = trajectory.squeeze(0)
    n_steps = traj.position.shape[-2]
    interp_dt = traj.dt.item() if traj.dt is not None else 0.02

    prev_position = traj.position[0, 0, :].unsqueeze(0)
    prev_velocity = torch.zeros_like(prev_position)
    prev_accel = torch.zeros_like(prev_position)
    prev_tool_position = _tool_position(planner, prev_position)

    for i in range(n_steps):
        if interrupt_check is not None and interrupt_check():
            break
        pos = traj.position[0, i, :].unsqueeze(0).clone()
        vel = traj.velocity[0, i, :].unsqueeze(0) if traj.velocity is not None else torch.zeros_like(pos)
        gripper_pos = gripper_pinch_fn()
        if gripper_pos is not None:
            pos[:, GRIPPER_JOINT_IDX] = gripper_pos
        wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
            planner, prev_position, prev_tool_position, pos, vel, interp_dt,
        )
        prev_accel = send_smoothed_segment(bridge, prev_position, prev_velocity, prev_accel, wp_position, wp_velocity, interp_dt)
        prev_position = wp_position
        prev_velocity = wp_velocity

    final_state = JointState.from_position(prev_position, joint_names=traj.joint_names)
    final_state.velocity = prev_velocity
    return final_state


def run_trajectory_teleop(
    hand_tracker: OpenXrHandTracker, calibrator: WorkspaceCalibrator, bridge: RebotArmBridge,
    viser_viz: ViserVisualizer, current_state: JointState, robot_neutral_orientation_wxyz: np.ndarray,
    enable_mpc_fallback: bool, control_dt: float = 0.03,
):
    config = MotionPlannerCfg.create(robot=ROBOT_YML, scene_model="collision_test.yml", max_goalset=10)
    planner = MotionPlanner(config)
    tool_frame = planner.tool_frames[0]
    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    mpc = None
    if enable_mpc_fallback:
        mpc_config = ModelPredictiveControlCfg.create(
            robot=ROBOT_YML, scene_model="collision_test.yml", use_cuda_graph=True,
            optimization_dt=control_dt, interpolation_steps=4, optimizer_collision_activation_distance=0.03,
        )
        mpc = ModelPredictiveControl(mpc_config)
        mpc.setup(current_state)
        kin_result = mpc.compute_kinematics(current_state)
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(kin_result.tool_poses.to_dict(), ordered_tool_frames=mpc.tool_frames, num_goalset=1),
            run_ik=False,
        )
        mpc.optimize_action_sequence(current_state)  # warm up its CUDA graph too, before the background thread starts

    bridge.start_command_thread()
    tracker = TargetPoseTracker(hand_tracker, calibrator, tool_frame, robot_neutral_orientation_wxyz)
    print(f"\n[{'trajectory_mpc' if enable_mpc_fallback else 'trajectory'} mode] tracking left hand -> {tool_frame}. Press Ctrl+C to exit.\n")

    def try_plan(target_poses, start_state, max_attempts=3):
        active_js = planner.kinematics.get_active_js(start_state.clone())
        return planner.plan_pose(
            GoalToolPose.from_poses(target_poses, num_goalset=1), active_js,
            use_implicit_goal=True, max_attempts=max_attempts,
        )

    last_target_poses = None
    home_position = planner.default_joint_state.position.clone().unsqueeze(0)

    while True:
        target_poses = tracker.get_target()
        if target_poses is None:
            time.sleep(control_dt)
            continue
        changed = last_target_poses is None or any(
            target_poses[k] != last_target_poses[k] for k in target_poses
        )
        if not changed:
            time.sleep(control_dt)
            continue
        last_target_poses = {k: v.clone() for k, v in target_poses.items()}

        result = try_plan(target_poses, current_state, max_attempts=3)
        if result is None or not result.success.any():
            home_state = JointState.from_position(home_position, joint_names=planner.joint_names)
            result = try_plan(target_poses, home_state, max_attempts=3)
            if result is not None and result.success.any():
                # result was already planned starting FROM home (line above)
                # -- physically move there first, then execute that same
                # plan below.
                print("  [trajectory] current pose can't reach target -- returning home first...")
                home_to_pose(bridge, home_position)
                current_state = JointState.from_position(home_position, joint_names=planner.joint_names)

        if result is not None and result.success.any():
            interp = result.get_interpolated_plan()

            def target_changed_since_last() -> bool:
                latest = tracker.get_target()
                return latest is not None and any(latest[k] != last_target_poses[k] for k in last_target_poses)

            current_state = _execute_trajectory(
                bridge, planner, interp, tracker.get_gripper_position, interrupt_check=target_changed_since_last,
            )
        elif mpc is not None:
            print("  [trajectory] target unreachable -- falling back to MPC (best-effort tracking)...")
            current_state = _mpc_fallback_until_replannable(
                tracker, mpc, bridge, viser_viz, current_state, try_plan, control_dt,
            )
        else:
            print("  [trajectory] target unreachable, and MPC fallback is disabled -- waiting for a reachable target.")

        viser_viz.set_joint_state(current_state.squeeze(0))


def _mpc_fallback_until_replannable(
    tracker: TargetPoseTracker, mpc, bridge, viser_viz, current_state, try_plan, control_dt, replan_check_every=10,
):
    """Continuous MPC tracking (replan-from-belief) until the live target
    becomes solvable by a full plan again, then returns so the caller can
    switch back.
    """
    believed_state = current_state
    prev_accel = torch.zeros_like(current_state.position)
    tick = 0

    while True:
        target_poses = tracker.get_target()
        if target_poses is None:
            time.sleep(control_dt)
            continue

        if tick % replan_check_every == 0:
            plan_result = try_plan(target_poses, believed_state, max_attempts=1)
            if plan_result is not None and plan_result.success.any():
                print("  [trajectory] target now solvable -- switching back to a generated plan.")
                interp = plan_result.get_interpolated_plan()
                return _execute_trajectory(bridge, mpc, interp, tracker.get_gripper_position)
        tick += 1

        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(target_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1), run_ik=False,
        )
        result = mpc.optimize_action_sequence(believed_state)
        has_action = result.action_sequence is not None and result.action_sequence.position.shape[1] > 0
        if not has_action:
            time.sleep(control_dt)
            continue

        gripper_pos = tracker.get_gripper_position()
        seq_pos = result.action_sequence.position.clone()
        seq_vel = result.action_sequence.velocity.clone()
        if gripper_pos is not None:
            seq_pos[:, :, GRIPPER_JOINT_IDX] = gripper_pos
        seq_vel[:, :, GRIPPER_JOINT_IDX] = 0.0
        n_wp = seq_pos.shape[1]
        wp_dt = control_dt / n_wp

        prev_position = believed_state.position
        prev_velocity = believed_state.velocity
        prev_tool_position = _tool_position(mpc, prev_position)
        for wp in range(n_wp):
            wp_position, wp_velocity, prev_tool_position = cap_cartesian_step(
                mpc, prev_position, prev_tool_position, seq_pos[:, wp, :], seq_vel[:, wp, :], wp_dt,
            )
            prev_accel = send_smoothed_segment(bridge, prev_position, prev_velocity, prev_accel, wp_position, wp_velocity, wp_dt)
            prev_position = wp_position
            prev_velocity = wp_velocity
            viser_viz.set_joint_state(JointState.from_position(wp_position, joint_names=mpc.joint_names).squeeze(0))

        believed_state = JointState.from_position(prev_position, joint_names=mpc.joint_names)
        believed_state.velocity = prev_velocity
        believed_state.acceleration = torch.zeros_like(prev_position)


# --- Setup + CLI ---------------------------------------------------------------
def run_teleop(
    mode: str, dry_run: bool, port: str, baud: int, viz_port: int, observe_only: bool,
    scene_model: str = "collision_test.yml",
):
    print("Connecting to headset via WiVRn/OpenXR (left-hand tracking)...")
    hand_tracker = OpenXrHandTracker()
    hand_tracker.start()
    print("OpenXR session ready.\n")

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT_YML),
        connect_ip="0.0.0.0", connect_port=viz_port,
        add_control_frames=False, visualize_robot_spheres=False, add_robot_to_scene=True,
    )

    # A throwaway MotionPlanner just to get joint_names/kinematics/default
    # pose consistently regardless of --mode, and for FK sampling in the
    # calibrator -- cheap relative to the mode-specific solver built below.
    setup_config = MotionPlannerCfg.create(robot=ROBOT_YML, scene_model=scene_model)
    setup_planner = MotionPlanner(setup_config)
    joint_names = setup_planner.joint_names
    default_position = setup_planner.default_joint_state.position.clone().unsqueeze(0)

    bridge = RebotArmBridge(
        joint_names=joint_names, motor_map=MOTOR_MAP, port=port, baud=baud,
        dry_run=dry_run, default_position=default_position, observe_only=observe_only,
    )

    try:
        if not observe_only:
            home_to_pose(bridge, default_position)
        real_state = bridge.read_joint_state()
        bridge.set_target(real_state.position, torch.zeros_like(real_state.position))
        current_state = JointState.from_position(real_state.position.clone(), joint_names=joint_names)
        current_state.velocity = torch.zeros_like(current_state.position)

        lower, upper = _cspace_joint_limits()
        calibrator = WorkspaceCalibrator(setup_planner, lower, upper)
        calibrator.calibrate_hand_range(hand_tracker)

        # Move to the workspace's geometric center before handing off
        # control, so "hand at calibrated neutral" starts out matching
        # "arm at its own workspace center". Target orientation is the
        # CURRENT end-effector orientation (unchanged) -- this move should
        # only reposition the arm, not force an arbitrary orientation change.
        tool_frame = setup_planner.tool_frames[0]
        current_orientation_wxyz = (
            setup_planner.compute_kinematics(current_state).tool_poses.to_dict()[tool_frame]
            .quaternion.cpu().squeeze(0).numpy()
        )
        center_pose = Pose.from_numpy(calibrator.robot_center.astype(np.float32), current_orientation_wxyz.astype(np.float32))
        center_result = setup_planner.plan_pose(
            GoalToolPose.from_poses({tool_frame: center_pose}, num_goalset=1),
            setup_planner.kinematics.get_active_js(current_state.clone()),
            use_implicit_goal=True, max_attempts=5,
        )
        robot_neutral_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        if center_result is not None and center_result.success.any():
            interp = center_result.get_interpolated_plan()
            final_pos = interp.position[0, 0, -1, :].unsqueeze(0)
            home_to_pose(bridge, final_pos)
            current_state = JointState.from_position(final_pos, joint_names=joint_names)
            current_state.velocity = torch.zeros_like(current_state.position)
            kin = setup_planner.compute_kinematics(current_state)
            robot_neutral_orientation_wxyz = kin.tool_poses.to_dict()[tool_frame].quaternion.cpu().squeeze(0).numpy()
        else:
            print("  [warn] couldn't plan to workspace center -- starting from the current/home pose instead.")
            kin = setup_planner.compute_kinematics(current_state)
            robot_neutral_orientation_wxyz = kin.tool_poses.to_dict()[tool_frame].quaternion.cpu().squeeze(0).numpy()

        viser_viz.set_joint_state(current_state.squeeze(0))

        if mode == "ik":
            run_ik_teleop(hand_tracker, calibrator, bridge, viser_viz, current_state, robot_neutral_orientation_wxyz)
        elif mode == "mpc":
            run_mpc_teleop(hand_tracker, calibrator, bridge, viser_viz, current_state, robot_neutral_orientation_wxyz)
        else:
            run_trajectory_teleop(
                hand_tracker, calibrator, bridge, viser_viz, current_state, robot_neutral_orientation_wxyz,
                enable_mpc_fallback=(mode == "trajectory_mpc"),
            )
    finally:
        hand_tracker.stop()
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Meta Quest 3 left-hand teleop for the rebot arm")
    parser.add_argument(
        "--mode", choices=["ik", "mpc", "trajectory", "trajectory_mpc"], default="ik",
        help="Control backend. 'ik': one-shot IK per target change (simplest, most proven -- "
        "recommended first). 'mpc': continuous MPC tracking. 'trajectory': motion planning, "
        "replans on target change, no fallback for unreachable targets. 'trajectory_mpc': same "
        "plus continuous MPC tracking as a fallback when the target isn't fully reachable.",
    )
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
        confirm = input("About to command REAL motors on /dev of your choosing. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)

    run_teleop(
        mode=args.mode, dry_run=not args.live, port=args.port, baud=args.baud,
        viz_port=args.viz_port, observe_only=args.observe_only,
    )


if __name__ == "__main__":
    main()
