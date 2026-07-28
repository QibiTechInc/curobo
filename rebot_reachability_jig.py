# SPDX-License-Identifier: Apache-2.0
"""Reachability analysis + mop-jig mounting-angle optimization for the reBot B601-DM arm.

The arm's gripper is being replaced with a rigid mop-cleaning jig: a
0.35-0.48m extendable rod ending in a passive-pitch joint (self-aligns flat
to a table under contact force) and a 20x20cm cylindrical mop head. Before
fabricating the mount, this answers: **at what roll/pitch/yaw angle should
the jig be bolted to link6** to maximize how much of a 1m x 1m table is both
reachable and pressable with >=10N of sustained downward force without
exceeding any joint's continuous torque rating.

Per user decisions: only the mounting ANGLE is searched (rod length and
robot/table height are fixed at nominal values); the search uses Optuna in
15-degree steps (a manufacturing constraint -- the mount can't be adjusted
more finely than that); only IK + hardware constraints are checked, not a
full dynamic/collision-avoidance motion plan.

Key architecture point (this was the main design constraint going in): a
naive Optuna loop that reloads the URDF and rebuilds collision spheres per
candidate angle would be far too slow to search hundreds of candidates.
Instead, the solver (and its self-collision spheres) is built ONCE with a
virtual `mop_tip` link attached to `link6` (cuRobo's `extra_links`/
`extra_collision_spheres`, the same mechanism `franka.yml` uses for
`attached_object`), and each candidate angle just overwrites that link's
fixed-transform tensor **in place** (`kinematics_config.fixed_transforms`) --
no URDF re-parse, no sphere regeneration, no solver rebuild. See
`set_mop_mount_angle()`.

A second design note worth recording: cuRobo's `terminal_pose_axes_weight_factor`
(relaxing pitch/yaw of the IK goal so the passive joint + the mop cylinder's
own rotational symmetry don't need to be matched) was tried first and found
to have a real bug for large deviations -- the axis-angle error's shared
quaternion scalar component isn't masked by the per-axis weight, so a large
rotation on a "relaxed" axis still contaminates the orientation-convergence
check (verified empirically). Instead, this script explicitly tries a coarse
discrete grid of heading/roll/tilt combinations per table point and calls
the point reachable if ANY combination succcaeeds and passes the torque check
-- more batch size, but not subject to that bug.

Usage:

.. code-block:: bash

   python rebot_reachability_jig.py --n-trials 60
   python rebot_reachability_jig.py --n-trials 5   # quick smoke test
   python rebot_reachability_jig.py --visualize     # after a search, view the result in Viser
"""

from __future__ import annotations

import argparse
import copy
import os
import threading
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pinocchio as pin
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import ContentPath, GoalToolPose, Pose

from rebot_mpc_control import REBOT_URDF_PATH, compute_gravity_torque

REBOT_YML_PATH = "/home/kendemu/curobo/rebot_b601_gripper.yml"
OUTPUT_YML_PATH = "/home/kendemu/curobo/rebot_b601_mop.yml"
OUTPUT_URDF_SNIPPET_PATH = "/home/kendemu/curobo/mop_jig_mount.urdf.xacro"
# write_merged_urdf() generates a NEW, uniquely-named file here each run
# (timestamped) rather than overwriting a fixed path, so older optimization
# runs' URDFs stay around for comparison/reference.
MERGED_URDF_DIR = "/home/kendemu/curobo"
REACHABILITY_DATA_PATH = "/home/kendemu/curobo/rebot_mop_reachability.npz"

# Locking gripper_joint1 at exactly its URDF lower limit (0.0) was found to
# make every single IK solve report infeasible regardless of pose -- a
# joint-limit boundary edge case, confirmed independent of self-collision
# (still happened with self_collision_check=False) and independent of
# position/orientation error (both were ~0). Locking at the yml's own
# default position sidesteps it entirely and was confirmed to work.
GRIPPER_LOCK_POSITION = 0.03575

# Rod length fixed at the midpoint of the 0.35-0.48m extendable range for
# this pass -- per user decision, only the mounting ANGLE is being searched.
ROD_LENGTH_M = 0.415
MOP_CYLINDER_DIAMETER_M = 0.20
MOP_CYLINDER_LENGTH_M = 0.05  # max 5cm height along the cylinder's own axis, per user

# Real Damiao torque ratings, from the user (matches the URDF's <limit
# effort=...> exactly): DM4340P (joints1-3) nominal=9 N*m/peak=27 N*m,
# DM4310 (joints4-6) nominal=3 N*m/peak=7 N*m. Nominal (continuous) is used
# as the pass/fail limit since cleaning is a sustained, quasi-static load,
# not a brief motion -- peak is reported as a secondary ceiling only.
JOINT_TORQUE_NOMINAL_NM = np.array([9.0, 9.0, 9.0, 3.0, 3.0, 3.0])
JOINT_TORQUE_PEAK_NM = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])

CLEANING_FORCE_N = 10.0  # required sustained downward push at the mop tip.

# Table workspace. Per the user's own simplification, the robot base is
# assumed mounted at table height, so the table surface is Z=0 in
# robot-base frame. TABLE_ORIGIN_XY is the offset from the robot base to the
# table's near corner in robot-base-frame X/Y: robot base sits right at the
# table's edge (X in [0,1], Y in [-1,1], ROS2/REP-103 convention -- X
# forward, Y left, Z up). Y widened to 2m per user request (was 1m/square).
TABLE_SIZE_X_M = 1.0
TABLE_SIZE_Y_M = 2.0
TABLE_ORIGIN_XY = (0.0, -1.0)  # table spans X in [0, 1], Y in [-1, 1], Z=0
TABLE_GRID_NX = 8   # ~0.143m spacing over 1m
TABLE_GRID_NY = 15  # same ~0.143m spacing over 2m

# Discretized orientation sweep per table point -- see module docstring for
# why this replaces cost-weight-based orientation relaxation. The mop head
# is a disc (20cm diameter, 5cm tall) whose FLAT CIRCULAR FACE touches the
# table -- its own axis of rotational symmetry is the same as the
# approach/pressing direction (local Z, see _R_CANON), confirmed by the
# user after an earlier wrong assumption (that the mop was a roller with
# its curved SIDE touching, axis horizontal/perpendicular to the approach
# direction -- see rebot_reachability_jig memory notes for that mixup).
# ROLL is rotation about that shared axis (fully free -- the disc is
# rotationally symmetric about it -- sampled coarsely since only our SEARCH
# is discretized, not the real hardware). TILT is the approach-direction
# tilt the passive pitch joint is assumed able to self-align through -- an
# ASSUMPTION; narrow or widen this if the real joint's mechanical range is
# known to differ.
HEADINGS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
ROLL_DEG = [0.0, 90.0, 180.0, 270.0]
# 0.0 listed first: build_orientation_candidates() enumerates tilt in this
# order, and visualize() picks the FIRST candidate that solves -- putting
# the canonical (untilted) value first means the demo pose shown defaults
# to "pointing straight down" instead of an arbitrarily large tilt. The
# search itself no longer depends on this ordering either way:
# find_feasible_solutions() explicitly picks the MINIMUM-|tilt| feasible
# candidate per table point (see REACHABILITY_QUALITY_WEIGHT), not just the
# first one found.
TILT_DEG = [0.0, -15.0, 15.0, -30.0, 30.0]

# Manufacturing constraint: the jig mount can only be set in 15-degree
# increments, so the search space itself is quantized to match (not just the
# reported result).
JIG_ANGLE_RANGE_DEG = (-180.0, 180.0)
JIG_ANGLE_STEP_DEG = 15.0

# The search objective balances two things: how MUCH of the table is
# reachable (area_score) and how close to vertical/untilted the reachable
# points are (quality_score) -- a point that's only reachable at a heavy
# TILT_DEG value still gets counted as feasible, but relies on the passive
# pitch hinge (mop_tip_joint) taking up the slack, which saturates its real
# +-75 deg mechanical range and drives large IK/MPC joint-configuration
# swings between neighboring, differently-tilted table points (confirmed
# empirically: the r-15/p+75/y+90 angle found by an area-only search had the
# passive joint pinned at its limit on ~65% of a cleaning sweep). Weighting
# in quality_score biases the search toward mount angles where MOST
# reachable points need little or no tilt, not just toward raw area.
REACHABILITY_AREA_WEIGHT = 0.5
REACHABILITY_QUALITY_WEIGHT = 0.5

# local Z = -world Z (mop tip points straight down at the table, and the
# mop disc's own symmetry axis, since its flat face is what touches).
_R_CANON = Rotation.from_matrix(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float))


def _rod_transform(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[np.ndarray, Rotation]:
    """Fixed transform from link6 to mop_tip for a given jig mounting angle.

    The rod's DIRECTION is determined by pitch+yaw only. roll is applied as
    a separate spin about the rod's own resulting axis, composed LAST, so it
    changes how the mop head is "clocked" without moving where the rod
    points. (An earlier version applied all three as a single
    from_euler("xyz", [roll, pitch, yaw]) -- since that composes roll
    FIRST/innermost, it actually changed the rod's pointing direction too,
    which looked wrong when rendered and confused the "roll should be a free
    spin" mental model. Verified numerically: with this composition, the rod
    position is bit-for-bit identical across all roll values for fixed
    pitch/yaw; only the orientation changes.)
    """
    direction_rot = Rotation.from_euler("xyz", [0.0, pitch_deg, yaw_deg], degrees=True)
    spin = Rotation.from_euler("z", roll_deg, degrees=True)
    rot = direction_rot * spin
    pos = rot.apply([0.0, 0.0, ROD_LENGTH_M])
    return pos, rot


def _rod_collision_spheres_mount_frame() -> List[Dict]:
    """Rod spheres in mop_MOUNT's own local frame: z=0 at the link6 joint,
    z=+ROD_LENGTH_M at the tip end -- matches the merged URDF's
    mop_mount_joint/mop_tip_joint structure (see write_merged_urdf).
    """
    n_rod = 6
    rod_radius = 0.015
    return [
        {"center": [0.0, 0.0, ROD_LENGTH_M * i / (n_rod - 1)], "radius": rod_radius}
        for i in range(n_rod)
    ]


def _cylinder_collision_spheres_tip_frame() -> List[Dict]:
    """Mop-head disc spheres in mop_TIP's own local frame: the disc's flat
    circular face (20cm diameter, 5cm tall) is what touches the table, so
    its axis of symmetry is local Z (see _R_CANON) -- spheres are spread
    across the local X/Y plane (a center sphere + a ring), not along Z,
    approximating a short wide disc rather than a long rod. (An earlier
    version spread these along local X, based on a wrong "roller with its
    curved side touching" assumption -- corrected by the user; see
    rebot_reachability_jig memory notes.)
    """
    disc_radius = MOP_CYLINDER_DIAMETER_M / 2.0
    half_height = MOP_CYLINDER_LENGTH_M / 2.0
    ring_r = disc_radius - half_height  # keep ring spheres' extent within the true disc radius
    spheres = [{"center": [0.0, 0.0, 0.0], "radius": half_height}]
    n_ring = 4
    for i in range(n_ring):
        theta = 2 * np.pi * i / n_ring
        spheres.append({
            "center": [float(ring_r * np.cos(theta)), float(ring_r * np.sin(theta)), 0.0],
            "radius": half_height,
        })
    return spheres


def _mop_collision_spheres() -> List[Dict]:
    """Closed-form spheres approximating the rod + mop cylinder, COMBINED
    into mop_tip's OWN local frame -- used by the single-link extra_links
    approach (build_base_robot_cfg/set_mop_mount_angle, the fast search
    path). Fixed regardless of the jig mounting angle (only the
    link6->mop_tip transform changes per candidate, not this local layout),
    and only needs regenerating if ROD_LENGTH_M or the cylinder dimensions
    change (neither varies in this pass).

    Convention: mop_tip's local +Z points back along the rod toward the
    mount -- the SAME axis the disc-shaped mop head is symmetric about (its
    flat face touches the table perpendicular to this axis, see _R_CANON),
    matching a real round mop/buffing pad with its handle rising straight
    up from the disc's center. Rod spheres are just
    _rod_collision_spheres_mount_frame() re-expressed in mop_tip's frame
    (mount-frame z=0..ROD_LENGTH_M maps to tip-frame z=-ROD_LENGTH_M..0).
    """
    rod_in_tip_frame = [
        {"center": [0.0, 0.0, c["center"][2] - ROD_LENGTH_M], "radius": c["radius"]}
        for c in _rod_collision_spheres_mount_frame()
    ]
    return rod_in_tip_frame + _cylinder_collision_spheres_tip_frame()


def build_base_robot_cfg() -> dict:
    """Robot config dict: gripper_joint1 locked AND the gripper's own links
    (gripper_link/gripper_left/gripper_right) stripped from the collision
    model entirely -- the real hardware has the gripper physically removed,
    replaced by the mop jig, so there's no reason to keep checking/rendering
    its (now-fictional) geometry. mop_tip is attached to link6 as a virtual
    tool frame via cuRobo's extra_links/extra_collision_spheres (no URDF
    edits needed).
    """
    with open(REBOT_YML_PATH) as f:
        base = yaml.safe_load(f)
    cfg = copy.deepcopy(base)
    kin = cfg["kinematics"]
    kin["lock_joints"] = {"gripper_joint1": GRIPPER_LOCK_POSITION}

    gripper_links = {"gripper_link", "gripper_left", "gripper_right"}
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n not in gripper_links]
    kin["mesh_link_names"] = [n for n in kin["mesh_link_names"] if n not in gripper_links]
    for name in gripper_links:
        kin["collision_spheres"].pop(name, None)
        kin["self_collision_buffer"].pop(name, None)
        kin["self_collision_ignore"].pop(name, None)
    for link_name, ignore_list in kin["self_collision_ignore"].items():
        kin["self_collision_ignore"][link_name] = [n for n in ignore_list if n not in gripper_links]

    pos0, rot0 = _rod_transform(0.0, 0.0, 0.0)
    quat0 = rot0.as_quat()  # xyzw
    kin["extra_links"] = {
        "mop_tip": {
            "fixed_transform": [
                float(pos0[0]), float(pos0[1]), float(pos0[2]),
                float(quat0[3]), float(quat0[0]), float(quat0[1]), float(quat0[2]),
            ],
            "joint_name": "mop_tip_joint",
            "joint_type": "FIXED",
            "link_name": "mop_tip",
            "parent_link_name": "link6",
        },
    }
    spheres = _mop_collision_spheres()
    kin["extra_collision_spheres"] = {"mop_tip": len(spheres)}
    kin["collision_spheres"]["mop_tip"] = spheres
    kin["collision_link_names"] = list(kin["collision_link_names"]) + ["mop_tip"]
    kin["self_collision_buffer"]["mop_tip"] = 0.0
    kin["self_collision_ignore"]["mop_tip"] = ["link6", "link5"]
    kin["self_collision_ignore"]["link6"] = list(
        set(kin["self_collision_ignore"].get("link6", [])) | {"mop_tip"}
    )
    kin["tool_frames"] = ["mop_tip"]
    return cfg


def build_ik_solver(max_batch_size: int, num_seeds: int = 8) -> InverseKinematics:
    cfg = build_base_robot_cfg()
    ik_cfg = InverseKinematicsCfg.create(
        robot=cfg,
        num_seeds=num_seeds,
        self_collision_check=True,
        max_batch_size=max_batch_size,
    )
    return InverseKinematics(ik_cfg)


def build_pin_model() -> Tuple[pin.Model, int]:
    """Separate pinocchio model (own instance, not shared with
    rebot_mpc_control.py's cached one) with a `mop_tip` OP_FRAME added at
    link6 -- used only for the Jacobian in the torque check; gravity torque
    itself reuses rebot_mpc_control.compute_gravity_torque directly (proven,
    already handles the gripper_joint2 mimic mapping).
    """
    model = pin.buildModelFromUrdf(REBOT_URDF_PATH)
    link6_frame_id = model.getFrameId("link6")
    pos0, rot0 = _rod_transform(0.0, 0.0, 0.0)
    placement = pin.SE3(rot0.as_matrix(), np.asarray(pos0))
    frame = pin.Frame(
        "mop_tip", model.frames[link6_frame_id].parentJoint, link6_frame_id,
        placement, pin.FrameType.OP_FRAME,
    )
    mop_tip_id = model.addFrame(frame)
    return model, mop_tip_id


def set_mop_mount_angle(
    ik: InverseKinematics, pin_model: pin.Model, mop_tip_pin_id: int,
    roll_deg: float, pitch_deg: float, yaw_deg: float,
) -> None:
    """Overwrites the link6->mop_tip transform IN PLACE on both the already-
    built cuRobo solver and the already-built pinocchio model -- no solver
    rebuild, no URDF re-parse, no collision-sphere regeneration. This is the
    core mechanism that makes searching hundreds of candidate angles fast.
    """
    pos, rot = _rod_transform(roll_deg, pitch_deg, yaw_deg)
    rot_mat = rot.as_matrix()

    kc = ik.kinematics.config.kinematics_config
    idx = kc.link_name_to_idx_map["mop_tip"]
    dtype = kc.fixed_transforms.dtype
    device = kc.fixed_transforms.device
    kc.fixed_transforms[idx, :3, :3] = torch.tensor(rot_mat, dtype=dtype, device=device)
    kc.fixed_transforms[idx, :3, 3] = torch.tensor(pos, dtype=dtype, device=device)

    pin_model.frames[mop_tip_pin_id].placement = pin.SE3(rot_mat, np.asarray(pos))


def build_table_grid() -> np.ndarray:
    xs = np.linspace(TABLE_ORIGIN_XY[0], TABLE_ORIGIN_XY[0] + TABLE_SIZE_X_M, TABLE_GRID_NX)
    ys = np.linspace(TABLE_ORIGIN_XY[1], TABLE_ORIGIN_XY[1] + TABLE_SIZE_Y_M, TABLE_GRID_NY)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # xx/yy shape (TABLE_GRID_NY, TABLE_GRID_NX)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=-1).astype(np.float32)


def largest_reachable_rectangle(feasible_grid: np.ndarray) -> Tuple[int, int, int, int]:
    """Largest axis-aligned all-feasible rectangle in the 2D feasibility
    grid, via the standard largest-rectangle-in-histogram method
    (O(rows*cols)). Returns inclusive grid-index bounds (row0, row1, col0,
    col1) -- rows index Y, columns index X, matching build_table_grid's
    meshgrid(..., indexing="xy") layout.

    This is a coarse approximation: it only certifies the SAMPLED grid
    points inside the rectangle are feasible, not every point in between
    (gaps between the TABLE_GRID_NX x TABLE_GRID_NY samples aren't checked).
    """
    n_rows, n_cols = feasible_grid.shape
    heights = np.zeros(n_cols, dtype=int)
    best_area = 0
    best = (0, 0, 0, 0)
    for r in range(n_rows):
        heights = np.where(feasible_grid[r], heights + 1, 0)
        stack: List[Tuple[int, int]] = []
        for c in range(n_cols + 1):
            h = int(heights[c]) if c < n_cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best_area:
                    best_area = area
                    best = (r - sh + 1, r, s, c - 1)
                start = s
            stack.append((start, h))
    return best


def reachable_rectangle_bounds_m(feasible_grid: np.ndarray) -> Dict[str, float]:
    """Converts largest_reachable_rectangle's grid-index bounds to a
    world-frame X/Y rectangle (meters), using the table's actual grid
    spacing.
    """
    row0, row1, col0, col1 = largest_reachable_rectangle(feasible_grid)
    dx = TABLE_SIZE_X_M / (TABLE_GRID_NX - 1)
    dy = TABLE_SIZE_Y_M / (TABLE_GRID_NY - 1)
    x_min = TABLE_ORIGIN_XY[0] + col0 * dx
    x_max = TABLE_ORIGIN_XY[0] + col1 * dx
    y_min = TABLE_ORIGIN_XY[1] + row0 * dy
    y_max = TABLE_ORIGIN_XY[1] + row1 * dy
    return {
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "width_x_m": x_max - x_min, "width_y_m": y_max - y_min,
        "area_m2": (x_max - x_min) * (y_max - y_min),
    }


def largest_reachable_half_circle(table_pts: np.ndarray, feasible: np.ndarray) -> Dict[str, float]:
    """Largest half-circle (semicircle) approximation of the reachable area,
    flat edge along the robot's mounting edge (X = TABLE_ORIGIN_XY[0]),
    bulging into the table (+X) -- complements the rectangle approximation
    with a shape that better matches how a single-base-point arm's
    workspace actually looks (roughly circular, centered somewhere along
    the mount edge), rather than an axis-aligned box.

    For each candidate center (edge_x, cy), the largest valid radius is
    exactly the distance to the NEAREST infeasible sampled grid point (any
    smaller radius excludes it, any larger one includes it) -- capped so
    the semicircle doesn't extend past the table's own physical bounds
    (there's no data, and no table, out there). Same grid-sampling caveat
    as largest_reachable_rectangle: only certifies the SAMPLED points
    inside it are feasible, not the full continuum.
    """
    edge_x = TABLE_ORIGIN_XY[0]
    y_min_table = TABLE_ORIGIN_XY[1]
    y_max_table = TABLE_ORIGIN_XY[1] + TABLE_SIZE_Y_M

    xs = table_pts[:, 0] - edge_x
    ys = table_pts[:, 1]
    feasible = feasible.reshape(-1)

    cy_candidates = np.linspace(y_min_table, y_max_table, 61)
    best_area = 0.0
    best = {"center_y": 0.0, "radius": 0.0}
    for cy in cy_candidates:
        dist = np.sqrt(xs ** 2 + (ys - cy) ** 2)
        infeasible_dist = dist[~feasible]
        r_data = float(infeasible_dist.min()) if infeasible_dist.size > 0 else float(dist.max())
        r_bounds = min(TABLE_SIZE_X_M, cy - y_min_table, y_max_table - cy)
        r = max(0.0, min(r_data, r_bounds))
        area = 0.5 * np.pi * r ** 2
        if area > best_area:
            best_area = area
            best = {"center_y": float(cy), "radius": float(r)}
    best["center_x"] = edge_x
    best["area_m2"] = best_area
    return best


def build_orientation_candidates() -> np.ndarray:
    """Returns (n_headings*n_roll*n_tilt, 4) wxyz quaternions.

    roll rotates about local Z (the mop disc's own symmetry/approach axis,
    see _R_CANON) -- applied innermost, BEFORE _R_CANON, specifically
    because a Z-rotation commutes with _R_CANON's own Z-mapping (rotating
    about Z first doesn't change where [0,0,1] ends up after _R_CANON is
    applied on top), so it's a genuinely free spin that doesn't perturb the
    approach direction. (An earlier version rotated about local X here --
    same bug as _rod_transform's original roll coupling: rotating about an
    axis OTHER than the one that ends up being the symmetry axis doesn't
    stay "free", it redirects the touching direction too.)
    """
    quats = []
    for heading in HEADINGS_DEG:
        for tilt in TILT_DEG:
            for roll in ROLL_DEG:
                R = (
                    Rotation.from_euler("z", heading, degrees=True)
                    * Rotation.from_euler("y", tilt, degrees=True)
                    * _R_CANON
                    * Rotation.from_euler("z", roll, degrees=True)
                )
                q = R.as_quat()  # xyzw
                quats.append([q[3], q[0], q[1], q[2]])
    return np.array(quats, dtype=np.float32)


def _orientation_tilt_degrees() -> np.ndarray:
    """abs(tilt) in degrees for each orientation candidate, in the same
    heading/tilt/roll nested order as build_orientation_candidates() --
    lets find_feasible_solutions() score how far off-vertical each
    candidate is.
    """
    tilts = []
    for _heading in HEADINGS_DEG:
        for tilt in TILT_DEG:
            for _roll in ROLL_DEG:
                tilts.append(abs(tilt))
    return np.array(tilts, dtype=np.float32)


class JigAngleEvaluator:
    """Owns the one-time-built IK solver + pinocchio model and evaluates
    candidate jig mounting angles via in-place mutation (see
    set_mop_mount_angle).

    The full per-trial batch (table points x orientation candidates) is much
    too large to solve in one shot -- self-collision cost buffers alone blew
    past 15GB of GPU memory at batch=10240/seeds=8 (confirmed empirically).
    Instead the solver is built with a much smaller `CHUNK_SIZE` batch and
    `_solve_all()` loops `solve_pose` over chunks, concatenating results --
    this still only builds the solver ONCE (no rebuild between chunks or
    between trials), it just calls the same batched solve multiple times.
    """

    CHUNK_SIZE = 1024  # confirmed safe (~3GB peak) at num_seeds=8 on a 16GB GPU.

    def __init__(self):
        self.table_pts = build_table_grid()
        self.orientations = build_orientation_candidates()
        self.orientation_tilt_deg = _orientation_tilt_degrees()
        self.max_tilt_deg = float(max(abs(t) for t in TILT_DEG))
        self.n_points = self.table_pts.shape[0]
        self.n_orient = self.orientations.shape[0]
        total = self.n_points * self.n_orient
        print(
            f"Building IK solver (chunk size {self.CHUNK_SIZE}, {total} total targets = "
            f"{self.n_points} table points x {self.n_orient} orientation candidates)..."
        )
        self.ik = build_ik_solver(max_batch_size=self.CHUNK_SIZE)
        self.pin_model, self.mop_tip_pin_id = build_pin_model()
        self.pin_data = self.pin_model.createData()
        self.tool = self.ik.kinematics.tool_frames[0]
        self.joint_names = self.ik.kinematics.joint_names  # ["joint1".."joint6"]

        device = self.ik.kinematics.config.kinematics_config.fixed_transforms.device
        positions = np.repeat(self.table_pts, self.n_orient, axis=0)
        quats = np.tile(self.orientations, (self.n_points, 1))
        self._goal_pos = torch.tensor(positions, dtype=torch.float32, device=device)
        self._goal_quat = torch.tensor(quats, dtype=torch.float32, device=device)

    def _solve_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Runs solve_pose over the full target set in CHUNK_SIZE-sized
        chunks (padding + truncating the final partial chunk as needed).

        Returns:
            success: (n_points, n_orient) bool
            solution: (n_points, n_orient, 6) joint1-6 positions
        """
        total = self.n_points * self.n_orient
        chunk = self.CHUNK_SIZE
        success_parts = []
        solution_parts = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            pos_chunk = self._goal_pos[start:end]
            quat_chunk = self._goal_quat[start:end]
            pad = chunk - pos_chunk.shape[0]
            if pad > 0:
                pos_chunk = torch.cat([pos_chunk, pos_chunk[-1:].expand(pad, -1)], dim=0)
                quat_chunk = torch.cat([quat_chunk, quat_chunk[-1:].expand(pad, -1)], dim=0)
            goal = Pose(position=pos_chunk, quaternion=quat_chunk)
            result = self.ik.solve_pose(GoalToolPose.from_poses({self.tool: goal}, num_goalset=1))
            n_valid = end - start
            success_parts.append(result.success.view(chunk).cpu().numpy()[:n_valid])
            solution_parts.append(
                result.js_solution.position.squeeze(1).cpu().numpy()[:n_valid, :6]
            )
        success = np.concatenate(success_parts, axis=0).reshape(self.n_points, self.n_orient)
        solution = np.concatenate(solution_parts, axis=0).reshape(self.n_points, self.n_orient, 6)
        return success, solution

    def compute_torque(self, q_arm: np.ndarray) -> np.ndarray:
        """q_arm: (6,) joint1-6 solution. Returns |gravity + 10N downward
        push at the mop tip| per joint (6,), N*m.

        Caveat: gravity torque comes from rebot_mpc_control's existing
        gripper-equipped model (the mop jig's own mass isn't modeled -- it
        wasn't provided). Treat this as an approximation; revisit if the
        mop assembly's mass turns out to differ substantially from the
        gripper's.
        """
        cspace_names = list(self.joint_names) + ["gripper_joint1"]
        cspace_pos = list(q_arm) + [GRIPPER_LOCK_POSITION]
        gravity_tau = np.array(compute_gravity_torque(cspace_names, cspace_pos)[:6])

        q_full = pin.neutral(self.pin_model)
        q_full[0:6] = q_arm
        q_full[6] = GRIPPER_LOCK_POSITION
        q_full[7] = GRIPPER_LOCK_POSITION
        pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.computeFrameJacobian(
            self.pin_model, self.pin_data, q_full, self.mop_tip_pin_id, pin.LOCAL_WORLD_ALIGNED,
        )
        F_ext = np.array([0.0, 0.0, -CLEANING_FORCE_N, 0.0, 0.0, 0.0])
        push_tau = (J.T @ F_ext)[:6]

        return np.abs(gravity_tau + push_tau)

    def _torque_ok(self, q_arm: np.ndarray) -> bool:
        return bool(np.all(self.compute_torque(q_arm) <= JOINT_TORQUE_NOMINAL_NM))

    def find_feasible_solutions(
        self, success: np.ndarray, solution: np.ndarray,
    ) -> Tuple[np.ndarray, List[Optional[np.ndarray]], np.ndarray]:
        """For each table point, among all orientation candidates that are
        both IK-reachable and torque-feasible (<= nominal), picks the one
        with the SMALLEST |tilt| -- minimizing reliance on the passive pitch
        hinge to take up the remaining slack (a point reachable only via a
        heavy tilt is technically "feasible" but saturates mop_tip_joint's
        real +-75 deg range and drives large configuration swings between
        neighboring points, see REACHABILITY_QUALITY_WEIGHT).

        Returns a (n_points,) bool feasibility mask, the chosen (6,) joint
        solution per point (or None if no candidate qualified), and the
        chosen candidate's |tilt| in degrees per point (0.0 where
        infeasible, ignored by callers via the feasibility mask) -- shared
        by evaluate() (area/quality scoring) and _print_torque_headroom()
        (needs the actual chosen solutions, so its reported torques are the
        ones actually driving the feasibility score).
        """
        feasible = np.zeros(self.n_points, dtype=bool)
        chosen: List[Optional[np.ndarray]] = [None] * self.n_points
        chosen_tilt_deg = np.zeros(self.n_points, dtype=np.float32)
        for p in range(self.n_points):
            best_tilt = None
            for o in range(self.n_orient):
                if not success[p, o]:
                    continue
                if not self._torque_ok(solution[p, o]):
                    continue
                tilt = self.orientation_tilt_deg[o]
                if best_tilt is None or tilt < best_tilt:
                    best_tilt = tilt
                    feasible[p] = True
                    chosen[p] = solution[p, o]
                    chosen_tilt_deg[p] = tilt
                    if tilt == 0.0:
                        break  # can't do better than untilted
        return feasible, chosen, chosen_tilt_deg

    def evaluate(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float]:
        """Returns (combined_score, area_score, quality_score).

        area_score: fraction of table points reachable (at some orientation)
        and torque-feasible under the 10N push -- same as before.
        quality_score: mean over ALL points of (1 - chosen_tilt/max_tilt) for
        feasible points and 0 for infeasible ones (same denominator as
        area_score, so both terms stay in [0, 1] and are directly
        comparable/combinable).
        combined_score: the weighted sum Optuna actually maximizes.
        """
        set_mop_mount_angle(self.ik, self.pin_model, self.mop_tip_pin_id, roll_deg, pitch_deg, yaw_deg)
        success, solution = self._solve_all()
        feasible, _, chosen_tilt_deg = self.find_feasible_solutions(success, solution)
        area_score = float(feasible.mean())
        quality_per_point = np.where(feasible, 1.0 - chosen_tilt_deg / self.max_tilt_deg, 0.0)
        quality_score = float(quality_per_point.mean())
        combined_score = (
            REACHABILITY_AREA_WEIGHT * area_score + REACHABILITY_QUALITY_WEIGHT * quality_score
        )
        return combined_score, area_score, quality_score


def run_search(n_trials: int, seed: int = 0):
    evaluator = JigAngleEvaluator()

    def objective(trial: optuna.Trial) -> float:
        roll = trial.suggest_float("roll_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        pitch = trial.suggest_float("pitch_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        yaw = trial.suggest_float("yaw_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        t0 = time.time()
        combined, area, quality = evaluator.evaluate(roll, pitch, yaw)
        dt = time.time() - t0
        trial.set_user_attr("area_score", area)
        trial.set_user_attr("quality_score", quality)
        print(
            f"trial {trial.number}: roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} "
            f"-> combined={combined:.3f} (area={area:.3f} quality={quality:.3f}) ({dt * 1000:.0f} ms)"
        )
        return combined

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study, evaluator


def _print_torque_headroom(evaluator: JigAngleEvaluator, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Also returns the (n_points,) feasibility mask at this angle, so the
    caller can save it for later --visualize use without re-solving.
    """
    set_mop_mount_angle(evaluator.ik, evaluator.pin_model, evaluator.mop_tip_pin_id, roll, pitch, yaw)
    success, solution = evaluator._solve_all()
    feasible, chosen, chosen_tilt_deg = evaluator.find_feasible_solutions(success, solution)

    worst_tau = np.zeros(6)
    for q_arm in chosen:
        if q_arm is not None:
            worst_tau = np.maximum(worst_tau, evaluator.compute_torque(q_arm))

    print("\nWorst-case |torque| across each point's CHOSEN (torque-compliant) solution at the best angle:")
    for i in range(6):
        print(
            f"  joint{i + 1}: {worst_tau[i]:.2f} N*m "
            f"(nominal limit {JOINT_TORQUE_NOMINAL_NM[i]:.1f}, peak ceiling {JOINT_TORQUE_PEAK_NM[i]:.1f})"
        )

    feasible_tilts = chosen_tilt_deg[feasible]
    if feasible_tilts.size > 0:
        print(
            f"\nChosen-orientation tilt across feasible points: mean={feasible_tilts.mean():.1f} deg, "
            f"median={np.median(feasible_tilts):.1f} deg, max={feasible_tilts.max():.1f} deg "
            f"(fraction untilted: {np.mean(feasible_tilts == 0.0):.2f})"
        )
    return feasible


def write_reachability_data(evaluator: JigAngleEvaluator, feasible: np.ndarray) -> None:
    feasible_grid = feasible.reshape(TABLE_GRID_NY, TABLE_GRID_NX)
    rect = reachable_rectangle_bounds_m(feasible_grid)
    print(
        f"\nLargest reachable rectangle (axis-aligned approximation, grid-sampled): "
        f"{rect['width_x_m']:.3f}m (X) x {rect['width_y_m']:.3f}m (Y) = {rect['area_m2']:.3f} m^2"
    )
    print(
        f"  X: [{rect['x_min']:.3f}, {rect['x_max']:.3f}]  Y: [{rect['y_min']:.3f}, {rect['y_max']:.3f}]  Z: 0.000 (table height)"
    )

    half_circle = largest_reachable_half_circle(evaluator.table_pts, feasible)
    print(
        f"\nLargest reachable half-circle (flat edge along the mount edge X="
        f"{TABLE_ORIGIN_XY[0]:.3f}, grid-sampled): radius={half_circle['radius']:.3f}m, "
        f"center_y={half_circle['center_y']:.3f}m, area={half_circle['area_m2']:.3f} m^2"
    )

    np.savez(
        REACHABILITY_DATA_PATH,
        table_pts=evaluator.table_pts,
        feasible=feasible,
        table_grid_nx=TABLE_GRID_NX,
        table_grid_ny=TABLE_GRID_NY,
        rect_x_min=rect["x_min"], rect_x_max=rect["x_max"],
        rect_y_min=rect["y_min"], rect_y_max=rect["y_max"],
        hc_center_x=half_circle["center_x"], hc_center_y=half_circle["center_y"], hc_radius=half_circle["radius"],
    )
    print(f"Wrote reachability map data: {REACHABILITY_DATA_PATH}")


def write_optimized_robot_yaml(roll: float, pitch: float, yaw: float) -> None:
    """Writes OUTPUT_YML_PATH pointing at a freshly-generated, COMPLETE merged
    URDF (gripper removed, mop jig added as real links -- see
    write_merged_urdf/build_merged_robot_cfg). This is what --visualize
    loads: mop_mount/mop_tip render through the same proven URDF-loading+FK
    pipeline as the rest of the arm, rather than hand-computed poses.
    """
    urdf_path = write_merged_urdf(roll, pitch, yaw)
    cfg = build_merged_robot_cfg(urdf_path)
    with open(OUTPUT_YML_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote optimized robot config: {OUTPUT_YML_PATH}")


def _mop_joint_link_xml(roll: float, pitch: float, yaw: float, indent: str = "  ") -> str:
    """The mop_mount_joint/mop_mount/mop_tip_joint/mop_tip XML block, shared
    by write_urdf_snippet() and write_merged_urdf(). mop_mount_joint (fixed)
    carries the rod's DIRECTION (pitch, yaw -- roll does not affect it, see
    _rod_transform's docstring). mop_tip_joint is the REAL hardware's passive
    pitch hinge (self-aligns the mop flat against the table under contact
    force, per the user's mechanical description) -- modeled as a real
    "revolute" joint (axis local-Y, perpendicular to the rod), not "fixed",
    so anything reading this URDF sees it's an actual movable DOF, not a
    rigid weld. Its value is locked at 0 (neutral/unloaded) in
    build_merged_robot_cfg for static visualization -- there's no live
    contact-force simulation here. `roll` (the mop cylinder's spin about its
    own long axis) is intentionally NOT baked into this joint: the cylinder
    is rotationally symmetric, so roll has no effect on its geometry and
    isn't a real, separate mechanical DOF the way pitch is.
    """
    _, p_rad, y_rad = np.deg2rad([0.0, pitch, yaw])
    pitch_limit_rad = np.deg2rad(75.0)  # real hardware's passive-hinge mechanical range
    return f"""{indent}<joint name="mop_mount_joint" type="fixed">
{indent}  <parent link="link6"/>
{indent}  <child link="mop_mount"/>
{indent}  <origin xyz="0 0 0" rpy="0 {p_rad:.6f} {y_rad:.6f}"/>
{indent}</joint>
{indent}<link name="mop_mount">
{indent}  <visual>
{indent}    <origin xyz="0 0 {ROD_LENGTH_M / 2:.4f}" rpy="0 0 0"/>
{indent}    <geometry>
{indent}      <cylinder radius="0.015" length="{ROD_LENGTH_M:.4f}"/>
{indent}    </geometry>
{indent}  </visual>
{indent}  <collision>
{indent}    <origin xyz="0 0 {ROD_LENGTH_M / 2:.4f}" rpy="0 0 0"/>
{indent}    <geometry>
{indent}      <cylinder radius="0.015" length="{ROD_LENGTH_M:.4f}"/>
{indent}    </geometry>
{indent}  </collision>
{indent}</link>
{indent}<joint name="mop_tip_joint" type="revolute">
{indent}  <parent link="mop_mount"/>
{indent}  <child link="mop_tip"/>
{indent}  <origin xyz="0 0 {ROD_LENGTH_M:.4f}" rpy="0 0 0"/>
{indent}  <axis xyz="0 1 0"/>
{indent}  <limit lower="{-pitch_limit_rad:.6f}" upper="{pitch_limit_rad:.6f}" effort="1.0" velocity="1.0"/>
{indent}</joint>
{indent}<link name="mop_tip">
{indent}  <!-- The mop head is a disc (20cm diameter, 5cm tall) whose flat
{indent}       circular face touches the table, confirmed by the user after
{indent}       an earlier wrong assumption here (that it was a roller with
{indent}       its curved side touching, requiring a 90 degree reorientation
{indent}       of the geometry). A URDF cylinder's axis is already local Z
{indent}       by default, which is also the rod's own axis (see mop_mount),
{indent}       so no extra origin rotation is applied here: the disc's flat
{indent}       face ends up perpendicular to the rod, exactly like a real
{indent}       round mop or buffing pad with its handle rising from the
{indent}       disc's center. -->
{indent}  <visual>
{indent}    <geometry>
{indent}      <cylinder radius="{MOP_CYLINDER_DIAMETER_M / 2:.4f}" length="{MOP_CYLINDER_LENGTH_M:.4f}"/>
{indent}    </geometry>
{indent}  </visual>
{indent}  <collision>
{indent}    <geometry>
{indent}      <cylinder radius="{MOP_CYLINDER_DIAMETER_M / 2:.4f}" length="{MOP_CYLINDER_LENGTH_M:.4f}"/>
{indent}    </geometry>
{indent}  </collision>
{indent}</link>
"""


def write_urdf_snippet(roll: float, pitch: float, yaw: float) -> None:
    """Standalone URDF joint+link fragment for the mop mount at the
    optimized angle -- for the mechanical designer to merge into the real
    URDF (or reference directly), not auto-merged into the robot URDF here.
    """
    snippet = f"""<?xml version="1.0"?>
<!-- Mop jig mount, optimized via rebot_reachability_jig.py.
     Roll={roll:.0f} deg, Pitch={pitch:.0f} deg, Yaw={yaw:.0f} deg (15-degree
     manufacturing steps). Rod length={ROD_LENGTH_M:.3f} m (fixed nominal --
     midpoint of the 0.35-0.48m extendable range). Merge this joint+link
     into reBot_B601_DM_with_gripper.urdf in place of the gripper, or
     reference directly for jig fabrication. -->
<robot name="mop_jig_mount">
{_mop_joint_link_xml(roll, pitch, yaw)}</robot>
"""
    with open(OUTPUT_URDF_SNIPPET_PATH, "w") as f:
        f.write(snippet)
    print(f"Wrote mop-mount URDF snippet: {OUTPUT_URDF_SNIPPET_PATH}")


# "gripper_joint" (link6->gripper_link, fixed) is the mount joint; the
# other two are the finger joints (gripper_link->gripper_left/right).
# Confirmed via direct inspection of the base URDF -- easy to miss since
# only "gripper_joint1"/"gripper_joint2" show up in cuRobo's own cspace
# (gripper_joint2 is a mimic, dropped automatically; "gripper_joint" itself
# never appears in cspace at all since it's fixed).
_GRIPPER_JOINTS = {"gripper_joint", "gripper_joint1", "gripper_joint2"}
_GRIPPER_LINKS = {"gripper_link", "gripper_left", "gripper_right"}


def write_merged_urdf(roll: float, pitch: float, yaw: float) -> str:
    """Writes a COMPLETE URDF (base reBot arm, gripper joints/links removed,
    mop_mount+mop_tip added as real links) at the optimized angle. This lets
    the final visualization use cuRobo/viser's own proven URDF-loading+FK
    rendering pipeline directly -- no hand-computed link poses or manually
    drawn primitives (which turned out hard to get right/verify -- see
    memory notes on the rod-orientation debugging).
    """
    tree = ET.parse(REBOT_URDF_PATH)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.get("name") in _GRIPPER_JOINTS:
            root.remove(joint)
    for link in root.findall("link"):
        if link.get("name") in _GRIPPER_LINKS:
            root.remove(link)

    mop_xml = _mop_joint_link_xml(roll, pitch, yaw, indent="  ")
    mop_fragment = ET.fromstring(f"<root>{mop_xml}</root>")
    for child in mop_fragment:
        root.append(child)

    timestamp = int(time.time())
    merged_urdf_path = os.path.join(
        MERGED_URDF_DIR, f"reBot_B601_DM_with_mop_r{roll:+.0f}_p{pitch:+.0f}_y{yaw:+.0f}_{timestamp}.urdf",
    )
    tree.write(merged_urdf_path, xml_declaration=True, encoding="utf-8")
    print(f"Wrote merged URDF (gripper removed, mop jig added): {merged_urdf_path}")
    return merged_urdf_path


def build_merged_robot_cfg(urdf_path: str) -> dict:
    """Robot config dict pointing at the merged URDF (see write_merged_urdf)
    -- used only for the final --visualize step, not the search (which
    keeps using the cheap extra_links in-place-mutation approach in
    build_base_robot_cfg/set_mop_mount_angle).
    """
    with open(REBOT_YML_PATH) as f:
        base = yaml.safe_load(f)
    cfg = copy.deepcopy(base)
    kin = cfg["kinematics"]
    kin["urdf_path"] = urdf_path
    # mop_tip_joint is a real revolute joint now (the passive pitch hinge,
    # see _mop_joint_link_xml) -- not part of cspace, so it must be locked
    # explicitly or cuRobo rejects the URDF ("contains joints that are not
    # active ... and are not explicitly locked", the same error gripper_joint1
    # hit earlier). Locked at 0 (neutral/unloaded): there's no contact-force
    # simulation here to determine where it would actually settle.
    kin["lock_joints"] = {"mop_tip_joint": 0.0}
    kin.pop("extra_links", None)
    kin.pop("extra_collision_spheres", None)

    # cspace.joint_names (and every other per-joint cspace array, all the
    # same length) still lists gripper_joint1 from the base gripper config
    # -- but that joint no longer exists at all in the merged URDF (removed
    # entirely, not just locked), so it must be dropped everywhere, not
    # locked. All these arrays are ordered to match joint_names, so trimming
    # the trailing (gripper) entry off each one keeps them aligned.
    cspace = kin["cspace"]
    n_joints = len(cspace["joint_names"])
    gripper_idx = cspace["joint_names"].index("gripper_joint1")
    for value in cspace.values():
        if isinstance(value, list) and len(value) == n_joints:
            del value[gripper_idx]

    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n not in _GRIPPER_LINKS]
    kin["mesh_link_names"] = [n for n in kin["mesh_link_names"] if n not in _GRIPPER_LINKS]
    for name in _GRIPPER_LINKS:
        kin["collision_spheres"].pop(name, None)
        kin["self_collision_buffer"].pop(name, None)
        kin["self_collision_ignore"].pop(name, None)
    for link_name, ignore_list in kin["self_collision_ignore"].items():
        kin["self_collision_ignore"][link_name] = [n for n in ignore_list if n not in _GRIPPER_LINKS]

    kin["collision_link_names"] = list(kin["collision_link_names"]) + ["mop_mount", "mop_tip"]
    kin["mesh_link_names"] = list(kin["mesh_link_names"]) + ["mop_mount", "mop_tip"]
    kin["collision_spheres"]["mop_mount"] = _rod_collision_spheres_mount_frame()
    kin["collision_spheres"]["mop_tip"] = _cylinder_collision_spheres_tip_frame()
    kin["self_collision_buffer"]["mop_mount"] = 0.0
    kin["self_collision_buffer"]["mop_tip"] = 0.0
    kin["self_collision_ignore"]["mop_mount"] = ["link6", "link5", "mop_tip"]
    kin["self_collision_ignore"]["mop_tip"] = ["mop_mount", "link6"]
    kin["self_collision_ignore"]["link6"] = list(
        set(kin["self_collision_ignore"].get("link6", [])) | {"mop_mount"}
    )
    kin["tool_frames"] = ["mop_tip"]
    return cfg


def _add_reachability_heatmap(server) -> None:
    """Renders the saved per-point feasibility mask (see
    write_reachability_data) as a green/red image laid flat over the table,
    same visual convention as cuRobo's own reachability_example (green =
    reachable + torque-OK, red = not).
    """
    if not os.path.exists(REACHABILITY_DATA_PATH):
        print(f"[info] {REACHABILITY_DATA_PATH} not found -- run a search first to see the reachability map.")
        return
    data = np.load(REACHABILITY_DATA_PATH)
    nx, ny = int(data["table_grid_nx"]), int(data["table_grid_ny"])
    feasible = data["feasible"].reshape(ny, nx)

    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[feasible] = [0, 200, 0]
    img[~feasible] = [200, 0, 0]

    center = (
        TABLE_ORIGIN_XY[0] + TABLE_SIZE_X_M / 2,
        TABLE_ORIGIN_XY[1] + TABLE_SIZE_Y_M / 2,
        0.001,  # tiny lift off the table plane to avoid z-fighting
    )
    server.scene.add_image(
        "/reachability_map", image=img, render_width=TABLE_SIZE_X_M, render_height=TABLE_SIZE_Y_M,
        position=center,
    )
    print(f"Reachability map: {int(feasible.sum())}/{feasible.size} table points feasible (green=OK, red=not).")

    if "rect_x_min" in data:
        x0, x1 = float(data["rect_x_min"]), float(data["rect_x_max"])
        y0, y1 = float(data["rect_y_min"]), float(data["rect_y_max"])
        z = 0.002
        corners = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=np.float32)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)
        server.scene.add_line_segments(
            "/reachable_rectangle", points=lines, colors=(255, 255, 0), line_width=4.0,
        )
        print(f"Largest reachable rectangle: {x1 - x0:.3f}m (X) x {y1 - y0:.3f}m (Y) (yellow outline)")

    if "hc_radius" in data:
        cx, cy, r = float(data["hc_center_x"]), float(data["hc_center_y"]), float(data["hc_radius"])
        z = 0.003
        n_arc = 32
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_arc)  # semicircle bulging toward +X
        arc_pts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta), np.full(n_arc, z)], axis=-1)
        flat_edge = np.array([[cx, cy - r, z], [cx, cy + r, z]], dtype=np.float32)
        segments = [[arc_pts[i], arc_pts[i + 1]] for i in range(n_arc - 1)]
        segments.append(flat_edge)
        server.scene.add_line_segments(
            "/reachable_half_circle", points=np.array(segments, dtype=np.float32),
            colors=(0, 255, 255), line_width=4.0,
        )
        print(f"Largest reachable half-circle: radius={r:.3f}m, center_y={cy:.3f}m (cyan outline)")


def _generate_cleaning_path(
    rect: Dict[str, float], n_rows: int = 6, n_cols: int = 6, direction: str = "x",
) -> np.ndarray:
    """Boustrophedon ("lawnmower") sweep of waypoints inside the reachable
    rectangle -- a simple stand-in "motion plan" for the table-cleaning
    animation. Per the user's own framing ("just use the ik pose for it
    instead"), this is deliberately NOT a real trajectory: each waypoint is
    solved as an independent IK pose, with no interpolation/smoothing or
    velocity/acceleration limiting between them (unlike the proven
    cap_cartesian_step/send_smoothed_segment pipeline the real hardware
    control scripts use -- not needed here since this is IK-only preview,
    not a hardware command).

    direction: "x" (default) runs each long stroke along X, stepping row-to-
    row in Y. "y" transposes this -- each long stroke runs along Y, stepping
    column-to-column in X.
    """
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")
    xs = np.linspace(rect["x_min"], rect["x_max"], n_cols)
    ys = np.linspace(rect["y_min"], rect["y_max"], n_rows)
    path = []
    if direction == "x":
        for i, y in enumerate(ys):
            row_xs = xs if i % 2 == 0 else xs[::-1]  # alternate direction each row
            for x in row_xs:
                path.append([x, y, 0.0])
    else:
        for i, x in enumerate(xs):
            col_ys = ys if i % 2 == 0 else ys[::-1]  # alternate direction each column
            for y in col_ys:
                path.append([x, y, 0.0])
    return np.array(path, dtype=np.float32)


def _compute_passive_pitch_rad(mount_rot_world: np.ndarray, pitch_limit_rad: float) -> float:
    """Best single-DOF hinge angle (about mop_mount's own local Y axis) to
    bring the mop disc's face as close to parallel-with-the-table as
    possible, given the rod's actual world orientation at a solved arm
    pose -- i.e. simulates the passive joint actually settling, instead of
    just locking it at a fixed neutral value (0), which is only done for
    the static default-pose display where nothing is touching the table
    anyway.

    A single hinge axis can't in general rotate an arbitrary direction
    exactly onto another (that needs 2 DOF) -- this finds the CLOSEST
    achievable angle: project both the disc's current axis and the target
    "straight down" direction onto the plane perpendicular to the hinge
    axis, then take the angle between those projections. Clamped to the
    joint's real mechanical range.
    """
    y_world = mount_rot_world @ np.array([0.0, 1.0, 0.0])
    y_world = y_world / np.linalg.norm(y_world)
    z_current = mount_rot_world @ np.array([0.0, 0.0, 1.0])  # disc axis before the hinge
    z_target = np.array([0.0, 0.0, -1.0])  # straight down -- disc face parallel to the table

    def project(v: np.ndarray) -> np.ndarray:
        return v - np.dot(v, y_world) * y_world

    v0, vt = project(z_current), project(z_target)
    n0, nt = np.linalg.norm(v0), np.linalg.norm(vt)
    if n0 < 1e-8 or nt < 1e-8:
        return 0.0
    v0, vt = v0 / n0, vt / nt
    angle = float(np.arctan2(np.dot(y_world, np.cross(v0, vt)), np.dot(v0, vt)))
    return float(np.clip(angle, -pitch_limit_rad, pitch_limit_rad))


def _interpolate_path(start: np.ndarray, end: np.ndarray, step_size_m: float) -> np.ndarray:
    """Linearly-spaced points from just after `start` up to and including
    `end`, spaced ~step_size_m apart (does not include `start` itself, so
    consecutive calls chain without duplicating a point).
    """
    n_steps = max(1, int(np.linalg.norm(np.asarray(end) - np.asarray(start)) / step_size_m))
    return np.array(
        [start + (end - start) * (s / n_steps) for s in range(1, n_steps + 1)], dtype=np.float32,
    )


def _generate_dense_cleaning_path(
    rect: Dict[str, float], n_rows: int = 10, n_cols: int = 10, step_size_m: float = 0.02,
    direction: str = "x",
) -> np.ndarray:
    """Same boustrophedon corner waypoints as _generate_cleaning_path, but
    with small linearly-interpolated steps between them -- feeding
    incremental (warm-started) IK a slowly-moving target, the same style
    rebot_ik_control.py relies on for smooth, jump-free motion (dragging
    the gizmo a little at a time), rather than solving each sparse corner
    independently from scratch (which can land in a totally different,
    equally-valid IK solution branch even for two very close Cartesian
    targets -- that's what caused the visible "joint jumps").

    direction: forwarded to _generate_cleaning_path ("x" or "y" -- which
    table axis the long sweep strokes run along).
    """
    corners = _generate_cleaning_path(rect, n_rows=n_rows, n_cols=n_cols, direction=direction)
    dense = [corners[0]]
    for i in range(1, len(corners)):
        dense.extend(_interpolate_path(corners[i - 1], corners[i], step_size_m))
    return np.array(dense, dtype=np.float32)


def solve_cleaning_sweep(
    n_rows: int = 10, n_cols: int = 10, step_size_m: float = 0.02, direction: str = "x",
) -> List:
    """Solves a DENSE, incrementally warm-started IK sequence across the
    reachable RECTANGLE (per the user's request -- rectangle only, not the
    half-circle) -- matching rebot_ik_control.py's approach (single seed,
    current_state warm-start every solve, LBFGS-only, no CUDA-graph-averse
    particle stage) instead of independently batch-solving each sparse
    waypoint from scratch. Warm-starting keeps consecutive solutions close
    together in joint space, which is what actually avoids the "joint
    jumps" -- combined with small Cartesian steps between targets (see
    _generate_dense_cleaning_path), each solve only needs to move a little
    from the last one.

    Returns one JointState per dense path point (holding the last good
    solution across any transient failures, so the sequence has no gaps),
    each with mop_tip_joint (the passive pitch hinge) set to the angle that
    best keeps the mop disc's face parallel to the table at that point's
    actual solved arm configuration (see _compute_passive_pitch_rad).
    """
    if not os.path.exists(REACHABILITY_DATA_PATH):
        print("[info] no reachability data found -- run a search first. Skipping cleaning sweep.")
        return []
    data = np.load(REACHABILITY_DATA_PATH)
    if "rect_x_min" not in data:
        print("[info] no reachable-rectangle data found. Skipping cleaning sweep.")
        return []
    rect = {
        "x_min": float(data["rect_x_min"]), "x_max": float(data["rect_x_max"]),
        "y_min": float(data["rect_y_min"]), "y_max": float(data["rect_y_max"]),
    }
    rect_path = _generate_dense_cleaning_path(
        rect, n_rows=n_rows, n_cols=n_cols, step_size_m=step_size_m, direction=direction,
    )

    print(f"Building incremental (warm-started) IK solver for the cleaning sweep...")
    ik_cfg = InverseKinematicsCfg.create(
        robot=OUTPUT_YML_PATH,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=True,
    )
    ik = InverseKinematics(ik_cfg)
    ik.config.use_lm_seed = False
    ik.config.exit_early = False
    tool = ik.kinematics.tool_frames[0]

    # Single-seed, warm-started LBFGS (matching rebot_ik_control.py) has no
    # global/multi-seed search fallback -- it can only move a SHORT distance
    # from wherever it currently is, the same way a human can only drag the
    # gizmo a little at a time. rebot_ik_control.py sidesteps a cold start by
    # setting its very FIRST target to exactly match the arm's current
    # position; here that means prepending a dense transition path from the
    # default pose's actual FK position into the rectangle's first corner
    # (confirmed empirically: without this, every single solve failed and
    # the arm never moved from its default pose at all).
    #
    # Orientation needs the same treatment. The disc only truly needs 4-DOF
    # IK (position + pointing generally down -- heading/spin about the
    # disc's own axis doesn't matter at all, since it's rotationally
    # symmetric, and the passive pitch hinge separately handles fine
    # alignment -- confirmed by the user, "we don't care [about heading],
    # any answer might be ok"). So rather than forcing one arbitrary fixed
    # heading for the whole sweep, the target is whichever "pointing down"
    # orientation needs the LEAST rotation from the default pose's own
    # actual starting orientation (the minimal swing that brings the
    # current approach axis to straight down, with zero added twist/heading
    # change). That minimizes, but does NOT eliminate, the first step's
    # orientation gap -- confirmed empirically the default pose's own
    # natural orientation can still be ~90 degrees from pointing at the
    # table, which a single warm-started LBFGS solve genuinely can't bridge
    # in one shot regardless of which valid target is chosen. So the gap is
    # ALSO spread across the transition steps via SLERP, exactly like the
    # position interpolation above, rather than jumped in one go.
    default_state = ik.kinematics.default_joint_state.unsqueeze(0)
    start_pose = ik.kinematics.get_link_poses(default_state.position, [tool])
    start_pos = start_pose.position[0, 0].cpu().numpy()
    start_quat_wxyz = start_pose.quaternion[0, 0].cpu().numpy()
    start_rot = Rotation.from_quat([start_quat_wxyz[1], start_quat_wxyz[2], start_quat_wxyz[3], start_quat_wxyz[0]])
    start_axis_world = start_rot.apply([0.0, 0.0, 1.0])
    swing, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [start_axis_world])
    sweep_rot = swing * start_rot
    sweep_quat_wxyz = np.array(
        [sweep_rot.as_quat()[3], *sweep_rot.as_quat()[:3]], dtype=np.float32,
    )

    transition_pos = _interpolate_path(start_pos, rect_path[0], step_size_m)
    n_transition = len(transition_pos)
    slerp = Slerp([0, 1], Rotation.concatenate([start_rot, sweep_rot]))
    transition_rots = slerp(np.linspace(0.0, 1.0, n_transition + 1)[1:])  # exclude t=0 (start_rot itself)
    transition_quat_wxyz = np.array(
        [[q[3], q[0], q[1], q[2]] for q in transition_rots.as_quat()], dtype=np.float32,
    )

    path = np.concatenate([transition_pos, rect_path], axis=0)
    quats_wxyz = np.concatenate(
        [transition_quat_wxyz, np.tile(sweep_quat_wxyz, (len(rect_path), 1))], axis=0,
    )
    print(f"  {n_transition} transition steps into the rectangle + {len(rect_path)} sweep steps = {len(path)} total.")

    from curobo.types import JointState

    # mop_mount's world orientation, per solved arm configuration, is needed
    # to compute the passive hinge angle -- loaded straight from the merged
    # URDF (already has mop_mount/mop_tip as real links, no custom frame
    # setup needed like the search's build_pin_model does for mop_tip alone).
    with open(OUTPUT_YML_PATH) as f:
        mop_cfg = yaml.safe_load(f)
    pin_model = pin.buildModelFromUrdf(mop_cfg["kinematics"]["urdf_path"])
    pin_data = pin_model.createData()
    mount_id = pin_model.getFrameId("mop_mount")
    pitch_limit_rad = np.deg2rad(75.0)  # matches _mop_joint_link_xml's joint limit

    current_state = ik.kinematics.default_joint_state.unsqueeze(0)  # (1, active_dof)
    solved_js = []
    n_success = 0
    for i in range(len(path)):
        pos_t = torch.tensor(path[i:i + 1], device="cuda", dtype=torch.float32)
        quat_t = torch.tensor(quats_wxyz[i:i + 1], device="cuda", dtype=torch.float32)
        # Same shape dance as rebot_ik_control.py's loop: works whether
        # current_state is the initial (1, active_dof) or a previous
        # solve's (1, 1, full_dof_incl_locked) js_solution -- get_active_js
        # filters down to active joints either way, and the
        # squeeze/unsqueeze pair converges both cases to (1, active_dof).
        active_js = ik.get_active_js(current_state.squeeze(0)).unsqueeze(0)
        result = ik.solve_pose(
            GoalToolPose.from_poses({tool: Pose(position=pos_t, quaternion=quat_t)}, num_goalset=1),
            current_state=active_js.squeeze(1).clone(), return_seeds=1,
        )
        if result.success.any():
            n_success += 1
        # Advance to the best-effort solution EVEN on failure (not just on
        # success): result.js_solution is always the closest pose LBFGS
        # found, just outside tolerance when it "fails". Freezing
        # current_state at the old position on any failure was the actual
        # bug behind most of the sequence failing -- confirmed empirically:
        # once one step misses, the still-advancing target creates an
        # ever-growing gap from the frozen state, snowballing into a long
        # cascade of failures (position/rotation error climbing step over
        # step) instead of the solver being able to catch back up.
        current_state = result.js_solution.clone()
        js = current_state.squeeze(0).squeeze(0) if current_state.position.dim() == 3 else current_state.squeeze(0)

        q_arm_t = js.position[:6]
        q_arm = q_arm_t.cpu().numpy()
        q_full = pin.neutral(pin_model)
        q_full[0:6] = q_arm
        pin.forwardKinematics(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)
        pitch_rad = _compute_passive_pitch_rad(pin_data.oMf[mount_id].rotation, pitch_limit_rad)

        # Always the 6 active joint names + mop_tip_joint -- NOT js.joint_names,
        # which is 6 names before any successful solve but 7 (already
        # including mop_tip_joint) afterward, so blindly appending would
        # duplicate it in the post-solve case.
        full_names = list(ik.kinematics.joint_names) + ["mop_tip_joint"]
        full_pos = torch.cat([q_arm_t, torch.tensor([pitch_rad], device="cuda", dtype=torch.float32)])
        solved_js.append(JointState.from_position(full_pos.unsqueeze(0), joint_names=full_names).squeeze(0))

    print(f"Cleaning sweep: {n_success}/{len(path)} dense waypoints solved (holding last solution on any misses).")
    return solved_js


def solve_cleaning_sweep_mpc(
    n_rows: int = 10, n_cols: int = 10, control_dt: float = 0.03, direction: str = "x",
) -> List:
    """MPC + MIT-dynamics-simulated version of the cleaning sweep -- reuses
    the proven reactive MPC pattern from rebot_mpc_control.py's
    run_interactive_mpc_control (replan from BELIEF every tick, never real
    feedback -- see that function's docstring for why, it's the confirmed
    fix for a real live-hardware runaway) and rebot_mit_sim.py's Damiao
    MIT-mode dynamics simulator (the same one rebot_motion_planning.py's
    --sim mode drives), instead of displaying raw IK solutions directly.
    MPC's own trajectory optimization (velocity/acceleration
    regularization) plus physically-simulated inertia/torque limits damp
    the per-tick target updates in a way plain incremental IK has no
    mechanism for -- this is expected to look smoother still than
    solve_cleaning_sweep()'s already-warm-started IK.

    Caveat (same one the torque check already carries): MitDynamicsSimulator
    is hardcoded to simulate the ORIGINAL gripper-equipped arm
    (rebot_mit_sim.REBOT_URDF_PATH) -- the mop assembly's own mass isn't
    modeled (never provided). The (now nonexistent in our merged URDF)
    gripper joint is simply locked at a fixed value for the simulator's own
    internal 7-DOF model; it has no effect on the 6 arm joints' simulated
    dynamics beyond that.
    """
    if not os.path.exists(REACHABILITY_DATA_PATH):
        print("[info] no reachability data found -- run a search first. Skipping cleaning sweep.")
        return []
    data = np.load(REACHABILITY_DATA_PATH)
    if "rect_x_min" not in data:
        print("[info] no reachable-rectangle data found. Skipping cleaning sweep.")
        return []
    rect = {
        "x_min": float(data["rect_x_min"]), "x_max": float(data["rect_x_max"]),
        "y_min": float(data["rect_y_min"]), "y_max": float(data["rect_y_max"]),
    }
    rect_path = _generate_dense_cleaning_path(
        rect, n_rows=n_rows, n_cols=n_cols, step_size_m=0.02, direction=direction,
    )

    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import JointState
    from rebot_mit_sim import CSPACE_JOINT_NAMES, DEFAULT_SIM_DT, MitDynamicsSimulator

    print("Building MPC + MIT-dynamics-simulated solver for the cleaning sweep...")
    config = ModelPredictiveControlCfg.create(
        robot=OUTPUT_YML_PATH,
        use_cuda_graph=True,
        optimization_dt=control_dt,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )
    mpc = ModelPredictiveControl(config)
    tool = mpc.tool_frames[0]

    home_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names,
    )
    home_state.velocity = torch.zeros_like(home_state.position)
    home_state.acceleration = torch.zeros_like(home_state.position)
    mpc.setup(home_state)
    kin_result = mpc.compute_kinematics(home_state)
    target_link_poses = kin_result.tool_poses.to_dict()
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=False,
    )

    # Same reasoning as solve_cleaning_sweep()'s transition path: MPC also
    # needs a reasonably close first target, not an arbitrary cold jump, to
    # track smoothly from tick one -- and heading is still free (4-DOF IK,
    # see solve_cleaning_sweep), so the target is whichever "pointing down"
    # orientation needs the least rotation from the home pose's own actual
    # orientation, with that unavoidable remainder spread across the
    # transition steps via SLERP rather than jumped in one go.
    start_pos = target_link_poses[tool].position[0].cpu().numpy()
    start_quat_wxyz = target_link_poses[tool].quaternion[0].cpu().numpy()
    start_rot = Rotation.from_quat([start_quat_wxyz[1], start_quat_wxyz[2], start_quat_wxyz[3], start_quat_wxyz[0]])
    start_axis_world = start_rot.apply([0.0, 0.0, 1.0])
    swing, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [start_axis_world])
    sweep_rot = swing * start_rot
    sweep_quat_wxyz = np.array([sweep_rot.as_quat()[3], *sweep_rot.as_quat()[:3]], dtype=np.float32)

    transition_pos = _interpolate_path(start_pos, rect_path[0], 0.02)
    n_transition = len(transition_pos)
    slerp = Slerp([0, 1], Rotation.concatenate([start_rot, sweep_rot]))
    transition_rots = slerp(np.linspace(0.0, 1.0, n_transition + 1)[1:])
    transition_quat_wxyz = np.array(
        [[q[3], q[0], q[1], q[2]] for q in transition_rots.as_quat()], dtype=np.float32,
    )
    path = np.concatenate([transition_pos, rect_path], axis=0)
    quats_wxyz = np.concatenate(
        [transition_quat_wxyz, np.tile(sweep_quat_wxyz, (len(rect_path), 1))], axis=0,
    )
    print(f"  {n_transition} transition steps into the rectangle + {len(rect_path)} sweep steps = {len(path)} total.")

    # rebot_mit_sim's simulator is hardcoded to the ORIGINAL 7-DOF
    # gripper-equipped cspace (CSPACE_JOINT_NAMES = [joint1..6,
    # gripper_joint1]) -- lock_gripper_at hard-locks that 7th DOF every
    # step regardless of what's passed in p_des/v_des for it.
    initial_q = np.concatenate([mpc.default_joint_position.cpu().numpy(), [GRIPPER_LOCK_POSITION]])
    mit_sim = MitDynamicsSimulator(initial_q, lock_gripper_at=GRIPPER_LOCK_POSITION)
    # Per cuRobo's own MPCSolverCfg docstring: "command_dt = optimization_dt /
    # interpolation_steps" -- each of the n_wp waypoints in one action_sequence
    # spans command_dt of real time, NOT control_dt. Simulating a full
    # control_dt per waypoint (as if wp_dt == control_dt) advances physics
    # ~interpolation_steps x too far per waypoint, which desyncs the
    # replan-from-belief state badly enough to pin the passive-pitch joint at
    # its mechanical limit almost the whole sweep (confirmed empirically:
    # 78% of ticks saturated at +-30 deg with the wrong substep count).
    command_dt = control_dt / config.interpolation_steps
    sim_substeps = max(1, round(command_dt / DEFAULT_SIM_DT))

    with open(OUTPUT_YML_PATH) as f:
        mop_cfg = yaml.safe_load(f)
    pin_model = pin.buildModelFromUrdf(mop_cfg["kinematics"]["urdf_path"])
    pin_data = pin_model.createData()
    mount_id = pin_model.getFrameId("mop_mount")
    pitch_limit_rad = np.deg2rad(75.0)  # matches _mop_joint_link_xml's joint limit

    believed_state = home_state
    solved_js = []
    n_valid_ticks = 0
    for i in range(len(path)):
        pos_t = torch.tensor(path[i:i + 1], device="cuda", dtype=torch.float32)
        quat_t = torch.tensor(quats_wxyz[i:i + 1], device="cuda", dtype=torch.float32)
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses({tool: Pose(position=pos_t, quaternion=quat_t)}, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
            run_ik=False,
        )
        mpc_result = mpc.optimize_action_sequence(believed_state)
        has_action = (
            mpc_result.action_sequence is not None and mpc_result.action_sequence.position.shape[1] > 0
        )
        if has_action:
            n_valid_ticks += 1
            seq_pos = mpc_result.action_sequence.position.clone()  # [1, n_wp, 6]
            seq_vel = mpc_result.action_sequence.velocity.clone()
            n_wp = seq_pos.shape[1]
            for wp in range(n_wp):
                p_des = np.concatenate([seq_pos[0, wp, :].cpu().numpy(), [GRIPPER_LOCK_POSITION]])
                v_des = np.concatenate([seq_vel[0, wp, :].cpu().numpy(), [0.0]])
                # Recomputed per waypoint (not once per outer tick), same as
                # rebot_motion_planning.py's mpc_fallback_loop.
                tau_g = compute_gravity_torque(CSPACE_JOINT_NAMES, mit_sim.cspace_position().tolist())
                for _ in range(sim_substeps):
                    mit_sim.step(p_des, v_des, DEFAULT_SIM_DT, tau_g=tau_g)
                    if not mit_sim.is_finite():
                        print(f"  [warn] MIT sim diverged at step {i}, waypoint {wp} -- aborting sweep early.")
                        print(f"Cleaning sweep (MPC+physics): {n_valid_ticks}/{i + 1} ticks had a valid action, {len(solved_js)} states recorded before divergence.")
                        return solved_js

            sim_pos_6 = mit_sim.cspace_position()[:6]
            sim_vel_6 = mit_sim.cspace_velocity()[:6]
            believed_state = JointState.from_position(
                torch.tensor(sim_pos_6[None], device="cuda", dtype=torch.float32), joint_names=mpc.joint_names,
            )
            believed_state.velocity = torch.tensor(sim_vel_6[None], device="cuda", dtype=torch.float32)
            believed_state.acceleration = torch.zeros_like(believed_state.position)

        q_arm = mit_sim.cspace_position()[:6]
        q_full = pin.neutral(pin_model)
        q_full[0:6] = q_arm
        pin.forwardKinematics(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)
        pitch_rad = _compute_passive_pitch_rad(pin_data.oMf[mount_id].rotation, pitch_limit_rad)

        full_names = list(mpc.joint_names) + ["mop_tip_joint"]
        full_pos = torch.cat([
            torch.tensor(q_arm, device="cuda", dtype=torch.float32),
            torch.tensor([pitch_rad], device="cuda", dtype=torch.float32),
        ])
        solved_js.append(JointState.from_position(full_pos.unsqueeze(0), joint_names=full_names).squeeze(0))

    print(f"Cleaning sweep (MPC+physics): {n_valid_ticks}/{len(path)} ticks had a valid MPC action.")
    return solved_js


def animate_cleaning_sweep(viser_viz, solved_js: List, dt: float = 0.15) -> None:
    """Loops through the precomputed waypoint JointStates, animating the
    arm -- meant to be run in a background thread, started by a GUI button
    (see visualize()), not automatically, matching rebot_motion_planning.py's
    button-triggered pattern rather than auto-starting.
    """
    if not solved_js:
        print("[info] no reachable waypoints to animate.")
        return
    while True:
        for js in solved_js:
            viser_viz.set_joint_state(js)
            time.sleep(dt)


def visualize(port: int = 8080, clean_mode: str = "ik", clean_direction: str = "x") -> None:
    """Launches Viser showing the arm reaching the table center, with the
    mop jig rendered as REAL URDF links (mop_mount, mop_tip -- see
    write_merged_urdf/build_merged_robot_cfg) through cuRobo/viser's own
    proven URDF-loading+FK pipeline, plus the saved reachability map
    overlaid on the table.

    Earlier versions of this function hand-computed link6/mop_tip world
    poses (via a separate pinocchio model) and drew the rod/mop as manually
    placed+oriented viser primitives. That turned out hard to get
    consistently right/verify visually (see rebot_reachability_jig memory
    notes) even though the underlying position math checked out against
    cuRobo's own FK -- rendering the jig as real URDF geometry sidesteps
    the whole problem, since it's driven by the exact same
    joint-state-to-mesh pipeline already proven correct for the rest of
    the arm.

    clean_mode: "ik" (solve_cleaning_sweep -- incremental warm-started
    single-seed IK, no physics) or "mpc" (solve_cleaning_sweep_mpc --
    reactive MPC + rebot_mit_sim's MIT-mode dynamics simulator, expected to
    look smoother still thanks to trajectory-level velocity/acceleration
    regularization and simulated inertia).
    clean_direction: "x" or "y" -- which table axis the lawnmower sweep's
    long strokes run along (see _generate_cleaning_path).
    """
    from curobo.viewer import ViserVisualizer

    if not os.path.exists(OUTPUT_YML_PATH):
        raise RuntimeError(f"{OUTPUT_YML_PATH} not found -- run a search first to generate it.")
    if clean_mode not in ("ik", "mpc"):
        raise ValueError(f"clean_mode must be 'ik' or 'mpc', got {clean_mode!r}")
    if clean_direction not in ("x", "y"):
        raise ValueError(f"clean_direction must be 'x' or 'y', got {clean_direction!r}")

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=OUTPUT_YML_PATH),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    server = viser_viz._server
    _add_reachability_heatmap(server)

    ik_cfg = InverseKinematicsCfg.create(robot=OUTPUT_YML_PATH, num_seeds=24, self_collision_check=True)
    ik = InverseKinematics(ik_cfg)
    viser_viz.set_joint_state(ik.kinematics.default_joint_state)
    print(f"Showing the arm at its initial/default pose at http://localhost:{port}")
    print(f"Cleaning sweep mode: {clean_mode}, direction: {clean_direction}")

    solved_js = (
        solve_cleaning_sweep(direction=clean_direction) if clean_mode == "ik"
        else solve_cleaning_sweep_mpc(direction=clean_direction)
    )
    animation_state = {"running": False}

    def on_start_cleaning(_event) -> None:
        if animation_state["running"]:
            return
        animation_state["running"] = True

        def run() -> None:
            try:
                animate_cleaning_sweep(viser_viz, solved_js)
            finally:
                animation_state["running"] = False

        threading.Thread(target=run, daemon=True).start()

    clean_btn = server.gui.add_button(f"Start Cleaning ({clean_mode}/{clean_direction})", color="green")
    clean_btn.on_click(on_start_cleaning)

    print(f"\nViewer running at http://localhost:{port}")
    print(f"Click 'Start Cleaning ({clean_mode}/{clean_direction})' to animate the lawnmower sweep over the reachable rectangle.")
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", help="Skip the search; visualize the last optimized result.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--clean-mode", choices=["ik", "mpc"], default="ik",
        help="Cleaning-sweep solver: 'ik' (incremental warm-started single-seed IK, no physics) "
             "or 'mpc' (reactive MPC + MIT-mode dynamics simulator). Only used with --visualize.",
    )
    parser.add_argument(
        "--clean-direction", choices=["x", "y"], default="x",
        help="Which table axis the lawnmower sweep's long strokes run along. Only used with --visualize.",
    )
    args = parser.parse_args()

    if args.visualize:
        visualize(port=args.port, clean_mode=args.clean_mode, clean_direction=args.clean_direction)
        return

    study, evaluator = run_search(n_trials=args.n_trials, seed=args.seed)

    best = study.best_params
    best_trial = study.best_trial
    print("\n=== Best jig mounting angle found ===")
    print(f"  roll={best['roll_deg']:+.0f} deg  pitch={best['pitch_deg']:+.0f} deg  yaw={best['yaw_deg']:+.0f} deg")
    print(
        f"  combined score: {study.best_value:.3f}  "
        f"(area={best_trial.user_attrs['area_score']:.3f}, quality={best_trial.user_attrs['quality_score']:.3f}, "
        f"weights area={REACHABILITY_AREA_WEIGHT}/quality={REACHABILITY_QUALITY_WEIGHT})"
    )
    if any(abs(best[k]) == JIG_ANGLE_RANGE_DEG[1] for k in ("roll_deg", "pitch_deg", "yaw_deg")):
        print(
            "  [warn] optimum sits at a search-space boundary "
            f"({JIG_ANGLE_RANGE_DEG}) -- consider widening the range."
        )

    feasible = _print_torque_headroom(evaluator, best["roll_deg"], best["pitch_deg"], best["yaw_deg"])
    write_reachability_data(evaluator, feasible)

    write_optimized_robot_yaml(best["roll_deg"], best["pitch_deg"], best["yaw_deg"])
    write_urdf_snippet(best["roll_deg"], best["pitch_deg"], best["yaw_deg"])
    print("\nRun with --visualize to view the optimized mount in Viser.")


if __name__ == "__main__":
    main()
