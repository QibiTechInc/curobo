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
import time
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pinocchio as pin
import torch
import yaml
from scipy.spatial.transform import Rotation

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import ContentPath, GoalToolPose, Pose

from rebot_mpc_control import REBOT_URDF_PATH, compute_gravity_torque

REBOT_YML_PATH = "/home/kendemu/curobo/rebot_b601_gripper.yml"
OUTPUT_YML_PATH = "/home/kendemu/curobo/rebot_b601_mop.yml"
OUTPUT_URDF_SNIPPET_PATH = "/home/kendemu/curobo/mop_jig_mount.urdf.xacro"
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
MOP_CYLINDER_LENGTH_M = 0.20

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
# table's edge (X in [0,1], Y in [-0.5,0.5], ROS2/REP-103 convention -- X
# forward, Y left, Z up).
TABLE_SIZE_M = 1.0
TABLE_ORIGIN_XY = (0.0, -0.5)  # table spans X in [0, 1], Y in [-0.5, 0.5], Z=0
TABLE_GRID_N = 8  # 8x8 = 64 points

# Discretized orientation sweep per table point -- see module docstring for
# why this replaces cost-weight-based orientation relaxation. ROLL is
# rotation about the mop cylinder's own long axis (fully free -- cylinder is
# rotationally symmetric -- sampled coarsely since only our SEARCH is
# discretized, not the real hardware). TILT is the approach-direction tilt
# the passive pitch joint is assumed able to self-align through -- an
# ASSUMPTION; narrow or widen this if the real joint's mechanical range is
# known to differ.
HEADINGS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
ROLL_DEG = [0.0, 90.0, 180.0, 270.0]
TILT_DEG = [-30.0, -15.0, 0.0, 15.0, 30.0]

# Manufacturing constraint: the jig mount can only be set in 15-degree
# increments, so the search space itself is quantized to match (not just the
# reported result).
JIG_ANGLE_RANGE_DEG = (-90.0, 90.0)
JIG_ANGLE_STEP_DEG = 15.0

# local X = cylinder's long axis (world +X at heading=tilt=roll=0);
# local Z = -world Z (mop tip points straight down at the table).
_R_CANON = Rotation.from_matrix(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float))


def _rod_transform(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[np.ndarray, Rotation]:
    """Fixed transform from link6 to mop_tip for a given jig mounting angle.

    The rod extends ROD_LENGTH_M along whatever direction the mount's own
    (rotated) local Z axis ends up pointing -- i.e. rotating the mount also
    rotates which way the rod points out from link6.
    """
    rot = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    pos = rot.apply([0.0, 0.0, ROD_LENGTH_M])
    return pos, rot


def _mop_collision_spheres() -> List[Dict]:
    """Closed-form spheres approximating the rod + mop cylinder, in mop_tip's
    OWN local frame -- fixed regardless of the jig mounting angle (only the
    link6->mop_tip transform changes per candidate, not this local layout),
    and only need regenerating if ROD_LENGTH_M or the cylinder dimensions
    change (neither varies in this pass).

    Convention: mop_tip's local +Z points back along the rod toward the
    mount; local X is the cylinder's long axis (see _R_CANON).
    """
    spheres = []
    n_rod = 6
    rod_radius = 0.015
    for i in range(n_rod):
        t = -ROD_LENGTH_M * i / (n_rod - 1)
        spheres.append({"center": [0.0, 0.0, t], "radius": rod_radius})
    n_cyl = 3
    cyl_radius = MOP_CYLINDER_DIAMETER_M / 2.0
    half_len = MOP_CYLINDER_LENGTH_M / 2.0
    for i in range(n_cyl):
        x = -half_len + MOP_CYLINDER_LENGTH_M * i / (n_cyl - 1)
        spheres.append({"center": [x, 0.0, 0.0], "radius": cyl_radius})
    return spheres


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
    xs = np.linspace(TABLE_ORIGIN_XY[0], TABLE_ORIGIN_XY[0] + TABLE_SIZE_M, TABLE_GRID_N)
    ys = np.linspace(TABLE_ORIGIN_XY[1], TABLE_ORIGIN_XY[1] + TABLE_SIZE_M, TABLE_GRID_N)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=-1).astype(np.float32)


def build_orientation_candidates() -> np.ndarray:
    """Returns (n_headings*n_roll*n_tilt, 4) wxyz quaternions."""
    quats = []
    for heading in HEADINGS_DEG:
        for tilt in TILT_DEG:
            for roll in ROLL_DEG:
                R = (
                    Rotation.from_euler("z", heading, degrees=True)
                    * Rotation.from_euler("y", tilt, degrees=True)
                    * _R_CANON
                    * Rotation.from_euler("x", roll, degrees=True)
                )
                q = R.as_quat()  # xyzw
                quats.append([q[3], q[0], q[1], q[2]])
    return np.array(quats, dtype=np.float32)


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
    ) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
        """For each table point, picks the FIRST orientation candidate that
        is both IK-reachable and torque-feasible (<= nominal). Returns a
        (n_points,) bool feasibility mask and, per point, the chosen (6,)
        joint solution (or None if no orientation candidate qualified) --
        shared by evaluate() (just needs the fraction) and
        _print_torque_headroom() (needs the actual chosen solutions, so its
        reported torques are the ones actually driving the feasibility
        score, not the worst across every attempted-and-rejected orientation).
        """
        feasible = np.zeros(self.n_points, dtype=bool)
        chosen: List[Optional[np.ndarray]] = [None] * self.n_points
        for p in range(self.n_points):
            for o in range(self.n_orient):
                if not success[p, o]:
                    continue
                if self._torque_ok(solution[p, o]):
                    feasible[p] = True
                    chosen[p] = solution[p, o]
                    break
        return feasible, chosen

    def evaluate(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> float:
        """Returns the fraction of table points that are both reachable (at
        some orientation candidate) and torque-feasible under the 10N push.
        """
        set_mop_mount_angle(self.ik, self.pin_model, self.mop_tip_pin_id, roll_deg, pitch_deg, yaw_deg)
        success, solution = self._solve_all()
        feasible, _ = self.find_feasible_solutions(success, solution)
        return float(feasible.mean())


def run_search(n_trials: int, seed: int = 0):
    evaluator = JigAngleEvaluator()

    def objective(trial: optuna.Trial) -> float:
        roll = trial.suggest_float("roll_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        pitch = trial.suggest_float("pitch_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        yaw = trial.suggest_float("yaw_deg", *JIG_ANGLE_RANGE_DEG, step=JIG_ANGLE_STEP_DEG)
        t0 = time.time()
        score = evaluator.evaluate(roll, pitch, yaw)
        dt = time.time() - t0
        print(
            f"trial {trial.number}: roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} "
            f"-> feasible_fraction={score:.3f} ({dt * 1000:.0f} ms)"
        )
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study, evaluator


def _print_torque_headroom(evaluator: JigAngleEvaluator, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Also returns the (n_points,) feasibility mask at this angle, so the
    caller can save it for later --visualize use without re-solving.
    """
    set_mop_mount_angle(evaluator.ik, evaluator.pin_model, evaluator.mop_tip_pin_id, roll, pitch, yaw)
    success, solution = evaluator._solve_all()
    feasible, chosen = evaluator.find_feasible_solutions(success, solution)

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
    return feasible


def write_reachability_data(evaluator: JigAngleEvaluator, feasible: np.ndarray) -> None:
    np.savez(
        REACHABILITY_DATA_PATH,
        table_pts=evaluator.table_pts,
        feasible=feasible,
        table_grid_n=TABLE_GRID_N,
    )
    print(f"Wrote reachability map data: {REACHABILITY_DATA_PATH}")


def write_optimized_robot_yaml(roll: float, pitch: float, yaw: float) -> None:
    cfg = build_base_robot_cfg()
    pos, rot = _rod_transform(roll, pitch, yaw)
    quat = rot.as_quat()  # xyzw
    cfg["kinematics"]["extra_links"]["mop_tip"]["fixed_transform"] = [
        float(pos[0]), float(pos[1]), float(pos[2]),
        float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]),
    ]
    with open(OUTPUT_YML_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote optimized robot config: {OUTPUT_YML_PATH}")


def write_urdf_snippet(roll: float, pitch: float, yaw: float) -> None:
    """Standalone URDF joint+link fragment for the mop mount at the
    optimized angle -- for the mechanical designer to merge into the real
    URDF (or reference directly), not auto-merged into the robot URDF here.
    """
    r_rad, p_rad, y_rad = np.deg2rad([roll, pitch, yaw])
    snippet = f"""<?xml version="1.0"?>
<!-- Mop jig mount, optimized via rebot_reachability_jig.py.
     Roll={roll:.0f} deg, Pitch={pitch:.0f} deg, Yaw={yaw:.0f} deg (15-degree
     manufacturing steps). Rod length={ROD_LENGTH_M:.3f} m (fixed nominal --
     midpoint of the 0.35-0.48m extendable range). Merge this joint+link
     into reBot_B601_DM_with_gripper.urdf in place of the gripper, or
     reference directly for jig fabrication. -->
<robot name="mop_jig_mount">
  <joint name="mop_mount_joint" type="fixed">
    <parent link="link6"/>
    <child link="mop_mount"/>
    <origin xyz="0 0 0" rpy="{r_rad:.6f} {p_rad:.6f} {y_rad:.6f}"/>
  </joint>
  <link name="mop_mount">
    <visual>
      <origin xyz="0 0 {ROD_LENGTH_M / 2:.4f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.015" length="{ROD_LENGTH_M:.4f}"/>
      </geometry>
    </visual>
  </link>
  <joint name="mop_tip_joint" type="fixed">
    <parent link="mop_mount"/>
    <child link="mop_tip"/>
    <origin xyz="0 0 {ROD_LENGTH_M:.4f}" rpy="0 0 0"/>
  </joint>
  <link name="mop_tip">
    <visual>
      <origin xyz="0 0 0" rpy="0 1.5707963 0"/>
      <geometry>
        <cylinder radius="{MOP_CYLINDER_DIAMETER_M / 2:.4f}" length="{MOP_CYLINDER_LENGTH_M:.4f}"/>
      </geometry>
    </visual>
  </link>
</robot>
"""
    with open(OUTPUT_URDF_SNIPPET_PATH, "w") as f:
        f.write(snippet)
    print(f"Wrote mop-mount URDF snippet: {OUTPUT_URDF_SNIPPET_PATH}")


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
    n = int(data["table_grid_n"])
    feasible = data["feasible"].reshape(n, n)

    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[feasible] = [0, 200, 0]
    img[~feasible] = [200, 0, 0]

    center = (
        TABLE_ORIGIN_XY[0] + TABLE_SIZE_M / 2,
        TABLE_ORIGIN_XY[1] + TABLE_SIZE_M / 2,
        0.001,  # tiny lift off the table plane to avoid z-fighting
    )
    server.scene.add_image(
        "/reachability_map", image=img, render_width=TABLE_SIZE_M, render_height=TABLE_SIZE_M,
        position=center,
    )
    print(f"Reachability map: {int(feasible.sum())}/{feasible.size} table points feasible (green=OK, red=not).")


def _add_mop_jig_primitives(server, link6_pos: np.ndarray, mop_tip_pos: np.ndarray, mop_tip_quat_wxyz: np.ndarray) -> None:
    """Draws the rod + mop cylinder as simple primitives (the URDF has no
    mesh for the jig -- it doesn't physically exist yet). viser cylinders
    have their height along local +Y by default (THREE.CylinderGeometry).
    """
    rod_vec = mop_tip_pos - link6_pos
    rod_len = float(np.linalg.norm(rod_vec))
    if rod_len > 1e-6:
        rod_rot, _ = Rotation.align_vectors([rod_vec / rod_len], [[0.0, 1.0, 0.0]])
        rod_quat = rod_rot.as_quat()  # xyzw
        server.scene.add_cylinder(
            "/mop_jig/rod", radius=0.015, height=rod_len, color=(120, 120, 120),
            wxyz=(rod_quat[3], rod_quat[0], rod_quat[1], rod_quat[2]),
            position=tuple((link6_pos + mop_tip_pos) / 2),
        )

    mop_axis_world = Rotation.from_quat(
        [mop_tip_quat_wxyz[1], mop_tip_quat_wxyz[2], mop_tip_quat_wxyz[3], mop_tip_quat_wxyz[0]]
    ).apply([1.0, 0.0, 0.0])  # local X = cylinder's long axis, see _R_CANON
    mop_rot, _ = Rotation.align_vectors([mop_axis_world], [[0.0, 1.0, 0.0]])
    mop_quat = mop_rot.as_quat()  # xyzw
    server.scene.add_cylinder(
        "/mop_jig/mop_head", radius=MOP_CYLINDER_DIAMETER_M / 2, height=MOP_CYLINDER_LENGTH_M,
        color=(80, 140, 220),
        wxyz=(mop_quat[3], mop_quat[0], mop_quat[1], mop_quat[2]),
        position=tuple(mop_tip_pos),
    )


def visualize(port: int = 8080) -> None:
    """Launches Viser showing: the arm (gripper stripped, see
    build_base_robot_cfg) reaching the table center with the mop jig drawn
    as primitives (the URDF has no jig mesh), and the saved reachability
    map for the optimized angle overlaid on the table.
    """
    from curobo.viewer import ViserVisualizer

    if not os.path.exists(OUTPUT_YML_PATH):
        raise RuntimeError(f"{OUTPUT_YML_PATH} not found -- run a search first to generate it.")

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=OUTPUT_YML_PATH),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=True,
        add_robot_to_scene=True,
    )
    server = viser_viz._server
    _add_reachability_heatmap(server)

    with open(OUTPUT_YML_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["kinematics"]["tool_frames"] = ["link6", "mop_tip"]  # so both poses are queryable below

    orientations = build_orientation_candidates()
    ik_cfg = InverseKinematicsCfg.create(
        robot=cfg, num_seeds=24, self_collision_check=True, max_batch_size=len(orientations),
    )
    ik = InverseKinematics(ik_cfg)

    center = np.array([
        TABLE_ORIGIN_XY[0] + TABLE_SIZE_M / 2, TABLE_ORIGIN_XY[1] + TABLE_SIZE_M / 2, 0.0,
    ], dtype=np.float32)
    pos_t = torch.tensor(center[None], device="cuda", dtype=torch.float32).expand(len(orientations), -1)
    quat_t = torch.tensor(orientations, device="cuda", dtype=torch.float32)
    result = ik.solve_pose(GoalToolPose.from_poses({"mop_tip": Pose(position=pos_t, quaternion=quat_t)}, num_goalset=1))
    success = result.success.view(-1)
    if success.any():
        first = int(torch.nonzero(success)[0].item())
        js = result.js_solution[first]
        viser_viz.set_joint_state(js.squeeze(0))

        # get_link_poses expects only the ACTIVE (unlocked) joints -- js.position
        # includes the locked gripper_joint1 as a trailing 7th column.
        active_position = js.position[..., : len(ik.kinematics.joint_names)]
        link_poses = ik.kinematics.get_link_poses(active_position, ["link6", "mop_tip"])
        link6_pos = link_poses.position[0, 0].cpu().numpy()
        mop_tip_pos = link_poses.position[0, 1].cpu().numpy()
        mop_tip_quat = link_poses.quaternion[0, 1].cpu().numpy()  # wxyz
        _add_mop_jig_primitives(server, link6_pos, mop_tip_pos, mop_tip_quat)
        print(f"Showing the arm reaching the table center at http://localhost:{port}")
    else:
        print(
            "Table-center pose not reachable at the optimized angle under any of the "
            f"{len(orientations)} orientation candidates tried -- showing default pose instead."
        )
        print(f"Viewer running at http://localhost:{port}")

    print("Press Ctrl+C to exit.")
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
    args = parser.parse_args()

    if args.visualize:
        visualize(port=args.port)
        return

    study, evaluator = run_search(n_trials=args.n_trials, seed=args.seed)

    best = study.best_params
    print("\n=== Best jig mounting angle found ===")
    print(f"  roll={best['roll_deg']:+.0f} deg  pitch={best['pitch_deg']:+.0f} deg  yaw={best['yaw_deg']:+.0f} deg")
    print(f"  feasible fraction of table: {study.best_value:.3f}")
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
