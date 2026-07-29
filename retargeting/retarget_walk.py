"""Retarget Isaac Lab's bundled humanoid_walk.npz (generic 28-DOF Humanoid, 15 body
keypoints) onto G1_MINIMAL_CFG via per-frame keypoint IK, producing an Isaac Lab AMP
motion .npz for the g1_amp task.

Why this source motion: it's the exact clip Isaac Lab's own humanoid_amp example uses to
validate AMP training, so it's a known-good, license-clean walk cycle - good for
validating the *retargeting pipeline* before attempting anything harder (a kick).

Approach: map each of the source's 15 body keypoints to its G1 equivalent (chosen so
"body origin at proximal joint" holds for both skeletons - see BODY_MAP below), scale
by G1/source leg-length ratio, then solve per-frame IK for G1's ~21 "posture-relevant"
DOF (legs, torso, shoulders+elbow-pitch) against the pure-numpy FK model in kinematics.py.
Fingers and elbow-roll (forearm twist, negligible effect on hand *position*) are held at
their default pose throughout - the source skeleton has no hand/finger data anyway.

Usage:
    python retarget_walk.py --out /workspace/kickbot/motions/g1_walk.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from kinematics import G1Kinematics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

SOURCE_NPZ = "/workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz"
INFO_JSON = "/workspace/kickbot/motions/g1_info.json"

# source (Humanoid-28) body name -> G1 body name. Both skeletons place a link's origin
# at its proximal joint (confirmed for G1 via g1_info.json: local_pos1 is [0,0,0] for
# every joint), so this is a like-for-like mapping, not a guess.
BODY_MAP = {
    "torso": "torso_link",
    "head": "head_link",
    "right_upper_arm": "right_shoulder_yaw_link",
    "right_lower_arm": "right_elbow_roll_link",
    "right_hand": "right_palm_link",
    "left_upper_arm": "left_shoulder_yaw_link",
    "left_lower_arm": "left_elbow_roll_link",
    "left_hand": "left_palm_link",
    "right_thigh": "right_hip_yaw_link",
    "right_shin": "right_knee_link",
    "right_foot": "right_ankle_roll_link",
    "left_thigh": "left_hip_yaw_link",
    "left_shin": "left_knee_link",
    "left_foot": "left_ankle_roll_link",
}

# DOFs solved by IK per frame (posture-relevant). Excluded: elbow_roll (forearm twist,
# doesn't move the hand target) and all finger joints (no source data for them) - these
# stay at default_joint_pos the whole clip.
ACTIVE_DOF_SUFFIXES = [
    "hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint",
    "ankle_pitch_joint", "ankle_roll_joint",
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_pitch_joint",
    "torso_joint",
]


def quat_apply(q_wxyz, v):
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).apply(v)


def quat_inv_apply(q_wxyz, v):
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).inv().apply(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--source", type=str, default=SOURCE_NPZ)
    parser.add_argument("--info", type=str, default=INFO_JSON)
    args = parser.parse_args()

    kin = G1Kinematics(args.info)
    src = np.load(args.source)
    src_names = list(src["body_names"])
    src_pos = src["body_positions"].astype(np.float64)  # (N, 15, 3)
    src_quat = src["body_rotations"].astype(np.float64)  # (N, 15, 4) wxyz
    fps = int(src["fps"])
    num_frames = src_pos.shape[0]
    src_pelvis_idx = src_names.index("pelvis")

    # --- scale: G1 leg length / source leg length, from a single ankle+pelvis pair ---
    g1_leg_len = np.linalg.norm(kin.body_pos_default["pelvis"] - kin.body_pos_default["right_ankle_roll_link"])
    src_leg_len = float(
        np.median(np.linalg.norm(src_pos[:, src_pelvis_idx] - src_pos[:, src_names.index("right_foot")], axis=-1))
    )
    scale = g1_leg_len / src_leg_len
    print(f"[INFO] G1 leg length={g1_leg_len:.4f}  source leg length={src_leg_len:.4f}  scale={scale:.4f}")

    active_dof_idx = [
        i for i, n in enumerate(kin.dof_names) if any(n.endswith(suf) for suf in ACTIVE_DOF_SUFFIXES)
    ]
    print(f"[INFO] {len(active_dof_idx)} active IK DOFs: {[kin.dof_names[i] for i in active_dof_idx]}")

    target_pairs = [(src_names.index(src_name), g1_name) for src_name, g1_name in BODY_MAP.items()]

    lower = kin.hard_lower[active_dof_idx]
    upper = kin.hard_upper[active_dof_idx]

    # Small temporal-regularization weight: penalize deviating from the previous frame's
    # solution, not just position error. Without this, least_squares (even warm-started)
    # occasionally jumps to a different but equally-valid arm configuration frame-to-frame
    # (e.g. an elbow-up/elbow-down style flip), producing multi-radian single-frame
    # discontinuities - caught by a velocity sanity check (>70 rad/s spikes at the shoulder).
    REG_WEIGHT = 0.3

    def residuals(x_active, root_pos, root_quat, targets_world, full_dof, x_prev):
        full_dof = full_dof.copy()
        full_dof[active_dof_idx] = x_active
        poses = kin.forward_kinematics(root_pos, root_quat, full_dof)
        res = []
        for g1_name, target in targets_world:
            pred_pos, _ = poses[g1_name]
            res.append(pred_pos - target)
        res.append(REG_WEIGHT * (x_active - x_prev))
        return np.concatenate(res)

    dof_positions = np.tile(kin.default_joint_pos, (num_frames, 1))
    root_pos_out = np.zeros((num_frames, 3))
    root_quat_out = np.zeros((num_frames, 4))

    g1_default_pelvis = kin.body_pos_default["pelvis"]
    src_pelvis0 = src_pos[0, src_pelvis_idx].copy()

    x0 = kin.default_joint_pos[active_dof_idx].copy()
    for t in range(num_frames):
        root_pos = g1_default_pelvis + scale * (src_pos[t, src_pelvis_idx] - src_pelvis0)
        root_quat = src_quat[t, src_pelvis_idx]
        root_quat_out[t] = root_quat
        root_pos_out[t] = root_pos

        targets_world = []
        for src_idx, g1_name in target_pairs:
            local_offset = scale * quat_inv_apply(root_quat, src_pos[t, src_idx] - src_pos[t, src_pelvis_idx])
            targets_world.append((g1_name, root_pos + quat_apply(root_quat, local_offset)))

        full_dof = kin.default_joint_pos.copy()
        sol = least_squares(
            residuals, x0, args=(root_pos, root_quat, targets_world, full_dof, x0),
            bounds=(lower, upper), method="trf", xtol=1e-8, ftol=1e-8,
        )
        x0 = sol.x  # warm start + regularization anchor for next frame
        dof_positions[t, active_dof_idx] = sol.x
        if t % 20 == 0 or t == num_frames - 1:
            print(f"[INFO] frame {t:3d}/{num_frames}: cost={sol.cost:.6f}")

    # --- finite-difference velocities ---
    dt = 1.0 / fps
    dof_velocities = np.gradient(dof_positions, dt, axis=0)

    # --- full-body FK for all 44 bodies, every frame ---
    body_positions = np.zeros((num_frames, len(kin.body_names), 3))
    body_rotations = np.zeros((num_frames, len(kin.body_names), 4))
    for t in range(num_frames):
        poses = kin.forward_kinematics(root_pos_out[t], root_quat_out[t], dof_positions[t])
        for i, name in enumerate(kin.body_names):
            body_positions[t, i], body_rotations[t, i] = poses[name]

    body_linear_velocities = np.gradient(body_positions, dt, axis=0)
    # angular velocity via finite-difference quaternion derivative: omega = 2 * vec(dq/dt * conj(q))
    body_angular_velocities = np.zeros_like(body_positions)
    for i in range(len(kin.body_names)):
        q = body_rotations[:, i]
        dq = np.gradient(q, dt, axis=0)
        for t in range(num_frames):
            q_conj = q[t] * np.array([1, -1, -1, -1])
            # quaternion product dq * conj(q), wxyz
            w1, x1, y1, z1 = dq[t]
            w2, x2, y2, z2 = q_conj
            prod = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])
            body_angular_velocities[t, i] = 2.0 * prod[1:]

    np.savez(
        args.out,
        fps=np.int64(fps),
        dof_names=np.array(kin.dof_names),
        body_names=np.array(kin.body_names),
        dof_positions=dof_positions.astype(np.float32),
        dof_velocities=dof_velocities.astype(np.float32),
        body_positions=body_positions.astype(np.float32),
        body_rotations=body_rotations.astype(np.float32),
        body_linear_velocities=body_linear_velocities.astype(np.float32),
        body_angular_velocities=body_angular_velocities.astype(np.float32),
    )
    print(f"[INFO] Wrote {args.out} ({num_frames} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
