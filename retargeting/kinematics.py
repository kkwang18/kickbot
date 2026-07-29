"""Pure-numpy/scipy forward and inverse kinematics for G1_MINIMAL_CFG.

Built from `motions/g1_info.json`, itself produced by `scripts/dump_g1_info.py` via raw
USD introspection of the actual spawned asset (not an externally-sourced URDF, so there's
no risk of it disagreeing with what Isaac Sim actually simulates). Covers all 37 DOF
(revolute) joints plus the 6 fixed joints needed to place every one of G1's 44 bodies,
notably `left_palm_link`/`right_palm_link` (the AMP task's key bodies), which attach to
their parent via a fixed joint and don't appear in the DOF list at all.

Quaternion convention: wxyz everywhere (matches this repo's motion `.npz` schema and USD's
`GfQuat.real`/`.imaginary` split), converted to/from scipy's xyzw at the boundary.

Joint-limit convention: `physics:lowerLimit`/`upperLimit` on a USD RevoluteJoint are in
*degrees* (a USD Physics schema quirk); everywhere else - `default_joint_pos`, DOF angles
passed into `forward_kinematics`, and the AMP motion `.npz` schema - angles are radians.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

AXIS_VEC = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def compose(pos_a, quat_a, pos_b, quat_b):
    """Compose two (pos, quat_wxyz) transforms: result = a * b."""
    ra = R.from_quat(wxyz_to_xyzw(quat_a))
    pos_c = pos_a + ra.apply(pos_b)
    quat_c = xyzw_to_wxyz((ra * R.from_quat(wxyz_to_xyzw(quat_b))).as_quat())
    return pos_c, quat_c


def inverse(pos, quat):
    r_inv = R.from_quat(wxyz_to_xyzw(quat)).inv()
    return -r_inv.apply(pos), xyzw_to_wxyz(r_inv.as_quat())


class G1Kinematics:
    def __init__(self, info_path: str):
        with open(info_path) as f:
            info = json.load(f)
        assert info["limit_units"] == "degrees"

        self.dof_names: list[str] = info["dof_names"]
        self.body_names: list[str] = info["body_names"]
        self.default_joint_pos = np.array(info["default_joint_pos"], dtype=np.float64)
        self.soft_lower = np.array(info["soft_joint_pos_lower"], dtype=np.float64)
        self.soft_upper = np.array(info["soft_joint_pos_upper"], dtype=np.float64)
        self.body_pos_default = {
            name: np.array(p) for name, p in zip(self.body_names, info["body_pos_w_default"])
        }
        self.body_quat_default = {
            name: np.array(q) for name, q in zip(self.body_names, info["body_quat_w_default"])
        }

        # per-DOF hard limits, in radians (source data is in degrees - see module docstring)
        self.hard_lower = np.deg2rad(np.array([j["lower_limit"] for j in info["joints"]], dtype=np.float64))
        self.hard_upper = np.deg2rad(np.array([j["upper_limit"] for j in info["joints"]], dtype=np.float64))

        # body tree: child_body_name -> edge description, root ("pelvis") has no entry
        self.tree: dict[str, dict] = {}
        for i, j in enumerate(info["joints"]):
            self.tree[j["body1_name"]] = {
                "parent": j["body0_name"],
                "kind": "revolute",
                "dof_index": i,
                "axis": j["axis"],
                "local_pos0": np.array(j["local_pos0"]),
                "local_rot0": np.array(j["local_rot0"]),
                "local_pos1": np.array(j["local_pos1"]),
                "local_rot1": np.array(j["local_rot1"]),
            }
        for j in info["fixed_joints"]:
            self.tree[j["body1_name"]] = {
                "parent": j["body0_name"],
                "kind": "fixed",
                "local_pos0": np.array(j["local_pos0"]),
                "local_rot0": np.array(j["local_rot0"]),
                "local_pos1": np.array(j["local_pos1"]),
                "local_rot1": np.array(j["local_rot1"]),
            }
        assert set(self.tree) | {"pelvis"} == set(self.body_names), (
            f"body tree doesn't cover all bodies: missing {set(self.body_names) - set(self.tree) - {'pelvis'}}"
        )

        # topological order (parents before children) for a single forward pass
        self._order: list[str] = []
        placed = {"pelvis"}
        remaining = set(self.tree)
        while remaining:
            progressed = False
            for name in list(remaining):
                if self.tree[name]["parent"] in placed:
                    self._order.append(name)
                    placed.add(name)
                    remaining.discard(name)
                    progressed = True
            assert progressed, f"cycle or missing parent in body tree, stuck on: {remaining}"

    def forward_kinematics(
        self, root_pos: np.ndarray, root_quat: np.ndarray, dof_pos: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """FK for the full body tree. dof_pos: (37,) radians, in self.dof_names order."""
        poses = {"pelvis": (root_pos, root_quat)}
        for name in self._order:
            edge = self.tree[name]
            parent_pos, parent_quat = poses[edge["parent"]]
            jf_pos, jf_quat = compose(parent_pos, parent_quat, edge["local_pos0"], edge["local_rot0"])
            if edge["kind"] == "revolute":
                theta = dof_pos[edge["dof_index"]]
                rot_joint = R.from_rotvec(theta * AXIS_VEC[edge["axis"]])
                jf_quat = xyzw_to_wxyz((R.from_quat(wxyz_to_xyzw(jf_quat)) * rot_joint).as_quat())
            inv_pos1, inv_quat1 = inverse(edge["local_pos1"], edge["local_rot1"])
            poses[name] = compose(jf_pos, jf_quat, inv_pos1, inv_quat1)
        return poses
