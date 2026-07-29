"""Dump G1_MINIMAL_CFG's full kinematic model to JSON: joint/body names, default pose,
soft joint limits, and (via raw USD introspection) each revolute joint's axis, local
frames in its parent/child body, and hard limits, plus the body parent/child tree.

This is a one-off introspection script (not part of the training pipeline) so the
retargeting pipeline can do forward/inverse kinematics for G1 in pure numpy, without
needing Isaac Sim in the loop and without depending on any externally-sourced URDF that
might not exactly match the USD asset actually spawned by G1_MINIMAL_CFG.

Approach: `root_physx_view.dof_paths[0]` gives the USD prim path of each DOF in the
exact order of `robot.data.joint_names` / the DOF tensors. For each such path, casting
the prim as `UsdPhysics.RevoluteJoint` exposes GetAxisAttr/GetLocalPos0Attr/
GetLocalRot0Attr/GetLocalPos1Attr/GetLocalRot1Attr/limits, and the base `UsdPhysics.Joint`
exposes body0/body1 relationship targets (the parent/child body prim paths). Axis is in
the joint's local frame (i.e. after localRot0/localRot1), not directly the body frame.

Usage:
    ./isaaclab.sh -p dump_g1_info.py --out /workspace/kickbot/motions/g1_info.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump G1_MINIMAL_CFG kinematic model.")
parser.add_argument("--out", type=str, required=True, help="Output JSON path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import omni.usd
import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import G1_MINIMAL_CFG  # isort:skip


def main():
    # Gravity off: this is a *kinematic* bind-pose dump, not a dynamics test. With gravity
    # on, PD control has real (physically correct) nonzero steady-state droop under load
    # (e.g. hip_pitch carries the whole upper body against stiffness=200) even when the
    # exact target is commanded - confirmed by cross-checking against a hand-computed FK
    # from the extracted joint frames, which reproduced the drooped pose, not the commanded
    # one. Disabling gravity here removes the disturbance torque so PD converges exactly to
    # the commanded default_joint_pos, giving a clean ground truth for validating kinematics.py.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    robot_cfg = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    # Directly WRITE the default joint state (not just command it as a PD target) so there's
    # no spring-damper transient to settle: 2 physics steps turned out not to be enough time
    # for stiffness/damping like 200/5 to travel from reset()'s initial state to a commanded
    # target (confirmed by disabling gravity, which barely changed the residual error -
    # ruling out steady-state droop and pointing at an unsettled transient instead). Writing
    # state directly, plus also holding it as the PD target so it doesn't immediately start
    # drifting away again, removes the transient entirely.
    default_pos = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(default_pos)
    robot.write_joint_state_to_sim(default_pos, default_vel)
    robot.set_joint_position_target(default_pos)
    robot.write_data_to_sim()
    sim_dt = sim.get_physics_dt()
    for _ in range(2):
        sim.step()
        robot.update(sim_dt)

    dof_names = robot.data.joint_names
    body_names = robot.data.body_names
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy().tolist()
    soft_lower = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy().tolist()
    soft_upper = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy().tolist()
    body_pos_default = robot.data.body_pos_w[0].cpu().numpy().tolist()
    body_quat_default = robot.data.body_quat_w[0].cpu().numpy().tolist()  # wxyz

    stage = omni.usd.get_context().get_stage()
    dof_paths = list(robot.root_physx_view.dof_paths[0])
    print(f"[INFO] dof_paths[0] ({len(dof_paths)}): {dof_paths}")

    def body_leaf_name(sdf_path: str) -> str:
        return str(sdf_path).rsplit("/", 1)[-1]

    joints = []
    for i, path in enumerate(dof_paths):
        prim = stage.GetPrimAtPath(path)
        base = UsdPhysics.Joint(prim)
        rev = UsdPhysics.RevoluteJoint(prim)
        body0_targets = base.GetBody0Rel().GetTargets()
        body1_targets = base.GetBody1Rel().GetTargets()
        joints.append(
            {
                "dof_index": i,
                "dof_name": dof_names[i],
                "prim_path": str(path),
                "prim_type": prim.GetTypeName(),
                "body0": str(body0_targets[0]) if body0_targets else None,
                "body1": str(body1_targets[0]) if body1_targets else None,
                "body0_name": body_leaf_name(body0_targets[0]) if body0_targets else None,
                "body1_name": body_leaf_name(body1_targets[0]) if body1_targets else None,
                "axis": rev.GetAxisAttr().Get() if rev else None,
                "local_pos0": list(base.GetLocalPos0Attr().Get()) if base.GetLocalPos0Attr().Get() else None,
                # stored as wxyz, matching the convention used everywhere else in this repo
                "local_rot0": [base.GetLocalRot0Attr().Get().real] + list(base.GetLocalRot0Attr().Get().imaginary)
                if base.GetLocalRot0Attr().Get()
                else None,
                "local_pos1": list(base.GetLocalPos1Attr().Get()) if base.GetLocalPos1Attr().Get() else None,
                "local_rot1": [base.GetLocalRot1Attr().Get().real] + list(base.GetLocalRot1Attr().Get().imaginary)
                if base.GetLocalRot1Attr().Get()
                else None,
                "lower_limit": rev.GetLowerLimitAttr().Get() if rev else None,
                "upper_limit": rev.GetUpperLimitAttr().Get() if rev else None,
            }
        )

    # 37 DOF joints only cover 37 of the 44 bodies (+ pelvis root = 38). The rest
    # (e.g. palm_link, head_link, imu_link, logo_link, pelvis_contour_link) attach via
    # fixed (0-DOF) joints, which don't appear in root_physx_view.dof_paths. Walk the
    # full stage under the robot prim to find those too, so the FK tree covers all 44
    # bodies - notably left/right_palm_link, which are the AMP task's key bodies.
    covered_paths = {str(p) for p in dof_paths}
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    fixed_joints = []
    for prim in Usd.PrimRange(robot_prim):
        if str(prim.GetPath()) in covered_paths:
            continue
        if not prim.IsA(UsdPhysics.Joint):
            continue
        base = UsdPhysics.Joint(prim)
        body0_targets = base.GetBody0Rel().GetTargets()
        body1_targets = base.GetBody1Rel().GetTargets()
        fixed_joints.append(
            {
                "prim_path": str(prim.GetPath()),
                "prim_type": prim.GetTypeName(),
                "body0": str(body0_targets[0]) if body0_targets else None,
                "body1": str(body1_targets[0]) if body1_targets else None,
                "body0_name": body_leaf_name(body0_targets[0]) if body0_targets else None,
                "body1_name": body_leaf_name(body1_targets[0]) if body1_targets else None,
                "local_pos0": list(base.GetLocalPos0Attr().Get()) if base.GetLocalPos0Attr().Get() else None,
                "local_rot0": [base.GetLocalRot0Attr().Get().real] + list(base.GetLocalRot0Attr().Get().imaginary)
                if base.GetLocalRot0Attr().Get()
                else None,
                "local_pos1": list(base.GetLocalPos1Attr().Get()) if base.GetLocalPos1Attr().Get() else None,
                "local_rot1": [base.GetLocalRot1Attr().Get().real] + list(base.GetLocalRot1Attr().Get().imaginary)
                if base.GetLocalRot1Attr().Get()
                else None,
            }
        )

    info = {
        "dof_names": dof_names,
        "body_names": body_names,
        "default_joint_pos": default_joint_pos,
        "soft_joint_pos_lower": soft_lower,
        "soft_joint_pos_upper": soft_upper,
        "body_pos_w_default": body_pos_default,
        "body_quat_w_default": body_quat_default,
        "joints": joints,
        "fixed_joints": fixed_joints,
        # limit_units clarifies a real gotcha: RevoluteJoint limits are in degrees per
        # USD Physics schema convention, while joint angles/dof_positions everywhere
        # else in this repo (robot.data.joint_pos, the AMP motion npz schema) are radians.
        "limit_units": "degrees",
    }
    with open(args_cli.out, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[INFO] dof_names ({len(dof_names)}): {dof_names}")
    print(f"[INFO] body_names ({len(body_names)}): {body_names}")
    for j in joints[:3]:
        print(f"[INFO] sample joint: {j}")
    print(f"[INFO] fixed_joints ({len(fixed_joints)}): {[f['body1_name'] for f in fixed_joints]}")
    for j in fixed_joints:
        print(f"[INFO] fixed joint: {j}")
    covered_bodies = {"pelvis"} | {j["body1_name"] for j in joints} | {j["body1_name"] for j in fixed_joints}
    missing = set(body_names) - covered_bodies
    print(f"[INFO] bodies covered: {len(covered_bodies)}/{len(body_names)}, missing: {missing}")
    print(f"[INFO] Wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
