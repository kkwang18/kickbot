"""Generate a placeholder G1 reference motion (static default pose, held for a few frames).

Purpose: validate the AMP plumbing (motion loader, reset strategy, discriminator
observation shapes) end-to-end before any real retargeted motion data exists.
Not meant to produce meaningful learned behavior.

Usage:
    ./isaaclab.sh -p gen_placeholder_motion.py --out /workspace/kickbot/motions/g1_placeholder.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a placeholder G1 motion npz.")
parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
parser.add_argument("--num_frames", type=int, default=10, help="Number of (identical) frames to write.")
parser.add_argument("--fps", type=int, default=30, help="FPS to record in the motion file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import G1_MINIMAL_CFG  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    robot_cfg = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    # let the data buffers populate at the default (rest) pose
    sim_dt = sim.get_physics_dt()
    for _ in range(2):
        sim.step()
        robot.update(sim_dt)

    dof_names = robot.data.joint_names
    body_names = robot.data.body_names

    dof_pos = robot.data.joint_pos[0].cpu().numpy()  # (D,)
    dof_vel = np.zeros_like(dof_pos)

    body_pos = robot.data.body_pos_w[0].cpu().numpy()  # (B, 3)
    # re-center xy on the root so the motion is defined in a robot-local frame
    body_pos[:, 0] -= body_pos[0, 0]
    body_pos[:, 1] -= body_pos[0, 1]
    body_quat = robot.data.body_quat_w[0].cpu().numpy()  # (B, 4), wxyz
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    num_frames = args_cli.num_frames
    dof_positions = np.tile(dof_pos, (num_frames, 1)).astype(np.float32)
    dof_velocities = np.tile(dof_vel, (num_frames, 1)).astype(np.float32)
    body_positions = np.tile(body_pos, (num_frames, 1, 1)).astype(np.float32)
    body_rotations = np.tile(body_quat, (num_frames, 1, 1)).astype(np.float32)
    body_linear_velocities = np.tile(body_lin_vel, (num_frames, 1, 1)).astype(np.float32)
    body_angular_velocities = np.tile(body_ang_vel, (num_frames, 1, 1)).astype(np.float32)

    np.savez(
        args_cli.out,
        fps=np.int64(args_cli.fps),
        dof_names=np.array(dof_names),
        body_names=np.array(body_names),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"[INFO] Wrote placeholder motion to {args_cli.out}")
    print(f"[INFO] dof: {len(dof_names)}, bodies: {len(body_names)}, frames: {num_frames}, fps: {args_cli.fps}")


if __name__ == "__main__":
    main()
    simulation_app.close()
