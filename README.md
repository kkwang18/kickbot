# kickbot

Motion-imitation training pipeline for a Unitree G1 humanoid performing a karate kick in Isaac Lab, using a mocap/video-derived reference motion.

## Environment

Runs against an existing Isaac Lab checkout at `/workspace/IsaacLab` (v2.3.0, conda env `env_isaaclab`). See `/workspace/isaaclab_commands.txt` on the training box for activation/run commands.

## Layout

- `retargeting/` — video → 3D pose → G1 skeleton retargeting pipeline
- `motions/` — reference motion `.npz` files (IsaacLab AMP schema: dof/body names, positions, rotations, velocities, fps)
- `tasks/` — IsaacLab task configs (G1 + AMP env, rewards, domain randomization)
- `scripts/` — training/eval entry points

## Status

Project scaffolding only — pipeline not yet implemented.
