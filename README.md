# kickbot

Motion-imitation training pipeline for a Unitree G1 humanoid performing a karate kick in Isaac Lab, using a mocap/video-derived reference motion.

## Environment

Runs against an existing Isaac Lab checkout at `/workspace/IsaacLab` (v2.3.0, conda env `env_isaaclab`). See `SETUP.md` for environment fixes/gotchas and a fresh-pod quick start.

## Layout

- `retargeting/` — video → 3D pose → G1 skeleton retargeting pipeline
- `motions/` — reference motion `.npz` files (IsaacLab AMP schema: dof/body names, positions, rotations, velocities, fps)
- `tasks/` — IsaacLab task configs (G1 + AMP env, rewards, domain randomization)
- `scripts/` — training/eval entry points

## Status

See `PROGRESS.md` for what's done and what's next. Short version: Isaac Lab's
AMP task is ported and validated on G1 with a placeholder reference motion;
real motion data (retargeted AMASS or the karate-kick video) is next.
