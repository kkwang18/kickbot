# Environment setup & runbook

Non-obvious fixes and gotchas for this RunPod GPU instance, kept here so they
don't need to be rediscovered if the pod is rebuilt.

## Isaac Lab install

- Isaac Lab v2.3.0 at `/workspace/IsaacLab` (git repo, origin `isaac-sim/IsaacLab`)
- Conda env `env_isaaclab` (Python 3.11.15) in `/workspace/miniconda3`
- Isaac Sim 5.1.0.0 (pip-installed), Torch 2.7.0+cu128
- Activate: `source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
- Run scripts via `./isaaclab.sh -p <script.py>` from `/workspace/IsaacLab`

## GPU

RTX 4090, 24GB VRAM. Host is multi-GPU; this pod is allocated the host's
second physical GPU slot (device minor 1) — `/dev/nvidia0` does not exist in
this container by design (that GPU belongs to another tenant), only
`/dev/nvidia1` exists. NVML/CUDA renumber the visible GPU as index 0
internally, so `nvidia-smi`/PyTorch are unaffected by this.

## Rendering fix (missing libraries)

A fresh container was missing standard X11/GL/EGL userspace libraries. This
broke Isaac Sim's Vulkan-based renderer with `ERROR_INCOMPATIBLE_DRIVER` on
`vkCreateInstance` any time `--video` or camera rendering was requested.
Plain headless (no rendering, e.g. `list_envs.py`) was unaffected — only the
rendering-enabled kit profile hit this. Root cause (confirmed via `strace`):
`libEGL.so.1` was missing entirely, which the NVIDIA Vulkan ICD
(`libGLX_nvidia.so`) depends on internally during its own init.

Fix:

```bash
apt-get install -y libegl1 libglu1-mesa libxt6
```

This is a container-level fix (not scoped to the conda env) — if this pod is
rebuilt from its base image, reinstall these before using `--video` or any
camera-based task.

## GitHub access

- Auth: `gh` CLI (installed via apt, not preinstalled), logged in as
  `kkwang18` via device-flow login:
  `gh auth login --hostname github.com --git-protocol https --web`.
  `gh auth setup-git` wires it up as the git credential helper — `git push`
  works with no further config.
- An SSH keypair was also generated (`~/.ssh/id_ed25519`) as an alternative
  but is unused — `gh`'s HTTPS auth is what's actually active.
- git identity (set globally): `user.name = Kenny Wang`,
  `user.email = kennywang98@gmail.com`.

## File transfer to local laptop

- `runpodctl send <file>` on the pod → `runpodctl receive <code>` locally.
  Relays through RunPod's servers, no SSH setup needed. Requires
  `runpodctl` installed locally too (`brew install runpod/runpodctl/runpodctl`
  on Mac, or grab the binary from RunPod's GitHub releases).
- Direct `scp` also works if your laptop's SSH key is already authorized on
  the pod: `scp -P $RUNPOD_TCP_PORT_22 root@$RUNPOD_PUBLIC_IP:<path> .` (both
  env vars are set inside the pod; look them up fresh each time since they
  change if the pod is recreated).

## Filesystem notes

`/workspace` is a MooseFS network filesystem (FUSE-mounted), not local disk.
Bulk sequential I/O is fast (370MB/s+ measured), but metadata-heavy
operations (recursive `find`/`grep` over many small files, `conda activate`)
can be slow — some commands here took 5-10+ minutes for this reason. Isaac
Sim's first boot (extension loading touches thousands of small files) is
similarly slow. If a command seems stuck, check CPU time before assuming a
hang.

## G1 asset

- Isaac Lab ships G1 configs in `isaaclab_assets.robots.unitree`: `G1_CFG`,
  `G1_MINIMAL_CFG`, `G1_29DOF_CFG`, `G1_INSPIRE_FTP_CFG`. The USD streams from
  NVIDIA's Nucleus cloud asset server on first spawn (needs outbound network
  access — confirmed working from this pod).
- This project standardizes on `G1_MINIMAL_CFG`: it's what Isaac Lab's own
  already-validated `Isaac-Velocity-Flat-G1-v0` / `Isaac-Velocity-Rough-G1-v0`
  tasks use (same DOF/joint layout as `G1_CFG`, fewer collision meshes for
  simulation speed).

## Task registration

Isaac Lab supports external task registration: any importable Python package
that calls `gym.register(...)` works with `--task` exactly like an in-tree
task. This project's task code lives in `tasks/` in this repo rather than
inside the vendored `/workspace/IsaacLab` source tree, which tracks the
`isaac-sim/IsaacLab` upstream and shouldn't carry project-specific code.
