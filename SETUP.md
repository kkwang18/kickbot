# Environment setup & runbook

Non-obvious fixes and gotchas for this RunPod GPU instance, kept here so they
don't need to be rediscovered if the pod is rebuilt.

## Quick start on a fresh pod

Everything under `/workspace` (this repo, the IsaacLab checkout, the entire
`env_isaaclab` conda env including Isaac Sim/torch/skrl, and our `.pth` task
registration file) lives on the persistent network volume and survives pod
deletion/recreation, as long as the **same volume** is attached to the new
pod. Only container-root state (`/root`, `/usr`, `/etc`) is lost. On a fresh
pod, attach the volume, then redo just these:

```bash
# 1. Rendering libs (missing from the base container image) - see "Rendering fix" below
apt-get install -y libegl1 libglu1-mesa libxt6

# 2. GitHub auth (gh CLI itself is also container-root, not just its login state)
apt-get install -y gh   # or the full keyring-based install if this fails, see "GitHub access" below
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git

# 3. git identity
git config --global user.name "Kenny Wang"
git config --global user.email "kennywang98@gmail.com"
```

Then `conda activate env_isaaclab` and everything (including
`Isaac-G1-AMP-Kick-v0`) should work exactly as it did before. If the new pod
lands on a different physical GPU slot, you may also hit the `/dev/nvidia0`
device-numbering issue again (see "GPU" below) - the symlink workaround takes
one command if so.

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

## Filesystem notes

`/workspace` is a MooseFS network filesystem (FUSE-mounted), not local disk.
Bulk sequential I/O is fast (370MB/s+ measured), but metadata-heavy
operations (recursive `find`/`grep` over many small files, `conda activate`)
can be slow — some commands here took 5-10+ minutes for this reason. Isaac
Sim's first boot (extension loading touches thousands of small files) is
similarly slow. If a command seems stuck, check CPU time before assuming a
hang.

## Standalone script teardown hangs

Short standalone scripts (e.g. spawn-a-robot-and-print-something one-offs, not
full training runs) have reliably hung for many minutes at
`simulation_app.close()` after printing all their real output - the actual
work finishes, only Isaac Sim's shutdown is stuck. Symptom: `kill -0 <pid>`
still reports alive with the python process burning CPU, long after the
script's own print statements are done. Don't wait for the process to exit as
your completion signal for these - watch the log content for your script's
own final print statement instead, then kill the process manually. Full
training runs via `train.py` have not shown this (they exit cleanly).

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

Being importable isn't enough on its own, though — something still has to
actually run `import g1_amp` so its `gym.register(...)` call executes before
`gym.make(--task ...)` looks it up. In-tree tasks get this for free because
`isaaclab_tasks/__init__.py` auto-imports every task subpackage; our external
package doesn't hook into that. Fix: a `.pth` file in the conda env's
site-packages dir, which Python's `site` module auto-executes at interpreter
startup for any line starting with `import`:

```
# /workspace/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/kickbot_tasks.pth
import sys; sys.path.insert(0, "/workspace/kickbot/tasks") if "/workspace/kickbot/tasks" not in sys.path else None; import g1_amp
```

This means `Isaac-G1-AMP-Kick-v0` is registered automatically in every
`isaaclab.sh -p ...` invocation in this env, with zero vendored-repo changes.
It's outside both this git repo and the IsaacLab checkout — lives in the
conda env itself at `.../env_isaaclab/lib/python3.11/site-packages/` — but
since that whole conda env is under `/workspace`, it's on the persistent
volume and survives a pod rebuild along with everything else there. No action
needed on a fresh pod as long as the same volume is attached.

## skrl AMP config version skew

The installed `skrl` (2.1.0) uses a newer dataclass-based `AMP_CFG` than the
one IsaacLab's own shipped example
(`isaaclab_tasks/direct/humanoid_amp/agents/skrl_walk_amp_cfg.yaml`) targets.
Several `agent:` fields were renamed/removed (e.g. `amp_state_preprocessor` ->
`amp_observation_preprocessor`, `lambda` -> `gae_lambda`,
`style_reward_weight` + `discriminator_reward_scale` merged into a single
`style_reward_scale`), and `clip_predicted_values` was removed entirely.
`tasks/g1_amp/agents/skrl_amp_cfg.yaml` has the corrected field names for this
skrl version. IsaacLab's own bundled `Isaac-Humanoid-AMP-Walk-Direct-v0` task
would hit the same `TypeError: AMP_CFG.__init__() got an unexpected keyword
argument` if run against this environment - it's a version-skew bug in the
shipped repo/env combo, not specific to our G1 port. Worth re-checking if
Isaac Lab or skrl gets upgraded on this box.
