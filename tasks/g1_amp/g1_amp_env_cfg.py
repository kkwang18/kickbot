# SPDX-License-Identifier: BSD-3-Clause
"""G1 AMP environment config.

Adapted from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env_cfg, which is hardcoded to
the generic 28-DOF HUMANOID_28_CFG skeleton. This version targets the Unitree G1
(G1_MINIMAL_CFG, 37 DOF / 44 bodies) instead.

Observation/action space sizes are computed from G1's actual DOF and key-body counts
(confirmed via runtime introspection, not guessed):
    action_space = num_dofs = 37
    observation_space = amp_observation_space
        = num_dofs (pos) + num_dofs (vel) + 1 (root height) + 6 (root orientation
          tangent+normal) + 3 (root lin vel) + 3 (root ang vel) + 3 * num_key_bodies
        = 37 + 37 + 1 + 6 + 3 + 3 + 3*4 = 99
"""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import G1_MINIMAL_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# repo layout: tasks/g1_amp/g1_amp_env_cfg.py -> repo root is two levels up
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MOTIONS_DIR = os.path.join(REPO_ROOT, "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """G1 AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces (see module docstring for derivation)
    observation_space = 99
    action_space = 37
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 99

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (G1's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot - reuse G1_MINIMAL_CFG as-is (same asset already validated by
    # Isaac-Velocity-Flat-G1-v0 / Isaac-Velocity-Rough-G1-v0), just relocated under the
    # per-env prim path. Unlike the original Humanoid-28 AMP config, we do NOT override
    # the actuators: G1_MINIMAL_CFG already ships tuned per-joint-group PD gains
    # (legs/feet/arms), which is lower risk than introducing untested gains here.
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpKickEnvCfg(G1AmpEnvCfg):
    """Placeholder task: static default-pose motion, for validating AMP plumbing on G1.

    Not a real karate-kick reference yet - swap `motion_file` for a retargeted motion
    once the video-to-3D-pose + retargeting pipeline produces one.
    """

    motion_file = os.path.join(MOTIONS_DIR, "g1_placeholder.npz")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    """Validation task: a real retargeted walk cycle (not a placeholder).

    Motion is `motions/g1_walk.npz`, produced by `retargeting/retarget_walk.py` from
    Isaac Lab's own bundled `humanoid_walk.npz` reference clip via per-frame keypoint IK
    against G1's actual kinematic tree (see `retargeting/kinematics.py`). Exists to prove
    the retargeting pipeline produces AMP-trainable motion on an easy, well-understood case
    before attempting it on real kick footage.
    """

    motion_file = os.path.join(MOTIONS_DIR, "g1_walk.npz")
