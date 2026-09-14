"""Custom Nature-CNN ActorCritic for rsl_rl, trained end-to-end with PPO.

rsl_rl's built-in `ActorCritic` only supports 1D observations (see the assertion in
`rsl_rl.modules.actor_critic.ActorCritic.__init__`). To train a CNN jointly with PPO,
this module defines a custom `ActorCritic` subclass that:

  1. Accepts a mix of 1D observation groups (proprioception, command) and one 4D
     image observation group.
  2. Runs the image group through a small conv net (Nature-CNN style, Mnih et al. 2015)
     to produce a flat feature vector, then concatenates it with the 1D groups.
  3. Registers the CNN as a submodule, so `self.policy.parameters()` in rsl_rl's PPO
     (rsl_rl/algorithms/ppo.py) automatically includes the CNN weights in the Adam
     optimizer -- no separate optimizer or manual gradient wiring needed.

Registration: rsl_rl's `OnPolicyRunner._construct_algorithm` resolves the policy class
via `eval(class_name)` inside `rsl_rl/runners/on_policy_runner.py`'s own module
namespace. So a custom class name is only resolvable if it has been injected into that
module's globals -- this file does that via monkey-patching at import time (see bottom).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import rsl_rl.runners.on_policy_runner as _on_policy_runner_module
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


class NatureCNN(nn.Module):
    """Small conv backbone, Nature-CNN style (Mnih et al. 2015 DQN), for small (e.g. 128x128) inputs."""

    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 540, 540)
            n_flatten = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU())

    def forward(self, image_nhwc: torch.Tensor) -> torch.Tensor:
        """image_nhwc: (N, H, W, C) float tensor, values in [0, 1]."""
        image_nchw = image_nhwc.permute(0, 3, 1, 2)
        return self.fc(self.conv(image_nchw))


class CNNActorCritic(ActorCritic):
    """ActorCritic with a learnable CNN for one image observation group.

    Not calling ActorCritic.__init__ (it asserts every obs group is 2D, which the
    image group violates). Instead we replicate the parts of it we need and route the
    image group through `NatureCNN` before concatenating with the 1D groups.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        cnn_feature_dim: int = 256,
        image_group_name: str = "image",
        **kwargs,
    ):
        nn.Module.__init__(self)
        if kwargs:
            print(f"CNNActorCritic.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")

        self.obs_groups = obs_groups
        self.image_group_name = image_group_name

        in_channels = obs[image_group_name].shape[-1]
        self.cnn = NatureCNN(in_channels=in_channels, out_dim=cnn_feature_dim)

        def flat_dim(group_names: list[str]) -> int:
            total = 0
            for group_name in group_names:
                if group_name == image_group_name:
                    total += cnn_feature_dim
                else:
                    assert len(obs[group_name].shape) == 2, (
                        f"Non-image obs group '{group_name}' must be 1D (got shape {obs[group_name].shape})."
                    )
                    total += obs[group_name].shape[-1]
            return total

        num_actor_obs = flat_dim(obs_groups["policy"])
        num_critic_obs = flat_dim(obs_groups["critic"])

        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        # [FIX] 2026-09-06: this always built nn.Identity() regardless of the
        # actor_obs_normalization flag (copy-pasted from a stub, never wired to the branch
        # that ActorCritic.__init__ uses). So actor_obs_normalization=True in
        # rsl_rl_ppo_cfg.py was silently a no-op -- critic obs (joint state, distances,
        # deviations) went into the MLP unnormalized. Suspected contributor to the
        # recurring value_function loss explosions (1e17-1e33) seen across this project's
        # history: an untrained linear critic head regressing directly on raw-scale,
        # unnormalized state features is a classic self-reinforcing PPO value divergence
        # setup, independent of any physics/contact-sensor change.
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        print(f"Actor MLP (CNN feature dim={cnn_feature_dim}): {self.actor}")

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()
        print(f"Critic MLP (CNN feature dim={cnn_feature_dim}): {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _cat_obs(self, obs, group_names: list[str]) -> torch.Tensor:
        feats = []
        for group_name in group_names:
            if group_name == self.image_group_name:
                feats.append(self.cnn(obs[group_name]))
            else:
                feats.append(obs[group_name])
        return torch.cat(feats, dim=-1)

    def get_actor_obs(self, obs):
        return self._cat_obs(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        return self._cat_obs(obs, self.obs_groups["critic"])

    # [FIX] 2026-09-06: value_function loss explosion (1e17-1e33) traced to
    # rsl_rl/algorithms/ppo.py:163 -- `reward += gamma * value * time_outs_mask`. Episodes
    # here hit time_out on ~100% of rollouts (episode length ~240 >> num_steps_per_env=24),
    # so on literally every iteration this critic's raw, unclamped `evaluate()` output is
    # injected directly into the stored "reward" for every env. One bad value prediction
    # becomes next iteration's regression target -> pushes the critic further -> feeds back
    # into the next bootstrap. Reproduced with obs normalization ALSO enabled (see
    # actor_obs_normalization/critic_obs_normalization above) -- normalization alone did not
    # stop it (still diverged by iter ~15-25 in the obsnorm_fix_smoke run), confirming the
    # loop is structural (bootstrap-of-self-output), not a scale/normalization issue.
    # Clamp range: max plausible per-step reward here is far below 16 (reaching_object<=0.8,
    # grasp/lift bonuses<=10+5 but mutually gated and not all firing every step, deviation
    # penalties small) -- so the infinite-horizon bound reward_max/(1-gamma) with a generous
    # reward_max=16, gamma=0.99 gives ~1600. +-2000 leaves headroom over any legitimate
    # return while hard-stopping the runaway before it reaches the range that overflows
    # float32 in the squared value loss (which is what actually produces grad_norm=inf).
    _VALUE_CLAMP = 2000.0

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return torch.clamp(self.critic(obs), -self._VALUE_CLAMP, self._VALUE_CLAMP)

    # [FIX] 2026-09-08: checkpoints saved before the 2026-09-06 obs-normalization fix (e.g.
    # backup_20260903_6dof_30k, backup_20260904_grasp5_30k) have no actor_obs_normalizer.*/
    # critic_obs_normalizer.* keys at all (they used nn.Identity(), which has no buffers).
    # rsl_rl's OnPolicyRunner.load() calls load_state_dict() with its hardcoded default
    # strict=True and no CLI override, so Play/resume on any pre-fix checkpoint raises
    # "Missing key(s)" before ever reaching the sim loop. Safe to relax *only* for those two
    # specific buffer prefixes: EmpiricalNormalization's untouched default (mean=0, var=1)
    # computes (x-0)/1=x -- mathematically Identity(), i.e. exactly the normalizer those
    # checkpoints actually were trained and played with. Still hard-fails on any OTHER
    # mismatch (wrong architecture, corrupt file, etc.) rather than silently accepting it.
    def load_state_dict(self, state_dict, strict=True):
        result = nn.Module.load_state_dict(self, state_dict, strict=False)
        allowed_prefixes = ("actor_obs_normalizer.", "critic_obs_normalizer.")
        unexplained_missing = [
            k for k in result.missing_keys if not k.startswith(allowed_prefixes)
        ]
        if unexplained_missing or result.unexpected_keys:
            raise RuntimeError(
                "CNNActorCritic.load_state_dict: mismatch beyond the known legacy-normalizer "
                f"gap -- missing={unexplained_missing}, unexpected={result.unexpected_keys}"
            )
        return True  # matches ActorCritic.load_state_dict's "resumes" flag contract


@configclass
class CNNActorCriticCfg(RslRlPpoActorCriticCfg):
    """Config for CNNActorCritic. Adds CNN-specific fields on top of the standard PPO policy cfg."""

    class_name: str = "CNNActorCritic"
    cnn_feature_dim: int = 256
    image_group_name: str = "image"


# register the class into rsl_rl's OnPolicyRunner module namespace so that
# `eval("CNNActorCritic")` in `OnPolicyRunner._construct_algorithm` resolves it.
_on_policy_runner_module.CNNActorCritic = CNNActorCritic
