from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TargetColorCommand(CommandTerm):
    """Episode마다 target 색상(0=red, 1=blue)을 랜덤 샘플링하는 command."""

    cfg: TargetColorCommandCfg

    def __init__(self, cfg: TargetColorCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.target_color_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.long, 
            device=self.device
        )

    def __str__(self) -> str:
        return f"TargetColorCommand:\n\tResampling time range: {self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        # command_manager.get_command()가 반환할 값. (num_envs,) 텐서면 됩니다.
    
        return self.target_color_buf

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.target_color_buf[env_ids] = torch.randint(
            low=0,
            high=2, 
            size=(len(env_ids),),
            device=self.device
        )
      
    def _update_command(self):
        pass


@configclass
class TargetColorCommandCfg(CommandTermCfg):
    class_type: type = TargetColorCommand