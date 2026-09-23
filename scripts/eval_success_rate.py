# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

from isaaclab_tasks.manager_based.manipulation.lift import mdp

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    success_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    total_episodes = torch.zeros(num_envs, dtype=torch.long, device=device)
    total_successes = 0
    # 색상별 카운트 (episode 끝나는 시점의 target_color 기준)
    total_successes_red = 0
    total_successes_blue = 0
    total_episodes_red = 0
    total_episodes_blue = 0
    # place 성공 조건: target 큐브가 goal position으로부터 이 거리 이내
    place_success_threshold = 0.15
    lift_success_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    place_success_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    total_lift_successes = 0
    total_place_successes = 0
    total_overall_successes = 0
    target_eval_steps = 5000

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        is_lifted = mdp.object_is_lifted_target_aware(env.unwrapped, minimal_height=0.15).bool()
        success_this_episode |= is_lifted
        lift_success_this_episode |= is_lifted

        # place 조건: target color 큐브 위치와 목표 위치(object_pose command) 사이 거리
        target_color = env.unwrapped.command_manager.get_command("target_color")
        object_rb = env.unwrapped.scene["object"]
        distractor_rb = env.unwrapped.scene["distractor"]
        robot = env.unwrapped.scene["robot"]
        command = env.unwrapped.command_manager.get_command("object_pose")
        from isaaclab.utils.math import combine_frame_transforms
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
        target_pos = torch.where(
            (target_color == 0).unsqueeze(-1),
            object_rb.data.root_pos_w,
            distractor_rb.data.root_pos_w,
        )
        place_dist = torch.norm(target_pos - des_pos_w, dim=1)
        is_placed = (place_dist < place_success_threshold) & is_lifted
        place_success_this_episode |= is_placed

        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            target_color = env.unwrapped.command_manager.get_command("target_color")
            done_colors = target_color[done_ids]
            done_success = success_this_episode[done_ids]

            total_successes += done_success.sum().item()
            total_episodes[done_ids] += 1

            done_lift = lift_success_this_episode[done_ids]
            done_place = place_success_this_episode[done_ids]
            total_lift_successes += done_lift.sum().item()
            total_place_successes += done_place.sum().item()
            total_overall_successes += (done_lift & done_place).sum().item()
            lift_success_this_episode[done_ids] = False
            place_success_this_episode[done_ids] = False

            red_mask = done_colors == 0
            blue_mask = done_colors == 1
            total_episodes_red += red_mask.sum().item()
            total_episodes_blue += blue_mask.sum().item()
            total_successes_red += done_success[red_mask].sum().item()
            total_successes_blue += done_success[blue_mask].sum().item()

            success_this_episode[done_ids] = False

        timestep += 1
        if timestep >= target_eval_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    total_eps = total_episodes.sum().item()
    rate = 100.0 * total_successes / max(total_eps, 1)
    rate_red = 100.0 * total_successes_red / max(total_episodes_red, 1)
    rate_blue = 100.0 * total_successes_blue / max(total_episodes_blue, 1)
    print(f"SUCCESS_RATE: {total_successes}/{total_eps} = {rate:.2f}%")
    print(f"SUCCESS_RATE_RED: {total_successes_red}/{total_episodes_red} = {rate_red:.2f}%")
    print(f"SUCCESS_RATE_BLUE: {total_successes_blue}/{total_episodes_blue} = {rate_blue:.2f}%")
    lift_rate = 100.0 * total_lift_successes / max(total_eps, 1)
    place_rate = 100.0 * total_place_successes / max(total_eps, 1)
    overall_rate = 100.0 * total_overall_successes / max(total_eps, 1)
    print(f"LIFT_RATE: {total_lift_successes}/{total_eps} = {lift_rate:.2f}%")
    print(f"PLACE_RATE: {total_place_successes}/{total_eps} = {place_rate:.2f}%")
    print(f"OVERALL_RATE: {total_overall_successes}/{total_eps} = {overall_rate:.2f}%")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
