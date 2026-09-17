# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_lifted_sustained(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    min_steps: int,
    max_ee_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    left_contact_sensor_name: str = "left_finger_contact",
    right_contact_sensor_name: str = "right_finger_contact",
    min_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward held only after the object stays above `minimal_height` AND near the end-effector
    AND in contact with at least one finger, for `min_steps` consecutive control steps.

    `object_is_lifted` alone fires on height only. In this task the object's resting height
    (see the object RigidObjectCfg init_state, z=0.055) is already slightly above the bonus
    threshold used elsewhere (minimal_height=0.05), so a height-only check can be nearly always
    true even for an untouched object -- adding an EE-proximity requirement and a consecutive-step
    streak rules out that case and rejects momentary contact bounces, which decay within a step or
    two once contact ends. The streak buffer is lazily attached to `env`; it self-resets on episode
    reset because right after reset the EE is at the fixed home pose (far from the newly-sampled
    object position), so `held` is False on the first post-reset step -- no explicit reset hook needed.

    [FIX] 2026-09-07: height+EE-proximity alone doesn't require the object to actually be
    touched. Live diagnostic on the iter-20500 checkpoint (grasp_contact_bonus stuck at 0.0000
    for 20000+ iterations despite this bonus firing regularly) showed force on both finger
    contact sensors reading exactly 0.0000 against the object *while the object was airborne and
    within max_ee_distance* -- i.e. the policy was satisfying this term by knocking/flinging the
    cube near the EE, not by holding it; a real touch (35.9N, against something else) preceded
    the airborne window but had already ended by the time `held` would have fired. Ballistic
    flight has zero object-finger contact force throughout, by construction, so requiring
    *current-step* contact from at least one finger closes this loophole without reproducing
    grasp_contact_bonus's much stricter bilateral condition (both fingers >= force_threshold
    simultaneously) -- that bonus targets grasp *quality*, this one only needs to rule out
    "untouched". min_contact_force=0.05N sits far below grasp_contact_reward's force_threshold
    (0.5N, tuned for "confident bilateral squeeze") and 700x below the 35.9N contact force
    actually measured live -- enough margin over sensor/solver noise (measured exactly 0.0 when
    truly untouched) without approaching a "real grip" bar this term isn't meant to enforce.
    Uses OR (either finger), not AND: this only needs to prove the object is being influenced by
    the gripper, not that both sides are engaged -- that stricter bar stays grasp_contact_bonus's
    job.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    left_sensor: ContactSensor = env.scene.sensors[left_contact_sensor_name]
    right_sensor: ContactSensor = env.scene.sensors[right_contact_sensor_name]
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    left_force = torch.norm(left_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    right_force = torch.norm(right_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    finger_contact = (left_force > min_contact_force) | (right_force > min_contact_force)

    held = (object_pos_w[:, 2] > minimal_height) & (ee_distance < max_ee_distance) & finger_contact

    if not hasattr(env, "_lift_streak_buf") or env._lift_streak_buf.shape[0] != env.num_envs:
        env._lift_streak_buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._lift_streak_buf = torch.where(
        held, env._lift_streak_buf + 1, torch.zeros_like(env._lift_streak_buf)
    )
    return (env._lift_streak_buf >= min_steps).float()


def grasp_proximity_reward(
    env: ManagerBasedRLEnv,
    max_distance: float,
    gripper_close_range: tuple[float, float],
    max_object_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["finger_joint"]),
) -> torch.Tensor:
    """Shaping reward for grasp *mechanics*, separate from the lift outcome.

    Fires (1.0) when all three hold simultaneously, else 0.0:
      1. object is within `max_distance` of the EE (near enough to be between the fingers)
      2. `finger_joint` position is inside `gripper_close_range` (partially-to-mostly closed,
         not wide open and not slammed fully shut on nothing)
      3. object linear speed is below `max_object_speed` (resting/held steady, not being
         knocked around)

    Unlike `object_is_lifted_sustained` this doesn't require height, so it can reward "fingers
    closed around the object, holding it still on the table" as a distinct, earlier skill from
    "lift it" -- meant to bridge the gap noted in Play mode: the policy learned to reach and curl
    upward without ever closing the gripper around the cube.

    `gripper_cfg.joint_ids` is resolved by name via the manager (matches the `joint_deviation_l1`
    pattern used elsewhere in this task) -- no hardcoded joint index.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    gripper_closed = (finger_pos > gripper_close_range[0]) & (finger_pos < gripper_close_range[1])

    object_speed = torch.norm(object.data.root_lin_vel_w, dim=1)

    close_enough = ee_distance < max_distance
    stable = object_speed < max_object_speed

    return (close_enough & gripper_closed & stable).float()


def grasp_contact_reward(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    max_object_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_contact_sensor_name: str = "left_finger_contact",
    right_contact_sensor_name: str = "right_finger_contact",
) -> torch.Tensor:
    """Bilateral-contact grasp detection, replacing the exploitable `grasp_proximity_reward` heuristic.

    2026-09-04: 30000iter 실측(peak bonus 0.88, grasp>0.01 82회)에서 Play mode 관찰 결과 정책이 실제
    grasp 대신 "밀어서 정지시키기(pushing)"를 학습함. `grasp_proximity_reward`의 `max_object_speed`
    조건(물체가 안 움직이면 통과)은 "잡혀서 정지"와 "밀려서 벽/테이블에 눌려 정지"를 구별 못 함 --
    두 경우 다 물체 속도가 0에 가까워짐. 이 함수는 그 대신 왼쪽/오른쪽 finger 각각에 붙인
    `ContactSensorCfg`(object 필터)의 실측 접촉력을 사용, "양쪽 finger가 동시에 물체에 힘을 가하고
    있음"을 요구함 -- 한쪽에서 미는 동작은 원천적으로 한쪽 finger에서만 힘이 잡히므로 이 조건을
    만족할 수 없음. dexsuite_kuka_allegro의 `contacts()` 패턴(다중 finger 동시 접촉 요구)을 그대로
    이식.

    Fires (1.0) when both hold simultaneously, else 0.0:
      1. both `left_contact_sensor_name` and `right_contact_sensor_name` report contact force
         magnitude above `force_threshold` against `object_cfg` (bilateral squeeze, not one-sided push)
      2. object linear speed is below `max_object_speed` (held steady, not being knocked around)
    """
    object: RigidObject = env.scene[object_cfg.name]
    left_sensor: ContactSensor = env.scene.sensors[left_contact_sensor_name]
    right_sensor: ContactSensor = env.scene.sensors[right_contact_sensor_name]

    left_force = torch.norm(left_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    right_force = torch.norm(right_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    bilateral_contact = (left_force > force_threshold) & (right_force > force_threshold)

    object_speed = torch.norm(object.data.root_lin_vel_w, dim=1)
    stable = object_speed < max_object_speed

    return (bilateral_contact & stable).float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_is_lifted_target_aware(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distractor_cfg: SceneEntityCfg = SceneEntityCfg("distractor"),
) -> torch.Tensor:
    """target_color command에 따라 object 또는 distractor 중 실제 target인 쪽의 높이를 보고 lift 여부 판단."""
    target_color = env.command_manager.get_command("target_color")  # (num_envs,), 0=red(object) or 1=blue(distractor)
    object: RigidObject = env.scene[object_cfg.name]
    distractor: RigidObject = env.scene[distractor_cfg.name]

    object_height = object.data.root_pos_w[:, 2]
    distractor_height = distractor.data.root_pos_w[:, 2]

    target_height = torch.where(target_color == 0, object_height, distractor_height)

    return torch.where(target_height > minimal_height, 1.0, 0.0)


def object_ee_distance_target_aware(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distractor_cfg: SceneEntityCfg = SceneEntityCfg("distractor"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """target_color command에 따라 object 또는 distractor 중 실제 target인 쪽까지의 거리로 reaching reward 계산."""
    target_color = env.command_manager.get_command("target_color")
    object: RigidObject = env.scene[object_cfg.name]
    distractor: RigidObject = env.scene[distractor_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object_dist = torch.norm(object.data.root_pos_w - ee_w, dim=1)
    distractor_dist = torch.norm(distractor.data.root_pos_w - ee_w, dim=1)

    target_dist = torch.where(target_color == 0, object_dist, distractor_dist)

    return 1 - torch.tanh(target_dist / std)

def home_pose_after_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distractor_cfg: SceneEntityCfg = SceneEntityCfg("distractor"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]),
) -> torch.Tensor:
    """큐브가 들린 상태일 때만, arm joint가 초기 자세에 가까울수록 보상."""
    target_color = env.command_manager.get_command("target_color")
    object: RigidObject = env.scene[object_cfg.name]
    distractor: RigidObject = env.scene[distractor_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    object_height = object.data.root_pos_w[:, 2]
    distractor_height = distractor.data.root_pos_w[:, 2]
    target_height = torch.where(target_color == 0, object_height, distractor_height)
    is_lifted = target_height > minimal_height  # (num_envs,) 불리언

    # 초기 자세 값 (순서: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
    home_pose = torch.tensor(
        [0.0, -1.712, 1.712, -1.571, -1.571, 0.0], device=env.device
    )
    current_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, 6)

    pose_error = torch.norm(current_pos - home_pose, dim=-1)
    reward = (1.0 - torch.tanh(pose_error / std)) * is_lifted.float()

    return reward