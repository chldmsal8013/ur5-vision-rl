from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import joint_deviation_l1, joint_vel_out_of_manual_limit
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
import isaaclab.sim as sim_utils


def raw_rgb_image(env, sensor_cfg):
    """Raw RGB image, normalized to [0, 1], as (N, H, W, 3). Fed to a learnable CNN in the policy."""
    camera = env.scene[sensor_cfg.name]
    rgb = camera.data.output["rgb"]  # (N, H, W, 3), uint8
    return rgb.float() / 255.0


@configclass
class ImageCfg(ObsGroup):
    """Raw camera image observation group (kept separate from the 1D 'policy' group)."""

    image = ObsTerm(func=raw_rgb_image, params={"sensor_cfg": SceneEntityCfg("tiled_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True



UR5_ROBOTIQ_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="/home/choi/ur5_robotiq_2f85.usd",
        # [FIX] 2026-09-04: left/right_inner_finger에 붙인 ContactSensorCfg가 "could not find any
        # bodies with contact reporter API" 에러로 실패 -- activate_contact_sensors는 rigid_props
        # 안이 아니라 UsdFileCfg(RigidObjectSpawnerCfg) 자체의 필드. 이걸 켜면 이 prim 트리 아래
        # 모든 rigid body에 PhysxContactReporter API가 붙음(finger 두 개만 골라 켜는 옵션은 없음).
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # [FIX] velocity_iteration_count 0 -> 1 -> 2. 1은 5-DoF 시절(position-only IK, orientation
        # 추적 없음)엔 충분했으나, 6-DoF 복원 + pose(orientation 포함) IK 도입 이후(2026-09-02)
        # 5000iter 스모크에서 iter 13부터 value_function loss가 1e17~1e29 규모로 재발 --
        # 0->1로 잡았던 이전 사고와 동일 계열의 재발. 1->2로 한 단계 더 강화.
        # 주의: 이 solver 강화는 증상 완화 조치일 가능성이 높음 -- 근본 원인 후보는 따로 있음
        # (ik_rel_env_cfg.py의 DifferentialIKControllerCfg 참고: dls lambda_val 기본값 0.01은
        # 최대 증폭 1/(2*lambda)=50배까지 허용하며, orientation을 처음 추적하기 시작한 지금
        # wrist 근처 특이점에서 한 스텝에 과도하게 큰 joint 목표를 낼 수 있음). 이번에도 재발하면
        # solver를 더 올리기보다 lambda_val을 올리는 쪽을 먼저 볼 것.
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": -1.571,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            stiffness=400.0,
            damping=60.0,
        ),
        # [FIX] effort/velocity limit + stiffness/damping ported from IsaacLab's own validated
        # Robotiq 2F-85 reference config (isaaclab_assets/robots/franka.py: FRANKA_ROBOTIQ_GRIPPER_CFG)
        # -- same physical gripper, different arm. Previous effort_limit_sim=10.0 was ~165x below
        # that reference, which likely capped achievable grip force well below what's needed to
        # hold a cube. Also splits out the inner-finger joints into their own low-PD "gripper_finger"
        # group instead of lumping them into fully-passive (0/0): the reference config's own comment
        # says this is "to enable the gripper to grasp in a parallel manner".
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_inner_finger_joint", "left_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint",
                              "right_inner_finger_knuckle_joint",
                              "left_inner_finger_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class UR5RobotiqCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # reaching_object weight: 가까이 가기만으론 부족 (Kimi)
        self.rewards.reaching_object.weight = 0.8  # Kimi: 0.5 -> 0.8 (가까이 가기도 필요)
        # Reward 중복 제거 (Claude Code 진단)
        self.rewards.lifting_object.weight = 0.0  # object_is_lifted_bonus와 중복
        self.rewards.object_goal_tracking.weight = 0.0  # pick 안정화만
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.reaching_object.params["std"] = 0.3
        
        self.scene.robot = UR5_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            scale=0.5,
            use_default_offset=True,
        )
        # Continuous gripper action (Binary → JointPositionActionCfg)
        # 이전 결과: Binary로 잡기 시도했지만 안정 grip 어려움 (throwing 발생)
        # 목적: Continuous로 부분 close 가능하게 함 (0.0~0.8 사이)
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            scale=0.4,  # action [-1, 1] → joint pos [-0.4, 0.4], offset과 함께 [0.0, 0.8]
            use_default_offset=False,
            offset=0.4,  # neutral position (half-closed)
        )
        self.commands.object_pose.body_name = "wrist_3_link"
        # Target cube (red)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        # Distractor cube (blue) - fixed position
        self.scene.distractor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.25, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # TiledCamera for Vision RL (top-down view)
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 1.0),
                rot=(0.0, 1.0, 0.0, 0.0),
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            width=540,  # DLSS 300px 통과 (540*0.58=313)
            height=540,  # DLSS 300px 통과
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.13]),
                ),
            ],
        )
        # [FIX] 2026-09-04: bilateral grasp-contact sensors (dexsuite_kuka_allegro 패턴 이식).
        # left_inner_finger/right_inner_finger는 USD(ur5_robotiq_2f85.usd)에 실존하는 body 이름
        # -- ee_frame이 이미 검증한 "{ENV_REGEX_NS}/Robot/ur5/..." 플래튼 규칙을 그대로 따라
        # "{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/..."로 유추함. 5분 검증 스텝에서
        # 씬 생성이 에러 없이 되는지로 실제 경로가 맞는지 확인할 것 -- 틀렸으면 IsaacLab이 스캔
        # 단계에서 명확한 prim-not-found 에러를 낸다.
        self.scene.left_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/left_inner_finger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/Cube"],
        )
        self.scene.right_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/right_inner_finger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/Cube"],
        )

        # [FIX] 2026-09-07: value_function loss still hit transient 1e13-1e28 spikes in the
        # valclamp_fix_smoke run (recovered iter 15->76, clean to iter 337, then relapsed) even
        # with the critic output clamped (cnn_policy.py CNNActorCritic.evaluate). Root-caused to
        # rsl_rl's own vendored [PATCH v3] comment (ppo.py) diagnosis: "a single diverged env"
        # produces a genuine large-but-finite RAW reward (not a critic artifact) that survives
        # the critic clamp because that clamp only bounds the bootstrap term, not the
        # environment's own reward. Episode_Reward/joint_vel reached -6.7e13 (weight -1e-4) in
        # the original fresh_check log at the explosion iter -- solving for v assuming it's
        # concentrated in one joint: sum(v^2)=6.7e17 -> v~8.2e8 rad/s. That is ~8 orders of
        # magnitude beyond anything physically possible for this arm (UR5 datasheet: 180 deg/s
        # ~= 3.14 rad/s per joint) or anything seen in a healthy iteration in these logs (typical
        # Episode_Reward/joint_vel -0.03 to -0.08 -> sum(v^2) 300-800 -> individual joints peaking
        # at roughly 6-20 rad/s during normal RL exploration, well above the real robot's rated
        # speed since sim joints aren't velocity-limited the way hardware is).
        # max_velocity=100.0 rad/s sits ~5-15x above any normal-iteration peak observed in these
        # logs (so it won't false-trigger on ordinary aggressive exploration) and ~6-7 orders of
        # magnitude below the measured divergence scale (~8e8 rad/s) -- not a fine-tuned number,
        # just needs to sit in the (very wide) gap between "normal" and "diverged", which it does
        # with enormous margin on both sides. Effect: instead of a diverged env running for up to
        # ~240 more steps (its full episode) silently emitting a poisoned reward every step, it
        # resets within 1 step of crossing the threshold -- cutting off the poison at the source,
        # upstream of the critic-clamp fix (which stays in place as a second layer of defense).
        self.terminations.joint_vel_divergence = DoneTerm(
            func=joint_vel_out_of_manual_limit,
            params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot")},
        )

        self.curriculum = None
        self.rewards.action_rate.weight = -1e-4
        self.rewards.joint_vel.weight = -1e-4
        self.rewards.reaching_object.params["std"] = 0.3
        
        # Object_is_lifted bonus reward (Kimi 조언 Step 3)
        # 목적: outcome-based reward - 실제 lift 성공에 강한 보상
        # 예상: 잡기 + 들어올리기 학습 유도 (fine-grained reach reward 보완)
        # [FIX] object_is_lifted -> object_is_lifted_sustained: object 초기/리셋 높이(0.055, 아래
        # RigidObjectCfg 참고)가 이 minimal_height(0.05)보다 이미 높아서, height 단독 조건은
        # 아무것도 안 해도 거의 항상 참이 될 수 있음. EE 근접 조건 + N스텝 연속 유지 조건을 추가해
        # "우연히 튕겨서 5cm 넘긴 것"과 "실제로 쥐고 버틴 것"을 구분함.
        self.rewards.object_is_lifted_bonus = RewTerm(
            func=mdp.object_is_lifted_sustained,
            weight=10.0,
            params={
                "minimal_height": 0.05,  # 15cm(unreachable) -> 5cm (Kimi 지적)
                # [FIX] 9시간/21320iter 실측 결과 Bonus가 사실상 0(1/21320회)이라 조건이 너무 엄격했음.
                # min_steps 10->5(0.2s->0.1s), max_ee_distance 0.06->0.12(2x)로 완화.
                # 주의: max_ee_distance를 너무 풀면 "안 잡았는데 근처를 지나가기만 해도 성공" 판정되는
                # 원래 버그(resting height 0.055 > minimal_height 0.05)의 위험이 다시 커짐 -- 이번에도
                # Bonus가 여전히 거의 안 뜨면 더 풀지 말고, height/proximity/sustain 각각을 분리해서
                # 로그를 찍어보는 게 다음 단계로 맞음.
                "min_steps": 5,
                "max_ee_distance": 0.12,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                # [FIX] 2026-09-07: live diagnostic on the iter-20500 checkpoint showed this bonus
                # firing regularly while grasp_contact_bonus stayed at 0.0000 for 20000+ iterations
                # -- both finger sensors read exactly 0.0000 force against the object during the
                # airborne window (verified against real, nonzero unfiltered contact force
                # elsewhere, e.g. 35.9N, ruling out a sensor/path bug). The policy was knocking/
                # flinging the cube near the EE, not holding it. min_contact_force=0.05 (either
                # finger, not both) closes that loophole -- see object_is_lifted_sustained's
                # docstring for the full margin reasoning.
                "left_contact_sensor_name": "left_finger_contact",
                "right_contact_sensor_name": "right_finger_contact",
                "min_contact_force": 0.05,
            },
        )

        # [FIX] grasp_proximity_reward: shaping term for grasp *mechanics* (fingers closed around
        # the object, holding it steady) as a stepping stone toward object_is_lifted_bonus (which
        # needs height too). Play mode showed the policy reaching + curling upward without ever
        # closing the gripper around the cube -- this rewards "closed + near + object not being
        # knocked around" even before any lift happens.
        # 값 근거: max_distance=3cm은 object_is_lifted_bonus의 max_ee_distance(12cm)보다 훨씬 타이트
        # -- 이건 "근처"가 아니라 "접촉/근접" 수준을 노림. gripper_close_range=(0.15,0.65)는
        # gripper_action의 실제 joint pos 범위([0,0.8], offset=0.4+scale=0.4)의 19~81% 구간으로,
        # "많이 닫힘"을 잡되 완전 개방(0)/완전 밀폐(0.8) 양극단은 제외한 설계값. max_object_speed=0.05
        # 는 "정지/안정" 휴리스틱. 셋 다 실측 검증된 값은 아님 -- 5000iter 스모크에서 이 reward가
        # 뜨는 빈도를 반드시 확인할 것.
        
        # [FIX] weight 2.0 -> 5.0 (2026-09-04). 30000iter 실측: peak bonus 0.88 (iter 27975),
        # top-down 자세로 큐브 근접까지는 학습됐으나 최종 손가락 닫기 동작이 부족. reaching_object
        # (weight 0.8)는 매 스텝 발생하는 dense reward라 episode 누적 기준 이 grasp shaping term을
        # 압도할 수 있음 -- weight를 올려 "닫아라" 신호를 키움. object_is_lifted_bonus(outcome,
        # weight 10.0)보다는 낮게 유지: shaping term이 outcome term과 동률/그 이상이면 정책이 굳이
        # height/sustain 조건까지 채우는 완전한 lift 대신 이 조건(더 쉬움, height 불필요)만 반복
        # 파밍할 유인이 생김 -- reward hacking 회피 목적으로 절반 수준인 5.0으로 제한.
        # [FIX] 2026-09-04: weight 0.0으로 은퇴. 30000iter 실측(peak bonus 0.88)에서 Play mode 관찰:
        # 정책이 실제 grasp 대신 "밀어서 물체를 정지시키기"를 학습함 -- 이 함수의 max_object_speed
        # 조건이 "잡혀서 정지"와 "밀려서 벽/테이블에 눌려 정지"를 구별 못 하는 게 원인으로 진단됨.
        # 코드/params는 롤백 대비로 남기고 weight만 죽임 -- 아래 grasp_contact_bonus(bilateral
        # contact sensor 기반)로 대체.
        self.rewards.grasp_proximity_bonus = RewTerm(
            func=mdp.grasp_proximity_reward,
            weight=0.0,
            params={
                "max_distance": 0.03,
                "gripper_close_range": (0.15, 0.65),
                "max_object_speed": 0.05,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            },
        )

        # [FIX] 2026-09-04: grasp_proximity_bonus를 대체하는 접촉력 기반 grasp reward.
        # dexsuite_kuka_allegro_env_cfg.py의 ContactSensorCfg + contacts() 패턴을 이식 -- 왼쪽/
        # 오른쪽 finger 둘 다 동시에 물체에 힘을 가해야만(양쪽에서 미는 게 아니라 진짜로 사이에
        # 끼워 쥠) fire. 한쪽만 미는 동작은 한쪽 센서에서만 힘이 잡히므로 통과 불가.
        # force_threshold=0.5N: dexsuite는 allegro 손가락 기준 1.0N을 썼으나 그건 훨씬 작고 가벼운
        # 손가락 접촉 기준값 -- 우리는 "확실한 grip력"이 아니라 "노이즈가 아닌 진짜 양쪽 접촉"만
        # 구분하면 되므로 더 낮은 값으로 민감하게 잡음. max_object_speed=0.05는 기존
        # grasp_proximity_bonus와 동일 값(안정/정지 기준 유지). weight=5.0은 object_is_lifted_bonus
        # (outcome, weight=10.0)보다 낮게 유지 -- shaping term이 outcome과 동률이면 정책이 height
        # 조건까지 채우는 완전한 lift 대신 이 조건(더 쉬움)만 반복 파밍할 유인이 생김.
        self.rewards.grasp_contact_bonus = RewTerm(
            func=mdp.grasp_contact_reward,
            weight=5.0,
            params={
                "force_threshold": 0.5,
                "max_object_speed": 0.05,
                "object_cfg": SceneEntityCfg("object"),
                "left_contact_sensor_name": "left_finger_contact",
                "right_contact_sensor_name": "right_finger_contact",
            },
        )

        # [FIX] 2026-09-04: object_dropping 종료에 페널티 없었음(터미네이션만 발생, reward 0) --
        # "진짜 잡았다가 놓친 것"과 "밀어서 테이블 밖으로 떨어뜨린 것"이 보상 관점에서 완전히
        # 동일하게 취급되어 pushing을 억제할 유인이 전혀 없었음. is_terminated_term은 IsaacLab
        # 표준 패턴(time_out은 자동 제외, term_keys로 지정한 termination만 페널티) -- object_dropping
        # 터미네이션 발생 시 1회성 페널티. weight=-5.0: reaching_object 등 dense reward와 값 크기가
        # 겹치지 않는 수준의 "확실히 아프지만" 학습 불안정을 유발할 만큼 크지 않은 값 -- 0/1 곱셈이라
        # 물리/그래디언트 폭주 위험은 없음(과거 사고들과 다른 종류의 안전한 reward).
        self.rewards.dropping_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-5.0,
            params={"term_keys": "object_dropping"},
        )

        # wrist_1, wrist_2, wrist_3 twist 방지 penalty (잡은 후 자세 유지)
        self.rewards.wrist_1_deviation = RewTerm(
            func=joint_deviation_l1,
            weight=-0.1,  # Kimi: -0.5 -> -0.1 (grasp 자세 허용)
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_1_joint"])},
        )
        self.rewards.wrist_2_deviation = RewTerm(
            func=joint_deviation_l1,
            weight=-0.1,  # Kimi: -0.5 -> -0.1
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_2_joint"])},
        )
        # [FIX] wrist_3 6-DoF 복원(2026-09-02)에 맞춰 재도입. 최초 6-DoF 시절(.bak_before_5dof)엔
        # -0.05였으나, 지금은 옆 두 wrist 관절과 동일한 -0.1로 맞춤 -- 애초에 twist 문제 때문에
        # wrist_3를 fixed로 죽였던 이력이 있으니, 세 관절 중 하나만 더 느슨한 패널티를 줄 근거가 없음.
        self.rewards.wrist_3_deviation = RewTerm(
            func=joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_3_joint"])},
        )

        # red_block의 body name이 "Cube"라서 base의 "Object" 참조 실패함
        # body_names 없이 root만 참조하도록 override
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")

        # Raw RGB image observation group, fed to a learnable CNN inside the policy (end-to-end PPO)
        self.observations.image = ImageCfg()


@configclass
class UR5RobotiqCubeLiftEnvCfg_PLAY(UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

@configclass
class UR5RobotiqCubeLiftEnvCfg_State(UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Vision observation 비활성화
        self.observations.image = None
        self.scene.tiled_camera = None