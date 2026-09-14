from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class UR5RobotiqCubeLiftEnvCfg_IK(joint_pos_env_cfg.UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # IK Action (arm만 override, gripper 유지)
        # [FIX] 6-DoF 복원(2026-09-02): wrist_3_joint가 USD에서 다시 revolute가 되면서 팔이
        # 6-DOF가 됐으므로 "pose" 명령(6D: pos+rot)이 더 이상 rank-deficient가 아님
        # (Jacobian 6x6, use_relative_mode=True라 action_dim은 7이 아니라 6 -- differential_ik.py
        # action_dim 프로퍼티: pose+relative는 (dx,dy,dz,droll,dpitch,dyaw)). scale=0.5는
        # franka의 검증된 ik_rel_env_cfg.py(pose+relative, scale=0.5) 참조값을 그대로 사용 --
        # position/orientation을 따로 스케일할 근거 있는 값이 없어 공식 레퍼런스를 그대로 씀.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                # [FIX] lambda_val 기본값 0.01 -> 0.1 (2026-09-02). dls의 최대 증폭 배율은
                # 1/(2*lambda_val) -- 0.01일 땐 특이점 근처에서 pose_error 한 스텝이 최대 50배
                # 증폭돼 delta_joint_pos로 나갈 수 있음. 6dof_solver_v2_resume 런에서 실측:
                # iter 2793에 joint_vel reward가 정상(-0.02)에서 -2.33으로 튀었다가 2794-2796엔
                # 정상 복귀, 그러다 2797에 -16026으로 폭발하고 프로세스가 그대로 멈춤(hang) --
                # 갑작스런 단발성 스파이크 패턴이라 solver 해상도 부족(점진적 drift)보다는 특이점
                # 근처에서의 큰 one-step 조인트 점프에 더 부합함. lambda_val=0.1이면 최대 증폭이
                # 5배로 줄어듦 -- IK 추종은 약간 둔해지지만(수렴 느려짐), lift 태스크는 픽셀 단위
                # orientation 정밀도가 필요 없어 감수 가능한 트레이드오프로 판단.
                ik_params={"lambda_val": 0.1},
            ),
            scale=0.5,
            # [FIX] 2026-09-02: lambda_val 튜닝(0.01->0.1)만으론 폭발 재발 방지 실패 -- 실측으로
            # 확인된 실제 원인은 DLS 증폭도, solver 해상도도 아니라 "Mean action noise std"가
            # 학습 진행에 따라 계속 커지는 것(1.00 -> 1.14, entropy_coef=0.005로 상한 없이 성장)이었음.
            # std가 커질수록 정책이 이따금 매우 큰 raw action을 샘플하고, scale=0.5를 거쳐도
            # processed_action(=IK에 보내는 pose 명령)이 설계 의도(±0.5 근방)를 훨씬 초과할 수
            # 있음 -- 이게 iter ~2700-2800대에서 solver/lambda 설정과 무관하게 반복 재발한 진짜
            # 원인. clip은 scale 적용 "이후"의 processed_action에 직접 상한을 걸어 std가 얼마나
            # 커지든 IK로 들어가는 명령 자체를 ±0.5(위치 0.5m/스텝, 회전 0.5rad/스텝)로 하드
            # 캡함 -- scale=0.5가 원래 의도한 "정상 범위"를 그대로 강제하는 값이라 임의로 고른
            # 숫자가 아님. clip 키의 joint_names 매칭은 이 액션의 pose 축(dx,dy,dz,droll,dpitch,
            # dyaw)에 대해 positional로 적용됨(task_space_actions.py DifferentialInverseKinematicsAction
            # 참고) -- 여기선 6개 전부 동일 범위라 ".*" 하나로 충분.
            clip={".*": (-0.5, 0.5)},
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.13]),
        )


@configclass
class UR5RobotiqCubeLiftEnvCfg_IK_PLAY(UR5RobotiqCubeLiftEnvCfg_IK):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # [FIX] 2026-09-08: default ViewerCfg (eye=(7.5,7.5,7.5), lookat=(0,0,0)) frames the
        # whole multi-env grid -- with --num_envs 1 for a demo recording this renders the robot
        # as a tiny distant cube, unusable for judging grasp quality. Object spawns at
        # (0.5, 0, 0.055) (see joint_pos_env_cfg.py's Object RigidObjectCfg); robot base sits at
        # the env origin. A closer 3/4 elevated angle centered on the object/gripper workspace.
        self.viewer.eye = (1.2, 1.0, 0.8)
        self.viewer.lookat = (0.4, 0.0, 0.15)
