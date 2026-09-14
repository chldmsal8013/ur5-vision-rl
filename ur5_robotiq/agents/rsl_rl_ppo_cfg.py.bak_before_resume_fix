from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.manager_based.manipulation.lift.config.ur5_robotiq.cnn_policy import CNNActorCriticCfg


@configclass
class UR5RobotiqLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "ur5_robotiq_lift"
    # NOTE: fresh run required -- CNNActorCritic's state_dict does not match the old
    # state-based checkpoint's ActorCritic, loading it would crash.
    resume = False
    # "policy"/"critic" obs sets each combine the 1D "policy" group (joint state, command,
    # last action) with the 4D "image" group (raw camera pixels, routed through the CNN
    # inside CNNActorCritic). See cnn_policy.py.
    obs_groups = {
        "policy": ["policy", "image"],
        "critic": ["policy", "image"],
    }
    policy = CNNActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        cnn_feature_dim=256,
        image_group_name="image",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
