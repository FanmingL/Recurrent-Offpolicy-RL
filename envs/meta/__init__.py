from gym.envs.registration import register
import gym
## off-policy variBAD benchmark

register(
    "PointRobot-v0",
    entry_point="envs.meta.toy_navigation.point_robot:PointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2},
)

register(
    "PointRobotSparse-v0",
    entry_point="envs.meta.toy_navigation.point_robot:SparsePointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2, "goal_radius": 0.2},
)

register(
    "Wind-v0",
    entry_point="envs.meta.toy_navigation.wind:WindEnv",
)

register(
    "HalfCheetahVel-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)
# DM_Hopper_gravity-v2
# DM_Hopper_dof_damping-v2
# DM_Hopper_gravity_dof_damping-v2
# DM_Hopper_body_mass-v2
# DM_HalfCheetah_gravity-v2
# DM_HalfCheetah_dof_damping-v2
# DM_HalfCheetah_gravity_dof_damping-v2
# DM_HalfCheetah_body_mass-v2
# DM_Walker2d_gravity-v2
# DM_Walker2d_dof_damping-v2
# DM_Walker2d_gravity_dof_damping-v2
# DM_Walker2d_body_mass-v2
# DM_Ant_gravity-v2
# DM_Ant_dof_damping-v2
# DM_Ant_gravity_dof_damping-v2
# DM_Ant_body_mass-v2
# DM_Humanoid_gravity-v2
# DM_Humanoid_dof_damping-v2
# DM_Humanoid_gravity_dof_damping-v2
# DM_Humanoid_body_mass-v2
for _env_name in ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']:
    for _rand_params in [['gravity'], ['dof_damping'], ['gravity', 'dof_damping'], ['body_mass']]:
        register(
            f"DM_{_env_name.split('-')[0]}_{'_'.join(_rand_params)}-v2",
            entry_point="envs.meta.wrappers:dynamics_mujoco_wrapper",
            kwargs={
                "env_name": _env_name,
                "rand_params": _rand_params,
                "log_scale_limit": 2.0 if _env_name == 'Humanoid-v2' else 3.0,
            },
            max_episode_steps=1000,
        )

## on-policy variBAD benchmark

register(
    "AntDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.ant_dir:AntDirEnv",
        "max_episode_steps": 200,
        "forward_backward": True,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "CheetahDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "HumanoidDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.humanoid_dir:HumanoidDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)
