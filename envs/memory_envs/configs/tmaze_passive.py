from typing import Tuple
from gym.envs.registration import register
from .terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"Mem-T-passive-{l}-v0"
env_name_cont_fn = lambda l: f"Mem-T-passive-{l}-cont-act-v0"


def create_fn(env_name, continuous_act_space=False):
    length = env_name
    env_name = env_name_cont_fn(length) if continuous_act_space else env_name_fn(length)
    register(
        env_name,
        entry_point="envs.memory_envs.tmaze:TMazeClassicPassive",
        kwargs=dict(
            corridor_length=length,
            penalty=-1.0 / length,  # NOTE: \sum_{t=1}^T -1/T = -1
            distract_reward=0.0,
            continuous_act_space=continuous_act_space
        ),
        max_episode_steps=length + 1,  # NOTE: has to define it here
    )

    return env_name


# def get_config():
#     config = ConfigDict()
#     config.create_fn = create_fn
#
#     config.env_type = "tmaze_passive"
#     config.terminal_fn = finite_horizon_terminal
#
#     config.eval_interval = 10
#     config.save_interval = 10
#     config.eval_episodes = 10
#
#     # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
#     config.env_name = 10
#     config.distract_reward = 0.0
#
#     return config
