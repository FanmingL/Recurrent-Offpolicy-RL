from typing import Tuple
from gym.envs.registration import register
from .terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"Mem-T-active-{l}-v0"
env_name_cont_fn = lambda l: f"Mem-T-active-{l}-cont-act-v0"

def create_fn(env_name, continuous_act_space=False):
    length = env_name
    env_name = env_name_cont_fn(length) if continuous_act_space else env_name_fn(length)
    register(
        env_name,
        entry_point="envs.memory_envs.tmaze:TMazeClassicActive",
        kwargs=dict(
            corridor_length=length,
            penalty=-1.0 / length,  # NOTE: \sum_{t=1}^T -1/T = -1
            distract_reward=0.0,
            continuous_act_space=continuous_act_space
        ),
        max_episode_steps=length + 2 * 1 + 1,  # NOTE: has to define it here
    )

    return env_name
