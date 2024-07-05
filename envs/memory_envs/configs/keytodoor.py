from typing import Tuple
from gym.envs.registration import register
from .terminal_fns import finite_horizon_terminal
from ..key_to_door import key_to_door
import copy

def create_fn(env_name, continuous_act_space=False):
    if continuous_act_space:
        env_name_fn = lambda distract: f"Mem-SR-{distract}-cont-act-v0"
    else:
        env_name_fn = lambda distract: f"Mem-SR-{distract}-v0"
    env_name_ = env_name
    env_name = env_name_fn(env_name)
    MAX_FRAMES_PER_PHASE_SR = copy.deepcopy(key_to_door.MAX_FRAMES_PER_PHASE_SR)

    MAX_FRAMES_PER_PHASE_SR.update({"distractor": env_name_})
    # optimal expected return: 1.0 * (~23) + 5.0 = 28. due to unknown number of respawned apples
    register(
        env_name,
        entry_point="envs.memory_envs.key_to_door.tvt_wrapper:KeyToDoor",
        kwargs=dict(
            flatten_img=True,
            one_hot_actions=False,
            apple_reward=1.0,
            final_reward=5.0,
            respawn_every=20,  # apple respawn after 20 steps
            REWARD_GRID=copy.deepcopy(key_to_door.REWARD_GRID_SR),
            max_frames=copy.deepcopy(MAX_FRAMES_PER_PHASE_SR),
            continuous_act_space=continuous_act_space
        ),
        max_episode_steps=sum(MAX_FRAMES_PER_PHASE_SR.values()), # no info if max episode steps = sum(MAX_FRAMES_PER_PHASE_SR.values())
    )
    return env_name


#
# def get_config():
#     config = ConfigDict()
#     config.create_fn = create_fn
#
#     config.env_type = "key_to_door"
#     config.terminal_fn = finite_horizon_terminal
#
#     config.eval_interval = 50
#     config.save_interval = 50
#     config.eval_episodes = 20
#
#     config.env_name = 60
#
#     return config
