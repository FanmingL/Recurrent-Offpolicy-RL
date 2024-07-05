import torch
from envs.make_pomdp_env import make_pomdp_env
from envs.pomdp_config import env_config
import gym
from typing import Dict
import numpy as np
import contextlib
import random
try:
    import envs.dmc
    has_dm_control = True
except:
    has_dm_control = False


@contextlib.contextmanager
def fixed_seed(seed):
    """上下文管理器，用于同时固定random和numpy.random的种子"""
    state_np = np.random.get_state()
    state_random = random.getstate()
    torch_random = torch.get_rng_state()
    if torch.cuda.is_available():
        # TODO: if not preproducable, check torch.get_rng_state
        torch_cuda_random = torch.cuda.get_rng_state()
        torch_cuda_random_all = torch.cuda.get_rng_state_all()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state_np)
        random.setstate(state_random)
        torch.set_rng_state(torch_random)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random)
            torch.cuda.set_rng_state_all(torch_cuda_random_all)


def make_env(env_name: str, seed: int) -> Dict:
    if env_name in env_config:
        with fixed_seed(seed):
            result = make_pomdp_env(env_name, seed)
        result['seed'] = seed
        return result
    else:
        with fixed_seed(seed):
            if env_name.startswith('dmc'):
                env = gym.make(env_name, seed=seed)
                max_episode_steps = env.unwrapped._max_episode_steps
            else:
                env = gym.make(env_name)
                max_episode_steps = env._max_episode_steps
        env.seed(seed)
        env.action_space.seed(seed+1)
        env.observation_space.seed(seed+2)
        result = {
            'train_env': env,
            'eval_env': env,
            'train_tasks': [],
            'eval_tasks': [None] * 20,
            'max_rollouts_per_task': 1,
            'max_trajectory_len': max_episode_steps,
            'obs_dim': env.observation_space.shape[0],
            'act_dim': env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n,
            'act_continuous': isinstance(env.action_space, gym.spaces.Box),
            'seed': seed,
            'multiagent': False
        }

        return result