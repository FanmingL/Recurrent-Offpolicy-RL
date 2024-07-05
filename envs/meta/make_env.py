import gym

from .wrappers import VariBadWrapper

# In VariBAD, they use on-policy PPO by vectorized env.
# In BOReL, they use off-policy SAC by single env.
import numpy as np
import contextlib
import random

@contextlib.contextmanager
def fixed_seed(seed):
    """上下文管理器，用于同时固定random和numpy.random的种子"""
    state_np = np.random.get_state()
    state_random = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state_np)
        random.setstate(state_random)

def make_env(env_id, episodes_per_task, seed=None, oracle=False, **kwargs):
    """
    kwargs: include n_tasks=num_tasks
    """
    with fixed_seed(seed):
        env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    env = VariBadWrapper(
        env=env,
        episodes_per_task=episodes_per_task,
        oracle=oracle,
    )
    return env
