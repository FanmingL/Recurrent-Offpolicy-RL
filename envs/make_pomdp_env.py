import numpy as np
import gym
import sys
from .pomdp_config import env_config
from typing import Dict
from gym.wrappers import RescaleAction
from .yang_domains import *

import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def make_pomdp_env(env_name: str, seed: int) -> Dict:
    config = env_config[env_name]
    result = _make_pomdp_env(env_name=env_name, seed=seed, **config)
    act_continuous = result['act_continuous']
    obs_dim = result['obs_dim']
    act_dim = result['act_dim']
    train_env, eval_env = result['train_env'], result['eval_env']
    def regularize_space(_env):
        if act_continuous:
            _env.action_space = gym.spaces.Box(low=-np.ones((act_dim,)),
                                                    high=np.ones(act_dim,),
                                                    dtype=np.float64)
        else:
            _env.action_space = gym.spaces.Discrete(act_dim)
        _env.observation_space = gym.spaces.Box(low=-np.inf * np.ones((obs_dim,)),
                                                     high=np.inf * np.ones((obs_dim,)),
                                                     dtype=np.float64)
    regularize_space(train_env)
    regularize_space(eval_env)
    return result

def _make_pomdp_env(
        env_type,
        env_name,
        seed,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs
):
    result_dict = {}
    # initialize environment
    assert env_type in [
        "meta",
        "pomdp",
        "credit",
        "rmdp",
        "generalize",
        "atari",
        "neorl2",
        "metapid",
        "yang_domains"
    ]
    result_dict['env_type'] = env_type

    if env_type == "meta":  # meta tasks: using varibad wrapper
        from .meta.make_env import make_env
        assert max_rollouts_per_task is not None, f'max_rollouts_per_task for meta envs should be set'
        if num_tasks is None:
            assert num_eval_tasks is not None, f'if num_tasks is set to None, num_eval_tasks should not be None, expected a integral'
        result_dict['train_env'] = make_env(
            env_name,
            max_rollouts_per_task,
            seed=seed,
            n_tasks=num_tasks,
            **kwargs,
        )  # oracle in kwargs

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        if result_dict['train_env'].n_tasks is not None:
            # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
            # split to train/eval tasks
            assert num_train_tasks >= num_eval_tasks > 0, f'num_train_tasks: {num_train_tasks}, num_eval_tasks: {num_eval_tasks}'
            if hasattr(result_dict['train_env'], 'get_all_task_idx'):
                shuffled_tasks = np.random.permutation(
                    result_dict['train_env'].get_all_task_idx()
                )
            else:
                shuffled_tasks = np.random.permutation(
                    result_dict['train_env'].unwrapped.get_all_task_idx()
                )
            result_dict['train_tasks'] = shuffled_tasks[:num_train_tasks]
            result_dict['eval_tasks'] = shuffled_tasks[-num_eval_tasks:]
        else:
            # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
            assert num_tasks == num_train_tasks == None
            assert (
                    num_eval_tasks > 0
            )  # to specify how many tasks to be evaluated each time
            result_dict['train_tasks'] = []
            result_dict['eval_tasks'] = num_eval_tasks * [None]

        # calculate what the maximum length of the trajectories is
        result_dict['max_rollouts_per_task'] = max_rollouts_per_task
        result_dict['max_trajectory_len'] = result_dict['train_env'].horizon_bamdp  # H^+ = k * H
    elif result_dict['env_type'] in [
        "pomdp",
        "credit",
    ]:  # pomdp/mdp task, using pomdp wrapper
        if env_name == 'Plating-v0':
            from . import plating_env
            result_dict['train_env'] = gym.make(env_name)

        elif env_name.startswith('Mem-'):
            from .memory_envs.make_env import make_env
            if env_name.startswith('Mem-T-passive'):
                from .memory_envs.configs.tmaze_passive import create_fn as create_fn_mem
                distractor_len = int(env_name.split('-')[3])
            elif env_name.startswith('Mem-T-active'):
                from .memory_envs.configs.tmaze_active import create_fn as create_fn_mem
                distractor_len = int(env_name.split('-')[3])
            elif env_name.startswith('Mem-SR'):
                from .memory_envs.configs.keytodoor import create_fn as create_fn_mem
                distractor_len = int(env_name.split('-')[2])
            else:
                raise NotImplementedError(f'{env_name} has not been considered!!')

            if 'cont-act' in env_name:
                cont_act = True
            else:
                cont_act = False

            create_fn_mem(distractor_len, continuous_act_space=cont_act)
            result_dict['train_env'] = make_env(env_name, seed)
        else:
            from . import pomdp
            from . import credit_assign
            result_dict['train_env'] = gym.make(env_name)

        assert num_eval_tasks > 0
        result_dict['train_env'].seed(seed)
        result_dict['train_env'].action_space.np_random.seed(seed)  # crucial

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]

        result_dict['max_rollouts_per_task'] = 1
        if hasattr(result_dict['train_env'], '_max_episode_steps'):
            result_dict['max_trajectory_len'] = result_dict['train_env']._max_episode_steps
        elif hasattr(result_dict['train_env'].unwrapped, '_max_episode_steps'):
            result_dict['max_trajectory_len'] = result_dict['train_env'].unwrapped._max_episode_steps
        elif hasattr(result_dict['train_env'], 'max_episode_steps'):
            result_dict['max_trajectory_len'] = result_dict['train_env'].max_episode_steps
        else:
            raise AttributeError(f'env does not have attribute _max_episode_steps')
    elif result_dict['env_type'] in [
        "neorl2"
    ]:  # pomdp/mdp task, using pomdp wrapper
        from . import dmsd
        from . import pipeline

        assert num_eval_tasks > 0
        result_dict['train_env'] = gym.make(env_name, seed=seed)
        result_dict['train_env'].seed(seed)
        result_dict['train_env'].action_space.np_random.seed(seed)  # crucial

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]

        result_dict['max_rollouts_per_task'] = 1
        if hasattr(result_dict['train_env'], '_max_episode_steps'):
            result_dict['max_trajectory_len'] = result_dict['train_env']._max_episode_steps
        elif hasattr(result_dict['train_env'].unwrapped, '_max_episode_steps'):
            result_dict['max_trajectory_len'] = result_dict['train_env'].unwrapped._max_episode_steps
        else:
            raise AttributeError(f'env does not have attribute _max_episode_steps')
    elif result_dict['env_type'] in ['metapid']:
        from . import metapid
        from .metapid.metapid import MetaPIDEnv
        train_envs = kwargs['train_envs']
        eval_envs = eval_envs
        obs_dim = kwargs['obs_dim']
        result_dict['train_env'] = MetaPIDEnv(train_envs=train_envs, ood_eval_envs=eval_envs, seed=seed, obs_dim=obs_dim, ood=False)
        result_dict['train_env'].seed(seed)
        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]

        result_dict['max_rollouts_per_task'] = 1
        result_dict['max_trajectory_len'] = result_dict['train_env'].unwrapped._max_episode_steps
        result_dict['multiagent'] = True

    elif env_type == "atari":
        from .atari import create_env

        assert num_eval_tasks > 0
        result_dict['train_env'] = create_env(env_name)
        result_dict['train_env'].seed(seed)
        result_dict['train_env'].action_space.np_random.seed(seed)  # crucial

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]

        result_dict['max_rollouts_per_task'] = 1
        result_dict['max_trajectory_len'] = result_dict['train_env']._max_episode_steps

    elif env_type == "rmdp":  # robust mdp task, using robust mdp wrapper
        from .rl_generalization import sunblaze_envs

        assert (
                num_eval_tasks > 0 and worst_percentile > 0.0 and worst_percentile < 1.0
        )
        result_dict['train_env'] = sunblaze_envs.make(env_name, **kwargs)  # oracle
        result_dict['train_env'].seed(seed)
        assert np.all(result_dict['train_env'].action_space.low == -1)
        assert np.all(result_dict['train_env'].action_space.high == 1)

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['worst_percentile'] = worst_percentile

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]

        result_dict['max_rollouts_per_task'] = 1
        result_dict['max_trajectory_len'] = result_dict['train_env']._max_episode_steps

    elif env_type == "generalize":
        from .rl_generalization import sunblaze_envs
        result_dict['train_env'] = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
        result_dict['train_env'].seed(seed)
        assert np.all(result_dict['train_env'].action_space.low == -1)
        assert np.all(result_dict['train_env'].action_space.high == 1)

        def check_env_class(env_name):
            if "Normal" in env_name:
                return "R"
            if "Extreme" in env_name:
                return "E"
            return "D"

        result_dict['train_env_name'] = check_env_class(env_name)

        result_dict['eval_envs'] = {}
        for env_name, num_eval_task in eval_envs.items():
            eval_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
            eval_env.seed(seed + 1)
            result_dict['eval_envs'][eval_env] = (
                check_env_class(env_name),
                num_eval_task,
            )  # several types of evaluation envs
            result_dict['eval_env'] = eval_env

        # logger.log(result_dict['train_env']_name, result_dict['train_env'])
        # logger.log(result_dict['eval_envs'])

        result_dict['train_tasks'] = []
        result_dict['max_rollouts_per_task'] = 1
        result_dict['max_trajectory_len'] = result_dict['train_env']._max_episode_steps

    elif env_type == "yang_domains":
        result_dict['train_env'] = gym.wrappers.RescaleAction(gym.make(env_name), -1, 1)
        # result_dict['train_env'] = gym.make(env_name)
        result_dict['train_env'].seed(seed)

        result_dict['eval_env'] = result_dict['train_env']
        result_dict['eval_env'].seed(seed + 1)

        result_dict['train_tasks'] = []
        result_dict['eval_tasks'] = num_eval_tasks * [None]
        
        result_dict['max_rollouts_per_task'] = 1
        result_dict['max_trajectory_len'] = result_dict['train_env'].spec.max_episode_steps
        
    else:
        raise ValueError

    # get action / observation dimensions
    if result_dict['train_env'].action_space.__class__.__name__ == "Box":
        # continuous action space
        result_dict['act_dim'] = result_dict['train_env'].action_space.shape[0]
        result_dict['act_continuous'] = True
    else:
        assert result_dict['train_env'].action_space.__class__.__name__ == "Discrete"
        result_dict['act_dim'] = result_dict['train_env'].action_space.n
        result_dict['act_continuous'] = False
    result_dict['obs_dim'] = result_dict['train_env'].observation_space.shape[0]  # include 1-dim done
    if not 'multiagent' in result_dict:
        result_dict['multiagent'] = False
    # logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)
    return result_dict
