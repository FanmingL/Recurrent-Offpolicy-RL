import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.*")
warnings.filterwarnings("ignore", ".*The DISPLAY environment variable is missing") # dmc
warnings.filterwarnings("ignore", category=FutureWarning) # art
import os

from gym import Space
import numpy as np
from ..policy_value_models.make_models import make_policy_model
import gym
from ..env_utils.make_env import make_env
import numpy as np
import torch
import random
import multiprocessing


def norm_act(act: np.ndarray, act_space) -> np.ndarray:
    if isinstance(act_space, gym.spaces.Box):
        return (act - act_space.low) / (act_space.high - act_space.low) * 2 - 1
    else:
        return act

def unorm_act(act: np.ndarray, act_space) -> np.ndarray:
    if isinstance(act_space, gym.spaces.Box):
        return (act + 1) / 2 * (act_space.high - act_space.low) + act_space.low
    else:
        return act

def n2t(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(data).to(torch.get_default_dtype()).to(device)

def n2t_2dim(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(data).to(torch.get_default_dtype()).to(device).reshape((-1, data.shape[-1]))
def t2n(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy()

@torch.no_grad()
def policy_eval(env, env_info, act_dim, policy, nrollout, device=None):
    if device is not None:
        policy.to(device)
    policy.eval()
    if device is None:
        device = torch.device('cpu')
    ep_rets = []
    ep_lens = []
    obs_dim = env.observation_space.shape[0]
    discrete_env = not env_info['act_continuous']
    infos = {}
    for i in range(nrollout):
        ep_ret = 0
        ep_len = 0
        if hasattr(env, 'meta_env_flag') and hasattr(env, 'n_tasks') and env.n_tasks is not None:
            eval_tasks = env_info['eval_tasks']
            idx = np.random.randint(0, len(eval_tasks))
            state_np = env.reset(eval_tasks[idx])
        else:
            state_np = env.reset()
        last_action_np = np.zeros((1, act_dim))
        last_state_np = np.zeros((1, obs_dim))
        reward_np = np.zeros((1, 1))
        rnn_hidden = policy.make_init_state(1, device=device)
        done = False
        while not done:
            act_mean, _, _, _, rnn_hidden, _ = policy.forward(
                state=n2t_2dim(state_np, device),
                lst_state=n2t_2dim(last_state_np, device),
                lst_action=n2t_2dim(last_action_np, device),
                rnn_memory=rnn_hidden,
                reward=n2t_2dim(reward_np, device)
            )
            act_mean = t2n(act_mean)
            act_mean = act_mean.squeeze(0)
            act_unormalized = unorm_act(act_mean, env.action_space)
            if discrete_env:
                act_unormalized = int(act_unormalized)
            next_state, reward, done, info = env.step(act_unormalized)

            last_state_np = state_np.copy()
            state_np = next_state.copy()
            reward_np[:] = reward
            if discrete_env:
                last_action_np = np.zeros((1, act_dim))
                last_action_np[..., int(act_mean)] = 1
            else:
                last_action_np = act_mean.copy()
            if info is not None:
                for k, v in info.items():
                    try:
                        v_float = float(v)
                    except Exception as _:
                        continue
                    if k not in infos:
                        infos[k+'Test'] = []

                    infos[k+'Test'].append(v_float)
            ep_ret += reward
            ep_len += 1
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
    env.close()
    return {
        'EpRetTest': ep_rets,
        'EpLenTest': ep_lens,
        **infos
    }

# def is_child_process():
#     # not suitable for processpool_executor
#     # if the current process is main process (multiprocessing.current_process().name == 'MainProcess'), its parent_process func will return None
#     return multiprocessing.parent_process() is not None

def eval_inprocess(policy_args, policy_state_dict, env_name, seed, env_making_seed, nrollout, eval_idx=None, ppid=None, base_algorithm='sac', device:str=None):
    # must not have ``self'' in the following scope
    if os.getpid() == ppid or ppid is None:
        print(
            f'WARNING: the function eval_inprocess re-seeded the random number generator, this function should be called in a child process!! PID: {os.getpid()}')
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    env_info = make_env(env_name, env_making_seed)
    policy = make_policy_model(policy_args, base_algorithm, not env_info['act_continuous'])
    policy.load_state_dict(policy_state_dict)
    env = env_info['eval_env']
    act_dim = env_info['act_dim']
    env.seed(seed + 5)
    env.action_space.seed(seed + 6)
    env.observation_space.seed(seed + 7)
    result = policy_eval(env, env_info, act_dim, policy, nrollout, device)
    return result, eval_idx



