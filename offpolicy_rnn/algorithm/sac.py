import concurrent.futures.process
import copy
import math
import time

from ..buffers.replay_memory import MemoryArray, Transition
from ..config.load_config import init_smart_logger
from ..models.contextual_model import ContextualModel
from ..parameter.ParameterSAC import Parameter
from ..policy_value_models.contextual_sac_value import ContextualSACValue
from ..policy_value_models.contextual_sac_policy import ContextualSACPolicy
from ..policy_value_models.contextual_sac_discrete_value import ContextualSACDiscreteValue
from ..policy_value_models.contextual_sac_discrete_policy import ContextualSACDiscretePolicy
from ..utility.timer import Timer
from ..utility.sample_utility import unorm_act, norm_act, n2t, n2t_2dim, t2n, eval_inprocess, policy_eval
from ..env_utils.make_env import make_env
from ..utility.ValueScheduler import CosineScheduler
from ..utility.count_parameters import count_parameters
from typing import List, Union, Tuple, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
import random
import torch
import smart_logger
from smart_logger import Logger
import os
import gym
from ..policy_value_models.make_models import make_policy_model, make_value_model


class SAC:
    def __init__(self, parameter):
        # 1. smart_logger init and parameter init
        self.parameter = parameter
        self.timer = Timer()
        self.logger = Logger(log_name=self.parameter.short_name)
        self.parameter.set_config_path(os.path.join(self.logger.output_dir, 'config'))
        self.parameter.save_config()
        self.logger(self.parameter)
        # 2. make env
        self.env_name = self.parameter.env_name
        self.env_info = make_env(self.env_name, self.parameter.seed)
        self.eval_env_info = make_env(self.env_name, self.parameter.seed + 1)
        self.env: gym.Env = self.env_info['train_env']
        self.eval_env: gym.Env = self.eval_env_info['train_env']
        self.max_episode_steps = self.env_info['max_trajectory_len']
        self.discrete_env = not self.env_info['act_continuous']
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env_info['act_dim']
        # 3. set random seed
        self._seed(self.parameter.seed)
        # 4. make policy and value
        self.policy_args = self._make_policy_args(self.parameter)
        self.value_args = self._make_value_args(self.parameter)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # if torch.has_mps:
        #     self.device = torch.device('mps')
        self.sample_device = self.device if self.parameter.cuda_inference else torch.device('cpu')
        self.optim_class = torch.optim.AdamW
        self.base_algorithm = self.parameter.base_algorithm if hasattr(self.parameter, 'base_algorithm') else 'sac'
        self.policy = make_policy_model(self.policy_args, self.base_algorithm, self.discrete_env)
        self.values: List[ContextualSACDiscreteValue] = [make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in range(self.parameter.value_net_num)]
        self.target_values: List[ContextualSACDiscreteValue] = [make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in range(self.parameter.value_net_num)]
        for value in self.values + self.target_values:
            value.to(self.device)
        self.policy.to(self.sample_device)
        self._value_update(tau=0.0)     # hard update
        self.logger(f'[Action space]: {self.env.action_space}')
        self.logger(f'[Observation space]: {self.env.observation_space}')
        if self.discrete_env:
            self.logger(f'[Notice] Action space {self.parameter.env_name} is discrete.')
            self.parameter.no_alpha_auto_tune = True
        if hasattr(self.parameter, 'no_alpha_auto_tune') and self.parameter.no_alpha_auto_tune:
            self.log_sac_alpha = torch.Tensor([math.log(self.parameter.sac_alpha)]).to(self.device).to(torch.get_default_dtype()).requires_grad_(True)
        else:
            self.log_sac_alpha = torch.Tensor([0.0]).to(self.device).to(torch.get_default_dtype()).requires_grad_(True)
        self.target_entropy = -np.prod(self.env.action_space.shape) * self.parameter.target_entropy_ratio if not self.discrete_env else self.parameter.target_entropy_ratio
        # 5. make policy and value optimizer
        self.optimizer_policy = self.optim_class(self.policy.parameters(True),
                                                 lr=self.parameter.policy_lr,
                                                 weight_decay=self.parameter.policy_l2_norm)
        value_parameters = [param for value in self.values for param in value.parameters(True)]
        value_embedding_parameters = [param for value in self.values for param in value.embedding_network.parameters(True)]

        self.value_parameters = value_parameters
        self.value_embedding_parameters = value_embedding_parameters
        self.optimizer_value = self.optim_class(value_parameters, lr=self.parameter.value_lr, weight_decay=self.parameter.value_l2_norm)
        self.optimizer_alpha = self.optim_class([self.log_sac_alpha], lr=self.parameter.alpha_lr)
        for value in self.values:
            value.train()
        for target_value in self.target_values:
            target_value.eval()
        self.policy.train()
        # 6. make replay buffer
        self.replay_buffer = MemoryArray(self.parameter.max_buffer_traj_num, smart_logger.get_customized_value('MAX_TRAJ_STEP'))

        # 7. init variables
        self.state_np = np.zeros((1, self.obs_dim))
        self.last_action_np = np.zeros((1, self.act_dim))
        self.last_state_np = np.zeros((1, self.obs_dim))
        self.reward_np = np.zeros((1, 1))
        self.sample_hidden = self._init_sample_hidden()

        self.env_reset()

        # 8. statistic variables
        self.sample_num = 0
        self.grad_num = 0
        self.start_time = time.time()

        # 9. sample process

        self.process_context = multiprocessing.get_context("spawn")
        self.process_pool = ProcessPoolExecutor(max_workers=self.parameter.test_nprocess, mp_context=self.process_context)
        self.instable_env = False

        # 10. learning scheduler [optional], if requiring setting the schedulers reload the init_lr_scheduler function
        self.actor_lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.critic_lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.target_entropy_scheduler: Optional[CosineScheduler] = None

        # 11. check whether nest-stack is allowed
        self.allow_nest_stack = self.allow_nest_stack_trajs()
        self.logger(f'policy parameter num: {count_parameters(self.policy)}, ({count_parameters(self.policy) / 1e9:.4f}) B')
        self.logger(f'value[0] parameter num: {count_parameters(self.values[0])}, ({count_parameters(self.values[0]) / 1e9:.4f}) B')


    def allow_nest_stack_trajs(self):
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network]:
            for i in range(len(rnn_base.layer_type)):
                if 'transformer' in rnn_base.layer_type[i]:
                    return False
                if 'gru' in rnn_base.layer_type[i]:
                    return False
        return True

    def init_lr_scheduler(self):
        pass

    def _init_sample_hidden(self):
        if self.parameter.randomize_first_hidden:
            sample_hidden = self.policy.make_init_state(1, self.sample_device)
        else:
            sample_hidden = self.policy.make_rnd_init_state(1, self.sample_device)
        return sample_hidden

    def env_reset(self):
        if hasattr(self.env, 'meta_env_flag') and hasattr(self.env, 'n_tasks') and self.env.n_tasks is not None:
            train_tasks = self.env_info['train_tasks']
            self.state_np = self.env.reset(train_tasks[np.random.randint(0, len(train_tasks))])
        else:
            self.state_np = self.env.reset()
        self.state_np = self.state_np.reshape((1, -1))
        self.last_action_np = np.zeros((1, self.act_dim))
        self.last_state_np = np.zeros((1, self.obs_dim))
        self.reward_np = np.zeros((1, 1))
        self.sample_hidden = self._init_sample_hidden()


    def env_step(self, next_obs: np.ndarray, act: Union[np.ndarray, float], reward: float, done: bool):
        self.last_state_np = self.state_np.copy()
        self.state_np = next_obs.copy()
        self.state_np = self.state_np.reshape((1, -1))
        if self.discrete_env:
            self.last_action_np = np.zeros((1, self.act_dim))
            self.last_action_np[..., int(act)] = 1
        else:
            self.last_action_np = act.copy()
        self.reward_np = np.array([[reward]])

        if done:
            self.env_reset()

    def _seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.env.seed(seed + 5)
        self.env.action_space.seed(seed + 6)
        self.env.observation_space.seed(seed + 7)

    def _value_update(self, tau):
        """
            if tau = 0, self <--- src_net
            if tau = 1, self <--- self
        """
        for value, target_value in zip(self.values, self.target_values):
            value: ContextualSACValue
            target_value: ContextualSACValue
            target_value.copy_weight_from(value, tau)

    def _make_policy_args(self, parameter: Parameter) -> Dict[str, Union[int, float, List[int]]]:
        policy_args = {
            'state_dim': self.obs_dim,
            'action_dim': self.act_dim,
            'embedding_size': parameter.policy_embedding_dim,
            'embedding_hidden': parameter.policy_embedding_hidden_size,
            'embedding_activations': parameter.policy_embedding_activations,
            'embedding_layer_type': parameter.policy_embedding_layer_type,
            'uni_model_hidden': parameter.policy_hidden_size,
            'uni_model_activations': parameter.policy_activations,
            'uni_model_layer_type': parameter.policy_layer_type,
            'fix_rnn_length': parameter.rnn_fix_length,
            'reward_input': parameter.reward_input,
            'last_action_input': not parameter.no_last_action_input,
            'last_state_input': hasattr(parameter, 'last_state_input') and parameter.last_state_input,
            'uni_model_input_mapping_dim': parameter.policy_uni_model_input_mapping_dim,
            'separate_encoder': hasattr(parameter, 'state_action_encoder') and parameter.state_action_encoder
        }
        if hasattr(parameter, 'base_algorithm') and parameter.base_algorithm == 'td3':
            policy_args['sample_std'] = parameter.sample_std
        return policy_args

    def _make_value_args(self, parameter: Parameter) -> Dict[str, Union[int, float, List[int]]]:
        value_args = {
            'state_dim': self.obs_dim,
            'action_dim': self.act_dim,
            'embedding_size': parameter.value_embedding_dim,
            'embedding_hidden': parameter.value_embedding_hidden_size,
            'embedding_activations': parameter.value_embedding_activations,
            'embedding_layer_type': parameter.value_embedding_layer_type,
            'uni_model_hidden': parameter.value_hidden_size,
            'uni_model_activations': parameter.value_activations,
            'uni_model_layer_type': parameter.value_layer_type,
            'fix_rnn_length': parameter.rnn_fix_length,
            'reward_input': parameter.reward_input,
            'last_action_input': not parameter.no_last_action_input,
            'last_state_input': hasattr(parameter, 'last_state_input') and parameter.last_state_input,
            'uni_model_input_mapping_dim': parameter.value_uni_model_input_mapping_dim,
            'separate_encoder': hasattr(parameter, 'state_action_encoder') and parameter.state_action_encoder
        }
        return value_args

    def warmup(self):
        self.env_reset()
        sample_cnt = 0
        while sample_cnt < self.parameter.random_num:
            done = False
            traj_len = 0
            while not done:
                act = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(act)
                act_normalized = norm_act(act, self.env.action_space)
                traj_len += 1
                self.replay_buffer.mem_push(Transition(
                    state=self.state_np,
                    last_state=self.last_state_np,
                    last_action=self.last_action_np,
                    action=act_normalized,
                    next_state=next_state.reshape((1, -1)),
                    reward=reward,
                    logp=None,
                    mask=1,
                    done=done,
                    timeout=traj_len >= self.max_episode_steps,
                    start=traj_len==1,
                    reward_input=self.reward_np,
                ))
                # done, transit state in env_step
                self.env_step(next_state, act_normalized, reward, done)
                sample_cnt += 1
        return sample_cnt

    def train_one_batch(self) -> Dict:
        return {}

    def train(self):
        self.logger(f'warming up! random test: random: {random.random()}, numpy: {np.random.random()}, torch: {torch.randn((1,))}, torch cuda: {torch.randn((1,), device=self.device).item()}')
        random_sample_num = self.warmup()
        self.logger(f'warmup done! got {random_sample_num} samples')
        self.sample_num += random_sample_num
        self.env_reset()
        ep_ret = 0
        ep_length = 0
        for iter_train in range(self.parameter.total_iteration):
            self.logger(f'random test: random: {random.random()}, numpy: {np.random.random()}, torch: {torch.randn((1,))}, torch cuda: {torch.randn((1,), device=self.device).item()}')
            # eval policy in additional process
            self.policy.to(torch.device('cpu'))
            state_dict = copy.deepcopy(self.policy.state_dict())
            if self.instable_env:
                self.process_pool = ProcessPoolExecutor(max_workers=self.parameter.test_nprocess,
                                                        mp_context=self.process_context)
            try:
                test_futures = [self.process_pool.submit(eval_inprocess, self.policy_args, state_dict, self.env_name, random.randint(0, 10000000),
                                                         self.env_info['seed'], self.parameter.test_nrollout, future_idx, os.getpid(), self.parameter.base_algorithm, 'cuda:0' if self.parameter.test_nprocess == 1 else 'cpu') for future_idx in range(self.parameter.test_nprocess)]
            except concurrent.futures.process.BrokenProcessPool:
                self.process_pool = ProcessPoolExecutor(max_workers=self.parameter.test_nprocess,
                                                        mp_context=self.process_context)
                test_futures = [self.process_pool.submit(eval_inprocess, self.policy_args, state_dict, self.env_name,
                                                         random.randint(0, 10000000),
                                                         self.env_info['seed'], self.parameter.test_nrollout,
                                                         future_idx, os.getpid(), self.parameter.base_algorithm, 'cuda:0' if self.parameter.test_nprocess == 1 else 'cpu') for future_idx in
                                range(self.parameter.test_nprocess)]
            # test_result = {}
            # for _ in range(3):
            #     try:
            #         test_result = policy_eval(self.eval_env, self.eval_env_info, self.act_dim, self.policy, self.parameter.test_nrollout, device=self.device)
            #         break
            #     except Exception as _:
            #         import traceback
            #         traceback.print_exc()
            # else:
            #     test_result = {}


            self.policy.train()
            self.policy.to(self.sample_device)
            # start training
            for step in tqdm(range(self.parameter.step_per_iteration)):
                # sample from env and save to replay buffer
                self.timer.register_point(tag='sample_in_env', level=1)
                with torch.no_grad():
                    _, _, act_sample, _, self.sample_hidden, _ = self.policy.forward(
                        state=n2t_2dim(self.state_np, self.sample_device),
                        lst_state=n2t_2dim(self.last_state_np, self.sample_device),
                        lst_action=n2t_2dim(self.last_action_np, self.sample_device),
                        rnn_memory=self.sample_hidden,
                        reward=n2t_2dim(self.reward_np, self.sample_device)
                    )
                    act_sample = t2n(act_sample)
                    act_sample = act_sample.squeeze(0)
                    act_unormalized = unorm_act(act_sample, self.env.action_space)
                if self.discrete_env:
                    act_unormalized = int(act_unormalized)
                next_state, reward, done, info = self.env.step(act_unormalized)
                self.timer.register_end(level=1)
                ep_ret += reward
                ep_length += 1
                self.timer.register_point(tag='mem_push', level=1)
                self.replay_buffer.mem_push(Transition(
                        state=self.state_np,
                        last_action=self.last_action_np,
                        last_state=self.last_state_np,
                        action=act_sample,
                        next_state=next_state.reshape((1, -1)),
                        reward=reward,
                        logp=None,
                        mask=1,
                        done=done,
                        timeout=ep_length >= self.max_episode_steps,
                        start=ep_length == 1,
                        reward_input=self.reward_np,
                    )
                )
                self.timer.register_end(level=1)
                self.env_step(next_state, act_sample, reward, done)

                if done:
                    self.logger.add_tabular_data(tb_prefix='Train', EpRet=ep_ret, EpLength=ep_length)
                    ep_ret = 0
                    ep_length = 0
                if self.sample_num % self.parameter.update_interval == 0 and self.sample_num >= self.parameter.start_train_num:
                    training_log = self.train_one_batch()
                    self.logger.add_tabular_data(tb_prefix='train', **training_log)
                    self.grad_num += 1
                self.sample_num += 1
            test_results = []
            for future in as_completed(test_futures):
                try:
                    test_results.append(future.result())
                except Exception:
                    import traceback
                    traceback.print_exc()
            if len(test_results) > 0:
                self.logger(f'fetched {len(test_results)}/{self.parameter.test_nprocess} data')
                test_results = sorted(test_results, key=lambda x: x[1])
                for test_log, _ in test_results:
                    self.logger.add_tabular_data(tb_prefix='performance', **test_log)

            else:
                self.logger(f'fail to fetch any data')
                self.instable_env = True
            # self.logger.add_tabular_data(tb_prefix='performance', **test_result)
            self.logger.log_tabular('iteration', iter_train, tb_prefix='timestep')
            self.logger.log_tabular('timestep', self.sample_num, tb_prefix='timestep')
            self.logger.log_tabular('grad_num', self.grad_num, tb_prefix='timestep')
            self.logger.log_tabular('time', time.time() - self.start_time, tb_prefix='timestep')
            self.logger.log_tabular('memory_size', self.replay_buffer.size, tb_prefix='buffer')
            self.logger.log_tabular('memory_trajectory_num', len(self.replay_buffer), tb_prefix='buffer')
            self.logger.add_tabular_data(tb_prefix='timer', **self.timer.summary(summation=True))
            self.logger.dump_tabular()
            self.logger(f'logdir: {self.logger.output_dir}')
            if iter_train % 25 == 0:
                self.save()
            if iter_train % 50 == 0 and self.parameter.backing_log:
                self.logger.sync_log_to_remote(replace=iter_train == 0, trial_num=5)
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()
            if self.critic_lr_scheduler is not None:
                self.critic_lr_scheduler.step()
            if self.target_entropy_scheduler is not None:
                self.target_entropy_scheduler.step()
                self.target_entropy = self.target_entropy_scheduler.get_value()
        # self.logger.sync_log_to_remote(replace=False, trial_num=5)

    def save(self, model_dir=None):
        model_path = os.path.join(self.logger.output_dir, 'model') if model_dir is None else model_dir
        self.policy.save(model_path)
        for i in range(len(self.values)):
            self.values[i].save(model_path, index=f'{i}')
            self.target_values[i].save(model_path, index=f'{i}-target')
        torch.save(self.log_sac_alpha, os.path.join(model_path, 'log_sac_alpha.pt'))

    def load(self, model_dir=None, load_policy=True, load_value=True):
        model_path = os.path.join(self.logger.output_dir, 'model') if model_dir is None else model_dir
        if load_policy:
            self.policy.load(model_path, map_location=self.sample_device)
        if load_value:
            for i in range(len(self.values)):
                self.values[i].load(model_path, index=f'{i}', map_location=self.device)
                self.target_values[i].load(model_path, index=f'{i}-target', map_location=self.device)
        self.log_sac_alpha = torch.load(os.path.join(model_path, 'log_sac_alpha.pt'), map_location=self.device)


