import math
from typing import Dict

from .sac import SAC
from ..buffers.transition_buffer.nested_replay_memory import NestedMemoryArray as NestedTransitionMemoryArray
from ..utility.q_value_guard import QValueGuard
import smart_logger
import torch
from ..utility.sample_utility import n2t
from ..policy_value_models.contextual_sac_policy import ContextualSACPolicy
from ..policy_value_models.contextual_sac_discrete_policy import ContextualSACDiscretePolicy
import copy
from ..policy_value_models.make_models import make_policy_model
from torch.cuda.amp import GradScaler

class SACFullLengthRNNEnsembleQ(SAC):
    def __init__(self, parameter):
        super().__init__(parameter)
        assert self.parameter.value_net_num == 1
        for item in self.parameter.value_layer_type:
            assert item.startswith('e')
        # for item in self.parameter.value_embedding_layer_type:
        #     assert item.startswith('e')

        for network in (self.values[0].embedding_network.layer_list + self.target_values[0].embedding_network.layer_list
                        + self.values[0].uni_network.layer_list + self.target_values[0].uni_network.layer_list):
            if hasattr(network, 'desire_ndim'):
                print(f'set desire_ndim')
                network.desire_ndim = 4
            if hasattr(network, 'in_proj') and hasattr(network.in_proj, 'desire_ndim'):
                print(f'set desire_ndim')
                network.in_proj.desire_ndim = 4
        self.logger(f'replay buffer skip len: {self._get_skip_len()}')
        if self._get_whether_require_amp():
            self.logger(f'enable AMP, introducing grad scalar!!')
            self.amp_scalar = GradScaler()
            self.amp_scalar_critic = GradScaler()
        else:
            self.amp_scalar = None
            self.amp_scalar_critic = None
        self.replay_buffer = NestedTransitionMemoryArray(self.parameter.max_buffer_transition_num, self.env_info['max_trajectory_len'], additional_history_len=self._get_skip_len())
        # self.Q_guard = QValueGuard(guard_min=True, guard_max=True, decay_ratio=1.0)
        if self.discrete_env:
            self.Q_guard = QValueGuard(guard_min=True, guard_max=True, decay_ratio=1.0)
        else:
            self.Q_guard = QValueGuard(guard_min=True, guard_max=True, decay_ratio=1-1e-3)

        # TODO: use target policy here
        self.target_policy = make_policy_model(self.policy_args, self.base_algorithm, self.discrete_env)
        self.target_policy.to(self.device)
        self.target_policy.copy_weight_from(self.policy, tau=0.0)
        self.target_policy.eval()
        # tune config

        self.logger(f'random test: policy L2 norm: {self.policy.l2_norm_square()}, value[0] L2 norm: {self.values[0].l2_norm_square()}')

    def _get_skip_len(self):
        skip_len = 0
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network]:
            for i in range(len(rnn_base.layer_type)):
                if 'smamba' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].d_conv, skip_len)
                elif 'mamba' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].mixer.d_conv, skip_len)
                elif 'conv1d' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].d_conv, skip_len)
        return skip_len + 1

    def _get_whether_require_amp(self):
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network]:
            for i in range(len(rnn_base.layer_type)):
                if 'gpt' in rnn_base.layer_type[i]:
                    return True
                    # try disabling GradScaler for GPT
                    # return False
        return False

    def _mask_mean(self, data: torch.Tensor, mask: torch.Tensor, valid_num: float) -> torch.Tensor:
        return (data * mask).sum() / valid_num

    def _target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                  target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            # full_state = torch.cat((state[..., :1, :], next_state), dim=-2)
            # full_last_action = torch.cat((last_action[..., :1, :], action), dim=-2)
            # full_reward = torch.cat((reward_input[..., :1, :], reward), dim=-2)
            # pi(s_0) ~ pi (s_T)
            _, _, next_act_sample, next_log_prob, _, _ = self.policy.forward(next_state, state, action,
                                                                                    policy_hidden, reward)
            # Q'(s_0) ~ Q' (s_T)
            next_target_Qs = [
                target_q_net.forward(next_state, state, action, next_act_sample, target_value_hiddens[q_ind],
                                     reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)

            # min_i next_Q_i
            min_next_target_Q = next_target_Qs[0].min(dim=0, keepdim=False).values - alpha_detach * next_log_prob
            target_Q = reward + (1 - done) * self.parameter.gamma * self.Q_guard.clamp(min_next_target_Q)

        return target_Q

    def _Q_loss(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        Q_output = [q_net.forward(state, last_state, last_action, action, value_hiddens[q_ind], reward_input) for q_ind, q_net in
              enumerate(self.values)]

        Qs = [item[0] for item in Q_output]
        embeddings = [item[1] for item in Q_output]
        Q_loss_list = [self._mask_mean((q_item - target_Q.unsqueeze(0)).pow(2).sum(dim=0, keepdim=False), rnd_mask, valid_num) for q_item in Qs]
        Q_loss = sum(Q_loss_list)
        info = {'Q': Qs}
        return Q_loss, embeddings, info

    def _policy_loss(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask,
                     valid_num):
        action_mean, embedding_output, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, policy_hidden,
                                                                                     reward_input)
        # self.logger(f'policy_hidden_input shape: {policy_hidden[0].shape}, policy_hidden_full shape: {_full_rnn_memory_policy[0].shape}')
        # self.logger(f'policy hidden error {self.batch_cnt}.{utd_idx}: {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().mean()} pm {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().std()}')
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, value_hiddens[q_ind], reward_input, detach_embedding=True)[0] for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = Qs_policy[0].min(dim=0, keepdim=False).values
        actor_loss = self._mask_mean((alpha_detach * log_prob) - min_Q, rnd_mask, valid_num)
        info = {'action_mean': action_mean, 'embedding_output': embedding_output, 'action_sample': act_sample}
        return actor_loss, log_prob, embedding_output, info

    def _alpha_loss(self, log_alpha, log_prob, rnd_mask, valid_num):
        alpha_loss = -self._mask_mean(log_alpha * (log_prob + self.target_entropy).detach(), rnd_mask, valid_num)
        return alpha_loss

    def _target_Q_discrete(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                           target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            action_one_hot = self.policy.action2onehot(action.long())
            # pi(s_0) ~ pi (s_T)
            _, _, next_act_sample, next_log_prob, _, _ = self.policy.forward(next_state, state, action,
                                                                             policy_hidden, reward)
            # Q'(s_0) ~ Q' (s_T)
            next_target_Qs = [
                target_q_net.forward(next_state, state, action_one_hot, next_act_sample, target_value_hiddens[q_ind],
                                     reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)
            # min_i next_Q_i
            min_next_target_Q = (next_target_Qs[0].min(dim=0, keepdim=False).values - alpha_detach * next_log_prob) * next_log_prob.exp()
            min_next_target_Q = min_next_target_Q.sum(dim=-1, keepdim=True)
            target_Q = reward + (1 - done) * self.parameter.gamma * self.Q_guard.clamp(min_next_target_Q)
        return target_Q

    def _Q_loss_discrete(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        Q_output = [q_net.forward(state, last_state, last_action, action, value_hiddens[q_ind], reward_input) for q_ind, q_net in
              enumerate(self.values)]
        Qs = [item[0] for item in Q_output]
        embeddings = [item[1] for item in Q_output]
        Qs = [self.policy.select_with_action(action.long(), Qs[0][i]) for i in range(Qs[0].shape[0])]
        Q_loss_list = [self._mask_mean((q_item - target_Q).pow(2), rnd_mask, valid_num) for q_item in Qs]
        Q_loss = sum(Q_loss_list)
        info = {'Q': Qs}
        return Q_loss, embeddings, info

    def _policy_loss_discrete(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach,
                              rnd_mask, valid_num):
        action_mean, embedding_output, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, policy_hidden,
                                                                                     reward_input)
        # self.logger(f'policy_hidden_input shape: {policy_hidden[0].shape}, policy_hidden_full shape: {_full_rnn_memory_policy[0].shape}')
        # self.logger(f'policy hidden error {self.batch_cnt}.{utd_idx}: {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().mean()} pm {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().std()}')
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, value_hiddens[q_ind], reward_input, detach_embedding=True)[0].unsqueeze(-1)
                     for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = Qs_policy[0].min(dim=0, keepdim=False).values.squeeze(-1)
        actor_loss = (((alpha_detach * log_prob) - min_Q) * log_prob.exp()).sum(dim=-1, keepdim=True)
        actor_loss = self._mask_mean(actor_loss, rnd_mask, valid_num)
        info = {'action_mean': action_mean, 'embedding_output': embedding_output, 'action_sample': act_sample}
        return actor_loss, log_prob, embedding_output, info

    def _alpha_loss_discrete(self, log_alpha, log_prob, rnd_mask, valid_num):
        probs = log_prob.exp()
        neg_entropy = (log_prob * probs).sum(dim=-1, keepdim=True)
        alpha_loss = -self._mask_mean(log_alpha * (neg_entropy + self.target_entropy).detach(), rnd_mask,
                                      valid_num)
        return alpha_loss

    def get_target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                     target_value_hiddens, alpha_detach):
        if self.discrete_env:
            return self._target_Q_discrete(state, last_state, action, last_action, next_state, done, reward, reward_input,
                                           policy_hidden, target_value_hiddens, alpha_detach)
        else:
            return self._target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                                  target_value_hiddens, alpha_detach)

    def get_Q_loss(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        if self.discrete_env:
            return self._Q_loss_discrete(state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num,
                                         target_Q)
        else:
            return self._Q_loss(state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q)

    def get_policy_loss(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask,
                        valid_num):
        if self.discrete_env:
            return self._policy_loss_discrete(state, last_state, last_action, policy_hidden, reward_input, value_hiddens,
                                              alpha_detach, rnd_mask, valid_num)
        else:
            return self._policy_loss(state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach,
                                     rnd_mask, valid_num)

    def get_alpha_loss(self, log_alpha, log_prob, rnd_mask, valid_num):
        if self.discrete_env:
            return self._alpha_loss_discrete(log_alpha, log_prob, rnd_mask, valid_num)
        else:
            return self._alpha_loss(log_alpha, log_prob, rnd_mask, valid_num)

    def _mask_rnd_select(self, mask, select_num):
        mask_1d = mask.view((-1,))
        mask_idx = mask_1d.nonzero()
        mask_idx_set_zero = mask_idx[torch.randperm(mask_idx.shape[0], device=mask.device)[:-select_num]]
        mask_1d[mask_idx_set_zero] = 0

    def _optim_policy(self, state, last_state, last_action, policy_hidden, reward_input,
                                                            value_hiddens, alpha_detach, mask,
                                                            valid_num):
        actor_loss, log_prob, actor_embedding, _ = self.get_policy_loss(state, last_state, last_action, policy_hidden, reward_input,
                                                    value_hiddens, alpha_detach, mask,
                                                    valid_num)
        losses = {}

        # 2.5 optimize actor
        self.optimizer_policy.zero_grad()
        if self.amp_scalar is not None:
            self.amp_scalar.scale(actor_loss).backward()
        else:
            actor_loss.backward()
        pi_grad_norm = 0
        if self.parameter.policy_max_gradnorm is not None:
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.parameter.policy_max_gradnorm,
                                                          norm_type=2)
            pi_grad_norm = pi_grad_norm.item()
        if self.parameter.policy_embedding_max_gradnorm is not None:
            torch.nn.utils.clip_grad_value_(self.policy.embedding_network.parameters(), self.parameter.policy_embedding_max_gradnorm)
            for layer in self.policy.embedding_network.layer_list:
                try:
                    for sublayer in layer.layers:
                        torch.nn.utils.clip_grad_value_([sublayer.mixer.A_log],1e-3)
                except Exception as _:
                    pass

            pi_grad_norm = 0
        # policy_grad_min, policy_grad_max, policy_grad_l2_norm_square = get_gradient_stats(self.policy.parameters())
        if self.amp_scalar is not None:
            self.amp_scalar.step(self.optimizer_policy)
            self.amp_scalar.update()
        else:
            self.optimizer_policy.step()
        return pi_grad_norm, actor_loss, log_prob, losses

    def _optim_value(self, state, last_state, last_action, action, mask, reward_input, value_hiddens, valid_num, target_Q):
        Q_loss, Q_embeddings, _ = self.get_Q_loss(state, last_state, last_action, action, mask, reward_input, value_hiddens,
                                 valid_num,
                                 target_Q)
        losses = {}

        # 2.3 optimize the value model
        self.optimizer_value.zero_grad()
        if self.amp_scalar_critic is not None:
            self.amp_scalar_critic.scale(Q_loss).backward()
        else:
            Q_loss.backward()
        q_grad_norm = 0
        if self.parameter.value_max_gradnorm is not None:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_parameters, self.parameter.value_max_gradnorm,
                                                         norm_type=2)
            q_grad_norm = q_grad_norm.item()
        if self.parameter.value_embedding_max_gradnorm is not None:
            torch.nn.utils.clip_grad_value_(self.value_embedding_parameters, self.parameter.value_embedding_max_gradnorm)
            for q in self.values:
                for layer in q.embedding_network.layer_list:
                    try:
                        for sublayer in layer.layers:
                            torch.nn.utils.clip_grad_value_([sublayer.mixer.A_log], 1e-3)
                    except Exception as _:
                        pass
            q_grad_norm = 0.0

        # value_grad_min, value_grad_max, value_grad_l2_norm_square = get_gradient_stats(self.value_parameters)
        if self.amp_scalar_critic is not None:
            self.amp_scalar_critic.step(self.optimizer_value)
            self.amp_scalar_critic.update()
        else:
            self.optimizer_value.step()
        return Q_loss, q_grad_norm, losses

    def train_one_batch(self) -> Dict:
        # 0. move policy to GPU
        self.policy.to(self.device)
        policy_update_cnt = 0
        pi_part_losses = {}
        # actor_loss = torch.zeros((1,))
        # alpha_loss = torch.zeros((1,))
        # log_prob_mean = 0
        # pi_grad_norm = 0
        policy_losses = dict()
        # 1. sample from replay buffer
        pi_part_losses_tune = {}
        q_part_losses_tune = {}

        for utd_idx in range(self.parameter.utd):
            self.timer.register_point(tag='sample_trajs', level=2)
            batch_data, batch_size, traj_valid_indicators, traj_len_array = self.replay_buffer.sample_trajs(self.parameter.sac_batch_size,
                                                                     None, randomize_mask=self.parameter.randomize_mask,
                                                                     valid_number_post_randomized=self.parameter.valid_number_post_randomized,
                                                                     equalize_data_of_each_traj=True,
                                                                                            random_trunc_traj=self.parameter.random_trunc_traj,
                                                                                            nest_stack_trajs=self.allow_nest_stack)
            self.timer.register_end(level=2)

            # ('state', 'last_action', 'action', 'next_state', 'reward', 'logp', 'mask', 'done')
            state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout, rnn_start = map(
                lambda x: getattr(batch_data, x), [
                    'state', 'last_state', 'action', 'last_action', 'next_state', 'done', 'mask', 'reward', 'reward_input', 'timeout', 'start'
                ])
            # self.logger(state.shape, action.shape, last_action.shape, next_state.shape, done.shape, mask.shape, reward.shape)
            # (1, 1000, 17) (1, 1000, 6) (1, 1000, 6) (1, 1000, 17) (1, 1000, 1) (1, 1000, 1) (1, 1000, 1)
            # 2. update networks
            state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout, rnn_start, traj_valid_indicators = map(
                lambda x: n2t(x, self.device), [
                    state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout, rnn_start, traj_valid_indicators
                ])
            # must before covering done
            # rnn_start[:, 1, :] == 1
            # total_rnn_start[:, 0, :] == 1
            # mask[:, 1, :] == 1
            # mask[:, 0, :] == 0
            total_rnn_start = rnn_start.clone()
            total_valid_indicators = traj_valid_indicators.clone()
            total_valid_indicators[torch.where(torch.diff(traj_valid_indicators, dim=-2) == 1)] = 1
            total_rnn_start[torch.where(torch.diff(total_rnn_start, dim=-2) == -1)] = 0
            done[timeout > 0] = 0
            # valid_num = mask.sum().item()
            alpha_detach = self.log_sac_alpha.exp().detach()
            # set RNN termination for LRU
            if self.parameter.randomize_first_hidden:
                policy_hidden = self.policy.make_rnd_init_state(state.shape[0], device=self.device)
                target_policy_hidden = policy_hidden
                target_hiddens = [target_value.make_rnd_init_state(state.shape[0], device=self.device) for target_value in
                                  self.target_values]
                value_hiddens = [value.make_rnd_init_state(state.shape[0], device=self.device) for value in self.values]
            else:
                policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
                target_policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
                target_hiddens = [target_value.make_init_state(state.shape[0], device=self.device) for target_value in
                                  self.target_values]
                value_hiddens = [value.make_init_state(state.shape[0], device=self.device) for value in self.values]
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(self.device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=self.device)), dim=-1)
                target_attention_mask = torch.cat((attention_mask[..., 1:], torch.zeros(
                    (attention_mask.shape[0], 1), device=self.device)), dim=-1)
                attention_mask = attention_mask.clone()
                attention_mask = attention_mask.to(torch.int)
                target_attention_mask = target_attention_mask.to(torch.int)

            target_policy_hidden.set_rnn_start(total_rnn_start)
            target_policy_hidden.set_mask(total_valid_indicators)
            target_policy_hidden.set_attention_concat_mask(target_attention_mask)
            for item in value_hiddens:
                item.set_rnn_start(rnn_start)
                item.set_attention_concat_mask(attention_mask)
                item.set_mask(traj_valid_indicators)
            for item in target_hiddens:
                item.set_rnn_start(total_rnn_start)
                item.set_attention_concat_mask(target_attention_mask)
                item.set_mask(total_valid_indicators)
            # if self.parameter.randomize_mask:
            #     self._mask_rnd_select(mask, self.parameter.valid_number_post_randomized)
            # end of RNN termination for LRU
            self.policy.eval()
            self.target_policy.eval()
            target_Q = self.get_target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, target_policy_hidden,
                                         target_hiddens, alpha_detach)

            self.Q_guard.update(target_Q * mask)
            # 2.2 compute Q value
            for value in self.values:
                value.train()
            valid_num = mask.sum()
            Q_loss, q_grad_norm, q_part_losses = self._optim_value(state, last_state, last_action, action, mask, reward_input,
                                                    value_hiddens, valid_num, target_Q)
            q_part_losses.update(q_part_losses_tune)
            self._value_update(tau=self.parameter.sac_tau)
            for value in self.values:
                value.eval()
            self.policy.train()
            # rnn_start for all linear RNN
            policy_hidden.set_rnn_start(rnn_start)
            # mask for mamba and conv1d!
            policy_hidden.set_mask(traj_valid_indicators)
            policy_hidden.set_attention_concat_mask(attention_mask)
            # 2.4 compute actor loss
            if self.grad_num % self.parameter.policy_update_per == 0 and (utd_idx + 1) / self.parameter.utd * self.parameter.policy_utd > policy_update_cnt:
                pi_grad_norm, actor_loss, log_prob, pi_part_losses = self._optim_policy(state, last_state, last_action, policy_hidden, reward_input,
                                                            value_hiddens, alpha_detach, mask,
                                                            valid_num)
                # 2.6 optimize alpha
                if self.parameter.no_alpha_auto_tune:
                    pass
                else:
                    alpha_loss = self.get_alpha_loss(self.log_sac_alpha, log_prob, mask, valid_num)
                    self.optimizer_alpha.zero_grad()
                    alpha_loss.backward()
                    self.optimizer_alpha.step()
                    with torch.no_grad():
                        if self.discrete_env:
                            self.log_sac_alpha.clamp_max_(5)
                            self.log_sac_alpha.clamp_min_(-10)
                        else:
                            self.log_sac_alpha.clamp_max_(1)
                    policy_losses['alpha_loss'] = alpha_loss.item()
                if self.discrete_env:
                    log_prob_mean = self._mask_mean((log_prob * log_prob.exp()).sum(dim=-1, keepdim=True), mask, valid_num).item()
                else:
                    log_prob_mean = self._mask_mean(log_prob, mask, valid_num).item()
                policy_update_cnt += 1
                policy_losses['log_prob'] = log_prob_mean
                policy_losses['actor_loss'] = actor_loss.item(),
                policy_losses['policy_grad_norm'] = pi_grad_norm
                policy_losses['policy_l2_norm_square'] = self.policy.l2_norm_square().item()
        # move policy to sample device
        self.policy.to(self.sample_device)
        return {
            # 'actor_loss': actor_loss.item(),
            'critic_loss': Q_loss.item(),
            # 'alpha_loss': alpha_loss.item(),
            'log_alpha': self.log_sac_alpha.item(),
            # 'log_prob': log_prob_mean,
            'real_batch_size': batch_size,
            'real_batch_traj_num': state.shape[0],
            # 'q1_mean': self._mask_mean(Qs[0], mask, valid_num).item(),
            'target_q_max': target_Q.abs().max().item(),
            # 'policy_l2_norm': pi_l2norm.item(),
            # 'value_l2_norm': q_l2norm.item(),
            # 'policy_grad_norm': pi_grad_norm,
            'value_grad_norm': q_grad_norm,
            'clip_min': self.Q_guard.get_min() if self.Q_guard.get_min() is not None else 0,
            'clip_max': self.Q_guard.get_max() if self.Q_guard.get_max() is not None else 0,
            # **self.timer.summary(summation=True)
            # 'policy'
            # 'policy_l2_norm_square': self.policy.l2_norm_square().item(),
            'q1_l2_norm_square': self.values[0].l2_norm_square().item(),
            'average_traj_len': self.replay_buffer.size / len(self.replay_buffer),
            'amp_scalar_pi': self.amp_scalar.get_scale() if self.amp_scalar is not None else 0,
            'amp_scalar_q': self.amp_scalar_critic.get_scale() if self.amp_scalar_critic is not None else 0,
            **policy_losses,
            **q_part_losses,
            **pi_part_losses,
            # 'policy_grad_min': policy_grad_min,
            # 'policy_grad_max': policy_grad_max,
            # 'policy_grad_l2_norm_square': policy_grad_l2_norm_square,
            # 'value_grad_min': value_grad_min,
            # 'value_grad_max': value_grad_max,
            # 'value_grad_l2_norm_square': value_grad_l2_norm_square,
        }
