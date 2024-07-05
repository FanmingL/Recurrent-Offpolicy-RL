import random
from typing import Dict

from .sac import SAC
from ..buffers.replay_memory import Transition
import torch
from ..utility.sample_utility import n2t
from ..models.torch_utility import get_gradient_stats
import numpy as np


class SAC_MLP_REDQ(SAC):
    def __init__(self, parameter):
        super().__init__(parameter)
        assert self.policy.uni_network.rnn_num + self.policy.embedding_network.rnn_num == 0, f'sac mlp does not support rnn policy'
        assert sum([value_net.uni_network.rnn_num + value_net.embedding_network.rnn_num for value_net in self.values]) == 0, f'sac mlp does not support rnn value function'
        assert self.parameter.rnn_fix_length == 0, f'sac mlp only support rnn_fix_length==0, got {self.parameter.rnn_fix_length}'

    def train_one_batch(self) -> Dict:
        # 0. move policy to GPU
        self.policy.to(self.device)
        policy_update_cnt = 0
        for utd_idx in range(self.parameter.utd):
            # 1. sample from replay buffer
            batch_data = self.replay_buffer.sample_transitions(self.parameter.sac_batch_size)
            # ('state', 'last_action', 'action', 'next_state', 'reward', 'logp', 'mask', 'done')
            state, last_state, action, last_action, next_state, done, reward, reward_input, timeout = map(lambda x: getattr(batch_data, x), [
                'state', 'last_state', 'action', 'last_action', 'next_state', 'done', 'reward', 'reward_input', 'timeout'
            ])
            # self.logger(state.shape, action.shape, last_action.shape, next_state.shape, done.shape, mask.shape, reward.shape)
            # (batch_size, 17) (batch_size, 6) (batch_size, 6) (batch_size, 17) (batch_size, 1) (batch_size, 1) (batch_size, 1)
            # 2. update networks
            state, last_state, action, last_action, next_state, done, reward, reward_input, timeout = map(lambda x: n2t(x, self.device), [
                state, last_state, action, last_action, next_state, done, reward, reward_input, timeout
            ])
            done[timeout > 0] = 0
            alpha = self.log_sac_alpha.exp()
            alpha_detach = alpha.detach()
            sampled_target_value_nets = [self.target_values[i] for i in np.random.permutation(self.parameter.value_net_num)[:self.parameter.redq_m]]
            with torch.no_grad():
                _, _, next_act_sample, next_log_prob, _, _ = self.policy.forward(next_state, state, action, None, reward)
                next_target_Qs = [target_q_net.forward(next_state, state, action, next_act_sample, None, reward)[0] for target_q_net in sampled_target_value_nets]
                # min_i next_Q_i
                min_next_target_Q = torch.cat(next_target_Qs, dim=-1).min(dim=-1, keepdim=True).values - alpha_detach * next_log_prob
                target_Q = reward + (1 - done) * self.parameter.gamma * min_next_target_Q
            # 2.2 compute Q value
            Qs = [q_net.forward(state, last_state, last_action, action, None, reward_input)[0] for q_net in self.values]
            Q_loss_list = [(q_item - target_Q).pow(2).mean() for q_item in Qs]
            Q_loss: torch.Tensor = sum(Q_loss_list)

            # 2.3 optimize the value model
            self.optimizer_value.zero_grad()
            Q_loss.backward()
            value_grad_min, value_grad_max, value_grad_l2_norm_square = get_gradient_stats(self.value_parameters)
            self.optimizer_value.step()
            self._value_update(tau=self.parameter.sac_tau)
            if (utd_idx + 1) / self.parameter.utd * self.parameter.policy_utd > policy_update_cnt:
                # 2.4 compute actor loss
                _, _, act_sample, log_prob, _, _ = self.policy.forward(state, last_state, last_action, None, reward_input)
                Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, None, reward_input)[0] for q_net in
                             self.values]

                mean_Q = torch.cat(Qs_policy, dim=-1).mean(dim=-1, keepdim=True)

                actor_loss = ((alpha_detach * log_prob) - mean_Q).mean()
                # 2.5 optimize actor
                self.optimizer_policy.zero_grad()
                actor_loss.backward()
                policy_grad_min, policy_grad_max, policy_grad_l2_norm_square = get_gradient_stats(
                    self.policy.parameters())
                self.optimizer_policy.step()

                # 2.6 optimize alpha
                alpha_loss = -(self.log_sac_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.optimizer_alpha.zero_grad()
                alpha_loss.backward()
                self.optimizer_alpha.step()
                with torch.no_grad():
                    self.log_sac_alpha.clamp_max_(1)
                policy_update_cnt += 1

        # move policy to sample device
        self.policy.to(self.sample_device)
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': Q_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'log_alpha': self.log_sac_alpha.item(),
            'log_prob': log_prob.mean().item(),
            'q1_mean': Qs[0].mean().item(),
            'target_q_max': target_Q.abs().max().item(),
            'policy_l2_norm_square': self.policy.l2_norm_square().item(),
            'q1_l2_norm_square': self.values[0].l2_norm_square().item(),
            'policy_grad_min': policy_grad_min,
            'policy_grad_max': policy_grad_max,
            'policy_grad_l2_norm_square': policy_grad_l2_norm_square,
            'value_grad_min': value_grad_min,
            'value_grad_max': value_grad_max,
            'value_grad_l2_norm_square': value_grad_l2_norm_square,
        }