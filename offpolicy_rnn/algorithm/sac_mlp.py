from typing import Dict

from .sac import SAC
from ..buffers.replay_memory import Transition
import torch
from ..utility.sample_utility import n2t
from ..models.torch_utility import get_gradient_stats
class SAC_MLP(SAC):
    def __init__(self, parameter):
        super().__init__(parameter)
        assert self.policy.uni_network.rnn_num + self.policy.embedding_network.rnn_num == 0, f'sac mlp does not support rnn policy'
        assert sum([value_net.uni_network.rnn_num + value_net.embedding_network.rnn_num for value_net in self.values]) == 0, f'sac mlp does not support rnn value function'
        assert self.parameter.rnn_fix_length == 0, f'sac mlp only support rnn_fix_length==0, got {self.parameter.rnn_fix_length}'

    def _target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach):
        with torch.no_grad():
            _, _, next_act_sample, next_log_prob, _, _ = self.policy.forward(next_state, state, action, None, reward)
            next_target_Qs = [target_q_net.forward(next_state, state, action, next_act_sample, None, reward)[0] for target_q_net in self.target_values]
            # min_i next_Q_i
            min_next_target_Q = torch.cat(next_target_Qs, dim=-1).min(dim=-1, keepdim=True).values - alpha_detach * next_log_prob
            target_Q = reward + (1 - done) * self.parameter.gamma * min_next_target_Q
        return target_Q

    def _Q_loss(self, state, last_state, last_action, action, reward_input, target_Q):
        Qs = [q_net.forward(state, last_state, last_action, action, None, reward_input)[0] for q_net in self.values]
        Q_loss_list = [(q_item - target_Q).pow(2).mean() for q_item in Qs]
        Q_loss: torch.Tensor = sum(Q_loss_list)
        return Q_loss

    def _policy_loss(self, state, last_state, last_action, reward_input, alpha_detach):
        _, _, act_sample, log_prob, _, _ = self.policy.forward(state, last_state, last_action, None, reward_input)
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, None, reward_input)[0] for q_net in self.values]

        min_Q = torch.cat(Qs_policy, dim=-1).min(dim=-1, keepdim=True).values
        actor_loss = ((alpha_detach * log_prob) - min_Q).mean()
        return actor_loss, log_prob

    def _alpha_loss(self, log_alpha, log_prob):
        alpha_loss = -(log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return alpha_loss

    def _target_Q_discrete(self, state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            action_one_hot = self.policy.action2onehot(action.long())

            # pi(s_0) ~ pi (s_T)
            _, _, next_act_sample, next_log_prob, _, _ = self.policy.forward(next_state, state, action_one_hot,
                                                                             None, reward)
            next_target_Qs = [target_q_net.forward(next_state, state, action_one_hot, next_act_sample, None, reward)[0].unsqueeze(-1) for
                              target_q_net in self.target_values]
            # min_i next_Q_i
            min_next_target_Q = (torch.cat(next_target_Qs, dim=-1).min(dim=-1,
                                                                      keepdim=False).values - alpha_detach * next_log_prob) * next_log_prob.exp()
            min_next_target_Q = min_next_target_Q.sum(dim=-1, keepdim=True)
            target_Q = reward + (1 - done) * self.parameter.gamma * min_next_target_Q
        return target_Q

    def _Q_loss_discrete(self, state, last_state, last_action, action, reward_input, target_Q):
        Qs = [q_net.forward(state, last_state, last_action, action, None, reward_input)[0] for q_ind, q_net in
              enumerate(self.values)]
        Qs = [self.policy.select_with_action(action.long(), q) for q in Qs]
        Q_loss_list = [(q_item - target_Q).pow(2).mean() for q_item in Qs]
        Q_loss = sum(Q_loss_list)
        return Q_loss

    def _policy_loss_discrete(self, state, last_state, last_action, reward_input, alpha_detach):
        _, _, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, None,
                                                                                     reward_input)
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, None, reward_input)[0].unsqueeze(-1) for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = torch.cat(Qs_policy, dim=-1).min(dim=-1, keepdim=False).values
        actor_loss = (((alpha_detach * log_prob) - min_Q) * log_prob.exp()).sum(dim=-1, keepdim=True)
        actor_loss = actor_loss.mean()
        return actor_loss, log_prob

    def _alpha_loss_discrete(self, log_alpha, log_prob):
        probs = log_prob.exp()
        alpha_loss = -(probs.detach() * (-log_alpha * (log_prob + self.target_entropy).detach())).mean()
        return alpha_loss

    def get_target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach):
        if self.discrete_env:
            return self._target_Q_discrete(state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach)
        else:
            return self._target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach)

    def get_Q_loss(self, state, last_state, last_action, action, reward_input, target_Q):
        if self.discrete_env:
            return self._Q_loss_discrete(state, last_state, last_action, action, reward_input, target_Q)
        else:
            return self._Q_loss(state, last_state, last_action, action, reward_input, target_Q)

    def get_policy_loss(self, state, last_state, last_action, reward_input, alpha_detach):
        if self.discrete_env:
            return self._policy_loss_discrete(state, last_state, last_action, reward_input, alpha_detach)
        else:
            return self._policy_loss(state, last_state, last_action, reward_input, alpha_detach)


    def get_alpha_loss(self, log_alpha, log_prob):
        if self.discrete_env:
            return self._alpha_loss_discrete(log_alpha, log_prob)
        else:
            return self._alpha_loss(log_alpha, log_prob)


    def train_one_batch(self) -> Dict:
        # 0. move policy to GPU
        self.policy.to(self.device)
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
        target_Q = self.get_target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, alpha_detach)
        # 2.2 compute Q value

        Q_loss: torch.Tensor = self.get_Q_loss(state, last_state, last_action, action, reward_input, target_Q)

        # 2.3 optimize the value model
        self.optimizer_value.zero_grad()
        Q_loss.backward()
        # value_grad_min, value_grad_max, value_grad_l2_norm_square = get_gradient_stats(self.value_parameters)
        self.optimizer_value.step()

        # 2.4 compute actor loss
        actor_loss, log_prob = self.get_policy_loss(state, last_state, last_action, reward_input, alpha_detach)
        # 2.5 optimize actor
        self.optimizer_policy.zero_grad()
        actor_loss.backward()
        # policy_grad_min, policy_grad_max, policy_grad_l2_norm_square = get_gradient_stats(self.policy.parameters())
        self.optimizer_policy.step()

        # 2.6 optimize alpha
        alpha_loss = self.get_alpha_loss(self.log_sac_alpha, log_prob)
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()
        with torch.no_grad():
            self.log_sac_alpha.clamp_max_(1)

        # move policy to sample device
        self.policy.to(self.sample_device)
        self._value_update(tau=self.parameter.sac_tau)
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': Q_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'log_alpha': self.log_sac_alpha.item(),
            'log_prob': log_prob.mean().item(),
            # 'q1_mean': Qs[0].mean().item(),
            'target_q_max': target_Q.abs().max().item(),
            'policy_l2_norm_square': self.policy.l2_norm_square().item(),
            'q1_l2_norm_square': self.values[0].l2_norm_square().item(),
            # 'policy_grad_min': policy_grad_min,
            # 'policy_grad_max': policy_grad_max,
            # 'policy_grad_l2_norm_square': policy_grad_l2_norm_square,
            # 'value_grad_min': value_grad_min,
            # 'value_grad_max': value_grad_max,
            # 'value_grad_l2_norm_square': value_grad_l2_norm_square,
        }