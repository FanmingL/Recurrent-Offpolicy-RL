from typing import Dict

from .sac import SAC
from ..buffers.replay_memory import Transition
import torch
from ..utility.sample_utility import n2t
from ..buffers.replay_memory_tail_padding import MemoryArrayTailZeroPadding
import smart_logger
from ..models.torch_utility import get_gradient_stats

class SACRNNSlice(SAC):
    def __init__(self, parameter):
        super().__init__(parameter)
        assert self.parameter.rnn_slice_length > 0, f'sac mlp only support rnn_fix_length > 0, got {self.parameter.rnn_slice_length}'
        self.replay_buffer = MemoryArrayTailZeroPadding(self.parameter.max_buffer_traj_num,
                                                  smart_logger.get_customized_value('MAX_TRAJ_STEP'),
                                                  fixed_length=self.parameter.rnn_slice_length)

    def _mask_mean(self, data: torch.Tensor, mask: torch.Tensor, valid_num: float) -> torch.Tensor:
        return (data * mask).sum() / valid_num

    def _target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden, target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            full_state = torch.cat((state[..., :1, :], next_state), dim=-2)
            full_last_state = torch.cat((last_state[..., :1, :], state), dim=-2)
            full_last_action = torch.cat((last_action[..., :1, :], action), dim=-2)
            full_reward = torch.cat((reward_input[..., :1, :], reward), dim=-2)
            # pi(s_0) ~ pi (s_T)
            _, _, full_act_sample, full_log_prob, _, _ = self.policy.forward(full_state, full_last_state, full_last_action,
                                                                             policy_hidden, full_reward)
            # Q'(s_0) ~ Q' (s_T)
            full_target_Qs = [
                target_q_net.forward(full_state, full_last_state, full_last_action, full_act_sample, target_value_hiddens[q_ind],
                                     full_reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)
            next_target_Qs = [item[..., 1:, :] for item in full_target_Qs]
            next_log_prob = full_log_prob[..., 1:, :]
            # min_i next_Q_i
            min_next_target_Q = torch.cat(next_target_Qs, dim=-1).min(dim=-1,
                                                                      keepdim=True).values - alpha_detach * next_log_prob
            target_Q = reward + (1 - done) * self.parameter.gamma * min_next_target_Q
        return target_Q

    def _Q_loss(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        Qs = [q_net.forward(state, last_state, last_action, action, value_hiddens[q_ind], reward_input)[0] for q_ind, q_net in
              enumerate(self.values)]
        Q_loss_list = [self._mask_mean((q_item - target_Q).pow(2), rnd_mask, valid_num) for q_item in Qs]
        Q_loss = sum(Q_loss_list)

        return Q_loss

    def _policy_loss(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask, valid_num):
        _, _, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, policy_hidden,
                                                                                     reward_input)
        # self.logger(f'policy_hidden_input shape: {policy_hidden[0].shape}, policy_hidden_full shape: {_full_rnn_memory_policy[0].shape}')
        # self.logger(f'policy hidden error {self.batch_cnt}.{utd_idx}: {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().mean()} pm {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().std()}')
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, value_hiddens[q_ind], reward_input)[0] for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = torch.cat(Qs_policy, dim=-1).min(dim=-1, keepdim=True).values
        actor_loss = self._mask_mean((alpha_detach * log_prob) - min_Q, rnd_mask, valid_num)
        return actor_loss, log_prob

    def _alpha_loss(self, log_alpha, log_prob, rnd_mask, valid_num):
        alpha_loss = -self._mask_mean(log_alpha * (log_prob + self.target_entropy).detach(), rnd_mask, valid_num)
        return alpha_loss

    def _target_Q_discrete(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden, target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            action_one_hot = self.policy.action2onehot(action.long())
            full_state = torch.cat((state[..., :1, :], next_state), dim=-2)
            full_last_state = torch.cat((last_state[..., :1, :], state), dim=-2)
            full_last_action = torch.cat((last_action[..., :1, :], action_one_hot), dim=-2)
            full_reward = torch.cat((reward_input[..., :1, :], reward), dim=-2)
            # pi(s_0) ~ pi (s_T)
            _, _, full_act_sample, full_log_prob, _, _ = self.policy.forward(full_state, full_last_state, full_last_action,
                                                                             policy_hidden, full_reward)
            # Q'(s_0) ~ Q' (s_T)
            full_target_Qs = [
                target_q_net.forward(full_state, full_last_state, full_last_action, full_act_sample, target_value_hiddens[q_ind],
                                     full_reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)
            next_target_Qs = [item[..., 1:, :].unsqueeze(-1) for item in full_target_Qs]
            next_log_prob = full_log_prob[..., 1:, :]
            # min_i next_Q_i

            min_next_target_Q = (torch.cat(next_target_Qs, dim=-1).min(dim=-1,
                                                                      keepdim=False).values - alpha_detach * next_log_prob) * next_log_prob.exp()
            min_next_target_Q = min_next_target_Q.sum(dim=-1, keepdim=True)
            target_Q = reward + (1 - done) * self.parameter.gamma * min_next_target_Q
        return target_Q

    def _Q_loss_discrete(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        Qs = [q_net.forward(state, last_state, last_action, action, value_hiddens[q_ind], reward_input)[0] for q_ind, q_net in
              enumerate(self.values)]
        Qs = [self.policy.select_with_action(action.long(), q) for q in Qs]
        Q_loss_list = [self._mask_mean((q_item - target_Q).pow(2), rnd_mask, valid_num) for q_item in Qs]
        Q_loss = sum(Q_loss_list)
        return Q_loss

    def _policy_loss_discrete(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask, valid_num):
        _, _, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, policy_hidden,
                                                                                     reward_input)
        # self.logger(f'policy_hidden_input shape: {policy_hidden[0].shape}, policy_hidden_full shape: {_full_rnn_memory_policy[0].shape}')
        # self.logger(f'policy hidden error {self.batch_cnt}.{utd_idx}: {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().mean()} pm {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().std()}')
        Qs_policy = [q_net.forward(state, last_state, last_action, act_sample, value_hiddens[q_ind], reward_input)[0].unsqueeze(-1) for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = torch.cat(Qs_policy, dim=-1).min(dim=-1, keepdim=False).values
        actor_loss = (((alpha_detach * log_prob) - min_Q) * log_prob.exp()).sum(dim=-1, keepdim=True)
        actor_loss = self._mask_mean(actor_loss, rnd_mask, valid_num)
        return actor_loss, log_prob

    def _alpha_loss_discrete(self, log_alpha, log_prob, rnd_mask, valid_num):
        probs = log_prob.exp()
        alpha_loss = -self._mask_mean(probs.detach() * (-log_alpha * (log_prob + self.target_entropy).detach()), rnd_mask, valid_num)
        return alpha_loss

    def get_target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden, target_value_hiddens, alpha_detach):
        if self.discrete_env:
            return self._target_Q_discrete(state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden, target_value_hiddens, alpha_detach)
        else:
            return self._target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden, target_value_hiddens, alpha_detach)

    def get_Q_loss(self, state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q):
        if self.discrete_env:
            return self._Q_loss_discrete(state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q)
        else:
            return self._Q_loss(state, last_state, last_action, action, rnd_mask, reward_input, value_hiddens, valid_num, target_Q)

    def get_policy_loss(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask, valid_num):
        if self.discrete_env:
            return self._policy_loss_discrete(state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask, valid_num)
        else:
            return self._policy_loss(state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask, valid_num)


    def get_alpha_loss(self, log_alpha, log_prob, rnd_mask, valid_num):
        if self.discrete_env:
            return self._alpha_loss_discrete(log_alpha, log_prob, rnd_mask, valid_num)
        else:
            return self._alpha_loss(log_alpha, log_prob, rnd_mask, valid_num)


    def train_one_batch(self) -> Dict:
        # 0. move policy to GPU
        self.policy.to(self.device)
        policy_update_cnt = 0
        # 1. sample from replay buffer
        for utd_idx in range(self.parameter.utd):
            batch_data = self.replay_buffer.sample_fix_length_sub_trajs(self.parameter.sac_batch_size, self.parameter.rnn_slice_length)
            # ('state', 'last_action', 'action', 'next_state', 'reward', 'logp', 'mask', 'done')
            state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout = map(lambda x: getattr(batch_data, x), [
                'state', 'last_state', 'action', 'last_action', 'next_state', 'done', 'mask', 'reward', 'reward_input', 'timeout'
            ])
            # self.logger(state.shape, action.shape, last_action.shape, next_state.shape, done.shape, mask.shape, reward.shape)
            # (1, 1000, 17) (1, 1000, 6) (1, 1000, 6) (1, 1000, 17) (1, 1000, 1) (1, 1000, 1) (1, 1000, 1)
            # 2. update networks
            state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout = map(lambda x: n2t(x, self.device), [
                state, last_state, action, last_action, next_state, done, mask, reward, reward_input, timeout
            ])
            done[timeout > 0] = 0
            valid_num = mask.sum().item()
            alpha = self.log_sac_alpha.exp()
            alpha_detach = alpha.detach()
            target_Q = self.get_target_Q(state, last_state, action, last_action, next_state, done, reward, reward_input, None, [None for _ in self.target_values], alpha_detach)
            # 2.2 compute Q value
            Q_loss = self.get_Q_loss(state, last_state, last_action, action, mask, reward_input, [None for _ in self.values], valid_num,
                                     target_Q)
            # 2.3 optimize the value model
            self.optimizer_value.zero_grad()
            Q_loss.backward()
            # value_grad_min, value_grad_max, value_grad_l2_norm_square = get_gradient_stats(self.value_parameters)
            self.optimizer_value.step()
            self._value_update(tau=self.parameter.sac_tau)
            if (utd_idx + 1) / self.parameter.utd * self.parameter.policy_utd > policy_update_cnt:
                # 2.4 compute actor loss
                actor_loss, log_prob = self.get_policy_loss(state, last_state, last_action, None, reward_input,
                                                            [None for _ in self.target_values], alpha_detach, mask, valid_num)
                # 2.5 optimize actor
                self.optimizer_policy.zero_grad()
                actor_loss.backward()
                # policy_grad_min, policy_grad_max, policy_grad_l2_norm_square = get_gradient_stats(self.policy.parameters())
                self.optimizer_policy.step()

                # 2.6 optimize alpha
                alpha_loss = self.get_alpha_loss(self.log_sac_alpha, log_prob, mask, valid_num)
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
            'log_prob': self._mask_mean(log_prob, mask, valid_num).item(),

            # 'q1_mean': self._mask_mean(Qs[0], mask, valid_num).item(),
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