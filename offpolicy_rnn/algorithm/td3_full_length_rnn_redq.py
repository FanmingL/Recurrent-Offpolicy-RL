from typing import Dict

from .sac import SAC
from ..buffers.replay_memory import Transition
import torch
from ..utility.sample_utility import n2t
from .td3_full_length_rnn_ensembleQ import TD3FullLengthRNNEnsembleQ
import numpy as np

class TD3FullLengthRNNREDQ(TD3FullLengthRNNEnsembleQ):
    def __init__(self, parameter):
        super().__init__(parameter)

    def _target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                  target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            # pi(s_0) ~ pi (s_T)
            action_mean, _, _, next_log_prob, _, _ = self.policy.forward(next_state, state, action,
                                                                             policy_hidden, reward)
            next_act_sample = torch.clamp(
                action_mean + torch.clamp(torch.randn_like(action_mean) * self.parameter.target_action_noise_std,
                                          -self.parameter.target_action_noise_clip,
                                          self.parameter.target_action_noise_clip), -1, 1)
            # Q'(s_0) ~ Q' (s_T)
            next_target_Qs = [
                target_q_net.forward(next_state, state, action, next_act_sample, target_value_hiddens[q_ind],
                                     reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)
            rnd_idx = np.random.permutation(next_target_Qs[0].shape[0])[:self.parameter.redq_m]
            next_target_Qs = [item[rnd_idx, :] for item in next_target_Qs]
            # min_i next_Q_i

            min_next_target_Q = next_target_Qs[0].min(dim=0, keepdim=False).values
            target_Q = reward + (1 - done) * self.parameter.gamma * self.Q_guard.clamp(min_next_target_Q)
        return target_Q


    def _policy_loss(self, state, last_state, last_action, policy_hidden, reward_input, value_hiddens, alpha_detach, rnd_mask,
                     valid_num):
        action_mean, embedding_output, act_sample, log_prob, _, _full_rnn_memory_policy = self.policy.forward(state, last_state, last_action, policy_hidden,
                                                                                     reward_input)
        # self.logger(f'policy_hidden_input shape: {policy_hidden[0].shape}, policy_hidden_full shape: {_full_rnn_memory_policy[0].shape}')
        # self.logger(f'policy hidden error {self.batch_cnt}.{utd_idx}: {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().mean()} pm {(_full_rnn_memory_policy[0].transpose(0, 1) - policy_hidden_output[0]).abs().std()}')
        Qs_policy = [q_net.forward(state, last_state, last_action, action_mean, value_hiddens[q_ind], reward_input, detach_embedding=True)[0] for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = Qs_policy[0].mean(dim=0, keepdim=False)
        actor_loss = self._mask_mean(- min_Q, rnd_mask, valid_num)
        info = {'action_mean': action_mean, 'embedding_output': embedding_output, 'action_sample': act_sample}
        return actor_loss, log_prob, embedding_output, info

