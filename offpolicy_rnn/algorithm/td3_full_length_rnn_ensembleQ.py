import math
from typing import Dict

from .sac import SAC
from ..buffers.replay_memory import Transition
from ..buffers.transition_buffer.nested_replay_memory import NestedMemoryArray as NestedTransitionMemoryArray
from ..utility.q_value_guard import QValueGuard
import smart_logger
import torch
from ..utility.sample_utility import n2t
from ..policy_value_models.contextual_sac_policy import ContextualSACPolicy
from ..policy_value_models.contextual_sac_discrete_policy import ContextualSACDiscretePolicy
import copy

from .sac_full_length_rnn_ensembleQ import SACFullLengthRNNEnsembleQ


class TD3FullLengthRNNEnsembleQ(SACFullLengthRNNEnsembleQ):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.parameter.no_alpha_auto_tune = True

    def _target_Q(self, state, last_state, action, last_action, next_state, done, reward, reward_input, policy_hidden,
                  target_value_hiddens, alpha_detach):
        with torch.no_grad():
            # 2.1 compute target value
            # full_state = torch.cat((state[..., :1, :], next_state), dim=-2)
            # full_last_action = torch.cat((last_action[..., :1, :], action), dim=-2)
            # full_reward = torch.cat((reward_input[..., :1, :], reward), dim=-2)
            # pi(s_0) ~ pi (s_T)
            action_mean, _, _, next_log_prob, _, _ = self.target_policy.forward(next_state, state, action,
                                                                                    policy_hidden, reward)
            next_act_sample = torch.clamp(action_mean + torch.clamp(torch.randn_like(action_mean) * self.parameter.target_action_noise_std,
                                                                    -self.parameter.target_action_noise_clip,
                                                                    self.parameter.target_action_noise_clip), -1, 1)
            # Q'(s_0) ~ Q' (s_T)
            next_target_Qs = [
                target_q_net.forward(next_state, state, action, next_act_sample, target_value_hiddens[q_ind],
                                     reward)[0] for q_ind, target_q_net in enumerate(self.target_values)]
            # Q'(s_1) ~ Q'(s_T)

            # min_i next_Q_i
            min_next_target_Q = next_target_Qs[0].min(dim=0, keepdim=False).values
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
        Qs_policy = [q_net.forward(state, last_state, last_action, action_mean, value_hiddens[q_ind], reward_input, detach_embedding=True)[0] for
                     q_ind, q_net in enumerate(self.values)]

        min_Q = Qs_policy[0].min(dim=0, keepdim=False).values
        actor_loss = self._mask_mean(- min_Q, rnd_mask, valid_num)
        info = {'action_mean': action_mean, 'embedding_output': embedding_output, 'action_sample': act_sample}
        return actor_loss, log_prob, embedding_output, info

    def _optim_policy(self, state, last_state, last_action, policy_hidden, reward_input,
                                                            value_hiddens, alpha_detach, mask,
                                                            valid_num):
        actor_loss, log_prob, actor_embedding, _ = self.get_policy_loss(state, last_state, last_action, policy_hidden, reward_input,
                                                    value_hiddens, alpha_detach, mask,
                                                    valid_num)
        losses = {}

        # 2.5 optimize actor
        self.optimizer_policy.zero_grad()
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
        self.optimizer_policy.step()

        return pi_grad_norm, actor_loss, log_prob, losses

    def _target_policy_update(self, tau):
        """
            if tau = 0, self <--- src_net
            if tau = 1, self <--- self
        """
        self.target_policy.copy_weight_from(self.policy, tau)


    def _optim_value(self, state, last_state, last_action, action, mask, reward_input, value_hiddens, valid_num, target_Q):
        Q_loss, Q_embeddings, _ = self.get_Q_loss(state, last_state, last_action, action, mask, reward_input, value_hiddens,
                                 valid_num,
                                 target_Q)
        losses = {}

        # 2.3 optimize the value model
        self.optimizer_value.zero_grad()
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
        self.optimizer_value.step()
        return Q_loss, q_grad_norm, losses
