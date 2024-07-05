import numpy as np
from ..models.contextual_model import ContextualModel
from ..models.mlp_base import MLPBase
import torch
from torch.functional import F
import os
from typing import List, Union, Tuple, Dict, Optional
from ..models.RNNHidden import RNNHidden
from .utils import nearest_power_of_two, nearest_power_of_two_half

class ContextualSACPolicyDoubleHead(ContextualModel):
    MAX_LOG_STD = 2.0
    MIN_LOG_STD = -20.0
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0, reward_input=False,
                 last_action_input=True, last_state_input=False, output_logstd=True):
        assert output_logstd is True
        if not uni_model_activations[-1] == 'linear':
            uni_model_activations = uni_model_activations[:-1] + ['linear']
        if not uni_model_layer_type[-1] == 'fc':
            raise NotImplementedError(f'It is not supported to construct {uni_model_layer_type[-1]} logstd and mean head! You can set the uni_model layer type to {uni_model_layer_type[:-1] + ["fc"]}')
        self.reward_input = reward_input
        self.last_action_input = last_action_input
        self.last_state_input = last_state_input
        self.reward_dim = 1 if self.reward_input else 0
        self.last_act_dim = action_dim if self.last_action_input else 0
        self.last_obs_dim = state_dim if self.last_state_input else 0
        if embedding_size == 'auto':
            embedding_size = nearest_power_of_two_half(state_dim)
        if uni_model_input_mapping_dim == 'auto':
            uni_model_input_mapping_dim = nearest_power_of_two(state_dim)
        # there is one mean head and one logstd head
        mean_head_hidden = []
        mean_head_activations = ['linear']
        mean_head_input = uni_model_hidden[-1]
        mean_head_output = action_dim

        logstd_head_hidden = []
        logstd_head_activations = ['linear']
        logstd_head_input = uni_model_hidden[-1]
        logstd_head_output = action_dim

        uni_model_hidden = uni_model_hidden[:-1]
        uni_model_activations = uni_model_activations[:-1]
        uni_model_layer_type = uni_model_layer_type[:-1]
        uni_model_input = state_dim
        uni_model_output = mean_head_input
        self.separate_encoder = separate_encoder = True
        if separate_encoder:
            basic_embedding_dim = 128
            self.state_encoder = torch.nn.Linear(state_dim, basic_embedding_dim)
            self.last_act_encoder = torch.nn.Linear(self.last_act_dim, basic_embedding_dim) if self.last_act_dim else None
            self.reward_encoder = torch.nn.Linear(self.reward_dim, basic_embedding_dim) if self.reward_dim else None
            self.last_obs_encoder = torch.nn.Linear(self.last_obs_dim, basic_embedding_dim) if self.last_obs_dim else None

            cum_dim = basic_embedding_dim
            if self.last_act_encoder is not None:
                cum_dim += basic_embedding_dim
            if self.last_obs_encoder is not None:
                cum_dim += basic_embedding_dim
            if self.reward_encoder is not None:
                cum_dim += basic_embedding_dim
        else:
            cum_dim = state_dim + self.reward_dim + self.last_act_dim + self.last_obs_dim
            self.state_encoder = torch.nn.Identity()
            self.last_act_encoder = torch.nn.Identity()
            self.reward_encoder = torch.nn.Identity()
            self.last_obs_encoder = torch.nn.Identity()

        super(ContextualSACPolicyDoubleHead, self).__init__(embedding_input_size=cum_dim,
                                                  embedding_size=embedding_size,
                                                  embedding_hidden=embedding_hidden,
                                                  embedding_activations=embedding_activations,
                                                  embedding_layer_type=embedding_layer_type,
                                                  uni_model_input_size=uni_model_input,
                                                  uni_model_output_size=uni_model_output,
                                                  uni_model_hidden=uni_model_hidden,
                                                  uni_model_activations=uni_model_activations,
                                                  uni_model_layer_type=uni_model_layer_type,
                                                  fix_rnn_length=fix_rnn_length,
                                                  uni_model_input_mapping_dim=uni_model_input_mapping_dim,
                                                  uni_model_input_mapping_activation=embedding_activations[-1],
                                                  name='ContextualSACPolicy')
        if separate_encoder:
            self.contextual_register_rnn_base_module(self.state_encoder, 'state_encoder')
            if self.last_act_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_act_encoder, 'last_act_encoder')
            if self.last_obs_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_obs_encoder, 'last_obs_encoder')
            if self.reward_encoder is not None:
                self.contextual_register_rnn_base_module(self.reward_encoder, 'reward_encoder')

        self.mean_head = MLPBase(mean_head_input, mean_head_output, mean_head_hidden, mean_head_activations)
        self.logstd_head = MLPBase(logstd_head_input, logstd_head_output, logstd_head_hidden, logstd_head_activations)
        self.contextual_register_rnn_base_module(self.mean_head, 'mean_head')
        self.contextual_register_rnn_base_module(self.logstd_head, 'logstd_head')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.soft_plus = torch.nn.Softplus()

    def _soft_clamp(self, x: torch.Tensor, _min, _max) -> torch.Tensor:
        if _max is not None:
            x = _max - self.soft_plus(_max - x)
        if _min is not None:
            x = _min + self.soft_plus(x - _min)
        return x

    def get_embedding_input(self, state, lst_state, lst_action, reward) -> torch.Tensor:
        embedding_inputs = [self.state_encoder(state)]
        if self.last_state_input:
            embedding_inputs.append(self.last_obs_encoder(lst_state))
        if self.last_action_input:
            embedding_inputs.append(self.last_act_encoder(lst_action))
        if self.reward_input:
            embedding_inputs.append(self.reward_encoder(reward))
        embedding_input = torch.cat(embedding_inputs, dim=-1)
        return embedding_input

    def forward(self, state: torch.Tensor, lst_state: torch.Tensor, lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden], reward: Optional[torch.Tensor]=None, detach_embedding: bool=False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RNNHidden, Optional[RNNHidden]
    ]:
        """
        :param state: (seq_len, dim) or (batch_size, seq_len, dim)
        :param lst_state: (seq_len, dim) or (batch_size, seq_len, dim)
        :param lst_action: (seq_len, dim) or (batch_size, seq_len, dim)
        :param reward: (seq_len, 1) or (batch_size, seq_len, dim)
        :param rnn_memory:
        :return:
        """
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        model_output, rnn_memory, embedding_output, full_rnn_memory = self.meta_forward(embedding_input, state, rnn_memory, detach_embedding)
        logstd = self.logstd_head.meta_forward(model_output)
        logit = self.mean_head.meta_forward(model_output)
        action_mean, action_sample, log_prob = self.process_model_out(logit, logstd)
        return action_mean, embedding_output, action_sample, log_prob, rnn_memory, full_rnn_memory

    def process_model_out(self, logit, logstd):
        action_mean = logit
        # logstd = self._soft_clamp(logstd, self.MIN_LOG_STD, self.MAX_LOG_STD)
        logstd = torch.clamp(logstd, self.MIN_LOG_STD, self.MAX_LOG_STD)
        action_std = logstd.exp()
        noise_pre = torch.randn_like(action_mean).detach()
        noise = noise_pre * action_std

        action_sample = action_mean + noise
        log_prob = (- 0.5 * noise_pre.pow(2) - (logstd + 0.5 * np.log(2 * np.pi))).sum(-1, keepdim=True)
        log_prob = log_prob - (2 * (- action_sample - self.soft_plus(-2 * action_sample) + np.log(2))).sum(-1,
                                                                                                           keepdim=True)
        action_mean = torch.tanh(action_mean)
        action_sample = torch.tanh(action_sample)
        return action_mean, action_sample, log_prob

    def forward_embedding(self, state: torch.Tensor, lst_state: torch.Tensor,
                          lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                          reward: Optional[torch.Tensor]):
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self.get_embedding(embedding_input, rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory
