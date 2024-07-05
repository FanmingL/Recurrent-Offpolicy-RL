import numpy as np
from ..models.contextual_model import ContextualModel
from ..models.mlp_base import MLPBase
import torch
from torch.functional import F
import os
from typing import List, Union, Tuple, Dict, Optional
from ..models.RNNHidden import RNNHidden
from .contextual_sac_policy import ContextualSACPolicy
from .utils import nearest_power_of_two, nearest_power_of_two_half
class ContextualSACDiscretePolicy(ContextualModel):
    MAX_LOG_STD = 2.0
    MIN_LOG_STD = -15.0
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type,
                 fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False, separate_encoder=False):
        if not uni_model_activations[-1] == 'linear':
            uni_model_activations = uni_model_activations[:-1] + ['linear']
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
        self.separate_encoder = separate_encoder
        if separate_encoder:
            basic_embedding_dim = 128
            self.state_encoder = torch.nn.Linear(state_dim, basic_embedding_dim)
            self.last_act_encoder = torch.nn.Linear(self.last_act_dim,
                                                    basic_embedding_dim) if self.last_act_dim else None
            self.reward_encoder = torch.nn.Linear(self.reward_dim, basic_embedding_dim) if self.reward_dim else None
            self.last_obs_encoder = torch.nn.Linear(self.last_obs_dim,
                                                    basic_embedding_dim) if self.last_obs_dim else None

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
        super(ContextualSACDiscretePolicy, self).__init__(embedding_input_size=cum_dim,
                                                  embedding_size=embedding_size,
                                                  embedding_hidden=embedding_hidden,
                                                  embedding_activations=embedding_activations,
                                                  embedding_layer_type=embedding_layer_type,
                                                  uni_model_input_size=state_dim,
                                                  uni_model_output_size=action_dim,
                                                  uni_model_hidden=uni_model_hidden,
                                                  uni_model_activations=uni_model_activations,
                                                  uni_model_layer_type=uni_model_layer_type,
                                                  fix_rnn_length=fix_rnn_length,
                                                  uni_model_input_mapping_dim=uni_model_input_mapping_dim,
                                                  uni_model_input_mapping_activation=embedding_activations[-1],
                                                  name='ContextualSACDiscretePolicy')
        if separate_encoder:
            self.contextual_register_rnn_base_module(self.state_encoder, 'state_encoder')
            if self.last_act_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_act_encoder, 'last_act_encoder')
            if self.last_obs_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_obs_encoder, 'last_obs_encoder')
            if self.reward_encoder is not None:
                self.contextual_register_rnn_base_module(self.reward_encoder, 'reward_encoder')

        self.state_dim = state_dim
        self.action_dim = action_dim

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
        action_mean, action_sample, log_probs, action_probs = self.process_model_out(model_output)
        return action_mean, embedding_output, action_sample, log_probs, rnn_memory, full_rnn_memory

    def process_model_out(self, model_output):
        probs = (model_output - torch.max(model_output, dim=-1, keepdim=True).values).exp()
        probs = probs / probs.sum(dim=-1, keepdim=True)
        probs = probs + 0.01
        probs = probs / probs.sum(dim=-1, keepdim=True)
        policy_dist = torch.distributions.Categorical(probs=probs)

        action_mean = policy_dist.mode.unsqueeze(-1)
        action_sample = policy_dist.sample().unsqueeze(-1)
        action_probs = policy_dist.probs
        # log_probs = F.log_softmax(model_output, dim=-1)
        log_probs = torch.log(policy_dist.probs)

        # policy_dist = torch.distributions.Categorical(logits=model_output)
        # action_mean = policy_dist.mode.unsqueeze(-1)
        # action_sample = policy_dist.sample().unsqueeze(-1)
        # action_probs = policy_dist.probs
        # log_probs = F.log_softmax(model_output, dim=-1)
        return action_mean, action_sample, log_probs, action_probs

    def select_with_action(self, action: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return data.gather(-1, action.long())

    def action2onehot(self, action: torch.Tensor):
        return F.one_hot(action.squeeze(-1).long(), num_classes=self.action_dim).float()

    def forward_embedding(self, state: torch.Tensor, lst_state: torch.Tensor,
                          lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                          reward: Optional[torch.Tensor]):
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self.get_embedding(embedding_input, rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory
