import numpy as np
from ..models.contextual_model import ContextualModel
import torch
import os
from ..models.RNNHidden import RNNHidden
from typing import List, Union, Tuple, Dict, Optional
from .utils import nearest_power_of_two, nearest_power_of_two_half

class ContextualSACValue(ContextualModel):
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False, separate_encoder=False, name='ContextualSACValue'):
        self.embedding_state_dim = state_dim
        self.reward_input = reward_input
        self.last_action_input = last_action_input
        self.last_state_input = last_state_input
        self.reward_dim = 1 if self.reward_input else 0
        self.last_act_dim = action_dim if self.last_action_input else 0
        self.last_obs_dim = state_dim if self.last_state_input else 0
        if embedding_size == 'auto':
            embedding_size = nearest_power_of_two_half(state_dim)
        if uni_model_input_mapping_dim == 'auto':
            uni_model_input_mapping_dim = nearest_power_of_two(state_dim + action_dim)
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

        uni_model_input_size = state_dim + action_dim
        self.state_input_encoder = torch.nn.Identity()
        self.action_input_encoder = torch.nn.Identity()
        if uni_model_input_mapping_dim > 0:
            if separate_encoder:
                self.state_input_encoder = torch.nn.Linear(state_dim, uni_model_input_mapping_dim)
                self.action_input_encoder = torch.nn.Linear(action_dim, uni_model_input_mapping_dim)
                uni_model_input_size = uni_model_input_mapping_dim * 2
                uni_model_input_mapping_dim = 0

        super(ContextualSACValue, self).__init__(embedding_input_size=cum_dim,
                                                 embedding_size=embedding_size,
                                                 embedding_hidden=embedding_hidden,
                                                 embedding_activations=embedding_activations,
                                                 embedding_layer_type=embedding_layer_type,
                                                 uni_model_input_size=uni_model_input_size,
                                                 uni_model_output_size=1,
                                                 uni_model_hidden=uni_model_hidden,
                                                 uni_model_activations=uni_model_activations,
                                                 uni_model_layer_type=uni_model_layer_type,
                                                 fix_rnn_length=fix_rnn_length,
                                                 uni_model_input_mapping_dim=uni_model_input_mapping_dim,
                                                 uni_model_input_mapping_activation=embedding_activations[-1],
                                                 name=name)
        self.uni_model_input_mapping_activation_func = self.embedding_network.activation_dict[embedding_activations[-1]]()
        if separate_encoder:
            self.contextual_register_rnn_base_module(self.state_encoder, 'state_encoder')
            if self.last_act_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_act_encoder, 'last_act_encoder')
            if self.last_obs_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_obs_encoder, 'last_obs_encoder')
            if self.reward_encoder is not None:
                self.contextual_register_rnn_base_module(self.reward_encoder, 'reward_encoder')
            if self.state_input_encoder is not None:
                self.contextual_register_rnn_base_module(self.state_input_encoder, 'state_input_encoder_q')
            if self.action_input_encoder is not None:
                self.contextual_register_rnn_base_module(self.action_input_encoder, 'action_input_encoder_q')

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

    def state_action(self, state, action):
        sa = torch.cat((
            self.state_input_encoder(state),
            self.action_input_encoder(action),
        ), dim=-1)
        if self.separate_encoder:
            return self.uni_model_input_mapping_activation_func(sa)
        else:
            return sa

    def forward(self, state: torch.Tensor, lst_state: torch.Tensor,
                lst_action: torch.Tensor, action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                reward: Optional[torch.Tensor], detach_embedding: bool=False) -> Tuple[
        torch.Tensor, torch.Tensor, RNNHidden, Optional[RNNHidden]
    ]:
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        value, rnn_memory, embedding_output, full_rnn_memory = \
            self.meta_forward(embedding_input, self.state_action(state, action), rnn_memory, detach_embedding)
        return value, embedding_output, rnn_memory, full_rnn_memory

    def forward_embedding(self, state: torch.Tensor, lst_state: torch.Tensor,
                lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                reward: Optional[torch.Tensor]):
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self.get_embedding(embedding_input, rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory
