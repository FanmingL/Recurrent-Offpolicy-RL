import numpy as np
from ..models.contextual_model import ContextualModel
import torch
import os
from ..models.RNNHidden import RNNHidden
from typing import List, Union, Tuple, Dict, Optional
from .utils import nearest_power_of_two, nearest_power_of_two_half
from .contextual_sac_value import ContextualSACValue
class ContextualTD3Value(ContextualSACValue):
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False, separate_encoder=False):
        super(ContextualTD3Value, self).__init__(state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim, reward_input,
                                                 last_action_input, last_state_input, separate_encoder, name='ContextualTD3Value')
