from .contextual_sac_policy import ContextualSACPolicy
from typing import List, Union, Tuple, Dict, Optional
from ..models.RNNHidden import RNNHidden
import torch

class ContextualTD3Policy(ContextualSACPolicy):
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False, separate_encoder=False, sample_std=0.1):
        super().__init__(state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim, reward_input,
                         last_action_input, last_state_input, separate_encoder, output_logstd=False, name='ContextualTD3Policy')
        self.sample_std = sample_std


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
        action_mean = torch.tanh(model_output)
        action_sample = torch.tanh(model_output) + torch.randn_like(model_output) * self.sample_std
        action_sample = torch.clamp(action_sample, -1, 1)
        # not used in TD3
        log_prob = torch.zeros_like(action_sample)
        return action_mean, embedding_output, action_sample, log_prob, rnn_memory, full_rnn_memory
