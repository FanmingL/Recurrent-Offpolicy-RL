from .contextual_sac_policy_single_head import ContextualSACPolicySingleHead
from .contextual_sac_policy_double_head import ContextualSACPolicyDoubleHead

class ContextualSACPolicy(ContextualSACPolicySingleHead):
# class ContextualSACPolicy(ContextualSACPolicyDoubleHead):
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False, separate_encoder=False,
                 output_logstd=True, name='ContextualSACPolicy'):
        super().__init__(state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim, reward_input,
                         last_action_input, last_state_input, separate_encoder, output_logstd,
                         name=name)
