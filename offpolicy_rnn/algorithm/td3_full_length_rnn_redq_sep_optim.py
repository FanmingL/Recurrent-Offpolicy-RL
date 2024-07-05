from typing import Dict, List

from .sac import SAC
from ..buffers.replay_memory import Transition
import torch
from ..utility.sample_utility import n2t
from .td3_full_length_rnn_redq import TD3FullLengthRNNREDQ
import numpy as np
from ..utility.ValueScheduler import CosineScheduler, LinearScheduler
from ..models.rnn_base import RNNBase


# def prepare_param_list(model, rnn_lr, policy_l2_norm):
#     param_list = []
#     for k, v in model.contextual_modules.items():
#         if isinstance(v, RNNBase):
#             for ind, (layer, layer_type) in enumerate(zip(v.layer_list, v.layer_type)):
#                 if v.check_is_rnn(layer_type):
#                     print(f'{layer_type} {type(layer)} in {k} is RNN!! Add it to RNN categories')
#                     param_list.append(
#                         {"params": list(layer.parameters(True)), 'lr': rnn_lr,
#                          'weight_decay': policy_l2_norm}
#                     )
#                 else:
#                     param_list.append(
#                         {"params": list(layer.parameters(True))}
#                     )
#             param_list.append({
#                 "params": list(v.activation_list.parameters(True))
#             })
#         else:
#             param_list.append(
#                 {"params": list(v.parameters(True))}
#             )
#     return param_list

def _insert_rnn_parameter(module, rnn_lr, l2_norm, name):
    return {"params": list(module.parameters(True)),
         'lr': rnn_lr,
         'weight_decay': l2_norm,
         "name": f"rnn-{name}"}


def _insert_mlp_parameter(module, rnn_lr, l2_norm, name):
    return {"params": list(module.parameters(True)),
            # 'lr': rnn_lr,
            # 'weight_decay': l2_norm,
            "name": f"mlp-{name}"}
def prepare_param_list(model, rnn_lr, l2_norm):
    param_list = []
    for k, v in model.contextual_modules.items():
        def rnn_parameter(mod):
            return _insert_rnn_parameter(mod, rnn_lr, l2_norm, type(v).__name__)
        def mlp_parameter(mod):
            return _insert_mlp_parameter(mod, rnn_lr, l2_norm, type(v).__name__)

        if k.endswith('encoder'):
            param_list.append(mlp_parameter(v))
        elif k == 'embedding_model':
            param_list.append(rnn_parameter(v.layer_list[0]))
            for i in range(1, len(model.embedding_network.layer_list) - 1):
                param_list.append(rnn_parameter(v.layer_list[i]))
            param_list.append(rnn_parameter(v.layer_list[-1]))
            param_list.append(rnn_parameter(v.activation_list))
        else:
            param_list.append(mlp_parameter(v))
    return param_list

class TD3FullLengthRNNREDQ_SEP_OPTIM(TD3FullLengthRNNREDQ):
    def __init__(self, parameter):
        super().__init__(parameter)

        policy_param_list = prepare_param_list(self.policy, self.parameter.rnn_policy_lr, self.parameter.policy_l2_norm)
        self.optimizer_policy = self.optim_class(policy_param_list,
                                                 lr=self.parameter.policy_lr,
                                                 weight_decay=self.parameter.policy_l2_norm)
        value_param_list = [item for value in self.values for item in prepare_param_list(value, self.parameter.rnn_value_lr, self.parameter.value_l2_norm)]

        self.optimizer_value = self.optim_class(value_param_list, lr=self.parameter.value_lr,
                                                weight_decay=self.parameter.value_l2_norm)
        # tune_config
        self.optimizer_policy_tune = self.optim_class(policy_param_list,
                                                      lr=self.parameter.policy_lr,
                                                      weight_decay=self.parameter.policy_l2_norm)

        self.optimizer_value_tune = self.optim_class(value_param_list, lr=self.parameter.value_lr,
                                                     weight_decay=self.parameter.value_l2_norm)

        self.init_lr_scheduler()


    def init_lr_scheduler(self):
        pass


