from .contextual_sac_policy import ContextualSACPolicy
from .contextual_td3_policy import  ContextualTD3Policy
from .contextual_sac_value import ContextualSACValue
from .contextual_td3_value import ContextualTD3Value
from .contextual_sac_discrete_policy import ContextualSACDiscretePolicy
from .contextual_sac_discrete_value import ContextualSACDiscreteValue
import torch
from typing import Optional, Union

def make_policy_model(policy_args, base_alg_name, discrete) -> Union[ContextualTD3Policy, ContextualSACPolicy, ContextualSACDiscretePolicy]:
    if base_alg_name == 'sac':
        if discrete:
            policy = ContextualSACDiscretePolicy(**policy_args)
        else:
            policy = ContextualSACPolicy(**policy_args)
    elif base_alg_name == 'td3':
        policy = ContextualTD3Policy(**policy_args)
    return policy

def make_value_model(value_args, base_alg_name, discrete) -> Union[ContextualTD3Value, ContextualSACValue, ContextualSACDiscreteValue]:
    if base_alg_name == 'sac':
        if discrete:
            value = ContextualSACDiscreteValue(**value_args)
        else:
            value = ContextualSACValue(**value_args)
    elif base_alg_name == 'td3':
        value = ContextualTD3Value(**value_args)
    return value
