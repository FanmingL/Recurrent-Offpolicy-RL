from ..algorithm.sac import SAC
from ..algorithm.sac_mlp import SAC_MLP
from ..parameter.ParameterSAC import Parameter
from ..algorithm.sac_rnn_slice import SACRNNSlice
from ..algorithm.sac_mlp_redq import SAC_MLP_REDQ
from ..algorithm.sac_full_length_rnn_ensembleQ import SACFullLengthRNNEnsembleQ
from ..algorithm.sac_full_length_rnn_redq import SACFullLengthRNNREDQ
from ..algorithm.sac_mlp_redq_ensemble_q import SAC_MLP_REDQ_EnsembleQ
from ..algorithm.sac_full_length_rnn_redq_sep_optim import SACFullLengthRNNREDQ_SEP_OPTIM
from ..algorithm.sac_full_length_rnn_ensembleQ_sep_optim import SACFullLengthRNNENSEMBLEQ_SEP_OPTIM
from ..algorithm.td3_full_length_rnn_ensembleQ import TD3FullLengthRNNEnsembleQ
from ..algorithm.td3_full_length_rnn_redq import TD3FullLengthRNNREDQ
from ..algorithm.td3_full_length_rnn_redq_sep_optim import TD3FullLengthRNNREDQ_SEP_OPTIM


def alg_init(parameter: Parameter) -> SAC:
    if parameter.alg_name == 'sac_no_train':
        sac = SAC(parameter)
    elif parameter.alg_name == 'sac_mlp':
        sac = SAC_MLP(parameter)
    elif parameter.alg_name == 'sac_mlp_redq':
        sac = SAC_MLP_REDQ(parameter)
    elif parameter.alg_name == 'sac_rnn_slice':
        assert parameter.rnn_slice_length > 0
        sac = SACRNNSlice(parameter)
    elif parameter.alg_name == 'sac_rnn_full_horizon_ensembleQ':
        sac = SACFullLengthRNNEnsembleQ(parameter)
    elif parameter.alg_name == 'sac_rnn_full_horizon_redQ':
        sac = SACFullLengthRNNREDQ(parameter)
    elif parameter.alg_name == 'sac_rnn_full_horizon_redQ_sep_optim':  # STAR it!!!!
        sac = SACFullLengthRNNREDQ_SEP_OPTIM(parameter)
    elif parameter.alg_name == 'td3_rnn_full_horizon_redQ_sep_optim':  # STAR it!!!!
        parameter.base_algorithm = 'td3'
        sac = TD3FullLengthRNNREDQ_SEP_OPTIM(parameter)
    elif parameter.alg_name == 'sac_rnn_full_horizon_ensemble_q_sep_optim':
        sac = SACFullLengthRNNENSEMBLEQ_SEP_OPTIM(parameter)
    elif parameter.alg_name == 'sac_mlp_redq_ensemble_q':
        sac = SAC_MLP_REDQ_EnsembleQ(parameter)
    elif parameter.alg_name == 'td3_rnn_full_horizon_ensembleQ':
        parameter.base_algorithm = 'td3'
        sac = TD3FullLengthRNNEnsembleQ(parameter)
    elif parameter.alg_name == 'td3_rnn_full_horizon_redQ':
        parameter.base_algorithm = 'td3'
        sac = TD3FullLengthRNNREDQ(parameter)
    else:
        raise NotImplementedError(f'Algorithm {parameter.alg_name} has not been implemented!')
    return sac
