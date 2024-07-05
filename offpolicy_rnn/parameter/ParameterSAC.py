from smart_logger.parameter.ParameterTemplate import ParameterTemplate
import smart_logger
import argparse

def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value

class Parameter(ParameterTemplate):
    def __init__(self, config_path=None, debug=False):
        super(Parameter, self).__init__(config_path, debug)

    def parse(self):
        parser = argparse.ArgumentParser(description=smart_logger.experiment_config.EXPERIMENT_TARGET)

        self.env_name = 'HalfCheetah-v2'
        parser.add_argument('--env_name', type=str, default=self.env_name,
                            help="name of the environment to run")

        self.alg_name = 'sac_mlp'
        parser.add_argument('--alg_name', type=str, default=self.alg_name,
                            help="name of the algorithm")

        self.seed = 1
        parser.add_argument('--seed', type=int, default=self.seed,
                            help="seed")

        self.policy_lr = 3e-4
        parser.add_argument('--policy_lr', type=float, default=self.policy_lr,
                            help="learning rate of the policy.")

        self.rnn_policy_lr = 1e-5
        parser.add_argument('--rnn_policy_lr', type=float, default=self.rnn_policy_lr,
                            help="learning rate of the rnn-part of the policy.")

        self.policy_l2_norm = 0.0
        parser.add_argument('--policy_l2_norm', type=float, default=self.policy_l2_norm,
                            help="ratio of L2 norm for policy learning.")

        self.policy_update_per = 1
        parser.add_argument('--policy_update_per', type=int, default=self.policy_update_per,
                            help="policy update per update iteration")

        self.policy_max_gradnorm = None
        parser.add_argument('--policy_max_gradnorm', type=float, default=self.policy_max_gradnorm,
                            help="The maximum norm of gradient for policy.")

        self.policy_embedding_max_gradnorm = None
        parser.add_argument('--policy_embedding_max_gradnorm', type=float, default=self.policy_embedding_max_gradnorm,
                            help="The maximum norm of gradient for policy embedding network.")

        self.alpha_lr = 1e-2
        parser.add_argument('--alpha_lr', type=float, default=self.alpha_lr,
                            help="learning rate of the entropy coefficient.")

        self.value_lr = 1e-3
        parser.add_argument('--value_lr', type=float, default=self.value_lr,
                            help="learning rate of the value.")

        self.rnn_value_lr = 1e-4
        parser.add_argument('--rnn_value_lr', type=float, default=self.rnn_value_lr,
                            help="learning rate of the rnn-part of the value.")

        self.value_max_gradnorm = None
        parser.add_argument('--value_max_gradnorm', type=float, default=self.value_max_gradnorm,
                            help="The maximum norm of gradient for value.")

        self.value_embedding_max_gradnorm = None
        parser.add_argument('--value_embedding_max_gradnorm', type=float, default=self.value_embedding_max_gradnorm,
                            help="The maximum norm of gradient for value embedding network.")

        self.value_l2_norm = 0.0
        parser.add_argument('--value_l2_norm', type=float, default=self.value_l2_norm,
                            help="ratio of L2 norm for value learning.")

        self.cuda_inference = False
        parser.add_argument('--cuda_inference', action='store_true',
                            help='whether use GPU to sample data')

        self.backing_log = False
        parser.add_argument('--backing_log', action='store_true',
                            help='if true, backing up log to remote machine')

        self.reward_input = False
        parser.add_argument('--reward_input', action='store_true',
                            help='whether use reward as embedding network input')

        self.last_state_input = False
        parser.add_argument('--last_state_input', action='store_true',
                            help='whether use last_state as embedding network input')

        self.randomize_mask = False
        parser.add_argument('--randomize_mask', action='store_true',
                            help='random choosing mask entity to zero, reserve a small amount.')

        self.random_trunc_traj = False
        parser.add_argument('--random_trunc_traj', action='store_true',
                            help='random trunking the trajectory (instead of use full trajectory to train). ')

        self.valid_number_post_randomized = 256
        parser.add_argument('--valid_number_post_randomized', type=int, default=self.valid_number_post_randomized,
                            help="how many number of data will reserve after entity masking")

        self.policy_uni_model_input_mapping_dim = 0
        parser.add_argument('--policy_uni_model_input_mapping_dim', type=str_or_int, default=self.policy_uni_model_input_mapping_dim,
                            help="mapping state to certain-dimension vector")

        self.value_uni_model_input_mapping_dim = 0
        parser.add_argument('--value_uni_model_input_mapping_dim', type=str_or_int,
                            default=self.value_uni_model_input_mapping_dim,
                            help="mapping state to certain-dimension vector")

        self.randomize_first_hidden = False
        parser.add_argument('--randomize_first_hidden', action='store_true',
                            help='whether use a random hidden state as the first hidden state or use a zero hidden')

        self.randomize_training_initial_hidden = False
        parser.add_argument('--randomize_training_initial_hidden', action='store_true',
                            help='In full rnn with slice and hidden pre-computing, whether disturbing the initial hidden state.')

        self.no_alpha_auto_tune = False
        parser.add_argument('--no_alpha_auto_tune', action='store_true',
                            help='Do not tune alpha automatically.')

        self.no_last_action_input = False
        parser.add_argument('--no_last_action_input', action='store_true',
                            help='do not use the last action as embedding network input (If true, the last action will not be input)')

        self.state_action_encoder = False
        parser.add_argument('--state_action_encoder', action='store_true',
                            help='Use encoder for state, action, and reward before inputting into embedding/value network.')

        # ---> value model
        self.value_hidden_size = [256, 128]
        parser.add_argument('--value_hidden_size', nargs='+', type=int, default=self.value_hidden_size,
                            help="architecture of the hidden layers of value")

        self.value_activations = ['relu', 'relu', 'linear']
        parser.add_argument('--value_activations', nargs='+', type=str,
                            default=self.value_activations,
                            help="activation of each layer of value")

        self.value_layer_type = ['fc', 'fc', 'fc']
        parser.add_argument('--value_layer_type', nargs='+', type=str,
                            default=self.value_layer_type,
                            help="net type of value")

        self.value_net_num = 2
        parser.add_argument('--value_net_num', type=int, default=self.value_net_num,
                            help="number of value network")

        self.utd = 1
        parser.add_argument('--utd', type=int, default=self.utd,
                            help="update to data (UTD) ratio")

        self.policy_utd = 1
        parser.add_argument('--policy_utd', type=int, default=self.policy_utd,
                            help="update to data (UTD) ratio of the policy")

        self.redq_m = 2
        parser.add_argument('--redq_m', type=int, default=self.redq_m,
                            help="use redq_m value networks to compute target Q")

        # ---> value embedding shape
        self.value_embedding_hidden_size = [256, 128, 64]
        parser.add_argument('--value_embedding_hidden_size', nargs='+', type=int, default=self.value_embedding_hidden_size,
                            help="architecture of the hidden layers of value")

        self.value_embedding_activations = ['relu', 'linear', 'relu', 'tanh']
        parser.add_argument('--value_embedding_activations', nargs='+', type=str,
                            default=self.value_embedding_activations,
                            help="activation of each layer of value")

        self.value_embedding_layer_type = ['fc', 'gru', 'fc', 'fc']
        parser.add_argument('--value_embedding_layer_type', nargs='+', type=str,
                            default=self.value_embedding_layer_type,
                            help="net type of value")

        self.value_embedding_dim = 16
        parser.add_argument('--value_embedding_dim', type=str_or_int, default=self.value_embedding_dim,
                            help="value embedding dim.")
        # ---> policy model
        self.policy_hidden_size = [256, 128]
        parser.add_argument('--policy_hidden_size', nargs='+', type=int, default=self.policy_hidden_size,
                            help="architecture of the hidden layers of Universe Policy")

        self.policy_activations = ['relu', 'relu', 'linear']
        parser.add_argument('--policy_activations', nargs='+', type=str,
                            default=self.policy_activations,
                            help="activation of each layer of Universe Policy")

        self.policy_layer_type = ['fc', 'fc', 'fc']
        parser.add_argument('--policy_layer_type', nargs='+', type=str,
                            default=self.policy_layer_type,
                            help="net type of Universe Policy")

        # ---> policy embedding shape
        self.policy_embedding_hidden_size = [256, 128, 64]
        parser.add_argument('--policy_embedding_hidden_size', nargs='+', type=int, default=self.policy_embedding_hidden_size,
                            help="architecture of the hidden layers of Universe Policy")

        self.policy_embedding_activations = ['relu', 'linear', 'relu', 'tanh']
        parser.add_argument('--policy_embedding_activations', nargs='+', type=str,
                            default=self.policy_embedding_activations,
                            help="activation of each layer of Universe Policy")

        self.policy_embedding_layer_type = ['fc', 'gru', 'fc', 'fc']
        parser.add_argument('--policy_embedding_layer_type', nargs='+', type=str,
                            default=self.policy_embedding_layer_type,
                            help="net type of Universe Policy")

        self.policy_embedding_dim = 16
        parser.add_argument('--policy_embedding_dim', type=str_or_int, default=self.policy_embedding_dim,
                            help="policy embedding dim.")

        self.test_nprocess = 5
        parser.add_argument('--test_nprocess', type=int, default=self.test_nprocess,
                            help="number of process for policy test")

        self.test_nrollout = 2
        parser.add_argument('--test_nrollout', type=int, default=self.test_nrollout,
                            help="number of rollouts for each test process.")

        self.total_iteration = 5000
        parser.add_argument('--total_iteration', type=int, default=self.total_iteration,
                            help="total SAC iteration.")

        self.gamma = 0.99
        parser.add_argument('--gamma', type=float, default=self.gamma,
                            help="discounted factor")

        self.information = "None"
        parser.add_argument('--information', type=str, default=self.information,
                            help="information")

        self.rnn_sample_max_batch_size = 300000
        parser.add_argument('--rnn_sample_max_batch_size', type=int, default=self.rnn_sample_max_batch_size,
                            help='max point num sampled from replay buffer per time')

        self.max_buffer_traj_num = 10000
        parser.add_argument('--max_buffer_traj_num', type=int, default=self.max_buffer_traj_num,
                            help='maximum trajectory num.')

        self.max_buffer_transition_num = int(1e6)
        parser.add_argument('--max_buffer_transition_num', type=int, default=self.max_buffer_transition_num,
                            help='maximum transition num for replay buffer.')

        self.sac_tau = 0.995
        parser.add_argument('--sac_tau', type=float, default=self.sac_tau,
                            help='ratio of coping value net to target value net')

        self.sac_alpha = 0.2
        parser.add_argument('--sac_alpha', type=float, default=self.sac_alpha,
                            help='sac temperature coefficient')

        self.target_entropy_ratio = 1.5
        parser.add_argument('--target_entropy_ratio', type=float, default=self.target_entropy_ratio,
                            help="target entropy")

        self.rnn_fix_length = 0
        parser.add_argument('--rnn_fix_length', type=int, default=self.rnn_fix_length,
                            help="fix the rnn memory length to rnn_fix_length")

        self.rnn_slice_length = 0
        parser.add_argument('--rnn_slice_length', type=int, default=self.rnn_slice_length,
                            help="slice length of RNN training")

        self.step_per_iteration = 1000
        parser.add_argument('--step_per_iteration', type=int, default=self.step_per_iteration,
                            help='timestep per training iteration')

        self.random_num = 20000
        parser.add_argument('--random_num', type=int, default=self.random_num,
                            help='sample random_num fully random samples,')

        self.start_train_num = 1000
        parser.add_argument('--start_train_num', type=int, default=self.start_train_num,
                            help='when to start training')

        self.update_interval = 1
        parser.add_argument('--update_interval', type=int, default=self.update_interval,
                            help="model update interval")

        self.sac_batch_size = 1024
        parser.add_argument('--sac_batch_size', type=int, default=self.sac_batch_size,
                            help='sac_batch_size valid transitions will be sampled from the replay buffer.')

        # ----------- Algorithm Choosing
        self.base_algorithm = "sac"
        parser.add_argument('--base_algorithm', type=str, default=self.base_algorithm,
                            help="backbone RL algorithm")

        # ------------ TD3 specific hyper-parameters
        self.sample_std = 0.1
        parser.add_argument('--sample_std', type=float, default=self.sample_std,
                            help="sample std of TD3")

        self.target_action_noise_std = 0.04
        parser.add_argument('--target_action_noise_std', type=float, default=self.target_action_noise_std,
                            help="noise of the target action in training")

        self.target_action_noise_clip = 0.12
        parser.add_argument('--target_action_noise_clip', type=float, default=self.target_action_noise_clip,
                            help="noise-clip of the target action in training")

        return parser.parse_args()


if __name__ == '__main__':
    def main():
        parameter = Parameter()
        print(parameter)
    main()