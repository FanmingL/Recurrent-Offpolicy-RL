import copy

import yaml
import os
import math
from smart_logger.scripts.generate_tmuxp_base import generate_tmuxp_file, make_cmd_array
import argparse
import subprocess
from typing import Dict
MAX_SUBWINDOW =2
MAX_PARALLEL = 12

def get_gpu_count():
    sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode('utf-8').split('\n')
    out_list = [x for x in out_list if x]
    return len(out_list)

def get_cmd_array(total_machine=8, machine_idx=0):
    """
    :return: cmd array: list[list[]], the item in the i-th row, j-th column of the return value denotes the
                        cmd in the i-th window and j-th sub-window
    """
    session_name = 'OffpolicyRNN'
    # 0. 代码运行路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 1. GPU设置
    GPUS = [_ for _ in range(get_gpu_count())]
    # 2. 环境变量设置
    environment_dict = dict(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=current_path,
        OMP_NUM_THREADS=1,
        TF_ENABLE_ONEDNN_OPTS=0,
        TF_CPP_MIN_LOG_LEVEL=2,
    )
    directory = current_path
    # 3. 启动脚本
    start_up_header = "/home/ubuntu/.conda/envs/py38/bin/python main.py"
    # 4. 基础参数
    num_ensemble = 8
    basic_activation = 'elu'
    embedding_output_activation = f'linear'
    rnn_type_name = 'smamba_b1_c8_s64_ff'
    common_ndim = 256
    parameters_base = dict(
        alg_name='sac_rnn_full_horizon_redQ_sep_optim',
        total_iteration=1500,
        target_entropy_ratio=1.0,
        test_nprocess=5,
        test_nrollout=1,

        value_embedding_layer_type=[f'fc', f'{rnn_type_name}', f'fc'],
        value_embedding_activations=[f'{basic_activation}', f'linear', embedding_output_activation],
        value_embedding_hidden_size=[common_ndim, common_ndim],

        value_hidden_size=[common_ndim, common_ndim, common_ndim],
        value_activations=[basic_activation, basic_activation, basic_activation, 'linear'],
        value_layer_type=[f'efc-{num_ensemble}', f'efc-{num_ensemble}', f'efc-{num_ensemble}', f'efc-{num_ensemble}'],

        policy_embedding_layer_type=[ f'fc', f'{rnn_type_name}', 'fc'],
        policy_embedding_activations=[f'{basic_activation}', 'linear',
                                      embedding_output_activation],
        policy_embedding_hidden_size=[common_ndim, common_ndim],

        policy_hidden_size=[common_ndim, common_ndim, common_ndim],
        policy_activations=[basic_activation, basic_activation, basic_activation, 'linear'],
        policy_layer_type=[f'fc', 'fc', f'fc', f'fc'],
        sac_tau=0.995,  # influence value loss divergence
        value_net_num=1,
        cuda_inference=True,
        random_num=5000,
        max_buffer_traj_num=5000,
        policy_embedding_dim=128,
        value_embedding_dim=128,
        alpha_lr=1e-4,
        policy_uni_model_input_mapping_dim=128,
        value_uni_model_input_mapping_dim=128,
        policy_update_per=2,
        reward_input=False,
        sac_batch_size=1999,
        state_action_encoder=True,
        last_state_input=True,
    )
    # 5. 遍历设置
    exclusive_candidates = dict(
        seed=[1],
        env_name=[
            'AntBLT-V-v0', 'HalfCheetahBLT-V-v0', 'HopperBLT-V-v0', 'WalkerBLT-V-v0',
            'AntBLT-P-v0', 'HalfCheetahBLT-P-v0', 'HopperBLT-P-v0', 'WalkerBLT-P-v0',
            ]
    )
    # 6. 单独设置
    aligned_candidates = dict(
        policy_lr=[3e-4],
        value_lr=[1e-3],
        rnn_policy_lr=[1e-5],
        information=['Mamba_0705'],
    )

    def task_is_valid(_task):
        _task['rnn_value_lr'] = _task['rnn_policy_lr']
        if _task['env_name'] == 'AntBLT-V-v0':
            _task['sac_batch_size'] = 999
        return True

    # 从这里开始不用再修改了
    cmd_array, session_name = make_cmd_array(
        directory, session_name, start_up_header, parameters_base, environment_dict,
        aligned_candidates, exclusive_candidates, GPUS, MAX_PARALLEL, MAX_SUBWINDOW,
        machine_idx, total_machine, task_is_valid, split_all=True, sleep_before=0.0, sleep_after=0.0, rnd_seed=None, task_time_interval=60,
    )
    # 上面不用修改了

    # 7. 额外命令
    cmd_array.append(['htop', 'watch -n 1 nvidia-smi'])
    return cmd_array, session_name


def main():
    parser = argparse.ArgumentParser(description=f'generate parallel environment')
    parser.add_argument('--machine_idx', '-idx', type=int, default=-1,
                        help="Server port")
    parser.add_argument('--total_machine_num', '-tn', type=int, default=8,
                        help="Server port")
    args = parser.parse_args()
    cmd_array, session_name = get_cmd_array(args.total_machine_num, args.machine_idx)
    generate_tmuxp_file(session_name, cmd_array, use_json=True, layout='even-horizontal')


if __name__ == '__main__':
    main()
