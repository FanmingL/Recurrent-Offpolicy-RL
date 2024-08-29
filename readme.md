# Off-Policy RL with RNN and Transformers

This repository is the official implementation of the paper "[Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate.](https://arxiv.org/abs/2405.15384)" It includes implementations of SAC and TD3 based on RNN and Transformer architectures.

🌟We presented some reproduced results and logs in the [result page](results.md).

## Features
The algorithms implemented in this repository have the following features:
1. We train the recurrent policy and values using full-length trajectories instead of sequence fragments;
2. To enhance training stability with full-length trajectories, we utilize the [Context-Encoder-Specific Learning Rate](https://arxiv.org/abs/2405.15384) (RESeL) technique;
3. [TODO] We provide a set of training hyperparameters that can achieve state-of-the-art performance in different environments of POMDP and MDP.

## Supported Layer Types
This repository supports the following neural network architectures. We have tested with training on `gru`, `mamba`, `smamba`, and `cgpt` layer types. The training speed from fastest to slowest is: `smamba`, `mamba`, `cgpt`, `gru`.

| Layer       | Layer ID | Parameters                                                | Notes                                                                                                                                                                                                                                                                |
|-------------|----------| --------------------------------------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GRU         | gru      |                                                           | PyTorch's built-in GRU, non-parallel, but still achieves good policy performance                                                                                                                                                                                     |
| Mamba       | smamba   | state_dim, conv1d_size, block_num, norm_type              | Official Mamba implementation, accelerated with selective_scan parallelization. Example: `smamba_s32_c16_b2_nln` means `state_dim=32`, `conv1d_size=16`, `block_num=2`, `norm_type=layer_norm`                                                                       |
| Mamba       | mamba    | state_dim, conv1d_size                                    | Mamba implemented with Triton, serial computation, significantly faster than the PyTorch implementation. Example: `mamba_s32_c16` means `state_dim=32`, `conv1d_size=16`                                                                                             |
| GILR        | gilr     |                                                           | Linear RNN structure implemented with Triton                                                                                                                                                                                                                         |
| LRU         | lru      |                                                           | Linear Recurrent Unit (LRU) implemented with Triton                                                                                                                                                                                                                                            |
| Transformer | cgpt     | head_num, block_num, dropout_prob, max_length, norm_type  | Custom GPT structure, accelerated with flash_attention for training and inference, using bf16 data type in multi-head-attention. Example: `cgpt_h8_l6_p0.1_ml1024_rms` means `head_num=8`, `block_num=6`, `dropout_prob=0.1`, `max_length=1024`, `norm_type=RMSNorm` |
| Transformer | gpt      | head_num, block_num, dropout_prob, max_length             | GPT structure from the flash_attn library, accelerated with flash_attention, using bf16 data type. Example: `gpt_h8_l6_p0.1_ml1024` means `head_num=8`, `block_num=6`, `dropout_prob=0.1`, `max_length=1024`                                                         |

## Dependencies
### Hardware
In the aforementioned network structures, GRU can be trained directly on a CPU machine. Mamba, GIRL, and LRU are implemented based on Triton, requiring training on GPU machines, while cgpt and gpt utilize flash_attention for acceleration, requiring the use of Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100).

### Environment
Since we have modified the CUDA source code of Mamba, it needs to be recompiled. This library also depends on some earlier RL training environments, so we strongly recommend using the Docker image we have prepared to run our algorithms. To pull the Docker image, use the following command:
```bash
docker pull core.116.172.93.164.nip.io:30670/public/luofanming:20240607150538
```
Alternatively, download it from this [link](https://box.nju.edu.cn/f/11384fd1c05641158dcd/)
```bash
wget -O 20240607_flash_attn_image.tar.gz https://box.nju.edu.cn/f/11384fd1c05641158dcd/?dl=1
docker load -i 20240607_flash_attn_image.tar.gz
```
To start the Docker container:
```bash
docker run --rm -it -v $PWD:/home/ubuntu/workspace --gpus all core.116.172.93.164.nip.io:30670/public/luofanming:20240607150538 /bin/bash
```

## Starting Training
We use Python files starting with `gen_tmuxp` to record the training hyperparameters. For instance, to start an experiment with the `smamba` structure, you can run:
```bash
cd /path/to/Recurrent-Offpolicy-RL
pip install -e .
python gen_tmuxp_mamba_mujoco.py
tmuxp load run_all.json
```
We present the reproducing results of each script at [here](results.md).


## Visualizing Results
We use [SmartLogger](https://github.com/FanmingL/SmartLogger) for log management. You can find the training logs in the directory named `logfile`. The most straightforward way to view the training process is to use TensorBoard:
```bash
tensorboard --logdir=./logfile
```
You can also use the rendering interface in SmartLogger to view experimental data:
```bash
python -m smart_logger.htmlpage -p 4008 -d /path/to/logfile -wks ~/Desktop/smartlogger_wks  -t local_plotting -u user -pw resel -cp 600
```
Visit [http://localhost:4008](http://localhost:4008) to view the training data, with the username `user` and password `resel`.

## Results

We present the reproducing results at [here](results.md).
- Mamba
  - [MuJoCo](results.md#mambamujoco)
  - [PyBullet POMDP](results.md#mambapybullet-pomdp)
  - [MuJoCo Gravity Randomization](results.md#mambamujoco-gravity-randomization)
  - [Meta-RL](results.md#mambameta-rl)
  - [DMControl](results.md#mambadeepmind-control)
- GPT
  - [MuJoCo](results.md#gptmujoco)
  - [PyBullet POMDP](results.md#gptpybullet-pomdp)
  
## Citation
```
@article{luo2024efficient,
  title={Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate},
  author={Luo, Fan-Ming and Tu, Zuolin and Huang, Zefang and Yu, Yang},
  journal={arXiv preprint arXiv:2405.15384},
  year={2024}
}
```

## Reference Repository
- [https://github.com/twni2016/pomdp-baselines](https://github.com/twni2016/pomdp-baselines)
- [https://github.com/FanmingL/ESCP](https://github.com/FanmingL/ESCP)
- [https://github.com/FanmingL/SmartLogger](https://github.com/FanmingL/SmartLogger)
- [https://github.com/sustcsonglin/flash-linear-rnn](https://github.com/sustcsonglin/flash-linear-rnn)
- [https://github.com/NicolasZucchet/minimal-LRU](https://github.com/NicolasZucchet/minimal-LRU)
- [https://github.com/zhihanyang2022/off-policy-continuous-control](https://github.com/zhihanyang2022/off-policy-continuous-control)
- [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)