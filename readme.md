# Off-Policy RL with RNN and Transformers

This repository is the official implementation of the paper "[Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate.](https://arxiv.org/abs/2405.15384)" It includes implementations of SAC and TD3 based on RNN and Transformer architectures.
## Features
The algorithms implemented in this repository have the following features:
1. We train the recurrent policy and values using full-length trajectories instead of sequence fragments;
2. To enhance training stability with full-length trajectories, we utilize the [Context-Encoder-Specific Learning Rate](https://arxiv.org/abs/2405.15384) (RESeL) technique;
3. [TODO] We provide a set of training hyperparameters that can achieve state-of-the-art performance in different environments of POMDP and MDP.

## Supported Layer Types
This repository supports the following neural network architectures. We have tested with training on `gru`, `mamba`, `smamba`, and `cgpt` layer types. The training speed from fastest to slowest is: `smamba`, `mamba`, `cgpt`, `gru`.

| Layer       | layer_type | Parameters                                                | Notes                                                                                                                                                                                                                                                                |
|-------------| ---------- | --------------------------------------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GRU         | gru        |                                                           | PyTorch's built-in GRU, non-parallel, but still achieves good policy performance                                                                                                                                                                                     |
| Mamba       | smamba     | state_dim, conv1d_size, block_num, norm_type              | Official Mamba implementation, accelerated with selective_scan parallelization. Example: `smamba_s32_c16_b2_nln` means `state_dim=32`, `conv1d_size=16`, `block_num=2`, `norm_type=layer_norm`                                                                       |
| Mamba       | mamba      | state_dim, conv1d_size                                    | Mamba implemented with Triton, serial computation, significantly faster than the PyTorch implementation. Example: `mamba_s32_c16` means `state_dim=32`, `conv1d_size=16`                                                                                             |
| GILR        | gilr       |                                                           | Linear RNN structure implemented with Triton                                                                                                                                                                                                                         |
| LRU         | lru        |                                                           | Linear Recurrent Unit (LRU) implemented with Triton                                                                                                                                                                                                                                            |
| Transformer | cgpt       | head_num, block_num, dropout_prob, max_length, norm_type  | Custom GPT structure, accelerated with flash_attention for training and inference, using bf16 data type in multi-head-attention. Example: `cgpt_h8_l6_p0.1_ml1024_rms` means `head_num=8`, `block_num=6`, `dropout_prob=0.1`, `max_length=1024`, `norm_type=RMSNorm` |
| Transformer | gpt        | head_num, block_num, dropout_prob, max_length             | GPT structure from the flash_attn library, accelerated with flash_attention, using bf16 data type. Example: `gpt_h8_l6_p0.1_ml1024` means `head_num=8`, `block_num=6`, `dropout_prob=0.1`, `max_length=1024`                                                         |

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
docker run --rm -it -v $PWD:/workspace --gpus all core.116.172.93.164.nip.io:30670/public/luofanming:20240607150538 /bin/bash
```

## Starting Training
We use Python files starting with `gen_tmuxp` to record the training hyperparameters. For instance, to start an experiment with the `cgpt` structure, you can run:
```bash
cd /path/to/Recurrent-Offpolicy-RL
pip install -e .
python gen_tmuxp_gpt.py
tmuxp load run_all.json
```

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
The [paper](https://arxiv.org/pdf/2405.15384) of RESeL provides extensive algorithm comparisons. The performance comparison on POMDP tasks is as follows, with GPIDE-ESS being the previous SOTA algorithm. 

|                     |           RESeL-Mamba (ours)           |    PPO-GRU    |    MF-RNN     | SAC-Transformer |   SAC-MLP    |   TD3-MLP    |                GPIDE-ESS                |      VRM       |    A2C-GRU    |
| :------------------ |:--------------------------------------:| :-----------: | :-----------: | :-------------: | :----------: | :----------: | :-------------------------------------: | :------------: | :-----------: |
| AntBLT-P-v0         | $\mathbf{2829} \pm\mathbf{56} ^\star$  | $2103\pm 80$  |  $352\pm 88$  |   $894\pm 36$   | $1147\pm 49$ | $897\pm 83$  |              $2597\pm 76$               |  $323\pm 37$   |  $916\pm 60$  |
| AntBLT-V-v0         | $\mathbf{1971} \pm\mathbf{60} ^\star$  | $690\pm 158$  | $1137\pm 178$ |   $692\pm 89$   | $651\pm 65$  | $476\pm 114$ |              $1017\pm 80$               |  $291\pm 23$   |  $264\pm 60$  |
| HalfCheetahBLT-P-v0 | $\mathbf{2900} \pm\mathbf{179} ^\star$ | $1460\pm 143$ | $2802\pm 88$  |  $1400\pm 655$  | $970\pm 47$  | $906\pm 19$  |              $2466\pm 129$              | $-1317\pm 217$ |  $353\pm 74$  |
| HalfCheetahBLT-V-v0 | $\mathbf{2678} \pm\mathbf{176} ^\star$ | $1072\pm 195$ | $2073\pm 69$  |  $-449\pm 723$  | $513\pm 77$  | $177\pm 115$ |              $1886\pm 165$              | $-1443\pm 220$ | $-412\pm 191$ |
| HopperBLT-P-v0      | $\mathbf{2769} \pm\mathbf{85} ^\star$  | $1592\pm 60$  | $2234\pm 102$ |  $1763\pm 498$  | $310\pm 35$  | $490\pm 140$ |              $2373\pm 568$              |  $557\pm 85$   |  $467\pm 78$  |
| HopperBLT-V-v0      |              $2480\pm 91$              | $438\pm 126$  | $1003\pm 426$ |  $240\pm 192$   |  $243\pm 4$  | $223\pm 28$  | $\mathbf{2537} \pm\mathbf{167} ^\star$ |  $476\pm 28$   | $301\pm 155$  |
| WalkerBLT-P-v0      | $\mathbf{2505} \pm\mathbf{96} ^\star$  | $651\pm 156$  | $940\pm 272$  |  $1150\pm 320$  | $483\pm 86$  | $505\pm 32$  |              $1502\pm 521$              |  $372\pm 96$   | $200\pm 104$  |
| WalkerBLT-V-v0      | $\mathbf{1901} \pm\mathbf{39} ^\star$  |  $423\pm 89$  |  $160\pm 38$  |   $39\pm 18$    | $214\pm 17$  | $214\pm 22$  |              $1701\pm 160$              |  $216\pm 71$   |   $26\pm 5$   |

The performance in classic MuJoCo tasks are as follows

|                |           RESeL-Mamba            |
| :------------- |:---------------------------------------:|
| Ant-v2         |  $\mathbf{8006} \pm\mathbf{63} ^\star$  |
| HalfCheetah-v2 | $\mathbf{16750} \pm\mathbf{432} ^\star$ |
| Hopper-v2      |  $\mathbf{4408} \pm\mathbf{5} ^\star$   |
| Humanoid-v2    | $\mathbf{10490} \pm\mathbf{381} ^\star$ |
| Walker2d-v2    | $\mathbf{8004} \pm\mathbf{150} ^\star$  |


We recently worked on combining RESeL with Transformer. Some of the experimental results are as follows:

|                     | RESeL-Transformer |
| :------------------ |:-----------------:|
| AntBLT-V-v0         |      $2369$       |
| HalfCheetah-v2      |      $16322$      |
| HalfCheetahBLT-P-v0 |      $2605$       |
| HalfCheetahBLT-V-v0 |      $2686$       |
| HopperBLT-V-v0      |      $2550$       |
| WalkerBLT-V-v0      |      $1314$       |


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