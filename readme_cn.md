# Off-Policy RL with RNN and Transformers
本仓库实现了基于RNN以及Transformer的SAC以及TD3。
## 特性
本仓库所实现的算法有以下特性
1. 我们使用全长的轨迹进行训练，而不是序列片段；
2. 为了提升全长轨迹的训练稳定性，我们使用了[Context-Encoder-Specific Learning Rate](https://arxiv.org/abs/2405.15384) (RESeL)技术；
3. [TODO] 我们对POMDP、MDP的不同环境提供了一套可达到SOTA效果的训练超参数。

## 支持的网络结构
本仓库支持以下的神经网络结构，已经在gru, mamba, smamba, cgpt两种layer_type上进行了训练测试，训练的速度从快到慢的排名为: smamba, mamba, cgpt, gru.

| Layer Name  | layer_type | 参数                                                     | 备注                                                                                                                                                                       |
| ----------- | ---------- | -------------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GRU         | gru        |                                                          | pytorch自带的GRU，无法并行，但策略性能并不差                                                                                                                                              |
| Mamba       | smamba     | state_dim, conv1d_size, block_num, norm_type,            | Mamba的官方实现，使用selective_scan并行加速。smamba_s32_c16_b2_nln表示state_dim=32, conv1d_size=16,block_num=2,norm_type=layer_norm                                                     |
| Mamba       | mamba      | state_dim, conv1d_size                                   | 使用triton实现的Mamba，串行计算，要比pytorch实现快不少。mamba_s32_c16表示state_dim=32, conv1d_size=16                                                                                         |
| GILR        | gilr       |                                                          | 使用triton实现的Linear RNN结构                                                                                                                                                  |
| LRU         | lru        |                                                          | 使用Linear Recurrent Unit (LRU)，串行计算。                                                                                                                                      |
| Transformer | cgpt       | head_num, block_num, dropout_prob, max_length, norm_type | 重新搭建的GPT结构，使用flash_attention加速训练与推理，multi-head-attention使用bf16的数据类型。cgpt_h8_l6_p0.1_ml1024_rms表示head_num=8,block_num=6,dropout_prob=0.1,max_length=0.1,norm_type=RMSNorm |
| Transformer | gpt        | head_num, block_num, dropout_prob, max_length            | flash_attn库中实现的GPT结构，使用flash_attention加速，使用bf16的数据类型。gpt_h8_l6_p0.1_ml1024表示head_num=8,block_num=6,dropout_prob=0.1,max_length=0.1                                       |

## 依赖
### 硬件
上述网络结构中，gru可以直接在CPU机器上训练，mamba, gilr, lru基于triton实现，要求在GPU机器上训练，而cgpt, gpt使用flash_attention进行加速，需要在Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100)。
### 环境
因为我们修改了Mamba的CUDA源码，需要重新编译，同时本库依赖于一些早期的RL训练环境，因此强烈推荐直接使用我们已经整理好的docker镜像运行我们的算法。使用docker将镜像pull下来
```bash
docker pull core.116.172.93.164.nip.io:30670/public/luofanming:20240607150538
```
或者从此[链接](https://box.nju.edu.cn/f/11384fd1c05641158dcd/)下载
```bash
wget -O 20240607_flash_attn_image.tar.gz https://box.nju.edu.cn/f/11384fd1c05641158dcd/?dl=1
docker load -i 20240607_flash_attn_image.tar.gz
```
启动环境
```bash
docker run --rm -it -v $PWD:/workspace --gpus all core.116.172.93.164.nip.io:30670/public/luofanming:20240607150538 /bin/bash
```

## 启动训练
我们用以`gen_tmuxp`开头的python文件来记录训练的超参数，如对于`cgpt`结构的策略，我们可以如下启动实验
```bash
python gen_tmuxp_gpt.py
tmuxp load run_all.json
```

## 可视化结果
我们使用[SmartLogger](https://github.com/FanmingL/SmartLogger)来进行日志管理，你可以在logfile中找到训练日志。
最直接的方法你可以直接使用tensorboard看训练过程的曲线
```bash
tensorboard --logdir=./logfile
```
你也可以使用smart_logger中的渲染界面来查看实验数据
```bash
python -m smart_logger.htmlpage -p 4008 -wks /path/to/logfile -t local_plotting -u user -pw resel -cp 600
```
你可以访问
[http://localhost:4008](http://localhost:4008)来查看训练数据，登录用户名user, 密码resel。

## 结果
RESeL的[论文](https://arxiv.org/pdf/2405.15384)中进行了大量算法对比，

在POMDP任务在的性能对比如下，其中GPIDE-ESS是先前的SOTA算法

|                     |              RESeL (ours)               |    PPO-GRU    |    MF-RNN     | SAC-Transformer |   SAC-MLP    |   TD3-MLP    |                GPIDE-ESS                |      VRM       |    A2C-GRU    |
| :------------------ | :-------------------------------------: | :-----------: | :-----------: | :-------------: | :----------: | :----------: | :-------------------------------------: | :------------: | :-----------: |
| AntBLT-P-v0         | $\mathbf{2829} \pm\mathbf{56} ^\star$  | $2103\pm 80$  |  $352\pm 88$  |   $894\pm 36$   | $1147\pm 49$ | $897\pm 83$  |              $2597\pm 76$               |  $323\pm 37$   |  $916\pm 60$  |
| AntBLT-V-v0         | $\mathbf{1971} \pm\mathbf{60} ^\star$  | $690\pm 158$  | $1137\pm 178$ |   $692\pm 89$   | $651\pm 65$  | $476\pm 114$ |              $1017\pm 80$               |  $291\pm 23$   |  $264\pm 60$  |
| HalfCheetahBLT-P-v0 | $\mathbf{2900} \pm\mathbf{179} ^\star$ | $1460\pm 143$ | $2802\pm 88$  |  $1400\pm 655$  | $970\pm 47$  | $906\pm 19$  |              $2466\pm 129$              | $-1317\pm 217$ |  $353\pm 74$  |
| HalfCheetahBLT-V-v0 | $\mathbf{2678} \pm\mathbf{176} ^\star$ | $1072\pm 195$ | $2073\pm 69$  |  $-449\pm 723$  | $513\pm 77$  | $177\pm 115$ |              $1886\pm 165$              | $-1443\pm 220$ | $-412\pm 191$ |
| HopperBLT-P-v0      | $\mathbf{2769} \pm\mathbf{85} ^\star$  | $1592\pm 60$  | $2234\pm 102$ |  $1763\pm 498$  | $310\pm 35$  | $490\pm 140$ |              $2373\pm 568$              |  $557\pm 85$   |  $467\pm 78$  |
| HopperBLT-V-v0      |              $2480\pm 91$               | $438\pm 126$  | $1003\pm 426$ |  $240\pm 192$   |  $243\pm 4$  | $223\pm 28$  | $\mathbf{2537} \pm\mathbf{167} ^\star$ |  $476\pm 28$   | $301\pm 155$  |
| WalkerBLT-P-v0      | $\mathbf{2505} \pm\mathbf{96} ^\star$  | $651\pm 156$  | $940\pm 272$  |  $1150\pm 320$  | $483\pm 86$  | $505\pm 32$  |              $1502\pm 521$              |  $372\pm 96$   | $200\pm 104$  |
| WalkerBLT-V-v0      | $\mathbf{1901} \pm\mathbf{39} ^\star$  |  $423\pm 89$  |  $160\pm 38$  |   $39\pm 18$    | $214\pm 17$  | $214\pm 22$  |              $1701\pm 160$              |  $216\pm 71$   |   $26\pm 5$   |

在Classic MuJoCo任务下的性能如下

|                |               RESeL (OURS)               |
| :------------- | :--------------------------------------: |
| Ant-v2         |  $\mathbf{8006} \pm\mathbf{63} ^\star$  |
| HalfCheetah-v2 | $\mathbf{16750} \pm\mathbf{432} ^\star$ |
| Hopper-v2      |  $\mathbf{4408} \pm\mathbf{5} ^\star$   |
| Humanoid-v2    | $\mathbf{10490} \pm\mathbf{381} ^\star$ |
| Walker2d-v2    | $\mathbf{8004} \pm\mathbf{150} ^\star$  |


我们近期尝试将RESeL与Transformer进行结合，部分实验结果如下

|                     | RESeL-Transformer |
| :------------------ |:-----------------:|
| AntBLT-V-v0         |       2369        |
| HalfCheetah-v2      |       16322       |
| HalfCheetahBLT-P-v0 |       2605        |
| HalfCheetahBLT-V-v0 |       2686        |
| HopperBLT-V-v0      |       2550        |
| WalkerBLT-V-v0      |       1314        |


## 引用
```
@article{luo2024efficient,
  title={Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate},
  author={Luo, Fan-Ming and Tu, Zuolin and Huang, Zefang and Yu, Yang},
  journal={arXiv preprint arXiv:2405.15384},
  year={2024}
}
```

## 参考代码仓库
- [https://github.com/twni2016/pomdp-baselines](https://github.com/twni2016/pomdp-baselines)
- [https://github.com/FanmingL/ESCP](https://github.com/FanmingL/ESCP)
- [https://github.com/FanmingL/SmartLogger](https://github.com/FanmingL/SmartLogger)
- [https://github.com/sustcsonglin/flash-linear-rnn](https://github.com/sustcsonglin/flash-linear-rnn)
- [https://github.com/NicolasZucchet/minimal-LRU](https://github.com/NicolasZucchet/minimal-LRU)
- [https://github.com/zhihanyang2022/off-policy-continuous-control](https://github.com/zhihanyang2022/off-policy-continuous-control)
- [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)