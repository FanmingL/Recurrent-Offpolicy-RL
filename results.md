# Results
We report on the replication of the results from the paper using this code. All logs can be found at [this url](https://box.nju.edu.cn/d/fe0603fd2bd5479eb8ee/). Reproducing work is still in progress. We will continue to update this page.

**Note:** Due to some atomic operations in Mamba during backpropagation, Mamba’s backpropagation has randomness that cannot be eliminated with fixing random seeds. Therefore, experiments based on the `smamba` layer cannot be fully reproduced. For more details, see this link: [state-spaces/mamba#137 (comment)](https://github.com/state-spaces/mamba/issues/137#issuecomment-1918483734). On the contrary, the `mamba` layer is implemented with Triton and is reproducible.


## Mamba+MuJoCo
The following results were reproduced using [gen_tmuxp_mamba_mujoco.py](gen_tmuxp_mamba_mujoco.py).

### Hopper-v2
final return: 4400

previous SOTA ([TD7](https://arxiv.org/abs/2306.02451)): 4075

vanilla [SAC](https://arxiv.org/abs/1801.01290): 3167

<img src="https://sky.luofm.site:13284/luofm/2024/07/07/668aaf49be50c.png" style="zoom:20%;" />

### Walker2d-v2
final return: 9495

previous SOTA ([TD7](https://arxiv.org/abs/2306.02451)): 7397

vanilla [SAC](https://arxiv.org/abs/1801.01290): 5681

<img src="https://sky.luofm.site:13284/luofm/2024/07/07/668aaf6bdb48b.png" style="zoom: 20%;" />

### Ant-v2

final return: 8454

SOTA ([TD7](https://arxiv.org/abs/2306.02451)): 10133

vanilla [SAC](https://arxiv.org/abs/1801.01290): 4615



<img src="https://sky.luofm.site:13284/luofm/2024/07/11/668f3ac1936fe.png" style="zoom: 20%;" />

### Humanoid-v2

final return: 10708

previous SOTA ([TD7](https://arxiv.org/abs/2306.02451)): 10281

vanilla [SAC](https://arxiv.org/abs/1801.01290): 6555



<img src="https://sky.luofm.site:13284/luofm/2024/07/11/668f7852a04e2.png" style="zoom: 20%;" />



## Mamba+PyBullet POMDP

The following results were reproduced using [gen_tmuxp_mamba_pomdp.py](gen_tmuxp_mamba_pomdp.py)

final return: 

|                                                                                  |   AntBLT-P-v0   |   AntBLT-V-v0   |HalfCheetahBLT-P-v0|HalfCheetahBLT-V-v0|HopperBLT-P-v0| HopperBLT-V-v0  |   WalkerBLT-P-v0   |WalkerBLT-V-v0|
|:---------------------------------------------------------------------------------|:---------------:|:---------------:|:-----------------:|:-----------------:|:------:|:---------------:|:------------------:|:------:|
| Mamba-0708                                                                       | $\mathbf{3045}$ | $\mathbf{1779}$ |  $\mathbf{3632}$  |  $\mathbf{2425}$  | $\mathbf{2663}$ |     $2513$      | $\mathbf{2499}$ | $\mathbf{2136}$ |
| Previous SOTA ([GPIDE](https://openreview.net/forum?id=pKnhUWqZTJ) & [MF-RNN](https://arxiv.org/abs/2110.05038)) |     $2597$      |     $1137$      |      $2802$       |      $2073$       | $2373$ | $\mathbf{2537}$ | $1502$ | $1701$ |

<img src="https://sky.luofm.site:13284/luofm/2024/07/08/668c03006f95b.png" style="zoom: 45%;" />


## Mamba+MuJoCo Gravity Randomization


The following results were reproduced using [gen_tmuxp_mamba_dynamics_rnd.py](gen_tmuxp_mamba_dynamics_rnd.py)

final return: 

|                                                                                                                            | DM-Ant-gravity-v2 |DM-HalfCheetah-gravity-v2|DM-Hopper-gravity-v2|DM-Humanoid-gravity-v2|DM-Walker2d-gravity-v2|
|:---------------------------------------------------------------------------------------------------------------------------|:-----------------:|:------:|:------------------:|:--------------------:|:--------------------:|
| Mamba-0708-gravity                                                                                                         |  $\mathbf{6185}$  | $\mathbf{7980}$ |  $\mathbf{3304}$   |   $\mathbf{7495}$    |   $\mathbf{5331}$    | 
| Previous SOTA ([ESCP](https://ojs.aaai.org/index.php/AAAI/article/view/20730) & [PEARL](https://arxiv.org/abs/1903.08254)) |      $4065$       | $7022$ |       $2683$       |        $3857$        |        $3242$        | 

<img src="https://sky.luofm.site:13284/luofm/2024/07/09/668cd7bc68acd.png" style="zoom: 45%;" />

## GPT+PyBullet POMDP

The following results were reproduced using [gen_tmuxp_gpt_pomdp.py](gen_tmuxp_gpt_pomdp.py)



final return: 

|                                                              |   AntBLT-P-v0   |   AntBLT-V-v0   | HalfCheetahBLT-P-v0 | HalfCheetahBLT-V-v0 | HopperBLT-P-v0  | HopperBLT-V-v0  | WalkerBLT-P-v0  | WalkerBLT-V-v0  |
| :----------------------------------------------------------- | :-------------: | :-------------: | :-----------------: | :-----------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|CGPT-0711-2 | $\mathbf{2934} $ | $\mathbf{2310}$ | $\mathbf{3190}$ | $ \mathbf{3099}$ | $2297$ | $2431$ | $\mathbf{2000}$ | $1372$ |
| Previous SOTA ([GPIDE](https://openreview.net/forum?id=pKnhUWqZTJ) & [MF-RNN](https://arxiv.org/abs/2110.05038)) |     $2597$      |     $1137$      |       $2802$        |       $2073$        |     $\mathbf{2373}$     | $\mathbf{2537}$ |     $1502$      |     $\mathbf{1701}$     |





<img src="https://sky.luofm.site:13284/luofm/2024/07/12/6690847b0b647.png" style="zoom: 45%;" />
