# Results
We report on the replication of the results from the paper using this code. All logs can be found at [this url](https://box.nju.edu.cn/d/fe0603fd2bd5479eb8ee/). Reproducing work is still in progress. We will continue to update this page.

[toc]

## Mamba+MuJoCo
The following results were reproduced using `gen_tmuxp_mamba_mujoco.py`.
### Hopper-v2
final return: 4400

previous SOTA (TD7): 4075

vanilla SAC: 3167

<img src="https://sky.luofm.site:13284/luofm/2024/07/07/668aaf49be50c.png" style="zoom:20%;" />

### Walker2d-v2
final return: 9495

previous SOTA (TD7): 7397

vanilla SAC: 5681

<img src="https://sky.luofm.site:13284/luofm/2024/07/07/668aaf6bdb48b.png" style="zoom: 20%;" />

## Mamba+PyBullet POMDP
final return: 

|               |   AntBLT-P-v0   |   AntBLT-V-v0   |HalfCheetahBLT-P-v0|HalfCheetahBLT-V-v0|HopperBLT-P-v0| HopperBLT-V-v0  |   WalkerBLT-P-v0   |WalkerBLT-V-v0|
|:--------------|:---------------:|:---------------:|:-----------------:|:-----------------:|:------:|:---------------:|:------------------:|:------:|
| Mamba-0708    | $\mathbf{3045}$ | $\mathbf{1779}$ |  $\mathbf{3632}$  |  $\mathbf{2425}$  | $\mathbf{2663}$ |     $2513$      | $\mathbf{2499}$ | $\mathbf{2136}$ |
| Previous SOTA (GPIDE & MF-RNN) |     $2597$      |     $1137$      |      $2802$       |      $2073$       | $2373$ | $\mathbf{2537}$ | $1502$ | $1701$ |

<img src="https://sky.luofm.site:13284/luofm/2024/07/08/668c03006f95b.png" style="zoom: 45%;" />
