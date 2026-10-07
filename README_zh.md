# DQA

[English](README.md)

DQA 是用于连续控制的离策略 actor-critic 算法。两个 critic 分别预测软回报的高斯分布，输出均值和标准差。计算 Bellman 目标时，在同一个下一状态采样多个动作，用全方差公式合并后续回报方差与动作之间的均值方差。

训练、经验回放、评估和模型保存均在 [`dqa_continuous_action.py`](dqa_continuous_action.py) 中实现，依赖 PyTorch 和 Gymnasium。

## 安装

使用 Python 3.10，在项目目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1
python -m pip install -r requirements.txt
```

环境依赖固定为 Gymnasium 0.29.1、MuJoCo 2.3.3 和 NumPy 2.2.6。需要其他 CUDA 构建时，按 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)选择对应的 wheel。训练不创建渲染窗口，无需配置显示服务。

## 训练

单个环境：

```bash
python dqa_continuous_action.py --env-id Hopper-v4 --seed 12345 --device cuda
```

没有 GPU 时使用 `--device cpu`。先运行一次短测，可检查环境、梯度更新和数据保存：

```bash
python dqa_continuous_action.py --env-id Hopper-v4 --device cpu \
  --total-timesteps 256 --learning-starts 32 --buffer-size 512 \
  --batch-size 32 --hidden-size 32 --num-target-action-samples 4 \
  --eval-episodes 2 --eval-frequency 256 --log-frequency 64 \
  --output-dir smoke
```

五个环境、五个种子的批量实验：

```bash
python run_multi_gpu.py --gpus 0,1 --jobs-per-gpu 1 \
  --env-ids Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4 \
  --seeds 12345 22345 32345 42345 52345
```

加上 `--dry-run` 只查看任务分配。`--jobs-per-gpu` 指定每张卡的并发数，其余任务排队运行。训练参数放在 `--` 后，例如：

```bash
python run_multi_gpu.py --gpus 0,1 -- --num-target-action-samples 32
```

设置了 `CUDA_VISIBLE_DEVICES` 时，`--gpus` 使用其中的原始编号。某个任务失败后，启动器会停止该批次的其余任务。

## 算法

critic 输出 `mu` 和 `sigma = softplus(raw_sigma) + 1e-6`。对于转移 `(s, a, r, s', d)`，从目标策略独立采样 M 个下一动作。每个动作选择均值较小的目标 critic，同时取该 critic 的均值和标准差。

令 `h_j = mu'_j - alpha * log pi_target(a'_j | s')`，则：

```text
mu_y    = r + gamma * (1 - d) * mean_j(h_j)
v_y     = gamma^2 * (1 - d)^2 * [mean_j(sigma'_j^2) + Var_j(h_j)]
sigma_y = sqrt(max(v_y, 0))
L_i     = E[(mu_i - mu_y)^2 + lambda * (sigma_i - sigma_y)^2]
```

`Var_j` 的分母为 `M - 1`，熵项也参与动作间方差的计算。`d` 只表示真正终止；时间上限导致的截断仍使用重置前的下一观测做 bootstrap。奖励缩放默认为 1。

`lambda = 1` 时，critic 损失是预测高斯与目标矩匹配高斯之间的 W2 平方距离。actor 最小化 `E[alpha * log pi(a | s) - min_i mu_i(s, a)]`。

这里假设给定状态动作后，奖励和转移是确定的。目标方差来自记录的下一状态上的策略动作采样，不包含额外的随机转移方差。样本方差无偏，但开平方得到的标准差在有限 M 下通常有偏。

## 默认参数

| 参数 | 设置 |
| --- | --- |
| 环境步数 | 1,000,000 |
| actor / critic 学习率 | 1e-4 / 1e-3 |
| 熵系数 | 固定 0.2 |
| 下一动作采样数 M | 8 |
| actor | 两层隐藏层，每层 256，ReLU |
| critic | 三层隐藏层，每层 256，LayerNorm + ReLU |
| 目标 actor | 使用 Polyak 平均 |
| gamma / tau | 0.99 / 0.005 |
| replay 容量 / batch size | 1,000,000 / 256 |
| 初始随机动作步数 | 10,000 |
| critic / actor 梯度裁剪 | 10 / 5 |
| 标准差损失权重 | 1 |
| 评估 | 每 10,000 步，10 个确定性策略回合 |

默认每个训练步分别更新一次 critic、actor 和目标网络。`--autotune` 开启自动熵系数，初始值为 0.2，优化器使用 actor 学习率。完整选项见 `python dqa_continuous_action.py --help`。

`--target-mode` 还提供三种消融：`mean_only` 保留多动作均值目标并关闭标准差损失；`conditional` 用一个下一动作传播条件标准差；`dirac` 用一次回报采样回归均值，并将目标标准差设为零。默认模式为 `dqa`。

## 参考结果

下表来自 **MuJoCo v3**，使用上述统一参数。先对每个种子在 900k 至 1M 步的 11 次评估取平均，再计算五个种子间的均值和样本标准差。

| 环境 | DQA 回报 |
| --- | ---: |
| Hopper-v3 | 3486.0 +/- 258.6 |
| Walker2d-v3 | 4872.6 +/- 291.4 |
| HalfCheetah-v3 | 12136.9 +/- 1498.1 |
| Ant-v3 | 6494.7 +/- 129.2 |
| Humanoid-v3 | 5586.9 +/- 520.1 |

各个种子的结果和评估曲线见 [`results/`](results/README.md)。五个环境使用同一套配置，保留全部五个种子。

上面的安装与训练命令使用原生 MuJoCo **v4** 环境。v3 结果依赖 `mujoco-py` 和 MuJoCo 2.1，不能直接把 v4 训练结果填入这张表。v3 的依赖和运行命令见[复现说明](results/README.md)。

## 输出文件

每次实验在 `runs/` 下创建独立目录，保存配置、实际软件版本、训练源码和以下数据：

| 文件 | 内容 |
| --- | --- |
| `evaluations.csv` | 独立评估及每个评估回合的回报 |
| `episodes.csv` | 训练回合的回报与长度 |
| `metrics.csv` | 损失、alpha、梯度范数与更新次数 |
| `sigma_diagnostics.csv` | 固定样本上两个 critic 的标准差统计 |
| `diagnostic_panel.npz` | 诊断使用的观测和动作样本 |
| `checkpoints/` | 定期和最终网络、优化器状态 |
| `status.json` | 完成状态与实际训练步数 |

收到 SIGINT 或 SIGTERM 后，程序会请求保存最终 checkpoint 再退出。目前提供模型保存，不提供精确断点续训。测试命令为：

```bash
python -m unittest discover -s tests -v
```

测试覆盖目标矩、熵项、critic 选择、终止与截断、动作范围、模型保存和任务调度。调度测试使用 CPU 子进程，无需 GPU。

## 许可

[MIT](LICENSE)。actor 与训练结构基于 [CleanRL](https://github.com/vwxyzjn/cleanrl)，上游来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
