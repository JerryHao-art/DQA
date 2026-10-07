# DQA

[中文说明](README_zh.md)

DQA is an off-policy actor-critic algorithm for continuous control. Each critic
predicts a Gaussian soft-return distribution. Its target combines the return
variance predicted at the next state with the variance across sampled next
actions, using the law of total variance.

The implementation uses PyTorch and Gymnasium. Training, evaluation, replay,
and checkpointing are in [`dqa_continuous_action.py`](dqa_continuous_action.py).

## Installation

Use Python 3.10. From the repository directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1
python -m pip install -r requirements.txt
```

For a different CUDA build, choose the wheel from the
[PyTorch installation page](https://pytorch.org/get-started/previous-versions/).
The environment dependencies are Gymnasium 0.29.1, MuJoCo 2.3.3, and NumPy 2.2.6.
Training does not require a display or rendering backend.

## Training

Run one experiment:

```bash
python dqa_continuous_action.py --env-id Hopper-v4 --seed 12345 --device cuda
```

Use `--device cpu` on a machine without CUDA. A short run checks the installation:

```bash
python dqa_continuous_action.py --env-id Hopper-v4 --device cpu \
  --total-timesteps 256 --learning-starts 32 --buffer-size 512 \
  --batch-size 32 --hidden-size 32 --num-target-action-samples 4 \
  --eval-episodes 2 --eval-frequency 256 --log-frequency 64 \
  --output-dir smoke
```

To run five environments and five seeds on two GPUs:

```bash
python run_multi_gpu.py --gpus 0,1 --jobs-per-gpu 1 \
  --env-ids Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4 \
  --seeds 12345 22345 32345 42345 52345
```

Add `--dry-run` to inspect the assignment. Increase `--jobs-per-gpu` to allow
concurrent experiments on each GPU. Extra training arguments follow `--`:

```bash
python run_multi_gpu.py --gpus 0,1 -- --num-target-action-samples 32
```

The launcher respects `CUDA_VISIBLE_DEVICES`; GPU identifiers refer to its
original entries. A failed run stops the remaining jobs in that launch.

## Method

Each critic outputs a mean and a positive standard deviation:

```text
Q_i(s, a) = (mu_i(s, a), sigma_i(s, a))
sigma_i = softplus(raw_sigma_i) + 1e-6
```

For a replay transition `(s, a, r, s', d)`, draw `M` independent actions from
the target policy at `s'`. For each action, select the target critic with the
smaller mean and take both its mean and standard deviation. With
`h_j = mu'_j - alpha * log pi_target(a'_j | s')`, the targets are

```text
mu_y    = r + gamma * (1 - d) * mean_j(h_j)
v_y     = gamma^2 * (1 - d)^2 * [mean_j(sigma'_j^2) + Var_j(h_j)]
sigma_y = sqrt(max(v_y, 0))
```

`Var_j` uses the sample-variance denominator `M - 1`. The entropy term is
included inside this variance. Rewards use `--reward-scale`, which defaults
to 1. Only true termination sets `d = 1`; time-limit truncation still
bootstraps from the observation before reset.

The loss for critic `i` is

```text
L_i = E[(mu_i - mu_y)^2 + lambda * (sigma_i - sigma_y)^2]
```

With `lambda = 1`, this is the squared Wasserstein-2 distance between the
predicted Gaussian and a Gaussian fitted to the target moments. The actor
minimizes `E[alpha * log pi(a | s) - min_i mu_i(s, a)]`.

The target calculation assumes deterministic rewards and transitions given
`(s, a)`. It estimates variance from policy actions at the recorded next
state; it does not estimate additional transition noise. The sampled variance
is unbiased, but taking its square root introduces finite-sample bias.

## Default configuration

| Parameter | Value |
| --- | --- |
| Environment steps | 1,000,000 |
| Actor / critic learning rate | 1e-4 / 1e-3 |
| Entropy coefficient | Fixed, 0.2 |
| Target action samples | 8 |
| Actor | 2 hidden layers, 256 units, ReLU |
| Critic | 3 hidden layers, 256 units, LayerNorm + ReLU |
| Target actor | Polyak average of the actor |
| Discount / Polyak coefficient | 0.99 / 0.005 |
| Replay capacity / batch size | 1,000,000 / 256 |
| Initial random-action steps | 10,000 |
| Critic / actor gradient clipping | 10 / 5 |
| Standard-deviation loss weight | 1 |
| Evaluation | 10 deterministic episodes every 10,000 steps |

The default schedule performs one critic update, one actor update, and one
target update per training step. `--autotune` enables entropy tuning, initialized
at 0.2 with the actor learning rate. Run `python dqa_continuous_action.py --help`
for all options.

Three ablations are available through `--target-mode`: `mean_only` keeps the
multi-action mean target and removes the scale loss; `conditional` uses one
next action and propagates its conditional scale; `dirac` regresses to one
sampled return with zero target scale. The default mode is `dqa`.

## Reference results

These are **MuJoCo v3** results for the default algorithm configuration above.
Each value averages the last 11 evaluations, from 900k to 1M steps, within a
seed, then reports the mean and sample standard deviation over five seeds.

| Environment | DQA return |
| --- | ---: |
| Hopper-v3 | 3486.0 +/- 258.6 |
| Walker2d-v3 | 4872.6 +/- 291.4 |
| HalfCheetah-v3 | 12136.9 +/- 1498.1 |
| Ant-v3 | 6494.7 +/- 129.2 |
| Humanoid-v3 | 5586.9 +/- 520.1 |

Per-seed values and evaluation curves are in [`results/`](results/README.md).
All environments use the same configuration; the table does not select the
best seed or a separate configuration for each task.

The installation above runs native MuJoCo **v4** environments. The v3 results
use the older MuJoCo 2.1 engine through `mujoco-py`; v3 and v4 scores are
different experimental settings. See [the reproduction notes](results/README.md)
for the v3 dependencies and command.

## Outputs

Each run creates its own directory under `runs/`.

| File | Contents |
| --- | --- |
| `config.json`, `runtime.json` | Arguments, software versions, device, and source hash |
| `evaluations.csv` | Deterministic evaluation returns, including individual episodes |
| `episodes.csv` | Training episode returns and lengths |
| `metrics.csv` | Losses, entropy coefficient, gradients, and update counts |
| `sigma_diagnostics.csv` | Critic scale statistics on a fixed observation/action sample |
| `diagnostic_panel.npz` | The observation/action sample used for diagnostics |
| `checkpoints/` | Periodic and final network and optimizer states |
| `status.json` | Completion state and training step count |

The training source is copied into each run directory. SIGINT and SIGTERM
request a final checkpoint before exit. Checkpoints can be used for policy
evaluation; exact training resumption is not implemented.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover target moments, entropy contributions, critic selection,
terminal/truncation handling, action bounds, checkpointing, and GPU scheduling.
The scheduling tests use CPU subprocesses and require no GPU.

## License

[MIT](LICENSE). The actor and training structure build on
[CleanRL](https://github.com/vwxyzjn/cleanrl). Upstream source and license
details are in [THIRD_PARTY.md](THIRD_PARTY.md).
