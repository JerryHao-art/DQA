# MuJoCo v3 reference runs

The data covers five environments and five training seeds per environment:
`12345`, `22345`, `32345`, `42345`, and `52345`. Every run completed 1,000,000
environment steps with the same configuration in [config.json](config.json).

| File | Contents |
| --- | --- |
| `evaluations.csv` | 2,525 evaluations, with individual episode returns and lengths |
| `per_seed.csv` | Tail return, final return, and normalized learning-curve AUC for each seed |
| `summary.csv` | Means and sample standard deviations across the five seeds |
| `config.json` | Shared training configuration, with the method named `dqa` |
| `provenance.json` | Recorded software versions and hashes of the original evaluation files and trainer |

Each evaluation uses ten deterministic episodes with seeds
`training_seed + 10000 + episode_index`, where the index runs from 0 to 9.
The tail metric averages eleven evaluation means from 900,000 through
1,000,000 steps, inclusive. AUC is the trapezoidal integral over the complete
0-to-1M curve, divided by 1M. Standard deviations use `ddof=1` across training
seeds. These statistics can be recomputed from `evaluations.csv`.

The data was collected with the standard-deviation-output critic and M=8.
The public trainer keeps that update rule and sets its defaults to this
configuration. The original trainer hash is preserved in `provenance.json`;
renaming the method and changing CLI defaults changes the public source hash.

## Legacy environment setup

The recorded stack is Python 3.10.19, PyTorch 2.4.1, NumPy 2.2.6,
Gymnasium 0.29.1, Gymnasium-Robotics 1.2.4, and mujoco-py 2.1.2.14.
The v3 environments use the MuJoCo 2.1 engine. Although the `mujoco` 2.3.3
Python package is also installed, it serves the native environments, not v3.

Create a separate Python 3.10 environment, install PyTorch as described in
the main README, and install the MuJoCo 2.1 Linux engine under
`~/.mujoco/mujoco210`. Engine downloads are available from the
[MuJoCo releases](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0).

On Ubuntu, install the build dependencies:

```bash
sudo apt-get install build-essential libosmesa6-dev libgl1-mesa-dev libglew-dev patchelf
python -m pip install -r requirements-legacy.txt
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
```

`mujoco-py` compiles its extension on first import. Its system requirements
are documented in the [mujoco-py installation guide](https://github.com/openai/mujoco-py#install-mujoco).
Check the actual v3 environment before launching training:

```bash
python -c "import gymnasium as gym; e = gym.make('Hopper-v3'); e.reset(seed=12345); e.step(e.action_space.sample()); e.close()"
```

Run the reference configuration with explicit v3 environment IDs:

```bash
python run_multi_gpu.py --gpus 0,1 --threads 1 \
  --env-ids Hopper-v3 Walker2d-v3 HalfCheetah-v3 Ant-v3 Humanoid-v3 \
  --seeds 12345 22345 32345 42345 52345 \
  --total-timesteps 1000000 -- \
  --policy-lr 1e-4 --q-lr 1e-3 --num-target-action-samples 8 \
  --alpha 0.2 --no-autotune
```

Changing the environment version, engine, or training hyperparameters produces
a different experiment. Exact numerical trajectories may also vary with
hardware and software versions.
