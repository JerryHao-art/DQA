# Attribution

DQA adapts the continuous-action SAC implementation from
[CleanRL](https://github.com/vwxyzjn/cleanrl), commit
`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`.
The reference file is
[`cleanrl/sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/sac_continuous_action.py).

The squashed Gaussian actor, action rescaling, entropy-regularized actor
objective, and target-network updates follow that implementation. DQA adds
Gaussian return critics, moment targets over multiple next actions, a target
actor, and the corresponding critic loss. The replay buffer, evaluation,
diagnostics, and experiment launcher are local implementations.

The full upstream license and its bundled notices are preserved in
[LICENSE.cleanrl](LICENSE.cleanrl). Its SHA-256 is
`c6c269472eecb009754856b5fe1029ee53b99b986f57bf22ad7c747b04799c69`.
The repository is distributed under [MIT](LICENSE).

Environment implementations are provided by
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and
[MuJoCo](https://github.com/google-deepmind/mujoco), installed as dependencies.
Legacy v3 environments additionally use
[mujoco-py](https://github.com/openai/mujoco-py).
