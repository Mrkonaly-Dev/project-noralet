# Operation Report 009 — Online Predictive Lifetime Learning

**Iteration:** 9
**Date:** 2026-08-24
**Status:** Complete

## Summary

Iteration 9 adds the first genuine individual lifetime neural plasticity to Project Noralet. A learning-enabled Noralet now predicts the fixed-size sensory representation of its own next lived Experience, compares that prediction with the representation produced after the real shared-world transition, and performs exactly one individual Adam update.

The implementation remains predictive only. It adds no reward, value, preference, reinforcement learning, policy-gradient training, action evaluation, candidate-action search or motivation. Prediction error changes the online sensory encoder, recurrent core and action-conditioned forward predictor. It never directly optimizes the acceleration, consume or signal action heads.

Lifetime learning is optional. A BaseBrain constructed without `NoraletLearningConfig` creates no predictor, frozen target encoder or optimizer and preserves the Iteration 8 autonomous path and initialization exactly.

## Lifetime learning flow

For every surviving autonomous transition, the implemented order is:

```text
NoraletExperience(t)
        ↓
online ExperienceEncoder + detached h(t-1)
        ↓
GRU produces graph-bearing h(t)
        ↓
fixed action heads + existing three RNG draws
        ↓
selected brain-native Action(t)
        ↓
PredictionModel(h(t), selected action vector)
        ↓
predicted z_target(t+1)
        ↓
all Noralets finish action selection
        ↓
one shared Simulation.step(...)
        ↓
WorldState(t+1)
        ↓
ExperienceBuilder
        ↓
actual NoraletExperience(t+1)
        ↓
frozen target ExperienceEncoder under no-grad
        ↓
actual z_target(t+1)
        ↓
MSE prediction error
        ↓
one backward + gradient clip + Adam step
```

Action creation and prediction both happen before the world transition. Target creation and learning happen only after the next state has been published. The update cannot recompute or replace the already selected action, `TickResult` or `WorldState(t+1)`.

## Stable sensory target

Every learning-enabled spawned brain owns a private `target_experience_encoder`. At spawn it is a deep copy of that individual's initial online `ExperienceEncoder` after the inherited model has been cloned to its execution device. Consequently, the online and target encoders are exactly parameter-equal at birth but use independent tensor storage.

Every target parameter is permanently configured with `requires_grad=False`. Target encoding runs under `torch.no_grad()`. Target parameters are excluded from Adam, retain `grad is None`, receive no backward path, and remain exactly unchanged throughout life. There is no EMA, moving-target update or later synchronization to the changing online encoder.

The target encoder has the same architecture and `experience_embedding_size` output as the online encoder. It receives exactly one `NoraletExperience(t+1)`. It is a fixed deterministic numerical transform, not a knowledgeable teacher. It receives no IDs, object categories, signal enums, exact coordinates, velocity, distance, Energy, condition, age, region data, events, death cause or action-success label.

## PredictionModel

`PredictionModel` is inherited BaseBrain structure and is instantiated only when lifetime learning is enabled. Its input is:

```text
current graph-bearing recurrent h(t)
+
selected 11-value brain-native action vector
```

The action vector contains:

```text
normalized acceleration motor command in [-1, 1]   1
consume intention in {0, 1}                         1
NONE/A_LEFT/A_RIGHT/.../D_RIGHT one-hot             9
                                                    --
total                                               11
```

The acceleration value is the bounded neural motor command before multiplication by the physical `max_acceleration`. The vector contains the selected consume and signal intentions even when the world later prevents their execution. It contains no applied acceleration, physical-unit scale, Energy cost, execution-success flag, action probability or RNG draw.

The predictor is one compact MLP:

```text
concat(h(t), action_vector)
        ↓
Linear(hidden_size + 11, predictor_hidden_size)
        ↓
tanh
        ↓
Linear(predictor_hidden_size, experience_embedding_size)
        ↓
unrestricted next-target prediction
```

It accepts only hidden and action tensors and imports no physical world types.

## Plastic parameters

Predictive loss may update exactly:

- the online `ExperienceEncoder`;
- the one `GRUCell` recurrent core;
- the `PredictionModel`.

It does not update:

- the frozen target `ExperienceEncoder`;
- acceleration-head parameters;
- consume-head parameters;
- signal-head parameters;
- the BaseBrain prototype;
- another living Noralet's parameters or optimizer.

The three action heads are excluded from the optimizer and frozen inside each learning-enabled individual. Their inherited values remain exact. Behaviour may nevertheless change indirectly because their input hidden representation is produced by the plastic online encoder and GRU.

The predictor uses a SHA-256 domain-separated deterministic initialization stream derived from `base_brain_seed`. The pre-existing encoder, GRU and action heads continue using the exact Iteration 8 initialization domain and parameter order. Tests confirm that enabling the predictor does not change any pre-existing initial parameter.

## Temporal gradient semantics

Iteration 9 implements one-transition truncated recurrence. The persistent input `h(t-1)` is detached before current processing. Gradient for the prediction of `t+1` can flow through:

```text
ExperienceEncoder(t) → GRU(t) → h(t) → PredictionModel
```

but cannot flow through earlier ticks. The graph-bearing prediction is retained only in one private pending context until the real next Experience exists. The separately stored persistent `h(t)` is detached immediately and has no `grad_fn`.

At most one pending transition can exist. A second action is rejected until the first pending transition is learned or explicitly discarded. After success or failure, the pending context is cleared and optimizer gradients are set back to `None`. There is no multi-tick graph, BPTT window, replay collection or retained completed transition.

## Optimizer/plasticity

Every learning-enabled individual owns a distinct `torch.optim.Adam` instance with:

- the configured positive finite `learning_rate`;
- default Adam moment coefficients;
- no weight decay;
- exactly the online encoder, GRU and predictor parameters.

`NoraletLearningConfig` contains only:

```text
learning_rate
max_gradient_norm
predictor_hidden_size
```

One surviving lived transition produces one MSE loss, one backward pass and one optimizer step. There are no repeated epochs or replay updates.

Before the step, all gradients are checked for finiteness and clipped by global norm over exactly the plastic parameter set. The observer metric reports the applied post-clipping norm. Non-finite target, prediction, loss or gradient state fails clearly before an optimizer step. Plastic parameters and Adam tensor state are checked again after the step. Gradients are cleared with `zero_grad(set_to_none=True)` before and after learning.

## Death semantics

A Noralet that dies during `t → t+1` has no `NoraletExperience(t+1)` and therefore no prediction target. Its pending graph is discarded and the runner removes the brain. No optimizer step or learning metric is produced for that terminal transition.

Boundary, Energy-depletion and controlled natural-death tests all verify the same rule. There is no synthetic death Experience, terminal vector, death label, reward, penalty or posthumous update. The individual's hidden state, target encoder, learned parameters and optimizer leave runner ownership together with the dead brain.

## Individual development

All learning-enabled Noralets still begin from exact copies of one BaseBrain, including equal predictor parameters and empty individual Adam state. Their online modules, target encoders, hidden states and optimizers use independent storage.

Controlled tests train only one of two clones and confirm that the other clone remains unchanged. Separate tests expose two initially identical brains to different Experience/action/next-Experience histories. Their online encoder, GRU and predictor parameters diverge while their frozen target encoders remain equal. This demonstrates the intended relationship:

```text
same inherited brain + different lived histories → different learned brains
```

## Predictive-learning diagnostic

A controlled CPU diagnostic repeatedly presented one brain with the same finite current Experience and the same deterministic next Experience. Action heads were held at fixed controlled motor outputs for the diagnostic, and each presentation remained a separate act/predict/learn transition with exactly one update.

Recorded result:

```text
updates: 100
initial 10-loss range: 0.020069962 .. 0.262112498
initial 10-loss mean:  0.119914844
final 10-loss range:   0.000001050 .. 0.000005518
final 10-loss mean:    0.000003877
action RNG draws:      300
```

The final ten-loss mean is about 0.0032% of the initial ten-loss mean. This demonstrates only that the implemented individual online network can improve prediction of a repeated learnable sensory transition. It is not evidence of understanding, intelligence, motivation, sentience or consciousness.

## Autonomous integration

`AutonomousSimulationRunner` remains the sole neural-world coordinator. For each tick it:

1. obtains all routed current Experiences;
2. makes every living brain act and create one pending prediction in canonical ID order;
3. calls `Simulation.step()` exactly once with the complete action map;
4. obtains routed Experiences from the published next state;
5. discards and removes dead brains;
6. performs one individual update for each survivor;
7. returns immutable action, physical tick and learning observer results.

No update begins until all actions have been selected and the shared world transition has completed. One individual's update cannot influence any action already chosen for that tick. Learning failures are wrapped with the routing identity for runtime diagnosis without placing identity in neural tensors.

`AutonomousTickResult.learning_results` contains one immutable `NoraletLearningResult(noralet_id, prediction_loss, gradient_norm)` for each brain that actually learned. Reading or serializing these values does not affect neural state, RNG or physics.

## Information boundary

The only prediction-target path is:

```text
WorldState(t+1)
        ↓
ExperienceBuilder
        ↓
NoraletExperience(t+1)
        ↓
frozen target ExperienceEncoder
        ↓
fixed-size sensory target
```

The brain and predictor import no `WorldState`, body-state, region, event, death-cause or exact Energy types. `NoraletBrain.learn()` accepts exactly `NoraletExperience`. The pending context contains only the graph-bearing prediction and selected brain-native action vector. Routing identity remains in the coordinator and observer metric, outside Experience and tensors.

No consume-success, signal-success, survival, boundary-danger, semantic category, good/bad action or other objective consequence label enters the loss.

## No motivation yet

Current Experience already contains subjective energy distress, condition distress and energetic exertion. The frozen target transform therefore allows the predictor to learn correlations between a current situation, selected action and later bodily sensation.

Iteration 9 assigns no utility or preferred sign to those sensations. Lower distress is not a reward; higher distress is not a punishment. Prediction error trains accuracy only. It is not used to choose actions, seek surprise, avoid surprise or evaluate candidate actions. There is no value function or motivation system, and prediction error never optimizes the motor heads.

## Determinism

The predictor is initialized through its own local domain-separated CPU generator. Target creation is an exact copy and uses no random initialization. Adam, MSE and target encoding add no stochastic operations, dropout, augmentation or sampling.

Action selection continues to consume exactly the existing three per-ID named-stream draws. Tests confirm that learning adds no draw to action, ecology, mortality, formation, global Python or global PyTorch RNG state.

On CPU, identical simulation seed, BaseBrain seed, brain configuration, learning configuration and world reproduce exact actions, prediction metrics, learned parameters, hidden states, world states, events and deaths. Body insertion order does not change canonical per-ID learning histories.

CUDA is tested for functional correctness and deterministic-algorithm compatibility on the current device. Bitwise equality between CPU and CUDA, different GPU models or different software stacks is not claimed.

## CUDA validation

The development environment reported:

```text
PyTorch: 2.13.0+cu130
torch.cuda.is_available(): True
PyTorch CUDA runtime: 13.0
GPU: NVIDIA GeForce RTX 3060
```

The focused CUDA learning step performed a real forward, target encode, MSE loss, backward, global gradient clip and Adam update. Plastic parameters changed; target encoder and action heads remained exact; hidden state, online model, target model, predictor and Adam moment tensors were device-compatible; all inspected values were finite.

A second six-tick CUDA autonomous smoke produced finite learning results and parameters, valid actions, evolving neural state and conserved world Energy without device mismatch. The two focused CUDA tests passed in `1.750s`.

## Tests and validation

The clean Iteration 8 baseline at commit `9e6767c` passed all 306 existing tests in `2.475s`.

Iteration 9 adds 48 focused tests across five modules, bringing discovery to 354 tests. The final complete command was:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: all 354 tests passed in `4.662s`, including all CUDA tests. There were no skips, failures or errors.

Additional validation commands:

```powershell
uv run python -m compileall -q src tests
uv lock --check
git diff --check
uv run noralet run --ticks 7 --seed 20260824
uv run python -m unittest discover -s tests -p "test_autonomous_determinism.py" -k finite_autonomous_smoke -v
uv run python -m unittest discover -s tests -p "test_learning_determinism_cuda.py" -k CudaLifetimeLearningTests -v
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Results:

- source and tests compiled successfully;
- the uv lock resolved 30 packages and is current;
- Git diff validation passed with only the repository's LF-to-CRLF conversion notices;
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260824`;
- the existing Iteration 8 20-tick no-learning autonomous smoke passed in `1.132s`;
- the controlled 100-update predictive diagnostic produced the loss reduction reported above;
- PyTorch/CUDA inspection reported `2.13.0+cu130`, CUDA available, runtime `13.0`, and `NVIDIA GeForce RTX 3060`;
- both focused CUDA predictive-learning tests passed;
- searches found no direct world-truth imports, semantic training labels, reward/value/policy code, replay, EMA target update, posthumous target or unrelated implementation.

The final architecture audit verified all sixteen requested invariants:

1. each target comes only from the individual's actual next Experience;
2. no objective world field becomes supervision;
3. the target encoder is frozen for life;
4. the target cannot move with the online representation;
5. prediction is conditioned on recurrent state and selected action;
6. the predictor receives intention before Experience reveals execution;
7. updates are private to one individual;
8. BaseBrain remains exact;
9. prediction loss is not motivation;
10. predictive gradients cannot optimize action heads;
11. recurrence is truncated to one transition;
12. there is no replay, epoch or offline training;
13. death creates no learning target;
14. learning cannot alter its causal predecessor action or transition;
15. learning introduces no random stream;
16. repeated experience changes weights and improves prediction.

## Deviations

There are no implementation deviations from the Iteration 9 instruction.

## Open implementation notes

- The permanently frozen target encoder defines an inherited numerical sensory coordinate system. Later research should treat this stable arbitrary representation as a design constraint when interpreting predictor outputs.
- Learning configuration belongs to BaseBrain construction rather than physical `SimulationConfig`; the world remains independent of whether its external neural coordinator enables individual plasticity.
- Observer `gradient_norm` is the actual post-clipping norm. This keeps the metric directly comparable to `max_gradient_norm`; the unclipped norm is intentionally not added as a fourth metric.
- PyTorch continues to emit its harmless optional NumPy-bridge warning because this implementation requires no NumPy operation.

## Git state

No commit or push was created.

The working tree began clean at commit `9e6767c` (`Add autonomous Noralet brains`). Iteration 9 modifies the neural public exports, BaseBrain, recurrent model/runtime, autonomous coordinator and shared neural test helper. It adds the focused learning configuration/value module, five predictive-learning test modules and this report. Physical simulation, Experience construction, architecture documents, renderer and unrelated implementation were not modified.
