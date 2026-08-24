# Operation Report 008 — NoraletBrain and Autonomous Action

**Iteration:** 8
**Date:** 2026-08-24
**Status:** Complete

## Summary

Iteration 8 introduces the first complete neural control substrate for Project Noralet. It adds a compact PyTorch `NoraletBrain`, variable-length set encoders, one persistent `GRUCell` hidden state per individual, a deterministic shared `BaseBrain` prototype, stochastic low-level action generation, an optional physical acceleration actuator limit, and a separate `AutonomousSimulationRunner`.

The implemented neural flow is:

```text
NoraletExperience(t) + h(t-1)
        ↓
set-based Experience encoder
        ↓
x(t)
        ↓
GRUCell
        ↓
h(t)
        ↓
acceleration / consume / signal heads
        ↓
three fixed isolated random draws
        ↓
ActionIntent(t)
        ↓
Simulation.step(all intents)
        ↓
WorldState(t+1)
```

There is intentionally no lifetime learning in this iteration. Hidden states change, but model parameters do not.

## PyTorch and device model

PyTorch is now a declared project dependency:

```toml
torch>=2.13.0

[tool.uv.sources]
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

The explicit official PyTorch CUDA 13.0 index makes the accelerator choice reproducible through the project configuration. The RTX 3060 development machine reports NVIDIA driver `610.74` with CUDA UMD `13.3`, which supports the selected CUDA 13.0 runtime. `uv.lock` records the CUDA build and its resolved transitive environment. After `uv sync`, the actual development environment reported:

```text
Python: 3.14.3
PyTorch: 2.13.0+cu130
CUDA available: True
PyTorch CUDA runtime: 13.0
CUDA device 0: NVIDIA GeForce RTX 3060
```

`NoraletBrainConfig.device` accepts `cpu`, `cuda` or `auto`. `cpu` is explicit CPU execution. `auto` selects CUDA only when `torch.cuda.is_available()` is true. An explicit `cuda` request raises a clear error when CUDA is unavailable and never silently falls back.

The CUDA smoke test ran successfully on the RTX 3060. It verified that the canonical BaseBrain prototype remains on CPU, spawned brain hidden state resides on CUDA, recurrent encoding and action selection produce finite tensors, and one autonomous world tick completes without a device mismatch.

`BaseBrain` enables PyTorch deterministic algorithms when the neural substrate is instantiated. The complete suite and focused CPU and CUDA autonomous smokes confirm that the required MLP, GRUCell and action-head operations work under this setting. Reproducibility is scoped to identical code, configuration, seeds, device/backend and initial world; no bitwise CPU-versus-CUDA or cross-hardware guarantee is claimed.

PyTorch 2.13's optional NumPy bridge emits a warning because NumPy is not installed. Neural construction and inference require no NumPy functionality, so NumPy was not added solely to silence that optional integration warning.

The root package exposes neural types lazily. Existing manual simulation and CLI startup therefore do not import PyTorch unless a neural API is requested.

## BaseBrain

`BaseBrain` is one inherited random neural prototype, not an evolved or pretrained policy. Construction receives:

- immutable `NoraletBrainConfig`;
- the existing Experience configuration for input dimensions;
- the existing signal configuration for signal-pattern dimensions;
- the physical actuator configuration.

The explicit `base_brain_seed` is domain-separated through SHA-256 and supplied to a local CPU `torch.Generator`. Every prototype parameter is initialized from that generator. Temporary default module construction is isolated inside `torch.random.fork_rng`, so BaseBrain creation does not consume or depend on mutable global PyTorch RNG state.

The canonical prototype stays on CPU. `BaseBrain.spawn()` deep-clones the complete module, moves only the clone to the configured neural device, and creates a zero hidden vector there. Therefore all initial brains begin parameter-equal to the prototype and to one another, while every parameter tensor has independent storage. Mutating one clone does not alter the prototype or another clone.

No per-Noralet weight perturbation or randomized initial hidden state exists.

## Experience encoder

`ExperienceEncoder` accepts exactly `NoraletExperience`. Its structure is:

```text
external percept
    appearance_pattern + direction_signal + proximity_signal
    → shared external MLP
    → sum pooling

signal percept
    signal_pattern + direction_signal + strength_signal
    → separate shared signal MLP
    → sum pooling

interoception
    energy_distress + condition_distress + energetic_exertion
    → interoception MLP

sensorimotor feedback
    complete public scalar/pattern structure
    → sensorimotor MLP

concatenated summaries
    → fusion MLP
    → x(t)
```

Each small MLP uses two learned linear layers with `tanh` activations. External and signal collections remain arbitrary-length. They are neither padded to an arbitrary maximum nor truncated. Every percept in a channel uses the same channel-specific MLP, and sum pooling makes the result permutation-invariant while preserving multiplicity. An empty set bypasses the MLP and produces an exact zero vector of the configured summary dimension.

Raw input dimensions are derived from the already validated sensory schemas:

```text
external input = appearance length + 2
signal input = signal-pattern length + 2
interoception input = 3
sensorimotor input = signal-pattern length + 6
```

The configurable learned dimensions are:

```text
external_percept_embedding_size
signal_percept_embedding_size
interoception_embedding_size
sensorimotor_embedding_size
experience_embedding_size
```

The encoder never converts appearance or signal patterns back into engine object classes or signal enums.

## Recurrent core

The model contains exactly one `torch.nn.GRUCell`:

```text
h(t) = GRUCell(x(t), h(t-1))
```

`hidden_size` is a positive configurable dimension. Each spawned `NoraletBrain` owns a separate zero-initialized hidden tensor on its configured device. `activate(experience)` executes one recurrent update and replaces the stored hidden tensor with the detached result. The next activation receives that stored result rather than a reset zero vector.

Different Experience histories can therefore separate initially identical individuals immediately while their parameters remain equal. Hidden state remains outside `WorldState`, body state and Experience. Removing a dead brain from the runner releases the runner's model and hidden-state ownership.

## Action system

The model exposes only three low-level action heads.

Acceleration produces one scalar `acceleration_loc`. Autonomous sampling consumes one uniform stream draw and converts it through the standard-normal inverse CDF to obtain `z`. It then applies:

```text
raw_motor_value = acceleration_loc
                  + acceleration_exploration_std * z

requested_acceleration = max_acceleration * tanh(raw_motor_value)
```

`acceleration_exploration_std` is finite and non-negative. The draw is still consumed when it is zero.

Consume produces one logit, converts it with a numerically stable sigmoid, and performs a Bernoulli comparison against the second uniform draw. Autonomous execution never substitutes a deterministic `0.5` threshold.

The signal head produces exactly nine raw logits. A stable softmax and the third uniform draw select exactly one of:

```text
NONE
A_LEFT  A_RIGHT
B_LEFT  B_RIGHT
C_LEFT  C_RIGHT
D_LEFT  D_RIGHT
```

The mapping to the existing `SignalEmissionIntent` happens outside the neural model. No output unit has an environmental or communicative semantic label, and one sampled category can create at most one emission.

Each autonomous brain owns one stream with a stable name of the form:

```text
brain:action:noralet:<ID-length>:<ID>
```

Every activation makes exactly three unconditional `random()` calls in this fixed order:

```text
1. acceleration standard-normal source draw
2. consume uniform draw
3. signal-category uniform draw
```

Sampling uses the simulation's explicit named Python RNG streams, never global Python RNG, global PyTorch RNG or CUDA sampling RNG.

## Physical actuator limit

`NoraletActuatorConfig.max_acceleration` is a finite positive physical body limit. When configured, `Simulation` applies the same saturation to neural and externally supplied manual intents:

```text
raw requested acceleration
        ↓
[-max_acceleration, +max_acceleration] physical saturation
        ↓
existing Energy affordability
        ↓
actually applied acceleration
```

This enforcement exists in both legacy and Energy-enabled motion paths. The neural `tanh` request is already bounded, but Simulation does not trust that property. Positive and negative manual saturation and the ordering relative to Energy affordability are covered directly.

The actuator configuration is optional. When absent, Iteration 1–7 manual simulations preserve their original unbounded-intent semantics. Autonomous neural control requires an actuator configuration.

## Autonomous coordinator

`AutonomousSimulationRunner` owns the routing-only map:

```text
Noralet simulation ID → independent NoraletBrain
```

At initialization it spawns one BaseBrain clone for every currently living Noralet and attaches the matching isolated action stream. Each autonomous step:

1. obtains an immutable tuple of `RoutedNoraletExperience` values for the current published moment;
2. activates every living brain exactly once in canonical identity order;
3. collects all resulting `ActionIntent` values without changing the world;
4. calls `Simulation.step()` once with the complete intent mapping;
5. removes brains whose bodies are absent after resolution;
6. returns an `AutonomousTickResult` containing the observer-visible routed intents and physical `TickResult`.

The route wrapper keeps `noralet_id` beside, not inside, `NoraletExperience`. Brain activation receives only the wrapper's Experience value; its ID was used earlier only to select the brain and its RNG stream.

Repeated Experience inspection does not activate brains or consume action RNG. Population insertion order cannot change per-ID autonomous histories. An individual killed by a physical death cause receives its last pre-transition activation, is removed after the transition, and never activates again. Removing it does not modify surviving brains or their RNG streams.

`Simulation.step(action_intents=...)` remains unchanged as the authoritative manual/external control interface. Simulation itself never chooses actions.

## Information boundary

The neural data path remains:

```text
objective WorldState
        ↓
ExperienceBuilder
        ↓
NoraletExperience
        ↓
NoraletBrain
```

`NoraletBrain.activate()` and `NoraletBrain.act()` accept exactly one `NoraletExperience` argument. The encoder/model import no `WorldState`, body, object-ID or signal-enum types.

The brain has no input containing:

- Noralet routing identity or tick number;
- absolute position, velocity or acceleration;
- exact Energy, condition or age;
- region or Environmental Energy data;
- object/source IDs;
- objective signal type, origin, sender or emission-direction enum;
- simulation events.

`WorldState` remains composed exclusively of CPU Python physical values and contains no tensor, `nn.Module`, CUDA object, hidden state or brain reference. Neural models and hidden tensors remain owned outside the physical simulation.

## No-learning guarantee

Autonomous activation uses `torch.no_grad()` and detaches every newly stored hidden state. The implementation contains:

- no optimizer;
- no `backward()` call;
- no parameter update;
- no learning rate;
- no prediction or value head;
- no reward/homeostatic training path;
- no training/replay buffer;
- no BaseBrain evolution.

All model parameters retain `requires_grad=True` for later work, but `.grad` remains `None` during autonomous inference. Multi-tick tests snapshot every parameter and confirm exact equality afterward. The only changing neural state in Iteration 8 is the per-Noralet hidden vector and observer-only activation count.

## Determinism

Determinism has two separately owned seeds:

```text
base_brain_seed
    → one local CPU prototype initialization

SimulationConfig.master_seed + stable per-ID stream name
    → each Noralet's persistent action-sampling RNG
```

Prototype construction restores global PyTorch RNG state. Cloning consumes no RNG. Neural inference uses deterministic algorithms. Action sampling has a fixed three-draw structure and cannot shift ecology, mortality or another Noralet's action stream.

CPU tests reproduce complete action, hidden-state, world-state, event and death histories for the same configuration and seeds. Reversing initial body insertion order produces the same canonical per-ID history. Observer reads do not shift neural or world execution.

Floating-point results are not claimed to be bitwise identical across CPU and CUDA, different GPU models, different PyTorch/CUDA versions or other backend changes.

## Architecture audit

All sixteen requested invariants were checked:

1. Brain activation receives Experience, never objective world truth.
2. All initial clones begin with exactly equal BaseBrain parameters.
3. Parameter storage and hidden state are independent per individual.
4. Each recurrent update uses current Experience and prior hidden state.
5. Hidden state persists across ticks until death.
6. Parameters remain exactly unchanged during life.
7. External and signal collections use shared MLPs and sum pooling without truncation.
8. Sensory patterns are processed numerically without semantic decoding.
9. Output is limited to continuous acceleration, binary consume and nine signal outcomes.
10. Simulation physically enforces the configured actuator bound.
11. Stochastic choices use explicit isolated streams.
12. One Noralet's draws cannot shift another's stream.
13. Every living brain acts from Experience(t) before one shared transition.
14. A dead Noralet's active brain runtime is removed.
15. Neural tensors/models never enter WorldState.
16. No prediction, value, reward, optimizer or learning system was introduced.

## Files changed

- `pyproject.toml` and `uv.lock` add and lock PyTorch.
- `src/noralet/brain/` adds focused config, encoder, model, prototype, individual runtime and autonomous coordinator modules.
- `src/noralet/noralets/actuators.py` adds the physical acceleration limit.
- `src/noralet/simulation/config.py` and `runtime.py` integrate optional actuator enforcement and routed Experience access.
- `src/noralet/simulation/experience.py` adds the coordinator-only route wrapper.
- public package exports expose the new physical and neural APIs; neural root exports are lazy.
- shared test helpers accept actuator configuration and construct compact neural worlds.
- six focused test modules cover actuator physics, BaseBrain/device behavior, encoding/recurrence, action sampling, coordination/lifecycle and autonomous determinism.
- `codex-reports/operation-report-008.md` records this iteration.

## Tests and validation

The clean pre-Iteration 8 baseline at commit `694619b` passed all 247 Iteration 1–7 tests in `0.720s`.

Iteration 8 adds 59 tests, bringing discovery to 306 tests. The final required command was:

```powershell
uv run python -m unittest discover -s tests -v
```

Result after installing the CUDA build: all 306 tests ran in `2.499s` and passed, including the CUDA-only smoke test. There were no skips, failures or errors.

Additional final validation commands:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
uv run python -m unittest discover -s tests -p "test_autonomous_determinism.py" -k cuda -v
uv run python -m compileall -q src tests
uv lock --check
git diff --check
uv run noralet run --ticks 7 --seed 20260824
uv run python -m unittest discover -s tests -p "test_autonomous_determinism.py" -k finite_autonomous_smoke -v
uv run python -c "... construct CPU BaseBrain, run one autonomous tick, print torch/device/determinism metadata ..."
```

Results:

- `uv sync` replaced `torch==2.13.0` with `torch==2.13.0+cu130` from the explicitly configured official index;
- the CUDA property check reported `2.13.0+cu130`, `True`, `13.0` and `NVIDIA GeForce RTX 3060`;
- the focused Iteration 8 CUDA autonomous smoke ran one test successfully in `5.692s` with no skip;
- source and tests compiled successfully;
- the uv lock resolved 30 packages and is current;
- Git diff validation passed, with only existing LF-to-CRLF conversion notices;
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260824` and imported no neural runtime;
- the focused 20-tick autonomous Energy-conserving smoke passed;
- the direct neural smoke constructed two independent brains, advanced one tick on CPU, and reported deterministic algorithms enabled;
- `torch==2.13.0+cu130`, CUDA available, PyTorch CUDA runtime `13.0`;
- the repository has no configured formatter, linter or static type checker, so no new tooling dependency was added.

The final diff was inspected for direct WorldState access, ID leakage into tensors, global action RNG use, shared clone storage, gradient accumulation, weight updates, premature learning/value/prediction code and unrelated changes.

## Deviations

There are no implementation deviations from the Iteration 8 instruction.

The conditionally defined CUDA smoke test executed successfully on the configured CUDA-enabled development environment.

## Open implementation notes

- The current PyTorch wheel's optional NumPy bridge warning is harmless for the tensor operations used here; NumPy remains deliberately absent because Iteration 8 has no concrete NumPy requirement.
- The BaseBrain interface records sensory vector lengths rather than specific configured pattern values. Semantically different experiment-supplied numerical patterns with the same shapes therefore remain valid neural inputs.
- The prototype remains on CPU even when clones target CUDA. This keeps initialization device-independent while preserving the documented same-device reproducibility scope for execution.

## Git state

No commit or push was created.

The working tree began clean at commit `694619b` (`Add Noralet signal communication`). This iteration modifies the dependency files, seven existing source/export/runtime files and two existing test helpers. It adds the seven focused `src/noralet/brain/` modules, the actuator module, one neural test-support module, six neural/actuator test modules and this report. Architecture and research documentation were read but not modified.
