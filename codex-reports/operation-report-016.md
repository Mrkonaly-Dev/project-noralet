# Operation Report 016 — Neutral BaseBrain Initialization

Date: 2026-08-25

## Motivation

Implemented a versioned neutral initialization scheme for genuinely new
BaseBrains. The change distinguishes minimal inherited neural/physiological
priors from accidental actuator bias: a newborn still has no knowledge of
boundaries, Energy, food, the world centre, other Noralets, signal meaning, or
survival strategy, but it no longer begins from arbitrary one-sided motor output
or action-space geometry that makes optional actuators active by default.

The initialization version is:

```text
002-neutral-actuator-baselines
```

Its compact recorded configuration is:

```text
acceleration_output_weight_scale: 0.01
initial_consume_probability:      0.05
initial_signal_probability:       0.05
```

## Architecture

The existing architecture remains unchanged:

```text
NoraletExperience
→ ExperienceEncoder
→ one GRU recurrent core
→ acceleration / consume / signal action heads
→ PredictionModel
```

No neural module, hidden-state hierarchy, sensory channel, actuator, intrinsic
reward, reflex, or world-state bypass was added. The encoder and predictor retain
seeded random initialization, so inherited diversity remains. The recurrent
hidden state remains exactly zero at birth.

The GRU now uses deterministic gate-wise Xavier input weights, gate-wise
orthogonal recurrent weights, and zero biases. This is a standard stable
tanh/sigmoid recurrent initialization; it does not install attractors,
oscillations, memories, environmental representations, or preferred hidden
states.

## Acceleration

The final acceleration-head bias initializes to exactly zero. Its weights use a
named small symmetric uniform range:

```text
[-acceleration_output_weight_scale, +acceleration_output_weight_scale]
```

The weights are deterministically derived from the BaseBrain seed without using
or advancing global PyTorch RNG state. Existing action sampling, tanh bounding,
actuator limits, and acceleration exploration standard deviation are unchanged.
Acceleration is not forced or clamped to zero during execution, and learning or
evolution may immediately move the policy.

## Consume

The consume output bias is initialized to the logit of the configured target:

```text
logit(0.05)
```

This produces a mostly inactive content-free baseline using the output bias
only. No Energy, food percept, distance, distress, or world state is inspected.
The consume weights remain random, and lifetime plasticity and evolution remain
free to change both weight and bias.

## Signals

The existing nine categorical actions and their sampling semantics remain
unchanged. The signal biases now encode:

```text
P(NONE)       = 0.95
P(each emit)  = 0.05 / 8
P(any signal) = 0.05
```

This removes the old uniform-nine-way combinatorial prior of `8/9` signal
emission without adding a runtime special case or assigning meaning to any
signal. A/B/C/D and left/right begin symmetric and meaningless. Signal-head
weights remain random, and learning and mutation can move all logits.

## Compatibility

Freshly constructed random BaseBrains use the neutral initializer. Future v1 and
v2 manifests, evolution states, and champion checkpoints record the initializer
version/configuration and whether explicit genome parameters override it.

Historical checkpoints without initialization provenance are identified as
explicit historical genomes with legacy unrecorded initialization. Loading a
saved genome still creates the current container and then applies every explicit
saved tensor exactly; it does not neutralize, reinterpret, or mutate that genome.

Focused tests created historical-format v1 and v2 champions, loaded and applied
their explicit parameters exactly, and verified unchanged file SHA-256 values.
The project's existing real v1 and v2 state/champion files also loaded through
the production loaders and retained these hashes before and after reading:

```text
v1 evolution-state.pt
AE5FD8FC2A107020E5406AD6B4F984F8AB024E49950B1CA873EAEBFDA852F95A

v1 champion/best.pt
51FF224052502230BE15BC1D4C94DD36137BB47E2AF211EAB1BB38DA3134F801

v2 evolution-state.pt
9054D18C214230F367815A6A65A60C0C48FD154CAA2350B087C76AAD9AB52467

v2 champion/best.pt
03DD8AC5E67E00E30456FB3AF18394398989FF9AA5C8C746B87689A85131173B
```

The loaded v1 state contained eight genomes at next generation 15 and its saved
champion contained 34 inherited tensors. The loaded v2 state contained four
genomes at next generation 3 and its saved champion also contained 34 inherited
tensors. No existing evolved champion or population file was written.

## Initialization audit

Command:

```text
uv run noralet research basebrain-initialization-audit \
  --samples 100 --seed 1 --device cpu
```

The audit constructs 100 fresh BaseBrains with deterministic distinct seeds and
activates each once on the same fixed synthetic neutral `NoraletExperience`. It
does not construct or run a world, sample survival, measure fitness, learn, or
evolve. It reads the normal pre-sampling action-distribution outputs, avoiding
finite action-sampling noise while auditing the actual initialized policy path.

Measured aggregate output:

```text
sample count:                       100
acceleration mean:                 -0.0000995635296931141
acceleration standard deviation:    0.0019580498978744093
fraction acceleration < 0:          0.58
fraction acceleration > 0:          0.42
consume activation probability:     0.05042320374138127
signal emission probability:        0.05040441689595743
signal NONE probability:            0.9495955831040426
```

Conditional probabilities given emission:

```text
A_LEFT   0.1251891371202745
A_RIGHT  0.12488840112302352
B_LEFT   0.12630315664105146
B_RIGHT  0.1239742080980618
C_LEFT   0.1251357885041892
C_RIGHT  0.12468576200571292
D_LEFT   0.12377858811643487
D_RIGHT  0.1260449583912518
```

These results establish neutral newborn actuator baselines only. They do not
establish or imply improved survival or generalization.

## Tests

Focused Iteration 16 coverage directly checks configuration/version validation,
exact action-head biases, symmetric multi-seed acceleration initialization,
stable GRU initialization, same-seed determinism, different-seed diversity,
the observer-only audit, historical v1/v2 loading, future provenance, lifetime
action plasticity, and evolutionary mutation:

```text
uv run python -m unittest discover \
  -s tests -p 'test_neutral_initialization.py' -v

Ran 10 tests in 2.646s
OK
```

Complete regression:

```text
uv run python -m unittest discover -s tests -v

Ran 474 tests in 50.045s
OK
```

Repository gates:

```text
uv run python -m compileall -q src tests
exit 0

uv lock --check
Resolved 34 packages in 1ms

git diff --check
exit 0
```

`git diff --check` emitted only the repository's existing LF→CRLF working-copy
notices and reported no whitespace error.

No long evolution, Research 001/002 rerun, survival experiment, or long world
simulation was performed.

## Deviations

- The audit reports the action distributions' deterministic probabilities and
  acceleration locations after normal activation rather than drawing one noisy
  action per brain. It still uses the production brain path and provides the
  requested population aggregates without conflating initialization with action
  RNG variance.
- The optional analytical old-versus-new helper was not added. The historical
  uniform categorical prior is already exactly `8/9`, while saved checkpoints
  preserve historical parameters without maintaining a second production
  initializer.
- PyTorch emits the pre-existing optional NumPy initialization warning because
  NumPy is not a project dependency. It does not affect the audit or tests.

## Git state

Implementation began from commit:

```text
8969e462cbe4a568a176c12f688fb9649f5a40f6
```

No commit or push was performed.
