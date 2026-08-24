# Operation Report 010 — Homeostatic Neuromodulation and Action Plasticity

**Iteration:** 10
**Date:** 2026-08-24
**Status:** Complete

## Summary

Iteration 10 adds optional individual homeostatic action plasticity beside the
existing predictive lifetime-learning system. Each enabled Noralet derives a
negative homeostatic drive from its own existing interoceptive distress,
records detached selected-action likelihood gradients in three persistent
eligibility traces, and uses the actual change in drive after a lived world
transition to modify only its acceleration, consume and signal action heads.

The implemented lifecycle is:

```text
Experience(t)
    → encoder + GRU → h(t)
    → sample the existing three stochastic actions
    → selected-action likelihood gradients at detached h(t)
    → decay and extend per-head eligibility
    → optional one-step prediction
    → one shared world transition
    → Experience(t+1) for survivors
    → existing predictive Adam update
    → bounded homeostatic modulation
    → direct clipped action-head update
```

There is no external reward, semantic action value, terminal penalty, planner
or candidate-action evaluation. When homeostatic plasticity is disabled, no
eligibility tensors or homeostatic observer results exist and the complete
Iteration 9 behaviour is preserved.

## Biological abstraction

The implementation uses the following functional abstraction:

```text
recent selected action pathway
        → decaying eligibility trace

later change in negative bodily distress
        → global homeostatic modulation

eligibility × modulation
        → gradual action-head plasticity
```

This is an engineering abstraction of neuromodulated / three-factor synaptic
plasticity. It is not a literal molecular neural model and makes no claim about
subjective experience.

## Homeostatic drive

`NoraletHomeostaticPlasticityConfig` contains two finite non-negative weights:

```text
energy_distress_weight
condition_distress_weight
```

At least one must be strictly positive. For `Interoception` supplied through
the current `NoraletExperience`, runtime computes:

```text
D(t) = (
    w_energy * energy_distress(t)
    + w_condition * condition_distress(t)
) / (w_energy + w_condition)
```

The two sensory inputs are already validated in `[0, 1]`, so `D` is also in
`[0, 1]`. `energetic_exertion` deliberately does not enter this formula.

The drive function accepts only brain-facing `Interoception` plus the immutable
plasticity configuration. It has no body, WorldState, exact Energy, condition,
age, region, event or mortality input. Exact objective physiology is excluded
because the only permitted path is:

```text
WorldState → ExperienceBuilder → NoraletExperience → Interoception → D
```

`D` is private transition/runtime context and is not added to Experience as a
new sensory channel.

## Modulatory signal

For one surviving lived transition:

```text
raw_homeostatic_improvement = D(t) - D(t+1)

modulation = tanh(
    raw_homeostatic_improvement
    / homeostatic_modulation_scale
)
```

`homeostatic_modulation_scale` must be positive and finite. Lower later drive
therefore creates positive modulation, higher later drive creates negative
modulation, and unchanged drive produces exactly zero. At floating-point tanh
saturation, the result is clamped only to the adjacent representable value
inside `(-1, +1)` so the documented open bound remains true.

The inherited organism-level polarity is limited to this rule: lower negative
distress is physiologically favourable and higher negative distress is
physiologically unfavourable. No meaning is attached to any external object,
boundary, motor command, signal category or other Noralet.

## Eligibility traces

Every enabled spawned brain creates zero tensors corresponding exactly to the
weight and bias tensors of:

```text
acceleration_head
consume_head
signal_head
```

The public observer snapshot retains these three named groups rather than
flattening them into one opaque tensor. Runtime traces are individual,
detached, finite, device-matched neural state and are not optimizer state.

For each selected action the relevant increment is:

```text
G(t) = gradient_theta log pi_theta(a_t | h_t detached)

E(t) = eligibility_decay * E(t-1) + G(t)
```

`eligibility_decay` is validated in `[0, 1)`. Traces persist after every
modulatory update and are extended again at the next action. They are not
cleared on neutral modulation. This provides compact delayed credit without a
fixed history buffer or a semantic causal label. Death destroys the pending
drive and eligibility ownership together with the removed brain.

## Action-head learning

### Acceleration

The existing first action draw still produces standard-normal `z`, followed by:

```text
raw_motor = acceleration_loc + acceleration_exploration_std * z
normalized_acceleration = tanh(raw_motor)
```

Homeostatic plasticity requires `acceleration_exploration_std > 0`. Runtime
retains the actual selected pre-tanh `raw_motor` value. When evaluating its
`Normal(acceleration_loc, acceleration_exploration_std)` log likelihood, the
selected sample is reconstructed as a detached tensor. The exploration
standard deviation remains configuration supplied and is not trained. This
avoids a reparameterized gradient path that would cancel the location score.

### Consume

The actually selected `0` or `1` is detached and evaluated with the negative
numerically stable binary cross-entropy-with-logits expression. Its gradient is
taken only with respect to the consume-head weight and bias.

### Signal

The actually selected one of the existing nine categories indexes a stable
`log_softmax` over the nine signal logits. Its gradient is taken only with
respect to the signal-head weight and bias. Signal categories retain no
semantic learning label.

### Detached recurrent context

All three likelihoods are recomputed from:

```text
h_action = h(t).detach()
```

`torch.autograd.grad` produces detached increments and does not populate any
parameter `.grad` field or retain a graph. Consequently this learning path has
no gradient route into the encoder or GRU. It also cannot touch the predictor
or frozen target encoder.

## Plasticity rule

After `Experience(t+1)` exists, runtime computes the global direction over all
named action-head traces:

```text
q_i = modulation * E_i
q_norm = combined_global_norm(q_i)

clip_factor = min(
    1,
    max_homeostatic_update_norm / q_norm
)

delta_theta_i =
    action_learning_rate * clip_factor * q_i

theta_i = theta_i + delta_theta_i
```

The zero-norm case uses a clip factor of one. Both the action learning rate and
maximum direction norm are positive finite values. The observer metric
`applied_update_norm` is the norm of the actual parameter delta after clipping
and multiplication by the separate action learning rate.

Before applying the update, runtime verifies finite modulation, eligibility,
direction norm, parameter state, update tensors and proposed parameter values.
It copies all already-validated proposed values under `torch.no_grad()` and
then verifies the resulting action heads again. NaNs are never silently zeroed.

Positive modulation increases the likelihood density/probability of recently
eligible selected actions. Negative modulation decreases it. No mask or hard
blocking mechanism exists, so all stochastic actions remain physically
possible.

## Learning-system separation

The parameter ownership is explicit and disjoint:

```text
predictive plasticity (existing individual Adam):
    online ExperienceEncoder
    GRU recurrent core
    PredictionModel

homeostatic plasticity (direct three-factor update):
    acceleration action head
    consume action head
    signal action head

permanently unchanged by both:
    frozen target ExperienceEncoder
    BaseBrain prototype
```

When both systems are enabled, action-head parameters require gradients only so
the selected-action likelihood increments can be calculated. They remain
excluded from predictive Adam. Predictive loss has no computation path through
the sampled action heads. Conversely, the homeostatic path receives detached
recurrent state and cannot optimize predictive components.

## Individual development

Each spawned Noralet owns independent online parameters, optimizer, target
encoder, hidden state and eligibility tensors. The BaseBrain contains only the
shared inherited initial action heads; it has no runtime traces and never
changes.

A controlled two-clone test presented identical initial brains with the same
current Experience and same selected-action draws, followed by opposite bodily
changes. One clone received positive modulation and the other negative
modulation. Their action-head parameters diverged while the prototype remained
exact. No result or tensor from one individual entered the other.

## Diagnostics

### Favourable selected-action adaptation

A deterministic 300-transition two-state experiment began with a neutral
consume-head weight and bias. The current negative distress was the prior
transition's actual outcome. Selecting the stochastic binary action caused the
next distress to be `0.2`; not selecting it caused the next distress to be
`0.8`. The action head received no semantic explanation of that relationship.

Measured probability at the fixed zero-hidden evaluation context:

```text
initial: 0.500000000
final:   0.527556002
```

### Unfavourable selected-action adaptation

The mirror experiment made selection produce next distress `0.8` and
non-selection produce next distress `0.2`:

```text
initial: 0.500000000
final:   0.473085046
```

These results demonstrate learned behavioural bias under internally derived
homeostatic modulation. They are not claims of intelligence or subjective
state.

### Delayed credit

A controlled acceleration sequence used a non-zero selected Normal deviation
at `t0`, then zero-deviation acceleration selections at the next two actions.
The first two bodily transitions were neutral; improvement arrived only after
the third action.

```text
eligibility_decay = 0.5
    residual acceleration eligibility norm: 1.042148829
    acceleration parameters changed:        yes
    modulation:                              0.964027580
    total applied update norm:               0.111268282

eligibility_decay = 0.0
    residual acceleration eligibility norm: 0.000000000
    acceleration parameters changed:        no
    modulation:                              0.964027580
    total applied update norm:               0.058052648
```

The zero-decay run still updated the consume/signal pathways eligible at the
current tick, hence its non-zero total update norm. The acceleration comparison
isolates the earlier action's delayed eligibility.

## Death semantics

The current drive and selected-action eligibility are created before the shared
world transition. A Noralet that dies during that transition has no next
Experience, so the coordinator performs neither a drive-after calculation nor
a modulatory update and emits no homeostatic observer record. It destroys the
brain's pending context and eligibility ownership before removing the brain.

Boundary, Energy-depletion and controlled natural-death tests cover this rule.
There is no synthetic terminal distress, terminal reward or death penalty. A
Noralet cannot learn directly from its own completed death.

## Information boundary

All causal homeostatic inputs originate in `NoraletExperience.interoception`.
The runtime module imports `NoraletExperience`, not WorldState or body state.
Pending homeostatic context contains exactly one field: the scalar current
drive derived from current Experience. It contains no exact Energy, condition,
region, execution result, event, death probability or routing identity.

Observer values (`modulation`, drive metrics, eligibility norm, update norm and
predictive metrics) are immutable outputs and never re-enter Experience or any
brain activation.

## No external reward

No reward, score, fitness, survival incentive, action-quality label or death
penalty variable/API was added. Environmental consumption events and other
objective outcomes never become learning labels.

The selected-action eligibility mathematics is explicitly related to the
score-function form used by stochastic policy-gradient methods:

```text
gradient log pi(selected action | detached neural context)
```

The third factor is not externally assigned reward. It is the continuous
bounded change in the organism's own existing negative interoceptive distress.
No value function, actor-critic system or Q-learning mechanism exists.

## Determinism

Homeostatic plasticity makes no random draw. Action selection still performs
exactly three unconditional reads, in the established order, from each
Noralet's existing isolated Python RNG stream:

```text
1. acceleration standard-normal source
2. consume Bernoulli source
3. signal categorical source
```

The selected values are reused for eligibility calculation. No global Python,
global PyTorch or CUDA RNG is consumed by learning. An enabled-vs-disabled
first-transition comparison produced identical actions, physical TickResult
and all simulation/action stream states.

Two complete CPU runs with identical world, seeds and neural/plasticity
configuration reproduced exact actions, predictive metrics, homeostatic
metrics, parameters, eligibility traces, hidden states, world history and
deaths. No CPU-versus-CUDA or cross-hardware bitwise guarantee is claimed.

## CUDA validation

The real development environment reported:

```text
PyTorch:                  2.13.0+cu130
torch.cuda.is_available:  True
PyTorch CUDA runtime:     13.0
GPU:                      NVIDIA GeForce RTX 3060
```

The focused three-tick CUDA smoke executed action sampling, detached
selected-action likelihood gradients, persistent CUDA eligibility tensors, the
world transition, finite homeostatic modulation, direct action-head updates,
predictive forward/backward/Adam updates and subsequent actions. Action heads
changed, the target encoder remained exact, all inspected tensors were finite,
all eligibility tensors stayed on CUDA, and total world Energy remained within
the established exact-conservation tolerance. The focused test passed in
`1.703s` without a skip or device mismatch.

## Tests and validation

The clean Iteration 9 baseline at commit `cc2ac15` passed all 354 existing tests
in `4.817s` before implementation.

Iteration 10 adds 39 focused tests across four modules, bringing discovery to
393 tests. The required final full-suite command was:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: all 393 tests passed in `6.623s`, including the real CUDA tests. There
were no failures, errors or skips in the configured CUDA environment.

Additional validation commands:

```powershell
uv run python -m compileall -q src tests
uv lock --check
git diff --check
uv run noralet run --ticks 7 --seed 20260824
uv run python -m unittest discover -s tests -p "test_homeostatic_diagnostics.py" -v
uv run python -m unittest discover -s tests -p "test_homeostatic_action_plasticity.py" -k delayed -v
uv run python -m unittest discover -s tests -p "test_homeostatic_integration.py" -k disabled_mode -v
uv run python -m unittest discover -s tests -p "test_homeostatic_integration.py" -k mixed_autonomous_smoke -v
uv run python -m unittest discover -s tests -p "test_homeostatic_integration.py" -k CudaHomeostaticLearningTests -v
uv run python -c "import torch; ..."
```

Results:

- source and tests compiled successfully;
- `uv.lock` resolved 30 packages and is current;
- Git whitespace validation passed (only repository LF-to-CRLF notices);
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260824`;
- both 300-transition positive/negative adaptation diagnostics passed in
  `2.493s`;
- the delayed-credit diagnostic passed in `1.148s`;
- the focused Iteration 9-disabled compatibility smoke passed in `1.428s`;
- the ten-tick mixed autonomous predictive + homeostatic smoke passed in
  `1.318s` with finite learning state and conserved Energy;
- the focused CUDA homeostatic/predictive test passed in `1.703s`.

The final architecture audit verified:

1. modulation uses only Noralet-facing distress sensations;
2. only negative-distress polarity is inherited, not world meaning;
3. action change requires recent eligibility plus later modulation;
4. probabilities change gradually and no action is blocked;
5. traces provide delayed credit;
6. predictive and action plasticity own disjoint parameter sets;
7. no semantic action value enters learning;
8. objective physiology cannot bypass Experience;
9. an outcome affects only later actions;
10. death produces no posthumous update;
11. plasticity remains individual;
12. no planner or counterfactual action search exists;
13. the Iteration 9 predictor remains intact;
14. the implementation makes no subjective-state claim.

## Files changed

- `src/noralet/brain/learning.py` adds immutable configuration, drive and
  modulation functions, named detached eligibility values and internal result.
- `src/noralet/brain/runtime.py` adds per-brain trace lifecycle, selected-action
  likelihood gradients and the direct clipped update.
- `src/noralet/brain/base.py` adds optional homeostatic configuration without
  changing inherited module architecture or initialization.
- `src/noralet/brain/coordinator.py` integrates survivor-only updates and
  separate immutable observer metrics.
- neural public exports expose the new operational APIs.
- the shared neural test helper accepts the optional configuration.
- four focused test modules cover formulas, traces, polarity, separation,
  adaptation, delayed credit, integration, death, determinism and CUDA.
- this report records Iteration 10.

No physical simulation, Experience construction, renderer, predictive neural
architecture, PyTorch dependency or unrelated implementation was changed.

## Deviations

There are no implementation deviations from the Iteration 10 instruction.

## Open implementation notes

- The controlled delayed-credit experiment shows that a decay of `0.5` retains
  measurable three-tick acceleration eligibility whereas zero decay does not.
  Whether the suitable timescale is longer or shorter in the normal ecology is
  an experimental question; no adaptive decay was introduced.
- The configured modulation and action learning rate produced small measurable
  probability changes in the controlled 300-transition diagnostic. Their
  strength in open autonomous ecology is not yet calibrated and should not be
  inferred from this capability test.
- The existing one-step predictor could later provide learned anticipatory
  action evaluation, but Iteration 10 does not use it for candidate actions,
  value estimation, planning or counterfactual rollout.

## Git state

No commit or push was created.

The working tree began clean at commit `cc2ac15` (`Add predictive lifetime
learning`). Iteration 10 modifies only the neural lifetime-learning runtime,
coordinator, public exports and shared neural test helper; it adds four focused
test modules and this operation report.
