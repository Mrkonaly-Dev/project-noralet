# Research 001 — Baseline Lifetime Adaptation

**Experiment ID:** `001-baseline-lifetime-adaptation`  
**Primary run:** `20260825T082114.799279Z-cd751b2db3`  
**Git commit:** `7d8e0241c29ed143e915bc5e73066ee3af2df9e2`  
**Status:** completed  
**Role:** first genuine research iteration of Project Noralet

## 1. Research question

The first research iteration asked:

> What does the current Noralet system actually do over a substantial lifetime, and what measurable effects are caused by its existing learning mechanisms?

The experiment was intentionally observational. It did not introduce a planner, new sensory channels, new learning rules, reward shaping or other neural architecture changes.

The current system under study contained recurrent autonomous Noralet brains, lifetime predictive learning, homeostatically modulated action plasticity, eligibility traces, a closed Energy ecology, lethal world boundaries, Noralet-to-Noralet signaling, and local perception/interoception.

## 2. Experimental design

The experiment used a 2 × 2 learning matrix:

| Condition | Predictive learning | Homeostatic action plasticity |
|---|---:|---:|
| `no-learning` | off | off |
| `predictive-only` | on | off |
| `homeostatic-only` | off | on |
| `full-current-brain` | on | on |

Protocol:

- 10 replicate seeds;
- 6 Noralets per run;
- 4 conditions;
- 40 total world runs;
- maximum 5,000 ticks per run;
- timeseries sampling every 10 ticks;
- CUDA execution on an NVIDIA GeForce RTX 3060;
- no respawn or reproduction;
- a run ended at total extinction or the maximum tick.

For each replicate, all four conditions used equivalent initial physical state, BaseBrain initialization and action RNG state. Only learning enablement differed.

The baseline world used lethal boundaries at `[-100, 100]`, no friction or passive braking, and a maximum acceleration magnitude of `0.25`.

## 3. Predefined hypotheses

The harness recorded five hypotheses before execution:

- **H1:** lifetime learning may affect survival and physiological regulation.
- **H2:** predictive learning may improve prediction.
- **H3:** homeostatic plasticity may alter action policy.
- **H4:** initially shared inherited brains may diverge through different lived histories.
- **H5:** signal behavior is descriptive in this pilot; meaningful communication is not presumed.

## 4. Primary quantitative result

All 40 runs completed successfully with zero technical failures.

All 40 runs became extinct.

Across all four conditions:

- total Noralets observed: **240**;
- world-boundary deaths: **240**;
- Energy-depletion deaths: **0**;
- natural deaths: **0**;
- survivors at 1,000 ticks: **0**;
- survivors at 2,500 ticks: **0**;
- survivors at 5,000 ticks: **0**.

The dominant result of Research 001 was therefore not a learning-condition difference.

It was that the current unevolved Noralets usually leave the finite world long before a substantial lifetime-learning experiment can occur.

## 5. Observed lifetimes

| Condition | Mean observed lifetime | Median observed lifetime |
|---|---:|---:|
| `no-learning` | 65.67 ticks | 46.5 ticks |
| `predictive-only` | 63.20 ticks | 44.0 ticks |
| `full-current-brain` | 61.17 ticks | 43.0 ticks |
| `homeostatic-only` | 71.63 ticks | 47.5 ticks |

These short observed lifetimes are not evidence that one learning condition is superior.

The experiment was designed around a possible 5,000-tick lifetime, while the typical observed lifetime was only a few dozen ticks. This creates a major timescale mismatch for evaluating lifetime adaptation.

## 6. Dominant failure mode

The current motion model has persistent velocity, no drag, no passive braking, lethal finite boundaries, and stochastic acceleration actions.

A small sustained directional bias therefore accumulates velocity and can move a Noralet rapidly toward a lethal edge.

Research 001 strongly suggests that random inherited BaseBrain initialization can produce substantial initial motor bias.

Because the BaseBrain seed also varied between replicates, the pilot sampled both different world histories and different random inherited neural initializations.

Within a replicate, condition comparisons remained fair because all four conditions shared the same inherited initialization. However, between-replicate variance includes inherited-brain variance.

This became an important confound for interpreting long-term learning.

## 7. Predictive learning observations

Predictive learning was active and produced substantial parameter change even during short lives.

Condition-level mean final parameter drift from inherited birth parameters:

| Condition | Encoder drift | GRU drift | Predictor drift |
|---|---:|---:|---:|
| `predictive-only` | 1.4763 | 1.5618 | 1.1856 |
| `full-current-brain` | 1.4872 | 1.5737 | 1.1834 |

Aggregate prediction loss changed from:

- `predictive-only`: `0.02266 → 0.02089`;
- `full-current-brain`: `0.02290 → 0.02163`.

However, most Noralets died too early for the configured first-100-update and last-100-update windows to become cleanly separated.

Therefore Research 001 does **not** provide a strong test of long-timescale predictive improvement.

The valid conclusion is narrower:

> Predictive lifetime plasticity was active and changed the neural system during life, but the pilot was too short-lived to evaluate its mature effect.

## 8. Homeostatic plasticity observations

Action-head drift was non-zero only in conditions with homeostatic plasticity:

- `full-current-brain`: mean drift `0.04091`;
- `homeostatic-only`: mean drift `0.03902`.

Mean absolute homeostatic modulation was approximately `0.00363` in `full-current-brain` and `0.00357` in `homeostatic-only`.

Mean homeostatic update norm was approximately `0.00128` in `full-current-brain` and `0.00115` in `homeostatic-only`.

Inspection of the raw run data showed an important qualitative pattern:

- ordinary expenditure ticks usually produced small worsening of bodily distress and therefore negative modulation;
- successful Energy ingestion produced the rare positive modulation events.

This matches the intended operational meaning of the homeostatic mechanism.

It does not yet establish that the resulting action plasticity improves ecological success.

## 9. Consumption and signaling

Across all conditions, successful Energy transfer occurred occasionally before death:

- `no-learning`: 33 successful consumptions;
- `predictive-only`: 38;
- `full-current-brain`: 37;
- `homeostatic-only`: 33.

Thus some Noralets did interact successfully with Consumable Energy even in their short lives.

Signal emission was extremely frequent.

The current signal action space contains one `NONE` option and eight directional signal options: A/B/C/D × left/right.

A nearly unstructured categorical policy therefore has a strong combinatorial prior toward emitting a signal rather than remaining silent.

Research 001 makes this action-space prior visible, but no claim about signal meaning is justified.

## 10. Main interpretation

Research 001 did not fail.

It found a more fundamental limitation than the one it was originally designed to measure.

The present system appears to have functional predictive plasticity, functional homeostatic action plasticity, functioning Energy interaction, and substantial inherited neural variation, but the random inherited BaseBrain often produces unstable motor behavior that ends life before long-term learning can be meaningfully studied.

This led to the hypothesis that Project Noralet is missing an evolutionary prehistory.

A biological newborn does not begin from arbitrary random motor circuitry. Its starting neural organization is already the product of prior evolutionary selection.

The current random BaseBrain is therefore better interpreted as a **pre-evolutionary neural population** than as a realistic primitive newborn brain.

## 11. Resulting direction

The next planned step is an external **BaseBrain evolutionary bootstrap**.

The intended separation is:

```text
EVOLUTIONARY TIMESCALE

many inherited BaseBrain variants
→ real Noralet lives
→ differential viability
→ selection
→ mutation
→ next inherited generation
```

while each individual life remains:

```text
INDIVIDUAL TIMESCALE

inherited BaseBrain
→ lifetime predictive learning
→ homeostatic action plasticity
→ death
```

Death must not become a posthumous negative neural signal.

A dead Noralet does not respawn and continue learning.

Adult learned weights should not automatically be copied into descendants.

Until true in-world reproduction exists, observed lifetime may be used only as an external **viability proxy for evolutionary bootstrap**, not as a claim of biological reproductive fitness.

## 12. Renderer follow-up

After Research 001, Renderer / Observer UI v1 was implemented as a separately planned engineering component.

The renderer does not alter the scientific result and is not itself a research-derived architecture change.

It enabled qualitative observation of individual trajectories that would be difficult to notice in aggregate CSV results.

The following observations are exploratory notes, not controlled evidence.

### 12.1 Seed `696969` — apparent group-following pattern

In a live renderer run using seed `696969`, several Noralets moved toward and crossed a lethal boundary. One remaining Noralet survived longer and later moved in the same general direction before also dying at the boundary.

Visually this resembled following behavior.

This must **not** currently be interpreted as social following.

Possible explanations include shared inherited motor bias, similar sensory conditions, recurrent-network trajectory dynamics, influence from other-Noralet perception, or influence from signal perception.

A controlled ablation would be required to distinguish these explanations.

### 12.2 Seed `94476`, population 5 — Noralet 3

A particularly notable live run used:

- simulation seed: `94476`;
- population: `5`;
- learning mode: full current brain.

Noralet 3 survived for **566 ticks**, much longer than the typical Research 001 lifetime.

Observed qualitatively:

1. it showed periods of slowing or near-stopping around the central area;
2. it moved back and forth rather than maintaining one simple directional trajectory;
3. it acquired Consumable Energy during life;
4. it later encountered Energy that was no longer usefully/fully taken;
5. it approached the right lethal boundary and appeared to stop or nearly stop before crossing it;
6. it subsequently travelled left;
7. it eventually died by crossing the **left** lethal boundary at tick 566.

The boundary-near slowdown is especially notable because boundary perception is available to the Noralet while absolute position is not.

However, this single observation does **not** establish intentional boundary avoidance, planning, fear, goal-directed foraging or understanding of death.

A random recurrent network can produce complex context-dependent trajectories.

The scientifically useful observation is only:

> The current neural architecture can already produce, from some inherited random initializations and lived histories, substantially richer motion than persistent one-direction acceleration.

This is encouraging for evolutionary bootstrap because it suggests that useful behavioral variation already exists before selection.

## 13. Behavioral probe seeds

Seeds such as `94476` may be retained as **qualitative behavioral probe seeds**.

They must not be used as the sole evolutionary fitness environment and must not be cherry-picked for selection.

Their role may instead be visual comparison of prototype versus evolved BaseBrains, qualitative regression, and hypothesis generation.

Quantitative evolutionary selection must still use multiple predefined training worlds and separate validation worlds.

## 14. Open research questions after Research 001

1. Can evolutionary selection produce inherited motor priors that reduce lethal boundary exits?
2. Can evolved BaseBrains survive long enough for lifetime predictive and homeostatic learning to become meaningfully testable?
3. Does Energy acquisition become more reliable under evolutionary selection?
4. Does excessive signal emission decrease when unnecessary signaling has an Energy cost?
5. Does boundary-near slowing become a reproducible percept-dependent behavior rather than a rare random trajectory?
6. How much behavior is inherited versus acquired during lifetime learning?
7. Does training-world improvement generalize to unseen validation seeds?
8. Do apparently social trajectories survive controlled signal/perception ablations?

These are questions, not conclusions.

## 15. Current conclusion

The strongest conclusion of Research 001 is:

> Project Noralet currently has enough neural and behavioral variability for learning mechanisms to operate, but random inherited neural initialization frequently terminates life before long-term adaptation can be evaluated.

The resulting next hypothesis is that a realistic primitive starting brain should not be arbitrary random weights.

An evolutionary bootstrap of inherited BaseBrain parameters is therefore the next research/engineering direction to test.

No claim about intelligence, intention, subjective experience or consciousness follows from Research 001.
