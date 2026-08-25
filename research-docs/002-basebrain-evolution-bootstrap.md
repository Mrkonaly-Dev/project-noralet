# Research 002 — BaseBrain Evolution Bootstrap

**Evolution ID:** `001-basebrain-bootstrap`  
**Primary lineage:** `20260825T121340.095069Z-8fbe71a650`  
**Status:** completed through generation 14 / 15 total generations  
**Role:** first evolutionary bootstrap study of inherited Noralet brain parameters

## 1. Research question

Research 001 showed that random inherited BaseBrain initialization frequently
produced unstable motor behavior and lethal boundary crossings before long-term
lifetime learning could be meaningfully studied.

Research 002 asked:

> Can external evolutionary selection improve the inherited BaseBrain
> initialization while preserving genuine lifetime learning inside each
> individual life?

The evolutionary system was intentionally separated from the Noralet's own
experience.

A Noralet did not observe fitness, generation, candidate identity, parentage,
ranking or validation performance.

Death remained terminal for the individual.

There was no death reward, terminal punishment, respawn, posthumous update or
inheritance of adult learned weights.

## 2. Evolutionary bootstrap model

Two timescales were used.

### Individual lifetime

```text
inherited BaseBrain
→ fresh Noralet
→ predictive lifetime learning
→ homeostatic action plasticity
→ lived world history
→ death
```

Adult learned parameters were discarded after evaluation.

### Evolutionary timescale

```text
population of inherited BaseBrains
→ multiple fresh lives
→ external viability measurement
→ selection
→ inherited parameter mutation
→ next generation
```

The evolved genome consisted of the inherited BaseBrain prototype parameters:
ExperienceEncoder, GRU, PredictionModel initialization, acceleration head,
consume head and signal head.

Neural topology and learning hyperparameters were not evolved.

## 3. Fitness

The bootstrap used one deliberately simple selection quantity:

> **fitness = mean observed lifetime ticks**

This was treated only as an external **viability proxy for evolutionary
bootstrap**.

No additional fitness reward was assigned for consuming Energy, avoiding
boundaries, moving, signaling, exploration or prediction quality.

## 4. Evolution environment

The evolution environment reused the existing world and neural mechanics.

The main deliberate difference from Research 001 was lower birth Energy:

- Research 001: `60 eU`;
- evolution bootstrap: `10 eU`.

World boundaries remained `[-100, 100]`, lethal, and the motion system retained
persistent velocity with no passive friction.

## 5. Pilot and continuation protocol

The first pilot used:

- population 8;
- 2 elites;
- parent pool 4;
- 2 fixed training world seeds;
- 2 separate fixed validation world seeds;
- 4 Noralets per world;
- 1,000-tick cap;
- mutation sigma `0.02`;
- birth Energy `10 eU`;
- full current-brain lifetime learning;
- CUDA execution.

The run completed 5 generations and was then deterministically resumed from its
saved `evolution-state.pt` checkpoint to a total of **15 generations**, without
changing the scientific configuration.

## 6. Fitness progression

| Generation | Best train | Mean train | Median train | Validation |
|---:|---:|---:|---:|---:|
| 0 | 63.750 | 43.469 | 41.625 | 81.000 |
| 1 | 122.000 | 60.469 | 48.188 | 105.000 |
| 2 | 144.375 | 82.891 | 75.313 | 129.375 |
| 3 | 160.375 | 101.797 | 98.938 | 129.125 |
| 4 | 160.375 | 113.469 | 119.250 | 129.125 |
| 5 | 167.875 | 126.203 | 129.875 | **173.500** |
| 6 | 167.875 | 140.719 | 142.438 | **173.500** |
| 7 | 207.875 | 145.750 | 156.750 | 149.375 |
| 8 | 207.875 | 143.719 | 141.813 | 149.375 |
| 9 | 207.875 | 131.906 | 114.500 | 149.375 |
| 10 | 213.875 | 149.844 | 151.125 | 113.750 |
| 11 | **222.000** | **163.672** | **175.625** | 125.500 |
| 12 | 222.000 | 151.125 | 160.125 | 125.500 |
| 13 | 222.000 | 155.563 | 145.563 | 125.500 |
| 14 | 222.000 | 144.438 | 147.125 | 125.500 |

Training viability improved strongly from the random Generation 0 population.

Between Generation 0 and Generation 14:

- best training fitness increased from `63.75` to `222`;
- mean training fitness increased from `43.47` to `144.44`;
- median training fitness increased from `41.63` to `147.13`.

The population therefore did not merely produce one isolated high-fitness
outlier. The distribution shifted toward longer survival.

## 7. Validation divergence

Validation initially improved together with training performance:

```text
Gen 0:   81.0
Gen 1:  105.0
Gen 2:  129.375
Gen 5:  173.5
```

After Generation 5, training and validation diverged:

```text
Gen 7:
training best 207.875
validation    149.375

Gen 10:
training best 213.875
validation    113.75

Gen 11:
training best 222.0
validation    125.5
```

This creates an important research hypothesis:

> Later generations may be specializing to the two fixed training world seeds
> rather than continuing to improve general viability.

This is **not yet established** as overfitting.

Only two validation worlds were used, so validation itself is noisy.

A larger unseen-world audit is needed before deciding whether later generations
genuinely generalize worse.

## 8. Death-mode development

Boundary death remained the dominant failure mode throughout evolution.

Generation 0 had a boundary-death fraction of `1.0`.

Later generations occasionally produced Energy-depletion deaths, showing that
some lives survived long enough for Energy maintenance to become limiting.

However, even in later generations the boundary-death fraction remained roughly
`0.94–1.0`.

The current evidence therefore supports only:

```text
boundary survival improved
→ some lives reach later physiological failure modes
→ lethal boundaries remain dominant
```

not:

```text
boundary problem solved
→ Energy becomes the main bottleneck
```

## 9. Qualitative champion observation

The Generation 3 champion (`g003-c005`) was watched in a fresh renderer life.

This was not a replay of an evolutionary evaluation episode.

The watched group showed visibly richer trajectories than the earliest random
BaseBrains:

- individuals did not all immediately accelerate in one common direction;
- movement histories diverged;
- direction changes were common;
- individuals survived for different durations;
- the population still eventually became extinct through boundary crossings.

This qualitative observation is consistent with the quantitative shift toward
longer survival.

It does not establish deliberate boundary avoidance, planning or foraging.

## 10. Historical checkpoints

The lineage preserved inherited snapshots including Generation 0, Generation 4,
Generation 5, Generation 10, Generation 14 and the current best-so-far genome.

This matters because the genome with the highest training fitness is not
necessarily the genome with the best general viability.

Generation 5 reached validation fitness `173.5`, while the later
training-fitness champion reached `125.5` on the two validation worlds.

## 11. Methodological concern: current evolution may be inefficient

The bootstrap successfully demonstrated that inherited neural parameters are
selectable.

However, the current training design may be unnecessarily expensive and may
encourage specialization.

Several features contribute:

1. **Fixed training worlds.**  
   The same small world-seed set is reused every generation. Selection can
   therefore favor inherited details that exploit those particular histories.

2. **Very small environment sample.**  
   The pilot uses only two training worlds per candidate. Fitness therefore has
   high environmental variance.

3. **Direct mutation of the entire inherited neural parameter vector.**  
   Independent Gaussian noise is applied directly to all inherited weights.
   This is simple and valid as a bootstrap, but it is a relatively blind search
   in a high-dimensional parameter space.

4. **Expensive candidate evaluation.**  
   Fitness requires full simulated lives with predictive and homeostatic
   learning. As evolution succeeds and Noralets live longer, evaluation becomes
   more computationally expensive.

5. **Sequential small-model execution.**  
   The current harness evaluates many tiny neural/world workloads rather than
   exploiting large batched simulation throughput. CUDA availability does not
   automatically imply efficient GPU use for this workload.

These are methodological and engineering concerns, not evidence that natural
selection itself is the wrong principle.

## 12. Proposed simplification direction

A future Evolution v2 should preserve natural-selection-style external
selection, but should separate:

```text
natural selection as the scientific principle
```

from:

```text
a naive fixed-seed genetic algorithm as the optimizer
```

A simpler and potentially better-aligned design would use an **environment
distribution** rather than permanent training worlds:

```text
each generation
→ deterministically sample fresh shared world seeds
→ evaluate every candidate on the same sampled environments
→ select
→ mutate
→ next generation receives new environments
```

A separate fixed benchmark/probe seed bank would be evaluated only periodically
and would never affect selection.

Conceptually:

```text
SELECTION ENVIRONMENTS
fresh shared sample every generation
→ selection pressure

BENCHMARK ENVIRONMENTS
fixed larger holdout bank
→ measurement only
```

This is closer to selection for general viability than repeated optimization
against two permanent worlds.

## 13. Optimizer question

The current elite + parent-pool + Gaussian-mutation method should be treated as
a successful proof of concept, not necessarily the final evolutionary optimizer.

Possible future approaches include:

- retaining simple mutation-selection but using changing environments;
- increasing environment diversity before increasing population size;
- using paired perturbations / evolution-strategy estimates for more
  sample-efficient inherited-parameter search;
- improving batched evaluation throughput without changing evolutionary
  semantics.

No optimizer replacement is yet accepted as an architecture decision.

The next design discussion should prefer the **smallest change that resolves
the measured problem**.

## 14. Next controlled experiment

Before interpreting the Generation 11 training champion as a better general
BaseBrain, perform a dedicated **generalization audit**.

Evaluate saved historical inherited genomes such as:

```text
generation-000.pt
generation-005.pt
generation-010.pt
generation-014.pt
best.pt
```

across a substantially larger set of previously unseen world seeds.

No selection or mutation should occur.

Measure mean/median observed lifetime, lifetime distribution across worlds,
boundary-death fraction, Energy-depletion fraction, consumed Energy and
between-world variability.

This should distinguish:

### Hypothesis A — noisy validation

Later genomes generalize better overall, and the two existing validation seeds
were simply noisy.

### Hypothesis B — evolutionary specialization

Generalization peaks around an earlier generation and later training improvement
is specific to the fixed training worlds.

## 15. Current conclusion

Research 002 supports three conclusions:

1. **Inherited BaseBrain parameters are evolutionarily selectable.**  
   Training viability improved substantially across generations.

2. **The current evolutionary protocol is not yet a satisfactory general
   training method.**  
   Training and validation diverged after early improvement, and computation
   becomes increasingly expensive as lifetimes grow.

3. **Natural selection should remain the outer principle, but the implementation
   should probably become simpler and more distributional.**  
   The next design should avoid treating a tiny permanent set of world seeds as
   the environment to be memorized.

No claim about intelligence, intention, consciousness or subjective experience
follows from this evolutionary result.
