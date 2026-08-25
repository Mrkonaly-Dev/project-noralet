# Operation Report 015 — Evolution v2: Fast Distributional Selection

Date: 2026-08-25

## Summary

Implemented the deliberately small second evolution protocol:

```text
002-basebrain-distributional-evolution
```

Evolution v2 keeps the successful mutation-only BaseBrain selection loop while
replacing permanent training worlds with fresh, deterministic, shared selection
worlds each generation. A separate fixed benchmark bank supplies the only
standardized longitudinal signal. Results are written beneath:

```text
evolution-results/002-basebrain-distributional-evolution/<run-id>/
```

Evolution Bootstrap v1 remains available and its saved lineage was not rewritten.
No neural architecture, sensory channel, lifetime-learning rule, world mechanic,
fitness component, or in-world reproduction behavior changed.

## Motivation

Research 002 established three practical facts about the completed 15-generation
v1 lineage:

- later saved genomes had substantially higher mean lifetime on the unseen-world
  bank (`69.656` at Generation 0 versus `144.000` for the final saved champion);
- fixed-world specialization remained unresolved rather than proven because the
  preregistered aggregate gate was still unclear;
- the small sequential workload measured `65.211` effective world ticks/s on CPU
  and `10.777` on CUDA, making CUDA approximately `6.05×` slower in wall time.

The existing simple evolutionary mechanism was therefore retained. This patch
changes environment sampling and measurement, not the optimizer.

## Protocol

The v2 defaults are:

```text
generations:                    20
device:                         CPU
population:                     8
elite count:                    2
parent pool:                    4
mutation sigma:                 0.02
selection worlds / generation: 4
fixed benchmark worlds:        8
benchmark interval:             5
Noralets / world:               4
max ticks:                      1000
birth Energy:                   10 eU
learning mode:                  full-current-brain
```

Every candidate is a fresh individual on every world. Predictive lifetime
learning and homeostatic action plasticity remain enabled during life; learned
adult state is discarded afterward. Only inherited BaseBrain prototype
parameters enter the next generation.

Each output contains the requested compact files:

```text
manifest.json
generations.csv
candidates.csv
benchmarks.csv
champion/best.pt
champion/benchmark-generation-XXX.pt
evolution-state.pt
summary.md
```

The state retains only the population genomes, accumulated scientific rows,
deterministic derivation inputs, current selection champion, benchmark-best
genome, fixed benchmark seeds, and optional fork provenance. It does not save
every candidate as a separate file.

## Forking

The CLI supports:

```text
uv run noralet evolution distributional \
  --generations 20 \
  --fork-from <v1-evolution-state.pt>
```

Fork is explicitly a new v2 lineage, not resume. The v1 checkpoint's final
population genomes are copied into newly identified v2 Generation 0 candidates.
Adult learned parameters, hidden state, optimizer state, eligibility traces, and
world state are not inherited.

The v2 manifest records the source evolution ID/run ID, resolved checkpoint path,
checkpoint SHA-256, completed generation count, source candidate IDs, new v2 IDs,
per-genome SHA-256 values, and v2 start generation 0.

The real fork smoke used the completed source lineage:

```text
evolution-results/001-basebrain-bootstrap/
20260825T121340.095069Z-8fbe71a650/evolution-state.pt
```

It recorded source completed generation `15`, copied all eight genomes, and
started v2 Generation `0`. The source SHA-256 was identical before and after:

```text
ae5fd8fc2a107020e5406ad6b4f984f8ab024e49950b1ca873eaebfda852f95a
```

## Selection

Selection remains:

```text
evaluate
→ stable rank by mean lifetime
→ copy elites unchanged
→ deterministic parents from the top parent pool
→ deterministic additive Gaussian mutation
```

There is no crossover, topology change, adaptive mutation, novelty objective,
reward shaping, new fitness term, or architecture mutation. Parent and mutation
seed derivation preserves the v1 deterministic scheme.

Candidate selection fitness is exactly the mean observed lifetime across the
current generation's selection lives. Food, movement, distress, signaling,
prediction error, and death cause are recorded or remain causal world effects;
none enters selection as an added reward.

## Benchmark

Selection seeds are derived with SHA-256 from the initial evolution seed,
generation index, and world slot. Every candidate in one generation receives the
same ordered bank, while different generations receive fresh banks.

Benchmark seeds are derived once under a separate role and remain fixed for the
lineage. An explicit even/odd partition makes selection and benchmark seed banks
disjoint in addition to their domain separation. Benchmark worlds never enter
candidate ranking or parent construction.

Only the current selection champion is benchmarked at Generation 0, every
configured interval, and the requested final generation. If a completed run is
later extended, a benchmark that existed only because the previous target was
final is retracted before continuation. This makes the extended output and future
benchmark schedule match an uninterrupted run to the new target.

`benchmarks.csv` records mean/median lifetime, world-mean standard deviation,
boundary/Energy-depletion/natural death fractions, and mean consumed Energy.
Benchmark-best uses highest benchmark mean lifetime with candidate ID as a stable
tie-break. Only:

```text
champion/best.pt
```

is labelled `benchmark-best`. Periodic checkpoint files are labelled as
benchmark-evaluated selection champions. Raw selection fitness on a different
generation cannot replace `best.pt`.

The generated `summary.md` explicitly states that between-generation selection
fitness is not standardized and uses the fixed benchmark rows for longitudinal
progress.

## Performance

CPU is the v2 default in the configuration, CLI, and normal Qt form. Explicit
`--device cuda` and optional resume device override remain supported and were
validated on the installed NVIDIA GeForce RTX 3060.

No CUDA optimization, GPU batching, multiprocessing, profiling framework, or
evaluation architecture redesign was added. Candidate/world evaluation remains
sequential.

## UI

The existing Evolution tab now uses Distributional Evolution v2 as its normal
protocol and shows the requested v2 fields, defaults, within-generation fitness
warning, and fixed-benchmark longitudinal role. Evolution Bootstrap v1 remains
selectable as a legacy protocol for fresh compatibility and its own checkpoints.

`Fork from previous evolution…` opens a normal Qt checkpoint picker, accepts only
v1 state, and clearly shows source lineage, checkpoint, source completed
generation, copied population, new v2 protocol, and new Generation 0. It never
calls the operation Resume.

Resume metadata is protocol-aware. Each checkpoint resumes through its own
existing CLI command/QProcess path, with checkpoint scientific fields locked.
Only total target generation and optional device override are emitted. A v1
checkpoint cannot be resumed through the v2 engine; it must be forked.

Watch Champion continues to construct a fresh full-learning life. For v2 it loads
`best.pt` and the observer is labelled:

```text
BENCHMARK-BEST CHAMPION WATCH
```

The panel shows source run, genome ID, source generation, benchmark mean/median,
fresh-life status, birth Energy, source/watch device, and new live seed/population.

## Determinism

The v2 selection and benchmark seed derivation uses stable, domain-separated
SHA-256 and never Python `hash()`. Candidate initialization, parent choice, and
mutation are also stateless deterministic derivations. Checkpoints therefore need
no ambient Python or global PyTorch RNG state.

The interrupted/resumed test begins with a two-generation target whose Generation
1 was benchmarked only because it was final, then resumes to four total
generations. Its `generations.csv`, `candidates.csv`, `benchmarks.csv`, final
population genome hashes, and next-generation index exactly match a direct
four-generation run.

## Tests

Focused v2 engine/UI coverage includes all requested seed, fairness, benchmark,
best checkpoint, fork, source-integrity, protocol-separation, resume, CPU-default,
CUDA, QProcess, and Watch Champion behavior.

```text
uv run python -m unittest \
  tests.test_distributional_evolution \
  tests.test_evolution_ui -v
```

Both focused modules passed. The explicit resume-equivalence plus real CUDA/watch
smoke also passed independently:

```text
Ran 2 tests in 6.265s
OK
```

Complete regression:

```text
uv run python -m unittest discover -s tests -v
Ran 464 tests in 46.221s
OK
```

Final repository gates:

```text
uv run python -m compileall -q src tests
exit 0

uv lock --check
Resolved 34 packages in 1ms

git diff --check
exit 0
```

`git diff --check` emitted only Git's existing LF→CRLF working-copy notices;
it reported no whitespace error.

The requested CPU CLI smoke ran two generations with four candidates, two
selection worlds, two Noralets/world, 50 maximum ticks, and two benchmark worlds:

```text
evolution-results/iteration15-cpu-smoke/
002-basebrain-distributional-evolution/
20260825T153714.879254Z-d63b651673
```

It used CPU without an explicit `--device`. A CLI resume then continued the same
result from two to three total generations. The prior Generation 1 terminal-only
benchmark was removed and the resumed final benchmark occurred at Generation 2,
leaving benchmark rows `0, 2`.

The real v1→v2 fork smoke result is:

```text
evolution-results/iteration15-fork-smoke/
002-basebrain-distributional-evolution/
20260825T153741.058927Z-62474ce23b
```

No long evolution study was run.

## Deviations

- The v2 engine is a focused module that reuses the existing genome, evaluation,
  ranking, mutation, checkpoint, and provenance primitives. The whole v1 engine
  was not generalized because that would broaden risk without reducing this
  deliberately small protocol implementation.
- Selection/benchmark parity partitioning is stricter than role-only domain
  separation and guarantees their seed integers cannot overlap.
- The existing optional NumPy initialization warning remains because NumPy is not
  a project dependency. It did not affect simulation, CUDA, or test outcomes.
- Generated smoke result directories remain ignored runtime artifacts.
- The user-owned untracked
  `research-docs/002-basebrain-evolution-bootstrap.md` was not modified.

## Git state

Implementation was performed on base commit:

```text
409256f675ad6352f162e5a07d2654d43a9fde4c
```

The worktree also retains the uncommitted Research 002 audit implementation and
operation report from the preceding instruction. No commit or push was performed.
