# Operation Report 013 — BaseBrain Evolution Bootstrap v1

## Summary

Iteration 13 adds a deterministic, headless, mutation-only evolutionary harness
for inherited BaseBrain initialization. It evaluates fresh Noralet lives with
the existing predictive and homeostatic lifetime learning enabled, selects only
on mean observed lifetime, persists resumable state and champion genomes, and
exposes the same harness through a lightweight Evolution UI tab.

The normal entry point is:

```text
uv run noralet evolution basebrain-bootstrap --generations 50 --device cuda
```

Generated runs are stored under:

```text
evolution-results/001-basebrain-bootstrap/<timestamp>-<config-fingerprint>/
```

`evolution-results/` is ignored by Git.

## Architecture boundary

- Neural topology, layer dimensions, learning rates, eligibility decay,
  perception schema, action semantics, and target construction are unchanged.
- World mechanics, tick phases, Energy laws, ecology, physiology, mortality,
  signals, and random-stream behavior are unchanged.
- There is no death reward, terminal penalty, terminal learning update,
  posthumous update, or respawn.
- Evolution is external to the world. A Noralet never observes candidate
  fitness, ranking, generation, parentage, or validation results.
- Every evaluated brain is a fresh clone of the inherited candidate. Adult
  predictive/homeostatic changes and recurrent state are discarded after the
  world evaluation and are never copied into the candidate genome.
- The frozen target encoder is not an independent genome. The existing
  `BaseBrain.spawn()` behavior still derives it from that fresh individual's
  inherited online encoder at birth.

## Evolution environment

Evolution Bootstrap Environment v1 reuses the Research 001 baseline builder and
all of its mechanics/configuration. Its one deliberate environmental difference
is initial stored body Energy:

- baseline Research 001 birth Energy: `60 eU`;
- Evolution Bootstrap v1 birth Energy: `10 eU`.

The exact evolution default is configurable with `--initial-energy`, but remains
`10 eU` in code and the manifest. No values were tuned after observing results.

The otherwise shared explicit configuration is:

- finite world `[-100, 100]`;
- left sparse `[-100, -25]`, central fertile `[-25, 25]`, and right sparse
  `[25, 100]`, each starting with 500 environmental Energy;
- formation probabilities infertile `0.001`, sparse `0.004`, fertile `0.012`;
  formation Energy `[4, 8]`, decay `0.002`, removal threshold `0.1`, spacing `3`;
- consumables at `-60, -20, 20, 60`, each starting with `20 eU`;
- bodies evenly placed between `-30` and `30`, zero velocity, age zero,
  condition one, deterministic perceptual signatures;
- body capacity `100`, existence cost `0.02/tick`, acceleration cost `0.1/unit`,
  consume radius `1`, and maximum acceleration `0.25`;
- unchanged Research 001 physiology, perception, signal, predictive-learning,
  and homeostatic-plasticity configuration.

The complete dataclass-expanded environment is written into every manifest.
The lower birth store changes the initial closed-system total but not Energy
conservation or transfer laws.

## Genome definition

`BaseBrainGenome` is a canonical, detached CPU snapshot of every named parameter
in the inherited BaseBrain prototype:

- ExperienceEncoder parameters;
- GRU recurrent-core parameters;
- PredictionModel parameters;
- acceleration-head parameters;
- consume-head parameters;
- signal-head parameters.

Parameter names, shapes, dtypes, and finite values are validated when loading.
The genome is stored on CPU for portable deterministic persistence, then copied
to the configured brain device by the existing spawn path. Architecture and
optimizer/hyperparameter state are not part of the genome.

## Evaluation protocol

Defaults per generation:

- candidates: `32`;
- shared training world seeds: `1101, 2203, 3301, 4409`;
- separate validation seeds: `5501, 6607, 7703, 8807`;
- Noralets per world: `6`;
- maximum ticks per world: `2000`;
- lives per training candidate: `4 × 6 = 24`;
- learning condition: `full-current-brain`;
- validation policy: evaluate that generation's best training candidate on all
  four validation worlds after every generation.

Every candidate within a generation receives the exact same ordered training
seed tuple. Validation seeds are disjoint and validation executes only after the
next population has already been created from training results.

Candidate fitness is exactly:

```text
mean observed lifetime ticks across all assigned training lives
```

Survivors at the cap contribute their truncated observed lifetime to v1 fitness
and are descriptively right-censored. This is documented as a **viability proxy
for evolutionary bootstrap**, not true reproductive fitness. Boundary, Energy
depletion, and natural death counts plus consumed Energy are descriptive only.
No auxiliary metric enters selection.

## Evolution algorithm

Generation 0 contains 32 independently initialized BaseBrain prototypes. Each
candidate receives its own SHA-256-derived BaseBrain seed; Generation 0 is not a
mutated founder population.

Later generations use:

- elite count: `4`;
- parent pool: top `8` by training fitness;
- population size: `32`;
- elites copied unchanged;
- remaining parent choices derived deterministically from the parent pool;
- independent additive Gaussian mutation on every inherited parameter tensor;
- mutation standard deviation: configurable, default `0.02`;
- no crossover, topology mutation, species, restarts, or adaptive schedule.

Training ties use stable candidate ID ordering. The current overall best genome
is retained when later fitness only ties it.

## Lifetime learning during evolution

Predictive lifetime learning and homeostatic action plasticity are both enabled
for every default evolution evaluation and champion watch. Each world creates a
new `BaseBrain` prototype loaded from the unchanged candidate genome, and the
existing runner spawns independent learned individuals from it. Only individual
copies receive online updates; the prototype/genome remains untouched.

Tests confirm that adult parameters change during a lived transition, while a
later fresh spawn returns exactly to the inherited genome instead of the adult
state.

## Result format

Each run contains:

- `manifest.json`: provenance, complete evolution/environment configuration,
  training/validation seeds, learning mode, seed derivation, and inheritance
  rules;
- `generations.csv`: best/mean/median/worst training fitness, champion ID,
  validation fitness, and death fractions per generation;
- `candidates.csv`: provenance, parent/elite/mutation metadata, training and
  optional validation fitness, lifetimes, deaths, population count, and
  descriptive consumed Energy per candidate;
- `champion/best.pt`: current overall best inherited genome;
- `champion/generation-000.pt` and every fifth-generation checkpoint, plus the
  final requested generation;
- `evolution-state.pt`: next generation, all population genomes and metadata,
  accumulated output rows, best-so-far state, complete configuration, and the
  stateless RNG scheme needed for continuation;
- `summary.md`: factual configuration, fitness/validation progression,
  champion, result location, and caveats.

Tensor checkpoints are loaded with PyTorch's `weights_only=True` mode.

## CLI

Normal run:

```text
uv run noralet evolution basebrain-bootstrap --generations 50 --device cuda
```

Resume or extend a saved run:

```text
uv run noralet evolution basebrain-bootstrap --generations 100 --resume evolution-results/001-basebrain-bootstrap/<run-id>/evolution-state.pt
```

Resume restores the saved scientific configuration and population; only the new
generation target and an optional explicit device override are applied. The CLI
also supports output root, population size, elite count, parent pool size,
mutation sigma, training/validation world counts, Noralets per world, maximum
ticks, initial Energy, initial seed, and champion cadence for controlled smoke
runs.

Progress identifies the result directory, generation, candidate index, training
fitness, validation fitness, and final output directory.

## UI

`uv run noralet ui` now includes an Evolution tab with:

- generation count and device selection;
- readable fixed/default protocol values and fitness definition;
- Start evolution and Stop controls;
- streamed child-process output and concise candidate/generation progress;
- result-directory discovery and Open Result Folder;
- Watch Champion for a completed run or a running result after its first
  champion checkpoint exists.

The tab invokes the actual headless CLI with `QProcess`; it contains no duplicate
evolution logic. Watch Champion loads `champion/best.pt`, creates a new
full-current-brain live session with the saved 10 eU environment setting, and
attaches it to the existing renderer. It starts at tick zero with fresh hidden,
optimizer, eligibility, and action-RNG state. It is a new life, not replay.

The ordinary Live Simulation reset path continues to create the normal 60 eU
baseline with random/default inherited BaseBrain parameters.

## Determinism

Generation-0 initialization, parent choice, and every child's mutation seed use
domain-separated SHA-256 derivation from the explicit initial evolution seed.
Python `hash()` and ambient Python/PyTorch RNG state are not used for evolutionary
choices. Gaussian mutation uses a dedicated CPU `torch.Generator` per child.

World/action/ecology/mortality randomness remains in the existing deterministic
named simulation streams. All candidates share training worlds; validation
worlds are separate.

A test compared a two-generation uninterrupted run with a one-generation run
resumed to generation two. `generations.csv` and `candidates.csv` were byte-for-
byte identical.

## Tests

Focused evolution command:

```text
QT_QPA_PLATFORM=offscreen uv run python -m unittest tests.test_evolution_genome tests.test_evolution_engine tests.test_evolution_ui -v
```

Result: `14/14` passed in `10.722s`, including inheritance purity, adult
non-inheritance, mutation, independent diversity, elitism, common-world fairness,
validation isolation, exact fitness, schema, resume equivalence, CLI, real CUDA,
Qt `QProcess`, and Watch Champion.

Compatibility command covering Research 001, headless CLI, baseline Live, and
the existing observer UI: `29/29` passed in `10.733s`.

Complete suite on final source:

```text
QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v
```

Result: `440/440` passed in `25.909s`.

Additional successful checks:

```text
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```

CPU CLI smoke completed two generations with two candidates, one training world,
one validation world, two Noralets/world, and a three-tick cap. CUDA CLI smoke
completed one equivalent generation on:

```text
torch.__version__: 2.13.0+cu130
torch.cuda.is_available(): True
torch.version.cuda: 13.0
torch.cuda.get_device_name(0): NVIDIA GeForce RTX 3060
```

The CUDA Watch Champion smoke loaded the saved genome exactly, started two fresh
Noralets at `10 eU`, ran three ticks, and produced six predictive plus six
homeostatic updates on CUDA.

The offscreen UI smoke ran the real evolution child process to exit code zero,
detected its result, enabled Watch Champion, started the fresh champion session,
and stepped it to tick one. The visually inspected screenshot is:

```text
evolution-results/ui-smoke/evolution-v1-styled.png
```

## Deviations

There are no implementation deviations from the requested Evolution Bootstrap
v1 scope. Validation used Qt's offscreen platform because no human-controlled
desktop session was available; the generated UI was inspected from the rendered
screenshot. The native Explorer window created by Open Result Folder was not
visually inspected.

The first ad-hoc Watch Champion print script referenced a nonexistent
`TickResult.state_before` observer field after it had already verified genome
identity. It was corrected to read the initial public `WorldState`, and the
complete smoke then passed. This was a validation-script typo, not a project-code
failure.

## Open notes

- The default `32 × 4 worlds × 2000 ticks × 50 generations` protocol is
  intentionally substantial; validation used only tiny overrides and did not
  run a scientific study.
- PyTorch emits the existing optional-NumPy warning because NumPy is not a
  dependency. The evolution, checkpoint, CUDA, UI, and test paths do not require
  NumPy and completed successfully.
- Fitness truncates capped lifetimes exactly as specified; generation and
  candidate outputs retain the interpretation caveat.

## Git state

Validation was performed from base commit
`8d553a4e4df891f93a84117e0b06bffcdda67f7d` with the Iteration 13 implementation
uncommitted. An unrelated deletion of
`research-docs/00-foundations/001-core-premise.md` and an unrelated untracked
`research-docs/README.md` appeared in the shared working tree during validation;
neither was created, modified, or restored by this iteration.

No commit or push was performed.
