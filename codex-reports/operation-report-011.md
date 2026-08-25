# Operation Report 011 — Baseline Lifetime Adaptation Research Harness

## Summary

Research Iteration 1 implemented a headless, reproducible experimental harness
for `001-baseline-lifetime-adaptation`. It executes a deterministic
condition-by-replicate matrix, observes existing public runtime results and
published state, streams sampled longitudinal data, and writes compact factual
summaries for later human/assistant interpretation.

The normal pilot command is:

```text
uv run noralet research baseline-lifetime-adaptation --seeds 10 --max-ticks 5000 --sample-every 10 --device cuda
```

The initial population defaults to six. `--population`, `--conditions`, and
`--output-root` allow controlled smoke/debug variants without changing the
normal protocol defaults.

## Architecture preservation

No causal simulation or neural behaviour changed. No simulation phase,
`NoraletExperience` field, neural architecture, learning target, action rule,
homeostatic equation, random-stream implementation, or existing default
configuration was modified. Research instrumentation is external to the
organism and consumes only observer-visible action intents, tick events,
learning results, homeostatic results, published experiences, model snapshots,
and current runtime state.

## Experiment protocol

The harness uses the complete naturally supported 2 × 2 learning matrix:

| Condition | Predictive lifetime learning | Homeostatic action plasticity |
|---|---:|---:|
| `no-learning` | disabled | disabled |
| `predictive-only` | enabled | disabled |
| `full-current-brain` | enabled | enabled |
| `homeostatic-only` | disabled | enabled |

The optional fourth condition required no architecture change: the existing
`BaseBrain` API already accepts homeostatic plasticity without predictive
learning. It is therefore included in the default protocol.

Pilot defaults:

- replicate seeds: integers 1 through 10;
- conditions: all four above;
- runs: 40 sequential runs;
- maximum duration: 5,000 ticks per run;
- timeseries cadence: every 10 ticks, plus tick 0 and a non-cadence final
  survivor tick;
- initial population: six Noralets;
- requested neural device: CUDA;
- stop rule: first of total extinction or tick 5,000;
- survivors at the tick limit: right-censored, never recorded as deaths;
- prediction-loss window: first and last `min(100, available_updates)`
  successful updates per individual.

Hypotheses H1–H5 are recorded in `manifest.json` before the first condition run:
lifetime effects, predictive-loss development, homeostatic action-policy
development, divergence through individual histories, and descriptive-only
signal behaviour. They are stored as hypotheses rather than conclusions.

## Baseline configuration

These are explicit Research 001 experimental choices frozen by the harness.
They are not new architecture constants and do not alter existing project
defaults.

- World: boundaries `[-100, 100]`.
- Regions: left sparse `[-100, -25]`, central fertile `[-25, 25]`, right sparse
  `[25, 100]`; each environmental pool starts with 500 Energy.
- Ecology: formation probabilities infertile `0.001`, sparse `0.004`, fertile
  `0.012`; formation Energy `[4, 8]`; decay `0.002`; removal threshold `0.1`;
  minimum point spacing `3`.
- Initial consumables: positions `-60, -20, 20, 60`, each with 20 Energy.
- Initial bodies: evenly spaced from `-30` to `30`; velocity `0`; Energy `60`;
  age `0`; condition `1`; deterministic two-value perceptual signatures.
- Body Energy: capacity `100`; existence cost `0.02/tick`; acceleration cost
  `0.1/unit`; consume radius `1`.
- Physiology: low-Energy ratio `0.4`; baseline condition loss `0.00002`;
  deprivation scale `0.001`, exponent `2`; base mortality hazard `0.00002`;
  age scale `5000`, age exponent `2`; condition hazard scale `0.0002`, exponent
  `2`; age hazard scale `0.00005`; interaction scale `0.0001`.
- Experience: vision radius `12`; consumable appearance `(0.72, -0.11)`;
  Noralet appearance `(0.09, 0.83)`; boundary appearance `(-0.44, 0.17)`;
  signature length `2`; Energy-distress exponent `2`; condition-distress
  exponent `1.5`; motor-effort scale `2`; ingestion scale `5`; exertion scale
  `3`.
- Signals: radius `20`; Energy cost `0.02`; patterns A
  `(0.91, -0.13, 0.27)`, B `(-0.22, 0.84, 0.31)`, C
  `(0.18, 0.36, -0.77)`, D `(-0.63, -0.24, 0.52)`.
- Actuator: maximum acceleration `0.25`.
- Brain dimensions: external percept embedding `4`; signal embedding `4`;
  interoception embedding `3`; sensorimotor embedding `4`; combined experience
  embedding `6`; GRU hidden size `7`; acceleration exploration standard
  deviation `0.2`.
- Predictive learning: learning rate `0.01`; maximum gradient norm `1`;
  predictor hidden size `8`.
- Homeostatic plasticity: Energy-distress weight `1`; condition-distress weight
  `1`; modulation scale `0.2`; eligibility decay `0.8`; action learning rate
  `0.05`; maximum update norm `2`.

The complete machine-readable values, including initial bodies and all seed
mappings, are also serialized in every manifest.

## Metrics

Run-level metrics include condition and three seed roles, status/technical
error, start/final/maximum ticks, extinction, survivors, runtime, requested
device resolution, population, total deaths, and boundary, Energy-depletion,
and natural-death counts.

Per-individual summaries include observed lifetime and censoring; initial,
final-observed, minimum, and sampled-mean physiology; distress means/maxima;
absolute displacement and velocity; selected consume attempts versus actual
positive Energy transfers; Energy consumed; actual signal counts by type and
direction; received percept counts; requested-acceleration statistics; exact
consume and nine-way signal-selection counts; predictive loss/update summaries;
homeostatic drive/modulation/eligibility/update summaries; and final encoder,
GRU, predictor, and action-head L2 drift from immutable birth snapshots.
Disabled components use null values where zero would be misleading.

Timeseries rows contain experimental condition, replicate, tick, Noralet ID,
alive status, objective position/velocity, Energy, physiological condition,
distress, exertion, per-transition predictive/homeostatic metrics when present,
hidden-state norm, displacement, and cumulative action/event counters. Exact
objective values remain external and never enter the brain-facing experience.

## Output format

Generated batches are ignored by Git and use:

```text
research-results/001-baseline-lifetime-adaptation/<run-id>/
```

- `manifest.json`: schema/version, timestamps, Git/Python/PyTorch/CUDA/GPU
  provenance, CLI arguments, condition definitions, SHA-256 seed mappings,
  complete baseline, hypotheses, cadence, stopping and interpretation rules.
- `run-summary.csv`: one stable-column row per condition × replicate.
- `noralet-summary.csv`: one stable-column row per observed individual lifetime.
- `timeseries.csv`: stable-column sampled longitudinal rows.
- `aggregate-summary.json`: condition-level counts, extinction/survival and
  observed/censored lifetime summaries, death causes, distress, consumption,
  signals, learning/homeostatic statistics, movement/action summaries, and
  parameter drift.
- `summary.md`: factual configuration, completion, condition, predictive,
  homeostatic, survival/physiology, behaviour, parameter-development, and caveat
  sections. It makes no automatic architecture or consciousness claim.

## Reproducibility

Each user-facing replicate seed deterministically derives separate simulation
and BaseBrain seeds with domain-separated SHA-256. Python `hash()` is not used.
All conditions for a replicate receive the same simulation seed, BaseBrain seed,
physical initial state, perceptual signatures, inherited Iteration 8 parameters,
zero hidden states, and named action-RNG state. Only the two learning enablement
flags differ.

A same-arguments/two-seed test executed complete batches twice and compared
`run-summary.csv` after removing wall-clock runtime plus complete
`noralet-summary.csv`, `timeseries.csv`, `aggregate-summary.json`, and manifest
scientific fields. All scientific values matched; only documented timestamps,
run IDs, and duration/provenance fields were excluded.

## Observation purity

The explicit causal-equivalence test ran otherwise identical full-current-brain
worlds with research recording off and on. After every tick it verified equal
`ActionIntent`s, tick results/events/learning metrics, and `WorldState`s. At the
end it also verified exact neural parameters, recurrent hidden states, and named
action RNG states. All comparisons passed.

## Resource behaviour

`timeseries.csv` is written incrementally. The harness retains scalar lifetime
accumulators, first/last 100-loss bounded windows, compact run/individual summary
rows, live brain references, and one immutable birth parameter snapshot per
individual in the current sequential run. It does not retain WorldState history,
hidden vectors, full per-tick weight tensors, or autograd graphs.

If all six individuals survive, the default 40-run pilot produces at most
120,240 sampled timeseries rows (`40 × 6 × 501`) plus 40 run rows and 240 lifetime
rows. Based on the compact schema this is expected to be a tens-of-megabytes
dataset, not an unbounded in-memory history.

## CPU validation

Command:

```text
uv run noralet research baseline-lifetime-adaptation --seeds 2 --max-ticks 3 --sample-every 2 --device cpu --population 2 --output-root research-results/final-validation-cpu
```

Result: 8/8 condition × replicate runs completed, 16 lifetime rows and 48
timeseries rows; every run reported resolved device `cpu`.

The normal CLI smoke also passed:

```text
uv run noralet run --ticks 5 --seed 12345
Completed 5 tick(s); final tick: 5; seed: 12345
```

## CUDA validation

Environment validation:

```text
torch.__version__: 2.13.0+cu130
torch.cuda.is_available(): True
torch.version.cuda: 13.0
torch.cuda.get_device_name(0): NVIDIA GeForce RTX 3060
```

Command:

```text
uv run noralet research baseline-lifetime-adaptation --seeds 2 --max-ticks 3 --sample-every 2 --device cuda --population 2 --output-root research-results/final-validation-cuda
```

Result: 8/8 condition × replicate runs completed, 16 lifetime rows and 48
timeseries rows; every run reported resolved device `cuda`. The focused CUDA
unittest also completed a full-current-brain autonomous research tick.

PyTorch emitted its existing optional-NumPy warning because NumPy is not a
project dependency; no research path uses NumPy and validation completed.

## Tests

Targeted research/CLI command:

```text
uv run python -m unittest tests.test_research_config tests.test_research_metrics tests.test_research_output tests.test_cli -v
```

Result after final source changes: 16/16 passed in 6.361 seconds.

Complete regression command:

```text
uv run python -m unittest discover -s tests -v
```

Result after final source changes: 408/408 passed in 11.511 seconds.

Additional final validation:

```text
uv run python -m compileall -q src tests
```

Result: passed with no output.

```text
uv lock --check
```

Result: passed; 30 packages resolved and the lockfile was current.

```text
git diff --check
```

Result: passed. Git printed only the existing Windows line-ending conversion
notices for three tracked text files; no whitespace error was reported. A
separate trailing-whitespace scan of the new untracked source, tests, and report
also found no matches.

## Deviations

No instruction deviations occurred. The optional homeostatic-only condition was
included because the existing architecture supports it naturally.

## Research limitations

This is a limited-seed pilot protocol. Ordinary lifetime means and medians are
explicitly labelled observed/censored summaries, not unbiased survival
estimates. Right-censoring may be present, smoke durations are not substantive
experimental evidence, and descriptive association does not establish a
mechanism. No intelligence or consciousness inference is made.

For deaths, `final_observed_energy` and `final_observed_condition` mean the last
published living values; the current death event exposes cause, tick, and
resolved position but no post-removal physiological body.

## Renderer status

Renderer remains a separately planned future engineering/observer component and was not part of this research iteration.

## Git state

No commit or push was performed. Generated validation results are ignored. The
working tree contains only the intended Research 001 harness, CLI, tests,
`.gitignore`, and this operation report changes.
