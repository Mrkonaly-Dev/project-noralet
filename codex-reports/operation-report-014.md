# Operation Report 014 — Fast Evolution Generalization + Performance Audit

Date: 2026-08-25

## Scope

This patch implements the deliberately small Research 002 audit requested after
the completed 15-generation BaseBrain Evolution Bootstrap lineage. It answers:

1. whether saved later inherited BaseBrains perform better on a deterministic
   unseen-world bank than earlier checkpoints; and
2. whether CUDA improves throughput for the existing small, sequential learned-
   lifetime workload.

It does not modify evolution, neural architecture, world mechanics, learning,
fitness, mutation, selection, checkpoint schemas, saved genomes, or scientific
result schemas from earlier work.

## Implementation

`noralet.research.evolution_audit` adds one compact observer-side harness. It:

- automatically locates `generation-000.pt`, `generation-005.pt`,
  `generation-010.pt`, `generation-014.pt`, and `best.pt` beneath a completed
  Evolution Bootstrap result;
- loads each checkpoint through the existing safe `load_champion()` path;
- calls the existing `evaluate_candidate()` fresh-life evaluator separately for
  every saved genome/world pair;
- uses full-current-brain lifetime learning in each fresh life and discards all
  adult learned state afterward;
- performs no ranking, selection, mutation, reproduction, or continued
  evolution;
- derives unseen and benchmark seed banks with SHA-256 under the named domain
  `project-noralet:research-002:evolution-generalization-audit:seed:v1`;
- writes only the five requested compact outputs.

The command is:

```text
uv run noralet research evolution-audit \
  --evolution-result <completed-evolution-result-directory>
```

`--audit-seed`, `--generalization-device`, and `--output-root` are optional
reproducibility/execution controls. They do not alter the fixed 5-genome, 8-world,
4-Noralet, 1,000-tick generalization protocol or the fixed 4-world, 4-Noralet,
250-tick performance protocol.

## Audit seed banks

Audit seed: `20260825`.

Excluded known evolution seeds:

- training: `1101`, `2203`;
- validation: `5501`, `6607`;
- documented qualitative probe: `94476`.

Unseen generalization worlds:

```text
2157287210111473669
8220454516905462506
7136369498497502814
3707609362642838765
5819872299436858616
3295240427133885173
6606366054786922742
6840103523807828689
```

CPU/CUDA benchmark worlds, derived under a separate role and also disjoint from
the unseen bank:

```text
9208378890936581344
8341227759016949069
2343147448247112603
5686149876834081524
```

No seed was hand-picked or changed after observing results. Every saved genome
received the exact same unseen bank.

## Full validation audit

The full requested generalization audit was run against lineage:

```text
evolution-results/001-basebrain-bootstrap/
20260825T121340.095069Z-8fbe71a650
```

Generalization used CPU because the preliminary identical-workload timing showed
that CUDA was substantially slower. Device choice does not change the fixed
genome/world/lifetime protocol and is recorded in the manifest.

Result directory:

```text
research-results/002-evolution-generalization-audit/
20260825T145214.238989Z-00ea3524e6
```

The run completed all `5 × 8 × 4 = 160` fresh learned lives with a 1,000-tick
cap. All lives terminated before the cap; there were no censored survivors.

## Generalization results

| Checkpoint | Saved identity | unseen mean | median | world-mean SD | boundary | Energy depletion | mean consumed Energy |
|---|---|---:|---:|---:|---:|---:|---:|
| generation-000 | g000-c005 | 69.656 | 63.0 | 7.910 | 100.0% | 0.0% | 13.679 |
| generation-005 | g005-c006 | 121.469 | 120.0 | 14.514 | 100.0% | 0.0% | 14.728 |
| generation-010 | g010-c002 | 140.188 | 110.0 | 39.403 | 100.0% | 0.0% | 15.927 |
| generation-014 | g011-c007 | 144.000 | 118.5 | 14.580 | 96.9% | 3.1% | 12.025 |
| best | g011-c007 | 144.000 | 118.5 | 14.580 | 96.9% | 3.1% | 12.025 |

The saved `generation-014.pt` and `best.pt` both identify the same generation-11
best-so-far candidate, so their scientific rows are identical as expected.

Factual result:

- unseen mean lifetime increased from `69.656` at Generation 0 to `121.469` at
  Generation 5, `140.188` at Generation 10, and `144.000` in the final saved
  champion;
- Generation 14's aggregate median (`118.5`) was slightly lower than Generation
  5's (`120.0`);
- Generation 10 had notably larger between-world variation;
- boundary death remained dominant, and there was one Energy-depletion death in
  the final/best checkpoint evaluation.

The decision rule was recorded before interpretation: Generation 14 must exceed
Generation 5 in mean without a lower median to produce `later genomes clearly
better`; the inverse comparison produces `earlier genome better`; otherwise the
signal is `unclear`. The resulting gate is therefore:

```text
Generalization signal:
- unclear

Fixed-world specialization concern:
- unresolved
```

The higher later unseen means weaken a simple claim that improvement was confined
to the two training worlds, but this compact audit does not establish a clear
monotonic generalization advantage and does not label the lineage overfit.

## CPU vs CUDA throughput

Checkpoint: `generation-005.pt` (`g005-c006`).

Protocol per repetition:

- four fixed benchmark worlds;
- four Noralets/world;
- maximum 250 ticks;
- full-current-brain lifetime learning;
- identical inherited genome and seeds;
- 801 world ticks executed;
- 2,164 Noralet activations/lived transitions.

One warm-up and three measured repetitions were run per device. Explicit CUDA
synchronization enclosed each measured wall-clock interval.

| Device | mean seconds | median seconds | effective ticks/s |
|---|---:|---:|---:|
| CPU | 12.283253 | 12.277869 | 65.211 |
| CUDA | 74.330222 | 74.164966 | 10.777 |

```text
CUDA speedup versus CPU: 0.165×
```

For this small, sequential workload CUDA is clearly slower: approximately
`6.05×` the CPU wall time. This is a throughput observation only. GPU utilization
was not measured or inferred.

Coarse phase timing was skipped. The existing causal runner step combines brain
activation, predictive learning, homeostatic update, and world orchestration;
separating those phases would require runtime instrumentation beyond this small
observer-only audit.

## Outputs

The completed result contains exactly:

```text
manifest.json
genome-summary.csv
world-results.csv
performance.json
summary.md
```

The manifest records Git SHA/dirty state, Python, PyTorch, CUDA, GPU, source
lineage, checkpoint paths/hashes/identities, both seed banks, both protocols,
devices, repetition counts, timing caveats, and the decision rules/gate.

All five source checkpoint SHA-256 hashes were recomputed after the full run and
still matched the manifest exactly.

## Tests

Focused audit coverage includes saved genome loading, deterministic shared seed
banks, exclusion of training/validation/probe seeds, unchanged checkpoint hashes,
explicit mutation/selection call guards, output schemas, summary sections, CPU
timing, and real CUDA timing:

```text
uv run python -m unittest tests.test_evolution_audit -v
Ran 4 tests in 2.512s
OK
```

CLI compatibility plus audit tests:

```text
uv run python -m unittest tests.test_cli tests.test_evolution_audit -v
Ran 6 tests in 5.997s
OK
```

Complete regression:

```text
uv run python -m unittest discover -s tests
Ran 454 tests in 32.405s
OK
```

## Deviations and notes

- The optional Evolution-tab `Run quick audit` button was not added. A second
  research-process lifecycle in the already stateful Evolution panel would
  materially expand this deliberately small patch. The instruction explicitly
  permits CLI-only operation, and the full audit requires one command.
- Progress output is explicitly flushed for CLI/QProcess-friendly streaming.
- The optional NumPy initialization warning remains because NumPy is not a
  project dependency; it did not affect audit, CUDA, or tests.
- The user-owned untracked research document
  `research-docs/002-basebrain-evolution-bootstrap.md` was read for the declared
  primary lineage and qualitative probe seed but was not modified.

Validation used base Git commit
`409256f675ad6352f162e5a07d2654d43a9fde4c`. The worktree was dirty as recorded
in the audit manifest. No commit or push was performed.
