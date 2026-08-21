# Operation Report 003 — Closed Energy Ecology

**Iteration:** 3
**Date:** 2026-08-21
**Status:** Complete

## Summary

Iteration 3 introduced an immutable, deterministic closed energy ecology for the finite continuous one-dimensional world. A configured world is partitioned into regions, each region owns a local Environmental Energy pool, and positive Consumable Energy exists as stationary discrete points.

The runtime now resolves proportional point decay, complete small-point dissolution, and region-dependent Environmental-to-Consumable formation. Every successful transition is checked against a fixed initial universe-energy baseline before publication.

Iteration 2 motion remains unchanged and physically independent from the ecology. Noralets do not store, sense, consume or spend energy.

## Files changed

- `src/noralet/world/regions.py` — adds immutable region definitions and the `INFERTILE`, `SPARSE` and `FERTILE` observer classifications.
- `src/noralet/world/energy.py` — adds environmental pools, consumable points, observable totals, formation probabilities, ecology configuration and the conservation error.
- `src/noralet/world/__init__.py` — exposes the implemented world-domain values.
- `src/noralet/simulation/config.py` — accepts an optional explicit energy ecology and validates its exact world partition.
- `src/noralet/simulation/state.py` — stores canonical immutable pools and points, the next deterministic point identity, and carefully summed energy totals.
- `src/noralet/simulation/runtime.py` — initializes the closed universe, resolves decay/dissolution/formation, uses isolated named streams, and audits conservation before state publication.
- `src/noralet/simulation/events.py` — adds immutable formation, decay and dissolution events.
- `src/noralet/simulation/__init__.py` and `src/noralet/__init__.py` — expose the Iteration 3 public API.
- `tests/energy_test_support.py` — provides explicit compact test ecology construction.
- `tests/test_energy_validation.py` — covers region partitions, boundary ownership, configuration and initial energy validation.
- `tests/test_energy_ecology.py` — covers formation, decay, dissolution, locality, timing, no diffusion, immutability and event phases.
- `tests/test_energy_determinism.py` — covers insertion-order independence, multi-tick reproducibility, RNG isolation, long-run conservation and failed-audit publication protection.
- `codex-reports/operation-report-003.md` — records this implementation iteration.

## Region model

`RegionDefinition` is a frozen value containing only:

```text
region_id
left
right
kind
```

`RegionKind` is a string enum with `INFERTILE`, `SPARSE` and `FERTILE`. These are observer-side formation classifications and have no effect on motion or any Noralet state.

`EnergyEcologyConfig` canonicalises regions by spatial coordinates and stable identity. `SimulationConfig` then requires them to cover the traversable world exactly. Empty definitions, duplicate identities, gaps, overlaps, outside extents and incomplete coverage are rejected.

The ownership convention is:

```text
[left, right)
```

for every non-final region, while the final region owns:

```text
[left, right]
```

Consequently every traversable coordinate, including shared boundaries and the world's final right boundary, has exactly one owning region.

## Environmental Energy

`EnvironmentalEnergyPool` is an immutable pair of stable region identity and finite non-negative energy measured in abstract `eU`. There is exactly one explicitly configured pool for each configured region. Missing, extra and duplicate pool identities are rejected.

Pools remain separate in `WorldState`; no global mutable environmental pool exists. Ecology resolution addresses each pool by region identity, and there is no direct transfer or diffusion between regions.

Initial Environmental Energy is supplied through `EnergyEcologyConfig.initial_environmental_energy`. It is never randomly invented.

## Consumable Energy

`ConsumableEnergyPoint` contains only a non-negative integer `point_id`, finite continuous `position`, and finite positive `energy`. Points are immutable and stationary. Their position determines their single owning region using the canonical interval convention.

Explicit initial points are accepted by `Simulation(..., initial_energy_points=...)`. Duplicate identities and invalid or out-of-world values are rejected. Generated identity is the next integer above the largest initial point identity, or zero when there are no initial points. New identities increase monotonically in canonical formation order; no UUIDs or hashes are used.

Formation performs an equal transfer from one region's Environmental Energy pool into one new point. A region gets at most one attempt per tick. It cannot form a point below the configured minimum or remove more energy than is locally available.

Existing points decay according to:

```text
E_next = E_current * (1 - decay_rate)
E_returned = E_current - E_next
```

`E_returned` goes to the point's owning region. If `E_next` is at or below the configured removal threshold, its complete remainder is also returned locally and the point is omitted from the successor state. No point remainder is discarded.

## Tick ecology order

For one `t -> t+1` transition, the runtime performs this sequence:

1. freeze `WorldState(t)` as the shared input and validate external action intentions;
2. calculate all Iteration 2 motion consequences from that same prior state;
3. decay all points that existed in `WorldState(t)`, in ascending point-ID order;
4. return decay transfers and complete dissolution remainders to their owning region pools;
5. attempt formation once per region in canonical left-to-right region order, using the post-decay pools;
6. construct the complete immutable `WorldState(t+1)`;
7. audit its universe-energy total against the initial baseline;
8. create the final observer result and publish the state only after the audit succeeds.

Newly formed points are not part of step 3 because that phase reads only `WorldState(t)`. Their first decay opportunity is the next transition.

Physical and ecological calculations are independent in Iteration 3. Neither subsystem observes a partially constructed successor state.

## Randomness

All ecology randomness uses the existing simulation-owned `DeterministicRandomStreams`. The runtime uses a separate stable stream for every region and purpose:

```text
energy:region:{region-id-length}:{region-id}:formation:trigger
energy:region:{region-id-length}:{region-id}:formation:amount
energy:region:{region-id-length}:{region-id}:formation:position
```

Every configured region consumes exactly one trigger draw per tick. Amount and position draws occur only for successful, sufficiently funded formation, and come from their own purpose streams. Activity in one region therefore cannot advance another region's sequences, and ecology activity cannot advance existing names such as `world` or `mortality`.

The same master seed, configuration, initial bodies, initial points and action sequence reproduce identical states and event histories. Region, pool, point, body and intent insertion order cannot change equivalent results.

## Energy conservation

The implemented Iteration 3 law is:

```text
E_total
=
E_environmental
+
E_consumable
=
constant
```

`Simulation` calculates `initial_total_energy` from every configured Environmental Energy pool and every explicit initial Consumable Energy point. `WorldState.energy_totals` exposes immutable environmental, consumable and combined observer totals, using `math.fsum` for careful summation.

Before every successor state is published, `Simulation.audit_energy_conservation` recomputes its total and compares it with the initial baseline. The comparison has zero relative tolerance and an absolute tolerance of `1e-9 eU`, solely for tiny binary floating-point representation effects.

Non-finite totals or differences beyond that tolerance raise `EnergyConservationError` with expected, observed and difference values. The runtime does not repair the totals and does not publish the invalid successor state.

Formation, proportional decay and dissolution are implemented only as equal source-to-destination transfers. Region-local pools never become negative, and a zero-energy universe cannot form energy.

## Events

The observer-only immutable event model now includes:

- `EnergyPointFormed` — region, new point identity, position, transferred energy and tick transition;
- `EnergyPointDecayed` — region, point identity, returned energy, post-decay energy and tick transition;
- `EnergyPointDissolved` — region, removed point identity, final returned remainder and tick transition.

Failed formation attempts do not emit events. Events do not drive mechanics.

The deterministic phase-first event order is:

1. `NoraletAccelerated` by ascending Noralet ID;
2. `NoraletMoved` by ascending Noralet ID;
3. `NoraletDied` by ascending Noralet ID;
4. `EnergyPointDecayed` by ascending point ID;
5. `EnergyPointDissolved` by ascending point ID;
6. `EnergyPointFormed` by canonical left-to-right region order;
7. `TickAdvanced`.

## Tests and validation

The pre-Iteration 3 suite contained 40 passing Iteration 1 and Iteration 2 tests.

The final suite contains 77 tests: all 40 existing tests plus 37 focused Iteration 3 tests. It was run once after the new test groups were added and twice more after final cleanup:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run python -m unittest discover -s tests -v
```

All three runs passed 77 of 77 tests. The final two runs completed in `0.423s` and `0.429s` respectively.

The suite covers valid and invalid partitions, shared boundaries, explicit totals, zero-energy persistence, guaranteed and impossible formation, fertility mapping, formation locality, amount capping, proportional decay, full dissolution, new-point timing, local return, no diffusion, immutable prior state, multi-region and multi-point transitions, deterministic event order, insertion-order independence, multi-tick reproducibility, region-stream isolation, unrelated-stream isolation, a 250-tick mixed conservation run, and direct conservation-failure protection. All previous acceleration, persistent-velocity, shared-position, lockstep and boundary-death tests remain green.

The headless CLI was smoke-tested:

```powershell
uv run noralet run --ticks 7 --seed 20260821
```

Result: successful exit with `Completed 7 tick(s); final tick: 7; seed: 20260821`.

Additional validation commands:

```powershell
uv run python -m compileall -q src tests
uv run python -c "from noralet import ..."
uv lock --check
git diff --check
rg -n '[ \t]+$' src tests codex-reports/operation-report-003.md
```

Results:

- source and tests compiled successfully;
- all Iteration 3 public API imports succeeded;
- the uv lockfile was current;
- `git diff --check` reported no whitespace errors;
- the explicit trailing-whitespace scan found no source, test or report matches.

The project has no configured formatter, linter or static type checker, so no new tool was introduced solely for this iteration. A scope scan and manual diff review found no Noralet-energy, consumption, cost, sensing, signalling, neural, rendering, replay, persistence or environmental-diffusion implementation.

The architecture audit confirmed all requested invariants: a closed universe; transfer-only formation and decay; region locality; no environmental diffusion; immutable prior state; delayed first decay for new points; explicit deterministic named randomness; order-independent state and event history; unchanged Iteration 2 physics; and no Noralet Energy.

## Deviations

There were no deviations from the Iteration 3 instruction or the current architecture documentation.

## Open implementation notes

- `SimulationConfig.energy_ecology=None` preserves the Iteration 1/2 constructor and existing headless CLI as an ecology-free zero-energy compatibility mode. It is not an experimental energy default. Ecology experiments must provide explicit regions, local initial pools and probabilities through the Python API.
- The removal threshold must be finite, non-negative and strictly below `formation_energy_min` so a newly formed valid point is not immediately in the dissolution range.
- Static region definitions remain in immutable run configuration; dynamic region-local pools and consumable points live in immutable `WorldState` values.

## Git state

No commit or push was created.

The working tree was clean before Iteration 3 began. This iteration modifies:

- `src/noralet/__init__.py`
- `src/noralet/simulation/__init__.py`
- `src/noralet/simulation/config.py`
- `src/noralet/simulation/events.py`
- `src/noralet/simulation/runtime.py`
- `src/noralet/simulation/state.py`

It adds:

- `src/noralet/world/__init__.py`
- `src/noralet/world/energy.py`
- `src/noralet/world/regions.py`
- `tests/energy_test_support.py`
- `tests/test_energy_validation.py`
- `tests/test_energy_ecology.py`
- `tests/test_energy_determinism.py`
- `codex-reports/operation-report-003.md`

All changes are scoped to Iteration 3 implementation, tests and reporting.
