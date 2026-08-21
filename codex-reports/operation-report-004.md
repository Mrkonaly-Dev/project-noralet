# Operation Report 004 — Noralet Energy, Consumption and Expenditure

**Iteration:** 4
**Date:** 2026-08-21
**Status:** Complete

## Summary

Iteration 4 introduced the third fundamental energy storage form, Noralet Energy, and completed the closed transfer cycle between region-local Environmental Energy, stationary Consumable Energy points and living Noralets.

Energy-enabled Noralets now have finite stored energy and shared configurable capacity. They can explicitly attempt local consumption, pay existence and applied-acceleration costs, receive only affordable acceleration, die deterministically from depletion, and return remaining stored energy when crossing a world boundary. Consumable points now obey configurable global minimum spacing, including symmetric same-tick formation-candidate conflict resolution.

The runtime audits all three energy forms before every publication. Energy-disabled Iteration 1–3 simulations continue through their prior motion/ecology path unchanged.

No condition, ageing, natural mortality, perception, signalling, neural, rendering or persistence system was added.

## Files changed

- `src/noralet/noralets/energy.py` — adds immutable `NoraletEnergyConfig` for capacity, existence cost, acceleration cost and consume radius.
- `src/noralet/noralets/body.py` — adds finite non-negative stored energy to immutable `NoraletBodyState`.
- `src/noralet/noralets/actions.py` — adds the explicit boolean `consume` attempt to `ActionIntent`.
- `src/noralet/noralets/__init__.py` — exposes the implemented Noralet energy configuration.
- `src/noralet/world/energy.py` — adds Noralet Energy to observer totals and configurable global minimum point spacing to the ecology.
- `src/noralet/simulation/config.py` — adds the optional Noralet-energy boundary and validates its required ecology and strict spacing/radius relation.
- `src/noralet/simulation/state.py` — includes living-body energy in carefully summed world totals.
- `src/noralet/simulation/events.py` — adds consumption, expenditure and death-release events plus energy-depletion death and machine-readable expenditure reasons.
- `src/noralet/simulation/runtime.py` — implements fair consumption, costs, affordable acceleration, energy-aware deaths, death release, spacing-aware formation and three-form conservation.
- `src/noralet/simulation/__init__.py` and `src/noralet/__init__.py` — expose the Iteration 4 public API.
- `tests/energy_test_support.py` — extends explicit test ecology construction with minimum spacing.
- `tests/noralet_energy_test_support.py` — provides compact explicit energy-enabled simulation construction for tests.
- `tests/test_noralet_energy_validation.py` — covers configuration, body energy, action, spacing, capacity and initial totals.
- `tests/test_consumption.py` — covers explicit targeting, tick-start reach, fair allocation, capacity redistribution, overflow and consumption/decay order.
- `tests/test_noralet_expenditure_death.py` — covers costs, affordable motion, local returns, depletion, boundary release, precedence and event phases.
- `tests/test_iteration4_determinism.py` — covers formation spacing/conflicts, no resampling, reproducibility, RNG isolation and long-run complete conservation.
- `codex-reports/operation-report-004.md` — records this implementation iteration.

## Noralet Energy model

`NoraletBodyState` remains a frozen observer-visible body value and now contains only:

```text
noralet_id
position
velocity
energy
```

Stored energy is canonicalised to a finite float and must be non-negative. In an energy-enabled simulation it must also not exceed the single configured `energy_capacity`. Capacity is finite, strictly positive, shared by all Noralets, and is not itself energy.

`NoraletEnergyConfig` contains exactly:

```text
energy_capacity
existence_energy_cost_per_tick
acceleration_energy_cost_per_unit
consume_radius
```

The costs must be finite and non-negative; the radius must be finite and positive. Activating this configuration requires an active `EnergyEcologyConfig`, because every expenditure and death release needs a destination inside the same closed energy universe.

Body energy defaults to zero for source compatibility. With `SimulationConfig.noralet_energy=None`, the runtime uses the preserved Iteration 1–3 transition path: zero-energy bodies do not pay costs or die from depletion, and motion remains unchanged. Supplying positive body energy without `NoraletEnergyConfig`, or initial energy above configured capacity, is rejected.

An energy-enabled body may validly begin at zero energy. It still receives its already-selected transition and can survive if explicit consumption restores energy before expenditure; otherwise it cannot enter the next world state.

## Consumption

`ActionIntent.consume` is an immutable boolean with default `False`. Consumption therefore never occurs merely because a body is near a point.

For `consume=True`, the resolver reads only `WorldState(t)` body and point positions. A point is accessible when:

```text
abs(point.position - body.position) <= consume_radius
```

The inclusive radius is fixed consistently. Moving into range during the current transition cannot enable consumption until the next tick. If a malformed state ever exposes several accessible points, the deterministic robustness rule selects smallest distance and then lowest stable point ID.

All Noralets targeting the same point are resolved simultaneously. The runtime performs deterministic equal-share water filling against each consumer's remaining capacity. Consumers unable to accept an equal share saturate at capacity, and their unused share is repeatedly redistributed equally among consumers with room.

Every positive allocation transfers exactly one observer-visible amount from the point to the receiving Noralet. If total capacity is insufficient, every accepted receiver fills appropriately and the unallocated energy remains in the original point. Fully consumed points are removed before decay. A partially consumed point remains stationary and its post-consumption remainder undergoes normal same-tick decay.

## Energy-point spacing

`EnergyEcologyConfig.minimum_energy_point_spacing` is finite and non-negative. Every pair of simultaneously existing points must satisfy:

```text
distance >= minimum_energy_point_spacing
```

The rule is global across region boundaries. Explicit initial points are sorted by position and identity before adjacent-distance validation, making acceptance independent of caller insertion order.

Energy-enabled configurations additionally require the strict clarity invariant:

```text
minimum_energy_point_spacing > 2 * consume_radius
```

This prevents two valid points from being simultaneously accessible to one Noralet.

Formation retains at most one attempt per region per tick. A triggered and funded region draws one amount and one position candidate, but does not transfer energy yet. A candidate closer than the minimum to any post-consumption, post-decay surviving point is rejected in place without movement, clamping, search or resampling.

All regions generate candidates before candidate-to-candidate resolution. Every candidate participating in a same-tick distance violation is rejected; no left-to-right or collection-order winner exists. Only accepted candidates receive deterministic point IDs and transfer their exact amount from the producing region's Environmental Energy pool.

The spacing default is zero for energy-disabled Iteration 3 compatibility. Explicit Iteration 4 configurations must opt into a value satisfying the strict consume-radius relation.

## Expenditure

After consumption, every body that existed in `WorldState(t)` pays up to `existence_energy_cost_per_tick`. If it stores less than the configured cost, it transfers only what it owns and reaches zero without becoming negative. The transfer goes to the region containing its `WorldState(t)` position, even if it later moves or dies.

Acceleration cost uses the configured linear model:

```text
requested_cost
=
acceleration_energy_cost_per_unit
*
abs(requested_acceleration)
```

If affordable, the full requested acceleration is applied and charged. Otherwise, with a positive coefficient:

```text
abs(applied_acceleration)
=
available_energy / acceleration_energy_cost_per_unit
```

using the requested sign. Only the represented applied acceleration is charged, and the actual transfer goes to the body's `WorldState(t)` region. Stored energy never becomes negative.

A zero coefficient preserves fully free requested acceleration without division. Zero requested acceleration and persistent-velocity coasting incur no acceleration cost. The semi-implicit Iteration 2 equations remain:

```text
v(t+1) = v(t) + applied_a(t)
x(t+1) = x(t) + v(t+1)
```

## Death

After consumption, both expenditure phases and movement, an in-bounds Noralet with zero stored energy receives `NoraletDeathCause.ENERGY_DEPLETION` and is absent from `WorldState(t+1)`. It may still have terminal acceleration and movement events because it existed at `WorldState(t)`.

Boundary semantics remain inclusive and unchanged. A resolved position strictly outside the world receives `WORLD_BOUNDARY`, with boundary death taking precedence if depletion also occurred.

Any positive stored energy remaining on death is transferred to Environmental Energy. A left crossing releases into the leftmost traversable region; a right crossing releases into the rightmost region. The transfer is exposed as `NoraletEnergyReleased`, no corpse retains energy, and the dead body is omitted from the successor state. Energy-depletion deaths normally have zero remainder and therefore no release event.

## Tick order

For one `t -> t+1` transition with Noralet Energy enabled, the implemented order is:

1. validate and freeze all action intentions;
2. determine consume targets from `WorldState(t)` positions;
3. resolve all same-point consumers with fair capacity-limited allocation;
4. transfer existence expenditure from every `WorldState(t)` body to its tick-start region;
5. calculate affordable applied acceleration;
6. transfer applied-acceleration expenditure to each tick-start region;
7. resolve every velocity and position with the Iteration 2 equations;
8. identify boundary deaths;
9. identify in-bounds energy-depletion deaths;
10. transfer positive remaining energy from dead bodies to the appropriate environment;
11. decay and dissolve post-consumption remainders of points that existed at `t`;
12. generate every region's single possible formation candidate;
13. reject candidates conflicting with surviving points;
14. reject all candidates participating in candidate-to-candidate spacing conflicts;
15. transfer Environmental to Consumable Energy only for accepted candidates;
16. construct the complete immutable `WorldState(t+1)`;
17. audit all three energy forms against the initial baseline;
18. construct the observer result and publish only the valid state.

No phase mutates or publishes `WorldState(t)`. New formations are not eligible for decay until the next transition.

## Conservation

The complete implemented law is:

```text
E_total
=
E_environmental
+
E_consumable
+
E_noralets
=
constant
```

`WorldState.energy_totals` exposes separate environmental, consumable and living-Noralet totals plus their combined total. Each category and the final result use `math.fsum` where aggregation is required.

`Simulation.initial_total_energy` is established from explicit initial region pools, points and body energy. Capacity is excluded. Before publication, `audit_energy_conservation` recomputes the three-form total and compares it to that immutable baseline with zero relative tolerance and the existing absolute tolerance of `1e-9 eU`, solely for binary floating-point representation effects.

Non-finite or out-of-tolerance results raise `EnergyConservationError` with expected, observed and difference values. The runtime does not repair totals and does not publish the invalid successor state.

Consumption, existence cost, acceleration cost, death release, point decay/dissolution and accepted formation are all source-to-destination transfers. Rejected formation candidates transfer nothing.

## Determinism

Bodies, points and regions retain their existing canonical orders. Consumption events are ordered by point ID and then Noralet ID. Water-filling calculations use the complete consumer set and stable identities, so body/action insertion order cannot produce a priority advantage.

Initial spacing checks sort by position and identity. Formation candidates are generated in canonical spatial region order, checked as a complete set, and all participants in any same-tick conflict lose symmetrically. Reversing input region, pool, point, body and action order therefore produces identical states and histories.

Consumption, expenditure, affordability and death use no randomness. Formation continues using the Iteration 3 region-and-purpose-specific trigger, amount and position streams. Spacing rejection does not resample. Existing unrelated streams such as `world` remain unshifted.

## Events

Iteration 4 adds immutable observer-only types:

- `EnergyConsumed` — Noralet ID, point ID, positive transferred amount and tick transition;
- `NoraletEnergySpent` — Noralet ID, destination region, positive amount, tick transition and machine-readable `EXISTENCE` or `ACCELERATION` reason;
- `NoraletEnergyReleased` — dead Noralet ID, destination region, positive returned amount and tick transition;
- `NoraletDeathCause.ENERGY_DEPLETION` — deterministic in-bounds depletion cause on the existing `NoraletDied` event.

Zero-transfer attempts and costs emit no transfer event. Events remain descriptive and do not drive mechanics.

The exact energy-enabled phase ordering is:

1. `EnergyConsumed` by point ID, then Noralet ID;
2. existence `NoraletEnergySpent` by Noralet ID;
3. acceleration `NoraletEnergySpent` by Noralet ID;
4. `NoraletAccelerated` by Noralet ID;
5. `NoraletMoved` by Noralet ID;
6. `NoraletDied` by Noralet ID;
7. `NoraletEnergyReleased` by Noralet ID;
8. `EnergyPointDecayed` by point ID;
9. `EnergyPointDissolved` by point ID;
10. `EnergyPointFormed` by canonical spatial region order;
11. `TickAdvanced`.

When Noralet Energy is disabled, the existing Iteration 3 physical/ecology event order remains unchanged.

## Tests and validation

The clean pre-Iteration 4 baseline passed all 77 existing Iteration 1–3 tests.

The final suite contains 120 tests: all 77 existing tests plus 43 focused Iteration 4 tests. The complete final suite passed three times:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run python -m unittest discover -s tests -v
```

Results: 120 of 120 tests passed. The three final runs completed in `0.452s`, `0.528s` and `0.512s`.

Coverage includes all requested configuration failures, complete initial totals, explicit and out-of-range consumption, inclusive reach, tick-start targeting, nearest/ID selection, global spacing, initial spacing, blocked formation, no resampling, symmetric candidate conflicts, fair equal shares, water-filling redistribution, capacity overflow retention, consumer-order independence, existence cost, insufficient energy, consume-before-cost survival, free coasting, linear acceleration cost, partial affordability, free coefficient, local return, deterministic depletion, terminal movement, both boundary releases, cause precedence, consumption-before-decay, full-consumption decay removal, event order, insertion-order determinism, audit failure protection, RNG isolation, generated-point spacing, an 80-tick replicated mixed history and a 300-tick mixed conservation run.

A separate deterministic stress script ran 30 generated energy-enabled worlds for 200 ticks each, with five initial Noralets, multiple points, random-but-seeded actions, consumption, expenditure, movement, deaths, formation, decay and dissolution. All 6,000 transitions passed the runtime conservation audit.

The existing headless CLI was smoke-tested:

```powershell
uv run noralet run --ticks 7 --seed 20260821
```

Result: successful exit with `Completed 7 tick(s); final tick: 7; seed: 20260821`.

Additional validation:

```powershell
uv run python -m compileall -q src tests
uv run python -c "from noralet import ..."
uv lock --check
git diff --check
rg -n '[ \t]+$' src tests codex-reports/operation-report-004.md
```

Results:

- source and tests compiled successfully;
- all Iteration 4 public API imports succeeded;
- the uv lockfile was current;
- `git diff --check` reported no errors;
- the explicit trailing-whitespace scan found no source, test or report matches.

The repository has no configured formatter, linter or static type checker, so no new validation tool was introduced. Manual scope and diff inspection found no unrelated implementation or out-of-scope condition, ageing, perception, signalling, neural, collision, diffusion, rendering, replay or persistence system.

The architecture audit confirmed all fourteen requested invariants: constant three-form total; transfer-only changes; explicit tick-start consumption; strict spacing clarity; fair simultaneous allocation; bounded non-negative body energy; free coasting; applied-only acceleration cost; local expenditure return; boundary-release conservation; deterministic depletion; unchanged movement equations; and no premature bodily, perceptual or neural architecture.

## Deviations

There were no deviations from the Iteration 4 instruction or the current architecture documentation.

## Open implementation notes

- `minimum_energy_point_spacing=0.0` remains available only as an Iteration 3 compatibility default. Activating `NoraletEnergyConfig` requires an explicitly compatible value strictly greater than twice the consume radius.
- Energy-disabled simulations accept the extended zero-energy body and default `consume=False` action representation but resolve through the preserved Iteration 1–3 mechanics. Positive body energy cannot enter that compatibility mode.
- Energy-enabled bodies starting at zero are not rejected during construction because the fixed transition order intentionally gives an already-selected consume action its opportunity before depletion is evaluated.

## Git state

No commit or push was created.

The working tree was clean before Iteration 4 began. This iteration modifies:

- `src/noralet/__init__.py`
- `src/noralet/noralets/__init__.py`
- `src/noralet/noralets/actions.py`
- `src/noralet/noralets/body.py`
- `src/noralet/simulation/__init__.py`
- `src/noralet/simulation/config.py`
- `src/noralet/simulation/events.py`
- `src/noralet/simulation/runtime.py`
- `src/noralet/simulation/state.py`
- `src/noralet/world/energy.py`
- `tests/energy_test_support.py`

It adds:

- `src/noralet/noralets/energy.py`
- `tests/noralet_energy_test_support.py`
- `tests/test_noralet_energy_validation.py`
- `tests/test_consumption.py`
- `tests/test_noralet_expenditure_death.py`
- `tests/test_iteration4_determinism.py`
- `codex-reports/operation-report-004.md`

Architecture and research documentation were read but not modified. All changes are scoped to Iteration 4 implementation, tests and reporting.
