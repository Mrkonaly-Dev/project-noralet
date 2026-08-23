# Operation Report 005 — Condition, Ageing and Natural Mortality

**Iteration:** 5
**Date:** 2026-08-23
**Status:** Complete

## Summary

Iteration 5 introduced objective Noralet age, persistent irreversible physiological condition, nonlinear low-energy deprivation wear, slow baseline ageing wear, and state-driven probabilistic natural mortality.

Natural mortality uses candidate updated age and condition, then one deterministic draw from an isolated per-Noralet stream. Boundary and energy-depletion deaths retain precedence. A naturally dying Noralet is removed immediately and returns all remaining stored energy to the region containing its resolved in-bounds position.

The existing lockstep motion, consumption, expenditure, ecology, immutable-state and three-form energy-conservation semantics remain intact. No condition recovery, health-bar state, subjective physiology, perception, signalling or neural system was introduced.

## Files changed

- `src/noralet/noralets/physiology.py` — adds immutable physiology configuration plus pure condition, mortality-hazard and natural-death-probability calculations.
- `src/noralet/noralets/body.py` — adds validated persistent `age_ticks` and `condition` values to the immutable body state.
- `src/noralet/noralets/__init__.py` — exposes the implemented physiology domain API.
- `src/noralet/simulation/config.py` — adds optional active physiology configuration and requires active Noralet Energy when it is enabled.
- `src/noralet/simulation/events.py` — adds machine-readable `NoraletDeathCause.NATURAL` to the existing death event family.
- `src/noralet/simulation/runtime.py` — adds candidate ageing, condition wear, isolated mortality draws, natural-death removal and local energy return in the canonical transition.
- `src/noralet/simulation/__init__.py` and `src/noralet/__init__.py` — expose the Iteration 5 public API.
- `tests/noralet_energy_test_support.py` — permits explicit physiology configuration in the existing energy-enabled test constructor.
- `tests/physiology_test_support.py` — provides compact explicit physiology test configuration.
- `tests/test_physiology_validation.py` — covers body and configuration validation, immutability and energy dependency.
- `tests/test_condition_ageing.py` — covers age advancement, baseline wear, nonlinear deprivation, accumulated history, monotonic condition and no recovery.
- `tests/test_natural_mortality.py` — covers the pure equations, extreme ages, controlled death/survival, precedence, event phases and local death release.
- `tests/test_mortality_determinism.py` — covers stream isolation, ordering, reproducibility, life-history separation, age separation and long-run mixed conservation.
- `codex-reports/operation-report-005.md` — records this implementation iteration.

## Age model

`NoraletBodyState.age_ticks` is a non-negative Python integer measured only in canonical simulation ticks. Callers may supply any explicit valid initial age; compatibility construction defaults it to `0`. Negative values, booleans and non-integers are rejected rather than clamped or converted.

When `NoraletPhysiologyConfig` is active, a body that survives boundary and energy-depletion resolution receives:

```text
candidate_age = current_age_ticks + 1
```

Natural mortality uses that candidate age. A naturally dying body therefore experienced the transition's age advancement for its mortality calculation even though no persistent corpse or death registry is created. A surviving body enters `WorldState(t+1)` with exactly that candidate age.

The optional physiology boundary preserves Iteration 1–4 compatibility: without active physiology, explicit age and condition values are retained unchanged and no mortality draw occurs. Active physiology requires the already configured closed Noralet-energy universe.

There are no years, days, life-stage labels, maximum-age rule or predetermined death tick.

## Condition model

`NoraletBodyState.condition` is a finite objective float in:

```text
0.0 <= condition <= 1.0
```

It defaults to `1.0` for compatibility and is observer-visible immutable world state, not a subjective Noralet input. Invalid initial values are rejected without clamping.

For each physiologically eligible survivor, the runtime reads the stored energy remaining after consumption, existence expenditure and acceleration expenditure:

```text
energy_ratio = stored_energy / energy_capacity

deprivation = max(
    0,
    (low_energy_condition_threshold_ratio - energy_ratio)
    / low_energy_condition_threshold_ratio
)

deprivation_condition_loss =
    deprivation_condition_loss_scale
    * deprivation ** deprivation_exponent

condition_loss =
    baseline_condition_loss_per_tick
    + deprivation_condition_loss

candidate_condition = max(
    0,
    current_condition - condition_loss
)
```

At or above the configured threshold, deprivation is zero and only baseline ageing wear applies. Below the threshold, normalized deprivation grows from `0` to `1`; the configurable exponent, required to be at least `1`, permits severe deprivation to matter disproportionately more than mild deprivation.

Condition never increases. Safe energy can prevent new deprivation-derived loss but cannot restore prior damage. Condition loss transfers no energy and does not maintain a separate starvation-history record: the persistent condition value itself carries accumulated life history. No generic per-tick condition event is emitted.

## Natural mortality model

The pure calculation implements:

```text
condition_deficit = 1 - condition

age_pressure =
(age / mortality_age_scale)
** mortality_age_exponent

condition_pressure =
condition_deficit
** mortality_condition_exponent

hazard =
    base_mortality_hazard
    + age_hazard_scale * age_pressure
    + condition_hazard_scale * condition_pressure
    + interaction_hazard_scale
      * age_pressure
      * condition_pressure

p_natural_death =
1 - exp(-hazard)
```

The age exponent must exceed `1`; the condition exponent must be at least `1`; every scale is finite and non-negative. The mortality age scale is finite and strictly positive. The base hazard may be zero for controlled experiments or a very small positive value for residual young/good-condition risk.

The age term, condition term and explicit multiplicative interaction allow strong state separation. Old age and poor condition can each raise risk, while their combination can be substantially more dangerous than either alone. Randomness only selects the exact death moment relative to this state-derived probability.

For numerical safety at extreme valid integer ages, age pressure is evaluated through the mathematically equivalent logarithmic form and overflowing hazard terms saturate at the largest finite float. Probability uses `-expm1(-hazard)`, which is equivalent to the required transform but accurate for very small hazards. If floating-point underflow would round the result to exactly `1.0`, it is represented as the largest float below `1.0`. Tests exercise an age with 10,001 decimal digits and confirm finite hazard, no exception or NaN, and `0 <= p < 1`.

## Mortality randomness

Every boundary/depletion survivor receives exactly one mortality draw per active-physiology tick from:

```text
mortality:noralet:{decimal-id-length}:{decimal-id}
```

The length-prefixed decimal ID makes the name stable and unambiguous. The existing simulation-owned `DeterministicRandomStreams` derives the stream seed with domain-separated SHA-256 from the master seed and exact name.

Each Noralet owns an independent cached `random.Random` stream. Additional draws or death for Noralet A cannot advance Noralet B's stream. Body insertion order, population processing count and unrelated ecology streams therefore cannot shift an existing Noralet's mortality sequence. Same seed, configuration, initial state and actions reproduce the same natural-death history.

Boundary and depletion deaths receive no mortality draw.

## Death resolution

The implemented precedence is:

```text
WORLD_BOUNDARY
>
ENERGY_DEPLETION
>
NATURAL
```

Boundary resolution occurs first, including simultaneous boundary/depletion cases. In-bounds zero-energy bodies then receive `ENERGY_DEPLETION`. Their positive remainders, if any, are returned using the existing release mechanism before natural mortality is evaluated for the remaining bodies.

A naturally dying body emits the existing `NoraletDied` with cause `NATURAL`, followed by `NoraletEnergyReleased` when its remainder is positive. The destination is the region owning its resolved in-bounds `t+1` position. The body is absent from `WorldState(t+1)`, and no corpse, death registry or fourth energy holder exists.

The observer event phases remain deterministic:

1. `EnergyConsumed` by point ID and Noralet ID;
2. existence `NoraletEnergySpent` by Noralet ID;
3. acceleration `NoraletEnergySpent` by Noralet ID;
4. `NoraletAccelerated` by Noralet ID;
5. `NoraletMoved` by Noralet ID;
6. boundary/depletion `NoraletDied` by Noralet ID;
7. their `NoraletEnergyReleased` events by Noralet ID;
8. natural `NoraletDied` by Noralet ID;
9. natural-death `NoraletEnergyReleased` by Noralet ID;
10. point decay, dissolution and formation phases;
11. `TickAdvanced`.

## Tick order

For active physiology, the exact `t -> t+1` transition is:

1. validate and freeze action intentions;
2. determine consumption targets from `WorldState(t)` positions;
3. resolve fair simultaneous capacity-limited consumption;
4. apply existence expenditure at tick-start regions;
5. calculate affordable acceleration;
6. transfer applied-acceleration expenditure at tick-start regions;
7. resolve velocity and position with the existing semi-implicit equations;
8. identify `WORLD_BOUNDARY` deaths;
9. identify in-bounds `ENERGY_DEPLETION` deaths;
10. return positive remaining energy from those deaths;
11. assign `candidate_age = age_ticks + 1` to remaining bodies;
12. calculate post-energy normalized deprivation;
13. calculate candidate condition from baseline and nonlinear deprivation wear;
14. calculate mortality hazard from candidate age and condition;
15. draw once from each eligible per-Noralet mortality stream;
16. identify `NATURAL` deaths;
17. return their complete remaining stored energy at resolved in-bounds positions;
18. decay existing post-consumption energy-point remainders;
19. dissolve points at or below the removal threshold;
20. generate region formation candidates;
21. resolve all existing-point and same-tick candidate spacing conflicts;
22. transfer Environmental Energy into accepted formations;
23. construct immutable `WorldState(t+1)` with only survivors and their candidate physiology;
24. audit complete universe energy against the initial baseline;
25. construct the observer result and publish the validated state.

No phase mutates `WorldState(t)`.

## Energy conservation

Age and condition are objective physical descriptors, not energy stores. Condition degradation and age advancement create or consume no energy. The universe law remains exactly:

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

Natural death is a complete Noralet-to-Environmental transfer. The runtime's existing pre-publication audit remains unchanged and includes only these three fundamental forms.

The new 300-tick mixed test exercises movement, consumption, existence and acceleration costs, ageing, condition degradation, natural mortality, depletion death, point formation, decay and dissolution. Every published state passes the runtime conservation audit, and the final total remains within the existing absolute tolerance of `1e-9 eU`.

## Tests and validation

The clean pre-Iteration 5 baseline passed all 120 existing Iteration 1–4 tests.

Iteration 5 adds 31 focused tests, bringing the complete suite to 151 tests. The final required command was run from the repository root:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: 151 of 151 tests passed in `0.618s`.

The 16 mortality, deterministic population and mixed-conservation tests were then repeated twice independently:

```powershell
uv run python -m unittest -v test_mortality_determinism test_natural_mortality
```

Results: 16 of 16 passed in `0.131s`, then 16 of 16 passed in `0.120s`.

The fixed-seed energy-regime experiment used 64 otherwise equivalent per-ID mortality streams in each group. At tick 15, well-maintained condition was `1.000` while chronically deprived condition was `0.808`. By tick 120, the well-maintained group had `0/64` deaths; the deprived group had `64/64` natural deaths, with mean death tick `40.6875`.

The fixed-seed age-regime experiment used the same 64 IDs, seed, condition and energy in both groups. After 20 ticks, the initially young group had `0/64` deaths while the initially age-1000 group had `60/64` deaths. Both controlled experiments use deliberately separated configurations and exact deterministic assertions rather than uncontrolled statistical tolerances.

Additional validation:

```powershell
uv run python -m compileall -q src tests
uv lock --check
uv run noralet run --ticks 7 --seed 20260823
uv run python -c "... Iteration 5 public API smoke ..."
git diff --check
rg -n '[ \t]+$' src tests
```

Results:

- source and tests compiled successfully;
- the uv lockfile was current;
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260823`;
- the Iteration 5 public imports and pure calculations succeeded;
- Git diff and trailing-whitespace checks passed.

The project has no configured formatter, linter or static type checker, so no additional validation dependency was introduced.

The final architecture audit confirmed all requested invariants: monotonic active age, irreversible condition, accumulated deprivation history, no recovery, nonlinear deprivation, no fixed lifespan, state-driven interacting hazard, residual per-body randomness, stream isolation, death precedence, local natural-death return, unchanged three-form conservation, objective-only physiology and no neural architecture.

## Deviations

There were no deviations from the Iteration 5 instruction or the current architecture documentation.

## Open implementation notes

- `NoraletPhysiologyConfig` is optional solely to preserve validated Iteration 1–4 compatibility paths. Explicit age and condition remain unchanged in that mode. Experiments enabling physiology must also enable Noralet Energy and its valid Energy Ecology.
- Candidate age and condition for a naturally dying Noralet are used in mortality resolution but are not retained in a corpse or added to `NoraletDied`; the existing minimal event remains sufficient for the requested observability.
- Very large ages use finite numerical saturation only after the configured equation exceeds representable floating-point range. This prevents overflow and NaN without introducing a maximum biological age or a forced-death branch.
- No condition-change event was added. Observer and debug tools can inspect age and condition directly from immutable living-body state.

## Git state

No commit or push was created.

The working tree was clean before Iteration 5 began. This iteration modifies:

- `src/noralet/__init__.py`
- `src/noralet/noralets/__init__.py`
- `src/noralet/noralets/body.py`
- `src/noralet/simulation/__init__.py`
- `src/noralet/simulation/config.py`
- `src/noralet/simulation/events.py`
- `src/noralet/simulation/runtime.py`
- `tests/noralet_energy_test_support.py`

It adds:

- `src/noralet/noralets/physiology.py`
- `tests/physiology_test_support.py`
- `tests/test_condition_ageing.py`
- `tests/test_mortality_determinism.py`
- `tests/test_natural_mortality.py`
- `tests/test_physiology_validation.py`
- `codex-reports/operation-report-005.md`

Architecture and research documentation were read but not modified. All changes are scoped to Iteration 5 implementation, tests and reporting.
