# Operation Report 002 — Noralet Bodies and 1D Motion

**Iteration:** 2
**Date:** 2026-08-21
**Status:** Complete

## Summary

Iteration 2 introduced a finite continuous one-dimensional world, immutable living Noralet bodies, externally supplied acceleration intentions, persistent velocity, semi-implicit Euler position updates, boundary-crossing death, and deterministic lockstep physical events.

The Iteration 1 clock, immutable transition, deterministic random-stream and headless CLI guarantees remain intact. No energy, region, perception, signal, collision, neural, renderer, persistence or replay systems were added.

## Files changed

- `src/noralet/simulation/config.py` — adds validated finite left and right world boundaries.
- `src/noralet/noralets/body.py` — adds immutable `NoraletBodyState` with simulation ID, position and velocity.
- `src/noralet/noralets/actions.py` — adds immutable acceleration-only `ActionIntent`.
- `src/noralet/noralets/__init__.py` — exposes the implemented Noralet value objects.
- `src/noralet/simulation/state.py` — stores living bodies in a canonical immutable tuple and provides ID lookup.
- `src/noralet/simulation/runtime.py` — accepts external intents and resolves all body motion, boundary deaths and events in lockstep.
- `src/noralet/simulation/events.py` — adds `NoraletAccelerated`, `NoraletMoved`, `NoraletDied` and the machine-readable `WORLD_BOUNDARY` cause.
- `src/noralet/simulation/__init__.py` and `src/noralet/__init__.py` — extend the small public API with the Iteration 2 types.
- `src/noralet/cli.py` — updates the help text while preserving the existing headless command and arguments.
- `tests/test_world_validation.py` — covers boundaries, initial bodies, identity uniqueness, finite values and initial placement.
- `tests/test_action_intents.py` — covers intent immutability and malformed, unknown or dead targets.
- `tests/test_motion.py` — covers integration order, persistent velocity, braking, reversal, lockstep multi-body motion, non-collision, boundaries, event ordering, immutability and multi-tick determinism.
- `codex-reports/operation-report-002.md` — records this implementation iteration.

## World representation

`SimulationConfig` contains the finite `left_boundary` and `right_boundary` and requires `left_boundary < right_boundary`. Non-numeric, non-finite, equal and reversed boundary values are rejected.

`NoraletBodyState` is a frozen value containing only:

```text
noralet_id
position
velocity
```

The integer `noralet_id` is simulation identity only; no perceptual identity system exists. Position and velocity must be finite and are stored as canonical Python floats.

`WorldState` contains the tick and an immutable tuple of currently living bodies. It rejects duplicate identities and non-body values, then canonicalises the tuple by ascending `noralet_id`. This gives equivalent initial collections identical state ordering regardless of caller insertion order.

`Simulation(config, initial_bodies=...)` accepts explicitly supplied bodies, begins at tick `0`, and rejects any initial position outside the inclusive configured interval. No population or physical value is randomly generated.

## Action-intent model

`Simulation.step(action_intents=...)` accepts a mapping from integer Noralet ID to frozen `ActionIntent`. The intent contains only finite one-dimensional acceleration. Constructing or supplying an intent does not alter a body; validation and all effects occur inside the later resolution phase.

A missing mapping entry means `acceleration = 0.0`. A `None` mapping therefore means zero acceleration for every living body.

Unknown and dead target IDs are rejected before world publication. Non-mapping collections, non-integer targets, non-`ActionIntent` values and non-finite acceleration are also rejected. A mapping inherently has at most one value per key, so the chosen input representation cannot contain duplicate conflicting intents for one ID.

## Motion resolution

For every living body, the runtime applies exactly:

```text
v(t+1) = v(t) + a(t)
x(t+1) = x(t) + v(t+1)
```

Resolution first snapshots and validates the complete intent mapping. It then calculates all accelerations from `WorldState(t)`, all updated velocities, all updated positions, boundary status, and finally the surviving `NoraletBodyState` values for `WorldState(t+1)`.

Zero acceleration leaves velocity unchanged. There is no friction, drag, passive braking, acceleration limit, energy cost, external force or inter-body collision. Opposing acceleration follows the same equation and may reduce, stop or reverse velocity.

Non-finite arithmetic results are rejected before the new state is published rather than allowing invalid physical values into the world.

## Boundary death

The configured interval is inclusive:

```text
left_boundary <= position <= right_boundary
```

A resolved position strictly outside that interval produces a `NoraletDied` event with cause `NoraletDeathCause.WORLD_BOUNDARY` and the actual out-of-bounds resolved position. The body is omitted from `WorldState(t+1)`; it is not clamped, bounced, stopped or retained as a corpse.

If the body moved before leaving the world, its terminal `NoraletMoved` event remains observable before the death event.

## Lockstep behaviour

The runtime never mutates bodies from `WorldState(t)`. It derives separate acceleration, velocity and position tables for every living body from the same prior state before constructing any successor body.

Body tuples are canonicalised by ascending simulation ID and intent values are looked up by ID rather than processed according to mapping order. Physical events use a deterministic phase-first order:

1. non-zero `NoraletAccelerated` events by ascending ID;
2. actual `NoraletMoved` events by ascending ID;
3. `NoraletDied` events by ascending ID;
4. the transition's `TickAdvanced` event.

Insertion order therefore cannot change the resulting world state or event history. Noralets may occupy the same coordinate and cross through one another without collision effects.

## Determinism

The Iteration 1 named random-stream implementation is unchanged. Iteration 2 physics consumes no random values.

With the same code, configuration, explicit initial bodies, action-intent sequence, master seed and compatible Python runtime, the complete world and event history is reproducible. Tests compare equivalent simulations across reversed body order, reversed intent-map order, multiple Noralets, multiple ticks and boundary deaths.

## Tests and validation

The pre-change Iteration 1 baseline was run through the project environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run python -m unittest discover -s tests -v
```

Result before implementation: 13 tests passed.

The final complete suite was run repeatedly:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: 40 tests passed, including all 13 Iteration 1 tests and 27 Iteration 2 tests. The final implementation passed the complete suite three times consecutively; an earlier complete Iteration 2 run also passed before the final cleanup.

The existing packaged CLI was smoke-tested:

```powershell
uv run noralet run --ticks 7 --seed 20260821
```

Result: successful exit with `Completed 7 tick(s); final tick: 7; seed: 20260821`.

Additional validation:

```powershell
uv run python -m compileall -q src tests
uv lock --check
git diff --check
rg -n '[ \t]+$' src tests
```

Results:

- source and tests compiled successfully;
- the uv lockfile was current;
- `git diff --check` reported no errors;
- the explicit trailing-whitespace scan found no matches.

A public-package API smoke check also constructed a body, submitted acceleration, and observed the expected `WorldState`, `NoraletAccelerated`, `NoraletMoved` and `TickAdvanced` values.

The repository has no configured formatter, linter or static type checker, so no additional tool was introduced solely for this iteration.

The architecture audit confirmed all ten requested invariants: one clock; immutable prior bodies; shared-state lockstep calculation; intent/effect separation; exact update equations; persistent velocity; no collisions; removal on boundary crossing; order-independent physical results; and no premature future subsystems.

## Deviations

There were no deviations from the Iteration 2 instruction or the current architecture documentation.

## Open implementation notes

- `SimulationConfig` retains finite default boundaries of `-100.0` and `100.0` so the Iteration 1 constructor and empty headless CLI remain backward compatible. These are defaults, not a settled experimental world scale; callers can and should provide explicit boundaries for experiments.
- `WorldState` contains only living bodies. Boundary-death information exists in the returned event history, not in persistent corpse or death-registry state.
- Floating-point overflow during motion resolution raises `OverflowError` before advancing the simulation clock.

## Git state

No commit or push was created.

The working tree was clean before Iteration 2 began. This iteration modifies:

- `src/noralet/__init__.py`
- `src/noralet/cli.py`
- `src/noralet/simulation/__init__.py`
- `src/noralet/simulation/config.py`
- `src/noralet/simulation/events.py`
- `src/noralet/simulation/runtime.py`
- `src/noralet/simulation/state.py`

It adds:

- `src/noralet/noralets/`
- `tests/test_action_intents.py`
- `tests/test_motion.py`
- `tests/test_world_validation.py`
- `codex-reports/operation-report-002.md`

All changes are scoped to Iteration 2 implementation, tests and reporting.
