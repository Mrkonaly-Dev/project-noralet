# Operation Report 001 — Deterministic Simulation Skeleton

**Iteration:** 1
**Date:** 2026-08-21
**Status:** Complete

## Summary

Iteration 1 implemented a minimal, dependency-free Python simulation runtime for an empty deterministic universe. A simulation begins at tick `0`, advances only through `Simulation.step()`, returns an immutable structured `TickResult`, owns independently seeded named random streams, and can run for a finite number of ticks from a headless CLI.

No Noralet, physical-world, energy, perception, action, neural, rendering, persistence or replay mechanics were introduced.

## Files changed

- `.gitignore` — ignores Python-generated caches, package artifacts and local environments.
- `pyproject.toml` — defines the minimal `src`-layout Python package and `noralet` console entry point without runtime dependencies.
- `src/noralet/__init__.py` — exposes the small public runtime API.
- `src/noralet/__main__.py` and `src/noralet/cli.py` — provide `python -m noralet run --ticks <n> --seed <seed>`.
- `src/noralet/simulation/config.py` — defines immutable `SimulationConfig` with an explicit master seed.
- `src/noralet/simulation/state.py` — defines immutable `WorldState`, containing only the authoritative tick.
- `src/noralet/simulation/runtime.py` — defines the authoritative `Simulation` and its explicit read/resolution/publication transition.
- `src/noralet/simulation/events.py` — defines the immutable structured `TickAdvanced` event.
- `src/noralet/simulation/tick.py` — defines immutable observer-facing `TickResult` values.
- `src/noralet/simulation/randomness.py` — defines stable independent named random streams.
- `src/noralet/simulation/__init__.py` — exposes the simulation package API.
- `tests/test_runtime.py` — covers clock advancement, state replacement, structured results, immutability and same-seed history.
- `tests/test_randomness.py` — covers reproducibility, different seeds, stream isolation, simulation ownership and cross-process hash independence.
- `tests/test_cli.py` — smoke-tests finite headless execution through the module entry point.
- `codex-reports/operation-report-001.md` — records this implementation iteration.

## Runtime design

`SimulationConfig` is a frozen value containing the required master seed. `Simulation` retains that configuration, creates the initial frozen `WorldState(tick=0)`, owns a `DeterministicRandomStreams` instance, and is the only runtime component that publishes a new canonical world state.

During `Simulation.step()`, the current state is held as the immutable read-phase state. The trivial Iteration 1 resolver creates a distinct `WorldState` for the next tick. A frozen `TickAdvanced` event and frozen `TickResult` are then constructed, after which the new state is published as canonical. Events are descriptive values in an immutable tuple; they have no callbacks and do not drive simulation behaviour.

The CLI constructs `SimulationConfig`, runs the requested number of steps and prints the final tick and seed. It contains no simulation authority of its own.

## Determinism

Every `Simulation` requires an explicit integer master seed. `DeterministicRandomStreams` derives each exact stream name from the master seed with a domain-separated SHA-256 digest; it never uses Python's process-randomised `hash()` and never calls module-global random functions.

Each name receives its own cached `random.Random` instance. Drawing from one instance cannot advance another stream, so adding draws to `world`, for example, does not change the sequence for `mortality`.

With the same code, configuration, initial state, master seed and compatible Python runtime, Iteration 1 produces the same tick history and named-stream sequences. Cross-process stability was tested with different `PYTHONHASHSEED` values. Cross-version random-algorithm identity is not claimed, and PyTorch/CUDA determinism is outside this iteration.

## Tests and validation

The following checks were run from the repository root:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
```

Result: 13 tests passed. The complete suite was run three times successfully; each run included the cross-process determinism and CLI smoke tests.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m noralet run --ticks 7 --seed 20260821
```

Result: successful exit with `Completed 7 tick(s); final tick: 7; seed: 20260821`.

```powershell
python -m compileall -q src tests
git diff --check
rg -n '[ \t]+$' .gitignore pyproject.toml src tests codex-reports\operation-report-001.md
```

Result: compilation and Git diff checks completed successfully, and the explicit scan found no trailing whitespace in the new files. `pyproject.toml` parsing and a public-package import/step smoke check also completed successfully. The repository had no pre-existing configured formatter, linter, type checker or test framework; the standard-library `unittest` runner was used and no new validation dependency was added.

The final architecture review confirmed:

1. `Simulation` owns the only authoritative clock.
2. Only `Simulation.step()` publishes clock advancement.
3. `WorldState(t)` is frozen and remains unchanged during the step.
4. Each step creates a distinct `WorldState(t+1)` at an explicit resolution boundary.
5. Results and events are immutable descriptive data and cannot drive the runtime merely by observation.
6. Randomness comes from explicit, simulation-owned, master-seed-derived named streams.
7. The implementation contains no premature Noralet, world-mechanics or neural architecture.

## Deviations

There were no deviations from the Iteration 1 instruction or the current simulation-runtime architecture.

## Open implementation notes

- The world state intentionally contains only `tick`; the resolution phase intentionally performs no domain mechanics.
- Source-checkout commands require either an installed package or `src` on `PYTHONPATH`. The package metadata supports normal installation and exposes both the module and console entry points.
- Named streams are created lazily. Future simulation subsystems can request exact names when those subsystems actually exist; no speculative stream registry was added.

## Git state

No commit or push was created.

The working tree already contained documentation work before Iteration 1 began:

- modified: `architecture-docs/03-noralets/001-noralet-foundations.md`
- untracked: `architecture-docs/01-system/`
- untracked: `architecture-docs/03-noralets/002-signal-system.md`
- untracked: `architecture-docs/04-learning/`

Those pre-existing files were read but not modified by this iteration.

Iteration 1 added the following untracked paths:

- `.gitignore`
- `pyproject.toml`
- `src/`
- `tests/`
- `codex-reports/operation-report-001.md`
