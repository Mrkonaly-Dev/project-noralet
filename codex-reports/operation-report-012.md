# Operation Report 012 — Renderer / Observer UI v1

## Summary

Implemented a compact PySide6/Qt Widgets desktop observer for the existing
Project Noralet runtime. The application is available through:

```text
uv run noralet ui
```

It contains a live autonomous baseline simulation and a separate Research 001
launcher. The renderer owns no simulation rules, neural state, objective state,
or random source. Existing headless simulation and research entry points remain
available and were regression-tested.

## Desktop technology and dependency

- PySide6/Qt Widgets `6.11.2` is declared in `pyproject.toml` and resolved in
  `uv.lock`.
- The UI uses `QMainWindow`, `QTabWidget`, `QPainter`, `QTimer`, and `QProcess`.
- Qt is imported lazily only after the `ui` CLI branch is selected. Importing
  `noralet.cli` does not load a PySide6 module.
- No plotting, web-server, or browser runtime was added.

## Architecture

The live path is deliberately thin:

```text
Qt controls -> LiveRunSetup -> shared baseline factory
                            -> AutonomousSimulationRunner
                            -> one authoritative step per requested tick

published state/results -> observer copies -> canvas and inspector
```

The baseline factory formerly private to the research harness was exposed as a
shared builder. Its numerical baseline, initial bodies, seed mapping, world
configuration, BaseBrain configuration, and learning-condition behavior are
unchanged. Research 001 still calls the same construction logic through its
existing wrapper.

The research path remains a separate headless process:

```text
Qt fields -> validated CLI arguments -> QProcess
          -> existing `noralet research baseline-lifetime-adaptation`
          -> streamed output/progress/result directory
```

The UI therefore never duplicates or embeds the scientific experiment loop.

## Live simulation controls

The Live tab provides:

- Start, Pause, single-tick Step, and fresh deterministic Reset;
- `1x`, `10x`, `100x`, and `Max` speed modes;
- simulation seed, population, device, maximum ticks, and all four existing
  learning conditions;
- automatic stop at extinction or the configured maximum tick;
- an always-visible current tick and run status.

Speed modes execute only bounded sequential bursts. They never coalesce,
interpolate, or skip authoritative runtime ticks. The current timer presets are
`1 @ 100 ms`, `2 @ 25 ms`, `4 @ 5 ms`, and `8 @ 0 ms`, keeping Qt event handling
responsive while allowing CUDA throughput.

Reset destroys the live session and constructs a new session from the displayed
setup. Tests verify exact equality with a fresh independently constructed run.

## World canvas

The visual language follows the supplied initial-plan image only as an art
direction: near-black space, restrained technical labels, a thin one-dimensional
world line, and luminous colored entities. No mechanics or obsolete labels were
copied from the image.

The canvas renders:

- the complete finite world and lethal left/right boundaries;
- subtle sparse/fertile ecology regions;
- current consumable Energy points;
- every living Noralet with deterministic observer-only color and ID;
- velocity and latest actually applied acceleration vectors;
- current A/B/C/D signals with direction and origin;
- a brief observer-only visual linger for signals and death flashes;
- click selection and a selection ring.

World-to-canvas mapping was tested at both boundaries, the center, and after
resize. Colors are a pure function of stable body data and use no simulation or
renderer RNG. Transient animation timing uses wall-clock time only in copied UI
objects and cannot enter the runtime.

## Inspector

Selecting a living Noralet shows current public values for:

- ID, age, position, velocity, Energy, and condition;
- Energy distress, condition distress, exertion, external percept count, and
  signal percept count;
- the latest available predictive and homeostatic learning metrics.

Unavailable metrics display an em dash. Selection is cleared safely when the
selected Noralet dies. The inspector reads the existing public Experience
boundary and latest runner results; it does not activate a brain or mutate a
world object.

## Research 001 launcher

The Research tab provides validated fields for seed count, maximum ticks,
sampling cadence, population, device, and condition checkboxes. Defaults match
the existing normal protocol: 10 seeds, 5,000 ticks, sampling every 10 ticks,
population 6, CUDA, and all four conditions.

The existing harness requires at least two unique replicate seeds, so the UI
enforces the same minimum instead of inventing a one-seed exception. A real
two-seed, one-condition, two-tick CPU batch was run through `QProcess`. Output
streaming, progress parsing, successful exit handling, manifest discovery, and
the Open Result Folder integration all passed. Stop first requests graceful
termination and then kills the child after 1.5 seconds if necessary; any newly
created partial result directory is retained and exposed.

## Observer non-causality

Automated tests establish the renderer boundary by comparing before/after:

- immutable published `WorldState` and runtime configuration;
- every named simulation random-stream state;
- every brain's parameters and recurrent hidden state;
- action RNG state;
- ticks, brain activations, events, and action history.

Repeated repaint, resize, selection, inspector refresh, and signal-glyph reads
changed none of these values. A second test compares six single UI ticks with
the same six ticks requested in UI bursts and obtains identical action/tick
history, objective state, neural parameters, hidden states, and RNG state.

## Visual and interaction smoke

The exact `uv run noralet ui` entry point started its Qt event loop and remained
running until intentionally terminated after the launch smoke. Offscreen
programmatic interaction then exercised Start, Pause, Step, Reset, `10x`,
`100x`, and `Max`, selection, live inspection, and the Research tab. The two
final screenshots were visually inspected for clipping, contrast, label
legibility, canvas mapping, selected-state visibility, and research output
layout:

```text
research-results/ui-smoke/renderer-v1-styled.png
research-results/ui-smoke/research-v1-styled.png
```

The environment did not provide a human-controlled visible desktop during
validation, so native Explorer appearance was not visually inspected. Its
`QDesktopServices.openUrl(QUrl.fromLocalFile(...))` call was verified directly
with the actual result-directory state and an integration-level mock of the OS
handoff.

## Headless and CUDA validation

Headless simulation:

```text
uv run noralet run --ticks 5 --seed 12345
Completed 5 tick(s); final tick: 5; seed: 12345
```

Headless research:

```text
uv run noralet research baseline-lifetime-adaptation --seeds 2 --max-ticks 2 --sample-every 1 --population 2 --device cpu --conditions no-learning
```

Result: 2/2 runs completed at tick 2 and produced a normal Research 001 result
directory.

CUDA environment:

```text
torch.__version__: 2.13.0+cu130
torch.cuda.is_available(): True
torch.version.cuda: 13.0
torch.cuda.get_device_name(0): NVIDIA GeForce RTX 3060
```

The focused UI CUDA test constructed a real `full-current-brain` live session,
executed three authoritative autonomous ticks, and verified finite neural
parameters on CUDA with no device mismatch.

PyTorch continues to emit its existing optional-NumPy warning because NumPy is
not a project dependency. Neither the runtime nor this UI requires NumPy, and
all validations completed.

## Tests and reproducibility checks

Focused observer/UI/process suite:

```text
QT_QPA_PLATFORM=offscreen uv run python -m unittest tests.test_ui_session tests.test_ui_qt -v
```

Result: 18/18 passed in 6.427 seconds.

Complete suite:

```text
QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v
```

Result: 426/426 passed in 16.706 seconds, including real CUDA tests.

Final project checks:

```text
uv sync
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```

All completed successfully. `uv sync` checked 15 installed packages and
`uv lock --check` resolved 34 locked packages.

## Scope boundary and repository state

No neural architecture, action policy, learning equation, simulation phase,
ecology rule, mortality rule, signal rule, causal random stream, research
metric, or experimental interpretation was changed. Evolution, reproduction,
multi-generation controls, replay, analysis dashboards, and consciousness
claims remain outside Renderer / Observer UI v1.

No commit or push was performed.
