# Operation Report 006 — Perception, Interoception and Sensorimotor Experience

**Iteration:** 6
**Date:** 2026-08-24
**Status:** Complete

## Summary

Iteration 6 introduces the first immutable Noralet-facing experience layer. A living Noralet can now receive a local variable-length external sensory field, derived homeostatic distress, and bounded sensorimotor consequences from the transition that produced the current world state.

The implementation adds no brain, learning, value/reward mechanism, semantic emotion, communication, sensory noise or fixed-size neural representation. Objective world truth remains authoritative and observer-visible; `NoraletExperience` is a deterministic restricted view derived from it.

Active experience requires the existing Energy, ecology and physiology systems plus an explicitly valid perceptual signature for every initial Noralet. With experience disabled, all Iteration 1–5 configuration and runtime paths retain their prior behaviour.

## Experience architecture

The implemented separation is:

```text
objective immutable WorldState(t)
          +
runtime-owned feedback from t-1 → t
          ↓
deterministic engine-side ExperienceBuilder
          ↓
immutable NoraletExperience(t)
```

`WorldState` continues to contain exact positions, velocities, Energy, age, condition, regions, pools and simulation identities. `_ExperienceBuilder` can read those objective values inside the engine only to calculate permitted sensations. The returned `NoraletExperience` contains exactly:

```text
external_percepts
interoception
sensorimotor_feedback
```

This boundary allows a future brain executor to receive domain experience values without being passed `WorldState`, events or exact transition accounting. The public experience values use frozen, slotted dataclasses and immutable tuples throughout.

## External perception

`NoraletExperienceConfig.vision_radius` is finite, positive and experiment-supplied. A living observer receives every visible:

- Consumable Energy point;
- other living Noralet;
- left world boundary;
- right world boundary.

Visibility is inclusive at `distance <= vision_radius`. There is no nearest-N limit, attention cap or occlusion. The observer's own body, dead Noralets, region divisions, region kinds, fertility and Environmental Energy pools are absent.

Every `ExternalPercept` contains only:

```text
appearance_pattern
direction_signal
proximity_signal
```

Direction is `-1.0`, `0.0` or `+1.0` for left, coincident or right. Boundary direction is forced left/right even when the observer is exactly on that boundary. For visible phenomena:

```text
proximity_signal = 1 - objective_distance / vision_radius
```

with only floating-point safety clamping. Exact distance and the configured radius do not enter experience.

Percepts are sorted mechanically from left to right by objective position. Exact-position ties use an internal stable kind rank and then internal identity; neither tie-break value is exposed. Collection order exists only for deterministic transport, serialization and testing. It does not encode salience or usefulness.

## Appearance system

The focused immutable configuration supplies distinct base appearance patterns for Consumable Energy, other Noralets and boundaries. Validation requires the three vectors to be finite, pairwise distinguishable, equal in length and non-empty.

Every external appearance has one uniform shape:

```text
base appearance pattern + signature component
```

Consumable Energy and boundaries receive an all-zero signature component. Another Noralet receives its stored persistent `NoraletBodyState.perceptual_signature`. Signature values are finite and immutable; an experience-enabled simulation requires every body to have exactly the configured positive signature length. Body reconstruction preserves the tuple exactly throughout life. Empty signatures remain valid only on compatibility runs where experience is disabled.

Simulation identity and perceptual signature remain separate. `ExternalPercept` contains no source ID, point ID, object-kind enum, Energy amount or other-object body facts.

## Interoception

The current body supplies three derived sensory values:

```text
energy_ratio = stored_energy / energy_capacity

energy_distress =
    (1 - energy_ratio) ** energy_distress_exponent

condition_distress =
    (1 - condition) ** condition_distress_exponent

energetic_exertion =
    1 - exp(-actual_transition_expenditure / exertion_sensation_scale)
```

`energy_distress_exponent` must be greater than `1`; `condition_distress_exponent` must be positive; the exertion scale must be positive. Distress values are bounded to `[0, 1]`, while exertion remains in `[0, 1)`. Numerically stable `-expm1` and the largest representable value below `1.0` handle large valid inputs without producing an invalid endpoint.

Energy and condition distress reflect the body in current published `WorldState(t)`. Exertion reflects the immediately preceding transition. Exact eU, Energy capacity, Energy ratio, objective condition and age are not included.

The two distress values are negative homeostatic sensory information only. Iteration 6 attaches no reward, value update, punishment, pain variable, preference or learning semantics to them.

## Sensorimotor feedback

The feedback generated by `t → t+1` becomes part of experience at `t+1`:

```text
motor_direction = sign(applied_acceleration)

motor_effort =
    1 - exp(-abs(applied_acceleration) / motor_effort_scale)

consume_activation =
    1 if a consume act was executed, else 0

ingestion_signal =
    1 - exp(-actual_consumed_energy / ingestion_sensation_scale)
```

Motor feedback uses applied acceleration, never the requested value. Coasting therefore moves the body while producing zero motor effort. Consume activation reports the motor act independently of outcome. Ingestion is positive only after a positive physical transfer and exposes neither a success flag nor the exact transferred eU.

Energetic exertion combines actual existence and acceleration expenditure without separating their reasons. At `t=0`, there is no prior transition, so motor, consumption, ingestion and exertion feedback is neutral while external perception and current-state distress remain active.

A boundary, depletion or natural death produces no `t+1` experience and no special death sensation.

## Runtime integration

The runtime stores an immutable private `_TransitionFeedback` for each survivor. During transition resolution it captures directly:

- `applied_acceleration` from the affordability calculation;
- `consume_attempt_executed` from the validated action intent;
- `consumed_energy` while fair allocations are committed;
- `actual_energy_expenditure` while existence and acceleration transfers are committed.

These values are created alongside the authoritative calculations. They are not reconstructed from `TickResult.events`, and the experience builder does not import or inspect event values. Feedback is published only after the candidate world passes the existing energy-conservation audit, matching state publication timing.

The read-only APIs are:

```python
simulation.experience_for(noralet_id)
simulation.experiences_for_all()
```

The first accepts only an integer routing identity and rejects unknown or dead identities with `KeyError`. The second returns an immutable tuple of domain experiences following the already canonical ascending living-identity order of `WorldState.bodies`. Routing identities are not embedded in the experiences.

Calling either API does not advance the tick, mutate state, consume Energy, alter physiology, draw RNG, change actions or emit events.

## Information intentionally hidden

None of the following can appear as a field in `NoraletExperience` or its contained sensory values:

- observer or source simulation IDs;
- tick number or event metadata;
- absolute position or universal direction explanation;
- exact distance, `dU`, velocity, speed or acceleration magnitude;
- exact stored Energy, capacity, ratio, transferred amount, expenditure amount or `eU`;
- exact condition, age or mortality probability;
- requested actions or other-Noralet actions;
- region IDs, region kinds, fertility or Environmental Energy pools;
- semantic object-type enums;
- expenditure-reason enums;
- `consume_success` or any equivalent semantic confirmation;
- danger, hunger, pain, fear, reward or value labels.

The engine retains objective values because physics and observability require them; only the derived experience boundary hides them from a future brain.

## Determinism

Experience generation contains no RNG and owns no sensory random stream. It is a pure observation transformation over immutable current state plus immutable last-transition feedback.

Tests confirm repeated reads return equal values without changing `WorldState`, tick or RNG state. Equivalent worlds are insensitive to input insertion order. External percept ordering remains stable across body and point ordering, including exact-position ties. Enabling and repeatedly reading experience does not change subsequent objective states, events, mortality draws or energy behaviour compared with an otherwise equivalent experience-disabled simulation.

## Architecture audit

All 17 requested invariants were checked against the final source and tests:

1. brain-facing values are sensory patterns rather than explanations;
2. no universal distance, Energy, position, velocity or age measurements are exposed;
3. visibility is local and inclusive;
4. the complete local field is returned without truncation;
5. configured patterns replace semantic object labels;
6. persistent signatures distinguish individuals under one shared Noralet base;
7. boundaries are perceptible without danger semantics;
8. other-body velocity is absent while temporal proximity changes remain observable;
9. Energy and condition are transformed into distress;
10. distress has no learning or reward role yet;
11. motor sensation reads actual applied execution;
12. consume success has no semantic flag;
13. feedback follows `t → t+1` causality;
14. dead Noralets have no next experience for all three current death causes;
15. observer events are non-causal;
16. sensory transformation has no randomness;
17. no brain or neural placeholder was introduced.

## Files changed

- `src/noralet/noralets/experience.py` — adds experience configuration and immutable brain-facing value types.
- `src/noralet/simulation/experience.py` — adds private transition facts and the deterministic experience builder.
- `src/noralet/noralets/body.py` — adds validated persistent perceptual signatures.
- `src/noralet/simulation/config.py` — adds optional experience configuration and body-system dependencies.
- `src/noralet/simulation/runtime.py` — captures transition facts, preserves signatures and exposes read-only experience APIs.
- `src/noralet/noralets/__init__.py`, `src/noralet/simulation/__init__.py` and `src/noralet/__init__.py` — export the public Iteration 6 domain API.
- `tests/noralet_energy_test_support.py` — permits explicit experience configuration in the shared energy simulation helper.
- `tests/experience_test_support.py` — adds compact explicit Iteration 6 constructors.
- `tests/test_experience_validation.py` — covers configuration, signatures, immutability, compatibility and exact public structures.
- `tests/test_external_perception.py` — covers visibility, all-object perception, spatial signals, appearances, boundaries, privacy, movement and stable individuals.
- `tests/test_interoception_sensorimotor.py` — covers distress, current-state timing, motor/consume/ingestion/exertion feedback and all death causes.
- `tests/test_experience_determinism.py` — covers read purity, stable all-living order, insertion independence, reproducibility and unchanged dynamics.
- `codex-reports/operation-report-006.md` — records this iteration.

## Tests and validation

The clean pre-Iteration 6 baseline passed all 151 existing Iteration 1–5 tests in `0.687s`.

Iteration 6 adds 51 focused tests, bringing the complete suite to 202 tests. The final required command was run from the repository root:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: 202 of 202 tests passed in `0.675s`.

Additional final validation:

```powershell
uv run python -m compileall -q src tests
uv lock --check
uv run noralet run --ticks 7 --seed 20260824
uv run python -c "... Iteration 6 public API smoke ..."
git diff --check
rg -n '[ \t]+$' src tests
```

Results:

- source and tests compiled successfully;
- the uv lockfile resolved its one package and was current;
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260824`;
- the five public Iteration 6 experience types imported successfully;
- Git diff and trailing-whitespace checks passed;
- the repository has no configured formatter, linter or static type checker, so no new validation dependency was added.

## Deviations

There is no implementation deviation from the requested experience semantics.

One requested test procedure has an inherited observability limitation: under the existing Iteration 4 affordability rule, an unaffordable acceleration is reduced by spending all remaining stored Energy, so the Noralet necessarily dies from Energy depletion in that same transition. The Iteration 6 rule forbidding posthumous experience therefore prevents inspecting a `t+1` motor sensation for that exact case. The focused test verifies the requested acceleration of `4.0` is physically reduced to the actual applied `1.5`, followed by no next experience. Living-body feedback is demonstrably constructed from the same `applied_accelerations` runtime value, never from requested intent. Prior physics and the no-posthumous-experience invariant were both preserved.

## Open implementation notes

- `experiences_for_all()` intentionally returns raw immutable experience values only. A future brain-execution coordinator can use the runtime's stable living-identity order for routing without adding identity to `NoraletExperience`.
- External perception is intentionally variable-length. A later encoder will have to consume that structure without weakening the current information boundary; no encoder design was introduced here.
- Unaffordable acceleration remains terminal under the current Energy law, as described under Deviations. A future change could make reduced-but-surviving execution observable only by deliberately changing that physical law, not by creating a death experience.

## Git state

No commit or push was created.

The working tree began clean. This iteration modifies:

- `src/noralet/__init__.py`
- `src/noralet/noralets/__init__.py`
- `src/noralet/noralets/body.py`
- `src/noralet/simulation/__init__.py`
- `src/noralet/simulation/config.py`
- `src/noralet/simulation/runtime.py`
- `tests/noralet_energy_test_support.py`

It adds:

- `src/noralet/noralets/experience.py`
- `src/noralet/simulation/experience.py`
- `tests/experience_test_support.py`
- `tests/test_experience_determinism.py`
- `tests/test_experience_validation.py`
- `tests/test_external_perception.py`
- `tests/test_interoception_sensorimotor.py`
- `codex-reports/operation-report-006.md`

Architecture and research documentation were read but not modified. All implementation changes are scoped to Iteration 6 experience, its tests and this report.
