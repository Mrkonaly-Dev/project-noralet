# Operation Report 007 — Signal Communication System

**Iteration:** 7
**Date:** 2026-08-24
**Status:** Complete

## Summary

Iteration 7 adds a deterministic, physical and semantically empty signal channel to the existing lockstep simulation. A living Noralet may request one of exactly four engine-side signal types in one 1D direction. An affordable request executes after acceleration expenditure and before motion, transfers its configured Energy cost to the local Environmental Energy pool, creates one transient objective signal, and supplies execution feedback to the surviving sender.

Other living Noralets derive a separate variable-length `signal_percepts` collection from the published active-signal state. This channel exposes only a configured numerical pattern, the direction of the signal's origin, and bounded strength. It does not expose a sender, signal type, origin, exact distance, or emission-direction label.

No brain, meaning, grammar, reward, learning, signal memory, noise, collision, propagation delay or random signal process was introduced.

## Signal action model

`SignalType` contains exactly the meaningless engine identifiers `A`, `B`, `C` and `D`. `SignalDirection` contains exactly `LEFT` and `RIGHT`. `SignalEmissionIntent` combines one valid type with one valid direction, and `ActionIntent.signal_emission` is either that single value or `None`; its shape cannot represent multiple or omnidirectional emissions.

`NoraletSignalConfig` validates a finite positive radius, a finite non-negative Energy cost, and four finite, pairwise-distinct sensory patterns of equal non-zero length. Activating it requires the Energy, ecology and Experience systems. A signal request is rejected when signal support is disabled.

Affordability is evaluated against the sender's stored Energy after consumption, existence expenditure and affordable-acceleration resolution. A request executes in full only when the remaining Energy is at least the configured signal cost. It is never reduced or partially emitted. A zero-cost configured signal executes normally without an Energy-spend event.

## Objective signal state

Each executed emission creates one immutable `ActiveSignal` containing:

```text
sender_noralet_id
signal_type
origin
emission_direction
```

This is engine/observer state, not brain-facing experience. It intentionally has no independent signal ID and stores no Energy. The origin is the sender's position in `WorldState(t)`, before the transition's motion.

Signals emitted during `t → t+1` are the only signals published in `WorldState(t+1)`. Signals that were active in `WorldState(t)` are not copied forward, so they expire during the following transition and are absent at `t+2` unless a new emission occurs. New signals replace the prior active set rather than accumulating.

Execution precedes death resolution. Consequently, boundary, Energy-depletion or natural death later in the same transition does not erase an already created final signal. It remains active and perceivable by surviving receivers for its normal one-tick lifetime. A dead sender receives no next experience.

## Signal Energy cost

Signal expenditure occurs after acceleration expenditure and before movement. On successful positive-cost emission, the complete configured cost is subtracted from Noralet Energy and transferred to the Environmental Energy pool containing the sender's tick-start position. Crossing a region boundary later in the transition does not change that destination.

The transfer produces the existing `NoraletEnergySpent` accounting event with reason `SIGNAL`. The same actual expenditure is included in the sender's general bounded `energetic_exertion` sensation. An unaffordable request changes neither Energy nor signal state and produces no execution feedback.

## Signal perception

Reception is derived deterministically from `WorldState(t+1).active_signals` at the receiver's current published position. Eligibility is inclusive:

```text
distance <= signal_radius
RIGHT: receiver_position >= origin
LEFT:  receiver_position <= origin
```

A colocated non-sender is therefore eligible for either emission direction. Every eligible simultaneous signal becomes a separate `SignalPercept`; multiplicity is preserved and no nearest-N, attention or collision rule is applied.

Brain-facing strength is:

```text
strength_signal = 1 - distance / signal_radius
```

It is bounded to `[0, 1]`, with `1` at the origin and `0` at the inclusive radius. `direction_signal` reports where the origin lies relative to the receiver: `-1` left, `0` colocated and `+1` right. It is not the sender's chosen emission-direction enum.

The sender is excluded from its own incoming signal collection. No receiver-facing field contains sender identity, sender signature, signal ID, origin, exact distance, engine signal type, emission direction or emission tick. Visual and signal perception stay independent, so a signal can be perceived while its sender is outside visual range, and visible individuals are not bound to signals.

## Signal sensory patterns

The four engine-side types are translated through the four experiment-supplied configuration patterns. `SignalPercept` contains the resulting numerical `signal_pattern`, never `SignalType.A` through `SignalType.D`. All patterns are immutable finite tuples with one common configured length, but the engine assigns them no semantics.

## Sensorimotor emission feedback

`SensorimotorFeedback` now includes:

```text
signal_emission_activation
signal_emission_pattern
signal_emission_direction
```

For a successfully executed emission during `t → t+1`, a surviving sender receives activation `1`, the same configured sensory pattern as the emitted channel, and own-action direction `-1` for `LEFT` or `+1` for `RIGHT` in its `t+1` experience. When no emission executes, activation and direction are `0` and the pattern is the fixed all-zero vector of configured pattern length. With the signal system disabled, compatibility defaults remain neutral and use an empty pattern.

Feedback records execution rather than request: an unaffordable request remains neutral. A sender that dies during the transition has no posthumous feedback even though its executed signal may remain in objective state for receivers.

## Experience integration

The immutable brain-facing structure is now:

```text
NoraletExperience
    external_percepts
    signal_percepts
    interoception
    sensorimotor_feedback
```

`signal_percepts` is a separate channel and does not alter or overload visual `external_percepts`. The runtime's private transition feedback supplies only the executed own-emission facts required for sensorimotor construction. Incoming reception is derived directly from objective active-signal state, not from observer events.

The future brain boundary intentionally hides `WorldState`, events, routing identity, sender identity, engine signal types, exact origin/distance, region data and exact Energy accounting. Existing Experience-disabled and signal-disabled behavior remains compatible.

## Determinism

Signal execution, lifecycle, reception and pattern conversion use no random stream. Objective signals are constructed in canonical Noralet identity order. Receiver-facing percepts are sorted only by visible content—`direction_signal`, `strength_signal`, then `signal_pattern`—and never intentionally by sender identity. Physically identical values remain duplicate equal percepts, making any residual tie observationally irrelevant.

Tests confirm body/source and action-map insertion-order independence, read purity, unchanged unrelated RNG streams, seed-independent signal perception, and identical full signal histories for the same seed, state and actions.

## Events

`SignalEmitted` records one successful physical emission with the sender ID, engine signal type, chosen direction, tick-start origin and transition ticks. No per-recipient `SignalReceived` events were added because reception is a pure derivation from active world state.

`NoraletEnergyExpenditureReason.SIGNAL` extends the existing expenditure event rather than duplicating Energy accounting. Observable transition ordering is:

```text
EnergyConsumed
existence NoraletEnergySpent
acceleration NoraletEnergySpent
signal NoraletEnergySpent
SignalEmitted
NoraletAccelerated
NoraletMoved
boundary/depletion NoraletDied
deterministic-death NoraletEnergyReleased
natural NoraletDied
natural-death NoraletEnergyReleased
EnergyPointDecayed / EnergyPointDissolved / EnergyPointFormed
TickAdvanced
```

Within each phase, events follow canonical Noralet or existing ecology order. Signal events are independent of action-map insertion order.

## Energy conservation

Successful positive-cost emission is only a transfer:

```text
Noralet Energy → local Environmental Energy
```

The runtime audit continues to enforce:

```text
E_environmental
+ E_consumable
+ E_noralets
= constant
```

Active signals are transient information-bearing physical state, not a fourth Energy storage form. Focused successful, terminal-emission and 100-tick mixed-signal tests preserve the initial three-form total within the existing conservation tolerance.

## Architecture audit

All sixteen requested invariants were checked:

1. `A`–`D` have no built-in semantics.
2. `ActionIntent` can hold at most one type/direction emission.
3. Reception is restricted to the selected half-line and configured radius.
4. Emission appears at `t+1` and expires by `t+2`.
5. `SignalPercept` has no sender identity.
6. Engine types become configured numerical sensory patterns.
7. Reception exposes bounded strength, not universal distance.
8. Visual and signal channels are independent.
9. Successful own emission appears in next sensorimotor feedback.
10. Unaffordable requests produce no executed-emission feedback.
11. Signal cost is a conserved Noralet-to-Environmental transfer.
12. Later sender death does not erase an executed final signal.
13. The sender never receives its own signal as incoming perception.
14. Communication consumes no RNG.
15. No meaning, grammar, reward or predefined use was introduced.
16. No brain or neural system was introduced.

## Files changed

- `src/noralet/noralets/signals.py` and `src/noralet/world/signals.py` add the signal action/configuration and objective-state types.
- `src/noralet/noralets/actions.py` adds the optional emission request.
- `src/noralet/noralets/experience.py` adds signal perception and own-emission feedback values.
- `src/noralet/simulation/config.py`, `state.py`, `events.py`, `experience.py` and `runtime.py` integrate dependencies, lifecycle, execution, Energy transfer, events and perception.
- Public package `__init__.py` files export the Iteration 7 API.
- Test support and four focused signal test modules cover validation, execution/lifetime, perception and determinism.
- `tests/test_experience_validation.py` updates the exact public Experience schema checks.
- `codex-reports/operation-report-007.md` records this iteration.

## Tests and validation

The clean pre-Iteration 7 baseline passed all 202 existing Iteration 1–6 tests in `0.668s`.

Iteration 7 adds 45 focused tests, bringing the complete suite to 247 tests. The final required command was run from the repository root:

```powershell
uv run python -m unittest discover -s tests -v
```

Result: 247 of 247 tests passed in `0.746s`.

Additional final validation:

```powershell
uv run python -m compileall -q src tests
uv lock --check
git diff --check
uv run noralet run --ticks 7 --seed 20260824
uv run python -c "from noralet import ActiveSignal, NoraletSignalConfig, SignalDirection, SignalEmissionIntent, SignalEmitted, SignalPercept, SignalType; print('Iteration 7 public API imports OK')"
rg -n '[ \t]+$' src tests codex-reports/operation-report-007.md
```

Results:

- source and tests compiled successfully;
- the uv lockfile resolved its one package and is current;
- Git diff validation passed; Git only reported the repository's existing LF-to-CRLF conversion notices;
- CLI output was `Completed 7 tick(s); final tick: 7; seed: 20260824`;
- all seven new public signal types imported successfully;
- the trailing-whitespace search returned no matches;
- the repository has no configured formatter, linter or static type checker, so no new validation dependency was added;
- the final diff was inspected for semantic meanings, identity leakage, event-driven perception, Energy mistakes, lifetime errors, neural placeholders and unrelated edits.

## Deviations

There are no implementation deviations from the Iteration 7 instruction.

## Open implementation notes

- Objective `ActiveSignal` retains sender identity solely so the engine can enforce no-self-reception; the identity is removed at the Experience boundary and is not used for percept ordering.
- A configured all-zero pattern is valid if it remains distinct from the other three. Own-emission activation independently distinguishes an executed all-zero channel pattern from neutral feedback.
- Signal radius remains independently configurable and is deliberately not constrained relative to visual radius.

## Git state

No commit or push was created.

The working tree began clean at commit `43b1653` (`Add noralet sensory experience`). This iteration modifies 14 tracked source/test files and adds seven implementation/test files plus this report. Architecture and research documentation were read but not modified. All changes are scoped to Iteration 7 signal communication, compatibility updates, tests and this operation report.
