# Noralet Signal System

**Status:** Initial architecture foundation
**Scope:** Primitive inter-Noralet signal emission and perception

## 1. Purpose

The signal system provides Noralets with a primitive communication channel.

Signals are intended exclusively for communication between Noralets.

The architecture defines the physical forms and transmission rules of signals, but deliberately assigns them **no predefined semantic meaning**.

If a shared meaning develops for a signal, that meaning should emerge through Noralet experience and interaction.

---

## 2. Participants

Only Noralets participate in the signal system.

A Noralet may:

* emit signals;
* perceive signals emitted by other Noralets.

Other environmental objects do not emit signals and do not respond to them.

Consumable energy, regions, boundaries and other non-Noralet world elements do not participate in signal communication.

---

## 3. Signal Types

The initial system uses a small fixed alphabet of discrete signal types.

Signals may be represented externally using letters:

```text
A
B
C
D
...
```

The exact number of available signal types remains configurable.

The letters are observer-side identifiers only.

They do not imply linguistic meaning.

For example:

```text
A ≠ danger
B ≠ energy
C ≠ come here
```

unless Noralets themselves eventually develop such conventions.

---

## 4. Semantic Neutrality

Every signal type begins without predefined meaning.

A newly instantiated Noralet may distinguish:

> this is an A-type signal

from:

> this is a B-type signal

but it does not know what either signal represents.

The architecture must not embed semantic associations between:

* signal type;
* environmental condition;
* sender intention;
* receiver response.

Any such relationship must emerge through learning.

---

## 5. Signal Emission

Signal emission is directional.

In the one-dimensional world, a Noralet may emit a signal toward one side:

```text
LEFT
```

or:

```text
RIGHT
```

This makes signal direction an intentional part of the Noralet's action.

A signal is therefore not automatically broadcast equally to every nearby Noralet.

---

## 6. Signal Reception

Signal perception is two-sided.

A Noralet may receive signals arriving from either side of its position.

Reception does not require the receiver to deliberately face or listen toward one direction.

Conceptually:

```text
LEFT  →  Noralet  ←  RIGHT
```

Both sides belong to the receiver's signal-perception space.

---

## 7. Signal Range

Signal perception has a finite radius.

A current candidate value is:

```text
16 dU
```

This value should remain configurable.

The proposed signal radius is somewhat larger than the initial visual-like perception radius.

This allows a Noralet to potentially receive a signal from an individual that it cannot currently perceive visually.

---

## 8. Signal Lifetime

A signal exists physically for a short duration.

The current proposed simulation lifetime is:

```text
1 tick
```

A signal emitted during one Noralet action therefore represents a brief communication event rather than a persistent world object.

The exact relationship between emission tick and reception tick depends on the final simulation-tick lifecycle and remains to be specified there.

---

## 9. Signal Strength

A received signal has a perceived strength related to the distance between sender and receiver.

Closer signals are stronger.

More distant signals are weaker.

A simple initial model may use a linear relationship.

For a `16 dU` signal radius, an illustrative mapping is:

```text
distance ≈ 1 dU   → strength ≈ 16
distance ≈ 8 dU   → strength ≈ 9
distance = 16 dU  → strength = 1
distance > 16 dU  → not perceived
```

Continuous distances may result in continuous signal-strength values.

The exact function remains configurable.

---

## 10. Distance Is Not Directly Exposed

The receiver should not receive the objective sender distance.

For example, the simulation may know:

```text
distance = 6.4 dU
```

but the Noralet should receive only the physical sensory consequence:

```text
signal strength
```

This allows distance-related interpretation to be learned.

A Noralet may eventually discover that stronger signals usually originate from closer sources without being given this relationship as symbolic knowledge.

---

## 11. Direction Perception

The receiver perceives the side from which the signal arrived.

Conceptually, a received signal contains subjective information corresponding to:

```text
type
direction
strength
```

For example:

```text
A
LEFT
strength = high
```

The exact neural encoding of these quantities remains undecided.

---

## 12. Sender Identity

The identity of the sender is not directly included in the received signal.

The receiver must not receive:

```text
sender = Noralet_17
```

Instead, the signal provides only its perceptible physical properties.

A receiver may potentially infer the sender by combining signal perception with other senses.

For example, if only one Noralet is visible on the side from which a signal arrives, the receiver may learn to associate the signal with that individual.

If several Noralets are present, or if the sender is outside visual range, the source may remain ambiguous.

This ambiguity is intentional.

---

## 13. Energy Cost

Signal emission requires a small amount of Noralet Energy.

The cost should be low enough for communication to be practical while ensuring that emitting signals is still a physical action with a consequence.

The exact `eU` cost is not yet defined.

Different signal types should initially have equivalent energy costs unless an experimental reason exists to make them different.

---

## 14. Visualisation

The observer renderer may visually display emitted signals.

A signal may remain visually animated for longer than its physical one-tick simulation lifetime to make communication understandable to a human observer.

For example, the renderer may animate a signal fading or travelling away from the sender across several display frames.

This visual persistence is observer-only.

It must not increase the signal's physical lifetime or allow additional Noralets to perceive it.

---

## 15. Signals and Visual Perception

Signal perception and visual-like perception are separate sensory channels.

A Noralet may therefore encounter situations such as:

```text
signal perceived
sender outside visual range
```

or:

```text
multiple Noralets visible
signal source uncertain
```

This separation creates opportunities for learned inference rather than exposing a unified symbolic world model.

---

## 16. Communication Emergence

The architecture provides:

* distinguishable signal types;
* voluntary signal emission;
* directional transmission;
* signal perception;
* signal strength;
* repeated social interaction.

It does **not** provide:

* vocabulary;
* syntax;
* predefined meanings;
* correct responses;
* communication rewards;
* social labels.

Whether Noralets develop stable conventions is an experimental outcome.

A signal type may acquire different meanings between different Noralet populations or simulation runs.

---

## 17. Open Questions

The following remain unresolved:

* exact number of signal types;
* exact signal radius;
* exact attenuation function;
* exact emission energy cost;
* whether signal strength is noisy;
* whether multiple simultaneous signals interfere;
* whether multiple signals of the same type can be individually distinguished;
* precise tick timing of signal emission and reception;
* whether signals propagate instantaneously within their range or use simulated travel time;
* whether one Noralet may emit more than one signal during a tick;
* exact neural encoding of signal type, direction and strength.

These decisions should be made only when required by the simulation lifecycle and NoraletBrain architecture.

---

## 18. Core Principle

Signals have **form without meaning**.

The world determines that a signal occurred, what physical type it had, where it came from and how strongly it was perceived.

The Noralets determine—through their own experience—whether that signal comes to mean anything at all.
