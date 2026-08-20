# Noralet Perception and Body

**Status:** Initial architecture foundation
**Scope:** Physical embodiment, local perception, internal energy perception and signal perception

## 1. Purpose

A Noralet exists through a body inside the simulated world.

It should not receive a symbolic description of objective world state.

Instead, it experiences a limited local view of its environment and selected internal bodily signals.

The distinction is fundamental:

```text
objective simulation state
        ↓
physical perception
        ↓
Noralet experience
        ↓
learned internal interpretation
```

The simulator knows what objects objectively are.

A Noralet must learn what its experiences mean.

---

## 2. Embodiment

Every living Noralet has a body represented within the physical world.

At minimum, the body has objective properties including:

* position;
* velocity;
* acceleration;
* stored energy;
* age;
* life/death state.

Additional body properties may be introduced later.

The existence of an objective body property does not imply that the Noralet can directly perceive its numerical value.

---

## 3. No Direct Access to Global Physical State

A Noralet should not natively receive:

* its absolute `x` coordinate;
* absolute world position;
* exact distance from world centre;
* current region identity;
* exact absolute velocity;
* exact numerical acceleration;
* global world dimensions;
* exact distance to the opposite side of the world;
* complete world state.

These are observer-side facts.

The Noralet instead develops knowledge from local experience.

---

## 4. Local Visual-Like Perception

A Noralet has a local spatial perception system analogous to extremely simplified vision.

It perceives a continuous area around itself extending in both directions along the one-dimensional world.

A candidate initial radius is:

```text
12 dU
```

This value is not yet considered a permanent architecture constant and should remain configurable.

Conceptually:

```text
-12 dU                SELF                 +12 dU
   |--------------------●---------------------|
```

Objects outside the perception radius are not visually perceived through this channel.

---

## 5. Perceived Objects

The local perception system contains objects that physically exist inside the perception range.

Important perceptible classes initially include:

* other Noralets;
* consumable energy;
* world boundaries.

A Noralet should be able to perceive that these are physically distinct kinds of things.

However, it should not receive their predefined semantic meaning.

For example, its input must not conceptually mean:

```text
"food at -5 dU"
"friend at +3 dU"
"dangerous cliff at +10 dU"
```

Instead, it perceives distinguishable object patterns whose significance must be learned.

---

## 6. Perceptual Categories Without Meaning

Objects of the same fundamental physical type should have perceptual similarity.

For example:

* consumable energy instances should share perceptual characteristics;
* Noralets should share perceptual characteristics;
* the world boundary should have its own distinct perceptual characteristics.

This gives a completely inexperienced Noralet the ability to notice:

> these two things appear to be the same kind of thing.

It does **not** give it knowledge such as:

> this is consumable energy.

Meaning should arise from interaction and experience.

For example, a Noralet may eventually associate a particular perceptual object type with an increase in its internal energy state.

---

## 7. Individual Noralet Recognition

Different Noralets should be perceptually identifiable as members of the same broad object class while remaining individually distinguishable.

Conceptually, a Noralet should be able to learn both:

> this entity is another thing like me

and:

> this is the same individual I encountered previously.

The system should not accomplish this by exposing an explicit database identifier such as:

```text
noralet_id = 17
```

Instead, individual Noralets should possess stable perceptual characteristics sufficient for learned recognition.

The exact representation of individual perceptual identity remains undecided.

---

## 8. Relative Spatial Information

A Noralet should perceive objects relative to itself rather than through global coordinates.

Its experience may contain information corresponding to:

* which side an object is on;
* relative local position;
* changes in relative position over time;
* potentially approximate relative distance.

The exact numerical representation supplied to the NoraletBrain has not yet been defined.

The essential requirement is that absolute world coordinates remain unavailable.

---

## 9. Learned Motion Awareness

A Noralet does not possess a native absolute velocity sensor.

The simulation may objectively know:

```text
v = +4.2 dU/kTick
```

but the Noralet should not directly receive this value.

Instead, it may infer its movement state from changes in visual perception over time.

For example:

```text
object appears progressively closer
        ↓
relative movement is occurring
```

A Noralet may therefore develop an internal estimate of:

* whether it is moving;
* in which direction it is moving;
* approximately how quickly it is moving;
* when it must begin slowing down.

These are learned interpretations rather than supplied physical variables.

This allows the Noralet to develop a body/world model from experience.

---

## 10. Self-Generated Acceleration

A Noralet may have direct awareness of the action it is currently attempting to produce.

For example, it may internally experience the fact that it is exerting itself to accelerate in one direction.

This does not require exposing an exact observer-side value such as:

```text
a = -0.0042
```

The distinction is:

```text
"I am exerting myself in this direction"
```

may be directly available,

while:

```text
"My absolute velocity is +4.2 dU/kTick"
```

is not.

The exact internal representation of motor intention remains part of later NoraletBrain design.

---

## 11. Internal Energy Perception

A Noralet should have native perception of its bodily energy condition.

However, it should not receive the exact numerical value of its stored energy.

The simulation may know:

```text
E = 437.2 eU
```

while the Noralet receives a subjective internal signal.

Conceptually:

```text
objective Noralet Energy
        ↓
body / interoception
        ↓
subjective internal energy state
```

This is analogous to biological interoception.

---

## 12. Energy Level and Energy Expenditure

Two aspects of energy perception should remain distinguishable.

### 12.1 Stored-energy state

A Noralet can internally experience how favourable or critical its current energy condition is.

This signal does not need to be linear.

For example, differences at very high energy may produce little subjective change, while approaching dangerous depletion may produce increasingly strong internal signals.

The Noralet should not necessarily know what this feeling means initially.

Its significance can be learned through experience.

### 12.2 Energy expenditure

A Noralet may also perceive increased energetic strain when its energy is being consumed more rapidly.

Strong acceleration, for example, may produce greater subjective exertion.

This allows it to experience something analogous to:

> this action is costly or tiring.

Again, this should not be represented as direct access to:

```text
dE/dt = -1.73 eU/tick
```

It is an internal bodily experience rather than an observer statistic.

---

## 13. Signal Perception

In addition to local visual-like perception, Noralets have a second major external sensory channel:

**signal perception**.

This is loosely analogous to hearing or communication through sound.

A Noralet may emit signals that other Noralets can perceive.

The physical signal forms may be predefined by the architecture.

Their meanings are not.

A completely inexperienced Noralet may therefore distinguish:

```text
signal A
signal B
signal C
```

without knowing what any of them represent.

Meaning may emerge through repeated social and environmental experience.

---

## 14. Signal Meaning

The architecture must avoid assigning semantic labels to signals from the Noralet's perspective.

A signal should not inherently mean:

* danger;
* food;
* come here;
* leave;
* affection;
* identity;
* yes;
* no.

Those meanings, if they arise, should be learned or socially established.

This distinction is central to studying emergent communication.

---

## 15. Visual Perception vs. Signal Perception

Visual-like perception and signal perception should remain distinct channels.

Visual perception primarily provides information about local spatial entities.

Signal perception primarily provides emitted communicative events.

The two systems do not need identical range or spatial precision.

Future architecture may allow signals to:

* travel farther than visual perception;
* weaken with distance;
* provide less exact spatial localisation;
* overlap when multiple Noralets emit signals.

These properties are not yet fixed.

---

## 16. Boundaries

World boundaries are perceptible physical features.

A Noralet should therefore have the possibility of detecting an approaching boundary before crossing it.

However, it is not born knowing:

> crossing this causes death.

The boundary is simply another distinguishable perceptual phenomenon until experience gives it significance.

Because an individual Noralet cannot learn from its own death afterward, boundary-related knowledge may ultimately depend on indirect experience, prior innate structure, observing others, communication, or other mechanisms to be designed later.

The architecture should not solve this problem prematurely.

---

## 17. Consumable Energy

Consumable energy is visible through the local perception system.

A Noralet is not initially told:

> this object restores energy.

It experiences the object as a particular perceptual category.

Through interaction, it may learn the relationship:

```text
encounter object
        ↓
consume object
        ↓
internal energy state improves
```

The learned meaning of the object can therefore emerge from direct bodily consequence.

---

## 18. Other Noralets

Other Noralets are locally perceptible entities.

A Noralet should have enough perceptual information to potentially learn:

* that other Noralets belong to a common class;
* that individual Noralets are persistent individuals;
* that they move independently;
* that they produce signals;
* that repeated encounters with the same individual are possible.

The architecture should not directly provide concepts such as:

* friend;
* enemy;
* parent;
* ally;
* stranger;
* relationship.

Such concepts, if they emerge, belong to learned internal representations.

---

## 19. Observer Knowledge vs. Noralet Experience

The observer may display exact information such as:

```text
Noralet #12
x = -56.89875
v = +3.7 dU/kTick
E = 421.4 eU
region = Sparse
```

The Noralet itself may experience only:

* nearby perceptual objects;
* their relative arrangement;
* changes across successive moments;
* received signals;
* internal energy condition;
* current exertion;
* its own attempted actions.

This separation must remain explicit in both implementation and documentation.

---

## 20. Open Questions

The following remain unresolved:

* exact local perception radius;
* exact encoding of perceived objects;
* whether relative distance is explicit or must itself be inferred;
* visual perceptual resolution;
* perceptual noise;
* object occlusion, if any;
* representation of individual Noralet appearance;
* signal alphabet size;
* signal range;
* signal propagation behaviour;
* signal attenuation with distance;
* source localisation for signals;
* simultaneous or overlapping signals;
* whether signal transmission takes time;
* exact energy interoception function;
* exact representation of exertion;
* additional bodily senses;
* whether ageing produces directly perceivable bodily effects.

These should be resolved only as required by later architecture decisions.

---

## 21. Core Principle

A Noralet should receive **experience, not explanation**.

The world may objectively contain:

* coordinates;
* velocities;
* regions;
* consumable energy;
* other Noralets;
* boundaries.

The Noralet should not be given those concepts as pre-labelled facts.

It should instead receive structured but initially meaningless sensory patterns and internal bodily signals from which useful interpretations may develop over time.

The architecture should provide enough perceptual regularity for learning to be possible without providing the learned meaning itself.
