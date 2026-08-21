# Noralet Foundations

**Status:** Initial architecture foundation
**Scope:** Noralet embodiment, perception, life, action, learning principles and death

## 1. Purpose

A Noralet is a living entity within the simulated world.

The architecture should model Noralets as embodied, continuously existing and individually learning entities rather than scripted agents executing predefined behavioural rules.

The simulation intentionally simplifies biological reality, but should preserve several fundamental properties of real living organisms:

* existence through a body;
* limited local perception;
* incomplete knowledge of the world;
* internal bodily state;
* energy requirements;
* self-generated action;
* continuous experience over time;
* individual learning during life;
* finite lifespan;
* irreversible death.

A Noralet should receive **experience rather than explanation**.

Its understanding of the world should develop from what happens to it.

---

## 2. Embodiment

Every living Noralet has a physical body represented inside the world.

At minimum, the objective simulation state of that body contains:

* position;
* velocity;
* acceleration;
* stored energy;
* age;
* life state.

Additional physical properties may be introduced later if required.

The existence of an objective physical property does not imply that the Noralet directly perceives it.

---

## 3. Objective State and Subjective Experience

The simulation has access to the exact physical state of every Noralet.

For example:

```text
x = -56.89875
v = +0.0041 dU/tick
E = 437.2 eU
region = Sparse
```

The Noralet does not receive this representation.

In particular, it has no native direct access to:

* absolute position;
* world coordinates;
* exact distance from the world centre;
* current region classification;
* absolute velocity;
* exact numerical acceleration;
* exact stored energy;
* global world dimensions;
* complete world state.

These are observer-side facts.

A Noralet experiences only information made available through its physical senses and internal bodily perception.

---

## 4. Local Spatial Perception

A Noralet has a local spatial sensory system analogous to an extremely simplified form of vision.

It perceives objects within a configurable radius around itself in both directions along the one-dimensional world.

A current candidate initial value is:

```text
12 dU
```

This is a configuration value rather than a fixed architectural constant.

Conceptually:

```text
-12 dU                SELF                 +12 dU
   |--------------------●---------------------|
```

Objects outside this range are not available through this sensory channel.

---

## 5. Perceptible World Objects

The initial visual-like sensory system should be capable of perceiving at least:

* other Noralets;
* consumable energy;
* world boundaries.

The Noralet should be capable of distinguishing different physical categories without receiving their semantic meaning.

A completely inexperienced Noralet may therefore perceive:

> this type of thing is different from that type of thing

without knowing:

> this is consumable energy

or:

> this is another Noralet.

Meaning must be learned from experience.

---

## 6. Perceptual Consistency

Objects belonging to the same fundamental category should share perceptual characteristics.

For example:

* different consumable-energy concentrations should appear related;
* all Noralets should share common perceptual characteristics;
* boundaries should form another distinguishable category.

This gives the learning system stable structure from which categories can potentially emerge.

The architecture must not attach learned semantic labels to these categories.

---

## 7. Individual Noralet Recognition

Noralets should be perceptually identifiable both as members of the same broad type and as distinct individuals.

A Noralet should therefore potentially be able to learn:

> this is another entity like me

while also learning:

> this is the same individual I encountered previously.

This must not be implemented by exposing internal identifiers such as:

```text
noralet_id = 17
```

Instead, every Noralet should have stable perceptual characteristics that allow individual recognition to be learned.

The exact representation of these characteristics remains undecided.

---

## 8. Relative Spatial Experience

Perception is relative to the observing Noralet.

A Noralet may perceive information corresponding to:

* which side an object is on;
* its local relative position;
* changes in that position across time;
* other perceptual characteristics of the object.

Absolute coordinates are never directly supplied.

The precise neural encoding of spatial perception remains undecided.

---

## 9. No Native Velocity Knowledge

A Noralet has no built-in sensor providing its absolute velocity.

The simulator may know:

```text
v = +4.2 dU/kTick
```

but this numerical value is observer knowledge.

A Noralet must instead infer movement from changing sensory experience.

For example:

```text
object becomes progressively closer
        ↓
relative movement is occurring
```

Over time, a Noralet may develop internal representations corresponding to:

* movement direction;
* approximate movement speed;
* stopping distance;
* expected future position.

These representations must emerge through learning rather than being supplied as explicit physical variables.

---

## 10. Self-Generated Action

A Noralet is capable of affecting its own physical state through actions.

Initial physical actions may include:

* accelerating toward one side;
* accelerating toward the other side;
* applying no acceleration;
* consuming accessible consumable energy;
* emitting a signal.

The exact action interface will be defined separately from the NoraletBrain implementation.

Doing nothing must remain a valid action.

---

## 11. Motor Awareness

A Noralet may directly experience its own attempted action.

For example, it may have internal awareness corresponding to:

> I am currently exerting myself toward this direction.

This is different from knowing its objective acceleration or velocity.

The Noralet does not need to receive:

```text
a = -0.0042 dU/tick²
```

It only needs access to the bodily or motor consequences of its own action.

This distinction allows learned body models to develop without exposing simulation-state variables directly.

---

## 12. Energy Interoception

A Noralet has native perception of its internal energetic condition.

However, it does not receive its exact stored-energy value.

The simulator may know:

```text
E = 437.2 eU
```

while the Noralet experiences only a subjective bodily signal.

Conceptually:

```text
objective stored energy
        ↓
body
        ↓
subjective internal sensation
```

This sensory channel represents simplified interoception.

---

## 13. Energy Level

The subjective energy signal should provide information about the Noralet's current physical condition without directly exposing a percentage or exact quantity.

The relationship does not need to be linear.

For example:

* differences between high energy levels may produce relatively weak subjective change;
* approaching severe depletion may produce increasingly strong bodily signals.

The Noralet is not born with the semantic knowledge that a particular internal sensation means imminent energy depletion.

It may learn this relationship through experience.

---

## 14. Exertion and Energy Consumption

A Noralet may also perceive how energetically demanding its current state or action is.

Strong acceleration, for example, may produce increased subjective exertion because energy is being consumed more rapidly.

This provides an internal experience analogous to physical effort or fatigue.

The Noralet should not receive an exact value such as:

```text
dE/dt = -1.73 eU/tick
```

Instead, it receives a bodily consequence from which the significance of energetic expenditure may be learned.

---

## 15. Boundaries

World boundaries are perceptible physical features.

A Noralet may therefore detect an approaching boundary if it lies within sensory range.

However, the Noralet is not born knowing:

> crossing this boundary causes death.

The boundary initially exists only as a perceptually distinct feature of the environment.

Its significance must arise from experience, observation, communication, innate structural bias, or some future combination of these mechanisms.

The exact solution should not be artificially hard-coded before the learning architecture is designed.

---

## 16. Consumable Energy

Consumable energy is perceptible through the local sensory system.

A Noralet is not initially given the semantic concept:

> this restores energy.

Instead, it may experience a sequence such as:

```text
perceive object
    ↓
interact with object
    ↓
internal energy condition changes
```

The relationship between that perceptual category and improved bodily state can therefore become learned knowledge.

---

## 17. Other Noralets

Other Noralets are perceptible physical entities.

The architecture should provide enough information for a Noralet to potentially learn that other Noralets:

* belong to a common physical category;
* are persistent individuals;
* move independently;
* perform actions;
* emit signals;
* can be encountered repeatedly.

The simulation must not directly provide higher-level social concepts such as:

* friend;
* enemy;
* stranger;
* ally;
* relationship;
* trust;
* affection.

If such representations develop, they should develop inside the Noralet through experience.

---

## 18. Learning Requirement

Each Noralet must possess its own neural learning system.

Long-term behaviour should emerge from genuine changes produced by the Noralet's individual experience.

The architecture must avoid reducing learning to predefined behavioural rules such as:

```text
if energy_low:
    seek_energy()
```

or manually encoded social strategies.

Instead:

```text
experience
    ↓
neural processing
    ↓
action
    ↓
consequence
    ↓
neural change
```

should be capable of producing lasting behavioural and representational change.

A Noralet late in its life should therefore be neurally different from the same Noralet near the beginning of its life as a consequence of its unique history.

---

## 19. Continuous Internal State

A Noralet should not behave as a stateless function evaluated independently once per tick.

The architecture should support persistent internal neural state across successive moments.

Conceptually:

```text
previous internal state
        +
current experience
        ↓
new internal state
        ↓
action
```

This allows past experience to influence present processing even before long-term neural learning is considered.

The exact architecture implementing this persistence remains undecided.

Possible implementations may eventually include recurrent networks, state-space architectures, memory systems or other mechanisms.

No specific model is selected by this document.

---

## 20. Innate Structure vs. Learned Knowledge

A Noralet does not need to begin life as a completely random neural system.

Real biological organisms are not born as unstructured blank networks.

The Noralet architecture may therefore contain **innate structure**.

Possible innate properties include:

* sensory pathways;
* motor pathways;
* interoception;
* mechanisms capable of learning;
* memory mechanisms;
* neural architectural organisation;
* basic ability to process temporal experience.

This must be distinguished from **innate semantic knowledge**.

A newly instantiated Noralet should not automatically know:

* what consumable energy is;
* that energy depletion is dangerous;
* what the boundary means;
* what another Noralet intends;
* what any signal means;
* which behaviour is socially desirable;
* where it is in the world;
* what its ultimate goal should be.

The intended principle is:

> **innate capacity to learn, not pre-written understanding of the world.**

The exact method used to create the initial NoraletBrain is intentionally unresolved.

---

## 21. Neural Architecture

Every Noralet is expected to possess an individual neural network or neural system responsible for its learned behaviour and internal representations.

The precise architecture is not yet selected.

Unresolved possibilities include:

* recurrent neural networks;
* predictive-learning systems;
* reinforcement-learning mechanisms;
* Hebbian or local learning;
* backpropagation-based online learning;
* world-model architectures;
* combinations of multiple mechanisms.

Selecting the learning architecture is a major research problem and should not be decided prematurely.

The architecture must be evaluated based on whether it can support genuine individual learning rather than on convenience alone.

---

## 22. No Externally Assigned Purpose

The NoraletBrain should not be given a manually defined high-level purpose such as:

```text
survive as long as possible
maximize energy
make friends
reproduce
explore
```

The physical world provides consequences and constraints.

The learning system may contain low-level biological structure necessary for learning and continued action, but higher-level goals, preferences and strategies should not simply be written into the Noralet.

A central experimental goal is to investigate what kinds of internal representations and behavioural priorities can emerge under these conditions.

---

## 23. Ageing

A Noralet has a finite life.

Age increases as simulation time progresses.

Age-related death does not necessarily occur at one identical fixed tick for every Noralet.

Long-term physical condition, including energy history, may eventually influence lifespan.

The exact ageing and mortality model remains part of world/body architecture and is not defined here.

A Noralet does not necessarily receive an explicit numerical age.

Whether ageing produces subjective bodily changes remains unresolved.

---

## 24. Death

Death is irreversible.

When a Noralet dies:

```text
living Noralet
      ↓
death
      ↓
NoraletBrain ceases to exist
Noralet internal state ceases to exist
body disappears
remaining stored energy returns to environment
```

There is no post-death Noralet state.

The Noralet does not receive a special experience representing its own death.

There is simply a final experienced moment followed by no subsequent experience.

The disappearance of the body and neural system is immediate at the simulation level unless future architecture introduces a specific reason to represent physical remains.

---

## 25. Initial Physical Equality

For early controlled experiments, Noralets should preferably begin with equivalent basic physical capabilities.

This may include equivalent:

* perception ranges;
* acceleration capabilities;
* energy capacity;
* bodily energy costs;
* signal capabilities.

This reduces confounding variables when studying differences that emerge through individual experience.

Later experiments may deliberately introduce biological variation.

Initial neural states may still require limited individual variation depending on the eventual learning architecture.

---

## 26. Observer Knowledge

The observer and analysis systems may display substantially more information than the Noralet itself experiences.

This may include:

* exact position;
* velocity;
* acceleration;
* energy;
* age;
* region;
* current neural activity;
* internal state;
* emitted signals;
* learning changes;
* event history.

Observer access must remain non-causal.

Displaying or recording internal information must never alter what the Noralet experiences or how the simulation develops.

---

## 27. Open Questions

The following remain intentionally unresolved:

* exact NoraletBrain architecture;
* method for producing the initial neural state;
* exact online-learning mechanism;
* whether pretraining is required;
* how innate neural structure is generated;
* persistent-memory architecture;
* sensory input encoding;
* exact visual perception radius;
* exact distance representation;
* perceptual noise;
* individual perceptual signatures;
* energy-interoception function;
* exertion representation;
* exact consumption interaction;
* additional bodily senses;
* relationship between ageing and bodily perception;
* whether different Noralets begin with neural variation;
* mechanisms by which higher-level goals may emerge.

These questions should be treated as research problems rather than filled with arbitrary implementation choices.

---

## 28. Core Principle

A Noralet should be a **learning living process**, not a scripted character.

The environment provides experiences.

The body provides consequences.

The neural system processes those experiences and changes because of them.

The architecture may give the Noralet the machinery required to learn, but should avoid giving it the conclusions it is intended to discover.

The long-term behaviour, internal representations, associations, communication meanings and potentially higher-level goals of a Noralet should arise primarily from its own lived history.
