# World Foundations

**Status:** Initial architecture foundation
**Scope:** Core physical world model and energy ecology
**Related visual concept:** `architecture-docs/00-overview/001-initial-plan.png`

## 1. Purpose

The Noralet world is intended to provide a minimal environment in which life has meaningful constraints without prescribing what a Noralet should ultimately value or pursue.

The world should satisfy several basic conditions:

* life is not guaranteed;
* life is finite;
* continued existence requires maintenance;
* other Noralets are present;
* survival does not depend solely on the individual;
* the environment provides enough freedom for Noralets to potentially develop their own learned goals, preferences and behavioural strategies.

The world should remain substantially simpler than biological reality while preserving selected principles that make real life consequential.

---

## 2. Space

The world uses a **continuous one-dimensional spatial axis**.

A Noralet has a real-valued physical position along this axis rather than occupying discrete cells.

Conceptually:

```text
<------------------------------------------------------------>
                         world axis
```

### 2.1 Finite traversable world

The traversable world occupies a finite interval.

The boundaries are not physical walls. A Noralet may move beyond the valid interval. Doing so means leaving the traversable world and results in death.

This represents a physical environmental hazard rather than an artificial movement restriction.

The exact world dimensions are not yet defined.

### 2.2 Noralet collision

Noralets do not use hard physical collision with one another.

Multiple Noralets may therefore occupy the same or nearby positions without blocking movement.

This prevents one-dimensional geometry from allowing individuals or groups to physically trap other Noralets simply by occupying the line.

The 1D axis should be understood as an abstraction of spatial relationships and distance, not as a literal one-dimensional physical tunnel.

---

## 3. Objective World State vs. Noralet Perception

The simulation has access to the true state of the world.

For example, internally it may know:

```text
Noralet position: x = -56.89875
Region: Fertile
```

A Noralet does **not** automatically receive this information.

In particular, Noralets should not directly know:

* their absolute coordinate;
* the coordinate of another Noralet;
* the name or identifier of their current region;
* their global position relative to the world boundaries;
* the complete state of the environment.

Instead, Noralets should experience the world through local perception.

Their understanding of location, distance, environmental structure and important places should potentially emerge from experience rather than from globally supplied coordinates or symbolic region labels.

The exact perception model is outside the scope of this document.

---

## 4. Regions

The world may be divided into spatial regions with different environmental properties.

Regions exist as part of the objective simulation state, but their symbolic classifications are observer-side concepts.

For example, a region may internally be described as:

```text
Fertile
```

A Noralet should not receive the word or category `Fertile`.

Instead, it may experience the consequences of being in that region, such as a greater local availability of consumable energy.

This allows environmental structure to exist without directly providing Noralets with a pre-interpreted map of the world.

The exact number, size, placement and properties of regions are not yet defined.

---

## 5. Energy System

The world uses a finite energy system.

Energy is represented using the abstract unit:

```text
eU — energy unit
```

`eU` is not intended to correspond directly to joules, kilocalories or any specific biological nutrient.

It represents biologically usable energy at the level of abstraction required by the simulation.

The world does **not** create an unlimited supply of energy.

Instead, energy moves between different states.

### 5.1 Energy states

The initial model contains three primary energy states:

```text
Environmental Energy
        ↓
Consumable Energy
        ↓
Noralet Energy
        ↓
Environmental Energy
```

#### Environmental Energy

Energy currently present in the environment but not directly consumable by a Noralet.

This acts as the world's underlying energy reserve.

#### Consumable Energy

Energy currently existing in a form that a Noralet is physically capable of consuming.

Consumable energy is the simplified equivalent of an energy source in biological life.

It is deliberately not modelled as conventional food. The simulation does not currently attempt to represent digestion, macronutrients, chemistry or other unnecessary biological complexity.

#### Noralet Energy

Usable energy currently stored within a living Noralet.

This energy supports continued existence and activity.

---

## 6. Energy Conservation

The total amount of energy in the closed world should remain finite.

Energy is transferred between states rather than being indefinitely generated or destroyed by normal world processes.

Conceptually:

```text
TOTAL ENERGY =
    environmental energy
  + consumable energy
  + energy stored in living Noralets
```

The exact accounting model is not yet specified, but the architecture should preserve the principle that energy used by Noralets ultimately returns to the environment.

This creates a closed ecological cycle rather than an infinite resource-spawning system.

---

## 7. Energy Availability

Environmental energy may become consumable energy over time.

The rate or probability of this conversion may depend on the region.

For example, a fertile region may make environmental energy available in consumable form more frequently or more efficiently than another region.

This does **not** mean that fertile regions generate energy from nothing.

They modify the availability or distribution of energy already present in the world.

Consumable energy may also return to environmental energy if it remains unused for sufficiently long.

The precise conversion mechanisms are not yet defined.

---

## 8. Noralet Energy Use

Living has an energy cost.

A Noralet's stored energy decreases through processes such as:

* continued existence;
* movement;
* potentially other physical actions introduced later.

Movement may have an additional cost dependent on the nature or intensity of that movement.

The exact energy expenditure formulas are not yet defined.

When a Noralet consumes consumable energy, some amount of energy is transferred into its internal Noralet Energy.

---

## 9. Death

Death is a physical state transition in the world.

At minimum, the initial system supports death from:

* energy depletion;
* ageing;
* leaving the traversable world.

Additional causes of death may be introduced later but are not part of the current foundation.

### 9.1 Energy depletion

If a Noralet can no longer maintain the energy required for continued existence, it dies.

Energy expended during its life returns to the environmental energy system.

### 9.2 Energy remaining at death

If a Noralet dies while still containing stored energy, the remaining energy returns to the environment.

Death therefore does not permanently remove the Noralet's remaining stored energy from the closed world.

### 9.3 Falling beyond the world boundary

Moving outside the finite traversable interval causes death.

The boundary is therefore an environmental danger rather than a collision surface or invisible wall.

---

## 10. Ageing and Finite Life

Noralet life is finite.

However, lifespan should not necessarily be represented by a single fixed maximum age shared by all individuals.

A Noralet that spends its life in consistently favourable physiological conditions may have a different expected lifespan from one that repeatedly experiences severe energy deprivation.

Therefore, ageing should eventually support a relationship between:

* chronological age;
* long-term physiological condition;
* history of energy availability;
* probability of death.

This does not imply that high energy directly purchases additional lifetime.

Instead, life history may influence physical deterioration and age-related mortality.

The exact ageing model remains undecided.

---

## 11. Initial Simulation Lifecycle

Reproduction is intentionally excluded from the first version of the world.

A simulation begins with a predefined population of already-living Noralets.

The initial experiments may use adult Noralets to avoid introducing developmental and reproductive systems before the basic ecology and cognition are understood.

Over the course of the simulation:

```text
initial Noralet population
        ↓
life, movement, energy use and interaction
        ↓
individual deaths
        ↓
last Noralet dies
        ↓
no living population remains
```

Once every Noralet has died, the simulation may contain only environmental and consumable energy.

With no living Noralet remaining to consume it, this energy simply remains within the world and may continue transitioning between environmental and consumable forms according to normal environmental rules.

This produces a natural terminal state for an experimental run.

---

## 12. Observer Knowledge

The renderer and analysis systems may display information unavailable to the Noralets themselves.

Examples include:

* exact coordinates;
* region identities;
* exact energy values;
* age;
* motion data;
* internal simulation status;
* emitted signals;
* event history.

This information exists for scientific observation and debugging.

Its presence in the renderer must never imply that the same information is available to a Noralet.

The observer layer is non-causal: visualisation must not influence the simulation.

---

## 13. Visualisation Concept

`architecture-docs/00-overview/001-initial-plan.png` represents an early visual direction for observing the world.

It demonstrates concepts such as:

* a continuous world axis;
* Noralets positioned along the axis;
* movement visualisation;
* signal emission;
* consumable energy;
* region information;
* observer-side Noralet information;
* event history;
* simulation time measured in ticks.

The image is **non-normative**.

Its specific values, labels, mechanics, distances, colours and displayed states must not be interpreted as final architecture decisions.

The visual design direction may be retained independently of the underlying simulation mechanics.

---

## 14. Open Questions

The following remain intentionally unresolved:

* exact world size;
* coordinate convention and origin;
* exact number and layout of regions;
* region properties;
* environmental energy spatial representation;
* mechanism for Environmental Energy → Consumable Energy;
* mechanism and timing for Consumable Energy → Environmental Energy;
* spatial representation of individual consumable energy sources;
* whether consumable energy exists as discrete concentrations or a continuous field;
* Noralet energy capacity;
* baseline cost of existence;
* movement energy costs;
* other action costs;
* ageing model;
* physiological-condition model;
* precise conditions for energy-depletion death;
* meaning and duration of one simulation tick;
* initial Noralet population size;
* initial energy distribution;
* perception system;
* how Noralets perceive world boundaries;
* how Noralets perceive consumable energy.

These questions should be resolved incrementally rather than fixed prematurely.

---

## 15. Core Principle

The world should provide **constraints, consequences and opportunities**, not externally assigned purpose.

Energy scarcity can make survival consequential.

Spatial structure can make learning useful.

Finite lifespan can make time consequential.

Other Noralets can make social behaviour consequential.

None of these should directly tell a Noralet what its goal is.

The architecture should instead create conditions in which useful behaviours, learned representations and potentially higher-level goals can emerge from experience.
