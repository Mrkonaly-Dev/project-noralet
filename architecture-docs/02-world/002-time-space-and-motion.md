# Time, Space and Motion

**Status:** Initial architecture foundation
**Scope:** Simulation time, spatial measurement, movement physics and observer-side physical quantities

## 1. Purpose

The Noralet world should use a small and internally consistent physical system.

The simulation is not intended to reproduce real-world physics in full. Instead, it adopts selected physical principles while removing unnecessary complexity.

The initial movement model is intentionally similar to motion in empty space:

* no drag;
* no friction;
* no passive braking;
* no external forces acting on Noralets;
* velocity persists until the Noralet changes it through acceleration.

---

## 2. Measurement Philosophy

The simulation uses its own abstract units rather than real-world units.

Real-world notation may still be reused where appropriate.

Examples:

```text
t = time
x = position
d = distance
E = energy
v = velocity
a = acceleration
```

The simulation should not use metres, seconds, joules, calories or other real-world units as its canonical physical measurements.

This avoids implying biological or physical fidelity that the simplified world does not provide.

---

## 3. Time

The canonical unit of simulation time is the:

```text
tick
```

A tick is one discrete update step of the world.

The simulation therefore progresses as:

```text
t = 0
t = 1
t = 2
t = 3
...
```

with:

```text
Δt = 1 tick
```

### 3.1 Meaning of a tick

Each tick represents another moment in which:

* the world exists in a state;
* environmental processes may change;
* Noralets may perceive their surroundings;
* Noralets may undergo internal change;
* Noralets may act;
* physical state may advance.

Not every property of the world must change during every tick.

A tick provides the opportunity for change.

### 3.2 Noralet experience of time

Noralets do not perceive the simulation as a sequence of externally visible computational steps.

From their perspective, successive experienced moments form their continuous existence.

The discrete nature of simulation time is observer knowledge.

### 3.3 Real-time rendering

The duration of a tick in real-world wall-clock time is not part of the world's physics.

A renderer may, for example, target:

```text
10 ticks / real second
```

which would visually correspond to approximately:

```text
0.1 real seconds / tick
```

This is a rendering or execution-speed choice, not a physical definition of the tick.

A headless simulation may execute ticks significantly faster than real time without changing the subjective simulation time experienced by Noralets.

### 3.4 Larger display units

For observer readability, multiples of ticks may be used.

For example:

```text
1 kTick = 1000 ticks
```

`kTick` is primarily a display and analysis convenience.

The simulation itself may continue using ticks as its canonical unit.

---

## 4. Distance and Position

The canonical unit of distance is:

```text
dU — distanceUnit
```

The world uses a continuous one-dimensional coordinate axis.

Adjacent integer coordinates are separated by exactly:

```text
1 dU
```

For example:

```text
x = 4.0
x = 5.0
```

are separated by:

```text
1 dU
```

Positions are not restricted to integer values.

A Noralet may occupy positions such as:

```text
x = 53.7824
```

The coordinate system exists objectively inside the simulation but is not directly exposed to Noralets.

---

## 5. Velocity

Velocity describes change in position over time.

Conceptually:

```text
v = Δx / Δt
```

The canonical simulation unit is:

```text
dU / tick
```

Because movement per individual tick may be numerically small, observer interfaces may display velocity using larger time scales, such as:

```text
dU / kTick
```

This does not change the underlying simulation physics.

### 5.1 Persistent motion

There is no passive loss of velocity.

If:

```text
a = 0
```

then velocity remains unchanged.

A Noralet moving at constant velocity therefore continues moving indefinitely unless:

* it accelerates;
* it dies;
* another physical mechanism capable of altering motion is introduced in a later architecture version.

The initial architecture includes no such external mechanism.

---

## 6. Acceleration

Acceleration describes change in velocity over time.

Conceptually:

```text
a = Δv / Δt
```

Its canonical unit is:

```text
dU / tick²
```

A Noralet changes its own movement state by producing acceleration.

Acceleration may point in either direction along the one-dimensional axis.

### 6.1 Starting movement

A stationary Noralet begins moving by accelerating.

Example:

```text
initial v = 0

Noralet accelerates right
        ↓
v > 0
```

### 6.2 Coasting

Once the desired velocity has been reached, the Noralet may stop accelerating.

Its existing velocity then persists without additional movement-specific energy cost.

### 6.3 Braking

There is no separate braking mechanic.

Stopping is produced by acceleration opposite to the current velocity.

Example:

```text
current v > 0
a < 0
```

The Noralet must therefore actively alter its velocity in order to stop.

It cannot instantly remove momentum.

---

## 7. Self-Generated Motion

In the initial architecture, only a Noralet itself may change its movement state.

There are initially no:

* environmental forces;
* wind;
* friction;
* drag;
* collisions capable of transferring momentum;
* gravitational acceleration along the world axis;
* forces produced by other Noralets.

This intentionally keeps the physical system minimal.

More complex physical interactions may be introduced later if they serve a clear experimental purpose.

---

## 8. Movement and Energy

Constant velocity does not create additional movement-specific energy expenditure.

A Noralet may therefore coast without paying continuously for motion itself.

Acceleration does require energy.

Conceptually:

```text
constant velocity
→ no additional movement cost

acceleration
→ energy cost
```

Braking also requires energy because braking is simply acceleration opposing the current velocity.

The energy cost should depend on the magnitude of acceleration.

The exact relationship is not yet fixed.

Possible models include:

```text
cost ∝ |a|
```

or nonlinear alternatives where stronger acceleration becomes disproportionately expensive.

The chosen model should remain simple enough to interpret experimentally.

---

## 9. Acceleration Limits

A Noralet should not necessarily be able to produce arbitrarily large acceleration in a single tick.

A body-level maximum may be defined:

```text
|a| ≤ a_max
```

This represents a physical limitation of the Noralet body.

The exact value of `a_max` is not yet defined.

There is currently no requirement for an equivalent hard maximum velocity.

A sufficiently long period of acceleration may therefore produce high velocity, at the cost of time and energy.

---

## 10. Consequences of Persistent Velocity

The movement system creates consequences that Noralets may need to learn.

For example, approaching the edge of the world at high velocity requires the Noralet to begin decelerating early enough to avoid crossing the boundary.

A Noralet cannot rely on an instantaneous stop command.

This allows spatial prediction, movement estimation and learned body/world models to become potentially useful without explicitly rewarding them.

---

## 11. World Scale

The exact numerical scale of the world remains configurable.

The intended initial experiments involve populations on the order of tens of Noralets rather than extremely large populations.

This keeps:

* simulation cost reasonable;
* individual Noralets observable;
* social interactions interpretable;
* full-life simulation histories manageable.

The architecture should not prevent substantially larger populations from being instantiated, but the initial design does not target thousands of simultaneously active Noralets.

### 11.1 Population-relative scale

World size may be configured relative to the initial Noralet population.

Conceptually:

```text
world_length =
    initial_population × target_space_per_noralet
```

Similarly, total world energy may scale with initial population.

This allows experiments with different population sizes while preserving roughly comparable environmental density.

Once a simulation begins, world size remains fixed even as Noralets die.

---

## 12. Region Layout

The initial world may use three environmental region classes:

```text
Infertile
Sparse
Fertile
```

A simple symmetric arrangement is currently preferred:

```text
VOID
  |
  | INFERTILE | SPARSE | FERTILE | SPARSE | INFERTILE |
                                                            |
                                                           VOID
```

The central area is fertile.

Moving outward leads through sparse regions and finally infertile regions near the world boundaries.

The primary distinction between these regions may initially be their effect on energy availability.

For example:

* **Fertile:** environmental energy becomes consumable relatively easily.
* **Sparse:** lower availability.
* **Infertile:** very low availability.

These region names are observer-side classifications.

Noralets are not directly informed which region they occupy.

---

## 13. Numerical Scale

Exact numerical scales remain intentionally configurable.

Current working expectations include:

* energy quantities commonly existing in the hundreds or thousands of `eU`;
* movement values chosen to remain human-readable;
* typical observer-facing velocity and acceleration magnitudes preferably remaining within manageable numeric ranges;
* world dimensions large enough for tens of Noralets to move and disperse without making interactions excessively rare.

These are design targets rather than fixed constants.

The final scale should be derived from meaningful relationships such as:

* typical lifetime;
* energy expenditure per tick;
* acceleration cost;
* perception radius;
* typical travel time between regions;
* population density.

Absolute numbers should be chosen after these relationships are better defined.

---

## 14. Observer Knowledge

The simulation and observer systems may know exact physical quantities including:

```text
x
v
a
t
region
```

A Noralet does not automatically have access to these values.

In particular, absolute velocity and absolute position are not native Noralet perceptions.

The distinction between objective physical state and subjective Noralet experience must remain explicit throughout the architecture.

---

## 15. Open Questions

The following remain unresolved or configurable:

* exact world length;
* population-to-world-size ratio;
* exact total energy scale;
* exact `a_max`;
* acceleration energy-cost function;
* typical movement speeds;
* precise region widths;
* exact regional energy-conversion behaviour;
* whether region boundaries transition sharply or gradually;
* renderer real-time tick rate;
* performance limits for large populations.

These values should remain configuration-level decisions until experimental requirements justify fixing defaults.
