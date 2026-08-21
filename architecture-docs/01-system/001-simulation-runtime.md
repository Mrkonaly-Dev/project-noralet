# Simulation Runtime Architecture

**Status:** Initial architecture foundation
**Scope:** Core simulation runtime, world-state ownership, tick execution, action resolution, determinism and observer integration

## 1. Purpose

The simulation runtime is the central authority responsible for advancing a Noralet universe through time.

It must provide a clean boundary between:

* physical world state;
* Noralet neural processing;
* action intentions;
* world resolution;
* observation and logging.

The runtime should remain independent of any graphical interface.

A simulation must be fully runnable in headless mode.

The same core runtime may later be observed through a visual renderer or controlled through a launcher without changing the underlying simulation behaviour.

---

## 2. Core Runtime Model

The central runtime object is:

```text
Simulation
```

A `Simulation` owns or coordinates:

```text
Simulation
├── SimulationConfig
├── WorldState
├── NoraletBrains
├── random number generators
├── event system
├── logging / observability
└── tick execution
```

Only the simulation runtime may advance simulation time.

Conceptually:

```text
Simulation.step()

WorldState(t)
    ↓
tick processing
    ↓
WorldState(t+1)
```

---

## 3. Simulation Authority

`Simulation` is the authoritative coordinator of a running universe.

Subsystems may calculate proposed changes, but they should not independently advance the world.

Examples include:

* NoraletBrains produce action intentions;
* environmental systems calculate possible transitions;
* mortality systems calculate death outcomes;
* physics calculates movement;
* energy systems calculate transfers.

The runtime coordinates these results into one consistent transition.

This prevents simulation logic from becoming distributed across unrelated components.

---

## 4. WorldState

`WorldState` represents the objective physical reality of the universe at one simulation tick.

Conceptually:

```text
WorldState(t)
```

may contain:

```text
tick

Noralet body states
    position
    velocity
    acceleration
    stored energy
    physiological condition
    age
    alive/dead state
    physical/perceptual identity

consumable energy points

regions

environmental energy

active physical signals

other future world-level state
```

The exact data structures will evolve as individual subsystems are implemented.

---

## 5. WorldState Does Not Contain NoraletBrains

The physical state of the universe and the neural state of its Noralets should remain architecturally separate.

`WorldState` contains the physical Noralet body.

It does not contain the PyTorch neural network itself.

Conceptually:

```text
WorldState
└── NoraletBodyState

NoraletBrainStore
└── NoraletBrain
```

A Noralet may therefore have:

```text
physical body state
+
individual neural state
```

without mixing GPU-specific neural objects into the physical world representation.

This separation improves:

* testing;
* serialization;
* replay;
* observability;
* CPU/GPU separation;
* future architecture replacement.

---

## 6. CPU and GPU Responsibility

The initial architecture should use the CPU for world simulation and the GPU for neural computation.

### CPU

The CPU should initially handle:

* world state;
* spatial calculations;
* regions;
* energy ecology;
* perception construction;
* movement physics;
* action resolution;
* mortality;
* events;
* logging;
* simulation control.

### GPU

The GPU should initially handle:

* perception encoding;
* recurrent NoraletBrain activation;
* hidden neural states;
* prediction;
* value/evaluation processing;
* lifetime neural learning;
* neural optimiser state.

The intended hardware target for development is a CUDA-capable consumer GPU.

The architecture should not require the physical world itself to be implemented as GPU tensors unless future profiling demonstrates a clear need.

---

## 7. Lockstep Tick Model

The universe uses synchronous lockstep ticks.

Every living Noralet experiences the same objective world moment:

```text
WorldState(t)
```

No Noralet may perceive changes produced by another Noralet during that same tick.

The order in which Noralets happen to be processed by Python or the GPU must not alter the world they experience.

The fundamental rule is:

> **All living Noralets perceive and decide from the same WorldState(t). Their combined actions and environmental processes create WorldState(t+1).**

---

## 8. Read Phase

During perception and NoraletBrain processing, the current world is logically read-only.

Conceptually:

```text
WorldState(t)
      ↓
local perception A → NoraletBrain A
local perception B → NoraletBrain B
local perception C → NoraletBrain C
```

No NoraletBrain directly modifies `WorldState(t)`.

Environmental systems should likewise avoid partially mutating the world while other entities are still processing the current tick.

---

## 9. ActionIntent

A NoraletBrain does not directly perform an action.

It produces an intention describing what the Noralet attempts to do.

Conceptually:

```text
NoraletBrain
    ↓
ActionIntent
```

A future `ActionIntent` may contain fields corresponding to:

```text
acceleration

consume attempt

signal emission
    signal type
    signal direction

other future physical actions
```

The intention represents attempted behaviour, not guaranteed physical outcome.

For example:

```text
consume = true
```

means:

> the Noralet attempts to consume an accessible consumable-energy point.

The world resolver decides what actually occurs.

---

## 10. Why Intentions Are Separate From Effects

Separating intention from physical effect is necessary for fair synchronous simulation.

Suppose two Noralets perceive the same consumable-energy point during tick `t`.

Both select:

```text
consume
```

If the first Noralet processed by Python immediately removed the point, the second Noralet would be unfairly affected by implementation order.

Instead:

```text
Noralet A → consume point X
Noralet B → consume point X
```

are both collected.

The resolver then sees both intentions simultaneously.

If the point contains:

```text
60 eU
```

and both are valid consumers:

```text
A receives 30 eU
B receives 30 eU
```

subject to future energy-capacity rules.

Execution order therefore does not determine physical priority.

---

## 11. Resolution Phase

Once all required intentions and environmental calculations have been collected, the runtime enters a resolution phase.

Conceptually:

```text
WorldState(t)
+
Noralet ActionIntents
+
environmental transitions
+
physical rules
+
mortality outcomes
        ↓
Resolver
        ↓
WorldState(t+1)
```

The resolver is responsible for converting proposed changes into one internally consistent next world state.

---

## 12. Resolution Responsibilities

The exact implementation may be divided into specialised systems, but tick resolution will eventually include processes such as:

* consumable-energy formation;
* consumable-energy decay;
* energy consumption;
* energy capacity handling;
* signal emission;
* signal lifetime;
* acceleration cost;
* velocity update;
* position update;
* baseline existence energy cost;
* physiological-condition changes;
* ageing;
* natural mortality;
* energy-depletion death;
* boundary death;
* energy return to the environment.

The precise order of these substeps must be explicitly defined as each subsystem is implemented.

It must never depend accidentally on dictionary iteration order, entity insertion order or other implementation details.

---

## 13. Position and Motion Transition

Movement is resolved as part of the transition:

```text
t → t+1
```

The world contains objective values such as:

```text
x(t)
v(t)
```

and the Noralet may attempt acceleration:

```text
a(t)
```

The physical movement system calculates:

```text
v(t+1)
x(t+1)
```

according to the world physics.

The exact numerical integration convention will be defined with the movement implementation.

All Noralets use the same convention.

---

## 14. Signals Across Ticks

Signal emission follows the lockstep model.

If a Noralet chooses to emit a signal during tick `t`, another Noralet cannot retroactively perceive that signal during its own processing of `WorldState(t)`.

Conceptually:

```text
tick t:
Noralet A chooses signal B

resolution:
signal enters physical world state

tick t+1:
eligible Noralets perceive signal B
```

The exact physical signal lifetime and ordering will be implemented consistently with the signal-system architecture.

---

## 15. Mortality

A Noralet that exists at `WorldState(t)` participates normally in that experienced moment.

During the transition toward `WorldState(t+1)`, it may die because of:

* energy depletion;
* leaving the traversable world;
* age- and condition-dependent natural mortality;
* future physical causes.

If death occurs:

```text
exists at t
does not exist at t+1
```

The NoraletBrain and its persistent internal neural state are destroyed as part of death processing.

Remaining stored energy returns to the environmental energy system according to the world-energy rules.

---

## 16. Neural Processing During a Tick

Every living Noralet undergoes neural activation every tick.

This remains true even when the resulting action is:

```text
do nothing
```

Conceptually:

```text
perception(t)
+
hidden_state(t-1)
        ↓
NoraletBrain
        ↓
hidden_state(t)
+
ActionIntent(t)
```

The absence of external action does not mean the NoraletBrain was inactive.

Continuous neural activation is part of the Noralet's ongoing simulated existence.

---

## 17. Neural Learning Across Tick Boundaries

The actual result of an action becomes available only after the world transitions to the next tick.

Conceptually:

```text
tick t

Noralet predicts t+1
Noralet selects action
        ↓
world resolution
        ↓

tick t+1

actual experience becomes available
        ↓
prediction / value / homeostatic learning
```

The exact learning implementation belongs to the learning subsystem.

The runtime only needs to preserve the temporal information required for that learning to occur correctly.

---

## 18. TickResult

`Simulation.step()` should return an observer-facing result describing what occurred during the transition.

Conceptually:

```text
TickResult
├── tick_before
├── tick_after
├── events
└── optional observer metadata
```

The `TickResult` should allow other systems to observe the simulation without influencing it.

---

## 19. Event System

Important state transitions should produce structured events.

Examples may eventually include:

```text
NoraletAccelerated

NoraletMoved

SignalEmitted

SignalReceived

EnergyPointFormed

EnergyPointDecayed

EnergyConsumed

NoraletConditionChanged

NoraletDied
```

Events serve several purposes:

* research logging;
* debugging;
* renderer animation;
* replay;
* experiment analysis.

Events describe what happened.

They do not cause behaviour merely because an observer is listening to them.

---

## 20. Renderer Integration

The renderer is an observer of simulation state and events.

It must never contain authoritative simulation logic.

Conceptually:

```text
Simulation
    ↓
TickResult
    ↓
Renderer
```

The renderer may use:

* world state;
* positions;
* velocities;
* signals;
* events;
* Noralet observer information;

to produce an animated representation.

The renderer may visually extend an event beyond its physical simulation duration.

For example, a signal existing physically for one tick may remain visible as an animation for several display frames.

This must not alter simulation behaviour.

---

## 21. Headless Execution

The simulation must run without a renderer.

A basic execution path should eventually support usage similar to:

```text
python -m noralet run <configuration>
```

A headless run may:

* execute ticks as quickly as computation allows;
* print periodic progress;
* record experiment data;
* terminate when configured conditions are met.

Real-world execution speed is independent of simulation time.

---

## 22. Future Graphical Runtime

A graphical application may later run the exact same simulation core.

Conceptually:

```text
CLI
   ┐
   ├── Simulation Core
   │
GUI
   ┘
```

The graphical application should not contain a second implementation of world rules.

A launcher or graphical configuration interface is not required for the initial implementation.

---

## 23. SimulationConfig

Simulation behaviour should be controlled through explicit configuration rather than scattered constants.

A future `SimulationConfig` may contain categories such as:

```text
world scale

initial population

energy scale

region layout

movement parameters

perception ranges

signal parameters

mortality parameters

neural parameters

random seed

deterministic mode

logging options
```

Architecture constants and experiment parameters should remain clearly distinguishable.

Many numerical values currently considered open questions are expected to become configuration values.

---

## 24. Deterministic Randomness

A simulation run receives an explicit master random seed.

Conceptually:

```text
seed = 12345
```

Random behaviour should never rely on uncontrolled global randomness.

The simulation should derive controlled random-number-generator streams for independent responsibilities.

Possible streams include:

```text
world randomness

energy randomness

mortality randomness

Noralet exploration randomness

BaseBrain / neural randomness
```

Subsystem-specific random streams reduce accidental coupling between unrelated random processes.

For example, adding a new random draw to the energy system should not unnecessarily change every later mortality draw.

---

## 25. Reproducibility Goal

In deterministic research mode, the intended relationship is:

```text
same code
+
same configuration
+
same initial state
+
same seed
+
same deterministic execution environment
        ↓
same simulation history
```

GPU neural computation introduces additional reproducibility concerns.

PyTorch and CUDA should therefore be configured for deterministic execution where practical during reproducible research runs.

Absolute determinism across different hardware, software or CUDA environments is not automatically guaranteed.

---

## 26. Run Metadata

Every research run should eventually record sufficient metadata to identify the exact experiment.

At minimum:

```text
master seed

simulation configuration

configuration hash

code version / Git commit

PyTorch version

CUDA version

device information

deterministic-mode status
```

This should make historical experiment results traceable to the runtime that produced them.

---

## 27. Deterministic and Performance Modes

The runtime may eventually support two execution priorities.

### Deterministic research mode

Prioritises:

* reproducibility;
* deterministic algorithms;
* controlled randomness;
* experiment traceability.

### Performance mode

Prioritises:

* simulation throughput;
* GPU efficiency;
* large experiment execution.

Performance mode may relax strict reproducibility requirements where necessary.

The initial implementation should prioritise correctness and reproducibility over maximum throughput.

---

## 28. State Mutation Strategy

`WorldState(t)` should be treated as logically immutable while Noralets perceive and decide.

This does not require physically deep-copying the complete world every tick.

A practical implementation may:

1. expose the current world as read-only during perception and neural processing;
2. collect proposed changes separately;
3. enter resolution only after all intentions are available;
4. apply the resolved changes to produce the next canonical state.

This preserves lockstep semantics without unnecessary copying.

---

## 29. Testing Philosophy

The simulation core should be designed for automated testing from the beginning.

Important test categories will eventually include:

* tick advancement;
* deterministic replay;
* seed isolation;
* physical movement;
* energy conservation;
* fair simultaneous consumption;
* signal timing;
* mortality;
* world-boundary behaviour;
* observer non-interference.

Subsystem behaviour should be testable without launching a renderer or neural training process whenever possible.

---

## 30. Initial Python Package Direction

A possible initial source layout is:

```text
src/noralet/
├── simulation/
│   ├── simulation.py
│   ├── state.py
│   ├── tick.py
│   └── config.py
│
├── world/
│   ├── space.py
│   ├── regions.py
│   ├── energy.py
│   └── physics.py
│
├── noralets/
│   ├── body.py
│   ├── perception.py
│   ├── actions.py
│   └── signals.py
│
├── brain/
│   ├── brain.py
│   ├── model.py
│   └── learning.py
│
├── observability/
│   ├── events.py
│   └── logging.py
│
└── cli/
    └── main.py
```

This structure is an initial organisational direction rather than a permanent public API.

Directories should be added as their corresponding systems are actually implemented.

The project should avoid creating empty abstraction layers merely to match a planned folder structure.

---

## 31. Iterative Implementation

The architecture is intended to be implemented incrementally.

The complete system should not be generated as a single implementation task.

Each iteration should:

1. implement a narrow architectural capability;
2. add tests;
3. validate existing behaviour;
4. produce an implementation report;
5. undergo review before the next iteration begins.

Later systems should build on previously validated foundations.

---

## 32. Iteration 1 Boundary

The first implementation iteration should establish only the deterministic runtime skeleton.

It should include concepts such as:

```text
SimulationConfig

Simulation

WorldState

tick counter

Simulation.step()

controlled random seed

basic structured events

headless CLI execution

determinism tests
```

It should intentionally exclude:

```text
Noralet physics

energy ecology

regions

signals

perception

PyTorch

NoraletBrain

learning

renderer

GUI
```

The first iteration exists to validate the runtime architecture before domain complexity is introduced.

---

## 33. Core Principle

The runtime should preserve one simple model:

> **The world has one objective state at tick `t`. Every living Noralet experiences that same moment from its own perspective. Their internal processing produces intentions, not immediate mutations. The simulation resolves all physical processes together and creates the next objective moment, `WorldState(t+1)`.**

Everything else should be built around this invariant.
