# Learning Foundations

**Status:** Initial architecture and research foundation
**Scope:** Noralet learning principles, persistent neural state, homeostasis, predictive learning and BaseBrain development

## 1. Purpose

Noralets must be genuinely learning entities rather than scripted systems whose meaningful behaviour is predetermined by the simulation.

The architecture should allow a Noralet's behaviour, internal representations, associations and strategies to change as a consequence of its own individual life experience.

The world provides:

* physical constraints;
* sensory experience;
* bodily consequences;
* other Noralets;
* opportunities for action.

It must not provide pre-written interpretations of those experiences.

The intended principle is:

> **The architecture provides the ability and pressure to learn, not the conclusions that should be learned.**

---

## 2. Two Timescales of Learning

Noralet development is expected to operate on two fundamentally different timescales.

### 2.1 BaseBrain development

Noralets do not begin from completely random neural systems.

A shared initial neural structure, referred to as the:

```text
BaseBrain
```

is produced before the individual experimental lives begin.

The long-term intention is for the BaseBrain itself to arise through a separate training process inspired by biological evolution.

Conceptually:

```text
evolution-like training
        ↓
     BaseBrain
        ↓
   individual birth
```

The BaseBrain represents inherited neural organisation and learning capability.

It should not contain a complete semantic understanding of the Noralet world.

### 2.2 Individual lifetime learning

When a Noralet is created, it receives its own instance of the BaseBrain.

From that moment onward, its neural system develops independently.

```text
BaseBrain
   ├── copy → Noralet A
   ├── copy → Noralet B
   └── copy → Noralet C
```

Each copy is subsequently modified by that Noralet's unique experience.

There is no automatic sharing of learned lifetime knowledge between living Noralets.

Consequently:

```text
same initial brain
+
different lives
        ↓
different developed brains
```

A Noralet late in life should be neurally different from the same Noralet near the beginning of life.

---

## 3. Innate Structure Is Not Innate Knowledge

The BaseBrain may contain substantial innate neural structure.

Possible inherited capabilities include:

* functioning sensory pathways;
* recurrent neural dynamics;
* motor output pathways;
* interoceptive processing;
* memory mechanisms;
* learning mechanisms;
* temporal processing;
* general structural biases that make learning possible.

This must remain distinct from semantic knowledge.

A newly created Noralet should not automatically know:

* what consumable energy is;
* that consuming it improves its energy state;
* that low energy is dangerous;
* what a world boundary represents;
* that crossing a boundary causes death;
* what another Noralet intends;
* the meaning of any signal;
* how fast it is moving;
* where it is in the world;
* what actions will produce particular external outcomes;
* what its higher-level goals should be.

The BaseBrain provides **capacity**, not a completed worldview.

---

## 4. Continuous Neural Existence

A NoraletBrain must not operate as a stateless function that independently processes every tick.

The Noralet has persistent neural activity across successive moments.

Conceptually:

```text
experience(t)
+
internal_state(t-1)
        ↓
    NoraletBrain
        ↓
internal_state(t)
+
action(t)
```

The internal neural state continues from one tick into the next.

This makes the current neural state dependent not only on the present sensory input, but also on what occurred before it.

A tick in which the Noralet performs no external action still contains neural activation and experience.

---

## 5. Fast State and Slow Learning

Two kinds of neural change must be distinguished.

### 5.1 Persistent internal state

The recurrent internal state changes continuously from tick to tick.

Conceptually:

```text
h(t-1) → h(t) → h(t+1)
```

This may contain temporary neural representations related to:

* recent perception;
* current expectations;
* remembered events;
* ongoing behavioural context;
* recent signals;
* bodily state;
* anticipated outcomes.

These interpretations are not required to exist as explicit symbolic variables.

They may exist only as neural activation patterns.

### 5.2 Neural weights

The NoraletBrain's weights represent slower, learned change.

Experience modifies these weights over the lifetime of the Noralet.

Conceptually:

```text
W
↓ experience
W'
↓ experience
W''
```

The persistent internal state represents what is currently active.

The changing weights represent what has been learned.

---

## 6. Candidate Recurrent Architecture

A recurrent neural architecture is currently preferred over a purely stateless network.

A simplified candidate structure is:

```text
visual perception ──┐
signal perception ──┤
interoception ──────┤
motor context ──────┘
         ↓
  Perception Encoder
         ↓
    recurrent core
         ↓
  persistent state
     ┌───┼─────────┐
     ↓   ↓         ↓
 Action  Prediction  Value /
 Head      Head      Evaluation
```

The recurrent core may initially be implemented using a mechanism such as a GRU.

This is a practical candidate, not yet a permanent architectural requirement.

Other recurrent or stateful neural architectures may later prove more appropriate.

---

## 7. Predictive Learning

A major proposed learning mechanism is continuous prediction of future experience.

At each moment, the NoraletBrain attempts to predict aspects of its next experience.

Examples may include:

* future local perception;
* future internal energy sensation;
* future physiological condition;
* future exertion;
* future received signals.

Conceptually:

```text
current experience
+
internal state
+
chosen action
        ↓
prediction of experience(t+1)
```

When the next tick occurs:

```text
predicted experience
        vs.
actual experience
        ↓
prediction error
```

The prediction error becomes a neural learning signal.

This creates a self-supervised learning process.

The Noralet does not require an externally prepared labelled dataset.

Its life itself generates the training data.

---

## 8. Why Prediction Matters

Predictive learning encourages the NoraletBrain to discover regularities in its environment.

For example:

```text
perceive a particular object
        ↓
approach it
        ↓
consume
        ↓
internal energy sensation changes
```

Repeated experience may cause the neural system to represent relationships between:

* perceptual patterns;
* actions;
* environmental changes;
* bodily consequences.

The architecture does not need to explicitly label the object as:

```text
"energy source"
```

The relationship can instead become encoded through prediction.

---

## 9. Homeostasis

Prediction alone explains how the Noralet may learn how the world behaves.

It does not by itself explain why one outcome should matter more than another.

For this reason, Noralets also possess biologically inspired **homeostatic learning signals**.

The body has physically favourable and unfavourable states.

Important factors initially include:

* internal energy;
* physiological condition;
* energetic exertion.

These bodily variables produce subjective interoceptive signals.

---

## 10. Homeostatic Distress

Strongly unfavourable bodily states create a negative learning signal.

Examples include:

* severe energy depletion;
* rapidly worsening energy state;
* poor physiological condition;
* worsening long-term bodily condition.

This signal may operationally be described as:

```text
pain
```

or:

```text
homeostatic distress
```

The term does not imply proof that the Noralet has human-like subjective pain.

It describes the functional role of the signal inside the learning system.

---

## 11. Positive Homeostatic Signal

The positive signal should not simply reward increasing energy without limit.

For example:

```text
E increases → positive reward
```

would create an undesirable incentive to accumulate energy indefinitely.

Instead, positive homeostatic learning should primarily correspond to:

* improvement from a bad bodily state;
* restoration toward a favourable range;
* maintenance of physiological stability.

Conceptually:

```text
critical → poor
    strong improvement

poor → stable
    improvement

stable → stable
    neutral or mildly favourable

stable → poor
    negative

poor → critical
    strongly negative
```

The intended target is homeostatic stability rather than energy maximisation.

---

## 12. Why the Neural Network "Cares"

A neural network does not intrinsically care about any input.

Providing:

```text
pain = 0.8
```

as an ordinary sensory input would not guarantee that the network avoids it.

Homeostatic signals therefore must participate directly in the learning mechanism.

They alter how neural parameters are updated.

Conceptually:

```text
experience
    ↓
action
    ↓
bodily consequence
    ↓
homeostatic change
    ↓
learning signal
    ↓
weight update
```

Actions and internal neural states associated with strongly negative future bodily consequences become less favourable.

Those associated with improved homeostasis may become more favourable.

The precise learning algorithm remains undecided.

---

## 13. Action Evaluation

A candidate NoraletBrain may learn to estimate the expected future homeostatic consequence of its current state and behaviour.

This can be represented by a value-like neural output.

Conceptually:

```text
internal neural state
        ↓
expected future bodily outcome
```

The action-selection system can then learn behavioural preferences based on experienced consequences.

Importantly, the architecture does not contain rules such as:

```python
if energy_low:
    find_energy()
```

Instead, behaviour must develop from learned associations between states, actions and consequences.

---

## 14. Pain and Fear

Pain-like distress and fear-like anticipation should be distinguished.

### 14.1 Pain / distress

Pain-like homeostatic distress corresponds to a currently unfavourable bodily condition.

Conceptually:

```text
bad bodily state now
        ↓
negative homeostatic signal
```

### 14.2 Fear

Fear should not initially exist as a manually defined variable or innate semantic state.

A fear-like state may emerge if the predictive neural system learns that a currently perceived situation is likely to produce future homeostatic distress.

Conceptually:

```text
current perception
+
past experience
+
learned prediction
        ↓
expected future distress
        ↓
anticipatory negative neural state
```

Operationally, such an anticipatory state may later be described as fear.

The Noralet does not need:

```text
fear = 0.8
```

for fear-like behaviour or internal dynamics to emerge.

---

## 15. Learned Motor Understanding

A Noralet is physically capable of producing motor outputs.

It should not necessarily begin with semantic knowledge of their external consequences.

For example, a particular motor output may physically produce acceleration toward one side.

The NoraletBrain may need to learn:

```text
this motor activation
        ↓
these perceptual changes follow
```

Through repeated experience, it may develop a learned body model.

The same principle applies to other actions such as consuming energy or producing signals.

---

## 16. Exploration

Early behaviour cannot depend entirely on already learned optimal actions.

A newly instantiated Noralet requires some mechanism allowing behavioural exploration.

Otherwise, an initially inactive neural policy could remain permanently inactive.

The action system should therefore contain some source of behavioural variation or stochasticity.

The precise exploration mechanism remains undecided.

Exploration may eventually itself become adaptive rather than remaining constant throughout life.

---

## 17. Online Lifetime Learning

Neural learning occurs during the Noralet's life.

The intended loop is:

```text
perceive
   ↓
neural activation
   ↓
act
   ↓
world changes
   ↓
experience consequence
   ↓
prediction / homeostatic errors
   ↓
neural learning
   ↓
next moment
```

Learning should not require stopping the Noralet's life and running a separate offline training phase.

The Noralet continuously exists while its neural system changes.

---

## 18. Temporal Credit Assignment

Consequences may occur several ticks after the action or neural state that contributed to them.

The learning system therefore requires a mechanism for temporal credit assignment.

For a recurrent neural implementation using gradient descent, a candidate mechanism is:

**Truncated Backpropagation Through Time (TBPTT).**

Instead of treating each tick independently, learning may operate across short temporal windows such as:

```text
t
t+1
t+2
...
t+N
```

The exact window size remains an implementation and research parameter.

TBPTT is currently a candidate mechanism rather than a fixed architectural commitment.

---

## 19. BaseBrain Evolution

The long-term BaseBrain training system should imitate selected properties of biological evolution.

The purpose is not to evolve fully competent adult behaviour.

Instead, evolutionary training should produce starting neural systems that are capable of useful lifetime development.

The desired outcome is therefore closer to:

```text
good learner
```

than:

```text
already knows how to survive
```

Evolutionary training may eventually influence:

* initial neural weights;
* architecture;
* sensory organisation;
* recurrent dynamics;
* learning rates;
* plasticity;
* exploration behaviour;
* homeostatic processing.

---

## 20. Learning to Learn

A longer-term possibility is that evolution does not merely optimise the BaseBrain's initial weights.

It may also optimise **how the brain changes during life**.

Conceptually:

```text
evolution
    ↓
learning mechanism
    ↓
individual experience
    ↓
lifetime adaptation
```

This enters the domain of meta-learning or learned plasticity.

It may provide a closer abstraction of biological evolution than manually fixing every lifetime-learning rule.

This is a future research direction and is not required for the first working NoraletBrain.

---

## 21. Death and Neural Continuity

A Noralet's lifetime neural development belongs only to that Noralet.

When the Noralet dies:

```text
persistent neural state → destroyed
learned lifetime weights → destroyed
body → removed
remaining energy → environment
```

There is no automatic transfer of lifetime memories or learned parameters to surviving Noralets.

Without reproduction, the Noralet's individual neural history ends completely at death.

---

## 22. Consciousness and Interpretation

The architecture must distinguish functional observations from claims about subjective experience.

Terms such as:

* pain;
* fear;
* memory;
* expectation;
* preference;

may be used as operational descriptions when neural behaviour satisfies suitable functional definitions.

Their use does not establish that a Noralet experiences these states in the same subjective sense as a human or other biological organism.

One purpose of the project is precisely to observe what kinds of internally coherent processes emerge without assuming the answer beforehand.

---

## 23. Initial Practical Direction

A reasonable first experimental NoraletBrain currently consists of:

```text
Perception Encoder
        ↓
GRU-like Recurrent Core
        ↓
persistent hidden state
        ↓
 ┌─────────────┬─────────────┐
 ↓             ↓             ↓
Action Head  Prediction    Value /
               Head       Evaluation
```

with:

* persistent hidden state across ticks;
* online predictive learning;
* homeostatic learning signals;
* behavioural exploration;
* lifetime weight change.

This is a starting implementation hypothesis.

It should be replaceable if experiments show that a different neural architecture or learning mechanism better satisfies the research goals.

---

## 24. Open Questions

Major unresolved questions include:

* exact neural architecture;
* hidden-state size;
* sensory encoder structure;
* action representation;
* prediction targets;
* predictive loss functions;
* value-learning mechanism;
* exact homeostatic signal function;
* relative weighting of energy and physiological condition;
* positive-signal definition;
* exploration mechanism;
* online optimiser;
* learning rate;
* TBPTT window length;
* gradient stability;
* memory beyond recurrent hidden state;
* neural plasticity during life;
* exact BaseBrain evolutionary training procedure;
* evolutionary fitness or selection mechanism;
* whether architecture itself can evolve;
* whether learning rules should eventually be learned;
* how much innate structure is necessary;
* how to distinguish useful emergent representation from superficial behavioural adaptation.

These are research and implementation questions and should be resolved experimentally.

---

## 25. Core Principle

A Noralet must not merely execute behaviour.

It must **develop**.

Its current behaviour should depend on:

```text
inherited neural structure
+
current internal state
+
current experience
+
its unique previous life experience
```

The BaseBrain provides the machinery with which life begins.

The body provides primitive consequences.

The world provides experience.

The Noralet's own lifetime provides the training data.

What it ultimately learns from that life should not already be written into the architecture.
