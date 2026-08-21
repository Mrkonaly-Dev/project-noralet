# Predictive Action Selection

**Status:** Initial architecture hypothesis
**Scope:** Tick-level NoraletBrain processing, action-conditioned prediction, future evaluation, action selection and lifetime learning
**Depends on:** `001-learning-foundations.md`

## 1. Purpose

A Noralet should not merely react directly to its current sensory input.

Its neural system should be capable of learning relationships between:

* current experience;
* possible actions;
* expected consequences;
* bodily outcomes.

The intended long-term behaviour is therefore closer to:

```text
What am I experiencing?
        ↓
What could I do?
        ↓
What is likely to happen if I do it?
        ↓
What kind of future would that produce?
        ↓
Act
```

These concepts do not need to exist symbolically inside the Noralet.

They may be represented entirely through learned neural activity.

---

## 2. Persistent Inputs to a Tick

At simulation tick `t`, a living NoraletBrain has three important sources of information:

### Current experience

The perception system produces the Noralet's current sensory and bodily experience.

This may include encoded information derived from:

* local visual-like perception;
* perceived Noralet identities or signatures;
* signal perception;
* subjective energy state;
* physiological condition;
* exertion;
* motor context.

This is represented conceptually as:

```text
perception(t)
```

### Previous hidden state

The NoraletBrain retains a recurrent internal neural state from the previous tick:

```text
h(t-1)
```

This allows present processing to depend on previous experience.

### Learned weights

The neural weights:

```text
W
```

represent the slower structure produced by inherited BaseBrain organisation and the Noralet's lifetime learning.

The three concepts have different roles:

```text
perception(t)
= what is happening now

h(t-1)
= what remains active from recent internal history

W
= how this Noralet has learned to process experience
```

---

## 3. Perception Encoding

Raw sensory information first passes through a neural encoder.

Conceptually:

```text
perception(t)
      ↓
   Encoder
      ↓
     x(t)
```

The purpose of the encoder is to transform heterogeneous sensory channels into a neural representation suitable for recurrent processing.

It must not add semantic labels such as:

```text
food
danger
friend
```

Any such meaning must remain learned.

---

## 4. Recurrent Processing

The encoded present experience is combined with the previous recurrent state.

A candidate initial recurrent mechanism is a GRU.

Conceptually:

```text
x(t)
+
h(t-1)
    ↓
recurrent core
    ↓
h(t)
```

The resulting:

```text
h(t)
```

is the Noralet's current fast internal neural state.

It is recomputed every tick.

The recurrent mechanism may learn to retain, overwrite or reinterpret information from previous moments depending on current experience.

---

## 5. Hidden State and Neural Weights

The architecture distinguishes fast neural state from long-term neural learning.

### Hidden state

```text
h(t-1) → h(t) → h(t+1)
```

The hidden state changes as part of ordinary neural activation.

It does not require a training step to change.

### Weights

```text
W → W'
```

Weights change only through lifetime learning.

Their modification is expected to be much slower and smaller than ordinary tick-to-tick neural activation.

This produces two distinct timescales:

```text
fast:
internal neural activity

slow:
learned neural structure
```

Additional intermediate timescales may be investigated later but are not part of the initial system.

---

## 6. Candidate Actions

The Noralet should not necessarily convert `h(t)` directly into one final deterministic action.

A candidate action mechanism may first produce possible actions or action parameters.

Possible physical outputs currently include:

* no acceleration;
* acceleration toward either direction with some magnitude;
* consumption attempt;
* signal emission;
* signal type;
* signal direction;
* no signal.

Conceptually:

```text
h(t)
  ↓
Action Proposal System
  ↓
candidate actions
```

Examples:

```text
A: do nothing

B: accelerate right

C: consume

D: accelerate left

E: emit signal B to the right
```

These are hypothetical actions.

They do not modify the world until one action configuration is selected and later resolved by the simulation.

---

## 7. Action-Conditioned Prediction

Prediction should eventually be useful for decision-making rather than existing only as an auxiliary training objective.

The prediction system therefore receives both:

```text
current internal state
+
candidate action
```

and attempts to predict the resulting future experience.

Conceptually:

```text
h(t)
+
candidate action
      ↓
Prediction Model
      ↓
predicted experience(t+1)
```

Different candidate actions may therefore produce different predicted futures.

Example:

```text
current state
    │
    ├── do nothing
    │      ↓
    │   predicted future A
    │
    ├── accelerate right
    │      ↓
    │   predicted future B
    │
    └── consume
           ↓
        predicted future C
```

---

## 8. Prediction Contents

The exact prediction target remains configurable.

Initial predictions may include aspects of:

* next visual-like perception;
* next perceived object positions;
* next signal perception;
* next subjective energy state;
* next physiological condition;
* next exertion state.

The initial practical model should focus on predicting **one simulation tick ahead**:

```text
t → t+1
```

Longer internal rollouts may be introduced later.

---

## 9. Learned World and Body Model

Through repeated prediction training, the NoraletBrain may learn relationships such as:

```text
this motor output
→ this kind of spatial change
```

or:

```text
this perceptual object
+
consume action
→ this bodily change
```

The Noralet does not need symbolic rules describing these relationships.

They may exist only as distributed neural representations.

The predictive system therefore acts as a learned model of both:

* external world dynamics;
* consequences for the Noralet's own body.

---

## 10. Future Evaluation

Predicted futures must be evaluated if prediction is to influence action selection.

A value-like neural system may estimate the expected homeostatic quality of a predicted future.

Conceptually:

```text
predicted future
      ↓
Evaluation / Value System
      ↓
expected future value
```

A candidate action leading toward worsening energy distress or physiological condition may receive a lower expected value.

A candidate associated with improving or stabilising homeostasis may receive a higher value.

---

## 11. Value Is Not Semantic Meaning

A value estimate does not mean the Noralet explicitly thinks:

```text
this action has value +0.72
```

The value is an internal neural quantity used during decision-making and learning.

Likewise, the architecture must not manually encode rules such as:

```text
consume = good
boundary = bad
```

The relationship between circumstances, actions and expected bodily consequences must be learned.

---

## 12. Homeostatic Foundation

The deepest externally supplied value structure remains bodily homeostasis.

Important primitive bodily consequences currently include:

* energy distress;
* physiological condition;
* energetic exertion.

The architecture may provide strong negative learning signals for severe bodily deterioration and favourable learning signals for restoration toward stable conditions.

These primitive signals provide the foundation from which more complex learned preferences may develop.

---

## 13. Action Selection

Once candidate actions have predicted and evaluated consequences, the Noralet selects an actual action.

Conceptually:

```text
candidate actions
      ↓
predicted futures
      ↓
evaluations
      ↓
Action Selection
      ↓
action(t)
```

Selection should not necessarily always choose the numerically highest-valued action.

Some exploration or stochasticity is required, especially during early life.

Otherwise, initially arbitrary behaviour could become permanently self-reinforcing.

The exact exploration mechanism remains undecided.

---

## 14. World Isolation During Thought

Internal candidate evaluation must not modify the physical world.

The Noralet may neurally consider several possibilities:

```text
action A
action B
action C
```

but only one becomes:

```text
action(t)
```

The physical simulation remains unchanged until all living Noralets have completed their processing for the tick.

This preserves the lockstep world model.

---

## 15. Shared Tick State

Every living Noralet perceives the same objective:

```text
WorldState(t)
```

from its own local perspective.

One Noralet being processed earlier by the computer must not give it access to physical changes produced by another Noralet during the same tick.

The general sequence is:

```text
WorldState(t)

       ↓

all Noralets perceive

       ↓

all NoraletBrains process

       ↓

all action intents collected

       ↓

central resolution

       ↓

WorldState(t+1)
```

---

## 16. Obtaining Real Experience

After action resolution, the simulation produces:

```text
WorldState(t+1)
```

The Noralet then receives:

```text
actual perception(t+1)
```

This provides ground truth for the prediction associated with the action it actually selected.

---

## 17. Prediction Error

The chosen action's predicted future can now be compared with real experience.

Conceptually:

```text
predicted experience(t+1)
        vs.
actual experience(t+1)
        ↓
prediction error
```

This error can train:

* the prediction system;
* the recurrent core;
* the perception encoder.

The shared recurrent representation therefore becomes increasingly useful for understanding the dynamics of the Noralet's experienced world.

---

## 18. Counterfactual Predictions

A Noralet may internally evaluate several candidate actions, but only one is physically performed.

Therefore, only the selected action provides direct real-world ground truth for that tick.

Example:

```text
imagined:
A
B
C

performed:
B
```

The simulation reveals what happened after `B`.

It does **not** reveal what would actually have happened after `A` or `C`.

Predictions for unselected actions remain counterfactual estimates learned indirectly from previous real experiences and generalisation.

The learning system must not receive artificial ground-truth outcomes for events that never occurred.

---

## 19. Homeostatic Outcome

The new tick also reveals the real bodily consequence of the previous action and world transition.

For example:

```text
energy distress:
high → low
```

or:

```text
physiological condition:
stable → worse
```

These changes produce homeostatic learning information.

This answers a different question from predictive error:

```text
prediction error:
"Was my model of what would happen correct?"

homeostatic signal:
"Was what actually happened physically favourable or unfavourable?"
```

Both are important.

---

## 20. Value Learning

The value/evaluation system should learn whether its expected bodily future matched the outcomes actually experienced.

Conceptually:

```text
expected future value
        vs.
experienced consequences
        ↓
value-learning error
```

This allows the Noralet to increasingly estimate which situations and behavioural directions are associated with favourable or unfavourable futures.

Exact temporal value-learning mathematics remain undecided.

Long-term consequences will require some form of temporal credit assignment.

---

## 21. Action Learning

The action-selection system must learn from the consequences of selected actions.

If an action repeatedly contributes to favourable future states under a particular internal context, it should become more likely to be selected in similar situations.

If it repeatedly contributes to strongly unfavourable futures, it should become less likely.

This must happen through neural learning rather than manually written behavioural rules.

The precise algorithm remains unresolved.

Because discrete action selection is not automatically differentiable, the first implementation will require an explicit learning method such as a policy/value learning approach or another suitable mechanism.

This should be chosen deliberately during implementation research rather than hidden behind generic backpropagation terminology.

---

## 22. Shared Representation

The action, prediction and value systems should share a substantial common neural representation.

Conceptually:

```text
                ┌→ action-related processing
                │
perception → recurrent core
                │
                ├→ prediction
                │
                └→ value evaluation
```

The individual output components may possess separate weights.

However, their learning may update the shared encoder and recurrent core.

This allows predictive learning to improve representations later used for decision-making.

The architecture should therefore not be understood as three independent brains.

They are specialised functions operating over a shared learned internal state.

---

## 23. Anticipation

This architecture allows behaviour to respond to predicted future consequences rather than only current distress.

For example, a Noralet may currently have:

```text
energy condition: good
physiological condition: good
```

while moving toward a dangerous world boundary.

If its learned model predicts:

```text
continue current behaviour
→ increasingly dangerous future
```

and an alternative action predicts:

```text
decelerate
→ safer future
```

the Noralet may act before suffering any current bodily harm.

This is functional anticipation.

---

## 24. Fear-Like States

A fear-like state may emerge when:

```text
current perception
+
current internal state
        ↓
prediction of future severe distress
        ↓
strong anticipatory negative neural state
```

No explicit:

```text
fear = value
```

variable is required.

If a stable internal neural pattern develops that predicts and helps avoid future severe bodily consequences, it may later be operationally investigated as a fear-like representation.

The term does not imply proof of human-like subjective fear.

---

## 25. One-Step Prediction First

The initial practical architecture should use short-horizon prediction:

```text
t → t+1
```

This makes training tractable and provides directly observable targets.

The recurrent system and value-learning mechanism may still encode longer-term consequences.

Once one-step modelling works reliably, future work may investigate multi-step internal rollouts:

```text
candidate action
      ↓
predicted t+1
      ↓
predicted t+2
      ↓
predicted t+3
```

Such internal simulation would move the system closer to explicit learned planning or imagination.

It is intentionally deferred.

---

## 26. Tick-Level Neural Lifecycle

A conceptual neural tick is:

```text
WorldState(t)
      ↓
Noralet perception(t)
      ↓
Encoder
      ↓
x(t)
      ↓
GRU + h(t-1)
      ↓
h(t)
      ↓
candidate actions
      ↓
action-conditioned predictions
      ↓
future evaluations
      ↓
exploratory action selection
      ↓
action intent
```

After all Noralets complete this process:

```text
all action intents
      ↓
central world resolution
      ↓
WorldState(t+1)
```

The next experience then provides:

```text
actual perception(t+1)
+
actual bodily consequence
```

which supports lifetime learning.

---

## 27. Learning Lifecycle

The complete conceptual cycle is:

```text
EXPERIENCE
    ↓
RECURRENT NEURAL ACTIVATION
    ↓
POSSIBLE ACTIONS
    ↓
PREDICTED CONSEQUENCES
    ↓
EXPECTED VALUE
    ↓
ACTION
    ↓
REAL WORLD CONSEQUENCE
    ↓
NEW EXPERIENCE
    ↓
 ┌────────────────────────────┐
 │ Prediction error           │
 │ Homeostatic consequence    │
 │ Value-learning error       │
 │ Action-learning signal     │
 └────────────────────────────┘
    ↓
NEURAL WEIGHT CHANGE
    ↓
NEXT EXPERIENCE
```

The Noralet continues existing throughout this process.

---

## 28. What Is Innate

The architecture may provide:

* neural structure;
* recurrent dynamics;
* perception pathways;
* motor pathways;
* predictive-learning capability;
* value-learning capability;
* homeostatic bodily signals;
* ability to explore;
* ability to change neural weights.

It must not directly provide:

* semantic object meanings;
* signal meanings;
* social meanings;
* movement understanding;
* environmental map knowledge;
* predetermined strategies;
* high-level goals.

---

## 29. What Must Be Learned

The Noralet may need to learn relationships such as:

```text
what different perceptions correspond to

what its own motor outputs cause

how objects behave

how other Noralets behave

what signals tend to predict

what actions alter bodily state

what circumstances tend to precede distress

what circumstances tend to restore stability

which action consequences are likely

which behaviours are useful over time
```

The exact internal representations are not prescribed.

---

## 30. Initial Implementation Hypothesis

A reasonable first neural prototype is:

```text
sensory inputs
      ↓
Perception Encoder
      ↓
GRU recurrent core
      ↓
persistent hidden state
      ↓
 ┌───────────────┬────────────────┐
 │               │                │
action-related   prediction       value
processing       model            model
 │               │                │
 └───────────────┴────────────────┘
                 ↓
           action selection
```

The precise PyTorch module graph is not fixed by this document.

Implementation experiments should remain free to simplify or revise this architecture while preserving the learning principles described above.

---

## 31. Open Questions

The following remain unresolved:

* exact action representation;
* whether candidate actions are explicitly enumerated or parametrically sampled;
* number of candidates evaluated per tick;
* prediction target representation;
* prediction-loss functions;
* recurrent hidden-state size;
* perception-encoder architecture;
* value-learning algorithm;
* action/policy-learning algorithm;
* exploration strategy;
* temporal discounting, if any;
* temporal credit-assignment mechanism;
* frequency of neural weight updates;
* training-window size;
* whether prediction and decision learning use the same optimiser;
* relative weighting of learning objectives;
* handling of simultaneous continuous and discrete actions;
* computational cost per Noralet;
* batching strategy across independently learning Noralets;
* future multi-step internal rollouts.

These are implementation and research questions and should be tested experimentally rather than fixed without evidence.

---

## 32. Core Principle

The Noralet should not merely learn:

> **When I see X, perform Y.**

It should increasingly be capable of learning:

> **When I experience something like this, different actions tend to produce different futures, and those futures have different consequences for my continued bodily state.**

Prediction makes experience understandable.

Homeostasis makes consequences matter.

Recurrent state connects moments.

Lifetime learning changes the Noralet because of what it has lived through.

Action selection turns those learned relationships back into behaviour.
