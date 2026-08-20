# Project Noralet — Documentation Protocol

Project Noralet maintains separate documentation domains for research, system design, and implementation history.

The purpose of this separation is to preserve a clear distinction between:

* what the project is trying to understand;
* how the experimental system is designed;
* and what was actually implemented during each development iteration.

All agents and contributors working on the repository may freely consult these documents when performing their tasks.

---

## 1. Documentation Structure

```text
/
├── README.md
├── DOCUMENTATION.md
│
├── research-docs/
├── architecture-docs/
└── codex-reports/
```

Each directory serves a different purpose and should not be used interchangeably.

---

# 2. `research-docs/`

`research-docs/` contains the scientific, conceptual, philosophical, and experimental record of Project Noralet.

It answers questions such as:

* What is Project Noralet trying to discover?
* What hypotheses are being investigated?
* What do we currently mean by concepts such as consciousness, self-model, emergence, or agency?
* How should experiments be structured?
* What was observed?
* How should the observations be interpreted?
* What alternative explanations exist?
* What ethical questions arise?

Research documents should describe the project independently of implementation details wherever possible.

Suggested structure:

```text
research-docs/
├── README.md
│
├── 00-foundations/
│   ├── 001-core-premise.md
│   ├── 002-research-questions.md
│   └── 003-terminology.md
│
├── 01-hypotheses/
├── 02-experiments/
├── 03-observations/
├── 04-analyses/
├── 05-ethics/
└── 06-conclusions/
```

The first foundational document is:

```text
research-docs/00-foundations/001-core-premise.md
```

Research documentation is primarily developed through the dedicated **Project Noralet Research discussion** and written with ChatGPT.

The documents should preserve the reasoning and assumptions that existed at the relevant stage of the project.

Research conclusions must not be silently rewritten to match later results.

When scientific interpretation materially changes, the change should be documented explicitly.

---

# 3. `architecture-docs/`

`architecture-docs/` contains the technical and architectural design of the experimental system.

It answers questions such as:

* How are Noralets represented?
* How does the simulated world operate?
* How do perception, action, learning, memory, or communication work?
* How is experimental state persisted?
* How is observability implemented?
* How are experiments reproduced?
* What technical constraints influence the system?
* Why were specific architectural choices made?

Suggested structure:

```text
architecture-docs/
├── README.md
│
├── 00-overview/
├── 01-system/
├── 02-agents/
├── 03-world/
├── 04-learning/
├── 05-observability/
├── 06-data/
└── 07-decisions/
```

Architecture documents describe the intended design.

They are not implementation reports.

Architecture documentation will normally be developed through architectural discussions with ChatGPT before or during implementation.

---

## 3.1 Architecture Decision Records

Important architectural decisions should be preserved as Architecture Decision Records where useful.

Example:

```text
architecture-docs/07-decisions/
├── ADR-001-example-decision.md
├── ADR-002-example-decision.md
└── ...
```

An ADR should generally record:

```text
Status
Context
Decision
Alternatives considered
Reasoning
Consequences
```

ADRs exist to preserve **why** the system was designed a certain way, not merely what the resulting implementation looks like.

Once accepted and implemented, an ADR should normally remain in the historical record.

If a decision is later replaced, the original ADR should be marked as superseded rather than silently rewritten.

---

# 4. `codex-reports/`

`codex-reports/` contains implementation and operation reports produced by Codex.

Example:

```text
codex-reports/
├── operation-report-001.md
├── operation-report-002.md
├── operation-report-003.md
└── ...
```

Each report corresponds to an implementation or development iteration.

Codex reports describe what actually happened during that iteration.

They may include:

* changes made;
* files created or modified;
* architectural components implemented;
* migrations;
* tests added or executed;
* test results;
* discovered issues;
* deviations from the requested design;
* technical compromises;
* unresolved concerns;
* relevant follow-up work.

The reports form an implementation audit trail.

They should be detailed enough for later review without requiring the reviewer to reconstruct the entire iteration from the Git diff alone.

Codex reports should generally be treated as historical records and should not be rewritten after the iteration except to correct clear factual errors.

---

# 5. Relationship Between Documentation and Implementation

The documentation hierarchy should normally be interpreted as:

```text
Research intent
      ↓
Architecture
      ↓
Implementation
      ↓
Codex report
      ↓
Review / research observation
```

Research documents define what is being investigated.

Architecture documents translate relevant research requirements into system design.

Codex implements the requested work using those documents as context.

Codex then creates an operation report describing the resulting implementation.

The implementation and report may subsequently be reviewed against both the architecture and research requirements.

---

# 6. Codex Access to Documentation

Codex is explicitly allowed and encouraged to browse:

```text
research-docs/
architecture-docs/
codex-reports/
README.md
DOCUMENTATION.md
```

when those documents are relevant to the current task.

Codex should use the documentation to understand:

* project intent;
* terminology;
* architectural constraints;
* previous decisions;
* experimental requirements;
* previous implementation history.

Existing documentation should be treated as project context, not as immutable truth.

If current instructions conflict with older documentation, the conflict should be surfaced rather than silently resolved through guesswork.

---

# 7. Source-of-Truth Boundaries

No single documentation domain is the universal source of truth for every question.

Use the following rule:

### Scientific or conceptual intent

Source:

```text
research-docs/
```

### Intended technical design

Source:

```text
architecture-docs/
```

### Actual current implementation

Source:

```text
source code
database schema
tests
configuration
```

### Historical implementation activity

Source:

```text
codex-reports/
Git history
```

An architecture document may describe intended behavior that has not yet been implemented.

A Codex report may describe an implementation that was later replaced.

The current source code and tests therefore remain authoritative for what the software currently does.

---

# 8. Document Naming

Documents should use stable numeric prefixes where ordering matters.

Example:

```text
001-core-premise.md
002-research-questions.md
003-terminology.md
```

Use lowercase kebab-case for filenames.

Prefer descriptive names over generic names such as:

```text
notes.md
new-plan.md
ideas.md
final.md
```

Avoid filenames whose meaning depends on the current date or conversation context.

---

# 9. Document Metadata

Important research and architecture documents should normally begin with metadata similar to:

```yaml
---
project: Project Noralet
document: Core Premise
id: FOUND-001
version: 0.1
status: draft
created: 2026-08-19
---
```

Possible statuses include:

```text
draft
active
accepted
superseded
archived
```

Additional metadata may be introduced later if needed.

Git remains the authoritative history of textual modifications.

Document version numbers should represent meaningful conceptual revisions rather than every minor edit.

---

# 10. Preservation of Research History

Project Noralet investigates questions where hindsight bias could significantly distort interpretation.

For this reason, foundational hypotheses, predictions, assumptions, and experimental criteria should be documented **before** the relevant experiment whenever practical.

Later results should not be retroactively inserted into earlier predictions.

Instead, create:

* observations;
* analyses;
* revised hypotheses;
* or new versions of the relevant document.

This preserves the distinction between:

**what was expected before the experiment**

and

**what was concluded after seeing the result.**

This distinction is particularly important for claims relating to emergent cognition or consciousness.

---

# 11. Documentation Philosophy

Documentation exists to make Project Noralet understandable and auditable, not to maximize document count.

Create a document when it preserves information that will remain useful across future conversations, iterations, experiments, or reviews.

Avoid documenting temporary implementation chatter that is already adequately represented by:

* the source code;
* Git commits;
* or a Codex operation report.

The system should remain rigorous without becoming bureaucratic.

---

# 12. General Rule

Before creating a new document, determine what kind of question it answers:

> **Why are we investigating this?**
> → `research-docs/`

> **How should the system work?**
> → `architecture-docs/`

> **What did Codex implement?**
> → `codex-reports/`

Maintaining this separation is part of the Project Noralet research protocol.
