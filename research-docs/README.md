# Research Documentation

This directory contains the human-interpreted research record for Project Noralet.

Keep this directory flat until the number of research documents makes further organization genuinely useful.

Each numbered research document should capture, where applicable:

- the research question or hypothesis;
- the experimental protocol;
- quantitative results;
- qualitative observations;
- interpretation;
- limitations and confounds;
- resulting research or architecture questions.

Generated machine-readable datasets belong in `research-results/`.

Implementation and validation reports produced by Codex belong in `codex-reports/`.

Architecture specifications remain in `architecture-docs/`.

## Research discipline

Project Noralet should distinguish clearly between:

1. **observation** — what was measured or seen;
2. **interpretation** — what may explain the observation;
3. **controlled evidence** — what has been isolated experimentally;
4. **architecture decision** — what is changed because of the evidence.

A surprising behavior should be recorded before it is explained.

> Record surprising behavior first; explain it only after controlled experiments.

Renderer observations are valid qualitative research notes, but visual behavior must not be described as intentional, social, goal-directed, intelligent or conscious unless a controlled experiment supports that interpretation.

## Current documents

- `001-baseline-lifetime-adaptation.md` — first quantitative lifetime-learning pilot and follow-up qualitative renderer observations.
