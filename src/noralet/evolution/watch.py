"""Fresh observer runs from saved inherited champion genomes."""

from __future__ import annotations

from pathlib import Path

from noralet.evolution.engine import load_champion
from noralet.research.config import LearningCondition
from noralet.ui.session import LiveRunSetup, LiveSession, create_live_session


def create_champion_live_session(
    champion_path: Path,
    setup: LiveRunSetup,
) -> tuple[LiveSession, dict[str, object]]:
    """Start a new learned life; this deliberately does not replay evaluation."""

    if not isinstance(setup, LiveRunSetup):
        raise TypeError("setup must be a LiveRunSetup")
    if setup.condition is not LearningCondition.FULL_CURRENT_BRAIN:
        raise ValueError("champion watch requires full-current-brain learning")
    genome, metadata = load_champion(Path(champion_path))
    initial_energy = float(metadata["initial_body_energy"])
    session = create_live_session(
        setup,
        inherited_genome=genome,
        initial_body_energy=initial_energy,
    )
    return session, metadata
