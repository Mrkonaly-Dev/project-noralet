"""Qt-independent live-session and shared-baseline tests."""

from __future__ import annotations

import unittest

import torch

from noralet.research.config import (
    LearningCondition,
    build_baseline_components,
)
from noralet.ui.session import LiveRunSetup, create_live_session


class LiveSessionTests(unittest.TestCase):
    def test_ui_session_matches_shared_baseline_factory(self) -> None:
        setup = LiveRunSetup(
            simulation_seed=77,
            population=3,
            device="cpu",
            maximum_ticks=10,
            condition=LearningCondition.PREDICTIVE_ONLY,
        )
        session = create_live_session(setup)
        expected_simulation, expected_base = build_baseline_components(
            initial_population=setup.population,
            device=setup.device,
            condition=setup.condition,
            simulation_seed=setup.simulation_seed,
            base_brain_seed=setup.base_brain_seed,
        )

        self.assertEqual(session.runner.simulation.state, expected_simulation.state)
        expected_parameters = expected_base.parameter_snapshot()
        for identity in session.runner.brain_ids:
            actual_parameters = session.runner.brain_for(identity).parameter_snapshot()
            self.assertEqual(len(actual_parameters), len(expected_parameters))
            for actual, expected in zip(
                actual_parameters,
                expected_parameters,
                strict=True,
            ):
                self.assertTrue(torch.equal(actual, expected))

    def test_one_session_step_is_one_autonomous_tick_and_activation(self) -> None:
        session = create_live_session(
            LiveRunSetup(
                simulation_seed=9,
                population=3,
                device="cpu",
                maximum_ticks=10,
                condition=LearningCondition.NO_LEARNING,
            )
        )
        before = {
            identity: session.runner.brain_for(identity).activation_count
            for identity in session.runner.brain_ids
        }

        result = session.step()

        self.assertIsNotNone(result)
        self.assertEqual(session.tick, 1)
        self.assertEqual(result.tick_result.tick_after, 1)
        for identity, count in before.items():
            self.assertEqual(
                session.runner.brain_for(identity).activation_count,
                count + 1,
            )

    def test_maximum_tick_stops_without_mutating_on_later_requests(self) -> None:
        session = create_live_session(
            LiveRunSetup(
                simulation_seed=3,
                population=2,
                device="cpu",
                maximum_ticks=2,
                condition=LearningCondition.NO_LEARNING,
            )
        )

        self.assertEqual(len(session.step_many(10)), 2)
        state = session.runner.simulation.state
        self.assertIsNone(session.step())
        self.assertIs(session.runner.simulation.state, state)
        self.assertEqual(session.completion_message, "Completed at maximum tick 2")


if __name__ == "__main__":
    unittest.main()
