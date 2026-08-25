"""Compact external body, Experience and learning inspector."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from noralet.ui.session import LiveSession


_EM_DASH = "—"

_BODY_FIELDS = (
    ("noralet_id", "Noralet ID"),
    ("age_ticks", "Age ticks"),
    ("position", "Position"),
    ("velocity", "Velocity"),
    ("stored_energy", "Stored Energy"),
    ("condition", "Condition"),
)

_EXPERIENCE_FIELDS = (
    ("energy_distress", "Energy distress"),
    ("condition_distress", "Condition distress"),
    ("energetic_exertion", "Energetic exertion"),
    ("external_percept_count", "External percepts"),
    ("signal_percept_count", "Signal percepts"),
)

_LEARNING_FIELDS = (
    ("prediction_loss", "Prediction loss"),
    ("gradient_norm", "Gradient norm"),
    ("homeostatic_drive_before", "Drive before"),
    ("homeostatic_drive_after", "Drive after"),
    ("homeostatic_modulation", "Modulation"),
    ("eligibility_norm", "Eligibility norm"),
    ("homeostatic_update_norm", "Update norm"),
)


class NoraletInspector(QWidget):
    """Display observer-only values without storing them in the simulation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value_labels: dict[str, QLabel] = {}
        self._values: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        title = QLabel("NORALET INSPECTOR")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addWidget(self._group("Identity / body", _BODY_FIELDS))
        layout.addWidget(self._group("Current Experience", _EXPERIENCE_FIELDS))
        layout.addWidget(self._group("Latest learning metrics", _LEARNING_FIELDS))
        layout.addStretch(1)
        self.clear()

    @property
    def values(self) -> dict[str, str]:
        return dict(self._values)

    def value_for(self, key: str) -> str:
        return self._values[key]

    def _group(
        self,
        title: str,
        fields: tuple[tuple[str, str], ...],
    ) -> QGroupBox:
        group = QGroupBox(title)
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 12, 8, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        for row, (key, label_text) in enumerate(fields):
            label = QLabel(label_text)
            value = QLabel(_EM_DASH)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value.setObjectName("inspectorValue")
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            self._value_labels[key] = value
        return group

    def clear(self) -> None:
        for key, label in self._value_labels.items():
            self._values[key] = _EM_DASH
            label.setText(_EM_DASH)

    def refresh(self, session: LiveSession | None, noralet_id: int | None) -> None:
        if session is None or noralet_id is None:
            self.clear()
            return
        try:
            body = session.runner.simulation.state.body(noralet_id)
            experience = session.runner.simulation.experience_for(noralet_id)
        except KeyError:
            self.clear()
            return

        values: dict[str, object | None] = {
            "noralet_id": body.noralet_id,
            "age_ticks": body.age_ticks,
            "position": body.position,
            "velocity": body.velocity,
            "stored_energy": body.energy,
            "condition": body.condition,
            "energy_distress": experience.interoception.energy_distress,
            "condition_distress": experience.interoception.condition_distress,
            "energetic_exertion": experience.interoception.energetic_exertion,
            "external_percept_count": len(experience.external_percepts),
            "signal_percept_count": len(experience.signal_percepts),
            "prediction_loss": None,
            "gradient_norm": None,
            "homeostatic_drive_before": None,
            "homeostatic_drive_after": None,
            "homeostatic_modulation": None,
            "eligibility_norm": None,
            "homeostatic_update_norm": None,
        }
        predictive = session.latest_learning.get(noralet_id)
        if predictive is not None:
            values["prediction_loss"] = predictive.prediction_loss
            values["gradient_norm"] = predictive.gradient_norm
        homeostatic = session.latest_homeostatic.get(noralet_id)
        if homeostatic is not None:
            values.update(
                {
                    "homeostatic_drive_before": (
                        homeostatic.homeostatic_drive_before
                    ),
                    "homeostatic_drive_after": homeostatic.homeostatic_drive_after,
                    "homeostatic_modulation": homeostatic.modulation,
                    "eligibility_norm": homeostatic.eligibility_norm,
                    "homeostatic_update_norm": homeostatic.applied_update_norm,
                }
            )
        for key, value in values.items():
            text = self._format(value)
            self._values[key] = text
            self._value_labels[key].setText(text)

    @staticmethod
    def _format(value: object | None) -> str:
        if value is None:
            return _EM_DASH
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
