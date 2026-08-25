"""PySide6 Qt Widgets shell for live observation and research launching."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time

from PySide6.QtCore import QProcess, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noralet.research.config import LearningCondition
from noralet.ui.canvas import WorldCanvas
from noralet.ui.inspector import NoraletInspector
from noralet.ui.research_launcher import (
    ProcessInvocation,
    ResearchLaunchSetup,
    build_research_invocation,
    latest_partial_result_directory,
    result_directory_from_line,
)
from noralet.ui.session import LiveRunSetup, LiveSession, create_live_session


_CONDITION_LABELS = {
    LearningCondition.NO_LEARNING: "No learning",
    LearningCondition.PREDICTIVE_ONLY: "Predictive only",
    LearningCondition.HOMEOSTATIC_ONLY: "Homeostatic only",
    LearningCondition.FULL_CURRENT_BRAIN: "Full current brain",
}

_SPEED_PRESETS = {
    "1x": (1, 100),
    "10x": (2, 25),
    "100x": (4, 5),
    "Max": (8, 0),
}


def _positive_spinbox(
    *,
    value: int,
    maximum: int,
) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(1, maximum)
    widget.setValue(value)
    return widget


def _ensure_observer_fonts() -> None:
    """Load Windows system fonts when the offscreen plugin exposes none."""

    if QFontDatabase.families():
        return
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = (
        windows_root / "Fonts" / "segoeui.ttf",
        windows_root / "Fonts" / "consola.ttf",
    )
    for path in candidates:
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    application = QApplication.instance()
    if isinstance(application, QApplication):
        application.setFont(QFont("Segoe UI", 10))


class LiveSimulationWidget(QWidget):
    """Timer-driven watch mode with a strictly observer-only presentation."""

    session_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session: LiveSession | None = None
        self._ticks_per_pulse = 1
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_timer)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addLayout(self._controls())

        self.canvas = WorldCanvas()
        self.inspector = NoraletInspector()
        self.inspector.setMinimumWidth(300)
        self.inspector.setMaximumWidth(380)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)
        root.addWidget(self._setup_group())

        self.canvas.selection_changed.connect(self._selection_changed)
        self.set_speed_mode("1x")
        self._refresh_observer_widgets()

    @property
    def timer_active(self) -> bool:
        return self._timer.isActive()

    def _controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        for button in (
            self.start_button,
            self.pause_button,
            self.step_button,
            self.reset_button,
        ):
            layout.addWidget(button)
        self.start_button.clicked.connect(self.start)
        self.pause_button.clicked.connect(self.pause)
        self.step_button.clicked.connect(self.step_once)
        self.reset_button.clicked.connect(lambda: self.reset_run())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(separator)
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_buttons: dict[str, QPushButton] = {}
        for label in _SPEED_PRESETS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, mode=label: self.set_speed_mode(mode)
            )
            self.speed_group.addButton(button)
            self.speed_buttons[label] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self.tick_label = QLabel("Tick: 0")
        self.tick_label.setObjectName("tickLabel")
        layout.addWidget(self.tick_label)
        return layout

    def _setup_group(self) -> QGroupBox:
        group = QGroupBox("Baseline world · run setup")
        grid = QGridLayout(group)
        self.seed_edit = QLineEdit("1")
        self.population_spin = _positive_spinbox(value=6, maximum=64)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA", "cuda")
        self.maximum_ticks_spin = _positive_spinbox(
            value=5_000,
            maximum=10_000_000,
        )
        self.condition_combo = QComboBox()
        for condition, label in _CONDITION_LABELS.items():
            self.condition_combo.addItem(label, condition.value)
        self.condition_combo.setCurrentIndex(
            self.condition_combo.findData(
                LearningCondition.FULL_CURRENT_BRAIN.value
            )
        )
        fields = (
            ("Simulation seed", self.seed_edit),
            ("Population", self.population_spin),
            ("Device", self.device_combo),
            ("Maximum ticks", self.maximum_ticks_spin),
            ("Learning mode", self.condition_combo),
        )
        for column, (label, widget) in enumerate(fields):
            grid.addWidget(QLabel(label), 0, column)
            grid.addWidget(widget, 1, column)
        self.status_label = QLabel("Ready · configure, then Start or Reset")
        self.status_label.setObjectName("statusLabel")
        grid.addWidget(self.status_label, 2, 0, 1, len(fields))
        return group

    def current_setup(self) -> LiveRunSetup:
        seed_text = self.seed_edit.text().strip()
        if not seed_text or seed_text in ("-", "+"):
            raise ValueError("Simulation seed must be an integer")
        return LiveRunSetup(
            simulation_seed=int(seed_text),
            population=self.population_spin.value(),
            device=str(self.device_combo.currentData()),
            maximum_ticks=self.maximum_ticks_spin.value(),
            condition=LearningCondition(str(self.condition_combo.currentData())),
        )

    def reset_run(self, *, show_errors: bool = True) -> bool:
        self.pause()
        try:
            session = create_live_session(self.current_setup())
        except Exception as error:
            self.session = None
            self.canvas.set_session(None)
            self.inspector.clear()
            self.status_label.setText(f"Setup failed: {error}")
            if show_errors:
                QMessageBox.critical(self, "Could not create run", str(error))
            return False
        self.session = session
        self.canvas.set_session(session)
        self.status_label.setText(
            f"Paused · {session.setup.condition.value} · "
            f"{session.setup.device}"
        )
        self.session_changed.emit(session)
        self._refresh_observer_widgets()
        return True

    def start(self) -> None:
        if self.session is None and not self.reset_run():
            return
        assert self.session is not None
        if not self.session.can_step:
            self.status_label.setText(
                self.session.completion_message or "Run completed"
            )
            return
        _, interval = _SPEED_PRESETS[self.current_speed_mode]
        self._timer.start(interval)
        self.status_label.setText(
            f"Running · {self.current_speed_mode} · "
            f"{self.session.setup.condition.value}"
        )

    def pause(self) -> None:
        self._timer.stop()
        if self.session is not None and self.session.can_step:
            self.status_label.setText(f"Paused at tick {self.session.tick}")

    def step_once(self) -> None:
        self.pause()
        if self.session is None and not self.reset_run():
            return
        self.advance_burst(1)

    @property
    def current_speed_mode(self) -> str:
        for mode, button in self.speed_buttons.items():
            if button.isChecked():
                return mode
        return "1x"

    def set_speed_mode(self, mode: str) -> None:
        if mode not in _SPEED_PRESETS:
            raise ValueError(f"unknown speed mode: {mode}")
        self._ticks_per_pulse, interval = _SPEED_PRESETS[mode]
        self.speed_buttons[mode].setChecked(True)
        if self._timer.isActive():
            self._timer.start(interval)
            self.status_label.setText(f"Running · {mode}")

    def advance_burst(self, count: int) -> tuple[object, ...]:
        if self.session is None:
            raise RuntimeError("no live session exists")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        results = []
        for _ in range(count):
            result = self.session.step()
            if result is None:
                break
            results.append(result)
            self.canvas.observe_latest_result()
        self._refresh_observer_widgets()
        self._stop_if_complete()
        return tuple(results)

    def _on_timer(self) -> None:
        try:
            self.advance_burst(self._ticks_per_pulse)
        except Exception as error:
            self.pause()
            self.status_label.setText(f"Run failed: {error}")
            QMessageBox.critical(self, "Live simulation failed", str(error))

    def _stop_if_complete(self) -> None:
        if self.session is None:
            return
        message = self.session.completion_message
        if message is not None:
            self._timer.stop()
            self.status_label.setText(message)

    def _selection_changed(self, noralet_id: object) -> None:
        selected = noralet_id if isinstance(noralet_id, int) else None
        self.inspector.refresh(self.session, selected)

    def _refresh_observer_widgets(self) -> None:
        tick = 0 if self.session is None else self.session.tick
        self.tick_label.setText(f"Tick: {tick}")
        self.canvas.synchronize_selection()
        self.inspector.refresh(self.session, self.canvas.selected_id)
        self.canvas.update()

    def shutdown(self) -> None:
        self._timer.stop()
        self.canvas.set_session(None)
        self.session = None


class ResearchWidget(QWidget):
    """QProcess launcher/progress view over the unchanged research CLI."""

    _PROGRESS_PATTERN = re.compile(
        r"\[(?P<current>\d+)/(?P<total>\d+)\]\s+"
        r"(?P<condition>[a-z-]+), replicate seed (?P<replicate>-?\d+)"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None
        self.result_directory = None
        self._output_buffer = ""
        self._started_at = 0.0
        self._invocation: ProcessInvocation | None = None
        self._cancel_requested = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        title = QLabel("001 · BASELINE LIFETIME ADAPTATION")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(self._setup_group())
        root.addLayout(self._buttons())
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("statusLabel")
        root.addWidget(self.progress_label)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(2_000)
        self.output.setPlaceholderText("Headless research progress appears here.")
        root.addWidget(self.output, 1)
        self.result_label = QLabel("Result directory: —")
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.result_label)

    def _setup_group(self) -> QGroupBox:
        group = QGroupBox("Research 001 protocol")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.seeds_spin = _positive_spinbox(value=10, maximum=10_000)
        self.seeds_spin.setMinimum(2)
        self.maximum_ticks_spin = _positive_spinbox(
            value=5_000,
            maximum=10_000_000,
        )
        self.sample_every_spin = _positive_spinbox(
            value=10,
            maximum=1_000_000,
        )
        self.population_spin = _positive_spinbox(value=6, maximum=64)
        self.device_combo = QComboBox()
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("Auto", "auto")
        form.addRow("Seeds", self.seeds_spin)
        form.addRow("Maximum ticks", self.maximum_ticks_spin)
        form.addRow("Sample every", self.sample_every_spin)
        form.addRow("Population", self.population_spin)
        form.addRow("Device", self.device_combo)
        layout.addLayout(form)
        checks = QHBoxLayout()
        self.condition_checks: dict[LearningCondition, QCheckBox] = {}
        for condition, label in _CONDITION_LABELS.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self.condition_checks[condition] = checkbox
            checks.addWidget(checkbox)
        checks.addStretch(1)
        layout.addLayout(checks)
        return group

    def _buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.run_button = QPushButton("Run experiment")
        self.stop_button = QPushButton("Stop")
        self.open_button = QPushButton("Open result folder")
        self.stop_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.run_button.clicked.connect(lambda: self.start_experiment())
        self.stop_button.clicked.connect(self.stop_experiment)
        self.open_button.clicked.connect(self.open_result_folder)
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.open_button)
        layout.addStretch(1)
        return layout

    def current_setup(self) -> ResearchLaunchSetup:
        conditions = tuple(
            condition
            for condition, checkbox in self.condition_checks.items()
            if checkbox.isChecked()
        )
        return ResearchLaunchSetup(
            seeds=self.seeds_spin.value(),
            maximum_ticks=self.maximum_ticks_spin.value(),
            sample_every_ticks=self.sample_every_spin.value(),
            population=self.population_spin.value(),
            device=str(self.device_combo.currentData()),
            conditions=conditions,
        )

    def invocation_for_current_fields(self) -> ProcessInvocation:
        return build_research_invocation(self.current_setup())

    def start_experiment(self, *, show_errors: bool = True) -> bool:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            return False
        try:
            invocation = self.invocation_for_current_fields()
        except Exception as error:
            self.progress_label.setText(f"Invalid setup: {error}")
            if show_errors:
                QMessageBox.warning(self, "Invalid research setup", str(error))
            return False

        self.output.clear()
        self.result_directory = None
        self.result_label.setText("Result directory: —")
        self.open_button.setEnabled(False)
        self._output_buffer = ""
        self._cancel_requested = False
        self._started_at = time.time()
        self._invocation = invocation
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(invocation.working_directory))
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_label.setText("Starting headless Research 001 process…")
        process.start(invocation.program, list(invocation.arguments))
        return True

    def stop_experiment(self) -> None:
        if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self._cancel_requested = True
        self.progress_label.setText("Cancellation requested…")
        self.process.terminate()
        QTimer.singleShot(1_500, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        if not text:
            return
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.ensureCursorVisible()
        self._output_buffer += text
        lines = self._output_buffer.splitlines(keepends=True)
        self._output_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._output_buffer = lines.pop()
        for line in lines:
            self._parse_output_line(line)

    def _parse_output_line(self, line: str) -> None:
        match = self._PROGRESS_PATTERN.search(line)
        if match is not None:
            self.progress_label.setText(
                f"Run {match.group('current')} / {match.group('total')} · "
                f"{match.group('condition')} · replicate "
                f"{match.group('replicate')}"
            )
        if self._invocation is not None:
            result = result_directory_from_line(
                line,
                self._invocation.working_directory,
            )
            if result is not None:
                self.result_directory = result
                self.result_label.setText(f"Result directory: {result}")

    def _process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_output()
        if self._output_buffer:
            self._parse_output_line(self._output_buffer)
            self._output_buffer = ""
        normal = exit_status == QProcess.ExitStatus.NormalExit
        successful = normal and exit_code == 0 and not self._cancel_requested
        if successful:
            self.progress_label.setText("Research batch completed successfully")
        else:
            if self.result_directory is None and self._invocation is not None:
                self.result_directory = latest_partial_result_directory(
                    self._invocation.working_directory,
                    started_after=self._started_at,
                )
            if self._cancel_requested:
                self.progress_label.setText("Research batch canceled · partial output may exist")
            else:
                self.progress_label.setText(
                    f"Research process failed with exit code {exit_code}"
                )
        if self.result_directory is not None:
            self.result_label.setText(
                f"{'Partial r' if not successful else 'R'}esult directory: "
                f"{self.result_directory}"
            )
            self.open_button.setEnabled(self.result_directory.is_dir())
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.Crashed and self._cancel_requested:
            return
        if self.process is not None:
            self.progress_label.setText(
                f"Research process error: {self.process.errorString()}"
            )
        if error == QProcess.ProcessError.FailedToStart:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def open_result_folder(self) -> None:
        if self.result_directory is None or not self.result_directory.is_dir():
            QMessageBox.warning(self, "Result unavailable", "No result folder is available.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_directory)))

    def shutdown(self) -> None:
        if self.process is None:
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1_500):
                self.process.kill()
                self.process.waitForFinished(1_000)
        self.process.deleteLater()
        self.process = None


class NoraletMainWindow(QMainWindow):
    """Single compact desktop shell for live observation and Research 001."""

    def __init__(self) -> None:
        super().__init__()
        _ensure_observer_fonts()
        self.setWindowTitle("Project Noralet · Observer UI")
        self.resize(1280, 780)
        self.setMinimumSize(1020, 650)
        self.tabs = QTabWidget()
        self.live = LiveSimulationWidget()
        self.research = ResearchWidget()
        self.tabs.addTab(self.live, "Live Simulation")
        self.tabs.addTab(self.research, "Research")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.live.shutdown()
        self.research.shutdown()
        event.accept()


_STYLESHEET = """
QWidget {
    background: #0a0d12;
    color: #cbd7e3;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow, QTabWidget::pane { background: #07090d; }
QTabWidget::pane { border: 1px solid #26313d; }
QTabBar::tab {
    background: #111821;
    color: #8092a3;
    padding: 8px 20px;
    border: 1px solid #26313d;
}
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #ff3579; }
QPushButton {
    background: #151e29;
    border: 1px solid #334252;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover { border-color: #6bdfff; color: #ffffff; }
QPushButton:checked { background: #3a1740; border-color: #ff4fa3; }
QPushButton:disabled { color: #586370; background: #10151b; }
QGroupBox {
    border: 1px solid #26313d;
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 8px;
    color: #8da2b5;
}
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {
    background: #0e141c;
    border: 1px solid #2d3a48;
    border-radius: 3px;
    padding: 4px;
    selection-background-color: #703159;
}
QLabel#tickLabel { font-size: 15pt; font-weight: 700; color: #ffffff; }
QLabel#statusLabel { color: #71dff2; padding: 3px; }
QLabel#pageTitle, QLabel#sectionTitle {
    color: #ffffff;
    font-size: 12pt;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#inspectorValue { color: #ffffff; font-family: "Consolas"; }
"""


def run_ui(argv: list[str] | None = None) -> int:
    """Create the Qt application only for the explicit UI entry point."""

    existing = QApplication.instance()
    owns_application = existing is None
    application = QApplication(sys.argv if argv is None else argv) if owns_application else existing
    assert isinstance(application, QApplication)
    application.setApplicationName("Project Noralet")
    application.setStyle("Fusion")
    application.setStyleSheet(_STYLESHEET)
    window = NoraletMainWindow()
    window.show()
    if not owns_application:
        return 0
    return application.exec()
