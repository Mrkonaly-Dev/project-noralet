"""Pure-observer Qt painting for the finite one-dimensional world."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from noralet.noralets.body import NoraletBodyState
from noralet.noralets.signals import SignalDirection
from noralet.simulation.events import NoraletDied, SignalEmitted
from noralet.ui.session import LiveSession
from noralet.world.signals import ActiveSignal


@dataclass(frozen=True, slots=True)
class SignalGlyph:
    signal_type: str
    origin: float
    direction: int

    @classmethod
    def from_active(cls, signal: ActiveSignal) -> SignalGlyph:
        return cls(
            signal_type=signal.signal_type.value,
            origin=signal.origin,
            direction=(
                -1 if signal.emission_direction is SignalDirection.LEFT else 1
            ),
        )


@dataclass(frozen=True, slots=True)
class _LingeringSignal:
    sender_noralet_id: int
    glyph: SignalGlyph
    created_at: float


@dataclass(frozen=True, slots=True)
class _DeathFlash:
    position: float
    cause: str
    created_at: float


def noralet_display_color(body: NoraletBodyState) -> QColor:
    """Return a deterministic observer color without any random source."""

    signature_value = math.fsum(
        (index + 1) * value
        for index, value in enumerate(body.perceptual_signature)
    )
    signature_code = int(round(abs(signature_value) * 10_000.0))
    hue = (body.noralet_id * 137 + signature_code * 29) % 360
    return QColor.fromHsv(hue, 185, 255)


class WorldCanvas(QWidget):
    """Render immutable published world state and UI-only transient copies."""

    selection_changed = Signal(object)

    _HORIZONTAL_MARGIN = 42.0
    _SIGNAL_LINGER_SECONDS = 0.75
    _DEATH_LINGER_SECONDS = 0.9

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: LiveSession | None = None
        self._selected_id: int | None = None
        self._lingering_signals: list[_LingeringSignal] = []
        self._death_flashes: list[_DeathFlash] = []
        self.setMinimumSize(640, 390)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    @property
    def session(self) -> LiveSession | None:
        return self._session

    @property
    def selected_id(self) -> int | None:
        return self._selected_id

    def set_session(self, session: LiveSession | None) -> None:
        self._session = session
        self._selected_id = None
        self._lingering_signals.clear()
        self._death_flashes.clear()
        self.selection_changed.emit(None)
        self.update()

    def select_noralet(self, noralet_id: int | None) -> None:
        if noralet_id is not None:
            if type(noralet_id) is not int:
                raise TypeError("noralet_id must be an integer or None")
            if self._session is None:
                noralet_id = None
            else:
                living = {
                    body.noralet_id
                    for body in self._session.runner.simulation.state.bodies
                }
                if noralet_id not in living:
                    noralet_id = None
        if self._selected_id != noralet_id:
            self._selected_id = noralet_id
            self.selection_changed.emit(noralet_id)
        self.update()

    def synchronize_selection(self) -> None:
        if self._session is None or self._selected_id is None:
            return
        living = {
            body.noralet_id
            for body in self._session.runner.simulation.state.bodies
        }
        if self._selected_id not in living:
            self.select_noralet(None)

    def observe_latest_result(self) -> None:
        """Copy only transient event facts needed for optional visual linger."""

        if self._session is None or self._session.latest_result is None:
            return
        now = time.monotonic()
        for event in self._session.latest_result.tick_result.events:
            if isinstance(event, SignalEmitted):
                self._lingering_signals = [
                    item
                    for item in self._lingering_signals
                    if item.sender_noralet_id != event.noralet_id
                ]
                self._lingering_signals.append(
                    _LingeringSignal(
                        sender_noralet_id=event.noralet_id,
                        glyph=SignalGlyph(
                            signal_type=event.signal_type.value,
                            origin=event.origin,
                            direction=(
                                -1
                                if event.emission_direction
                                is SignalDirection.LEFT
                                else 1
                            ),
                        ),
                        created_at=now,
                    )
                )
            elif isinstance(event, NoraletDied):
                self._death_flashes.append(
                    _DeathFlash(
                        position=event.resolved_position,
                        cause=event.cause.value,
                        created_at=now,
                    )
                )
        self.synchronize_selection()

    def observer_signal_glyphs(self) -> tuple[SignalGlyph, ...]:
        if self._session is None:
            return ()
        return tuple(
            SignalGlyph.from_active(signal)
            for signal in self._session.runner.simulation.state.active_signals
        )

    def world_to_canvas(self, position: float) -> float:
        if self._session is None:
            raise RuntimeError("a live session is required for coordinate mapping")
        config = self._session.runner.simulation.config
        usable = max(1.0, self.width() - 2.0 * self._HORIZONTAL_MARGIN)
        fraction = (
            (float(position) - config.left_boundary)
            / (config.right_boundary - config.left_boundary)
        )
        return self._HORIZONTAL_MARGIN + fraction * usable

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#05070b"))
        if self._session is None:
            painter.setPen(QColor("#607080"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Reset or Start to create the Baseline world",
            )
            return

        self._prune_transients()
        baseline_y = self.height() * 0.56
        self._draw_regions(painter, baseline_y)
        self._draw_world_line(painter, baseline_y)
        self._draw_energy(painter, baseline_y)
        self._draw_signals(painter, baseline_y)
        self._draw_bodies(painter, baseline_y)
        self._draw_deaths(painter, baseline_y)

    def _draw_regions(self, painter: QPainter, baseline_y: float) -> None:
        ecology = self._session.runner.simulation.config.energy_ecology
        if ecology is None:
            return
        colors = {
            "infertile": QColor(45, 36, 50, 48),
            "sparse": QColor(26, 53, 66, 58),
            "fertile": QColor(67, 28, 75, 68),
        }
        top = baseline_y - 78.0
        height = 156.0
        painter.setPen(Qt.PenStyle.NoPen)
        for region in ecology.regions:
            left = self.world_to_canvas(region.left)
            right = self.world_to_canvas(region.right)
            painter.setBrush(colors[region.kind.value])
            painter.drawRect(QRectF(left, top, right - left, height))
            painter.setPen(QColor(126, 139, 151, 120))
            painter.drawText(
                QRectF(left + 6.0, top + 6.0, max(0.0, right - left - 12.0), 18.0),
                Qt.AlignmentFlag.AlignLeft,
                f"{region.region_id} · {region.kind.value}",
            )
            painter.setPen(Qt.PenStyle.NoPen)

    def _draw_world_line(self, painter: QPainter, y: float) -> None:
        config = self._session.runner.simulation.config
        left = self.world_to_canvas(config.left_boundary)
        right = self.world_to_canvas(config.right_boundary)
        painter.setPen(QPen(QColor("#687583"), 1.2))
        painter.drawLine(QPointF(left, y), QPointF(right, y))
        painter.setPen(QPen(QColor("#ff315a"), 3.0))
        painter.drawLine(QPointF(left, y - 30.0), QPointF(left, y + 30.0))
        painter.drawLine(QPointF(right, y - 30.0), QPointF(right, y + 30.0))
        painter.setPen(QColor("#ff7890"))
        painter.drawText(QPointF(left + 7.0, y + 48.0), "LETHAL")
        painter.drawText(QPointF(right - 52.0, y + 48.0), "LETHAL")

    def _draw_energy(self, painter: QPainter, y: float) -> None:
        points = self._session.runner.simulation.state.energy_points
        for point in points:
            x = self.world_to_canvas(point.position)
            radius = min(8.0, 3.0 + math.sqrt(point.energy) * 0.45)
            for scale, alpha in ((2.8, 22), (1.8, 45)):
                painter.setBrush(QColor(43, 239, 215, alpha))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(x, y), radius * scale, radius * scale)
            painter.setBrush(QColor("#42f5d4"))
            painter.drawEllipse(QPointF(x, y), radius, radius)

    def _draw_bodies(self, painter: QPainter, y: float) -> None:
        state = self._session.runner.simulation.state
        for body in state.bodies:
            x = self.world_to_canvas(body.position)
            color = noralet_display_color(body)
            for radius, alpha in ((23.0, 18), (15.0, 36), (10.0, 62)):
                glow = QColor(color)
                glow.setAlpha(alpha)
                painter.setBrush(glow)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(x, y), radius, radius)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#dff9ff"), 1.0))
            painter.drawEllipse(QPointF(x, y), 6.5, 6.5)
            if body.noralet_id == self._selected_id:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
                painter.drawEllipse(QPointF(x, y), 13.0, 13.0)
            painter.setPen(QColor("#d7e4ef"))
            painter.drawText(QPointF(x - 9.0, y - 17.0), f"N{body.noralet_id}")
            self._draw_vector(
                painter,
                x,
                y + 24.0,
                body.velocity,
                QColor("#68d7ff"),
                visual_scale=13.0,
            )
            acceleration = self._session.latest_applied_acceleration.get(
                body.noralet_id,
                0.0,
            )
            self._draw_vector(
                painter,
                x,
                y + 36.0,
                acceleration,
                QColor("#ff4fa3"),
                visual_scale=70.0,
            )

    def _draw_vector(
        self,
        painter: QPainter,
        x: float,
        y: float,
        value: float,
        color: QColor,
        *,
        visual_scale: float,
    ) -> None:
        if abs(value) < 1e-12:
            return
        length = max(-38.0, min(38.0, value * visual_scale))
        if abs(length) < 4.0:
            length = math.copysign(4.0, length)
        end = x + length
        painter.setPen(QPen(color, 1.6))
        painter.drawLine(QPointF(x, y), QPointF(end, y))
        direction = 1.0 if length > 0.0 else -1.0
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(
            QPolygonF(
                (
                    QPointF(end, y),
                    QPointF(end - direction * 5.0, y - 3.5),
                    QPointF(end - direction * 5.0, y + 3.5),
                )
            )
        )

    def _draw_signals(self, painter: QPainter, y: float) -> None:
        now = time.monotonic()
        glyphs: list[tuple[SignalGlyph, float]] = [
            (glyph, 1.0) for glyph in self.observer_signal_glyphs()
        ]
        active_senders = {
            signal.sender_noralet_id
            for signal in self._session.runner.simulation.state.active_signals
        }
        glyphs.extend(
            (
                item.glyph,
                max(
                    0.0,
                    1.0 - (now - item.created_at) / self._SIGNAL_LINGER_SECONDS,
                ),
            )
            for item in self._lingering_signals
            if item.sender_noralet_id not in active_senders
        )
        signal_colors = {
            "A": QColor("#a34cff"),
            "B": QColor("#ff4fa3"),
            "C": QColor("#f7c948"),
            "D": QColor("#5ce1e6"),
        }
        for glyph, opacity in glyphs:
            x = self.world_to_canvas(glyph.origin)
            direction = float(glyph.direction)
            end = x + direction * 44.0
            color = QColor(signal_colors[glyph.signal_type])
            color.setAlpha(max(20, int(220 * opacity)))
            painter.setPen(QPen(color, 2.0))
            painter.drawLine(QPointF(x, y - 29.0), QPointF(end, y - 29.0))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(
                QPolygonF(
                    (
                        QPointF(end, y - 29.0),
                        QPointF(end - direction * 7.0, y - 34.0),
                        QPointF(end - direction * 7.0, y - 24.0),
                    )
                )
            )
            painter.setPen(color)
            painter.drawText(QPointF(x - 4.0, y - 39.0), glyph.signal_type)

    def _draw_deaths(self, painter: QPainter, y: float) -> None:
        now = time.monotonic()
        for item in self._death_flashes:
            fraction = max(
                0.0,
                1.0 - (now - item.created_at) / self._DEATH_LINGER_SECONDS,
            )
            x = self.world_to_canvas(item.position)
            radius = 8.0 + (1.0 - fraction) * 22.0
            color = QColor(255, 49, 90, int(180 * fraction))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 2.0))
            painter.drawEllipse(QPointF(x, y), radius, radius)
            painter.drawLine(
                QPointF(x - radius * 0.6, y - radius * 0.6),
                QPointF(x + radius * 0.6, y + radius * 0.6),
            )
            painter.drawLine(
                QPointF(x - radius * 0.6, y + radius * 0.6),
                QPointF(x + radius * 0.6, y - radius * 0.6),
            )

    def _prune_transients(self) -> None:
        now = time.monotonic()
        self._lingering_signals = [
            item
            for item in self._lingering_signals
            if now - item.created_at <= self._SIGNAL_LINGER_SECONDS
        ]
        self._death_flashes = [
            item
            for item in self._death_flashes
            if now - item.created_at <= self._DEATH_LINGER_SECONDS
        ]

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._session is None:
            return
        click_x = event.position().x()
        nearest: tuple[float, int] | None = None
        for body in self._session.runner.simulation.state.bodies:
            distance = abs(click_x - self.world_to_canvas(body.position))
            candidate = (distance, body.noralet_id)
            if nearest is None or candidate < nearest:
                nearest = candidate
        self.select_noralet(
            nearest[1] if nearest is not None and nearest[0] <= 14.0 else None
        )
