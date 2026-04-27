from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agents import RobotAgent
    from monitoring_agent.descriptive_models import FactoryContext, IslandContext, MachineContext
    from monitoring_agent.kpis import FactoryKPI, IslandKPI, MachineKPI
    from model import HolonicSchedulingModel


class Acceptance:
    @staticmethod
    def linear_closeness(actual: float, ref: float, tolerance: float) -> float:
        if tolerance <= 0:
            return 1.0 if actual == ref else 0.0
        return max(0.0, 1.0 - abs(actual - ref) / tolerance)

    @staticmethod
    def safety_band(actual: float, low: float, high: float, tolerance: float) -> float:
        if low <= actual <= high:
            return 1.0
        if tolerance <= 0:
            return 0.0
        if actual < low:
            return max(0.0, 1.0 - (low - actual) / tolerance)
        return max(0.0, 1.0 - (actual - high) / tolerance)


class IDMCalculator:
    def __init__(self, weights: Dict[str, float]):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def value(self, acceptances: Dict[str, float]) -> float:
        penalty = 0.0
        for key, weight in self.weights.items():
            penalty += weight * (1.0 - float(acceptances.get(key, 0.0)))
        return max(0.0, min(1.0, 1.0 - penalty))


class TrajectoryIDM:
    """Cumulative IDM over a fixed mission length."""

    def __init__(self, weights: Dict[str, float], mission_length: int, penalty_scale: float = 1.0):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.mission_length = max(1, mission_length)
        self.penalty_scale = penalty_scale
        self.cumulative_sum = 0.0
        self.last_idm = 1.0

    def reset(self, mission_length: int | None = None):
        if mission_length is not None:
            self.mission_length = max(1, mission_length)
        self.cumulative_sum = 0.0
        self.last_idm = 1.0

    def update(self, actuals: Dict[str, float], refs: Dict[str, float], safety_pct: float, perf_pct: float) -> float:
        step_penalty = 0.0
        for key, weight in self.weights.items():
            ref = float(refs.get(key, 0.0))
            act = float(actuals.get(key, 0.0))

            band = safety_pct * max(1.0, abs(ref))
            low = ref * (1.0 - safety_pct)
            high = ref * (1.0 + safety_pct)
            acceptance_s = Acceptance.safety_band(act, low, high, band)

            tolerance = perf_pct * max(1.0, abs(ref))
            acceptance_p = Acceptance.linear_closeness(act, ref, tolerance)

            step_penalty += 0.5 * weight * (1.0 - acceptance_s) + 0.5 * weight * (1.0 - acceptance_p)

        self.cumulative_sum += step_penalty * self.penalty_scale
        idm = 1.0 - self.cumulative_sum / self.mission_length
        idm = max(0.0, min(1.0, idm))
        self.last_idm = min(self.last_idm, idm)
        return self.last_idm

    def add_penalty(self, amount: float) -> float:
        if amount <= 0:
            return self.last_idm
        self.cumulative_sum += amount
        idm = 1.0 - self.cumulative_sum / self.mission_length
        idm = max(0.0, min(1.0, idm))
        self.last_idm = min(self.last_idm, idm)
        return self.last_idm


@dataclass(slots=True)
class ProjectionSummary:
    horizon: int
    late_ratio: float = 0.0
    avg_late: float = 0.0
    expected_backlog: float = 0.0
    backlog_ratio: float = 1.0


@dataclass(slots=True)
class SimulationAnalysisResult:
    reference: dict[str, float]
    behavior_signals: dict[str, float] = field(default_factory=dict)
    situation_labels: tuple[str, ...] = ()
    projection: ProjectionSummary = field(default_factory=lambda: ProjectionSummary(horizon=0))


class MachineAnalysis:
    def idm_drop_too_fast(self, history: list[dict[str, float]], threshold: float, window: int = 6) -> bool:
        if len(history) < window:
            return False
        drop = history[-window]["idm"] - history[-1]["idm"]
        return drop > threshold

    def prognostic_assessment(self, model: "HolonicSchedulingModel", robot: "RobotAgent") -> ProjectionSummary:
        if robot.current is None and not robot.queue:
            return ProjectionSummary(horizon=model.robot_forecast_horizon)

        now = model.steps
        horizon = model.robot_forecast_horizon
        projected: list[tuple[object, int]] = []
        time_cursor = now + (robot.remaining if robot.current is not None else 0)
        if robot.current is not None:
            projected.append((robot.current, time_cursor))

        for task in list(robot.queue):
            time_cursor += task.nominal_duration
            projected.append((task, time_cursor))
            if time_cursor - now > horizon:
                break

        if not projected:
            return ProjectionSummary(horizon=horizon)

        late = [max(0, finish - task.due_time) for task, finish in projected]
        late_ratio = sum(1 for value in late if value > 0) / max(1, len(late))
        avg_late = sum(late) / max(1, len(late))
        return ProjectionSummary(horizon=horizon, late_ratio=late_ratio, avg_late=avg_late)

    def analyse(
        self,
        model: "HolonicSchedulingModel",
        robot: "RobotAgent",
        context: "MachineContext",
        kpi: "MachineKPI",
    ) -> SimulationAnalysisResult:
        reference = context.reference
        signals = {
            "utilization_gap": kpi.utilization - float(reference["utilization"]),
            "delay_gap": kpi.delay - float(reference["delay"]),
            "availability_gap": kpi.availability - float(reference["availability"]),
            "queue_gap": kpi.queue_length - float(reference["queue_length"]),
        }
        labels: list[str] = []
        if signals["delay_gap"] > 0:
            labels.append("delayed_execution")
        if signals["queue_gap"] > 0:
            labels.append("queue_pressure")
        if signals["availability_gap"] < 0:
            labels.append("availability_loss")
        projection = self.prognostic_assessment(model, robot)
        return SimulationAnalysisResult(
            reference=reference,
            behavior_signals=signals,
            situation_labels=tuple(labels),
            projection=projection,
        )


class IslandAnalysis:
    def prognostic_assessment(
        self,
        model: "HolonicSchedulingModel",
        context: "IslandContext",
        kpi: "IslandKPI",
    ) -> ProjectionSummary:
        horizon = model.station_forecast_horizon
        expected_backlog = max(0.0, kpi.backlog + kpi.queue_growth * max(1, horizon // 3))
        nominal_backlog = max(1.0, float(context.reference["backlog"]) + 1.0)
        return ProjectionSummary(
            horizon=horizon,
            avg_late=kpi.tardiness_mean,
            expected_backlog=expected_backlog,
            backlog_ratio=expected_backlog / nominal_backlog,
        )

    def analyse(
        self,
        model: "HolonicSchedulingModel",
        context: "IslandContext",
        kpi: "IslandKPI",
    ) -> SimulationAnalysisResult:
        reference = context.reference
        signals = {
            "throughput_gap": kpi.throughput - float(reference["throughput"]),
            "backlog_gap": kpi.backlog - float(reference["backlog"]),
            "tardiness_gap": kpi.tardiness_mean - float(reference["tardiness_mean"]),
            "availability_gap": kpi.availability - float(reference["availability"]),
        }
        labels: list[str] = []
        if signals["throughput_gap"] < 0:
            labels.append("throughput_loss")
        if signals["backlog_gap"] > 0:
            labels.append("backlog_growth")
        if signals["tardiness_gap"] > 0:
            labels.append("schedule_risk")
        projection = self.prognostic_assessment(model, context, kpi)
        return SimulationAnalysisResult(
            reference=reference,
            behavior_signals=signals,
            situation_labels=tuple(labels),
            projection=projection,
        )


class FactoryAnalysis:
    def prognostic_assessment(
        self,
        model: "HolonicSchedulingModel",
        context: "FactoryContext",
        kpi: "FactoryKPI",
    ) -> ProjectionSummary:
        horizon = max(model.station_forecast_horizon, 12)
        remaining = len(context.released_token_ids)
        expected_backlog = max(0.0, remaining - max(1, model.completed_tokens))
        nominal_backlog = max(1.0, float(context.reference["backlog"]) + 1.0)
        return ProjectionSummary(
            horizon=horizon,
            avg_late=kpi.tardiness_mean,
            expected_backlog=expected_backlog,
            backlog_ratio=expected_backlog / nominal_backlog,
        )

    def analyse(
        self,
        model: "HolonicSchedulingModel",
        context: "FactoryContext",
        kpi: "FactoryKPI",
    ) -> SimulationAnalysisResult:
        reference = context.reference
        signals = {
            "throughput_gap": kpi.throughput - float(reference["throughput"]),
            "backlog_gap": kpi.backlog - float(reference["backlog"]),
            "tardiness_gap": kpi.tardiness_mean - float(reference["tardiness_mean"]),
            "availability_gap": kpi.availability - float(reference["availability"]),
        }
        labels: list[str] = []
        if signals["backlog_gap"] > 0:
            labels.append("factory_backlog_growth")
        if signals["tardiness_gap"] > 0:
            labels.append("factory_schedule_risk")
        projection = self.prognostic_assessment(model, context, kpi)
        return SimulationAnalysisResult(
            reference=reference,
            behavior_signals=signals,
            situation_labels=tuple(labels),
            projection=projection,
        )
