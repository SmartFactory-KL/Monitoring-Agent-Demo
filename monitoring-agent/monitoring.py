from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from model import HolonicSchedulingModel
    from agents import RobotAgent


@dataclass
class ResourceKPI:
    utilization: float = 0.0
    avg_waiting_time: float = 0.0
    queue_length: int = 0
    delay: float = 0.0
    availability: float = 1.0
    completed: int = 0


@dataclass
class StationKPI:
    throughput: int = 0
    backlog: int = 0
    tardiness_mean: float = 0.0
    tardiness_max: float = 0.0
    workload_imbalance: float = 0.0
    queue_growth: float = 0.0
    availability: float = 1.0


@dataclass
class FactoryKPI:
    throughput: int = 0
    backlog: int = 0
    tardiness_mean: float = 0.0
    availability: float = 1.0


class Acceptance:
    @staticmethod
    def linear_closeness(actual: float, ref: float, tolerance: float) -> float:
        """Linear acceptance in [0,1]. 1 means perfect."""
        if tolerance <= 0:
            return 1.0 if actual == ref else 0.0
        return max(0.0, 1.0 - abs(actual - ref) / tolerance)

    @staticmethod
    def safety_band(actual: float, low: float, high: float, tolerance: float) -> float:
        """1 inside [low, high], linear decay outside by tolerance."""
        if low <= actual <= high:
            return 1.0
        if tolerance <= 0:
            return 0.0
        if actual < low:
            return max(0.0, 1.0 - (low - actual) / tolerance)
        return max(0.0, 1.0 - (actual - high) / tolerance)


class IDMCalculator:
    """Simplified discrete IDM inspired by the papers."""

    def __init__(self, weights: Dict[str, float]):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def value(self, acceptances: Dict[str, float]) -> float:
        penalty = 0.0
        for key, weight in self.weights.items():
            penalty += weight * (1.0 - float(acceptances.get(key, 0.0)))
        return max(0.0, min(1.0, 1.0 - penalty))


class TrajectoryIDM:
    """Cumulative IDM over a fixed mission length (monotonic non-increasing)."""

    def __init__(self, weights: Dict[str, float], mission_length: int):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.mission_length = max(1, mission_length)
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
            A_s = Acceptance.safety_band(act, low, high, band)

            tol_p = perf_pct * max(1.0, abs(ref))
            A_p = Acceptance.linear_closeness(act, ref, tol_p)

            # safety and performance share the KPI weight equally
            step_penalty += 0.5 * weight * (1.0 - A_s) + 0.5 * weight * (1.0 - A_p)

        self.cumulative_sum += step_penalty
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


class RobotMonitor:
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.idm = TrajectoryIDM(
            {
                "utilization": 0.40,
                "delay": 0.30,
                "availability": 0.30,
            },
            mission_length=120,
        )
        self.history: List[Dict[str, float]] = []

    def evaluate(self, model: "HolonicSchedulingModel", robot: "RobotAgent") -> Tuple[ResourceKPI, float]:
        now = model.steps
        utilization = robot.busy_steps / max(1, now)
        waiting_samples = robot.waiting_samples[-20:]
        avg_wait = statistics.fmean(waiting_samples) if waiting_samples else 0.0
        queue_len = len(robot.queue)
        delay = max(0.0, robot.completed_lateness_sum / max(1, robot.completed_count))
        availability = 1.0 - (robot.down_steps / max(1, now))

        kpi = ResourceKPI(
            utilization=utilization,
            avg_waiting_time=avg_wait,
            queue_length=queue_len,
            delay=delay,
            availability=availability,
            completed=robot.completed_count,
        )

        ref = model.reference_for_robot(robot.robot_id, now)
        idm = self.idm.update(
            {
                "utilization": kpi.utilization,
                "delay": kpi.delay,
                "availability": kpi.availability,
            },
            {
                "utilization": ref["utilization"],
                "delay": ref["delay"],
                "availability": ref["availability"],
            },
            safety_pct=0.10,
            perf_pct=0.10,
        )

        self.history.append(
            {
                "step": now,
                "robot_id": robot.robot_id,
                "utilization": kpi.utilization,
                "avg_waiting_time": kpi.avg_waiting_time,
                "queue_length": kpi.queue_length,
                "delay": kpi.delay,
                "availability": kpi.availability,
                "idm": idm,
            }
        )
        return kpi, idm


class StationMonitor:
    def __init__(self):
        self.idm = TrajectoryIDM(
            {
                "throughput": 0.40,
                "tardiness": 0.30,
                "availability": 0.30,
            },
            mission_length=120,
        )
        self.history: List[Dict[str, float]] = []
        self._last_backlog = 0

    def evaluate(self, model: "HolonicSchedulingModel", station_id: int | None = None) -> Tuple[StationKPI, float]:
        now = model.steps
        if station_id is None:
            station_tokens = model.tokens
            station_robots = model.robots
        else:
            station_tokens = [t for t in model.tokens if t.station_id == station_id]
            station_robots = model.robots_for_station(station_id)

        throughput = sum(1 for t in station_tokens if t.state == "done")
        backlog = sum(1 for t in station_tokens if t.state != "done" and t.release_time <= now)
        tardiness_values = [
            max(0, (t.process_end_time or now) - t.due_time)
            for t in station_tokens
            if t.state == "done"
        ]
        tardiness_mean = statistics.fmean(tardiness_values) if tardiness_values else 0.0
        tardiness_max = max(tardiness_values) if tardiness_values else 0.0
        queue_lengths = [len(r.queue) for r in station_robots]
        workload_imbalance = statistics.pstdev(queue_lengths) if len(queue_lengths) > 1 else 0.0
        queue_growth = backlog - self._last_backlog
        self._last_backlog = backlog
        availability = statistics.fmean(r.current_kpi.availability for r in station_robots) if station_robots else 1.0

        kpi = StationKPI(
            throughput=throughput,
            backlog=backlog,
            tardiness_mean=tardiness_mean,
            tardiness_max=tardiness_max,
            workload_imbalance=workload_imbalance,
            queue_growth=queue_growth,
            availability=availability,
        )

        ref = model.reference_for_station(now, station_id=station_id)
        idm = self.idm.update(
            {
                "throughput": kpi.throughput,
                "tardiness": kpi.tardiness_mean,
                "availability": kpi.availability,
            },
            {
                "throughput": ref["throughput"],
                "tardiness": ref["tardiness_mean"],
                "availability": ref["availability"],
            },
            safety_pct=0.10,
            perf_pct=0.10,
        )

        self.history.append(
            {
                "step": now,
                "throughput": kpi.throughput,
                "backlog": kpi.backlog,
                "tardiness_mean": kpi.tardiness_mean,
                "tardiness_max": kpi.tardiness_max,
                "workload_imbalance": kpi.workload_imbalance,
                "queue_growth": kpi.queue_growth,
                "availability": kpi.availability,
                "idm": idm,
            }
        )
        return kpi, idm


class FactoryMonitor:
    def __init__(self):
        self.idm = TrajectoryIDM(
            {
                "throughput": 0.45,
                "tardiness": 0.30,
                "availability": 0.25,
            },
            mission_length=120,
        )
        self.history: List[Dict[str, float]] = []
        self._last_backlog = 0

    def evaluate(self, model: "HolonicSchedulingModel") -> Tuple[FactoryKPI, float]:
        now = model.steps
        throughput = sum(1 for t in model.tokens if t.state == "done")
        backlog = sum(1 for t in model.tokens if t.state != "done" and t.release_time <= now)
        tardiness_values = [
            max(0, (t.process_end_time or now) - t.due_time)
            for t in model.tokens
            if t.state == "done"
        ]
        tardiness_mean = statistics.fmean(tardiness_values) if tardiness_values else 0.0
        availability = statistics.fmean(s.current_kpi.availability for s in model.robots) if model.robots else 1.0

        kpi = FactoryKPI(
            throughput=throughput,
            backlog=backlog,
            tardiness_mean=tardiness_mean,
            availability=availability,
        )

        ref = model.reference_for_factory(now)
        idm = self.idm.update(
            {
                "throughput": kpi.throughput,
                "tardiness": kpi.tardiness_mean,
                "availability": kpi.availability,
            },
            {
                "throughput": ref["throughput"],
                "tardiness": ref["tardiness_mean"],
                "availability": ref["availability"],
            },
            safety_pct=0.10,
            perf_pct=0.10,
        )

        self.history.append(
            {
                "step": now,
                "throughput": kpi.throughput,
                "backlog": kpi.backlog,
                "tardiness_mean": kpi.tardiness_mean,
                "availability": kpi.availability,
                "idm": idm,
            }
        )
        return kpi, idm
