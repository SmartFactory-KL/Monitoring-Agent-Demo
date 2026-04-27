from __future__ import annotations

import statistics

from .decision_models import (
    FactoryDecisionModel,
    IslandDecisionModel,
    MachineDecisionModel,
    MonitoringDecision,
)
from .descriptive_models import ContextModel
from .inference_models import (
    FactoryAnalysis,
    IslandAnalysis,
    MachineAnalysis,
    TrajectoryIDM,
)
from .kpis import FactoryKPI, IslandKPI, MachineKPI


class MachineMonitor:
    """Machine-level Monitoring Agent following the paper's architecture."""

    def __init__(self, robot_id: int, tolerance_pct: float = 0.25, penalty_scale: float = 1.0):
        self.robot_id = robot_id
        self.tolerance_pct = tolerance_pct
        self.context_model = ContextModel()
        self.analysis_model = MachineAnalysis()
        self.decision_model = MachineDecisionModel()
        self.idm = TrajectoryIDM(
            {
                "utilization": 0.40,
                "delay": 0.30,
                "availability": 0.30,
            },
            mission_length=120,
            penalty_scale=penalty_scale,
        )
        self.history: list[dict[str, float | str]] = []
        self.last_decision = MonitoringDecision()
        self.last_analysis = None

    def data_acquisition_and_manipulation(self, model, robot) -> tuple[object, MachineKPI]:
        now = model.steps
        utilization = robot.busy_steps / max(1, now)
        waiting_samples = robot.waiting_samples[-20:]
        avg_wait = statistics.fmean(waiting_samples) if waiting_samples else 0.0
        queue_len = len(robot.queue)
        delay = max(0.0, robot.completed_lateness_sum / max(1, robot.completed_count))
        availability = 1.0 - (robot.down_steps / max(1, now))

        kpi = MachineKPI(
            utilization=utilization,
            avg_waiting_time=avg_wait,
            queue_length=queue_len,
            delay=delay,
            availability=availability,
            completed=robot.completed_count,
        )
        context = self.context_model.machine_context(model, robot)
        return context, kpi

    def simulation_analysis(self, model, robot, context, kpi: MachineKPI) -> tuple[float, object, bool]:
        idm = self.idm.update(
            {
                "utilization": kpi.utilization,
                "delay": kpi.delay,
                "availability": kpi.availability,
            },
            {
                "utilization": context.reference["utilization"],
                "delay": context.reference["delay"],
                "availability": context.reference["availability"],
            },
            safety_pct=self.tolerance_pct,
            perf_pct=self.tolerance_pct,
        )
        analysis = self.analysis_model.analyse(model, robot, context, kpi)
        idm_drop = self.analysis_model.idm_drop_too_fast(
            self.history,
            threshold=model.robot_resequence_drop,
        )
        return idm, analysis, idm_drop

    def evaluation_and_decision(self, model, context, kpi: MachineKPI, idm: float, analysis, idm_drop: bool):
        return self.decision_model.evaluate(model, context, kpi, idm, analysis, idm_drop)

    def evaluate(self, model, robot) -> tuple[MachineKPI, float]:
        context, kpi = self.data_acquisition_and_manipulation(model, robot)
        idm, analysis, idm_drop = self.simulation_analysis(model, robot, context, kpi)
        decision = self.evaluation_and_decision(model, context, kpi, idm, analysis, idm_drop)

        self.last_analysis = analysis
        self.last_decision = decision
        self.history.append(
            {
                "step": model.steps,
                "robot_id": robot.robot_id,
                "utilization": kpi.utilization,
                "avg_waiting_time": kpi.avg_waiting_time,
                "queue_length": kpi.queue_length,
                "delay": kpi.delay,
                "availability": kpi.availability,
                "idm": idm,
                "decision": decision.action,
                "severity": decision.severity,
            }
        )
        return kpi, idm


class IslandMonitor:
    """Island-level Monitoring Agent with aggregation and intervention advice."""

    def __init__(self, tolerance_pct: float = 0.25, penalty_scale: float = 1.0):
        self.tolerance_pct = tolerance_pct
        self.context_model = ContextModel()
        self.analysis_model = IslandAnalysis()
        self.decision_model = IslandDecisionModel()
        self.idm = TrajectoryIDM(
            {
                "throughput": 0.40,
                "tardiness": 0.30,
                "availability": 0.30,
            },
            mission_length=120,
            penalty_scale=penalty_scale,
        )
        self.history: list[dict[str, float | str]] = []
        self.last_decision = MonitoringDecision()
        self.last_analysis = None
        self._last_backlog = 0

    def data_acquisition_and_manipulation(self, model, island_id: int | None = None) -> tuple[object, IslandKPI]:
        now = model.steps
        if island_id is None:
            island_tokens = model.tokens
            island_robots = model.robots
        else:
            island_tokens = [t for t in model.tokens if t.station_id == island_id]
            island_robots = model.robots_for_station(island_id)

        throughput = sum(1 for t in island_tokens if t.state == "done")
        backlog = sum(1 for t in island_tokens if t.state != "done" and t.release_time <= now)
        tardiness_values = [
            max(0, (t.process_end_time or now) - t.due_time)
            for t in island_tokens
            if t.state == "done"
        ]
        tardiness_mean = statistics.fmean(tardiness_values) if tardiness_values else 0.0
        tardiness_max = max(tardiness_values) if tardiness_values else 0.0
        queue_lengths = [len(r.queue) for r in island_robots]
        workload_imbalance = statistics.pstdev(queue_lengths) if len(queue_lengths) > 1 else 0.0
        queue_growth = backlog - self._last_backlog
        self._last_backlog = backlog
        availability = statistics.fmean(r.current_kpi.availability for r in island_robots) if island_robots else 1.0

        kpi = IslandKPI(
            throughput=throughput,
            backlog=backlog,
            tardiness_mean=tardiness_mean,
            tardiness_max=tardiness_max,
            workload_imbalance=workload_imbalance,
            queue_growth=queue_growth,
            availability=availability,
        )
        context = self.context_model.island_context(model, island_id)
        return context, kpi

    def simulation_analysis(self, model, context, kpi: IslandKPI) -> tuple[float, object]:
        idm = self.idm.update(
            {
                "throughput": kpi.throughput,
                "tardiness": kpi.tardiness_mean,
                "availability": kpi.availability,
            },
            {
                "throughput": context.reference["throughput"],
                "tardiness": context.reference["tardiness_mean"],
                "availability": context.reference["availability"],
            },
            safety_pct=self.tolerance_pct,
            perf_pct=self.tolerance_pct,
        )
        analysis = self.analysis_model.analyse(model, context, kpi)
        return idm, analysis

    def evaluation_and_decision(self, model, context, kpi: IslandKPI, idm: float, analysis):
        return self.decision_model.evaluate(model, context, kpi, idm, analysis)

    def evaluate(self, model, island_id: int | None = None) -> tuple[IslandKPI, float]:
        context, kpi = self.data_acquisition_and_manipulation(model, island_id=island_id)
        idm, analysis = self.simulation_analysis(model, context, kpi)
        decision = self.evaluation_and_decision(model, context, kpi, idm, analysis)

        self.last_analysis = analysis
        self.last_decision = decision
        self.history.append(
            {
                "step": model.steps,
                "throughput": kpi.throughput,
                "backlog": kpi.backlog,
                "tardiness_mean": kpi.tardiness_mean,
                "tardiness_max": kpi.tardiness_max,
                "workload_imbalance": kpi.workload_imbalance,
                "queue_growth": kpi.queue_growth,
                "availability": kpi.availability,
                "idm": idm,
                "decision": decision.action,
                "severity": decision.severity,
            }
        )
        return kpi, idm


class FactoryMonitor:
    """Factory-level Monitoring Agent for cross-island aggregation."""

    def __init__(self, tolerance_pct: float = 0.25, penalty_scale: float = 1.0):
        self.tolerance_pct = tolerance_pct
        self.context_model = ContextModel()
        self.analysis_model = FactoryAnalysis()
        self.decision_model = FactoryDecisionModel()
        self.idm = TrajectoryIDM(
            {
                "throughput": 0.45,
                "tardiness": 0.30,
                "availability": 0.25,
            },
            mission_length=120,
            penalty_scale=penalty_scale,
        )
        self.history: list[dict[str, float | str]] = []
        self.last_decision = MonitoringDecision()
        self.last_analysis = None

    def data_acquisition_and_manipulation(self, model) -> tuple[object, FactoryKPI]:
        now = model.steps
        throughput = sum(1 for t in model.tokens if t.state == "done")
        backlog = sum(1 for t in model.tokens if t.state != "done" and t.release_time <= now)
        tardiness_values = [
            max(0, (t.process_end_time or now) - t.due_time)
            for t in model.tokens
            if t.state == "done"
        ]
        tardiness_mean = statistics.fmean(tardiness_values) if tardiness_values else 0.0
        availability = statistics.fmean(r.current_kpi.availability for r in model.robots) if model.robots else 1.0

        kpi = FactoryKPI(
            throughput=throughput,
            backlog=backlog,
            tardiness_mean=tardiness_mean,
            availability=availability,
        )
        context = self.context_model.factory_context(model)
        return context, kpi

    def simulation_analysis(self, model, context, kpi: FactoryKPI) -> tuple[float, object]:
        idm = self.idm.update(
            {
                "throughput": kpi.throughput,
                "tardiness": kpi.tardiness_mean,
                "availability": kpi.availability,
            },
            {
                "throughput": context.reference["throughput"],
                "tardiness": context.reference["tardiness_mean"],
                "availability": context.reference["availability"],
            },
            safety_pct=self.tolerance_pct,
            perf_pct=self.tolerance_pct,
        )
        analysis = self.analysis_model.analyse(model, context, kpi)
        return idm, analysis

    def evaluation_and_decision(self, model, context, kpi: FactoryKPI, idm: float, analysis):
        return self.decision_model.evaluate(model, context, kpi, idm, analysis)

    def evaluate(self, model) -> tuple[FactoryKPI, float]:
        context, kpi = self.data_acquisition_and_manipulation(model)
        idm, analysis = self.simulation_analysis(model, context, kpi)
        decision = self.evaluation_and_decision(model, context, kpi, idm, analysis)

        self.last_analysis = analysis
        self.last_decision = decision
        self.history.append(
            {
                "step": model.steps,
                "throughput": kpi.throughput,
                "backlog": kpi.backlog,
                "tardiness_mean": kpi.tardiness_mean,
                "availability": kpi.availability,
                "idm": idm,
                "decision": decision.action,
                "severity": decision.severity,
            }
        )
        return kpi, idm
