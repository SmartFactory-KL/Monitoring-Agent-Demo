from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .inference_models import SimulationAnalysisResult

if TYPE_CHECKING:
    from monitoring_agent.descriptive_models import FactoryContext, IslandContext, MachineContext
    from monitoring_agent.kpis import FactoryKPI, IslandKPI, MachineKPI
    from model import HolonicSchedulingModel


@dataclass
class MonitoringDecision:
    action: str = "observe"
    severity: str = "normal"
    request_support: bool = False
    recommend_self_resequence: bool = False
    recommend_island_resequence: bool = False
    recommend_factory_replan: bool = False
    override_plan_steps: int = 0
    reasons: list[str] = field(default_factory=list)


class MachineDecisionModel:
    def evaluate(
        self,
        model: "HolonicSchedulingModel",
        context: "MachineContext",
        kpi: "MachineKPI",
        idm: float,
        analysis: SimulationAnalysisResult,
        idm_drop_too_fast: bool,
    ) -> MonitoringDecision:
        reasons: list[str] = []
        resequence = False

        if idm_drop_too_fast:
            reasons.append("idm_drop")
            resequence = True
        if idm < model.robot_resequence_threshold:
            reasons.append("resequence_threshold")
            resequence = True
        if analysis.projection.late_ratio >= model.robot_forecast_late_ratio:
            reasons.append("late_ratio_forecast")
            resequence = True
        if analysis.projection.avg_late >= model.robot_forecast_avg_late:
            reasons.append("avg_late_forecast")
            resequence = True

        request_support = idm < model.robot_support_threshold
        if request_support:
            reasons.append("support_threshold")

        if idm < model.robot_critical_threshold:
            severity = "critical"
        elif resequence or request_support:
            severity = "warning"
        else:
            severity = "normal"

        if request_support and not resequence:
            action = "request_support"
        elif resequence:
            action = "self_resequence"
        else:
            action = "observe"

        return MonitoringDecision(
            action=action,
            severity=severity,
            request_support=request_support,
            recommend_self_resequence=resequence,
            override_plan_steps=2 if resequence else 0,
            reasons=reasons,
        )


class IslandDecisionModel:
    def evaluate(
        self,
        model: "HolonicSchedulingModel",
        context: "IslandContext",
        kpi: "IslandKPI",
        idm: float,
        analysis: SimulationAnalysisResult,
    ) -> MonitoringDecision:
        reasons: list[str] = []
        recommend_resequence = False
        recommend_replan = False

        if idm < model.station_support_threshold:
            reasons.append("support_threshold")
            recommend_resequence = True
        if analysis.projection.backlog_ratio > model.station_forecast_backlog_ratio:
            reasons.append("backlog_forecast")
            recommend_resequence = True
        if analysis.projection.avg_late > model.station_forecast_tardiness:
            reasons.append("tardiness_forecast")
            recommend_resequence = True
        if idm < model.station_replan_threshold:
            reasons.append("replan_threshold")
            recommend_replan = True

        if recommend_replan:
            action = "recommend_factory_replan"
            severity = "critical"
        elif recommend_resequence:
            action = "recommend_island_resequence"
            severity = "warning"
        else:
            action = "observe"
            severity = "normal"

        return MonitoringDecision(
            action=action,
            severity=severity,
            recommend_island_resequence=recommend_resequence,
            recommend_factory_replan=recommend_replan,
            reasons=reasons,
        )


class FactoryDecisionModel:
    def evaluate(
        self,
        model: "HolonicSchedulingModel",
        context: "FactoryContext",
        kpi: "FactoryKPI",
        idm: float,
        analysis: SimulationAnalysisResult,
    ) -> MonitoringDecision:
        reasons: list[str] = []
        recommend_replan = False

        if analysis.projection.backlog_ratio > 1.10:
            reasons.append("factory_backlog_forecast")
            recommend_replan = True
        if kpi.tardiness_mean > 0:
            reasons.append("factory_tardiness")
            recommend_replan = True
        if idm < 0.85:
            reasons.append("factory_idm")
            recommend_replan = True

        return MonitoringDecision(
            action="recommend_factory_replan" if recommend_replan else "observe",
            severity="warning" if recommend_replan else "normal",
            recommend_factory_replan=recommend_replan,
            reasons=reasons,
        )
