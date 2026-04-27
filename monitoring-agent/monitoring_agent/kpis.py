from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MachineKPI:
    utilization: float = 0.0
    avg_waiting_time: float = 0.0
    queue_length: int = 0
    delay: float = 0.0
    availability: float = 1.0
    completed: int = 0


@dataclass
class IslandKPI:
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
