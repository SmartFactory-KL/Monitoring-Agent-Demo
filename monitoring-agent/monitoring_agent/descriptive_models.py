from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import RobotAgent
    from model import HolonicSchedulingModel


@dataclass(slots=True)
class MachineContext:
    step: int
    machine_id: int
    island_id: int | None
    skills: tuple[str, ...]
    queue_token_ids: tuple[str, ...]
    planned_token_ids: tuple[str, ...]
    current_token_id: str | None
    buffer_capacity: int
    current_idm: float
    reference: dict[str, float]


@dataclass(slots=True)
class IslandContext:
    step: int
    island_id: int | None
    robot_ids: tuple[int, ...]
    released_token_ids: tuple[str, ...]
    active_order_ids: tuple[str, ...]
    buffer_load: int
    reference: dict[str, float]


@dataclass(slots=True)
class FactoryContext:
    step: int
    island_ids: tuple[int, ...]
    active_order_ids: tuple[str, ...]
    released_token_ids: tuple[str, ...]
    completed_tokens: int
    reference: dict[str, float]


class ContextModel:
    """Context builders that mirror the paper's descriptive model layer."""

    def machine_context(self, model: "HolonicSchedulingModel", robot: "RobotAgent") -> MachineContext:
        planned_token_ids = tuple(
            t.token_id
            for t in model.tokens
            if t.planned_robot_id == robot.robot_id and t.state != "done"
        )
        return MachineContext(
            step=model.steps,
            machine_id=robot.robot_id,
            island_id=robot.station_id,
            skills=tuple(robot.skills),
            queue_token_ids=tuple(t.token_id for t in robot.queue),
            planned_token_ids=planned_token_ids,
            current_token_id=robot.current.token_id if robot.current is not None else None,
            buffer_capacity=robot.buffer_capacity,
            current_idm=robot.current_idm,
            reference=model.reference_for_robot(robot.robot_id, model.steps),
        )

    def island_context(self, model: "HolonicSchedulingModel", island_id: int | None) -> IslandContext:
        if island_id is None:
            island_tokens = model.tokens
            island_robots = model.robots
        else:
            island_tokens = [t for t in model.tokens if t.station_id == island_id]
            island_robots = model.robots_for_station(island_id)

        active_orders = tuple(
            sorted(
                {
                    t.root_order_id
                    for t in island_tokens
                    if t.release_time <= model.steps and t.state != "done"
                }
            )
        )
        released_tokens = tuple(
            t.token_id
            for t in island_tokens
            if t.release_time <= model.steps and t.state != "done"
        )
        buffer_load = sum(len(r.queue) for r in island_robots)
        return IslandContext(
            step=model.steps,
            island_id=island_id,
            robot_ids=tuple(r.robot_id for r in island_robots),
            released_token_ids=released_tokens,
            active_order_ids=active_orders,
            buffer_load=buffer_load,
            reference=model.reference_for_station(model.steps, station_id=island_id),
        )

    def factory_context(self, model: "HolonicSchedulingModel") -> FactoryContext:
        active_orders = tuple(
            sorted(
                {
                    t.root_order_id
                    for t in model.tokens
                    if t.release_time <= model.steps and t.state != "done"
                }
            )
        )
        released_tokens = tuple(
            t.token_id for t in model.tokens if t.release_time <= model.steps and t.state != "done"
        )
        return FactoryContext(
            step=model.steps,
            island_ids=tuple(model.station_ids),
            active_order_ids=active_orders,
            released_token_ids=released_tokens,
            completed_tokens=model.completed_tokens,
            reference=model.reference_for_factory(model.steps),
        )
