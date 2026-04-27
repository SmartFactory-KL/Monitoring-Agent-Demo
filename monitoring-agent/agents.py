from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, TYPE_CHECKING

from mesa import Agent

from monitoring_agent import MachineKPI, MachineMonitor
from orders import TaskToken
from buffers import Buffer

if TYPE_CHECKING:
    from model import HolonicSchedulingModel


class RobotAgent(Agent):
    def __init__(self, model: "HolonicSchedulingModel", robot_id: int, skills: list[str], speed_factor: float, station_id: int):
        super().__init__(model)
        self.robot_id = robot_id
        self.station_id = station_id
        self.skills = skills
        self.speed_factor = speed_factor
        self.queue: Deque[TaskToken] = deque()
        self.current: Optional[TaskToken] = None
        self.remaining: int = 0
        self.buffer_capacity = 10
        self.input_buffer = Buffer(name=f"robot-{robot_id}-in", size=self.buffer_capacity)
        self.output_buffer = Buffer(name=f"robot-{robot_id}-out", size=self.buffer_capacity)

        self.busy_steps = 0
        self.completed_count = 0
        self.completed_lateness_sum = 0.0
        self.waiting_samples: list[float] = []
        self.machine_monitor = MachineMonitor(
            robot_id,
            tolerance_pct=model.idm_tolerance,
            penalty_scale=model.idm_penalty_scale,
        )
        self.monitor = self.machine_monitor
        self.current_idm = 1.0
        self.current_kpi = MachineKPI()
        self.down_steps = 0
        self.down_remaining = 0
        self.failure_rate = 0.006
        self.repair_time_range = (2, 6)
        self.last_support_step = -999
        self.override_plan_steps = 0
        self.hold_steps = 0
        self.last_trade_step = -999
        self.last_resequence_step = -999

    def accepts(self, task: TaskToken) -> bool:
        return task.required_skill in self.skills

    def enqueue(self, task: TaskToken):
        if len(self.queue) >= self.buffer_capacity:
            return False
        task.assigned_robot_id = self.robot_id
        task.robot_queue_enter_time = self.model.steps
        task.state = "waiting_robot"
        self.queue.append(task)
        return True

    def start_next(self):
        if self.current is None and self.queue:
            now = self.model.steps

            if self.override_plan_steps > 0:
                self.override_plan_steps -= 1
                planned = [t for t in self.queue if t.planned_robot_id == self.robot_id]
                planned_ready = [t for t in planned if t.planned_start is None or t.planned_start <= now]
                if planned_ready:
                    planned_first = min(planned_ready, key=lambda t: (t.planned_start or 0, t.due_time))
                    task = min(planned_ready, key=lambda t: (t.due_time, t.release_time, t.token_id))
                    if task != planned_first or not self.model.is_next_planned(self.robot_id, task.token_id):
                        self.model.log_robot_resequence(self.robot_id)
                    self.hold_steps = 0
                else:
                    task = min(self.queue, key=lambda t: (t.due_time, t.release_time, t.token_id))
                    self.model.log_robot_resequence(self.robot_id)
                    self.hold_steps = 0
                self.queue.remove(task)
            else:
                planned = [t for t in self.queue if t.planned_robot_id == self.robot_id]
                planned_ready = [t for t in planned if t.planned_start is None or t.planned_start <= now]
                if planned_ready:
                    task = min(planned_ready, key=lambda t: (t.planned_start or 0, t.due_time, t.token_id))
                    if not self.model.is_next_planned(self.robot_id, task.token_id):
                        self.model.log_robot_resequence(self.robot_id)
                    self.queue.remove(task)
                    self.hold_steps = 0
                elif planned:
                    # planned task not yet due, hold position
                    self.hold_steps += 1
                    if self.hold_steps >= 3 and self.current_idm < 0.95:
                        self.override_plan_steps = max(self.override_plan_steps, 2)
                    return
                else:
                    # no plan for these tasks -> resequence locally
                    task = min(self.queue, key=lambda t: (t.due_time, t.release_time, t.token_id))
                    self.queue.remove(task)
                    self.model.log_robot_resequence(self.robot_id)
                    self.hold_steps = 0

            self.current = task
            wait = self.model.steps - (task.robot_queue_enter_time or self.model.steps)
            self.waiting_samples.append(wait)
            task.state = "processing"
            task.process_start_time = self.model.steps
            self.model.log_token_event(task)
            stochastic = self.model.random.uniform(0.85, 1.35)
            self.remaining = max(1, math.ceil(task.nominal_duration * stochastic / self.speed_factor))

    def finish_current(self):
        assert self.current is not None
        task = self.current
        task.state = "done"
        task.process_end_time = self.model.steps
        self.completed_count += 1
        self.completed_lateness_sum += max(0, task.process_end_time - task.due_time)
        self.model.completed_tokens += 1
        self.model.token_completion_times.append(task.process_end_time)
        self.model.register_task_completion(task)
        self.output_buffer.add(task)
        station_out = self.model.get_station_output_buffer(self.station_id)
        if station_out is not None:
            station_out.add(task)
        self.model.actual_completions[self.model.steps] = self.model.actual_completions.get(self.model.steps, 0) + 1
        self.current = None
        self.remaining = 0

    def step(self):
        if self.down_remaining > 0:
            self.down_steps += 1
            self.down_remaining -= 1
            self.current_kpi, self.current_idm = self.machine_monitor.evaluate(self.model, self)
            return

        if self.model.random.random() < self.failure_rate:
            self.down_remaining = self.model.random.randint(*self.repair_time_range)
            self.down_steps += 1
            self.current_kpi, self.current_idm = self.machine_monitor.evaluate(self.model, self)
            return

        self.start_next()
        if self.current is not None:
            self.busy_steps += 1
            self.remaining -= 1
            if self.remaining <= 0:
                self.finish_current()
        self.current_kpi, self.current_idm = self.machine_monitor.evaluate(self.model, self)

        decision = self.machine_monitor.last_decision
        resequence_allowed = self.model.steps - self.last_resequence_step >= self.model.robot_resequence_cooldown
        if resequence_allowed and decision.recommend_self_resequence:
            if self.resequence_queue():
                self.override_plan_steps = max(self.override_plan_steps, decision.override_plan_steps)

        if decision.request_support:
            if self.model.steps - self.last_support_step >= self.model.robot_support_cooldown:
                self.model.register_support_request(self.robot_id)

    def resequence_queue(self) -> bool:
        if len(self.queue) < 2:
            return False
        old_order = [t.token_id for t in self.queue]
        # prioritize tasks already at station (transport done) and earlier due
        tasks = sorted(
            self.queue,
            key=lambda t: (
                0 if t.state == "waiting_robot" else 1,
                t.due_time,
                t.release_time,
                t.token_id,
            ),
        )
        new_order = [t.token_id for t in tasks]
        if new_order == old_order:
            return False
        self.queue = deque(tasks)
        self.last_resequence_step = self.model.steps
        self.model.log_robot_resequence(self.robot_id)
        return True

    def _idm_drop_too_fast(self) -> bool:
        hist = self.monitor.history
        window = 6
        if len(hist) < window:
            return False
        drop = hist[-window]["idm"] - hist[-1]["idm"]
        return drop > self.model.robot_resequence_drop

    def _forecast_trouble(self) -> bool:
        # Simple "petri-like" forecast: assume current order of work and
        # simulate completion times over a short horizon.
        if self.current is None and not self.queue:
            return False

        now = self.model.steps
        horizon = self.model.robot_forecast_horizon
        threshold_ratio = self.model.robot_forecast_late_ratio
        threshold_avg = self.model.robot_forecast_avg_late

        projected = []
        time_cursor = now + (self.remaining if self.current is not None else 0)
        # current task first (if any)
        if self.current is not None:
            projected.append((self.current, time_cursor))

        for task in list(self.queue):
            time_cursor += task.nominal_duration
            projected.append((task, time_cursor))
            if time_cursor - now > horizon:
                break

        if not projected:
            return False

        late = [max(0, finish - task.due_time) for task, finish in projected]
        late_ratio = sum(1 for v in late if v > 0) / max(1, len(late))
        avg_late = sum(late) / max(1, len(late))
        return late_ratio >= threshold_ratio or avg_late >= threshold_avg


class AGVAgent(Agent):
    def __init__(self, model: "HolonicSchedulingModel", agv_id: int, travel_time: int = 2):
        super().__init__(model)
        self.agv_id = agv_id
        self.travel_time = travel_time
        self.queue: Deque[TaskToken] = deque()
        self.current: Optional[TaskToken] = None
        self.remaining: int = 0

    def enqueue_transport(self, task: TaskToken):
        task.state = "waiting_agv"
        task.assigned_agv_id = self.agv_id
        self.queue.append(task)

    def choose_robot(self, task: TaskToken) -> Optional[RobotAgent]:
        station_id = task.station_id
        station_robots = self.model.robots_for_station(station_id) if station_id else self.model.robots
        if task.planned_robot_id is not None:
            target = next((r for r in station_robots if r.robot_id == task.planned_robot_id), None)
            if target is not None and target.accepts(task):
                return target

        candidates = [r for r in station_robots if r.accepts(task)]
        if not candidates:
            return None
        return min(candidates, key=lambda r: len(r.queue) + (1 if r.current else 0))

    def step(self):
        if self.current is None and self.queue:
            self.current = self.queue.popleft()
            self.current.agv_pickup_time = self.model.steps
            self.current.state = "in_transport"
            target_robot = self.choose_robot(self.current)
            self.current.transport_target_robot_id = target_robot.robot_id if target_robot else None
            self.model.log_token_event(self.current)
            if self.current.transport_time is None:
                target_id = self.current.transport_target_robot_id or self.current.planned_robot_id
                self.current.transport_time = self.model.compute_transport_time(self.current.from_robot_id, target_id)
            self.current.transport_start_step = self.model.steps
            self.current.transport_end_step = self.model.steps + (self.current.transport_time or self.travel_time)
            self.remaining = self.current.transport_time or self.travel_time

        if self.current is not None:
            self.remaining -= 1
            if self.remaining <= 0:
                delivered = False
                if self.current.transport_target_robot_id is not None:
                    target = next(
                        (r for r in self.model.robots if r.robot_id == self.current.transport_target_robot_id),
                        None,
                    )
                    if target and target.enqueue(self.current):
                        delivered = True
                if not delivered:
                    station_buffer = self.model.get_station_input_buffer(self.current.station_id)
                    if station_buffer and station_buffer.add(self.current):
                        self.current.state = "waiting_station"
                        self.model.log_token_event(self.current)
                        self.current = None
                    else:
                        self.current.state = "waiting_agv"
                        self.queue.appendleft(self.current)
                        self.current = None
                else:
                    self.model.log_token_event(self.current)
                    self.current = None
