from __future__ import annotations

import random
import statistics
from typing import Dict, List
from collections import deque

from mesa import Model
from mesa.datacollection import DataCollector
from agents import AGVAgent, RobotAgent
from buffers import Buffer
from monitoring_agent import IslandKPI, IslandMonitor, FactoryKPI, FactoryMonitor
from factory import FactoryController
from orders import OrderNode, TaskToken
import json
from pathlib import Path


class HolonicSchedulingModel(Model):
    def __init__(
        self,
        seed: int = 42,
        n_orders: int = 10,
        max_steps: int = 600,
        idm_tolerance: float = 0.25,
        idm_penalty_scale: float = 1.0,
    ):
        super().__init__(seed=seed)
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.idm_tolerance = idm_tolerance
        self.idm_penalty_scale = idm_penalty_scale
        self.mission_length = max_steps
        self.robot_mission_length = max(12, max_steps // 4)
        self.robot_resequence_threshold = 0.82
        self.robot_resequence_drop = 0.06
        self.robot_resequence_cooldown = 8
        self.robot_resequence_penalty = 0.25
        self.robot_resequence_penalty_cooldown = 4
        self.robot_support_threshold = 0.80
        self.robot_critical_threshold = 0.72
        self.robot_support_cooldown = 6
        self.robot_forecast_horizon = 8
        self.robot_forecast_late_ratio = 0.35
        self.robot_forecast_avg_late = 3.0
        self.robot_trade_cooldown = 8
        self.skip_stage_prob = 0.25
        self.station_support_threshold = 0.92
        self.station_support_drop = 0.02
        self.station_support_critical = 0.78
        self.station_replan_threshold = 0.75
        self.station_replan_drop = 0.00
        self.station_forecast_horizon = 12
        self.station_forecast_backlog_ratio = 1.15
        self.station_forecast_tardiness = 4.0
        self._last_station_idm: dict[int, float] = {}
        self._last_station_forecast_step: int | None = None
        self.mission_epoch = 0
        self.factory_replan_cooldown = 60
        self.factory_recent_reschedule_window = 40
        self._last_factory_replan_step: int | None = None
        self.support_requests: dict[int, list[int]] = {}
        self.robot_resequence_events: list[tuple[int, int]] = []
        self.robot_resequence_penalty_steps: dict[int, int] = {}
        self.robot_mission_reset_events: list[tuple[int, int]] = []
        self.station_reschedule_events: list[tuple[int, int, str]] = []
        self.station_replan_events: list[tuple[int, int]] = []
        self.factory_reschedule_events: list[tuple[int, int, int]] = []
        self.order_progress: dict[str, int] = {}
        self.order_last_robot: dict[str, int] = {}
        self.rolling_backlog_min = 6
        self.rolling_batch_size = 2
        self.max_total_orders = 24
        self.token_events: list[dict] = []
        self.actual_completions: dict[int, int] = {}
        self.planned_completions: dict[int, int] = {}
        self._last_forecast_step: int | None = None
        self.log_path = Path("sim_step_log.jsonl")
        self.log_path.write_text("", encoding="utf-8")

        self.island_monitors: dict[int, IslandMonitor] = {}
        self.island_idm: dict[int, float] = {}
        self.island_kpi: dict[int, IslandKPI] = {}
        self.station_monitors = self.island_monitors
        self.station_idm = self.island_idm
        self.station_kpi = self.island_kpi

        self.factory_monitor = FactoryMonitor(
            tolerance_pct=self.idm_tolerance,
            penalty_scale=self.idm_penalty_scale * 0.45,
        )
        self.factory_idm = 1.0
        self.factory_kpi = FactoryKPI()

        self.next_token_id = 0
        self.next_order_id = n_orders
        self.stage_station_assignment: dict[tuple[str, str], int] = {}
        self.orders: List[OrderNode] = self._generate_orders(n_orders)
        self.tokens: List[TaskToken] = self._flatten_orders_to_tokens(self.orders)
        self.release_index = 0

        self.completed_tokens = 0
        self.token_completion_times: List[int] = []

        self.station_ids = [1]
        self.island_ids = self.station_ids
        self.station_skills = {1: {"SE", "FE"}}
        robot_specs = [
            (1, ["JOIN", "RIVET", "BOLT", "SE", "FE"], 1.00, 1),
            (2, ["SEAL", "CHECK", "BOLT", "SE", "FE"], 0.95, 1),
            (3, ["JOIN", "SEAL", "BOLT", "FE"], 0.98, 1),
        ]
        self.robots: List[RobotAgent] = []
        for rid, skills, speed, station_id in robot_specs:
            robot = RobotAgent(self, rid, skills, speed, station_id)
            robot.monitor.idm.reset(self.robot_mission_length)
            robot.monitor.idm.penalty_scale = self.idm_penalty_scale * 0.7
            self.robots.append(robot)

        self.agv = AGVAgent(self, 100, travel_time=2)
        self.station_input_buffers = {sid: Buffer(name=f"station-{sid}-in", size=20) for sid in self.station_ids}
        self.station_output_buffers = {sid: Buffer(name=f"station-{sid}-out", size=20) for sid in self.station_ids}
        self._plan_initial_schedule()
        self._build_planned_sequences()
        self._build_planned_completions()
        self.mission_length = self._compute_planned_makespan() or self.max_steps
        for sid in self.station_ids:
            monitor = IslandMonitor(
                tolerance_pct=self.idm_tolerance,
                penalty_scale=self.idm_penalty_scale,
            )
            monitor.idm.reset(self.mission_length)
            self.island_monitors[sid] = monitor
            self.island_idm[sid] = 1.0
            self.island_kpi[sid] = IslandKPI()
            self._last_station_idm[sid] = 1.0
            self.support_requests[sid] = []

        self.factory_monitor.idm.reset(self.mission_length)
        self.factory = FactoryController(self)

        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.steps,
                "completed_tokens": lambda m: m.completed_tokens,
                "factory_idm": lambda m: m.factory_idm,
                "avg_island_idm": lambda m: statistics.fmean(m.island_idm.values()) if m.island_idm else 1.0,
                "avg_station_idm": lambda m: statistics.fmean(m.station_idm.values()) if m.station_idm else 1.0,
                "avg_robot_idm": lambda m: statistics.fmean(r.current_idm for r in m.robots),
            }
        )

    # -------------------------
    # Order generation
    # -------------------------
    def _generate_orders(self, n_orders: int) -> List[OrderNode]:
        orders: List[OrderNode] = []
        for i in range(n_orders):
            orders.append(self._build_order(i, release_base=i * 5))
        return orders

    def _build_order(self, order_idx: int, release_base: int | None = None) -> OrderNode:
        release = self.steps if release_base is None else release_base
        due = release + self.random.randint(18, 32)

        family = self.random.choice(["GOLF", "TIGUAN"])
        root = OrderNode(
            node_id=f"O{order_idx}",
            family=family,
            release_time=release,
            due_time=due,
            required_skill="bundle",
            nominal_process_time=0,
        )

        stage_durations = {
            "GOLF": {"SE": 16, "FE": 12},
            "TIGUAN": {"SE": 18, "FE": 12},
        }
        subtask_fractions = [
            ("JOIN", 0.4),
            ("RIVET", 0.2),
            ("BOLT", 0.2),
            ("SEAL", 0.1),
            ("CHECK", 0.1),
        ]

        seq = 0
        total_duration = 0
        mode_rand = self.random.random()

        if mode_rand < self.skip_stage_prob:
            # skip station grouping: direct subtasks under root (execution layer still exists)
            combined = sum(stage_durations[family].values())
            for subtask, frac in subtask_fractions:
                sub_duration = max(1, int(round(combined * frac)))
                leaf = OrderNode(
                    node_id=f"O{order_idx}.{subtask}",
                    family=family,
                    release_time=release,
                    due_time=due,
                    required_skill=subtask,
                    nominal_process_time=sub_duration,
                    parent_id=root.node_id,
                )
                leaf.seq_index = seq
                seq += 1
                total_duration += sub_duration
                root.children.append(leaf)
        else:
            # normal: SE/FE with subtasks
            for stage in ["SE", "FE"]:
                stage_duration = stage_durations[family][stage]
                suborder = OrderNode(
                    node_id=f"O{order_idx}.{stage}",
                    family=family,
                    release_time=release,
                    due_time=due,
                    required_skill=stage,
                    nominal_process_time=stage_duration,
                    parent_id=root.node_id,
                )
                for subtask, frac in subtask_fractions:
                    sub_duration = max(1, int(round(stage_duration * frac)))
                    leaf = OrderNode(
                        node_id=f"O{order_idx}.{stage}.{subtask}",
                        family=family,
                        release_time=release,
                        due_time=due,
                        required_skill=subtask,
                        nominal_process_time=sub_duration,
                        parent_id=suborder.node_id,
                    )
                    leaf.seq_index = seq
                    seq += 1
                    total_duration += sub_duration
                    suborder.children.append(leaf)
                root.children.append(suborder)

        # adjust due date based on total processing time with slack
        due = release + total_duration + self.random.randint(8, 16)
        root.due_time = due
        for node in root.children:
            node.due_time = due
            for leaf in node.flatten_leaves():
                leaf.due_time = due

        return root

    def _flatten_orders_to_tokens(self, orders: List[OrderNode]) -> List[TaskToken]:
        tokens: List[TaskToken] = []
        nominal_clock = 0
        for root in orders:
            for leaf in root.flatten_leaves():
                stage = None
                if leaf.required_skill in ("SE", "FE"):
                    stage = leaf.required_skill
                elif leaf.parent_id:
                    stage = leaf.parent_id.split(".")[-1]
                station_id = self._station_for_stage(root.node_id, stage)
                token = TaskToken(
                    token_id=f"T{self.next_token_id}",
                    root_order_id=root.node_id,
                    node=leaf,
                    required_skill=leaf.required_skill,
                    nominal_duration=leaf.nominal_process_time,
                    release_time=leaf.release_time,
                    due_time=leaf.due_time,
                    nominal_finish_target=nominal_clock + leaf.nominal_process_time + 2,
                    seq_index=leaf.seq_index or 0,
                    seq_total=len(root.children),
                    station_id=station_id,
                    stage=stage,
                )
                nominal_clock += leaf.nominal_process_time
                tokens.append(token)
                self.next_token_id += 1

        tokens.sort(key=lambda t: (t.release_time, t.due_time, t.token_id))
        return tokens

    # -------------------------
    # Planning (initial schedule)
    # -------------------------
    def _plan_initial_schedule(self):
        """Simple FJSS list scheduling with precedence constraints."""
        robot_available = {r.robot_id: 0 for r in self.robots}
        robot_skill = {r.robot_id: r.skills for r in self.robots}
        order_last_finish: dict[str, int] = {}

        # schedule per order, in its internal sequence
        for order in self.orders:
            seq = sorted(order.flatten_leaves(), key=lambda n: n.seq_index or 0)
            for node in seq:
                task = next((t for t in self.tokens if t.node.node_id == node.node_id), None)
                if task is None:
                    continue
                station_robots = self.robots_for_station(task.station_id)
                candidates = [
                    rid
                    for rid, sk in robot_skill.items()
                    if task.required_skill in sk and any(r.robot_id == rid for r in station_robots)
                ]
                if not candidates:
                    continue
                rid = min(candidates, key=lambda r: robot_available[r])
                start = max(task.release_time, robot_available[rid], order_last_finish.get(order.node_id, 0))
                finish = start + task.nominal_duration
                robot_available[rid] = finish
                order_last_finish[order.node_id] = finish
                task.nominal_finish_target = finish
                task.planned_robot_id = rid
                task.planned_start = start
                task.planned_finish = finish

    def _build_planned_sequences(self):
        self.planned_sequence: dict[int, list[str]] = {}
        self.planned_pos: dict[str, int] = {}
        for r in self.robots:
            tasks = [t for t in self.tokens if t.planned_robot_id == r.robot_id]
            tasks.sort(key=lambda t: (t.planned_start or 0, t.planned_finish or 0, t.token_id))
            seq = [t.token_id for t in tasks]
            self.planned_sequence[r.robot_id] = seq
            for idx, tid in enumerate(seq):
                self.planned_pos[tid] = idx

    def _build_planned_completions(self):
        self.planned_completions = {}
        for t in self.tokens:
            if t.nominal_finish_target is None:
                continue
            step = int(t.nominal_finish_target)
            self.planned_completions[step] = self.planned_completions.get(step, 0) + 1

    def _dispatch_station_buffers(self):
        for sid in self.station_ids:
            station_buffer = self.station_input_buffers[sid]
            if len(station_buffer) == 0:
                continue
            remaining = []
            while len(station_buffer) > 0:
                task = station_buffer.pop()
                target = None
                station_robots = self.robots_for_station(sid)
                if task.planned_robot_id is not None:
                    target = next((r for r in station_robots if r.robot_id == task.planned_robot_id), None)
                    if target and not target.accepts(task):
                        target = None
                if target is None:
                    candidates = [r for r in station_robots if r.accepts(task)]
                    if candidates:
                        target = min(candidates, key=lambda r: len(r.queue))
                if target is None or not target.enqueue(task):
                    remaining.append(task)
                else:
                    self.log_token_event(task)
            for task in remaining:
                station_buffer.add(task)

    def _negotiate_station_robots(self):
        for sid in self.station_ids:
            robots = self.robots_for_station(sid)
            if len(robots) < 2:
                continue
            for i in range(len(robots)):
                for j in range(i + 1, len(robots)):
                    ra = robots[i]
                    rb = robots[j]
                    if self.steps - min(ra.last_trade_step, rb.last_trade_step) < self.robot_trade_cooldown:
                        continue
                    if not ra.queue or not rb.queue:
                        continue
                    ta = next((t for t in ra.queue if rb.accepts(t)), None)
                    tb = next((t for t in rb.queue if ra.accepts(t)), None)
                    if ta is None or tb is None:
                        continue

                    score_a = self._forecast_queue_score(ra, list(ra.queue))
                    score_b = self._forecast_queue_score(rb, list(rb.queue))

                    new_a = list(ra.queue)
                    new_b = list(rb.queue)
                    new_a.remove(ta)
                    new_b.remove(tb)
                    new_a.insert(0, tb)
                    new_b.insert(0, ta)
                    score_a_new = self._forecast_queue_score(ra, new_a)
                    score_b_new = self._forecast_queue_score(rb, new_b)

                    if score_a_new < score_a and score_b_new < score_b:
                        ra.queue = deque(new_a)
                        rb.queue = deque(new_b)
                        ra.last_trade_step = self.steps
                        rb.last_trade_step = self.steps
                        break

    def _forecast_queue_score(self, robot: RobotAgent, queue_list: list[TaskToken]) -> float:
        now = self.steps
        horizon = self.robot_forecast_horizon
        time_cursor = now + (robot.remaining if robot.current is not None else 0)
        projected = []
        if robot.current is not None:
            projected.append((robot.current, time_cursor))

        for task in queue_list:
            time_cursor += task.nominal_duration
            projected.append((task, time_cursor))
            if time_cursor - now > horizon:
                break

        if not projected:
            return 0.0

        late = [max(0, finish - task.due_time) for task, finish in projected]
        late_ratio = sum(1 for v in late if v > 0) / max(1, len(late))
        avg_late = sum(late) / max(1, len(late))
        return avg_late + 5.0 * late_ratio

    def is_next_planned(self, robot_id: int, token_id: str) -> bool:
        seq = self.planned_sequence.get(robot_id, [])
        if not seq:
            return True
        # find first not yet completed token for this robot
        for tid in seq:
            tok = next((t for t in self.tokens if t.token_id == tid), None)
            if tok is None:
                continue
            if tok.state != "done":
                return tid == token_id
        return True

    # -------------------------
    # Rolling horizon: inject new orders
    # -------------------------
    def _maybe_add_orders(self):
        if len(self.orders) >= self.max_total_orders:
            return
        backlog = sum(1 for t in self.tokens if t.state != "done" and t.release_time <= self.steps)
        if backlog >= self.rolling_backlog_min:
            return
        for _ in range(self.rolling_batch_size):
            if len(self.orders) >= self.max_total_orders:
                break
            order = self._build_order(self.next_order_id)
            self.next_order_id += 1
            self.orders.append(order)
            new_tokens = self._flatten_orders_to_tokens([order])
            self.tokens.extend(new_tokens)
            self._plan_new_tokens(new_tokens)
        self._build_planned_sequences()
        self._build_planned_completions()

    def _plan_new_tokens(self, new_tokens: list[TaskToken]):
        robot_available = {r.robot_id: 0 for r in self.robots}
        for t in self.tokens:
            if t.planned_finish is not None and t.planned_robot_id is not None:
                robot_available[t.planned_robot_id] = max(robot_available[t.planned_robot_id], t.planned_finish)

        robot_skill = {r.robot_id: r.skills for r in self.robots}
        order_last_finish: dict[str, int] = {t.root_order_id: 0 for t in new_tokens}

        new_orders = {t.root_order_id for t in new_tokens}
        for order in [o for o in self.orders if o.node_id in new_orders]:
            seq = sorted(order.flatten_leaves(), key=lambda n: n.seq_index or 0)
            for node in seq:
                task = next((t for t in new_tokens if t.node.node_id == node.node_id), None)
                if task is None:
                    continue
                station_robots = self.robots_for_station(task.station_id)
                candidates = [
                    rid
                    for rid, sk in robot_skill.items()
                    if task.required_skill in sk and any(r.robot_id == rid for r in station_robots)
                ]
                if not candidates:
                    continue
                rid = min(candidates, key=lambda r: robot_available[r])
                start = max(task.release_time, robot_available[rid], order_last_finish.get(order.node_id, 0))
                finish = start + task.nominal_duration
                robot_available[rid] = finish
                order_last_finish[order.node_id] = finish
                task.nominal_finish_target = finish
                task.planned_robot_id = rid
                task.planned_start = start
                task.planned_finish = finish

    def _forecast_and_intervene(self):
        for sid in self.station_ids:
            if self._forecast_idm_breach(station_id=sid, horizon=12, threshold=0.9):
                if self._last_forecast_step is None or self.steps - self._last_forecast_step >= 25:
                    rescheduled = self._apply_reschedule_action("full", station_id=sid)
                    for r in rescheduled:
                        r.monitor.idm.reset(self.robot_mission_length)
                        r.last_support_step = self.steps
                        self.robot_mission_reset_events.append((self.steps, r.robot_id))
                    self.station_reschedule_events.append((self.steps, sid, "full"))
                    self._last_forecast_step = self.steps
            if self._petri_forecast_station_breach(station_id=sid):
                if self._last_station_forecast_step is None or self.steps - self._last_station_forecast_step >= 30:
                    rescheduled = self._apply_reschedule_action("full", station_id=sid)
                    for r in rescheduled:
                        r.monitor.idm.reset(self.robot_mission_length)
                        r.last_support_step = self.steps
                        self.robot_mission_reset_events.append((self.steps, r.robot_id))
                    self.station_reschedule_events.append((self.steps, sid, "full"))
                    self._last_station_forecast_step = self.steps

    def _forecast_idm_breach(self, station_id: int, horizon=10, threshold=0.9):
        hist = self.island_monitors[station_id].history
        if len(hist) < 5:
            return False
        recent = [h["idm"] for h in hist[-5:]]
        slope = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        projected = recent[-1] + slope * horizon
        if projected < threshold:
            return True

        now = self.steps
        planned = sum(self.planned_completions.get(step, 0) for step in range(now + 1, now + horizon + 1))
        actual = sum(self.actual_completions.get(step, 0) for step in range(now - horizon + 1, now + 1))
        if planned > 0 and actual / max(1, horizon) < planned / max(1, horizon) * 0.6:
            return True
        return False

    def _petri_forecast_station_breach(self, station_id: int) -> bool:
        # Simple "petri-like" forecast: backlog evolution from arrivals vs. service capacity.
        horizon = self.station_forecast_horizon
        now = self.steps

        # expected arrivals in horizon
        arrivals = sum(
            1
            for t in self.tokens
            if t.station_id == station_id and now < t.release_time <= now + horizon
        )

        # current backlog
        backlog = sum(
            1
            for t in self.tokens
            if t.station_id == station_id and t.state != "done" and t.release_time <= now
        )

        # service capacity estimate (avg nominal duration per robot)
        active = [
            t
            for t in self.tokens
            if t.station_id == station_id and t.state in ("waiting_robot", "processing")
        ]
        avg_dur = statistics.fmean([t.nominal_duration for t in active]) if active else 3.0
        capacity = int((horizon * len(self.robots_for_station(station_id))) / max(1.0, avg_dur))

        projected_backlog = max(0, backlog + arrivals - capacity)
        if backlog > 0 and projected_backlog / max(1, backlog) >= self.station_forecast_backlog_ratio:
            return True

        # tardiness forecast: if many tokens will be late within horizon
        late = 0
        considered = 0
        time_cursor = now
        for t in sorted(
            [x for x in self.tokens if x.station_id == station_id and x.state != "done"],
            key=lambda z: (z.due_time, z.release_time),
        ):
            time_cursor += t.nominal_duration
            if time_cursor - now > horizon:
                break
            considered += 1
            if time_cursor > t.due_time:
                late += 1
        if considered > 0 and (late / considered) >= 0.4:
            return True
        return False

    # -------------------------
    # Reference trajectories for monitoring
    # -------------------------
    def reference_for_robot(self, robot_id: int, step: int) -> Dict[str, float]:
        skills = next(r.skills for r in self.robots if r.robot_id == robot_id)
        released_skill_tasks = [t for t in self.tokens if t.required_skill in skills and t.release_time <= step]
        n_candidates = max(1, sum(1 for r in self.robots if any(s in r.skills for s in skills)))
        nominal_load = sum(t.nominal_duration for t in released_skill_tasks) / n_candidates
        nominal_util = min(0.90, nominal_load / max(1, step + 1))
        return {
            "utilization": nominal_util,
            "avg_waiting_time": 1.0,
            "queue_length": 1.0,
            "delay": 0.0,
            "availability": 1.0,
        }

    def reference_for_station(self, step: int, station_id: int | None = None) -> Dict[str, float]:
        if station_id is None:
            station_tokens = self.tokens
        else:
            station_tokens = [t for t in self.tokens if t.station_id == station_id]
        nominal_done = sum(1 for t in station_tokens if (t.nominal_finish_target or 0) <= step)
        nominal_backlog = sum(1 for t in station_tokens if t.release_time <= step) - nominal_done
        return {
            "throughput": nominal_done,
            "backlog": max(0, nominal_backlog),
            "tardiness_mean": 0.0,
            "workload_imbalance": 0.5,
            "availability": 1.0,
        }

    def reference_for_factory(self, step: int) -> Dict[str, float]:
        nominal_done = sum(1 for t in self.tokens if (t.nominal_finish_target or 0) <= step)
        nominal_backlog = sum(1 for t in self.tokens if t.release_time <= step) - nominal_done
        return {
            "throughput": nominal_done,
            "backlog": max(0, nominal_backlog),
            "tardiness_mean": 0.0,
            "availability": 1.0,
        }

    def _station_for_stage(self, order_id: str, stage: str | None) -> int | None:
        if stage is None:
            return 1
        key = (order_id, stage)
        if key in self.stage_station_assignment:
            return self.stage_station_assignment[key]
        # single-station setup: all stages map to station 1
        station_id = 1
        self.stage_station_assignment[key] = station_id
        return station_id

    def robots_for_station(self, station_id: int | None) -> list[RobotAgent]:
        if station_id is None:
            return self.robots
        return [r for r in self.robots if r.station_id == station_id]

    def get_station_input_buffer(self, station_id: int | None):
        if station_id is None:
            return None
        return self.station_input_buffers.get(station_id)

    def get_station_output_buffer(self, station_id: int | None):
        if station_id is None:
            return None
        return self.station_output_buffers.get(station_id)

    # -------------------------
    # Runtime control
    # -------------------------
    def release_new_tokens(self):
        now = self.steps
        for token in self.tokens:
            if token.state == "planned" and token.release_time <= now and self._order_allows_release(token):
                if token.from_robot_id is None:
                    token.from_robot_id = self.order_last_robot.get(token.root_order_id)
                if token.station_id is None:
                    token.station_id = self._station_for_stage(token.root_order_id, token.stage)
                token.state = "waiting_agv"
                self.log_token_event(token)
                self.agv.enqueue_transport(token)

    def step(self):
        self._maybe_add_orders()
        self.release_new_tokens()
        self._dispatch_station_buffers()
        self.agents.shuffle_do("step")
        self._dispatch_station_buffers()
        self._negotiate_station_robots()
        for sid in self.station_ids:
            kpi, idm = self.island_monitors[sid].evaluate(self, island_id=sid)
            self.island_kpi[sid] = kpi
            self.island_idm[sid] = idm
        self.factory_kpi, self.factory_idm = self.factory_monitor.evaluate(self)
        self._forecast_and_intervene()
        self._handle_support_requests()
        self.factory.step()
        self.datacollector.collect(self)
        self._log_step()

    def run(self):
        for _ in range(self.max_steps):
            if self.completed_tokens >= len(self.tokens):
                break
            self.step()

    # -------------------------
    # Reporting
    # -------------------------
    def final_report(self) -> str:
        avg_robot_idm = statistics.fmean(r.current_idm for r in self.robots)
        plan_quality = self.plan_quality_report()
        lines = []
        lines.append("=== Holonic Scheduling Demo Report ===")
        lines.append(f"Steps executed: {self.steps}")
        lines.append(f"Completed tokens: {self.completed_tokens}/{len(self.tokens)}")
        lines.append(f"Factory IDM: {self.factory_idm:.3f}")
        lines.append(f"Average robot IDM: {avg_robot_idm:.3f}")
        lines.append(f"Factory backlog: {self.factory_kpi.backlog}, tardiness_mean={self.factory_kpi.tardiness_mean:.2f}")
        lines.append(f"Factory availability: {self.factory_kpi.availability:.3f}")
        lines.append(f"Resequence events: {len(self.robot_resequence_events)}")
        lines.append(f"Island replans: {len(self.station_replan_events)}")

        lines.append("\nIsland details:")
        for sid in self.station_ids:
            kpi = self.island_kpi[sid]
            lines.append(
                "  Island {0} -> IDM={1:.3f}, backlog={2}, tardiness_mean={3:.2f}, availability={4:.2f}".format(
                    sid,
                    self.island_idm[sid],
                    kpi.backlog,
                    kpi.tardiness_mean,
                    kpi.availability,
                )
            )
        lines.append("\nPlan feasibility / quality:")
        for line in plan_quality:
            lines.append(f"  {line}")
        lines.append("\nRobot details:")
        for r in self.robots:
            lines.append(
                "  Robot {0} ({1}) -> IDM={2:.3f}, util={3:.2f}, queue={4}, wait={5:.2f}, completed={6}".format(
                    r.robot_id,
                    "/".join(r.skills),
                    r.current_idm,
                    r.current_kpi.utilization,
                    r.current_kpi.queue_length,
                    r.current_kpi.avg_waiting_time,
                    r.completed_count,
                )
            )
        lines.append("\nFirst 10 tasks:")
        for t in self.tokens[:10]:
            lines.append(
                "  {0} root={1} skill={2} due={3} robot={4} end={5} state={6}".format(
                    t.token_id,
                    t.root_order_id,
                    t.required_skill,
                    t.due_time,
                    t.assigned_robot_id,
                    t.process_end_time,
                    t.state,
                )
            )
        return "\n".join(lines)

    # -------------------------
    # Plan quality
    # -------------------------
    def plan_quality_report(self) -> list[str]:
        issues = []
        planned = [t for t in self.tokens if t.planned_robot_id is not None]
        if len(planned) != len(self.tokens):
            issues.append(f"Unplanned tasks: {len(self.tokens) - len(planned)}")

        # check precedence order in planned sequence
        precedence_violations = 0
        for order in self.orders:
            seq = sorted(order.children, key=lambda n: n.seq_index or 0)
            times = []
            for node in seq:
                t = next((x for x in self.tokens if x.node.node_id == node.node_id), None)
                if t is None or t.planned_start is None:
                    continue
                times.append(t.planned_start)
            if times != sorted(times):
                precedence_violations += 1
        if precedence_violations:
            issues.append(f"Orders with precedence violations in plan: {precedence_violations}")

        # planned robot utilization and makespan
        robot_windows: dict[int, list[tuple[int, int]]] = {r.robot_id: [] for r in self.robots}
        for t in planned:
            robot_windows[t.planned_robot_id].append((t.planned_start or 0, t.planned_finish or 0))

        makespan = 0
        total_busy = 0
        for rid, windows in robot_windows.items():
            if not windows:
                continue
            windows.sort()
            start = min(w[0] for w in windows)
            end = max(w[1] for w in windows)
            makespan = max(makespan, end)
            busy = sum(max(0, w[1] - w[0]) for w in windows)
            total_busy += busy
            horizon = max(1, end - start)
            util = busy / horizon
            issues.append(f"Planned util robot {rid}: {util:.2f}")

        if makespan > 0:
            issues.append(f"Planned makespan: {makespan}")
            overall_util = total_busy / (makespan * max(1, len(self.robots)))
            issues.append(f"Planned overall util: {overall_util:.2f}")

        # bottleneck skill
        skill_counts = {}
        for r in self.robots:
            for s in r.skills:
                skill_counts[s] = skill_counts.get(s, 0) + 1
        skill_load = {}
        for t in self.tokens:
            skill_load[t.required_skill] = skill_load.get(t.required_skill, 0) + t.nominal_duration
        if skill_load:
            bottleneck = max(skill_load.items(), key=lambda kv: kv[1] / max(1, skill_counts.get(kv[0], 1)))
            issues.append(f"Bottleneck skill: {bottleneck[0]} (load {bottleneck[1]})")

        return issues if issues else ["No obvious plan issues detected"]

    def _compute_planned_makespan(self) -> int:
        planned = [t for t in self.tokens if t.planned_finish is not None]
        if not planned:
            return 0
        return max(int(t.planned_finish or 0) for t in planned)

    # -------------------------
    # Replanning
    # -------------------------
    def _replan_mission(self, station_id: int):
        now = self.steps
        remaining = [
            t
            for t in self.tokens
            if t.state != "done" and t.release_time <= now and t.station_id == station_id
        ]
        remaining.sort(key=lambda t: (t.due_time, t.release_time, t.token_id))
        clock = now
        for t in remaining:
            clock += t.nominal_duration
            t.nominal_finish_target = clock

        self.mission_epoch = now
        self.island_idm[station_id] = 1.0
        self.island_monitors[station_id]._last_backlog = 0
        self.island_monitors[station_id].idm.reset(self.mission_length)
        self.station_replan_events.append((self.steps, station_id))
        # island-level replan implies suborder replan on robots
        station_robots = self.robots_for_station(station_id)
        self._reschedule_robots(station_robots)
        for r in station_robots:
            r.monitor.idm.reset(self.robot_mission_length)
            r.last_support_step = self.steps
            self.robot_mission_reset_events.append((self.steps, r.robot_id))


    def register_support_request(self, robot_id: int):
        station_id = next((r.station_id for r in self.robots if r.robot_id == robot_id), None)
        if station_id is None:
            return
        if robot_id not in self.support_requests[station_id]:
            self.support_requests[station_id].append(robot_id)

    def _handle_support_requests(self):
        for sid in self.station_ids:
            if not self.support_requests[sid]:
                continue

            drop = self._last_station_idm.get(sid, 0.0) - self.station_idm.get(sid, 1.0)
            station_concerned = self.station_idm[sid] < self.station_support_threshold or drop > self.station_support_drop

            if not station_concerned:
                self.support_requests[sid].clear()
                continue

            action = self._select_reschedule_action(sid)
            reschedule_robots = self._apply_reschedule_action(action, station_id=sid)

            for r in reschedule_robots:
                r.monitor.idm.reset(self.robot_mission_length)
                r.last_support_step = self.steps
                self.robot_mission_reset_events.append((self.steps, r.robot_id))

            self.station_reschedule_events.append((self.steps, sid, action))

            self.support_requests[sid].clear()

    def _select_reschedule_action(self, station_id: int) -> str:
        station_robots = self.robots_for_station(station_id)
        dying = sum(1 for r in station_robots if r.current_idm < self.robot_critical_threshold)
        if dying >= max(1, len(station_robots) // 2):
            return "full"
        if self.station_idm[station_id] < self.station_support_critical:
            return "full"

        candidates = ["local", "skill", "full"]
        scores = {c: self._score_reschedule_action(c, station_id) for c in candidates}
        return min(scores, key=scores.get)

    def _score_reschedule_action(self, action: str, station_id: int) -> float:
        tasks = [t for t in self.tokens if t.state != "done" and t.station_id == station_id]
        if not tasks:
            return 0.0

        now = self.steps
        station_robots = self.robots_for_station(station_id)
        robot_loads = {r.robot_id: 0 for r in station_robots}
        robot_skill = {r.robot_id: r.skills for r in station_robots}

        if action == "local":
            impacted = set(self.support_requests[station_id])
            for r in station_robots:
                if r.robot_id in impacted:
                    robot_loads[r.robot_id] = sum(t.nominal_duration for t in r.queue)
                else:
                    robot_loads[r.robot_id] = len(r.queue) * 2
        else:
            per_skill: dict[str, list[TaskToken]] = {}
            for t in tasks:
                per_skill.setdefault(t.required_skill, []).append(t)
            for skill, tlist in per_skill.items():
                candidates = [rid for rid, sk in robot_skill.items() if skill in sk]
                if not candidates:
                    continue
                for t in sorted(tlist, key=lambda x: (x.due_time, x.release_time, x.token_id)):
                    rid = min(candidates, key=lambda r: robot_loads[r])
                    robot_loads[rid] += t.nominal_duration

        # proxy objective: tardiness risk + imbalance
        avg_load = sum(robot_loads.values()) / max(1, len(robot_loads))
        imbalance = statistics.pstdev(list(robot_loads.values())) if len(robot_loads) > 1 else 0.0
        tardiness_risk = 0.0
        for t in tasks:
            tardiness_risk += max(0, (now + t.nominal_duration) - t.due_time)
        tardiness_risk /= max(1, len(tasks))

        return 0.6 * tardiness_risk + 0.4 * imbalance

    def _apply_reschedule_action(self, action: str, station_id: int) -> list[RobotAgent]:
        if action == "local":
            reschedule_ids = set(self.support_requests[station_id])
            robots = [r for r in self.robots_for_station(station_id) if r.robot_id in reschedule_ids]
            self._reschedule_robots(robots)
            return robots
        if action == "skill":
            reschedule_ids = set(self.support_requests[station_id])
            for r in self.robots_for_station(station_id):
                if r.current_idm < self.robot_support_threshold:
                    reschedule_ids.add(r.robot_id)
            robots = [r for r in self.robots_for_station(station_id) if r.robot_id in reschedule_ids]
            self._reschedule_robots(robots)
            return robots

        # full
        station_robots = self.robots_for_station(station_id)
        self._reschedule_robots(station_robots)
        return station_robots[:]

    def _reschedule_robots(self, robots: list[RobotAgent]):
        if not robots:
            return
        now = self.steps
        for skill in set(s for r in robots for s in r.skills):
            skill_robots = [r for r in robots if skill in r.skills]
            tasks = []
            for r in skill_robots:
                while r.queue:
                    tasks.append(r.queue.popleft())
            tasks.sort(key=lambda t: (t.due_time, t.release_time, t.token_id))
            for idx, task in enumerate(tasks):
                target = skill_robots[idx % len(skill_robots)]
                target.enqueue(task)

        # new mission plan for rescheduled robots: set new reference finish targets
        for r in robots:
            start = now + (r.remaining if r.current is not None else 0)
            clock = start
            for t in list(r.queue):
                t.planned_robot_id = r.robot_id
                t.planned_start = clock
                clock += t.nominal_duration
                t.nominal_finish_target = clock
                t.planned_finish = clock
        self._build_planned_sequences()

    def _factory_reschedule_stages(self, source_station: int, target_station: int, max_orders: int = 3):
        moved = 0
        for order in self.orders:
            key = (order.node_id, "FE")
            if key not in self.stage_station_assignment:
                continue
            if self.stage_station_assignment[key] != source_station:
                continue
            # only move if all FE subtasks for this order are still planned
            fe_tokens = [t for t in self.tokens if t.root_order_id == order.node_id and t.stage == "FE"]
            if not fe_tokens or any(t.state != "planned" for t in fe_tokens):
                continue
            for t in fe_tokens:
                t.station_id = target_station
                t.planned_robot_id = None
                t.planned_start = None
                t.planned_finish = None
            self.stage_station_assignment[key] = target_station
            moved += 1
            if moved >= max_orders:
                break
        if moved > 0:
            self.factory_reschedule_events.append((self.steps, source_station, target_station))

    # -------------------------
    # Precedence tracking
    # -------------------------
    def _order_allows_release(self, token: TaskToken) -> bool:
        done_index = self.order_progress.get(token.root_order_id, -1)
        return token.seq_index <= done_index + 1

    def register_task_completion(self, token: TaskToken):
        current = self.order_progress.get(token.root_order_id, -1)
        if token.seq_index > current:
            self.order_progress[token.root_order_id] = token.seq_index
        if token.assigned_robot_id is not None:
            self.order_last_robot[token.root_order_id] = token.assigned_robot_id
        self.log_token_event(token)

    def compute_transport_time(self, from_robot_id: int | None, to_robot_id: int | None) -> int:
        if to_robot_id is None:
            return 2
        if from_robot_id is None:
            return 2
        return 1 if from_robot_id == to_robot_id else 3

    # -------------------------
    # Event logging
    # -------------------------
    def log_robot_resequence(self, robot_id: int):
        self.robot_resequence_events.append((self.steps, robot_id))
        last_penalty_step = self.robot_resequence_penalty_steps.get(robot_id, -10_000)
        if self.steps - last_penalty_step < self.robot_resequence_penalty_cooldown:
            return

        # Penalize resequencing, but not so aggressively that one busy robot is forced to zero.
        for r in self.robots:
            if r.robot_id == robot_id:
                r.monitor.idm.add_penalty(self.robot_resequence_penalty)
                self.robot_resequence_penalty_steps[robot_id] = self.steps
                break

    def log_token_event(self, token: TaskToken):
        self.token_events.append(
            {
                "step": self.steps,
                "token_id": token.token_id,
                "state": token.state,
                "robot_id": token.assigned_robot_id,
            }
        )

    def _log_step(self):
        snapshot = {
            "step": self.steps,
            "factory": {
                "idm": self.factory_idm,
                "backlog": self.factory_kpi.backlog,
                "tardiness_mean": self.factory_kpi.tardiness_mean,
                "availability": self.factory_kpi.availability,
            },
            "islands": [
                {
                    "id": sid,
                    "idm": self.island_idm[sid],
                    "backlog": self.island_kpi[sid].backlog,
                    "tardiness_mean": self.island_kpi[sid].tardiness_mean,
                    "availability": self.island_kpi[sid].availability,
                    "buffer_in": len(self.station_input_buffers[sid]),
                    "buffer_out": len(self.station_output_buffers[sid]),
                }
                for sid in self.station_ids
            ],
            "robots": [
                {
                    "id": r.robot_id,
                    "idm": r.current_idm,
                    "utilization": r.current_kpi.utilization,
                    "delay": r.current_kpi.delay,
                    "availability": r.current_kpi.availability,
                    "queue": len(r.queue),
                }
                for r in self.robots
            ],
            "tokens": self._token_positions(),
            "events": {
                "resequences": len(self.robot_resequence_events),
                "reschedules": len(self.station_reschedule_events),
                "replans": len(self.station_replan_events),
            },
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

    def _token_positions(self):
        # grid coordinates used for replay
        robot_positions = {
            1: (4, 5),
            2: (6, 5),
            3: (5, 3),
        }
        station_in = {1: (2, 4)}
        station_out = {1: (7, 4)}
        supplier_in = (1, 7)
        last = {}
        for e in self.token_events:
            if e["step"] > self.steps:
                break
            last[e["token_id"]] = e
        token_by_id = {t.token_id: t for t in self.tokens}
        tokens = []
        for tid, e in last.items():
            state = e["state"]
            token = token_by_id.get(tid)
            if state == "waiting_agv":
                if token and token.from_robot_id is not None:
                    sx, sy = robot_positions.get(token.from_robot_id, station_in.get(1, (2, 4)))
                else:
                    sx, sy = supplier_in
                tokens.append({"id": tid, "x": sx, "y": sy, "state": state})
            elif state == "in_transport":
                if token and token.from_robot_id is not None:
                    sx, sy = robot_positions.get(token.from_robot_id, station_in.get(1, (2, 4)))
                else:
                    sx, sy = supplier_in
                if token and token.transport_target_robot_id is not None:
                    ex, ey = robot_positions.get(token.transport_target_robot_id, station_in.get(1, (2, 4)))
                else:
                    ex, ey = station_in.get(token.station_id or 1, (2, 4))
                if token and token.transport_start_step is not None and token.transport_end_step is not None:
                    denom = max(1, token.transport_end_step - token.transport_start_step)
                    frac = (self.steps - token.transport_start_step) / denom
                    frac = min(max(frac, 0.0), 1.0)
                else:
                    frac = 0.0
                x = sx + (ex - sx) * frac
                y = sy + (ey - sy) * frac
                tokens.append({"id": tid, "x": x, "y": y, "state": state})
            elif state == "waiting_station":
                sin = station_in.get(token.station_id or 1, (2, 4)) if token else (2, 4)
                tokens.append({"id": tid, "x": sin[0], "y": sin[1], "state": state})
            elif state == "waiting_robot":
                rx, ry = robot_positions.get(e.get("robot_id") or 1, (4, 5))
                tokens.append({"id": tid, "x": rx - 1, "y": ry, "state": state})
            elif state == "processing":
                rx, ry = robot_positions.get(e.get("robot_id") or 1, (4, 5))
                tokens.append({"id": tid, "x": rx, "y": ry, "state": state})
            elif state == "done":
                sout = station_out.get(token.station_id or 1, (7, 4)) if token else (7, 4)
                tokens.append({"id": tid, "x": sout[0], "y": sout[1], "state": state})
        return tokens
