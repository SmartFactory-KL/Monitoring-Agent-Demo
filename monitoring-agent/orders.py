from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OrderNode:
    """Self-similar order node.

    A node can be a full order or a suborder. Each node can contain children.
    Leaf nodes are executable work packages.
    """

    node_id: str
    family: str
    release_time: int
    due_time: int
    required_skill: str
    nominal_process_time: int
    quantity: int = 1
    children: List["OrderNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    seq_index: Optional[int] = None

    # runtime state
    created_at: Optional[int] = None
    assigned_at: Optional[int] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    transported_at: Optional[int] = None
    assigned_robot_id: Optional[int] = None
    state: str = "planned"  # planned, released, in_transport, waiting, processing, done

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flatten_leaves(self) -> List["OrderNode"]:
        if self.is_leaf():
            return [self]
        leaves: List[OrderNode] = []
        for child in self.children:
            leaves.extend(child.flatten_leaves())
        return leaves


@dataclass
class TaskToken:
    """Executable leaf task derived from an order hierarchy."""

    token_id: str
    root_order_id: str
    node: OrderNode
    required_skill: str
    nominal_duration: int
    release_time: int
    due_time: int
    seq_index: int = 0
    seq_total: int = 1
    planned_robot_id: Optional[int] = None
    planned_start: Optional[int] = None
    planned_finish: Optional[int] = None
    station_id: Optional[int] = None
    stage: Optional[str] = None
    from_robot_id: Optional[int] = None
    transport_target_robot_id: Optional[int] = None
    transport_time: Optional[int] = None
    transport_start_step: Optional[int] = None
    transport_end_step: Optional[int] = None

    state: str = "planned"  # planned, waiting_agv, in_transport, waiting_robot, processing, done
    assigned_robot_id: Optional[int] = None
    assigned_agv_id: Optional[int] = None
    agv_pickup_time: Optional[int] = None
    robot_queue_enter_time: Optional[int] = None
    process_start_time: Optional[int] = None
    process_end_time: Optional[int] = None
    nominal_finish_target: Optional[int] = None
