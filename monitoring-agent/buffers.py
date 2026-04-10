from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from orders import TaskToken


@dataclass
class Buffer:
    name: str
    size: int
    content: List[TaskToken] = field(default_factory=list)

    def has_space(self) -> bool:
        return len(self.content) < self.size

    def add(self, task: TaskToken) -> bool:
        if not self.has_space():
            return False
        self.content.append(task)
        return True

    def pop(self) -> TaskToken | None:
        if not self.content:
            return None
        return self.content.pop(0)

    def __len__(self) -> int:
        return len(self.content)
