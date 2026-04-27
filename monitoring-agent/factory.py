from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import HolonicSchedulingModel


class FactoryController:
    """Higher-level holon controlling island mission resets and container reschedules."""

    def __init__(self, model: "HolonicSchedulingModel"):
        self.model = model

    def step(self):
        m = self.model
        for sid in m.station_ids:
            last_idm = m._last_station_idm.get(sid, m.station_idm.get(sid, 1.0))
            drop = last_idm - m.station_idm.get(sid, 1.0)
            m._last_station_idm[sid] = m.station_idm.get(sid, 1.0)

            if m.station_idm.get(sid, 1.0) >= m.station_replan_threshold:
                continue

            if m._last_factory_replan_step is not None:
                if m.steps - m._last_factory_replan_step < m.factory_replan_cooldown:
                    continue

            recent_reschedule = False
            if m.station_reschedule_events:
                last_resched_step = next(
                    (ev[0] for ev in reversed(m.station_reschedule_events) if ev[1] == sid),
                    None,
                )
                if last_resched_step is not None and m.steps - last_resched_step <= m.factory_recent_reschedule_window:
                    recent_reschedule = True

            cant_help = m.station_idm[sid] < m.station_support_critical or (recent_reschedule and drop > 0.0)
            if cant_help:
                # Factory tries to move FE containers to island 1 before forcing a replan.
                if sid == 2 and m.station_idm.get(1, 1.0) > m.station_support_critical:
                    m._factory_reschedule_stages(source_station=2, target_station=1, max_orders=2)
                else:
                    m._replan_mission(sid)
                    m._last_factory_replan_step = m.steps
