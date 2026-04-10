from __future__ import annotations

from model import HolonicSchedulingModel
from pathlib import Path
import shutil


def run_demo(seed: int = 42, n_orders: int = 10, max_steps: int = 600, visu: bool | None = None):
    def model_factory():
        return HolonicSchedulingModel(seed=seed, n_orders=n_orders, max_steps=max_steps)

    model = model_factory()
    model.run()
    print(model.final_report())

    try:
        src = Path("sim_step_log.jsonl")
        dst = Path("godot_replay") / "data" / "sim_step_log.jsonl"
        if src.exists() and dst.parent.exists():
            shutil.copyfile(src, dst)
            print(f"\nCopied log to {dst}")
    except Exception:
        pass

    try:
        df = model.datacollector.get_model_vars_dataframe()
        print("\nLast 10 model rows:")
        print(df.tail(10).to_string(index=False))
    except Exception:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        factory_series = model.factory_monitor.history
        if factory_series:
            steps = [r["step"] for r in factory_series]
            factory_idm = [r["idm"] for r in factory_series]
        else:
            steps, factory_idm = [], []

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(steps, factory_idm, label="Factory IDM", linewidth=2)
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel("IDM")
        axes[0].set_title("Factory IDM")
        axes[0].legend(loc="lower left")

        for sid, mon in model.station_monitors.items():
            hist = mon.history
            if not hist:
                continue
            s_steps = [h["step"] for h in hist]
            s_idm = [h["idm"] for h in hist]
            axes[1].plot(s_steps, s_idm, alpha=0.8, label=f"Station {sid} IDM")
        for t, sid, action in model.station_reschedule_events:
            color = {"local": "orange", "skill": "goldenrod", "full": "red"}.get(action, "orange")
            axes[1].axvline(t, color=color, alpha=0.6, linewidth=1)
        for t, sid in model.station_replan_events:
            axes[1].axvline(t, color="blue", alpha=0.6, linewidth=1, linestyle="--")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("IDM")
        axes[1].set_title("Station IDMs (red/gold/orange=reschedule, blue=station replan)")
        axes[1].legend(loc="lower left")

        for r in model.robots:
            hist = r.monitor.history
            r_steps = [h["step"] for h in hist]
            r_idm = [h["idm"] for h in hist]
            axes[2].plot(r_steps, r_idm, alpha=0.8, label=f"Robot {r.robot_id}")

        if model.robot_resequence_events:
            rs_steps = [e[0] for e in model.robot_resequence_events]
            rs_vals = [next((h["idm"] for h in model.robots[e[1]-1].monitor.history if h["step"] == e[0]), None) for e in model.robot_resequence_events]
            rs_vals = [v if v is not None else 0.0 for v in rs_vals]
            axes[2].scatter(rs_steps, rs_vals, s=18, color="purple", label="Robot resequence")

        if model.robot_mission_reset_events:
            rm_steps = [e[0] for e in model.robot_mission_reset_events]
            rm_vals = [1.0 for _ in model.robot_mission_reset_events]
            axes[2].scatter(rm_steps, rm_vals, s=22, color="green", label="Robot mission reset")

        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("IDM")
        axes[2].set_title("Robot IDMs")
        axes[2].legend(loc="lower left", ncol=2)

        plt.tight_layout()
        plt.savefig("idm_plot.png", dpi=140)
        print("\nSaved plot to idm_plot.png")
    except Exception as e:
        print("\nPlot skipped:", e)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def color_for(order_id: str) -> tuple[float, float, float]:
            h = abs(hash(order_id)) % 360
            return (0.4 + 0.6 * (h % 3) / 3.0, 0.4 + 0.6 * (h % 5) / 5.0, 0.4 + 0.6 * (h % 7) / 7.0)

        planned = [t for t in model.tokens if t.planned_start is not None and t.planned_finish is not None]
        if planned:
            fig, ax = plt.subplots(figsize=(10, 6))
            y_map = {r.robot_id: i for i, r in enumerate(model.robots)}
            for t in planned:
                y = y_map.get(t.planned_robot_id, 0)
                ax.barh(y, t.planned_finish - t.planned_start, left=t.planned_start, color=color_for(t.root_order_id))
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels([f"Robot {r.robot_id}" for r in model.robots])
            ax.set_xlabel("Time")
            ax.set_title("Planned Gantt (Initial Schedule)")
            plt.tight_layout()
            plt.savefig("gantt_planned.png", dpi=140)
            print("Saved planned Gantt to gantt_planned.png")

        actual = [t for t in model.tokens if t.process_start_time is not None and t.process_end_time is not None]
        if actual:
            fig, ax = plt.subplots(figsize=(10, 6))
            y_map = {r.robot_id: i for i, r in enumerate(model.robots)}
            for t in actual:
                y = y_map.get(t.assigned_robot_id, 0)
                ax.barh(y, t.process_end_time - t.process_start_time, left=t.process_start_time, color=color_for(t.root_order_id))
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels([f"Robot {r.robot_id}" for r in model.robots])
            ax.set_xlabel("Time")
            ax.set_title("Actual Gantt (Realized Execution)")
            plt.tight_layout()
            plt.savefig("gantt_actual.png", dpi=140)
            print("Saved actual Gantt to gantt_actual.png")

        # Station-level Gantt (planned)
        planned = [t for t in model.tokens if t.planned_start is not None and t.planned_finish is not None]
        if planned:
            fig, ax = plt.subplots(figsize=(10, 5))
            station_ids = sorted(set(t.station_id for t in planned if t.station_id is not None))
            y_map = {sid: i for i, sid in enumerate(station_ids)}
            for t in planned:
                if t.station_id is None:
                    continue
                y = y_map.get(t.station_id, 0)
                ax.barh(y, t.planned_finish - t.planned_start, left=t.planned_start, color=color_for(t.root_order_id))
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels([f"Station {sid}" for sid in station_ids])
            ax.set_xlabel("Time")
            ax.set_title("Planned Gantt (Station Level)")
            plt.tight_layout()
            plt.savefig("gantt_station_planned.png", dpi=140)
            print("Saved station planned Gantt to gantt_station_planned.png")

        # Station-level Gantt (actual)
        actual = [t for t in model.tokens if t.process_start_time is not None and t.process_end_time is not None]
        if actual:
            fig, ax = plt.subplots(figsize=(10, 5))
            station_ids = sorted(set(t.station_id for t in actual if t.station_id is not None))
            y_map = {sid: i for i, sid in enumerate(station_ids)}
            for t in actual:
                if t.station_id is None:
                    continue
                y = y_map.get(t.station_id, 0)
                ax.barh(y, t.process_end_time - t.process_start_time, left=t.process_start_time, color=color_for(t.root_order_id))
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels([f"Station {sid}" for sid in station_ids])
            ax.set_xlabel("Time")
            ax.set_title("Actual Gantt (Station Level)")
            plt.tight_layout()
            plt.savefig("gantt_station_actual.png", dpi=140)
            print("Saved station actual Gantt to gantt_station_actual.png")
    except Exception as e:
        print("\nGantt plots skipped:", e)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Station KPI evolution
        sh = model.factory_monitor.history
        if sh:
            steps = [r["step"] for r in sh]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, [r["throughput"] for r in sh], label="throughput")
            ax.plot(steps, [r["tardiness_mean"] for r in sh], label="tardiness_mean")
            ax.plot(steps, [r["availability"] for r in sh], label="availability")
            ax.set_xlabel("Step")
            ax.set_title("Factory KPI Evolution")
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig("kpi_station.png", dpi=140)
            print("Saved factory KPI plot to kpi_station.png")

        # Robot KPI evolution (one subplot per robot)
        fig, axes = plt.subplots(len(model.robots), 1, figsize=(10, 3 * len(model.robots)), sharex=True)
        if len(model.robots) == 1:
            axes = [axes]
        for ax, r in zip(axes, model.robots):
            rh = r.monitor.history
            if not rh:
                continue
            steps = [x["step"] for x in rh]
            ax.plot(steps, [x["utilization"] for x in rh], label="utilization")
            ax.plot(steps, [x["delay"] for x in rh], label="delay")
            ax.plot(steps, [x["availability"] for x in rh], label="availability")
            ax.set_title(f"Robot {r.robot_id} KPI Evolution")
            ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("kpi_robots.png", dpi=140)
        print("Saved robot KPI plot to kpi_robots.png")
    except Exception as e:
        print("\nKPI plots skipped:", e)

    try:
        import json

        def node_to_dict(node):
            return {
                "id": node.node_id,
                "skill": node.required_skill,
                "start": node.started_at,
                "end": node.finished_at,
                "children": [node_to_dict(c) for c in node.children],
            }

        hierarchy = [node_to_dict(o) for o in model.orders]
        with open("order_hierarchy.json", "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)
        print("Saved order hierarchy to order_hierarchy.json")
    except Exception as e:
        print("\nHierarchy export skipped:", e)


if __name__ == "__main__":
    run_demo(seed=7, n_orders=100, max_steps=600, visu=True)
