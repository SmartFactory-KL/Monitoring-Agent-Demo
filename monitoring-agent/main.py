from __future__ import annotations

import csv
import json
from model import HolonicSchedulingModel
from pathlib import Path
import shutil


def make_idm_plot(model: HolonicSchedulingModel, output_base: str = "idm_plot", show_axis_labels: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.6), sharex=True)
    fig.patch.set_facecolor("white")

    island_color = "#4C78A8"
    idm_color = "black"
    availability_color = "#059669"
    utilization_color = "#72B7B2"
    delay_color = "#D95F02"
    wait_color = "#7C3AED"
    backlog_color = "#2563EB"
    event_color = "#C44E52"
    reset_color = "#6B7280"
    grid_color = "#D1D5DB"
    legend_face = "white"
    legend_edge = "#9CA3AF"

    def add_legend_box(ax, handles, labels, box, ncol=1, fontsize=9):
        legend_ax = ax.inset_axes(box, zorder=30)
        legend_ax.set_facecolor(legend_face)
        for spine in legend_ax.spines.values():
            spine.set_edgecolor(legend_edge)
            spine.set_linewidth(0.8)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        legend_ax.patch.set_alpha(1.0)
        legend = legend_ax.legend(
            handles,
            labels,
            loc="center left",
            frameon=False,
            ncol=ncol,
            fontsize=fontsize,
        )
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        return legend_ax, legend

    ax_island = axes[0]
    ax_island_buffer = ax_island.twinx()
    ax_island_tardiness = ax_island.twinx()
    ax_island_state = ax_island.twinx()
    ax_island_tardiness.spines["right"].set_position(("axes", 1.14))
    ax_island_tardiness.spines["right"].set_visible(True)
    ax_island_state.spines["right"].set_position(("axes", 1.28))
    ax_island_state.spines["right"].set_visible(True)

    island_handles = []

    for sid, mon in model.island_monitors.items():
        hist = mon.history
        if not hist:
            continue
        s_steps = [h["step"] for h in hist]
        s_idm = [h["idm"] for h in hist]
        s_backlog = [h["backlog"] for h in hist]
        s_tardiness = [h["tardiness_mean"] for h in hist]
        s_availability = [h["availability"] for h in hist]
        line_island_idm, = ax_island.plot(
            s_steps,
            s_idm,
            linewidth=2,
            alpha=0.95,
            color=idm_color,
            label=f"Island {sid} IDM",
        )
        line_backlog, = ax_island_buffer.plot(
            s_steps,
            s_backlog,
            linewidth=1.4,
            alpha=0.9,
            color=backlog_color,
            label="Buffer",
        )
        line_tardiness, = ax_island_tardiness.plot(
            s_steps,
            s_tardiness,
            linewidth=1.4,
            alpha=0.9,
            color=delay_color,
            linestyle="--",
            label="Tardiness",
        )
        line_availability, = ax_island_state.plot(
            s_steps,
            s_availability,
            linewidth=1.4,
            alpha=0.9,
            color=availability_color,
            linestyle="-",
            label="Availability",
        )
        island_handles.extend([line_island_idm, line_backlog, line_tardiness, line_availability])
        step_to_idm = {h["step"]: h["idm"] for h in hist}
        reset_steps = sorted({t for t, _robot_id in model.robot_mission_reset_events})
        for idx, t in enumerate(reset_steps):
            y = step_to_idm.get(t)
            if y is None:
                continue
            event = ax_island.scatter(
                [t],
                [y],
                s=18,
                color=event_color,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
                label="Island self-resequence" if idx == 0 else None,
            )
            if idx == 0:
                island_handles.append(event)
    ax_island.set_ylim(0.0, 1.05)
    ax_island_buffer.set_ylim(bottom=0.0)
    ax_island_tardiness.set_ylim(bottom=0.0)
    ax_island_state.set_ylim(0.0, 1.05)
    ax_island.set_ylabel("IDM [-]" if show_axis_labels else "")
    ax_island_buffer.set_ylabel("Buffer [pcs.]" if show_axis_labels else "")
    ax_island_tardiness.set_ylabel("Tardiness [min]" if show_axis_labels else "")
    ax_island_state.set_ylabel("Availability [-]" if show_axis_labels else "")
    ax_island_buffer.yaxis.labelpad = 10
    ax_island_tardiness.yaxis.labelpad = 14
    ax_island_state.yaxis.labelpad = 18
    ax_island.set_title("Island IDM and KPIs")
    ax_island.grid(True, axis="y", linestyle=":", color=grid_color, alpha=0.5)
    handles = island_handles
    labels = [h.get_label() for h in handles]
    unique = dict(zip(labels, handles))
    add_legend_box(ax_island_state, list(unique.values()), list(unique.keys()), [0.03, 0.04, 0.55, 0.30], ncol=1, fontsize=8.5)

    selected_machine = min(model.robots, key=lambda r: r.current_idm)
    hist = selected_machine.monitor.history
    m_steps = [h["step"] for h in hist]
    m_idm = [h["idm"] for h in hist]
    m_avail = [h["availability"] for h in hist]
    m_delay = [h["delay"] for h in hist]
    m_buffer = [h["queue_length"] for h in hist]

    ax_machine = axes[1]
    ax_machine_buffer = ax_machine.twinx()
    ax_machine_tardiness = ax_machine.twinx()
    ax_machine_state = ax_machine.twinx()
    ax_machine_tardiness.spines["right"].set_position(("axes", 1.14))
    ax_machine_tardiness.spines["right"].set_visible(True)
    ax_machine_state.spines["right"].set_position(("axes", 1.28))
    ax_machine_state.spines["right"].set_visible(True)

    line_idm, = ax_machine.plot(m_steps, m_idm, linewidth=2.0, color=idm_color, label=f"Machine {selected_machine.robot_id} IDM")
    line_avail, = ax_machine_state.plot(m_steps, m_avail, linewidth=1.5, color=availability_color, label="Availability")
    line_delay, = ax_machine_tardiness.plot(m_steps, m_delay, linewidth=1.5, color=delay_color, linestyle="--", label="Tardiness")
    line_buffer, = ax_machine_buffer.plot(m_steps, m_buffer, linewidth=1.4, color=backlog_color, label="Buffer")

    if model.robot_resequence_events:
        rs_steps = [e[0] for e in model.robot_resequence_events if e[1] == selected_machine.robot_id]
        rs_vals = [next((h["idm"] for h in hist if h["step"] == step), None) for step in rs_steps]
        rs_vals = [v if v is not None else 0.0 for v in rs_vals]
        ax_machine.scatter(
            rs_steps,
            rs_vals,
            s=16,
            color=event_color,
            edgecolors="white",
            linewidths=0.3,
            label="Machine self-resequence",
            zorder=4,
        )

    ax_machine.set_ylim(0.0, 1.05)
    ax_machine_buffer.set_ylim(bottom=0.0)
    ax_machine_tardiness.set_ylim(bottom=0.0)
    ax_machine_state.set_ylim(0.0, 1.05)
    ax_machine.set_xlabel("Time [min]" if show_axis_labels else "")
    ax_machine.set_ylabel("IDM [-]" if show_axis_labels else "")
    ax_machine_buffer.set_ylabel("Buffer [pcs.]" if show_axis_labels else "")
    ax_machine_tardiness.set_ylabel("Tardiness [min]" if show_axis_labels else "")
    ax_machine_state.set_ylabel("Availability [-]" if show_axis_labels else "")
    ax_machine_buffer.yaxis.labelpad = 10
    ax_machine_tardiness.yaxis.labelpad = 14
    ax_machine_state.yaxis.labelpad = 18
    ax_machine.set_title(f"Machine {selected_machine.robot_id} IDM and KPIs")
    ax_machine.grid(True, axis="y", linestyle=":", color=grid_color, alpha=0.5)
    handles = [line_idm, line_buffer, line_delay, line_avail]
    extra_handles, extra_labels = ax_machine.get_legend_handles_labels()
    for handle, label in zip(extra_handles, extra_labels):
        if label not in [h.get_label() for h in handles]:
            handles.append(handle)
    labels = [h.get_label() for h in handles]
    unique = dict(zip(labels, handles))
    add_legend_box(ax_machine_state, list(unique.values()), list(unique.keys()), [0.02, 0.04, 0.58, 0.18], ncol=2, fontsize=8.0)

    fig.tight_layout(pad=0.6, h_pad=0.45)
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.savefig(f"{output_base}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"\nSaved plot to {output_base}.pdf and {output_base}.svg")


def run_demo(
    seed: int = 42,
    n_orders: int = 10,
    max_steps: int = 600,
    visu: bool | None = None,
    idm_tolerance: float = 0.25,
    idm_penalty_scale: float = 1.0,
    output_prefix: str = "",
    save_aux_outputs: bool = True,
):
    def model_factory():
        return HolonicSchedulingModel(
            seed=seed,
            n_orders=n_orders,
            max_steps=max_steps,
            idm_tolerance=idm_tolerance,
            idm_penalty_scale=idm_penalty_scale,
        )

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
        plot_base = f"{output_prefix}idm_plot" if output_prefix else "idm_plot"
        make_idm_plot(model, plot_base)
    except Exception as e:
        print("\nPlot skipped:", e)

    if save_aux_outputs:
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
                ax.set_yticklabels([f"Island {sid}" for sid in station_ids])
                ax.set_xlabel("Time")
                ax.set_title("Planned Gantt (Island Level)")
                plt.tight_layout()
                plt.savefig("gantt_island_planned.png", dpi=140)
                print("Saved island planned Gantt to gantt_island_planned.png")

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
                ax.set_yticklabels([f"Island {sid}" for sid in station_ids])
                ax.set_xlabel("Time")
                ax.set_title("Actual Gantt (Island Level)")
                plt.tight_layout()
                plt.savefig("gantt_island_actual.png", dpi=140)
                print("Saved island actual Gantt to gantt_island_actual.png")
        except Exception as e:
            print("\nGantt plots skipped:", e)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

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
                plt.savefig("kpi_factory.png", dpi=140)
                print("Saved factory KPI plot to kpi_factory.png")

            fig, axes = plt.subplots(len(model.robots), 1, figsize=(10, 3 * len(model.robots)), sharex=True)
            if len(model.robots) == 1:
                axes = [axes]
            for ax, r in zip(axes, model.robots):
                rh = r.machine_monitor.history
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

    return model


def run_idm_sweep(n_runs: int = 10, n_orders: int = 20, max_steps: int = 100):
    output_dir = Path("idm_sweep")
    output_dir.mkdir(exist_ok=True)

    configs = [
        {"run": i + 1, "seed": 7 + i, "tolerance": 0.25 + 0.03 * i, "penalty_scale": 1.0 - 0.05 * i}
        for i in range(n_runs)
    ]

    summary_rows = []
    for cfg in configs:
        run_name = f"run_{cfg['run']:02d}"
        print(
            f"\n=== Sweep {run_name}: seed={cfg['seed']}, tolerance={cfg['tolerance']:.2f}, "
            f"penalty_scale={cfg['penalty_scale']:.2f} ==="
        )
        model = run_demo(
            seed=cfg["seed"],
            n_orders=n_orders,
            max_steps=max_steps,
            visu=False,
            idm_tolerance=cfg["tolerance"],
            idm_penalty_scale=cfg["penalty_scale"],
            output_prefix=str(output_dir / f"{run_name}_"),
            save_aux_outputs=False,
        )
        robot_min_idm = min(min((h["idm"] for h in r.machine_monitor.history), default=1.0) for r in model.robots)
        island_min_idm = min(min((h["idm"] for h in mon.history), default=1.0) for mon in model.island_monitors.values())
        summary_rows.append(
            {
                "run": cfg["run"],
                "seed": cfg["seed"],
                "tolerance": f"{cfg['tolerance']:.2f}",
                "penalty_scale": f"{cfg['penalty_scale']:.2f}",
                "min_robot_idm": f"{robot_min_idm:.3f}",
                "min_island_idm": f"{island_min_idm:.3f}",
                "island_reschedules": len(model.station_reschedule_events),
                "island_replans": len(model.station_replan_events),
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved sweep summary to {summary_path}")


if __name__ == "__main__":
    run_idm_sweep(n_runs=10, n_orders=20, max_steps=300)
