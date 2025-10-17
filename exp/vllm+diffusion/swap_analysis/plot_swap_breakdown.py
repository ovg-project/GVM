#!/usr/bin/env python3
import argparse
import json
import sqlite3
from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Lighter-looking hatches
mpl.rcParams["hatch.linewidth"] = 0.6  # thinner hatch strokes

CASE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

SQL = """
WITH
-- pick the second request window by time
params AS (
  SELECT
    e.start AS t_start,
    e."end" AS t_end
  FROM NVTX_EVENTS e
  WHERE e.text LIKE 'req%:%'      -- matches 'req1: ...', 'req2: ...', etc.
  ORDER BY e.start
  LIMIT 1 OFFSET 1                 -- 0-based: OFFSET 1 = the second request
),

raw_gpu AS (
  SELECT s.start, s."end"
  FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS AS s, params p
  WHERE s.start < p.t_end AND s."end" > p.t_start
),
raw_htod AS (
  SELECT m.start, m."end"
  FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m, params p
  WHERE m.copyKind IN (1, 11)  -- HtoD
    AND m.start < p.t_end AND m."end" > p.t_start
),
raw_dtoh AS (
  SELECT m.start, m."end"
  FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m, params p
  WHERE m.copyKind IN (2, 12)  -- DtoH
    AND m.start < p.t_end AND m."end" > p.t_start
),

gpu AS (
  SELECT MAX(start, p.t_start) AS start, MIN("end", p.t_end) AS "end"
  FROM raw_gpu, params p
),
htod AS (
  SELECT MAX(start, p.t_start) AS start, MIN("end", p.t_end) AS "end"
  FROM raw_htod, params p
),
dtoh AS (
  SELECT MAX(start, p.t_start) AS start, MIN("end", p.t_end) AS "end"
  FROM raw_dtoh, params p
),

points AS (
  SELECT start AS t, +1 AS g,  0 AS hd, 0 AS dh FROM gpu
  UNION ALL SELECT "end",   -1,  0,    0        FROM gpu
  UNION ALL SELECT start,    0, +1,    0        FROM htod
  UNION ALL SELECT "end",    0, -1,    0        FROM htod
  UNION ALL SELECT start,    0,  0,   +1        FROM dtoh
  UNION ALL SELECT "end",    0,  0,   -1        FROM dtoh
),

scan AS (
  SELECT
    t,
    SUM(g)  OVER (ORDER BY t, g DESC, hd DESC, dh DESC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS g_depth,
    SUM(hd) OVER (ORDER BY t, g DESC, hd DESC, dh DESC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS hd_depth,
    SUM(dh) OVER (ORDER BY t, g DESC, hd DESC, dh DESC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS dh_depth
  FROM points
),

segments AS (
  SELECT
    t                                         AS t0,
    LEAD(t) OVER (ORDER BY t)                 AS t1,
    g_depth,
    hd_depth,
    dh_depth
  FROM scan
),

durations AS (
  SELECT
    SUM(CASE WHEN (hd_depth > 0 OR dh_depth > 0)
             THEN (t1 - t0) ELSE 0 END) AS io_any_ns,
    SUM(CASE WHEN hd_depth > 0
             THEN (t1 - t0) ELSE 0 END) AS io_htod_ns,
    SUM(CASE WHEN dh_depth > 0
             THEN (t1 - t0) ELSE 0 END) AS io_dtoh_ns,
    SUM(CASE WHEN g_depth > 0 AND (hd_depth = 0 AND dh_depth = 0)
             THEN (t1 - t0) ELSE 0 END) AS cpu_handler_ns
  FROM segments
  WHERE t1 IS NOT NULL
),

totals AS (
  SELECT 
    (p.t_end - p.t_start) AS total_ns,
    (p.t_end - p.t_start) / 1e9 AS total_seconds
  FROM params p
)

SELECT
  total_seconds                              AS "total_runtime_seconds",
  io_any_ns        / 1e9                     AS "pcie_io_seconds",
  io_htod_ns       / 1e9                     AS "pcie_io_htod_seconds",
  io_dtoh_ns       / 1e9                     AS "pcie_io_dtoh_seconds",
  cpu_handler_ns   / 1e9                     AS "cpu_pagefault_handler_seconds",
  (io_any_ns + cpu_handler_ns) / 1e9         AS "gpu_fault_window_seconds_est",
  100.0 * io_any_ns      / total_ns          AS "pct_pcie_io_of_run",
  100.0 * cpu_handler_ns / total_ns          AS "pct_cpu_handler_of_run",
  100.0 * (io_any_ns + cpu_handler_ns) / total_ns AS "pct_gpu_fault_window_of_run"
FROM durations, totals;
"""


def get_fault_metrics(db_path: str) -> Dict[str, Any]:
    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        row = cur.execute(SQL).fetchone()
        if row is None:
            return {}
        return {k: row[k] for k in row.keys()}
    finally:
        con.close()


def _fmt_seconds(x: float) -> str:
    if x >= 10:
        return f"{x:.1f}s"
    elif x >= 1:
        return f"{x:.2f}s"
    elif x >= 0.01:
        return f"{x:.3f}s"
    else:
        return f"{x:.4f}s"


def plot_stacked_bar(metrics: Dict[str, Any], out_path: str) -> None:
    total = float(metrics.get("total_runtime_seconds", 0.0))
    htod = float(metrics.get("pcie_io_htod_seconds", 0.0))
    dtoh = float(metrics.get("pcie_io_dtoh_seconds", 0.0))
    cpu = float(metrics.get("cpu_pagefault_handler_seconds", 0.0))
    pcie = htod + dtoh
    other = max(0.0, total - (pcie + cpu))

    segs = [
        ("Diffusion Inference", other, ".", CASE_COLORS[3]),
        ("CPU Page-Fault handler", cpu, "x", CASE_COLORS[2]),
        ("PCIe I/O DtoH", dtoh, "\\", CASE_COLORS[1]),
        ("PCIe I/O HtoD", htod, "/", CASE_COLORS[0]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    plt.subplots_adjust(bottom=0.38, top=0.83)

    ylab = ["Run"]
    left = 0.0
    text_pad = 0.05

    for label, width, hatch, color in segs:
        if width <= 0:
            continue
        bars = ax.barh(
            ylab,
            [width],
            left=left,
            edgecolor="black",
            facecolor=color,
            hatch=hatch,
            linewidth=1.0,
            label=label,
        )
        rect = bars.patches[0]
        xmid = rect.get_x() + rect.get_width() / 2.0
        y_top = rect.get_y() + rect.get_height() + text_pad
        ax.text(
            xmid,
            y_top,
            _fmt_seconds(width),
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
        left += width

    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, max(total, left) * 1.05)
    ax.set_yticks([])
    ax.margins(y=0.35)

    legend_hatch_map = {"/": "////", "\\": "\\\\\\\\", "x": "xx", ".": ".."}
    legend_elems = [
        Patch(
            edgecolor="black",
            facecolor=color,
            hatch=legend_hatch_map.get(h, h),
            linewidth=1.2,
            label=lab,
        )
        for (lab, _, h, color) in segs
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        handlelength=2.2,
        handleheight=1.2,
        borderpad=0.3,
        labelspacing=0.6,
    )

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="Path to report.sqlite")
    ap.add_argument("--out", default="", help="Output plot path (e.g. fault_bar.png)")
    ap.add_argument("--dump-json", action="store_true", help="Also print metrics JSON")
    args = ap.parse_args()

    metrics = get_fault_metrics(args.db)
    if not metrics:
        print("{}")
        return

    if args.dump_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))

    plot_stacked_bar(metrics, args.out)


if __name__ == "__main__":
    main()
