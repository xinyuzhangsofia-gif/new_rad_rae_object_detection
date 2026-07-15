from datetime import datetime
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"(.+?)\s*$"
)


def parse_terminal_epoch_table(text):
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = ROW_PATTERN.match(line)
        if match is None:
            continue
        rows.append(
            {
                "epoch": int(match.group(1)),
                "bev@0.3": float(match.group(2)),
                "bev@0.5": float(match.group(3)),
                "3d@0.3": float(match.group(4)),
                "3d@0.5": float(match.group(5)),
                "p": float(match.group(6)),
                "r": float(match.group(7)),
                "f1": float(match.group(8)),
                "checkpoint": match.group(9),
            }
        )
    return rows


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def infer_run_name(rows):
    if len(rows) == 0:
        return "epoch_metrics"
    checkpoint_path = Path(rows[0]["checkpoint"])
    if checkpoint_path.parent.name:
        return checkpoint_path.parent.name
    return "epoch_metrics"


def default_output_path(rows):
    run_name = sanitize_filename(infer_run_name(rows))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "evaluation_plots" / "png_photos" / f"{timestamp}_sedan_only_epoch_curve_{run_name}.png"


def default_table_output_path(run_name):
    run_name = sanitize_filename(run_name or "epoch_metrics")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "evaluation_plots" / "png_photos" / f"{timestamp}_sedan_only_epoch_table_{run_name}.png"


def prepare_matplotlib_env():
    mpl_config_dir = ROOT / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def history_to_epoch_table_rows(history):
    rows = []
    for row in history:
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "bev@0.3": float(row.get("official_bev_mAP_0.3", 0.0)),
                "bev@0.5": float(row.get("official_bev_mAP_0.5", 0.0)),
                "3d@0.3": float(row.get("official_3d_mAP_0.3", 0.0)),
                "3d@0.5": float(row.get("official_3d_mAP_0.5", 0.0)),
                "p": float(row.get("official_detection_precision", 0.0)),
                "r": float(row.get("official_detection_recall", 0.0)),
                "f1": float(row.get("official_detection_f1", 0.0)),
            }
        )
    return rows


def save_epoch_table_image(history, output_path, run_name):
    prepare_matplotlib_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = history_to_epoch_table_rows(history)
    if len(rows) == 0:
        raise ValueError("No epoch rows available to render.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["epoch", "bev@0.3", "bev@0.5", "3d@0.3", "3d@0.5", "p", "r", "f1"]
    table_text = [
        [
            str(row["epoch"]),
            f"{row['bev@0.3']:.4f}",
            f"{row['bev@0.5']:.4f}",
            f"{row['3d@0.3']:.4f}",
            f"{row['3d@0.5']:.4f}",
            f"{row['p']:.4f}",
            f"{row['r']:.4f}",
            f"{row['f1']:.4f}",
        ]
        for row in rows
    ]

    best_row = max(rows, key=lambda item: item["bev@0.3"])
    final_row = rows[-1]
    row_height = 0.34
    fig_height = max(4.2, 1.7 + (len(rows) * row_height))

    fig, ax = plt.subplots(figsize=(10.8, fig_height), dpi=180)
    ax.axis("off")

    table = ax.table(
        cellText=table_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.22)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        cell.set_linewidth(0.65)
        if row_idx == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f8fafc")

        if row_idx > 0 and rows[row_idx - 1]["epoch"] == best_row["epoch"]:
            cell.set_facecolor("#ecfccb")

    title = (
        f"Sedan-Only Epoch Evaluation Table\n"
        f"Run: {run_name} | Best bev@0.3: epoch {best_row['epoch']} = {best_row['bev@0.3']:.4f} | "
        f"Final f1: {final_row['f1']:.4f}"
    )
    fig.suptitle(title, fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.92, left=0.03, right=0.97, bottom=0.03)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_epoch_metrics(rows, output_path):
    prepare_matplotlib_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in rows]
    best_row = max(rows, key=lambda item: item["bev@0.3"])
    final_row = rows[-1]
    run_name = infer_run_name(rows)

    fig = plt.figure(figsize=(14, 9.0), dpi=180)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.32)
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[1, 0])

    ap_specs = [
        ("bev@0.3", "#2563eb", "-", 2.4),
        ("bev@0.5", "#60a5fa", "--", 1.8),
        ("3d@0.3", "#dc2626", "-", 2.2),
        ("3d@0.5", "#f59e0b", "--", 1.8),
    ]
    pr_specs = [
        ("p", "#7c3aed", "-", 2.1),
        ("r", "#059669", "-", 2.1),
        ("f1", "#ea580c", "-", 2.1),
    ]

    for key, color, linestyle, linewidth in ap_specs:
        values = [row[key] for row in rows]
        ax_ap.plot(
            epochs,
            values,
            marker="o",
            markersize=4.2,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=key,
        )

    for key, color, linestyle, linewidth in pr_specs:
        values = [row[key] for row in rows]
        ax_pr.plot(
            epochs,
            values,
            marker="o",
            markersize=4.0,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=key,
        )

    ax_ap.scatter(
        [best_row["epoch"]],
        [best_row["bev@0.3"]],
        s=90,
        color="#111827",
        zorder=5,
        label=f"best bev@0.3: epoch {best_row['epoch']}",
    )
    ax_ap.annotate(
        f"best {best_row['bev@0.3']:.4f}",
        xy=(best_row["epoch"], best_row["bev@0.3"]),
        xytext=(12, -20),
        textcoords="offset points",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "lw": 0.8},
    )

    for ax in (ax_ap, ax_pr):
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_xlim(min(epochs), max(epochs))
        ax.set_xticks(epochs)
        ax.tick_params(axis="x", labelrotation=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_ap.set_ylabel("AP Value")
    ax_pr.set_ylabel("P / R / F1")
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylim(0.0, max(0.7, max(row[key] for row in rows for key in ("p", "r", "f1")) * 1.08))
    ax_ap.legend(loc="upper left", ncol=5, frameon=False)
    ax_pr.legend(loc="upper left", ncol=3, frameon=False)

    summary_text = (
        f"Run: {run_name}\n"
        f"Best epoch: {best_row['epoch']} | bev@0.3={best_row['bev@0.3']:.4f} | 3d@0.3={best_row['3d@0.3']:.4f}\n"
        f"Final epoch: {final_row['epoch']} | bev@0.3={final_row['bev@0.3']:.4f} | "
        f"3d@0.3={final_row['3d@0.3']:.4f} | p={final_row['p']:.4f} | r={final_row['r']:.4f} | f1={final_row['f1']:.4f}"
    )
    fig.suptitle("Sedan-Only Epoch Metrics", fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.935,
        summary_text,
        ha="center",
        va="top",
        fontsize=10.2,
        color="#374151",
        linespacing=1.45,
    )

    fig.subplots_adjust(top=0.885, left=0.07, right=0.98, bottom=0.07)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python sedan_only/plot_epoch_terminal_table.py <table_txt_path> [output_png_path]"
        )

    table_path = Path(sys.argv[1])
    text = table_path.read_text(encoding="utf-8")
    rows = parse_terminal_epoch_table(text)
    if len(rows) == 0:
        raise ValueError(f"No epoch rows parsed from {table_path}")

    if len(sys.argv) == 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = default_output_path(rows)

    plot_epoch_metrics(rows, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
