# coding=utf-8
"""Plot a run's loss_history.csv as a train/validation curve.

The point of the figure is the gap: training loss falling while validation loss
rises is overfitting, and it is much easier to see than to argue about. A run
that was cut short and a run that diverged look completely different here, so
the plot settles that question directly.

Training calls this automatically; run it by hand to re-plot an old run.

Usage:
    python3 finetuning/plot_loss_history.py output/loss_history.csv
    python3 finetuning/plot_loss_history.py output/loss_history.csv --out figure.png
"""
import argparse
import csv
import os

# Categorical slots 1 and 2 of the reference palette. Validated for colour-vision
# deficiency (worst adjacent pair dE 24.7 protan), but the line styles below carry
# the same distinction so the figure also survives greyscale printing.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#c3c2b7"
SURFACE = "#fcfcfb"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="loss_history.csv written by the training script")
    parser.add_argument("--out", default=None, help="Output PNG; defaults to alongside the CSV")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--mark_epoch", type=int, default=0,
        help="Draw a vertical marker here, e.g. the epoch the adapters unfroze",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with open(csv_path, encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["record_type"] == "epoch"]
    if not rows:
        raise SystemExit(f"No epoch rows in {csv_path}")
    return rows


def column(rows, name):
    """Return (epochs, values) for the rows where this column is populated."""
    epochs, values = [], []
    for row in rows:
        raw = row.get(name, "")
        if raw not in (None, ""):
            epochs.append(int(row["epoch"]))
            values.append(float(raw))
    return epochs, values


def plot(csv_path, out_path, title, mark_epoch=0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("This script needs matplotlib: pip install matplotlib")

    rows = read_rows(csv_path)
    epochs, train = column(rows, "train_main_loss")
    _, val = column(rows, "val_main_loss")
    delta_epochs, delta = column(rows, "val_emotion_delta")
    shuffle_epochs, shuffle = column(rows, "val_emotion_shuffle_delta")
    has_delta = bool(delta) or bool(shuffle)

    height = 7.2 if has_delta else 4.2
    figure, axes = plt.subplots(
        2 if has_delta else 1, 1, figsize=(9, height), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]} if has_delta else None,
    )
    panels = axes if has_delta else [axes]
    figure.patch.set_facecolor(SURFACE)

    top = panels[0]
    top.set_facecolor(SURFACE)
    # Final values ride in the legend and the gap rides in the title, both of
    # which sit outside the data area, so no annotation can land on a curve.
    top.plot(epochs, train, color=TRAIN_COLOR, linewidth=2,
             label=f"train  (final {train[-1]:.4f})")
    top.plot(epochs, val, color=VAL_COLOR, linewidth=2, linestyle="--",
             label=f"validation  (final {val[-1]:.4f})")

    best_index = min(range(len(val)), key=lambda i: val[i])
    best_epoch, best_value = epochs[best_index], val[best_index]
    top.axvline(best_epoch, color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)
    top.annotate(
        f"best val  ep{best_epoch}  {best_value:.4f}",
        xy=(best_epoch, best_value), xytext=(6, 10), textcoords="offset points",
        color=INK_MUTED, fontsize=9,
    )
    top.set_ylabel("main loss (CE)", color=INK)
    heading = title or os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    top.set_title(f"{heading}   |   final gap {val[-1] - train[-1]:+.4f}", color=INK)
    top.legend(frameon=False, labelcolor=INK, loc="lower left")

    if has_delta:
        bottom = panels[1]
        bottom.set_facecolor(SURFACE)
        if delta:
            bottom.plot(delta_epochs, delta, color=TRAIN_COLOR, linewidth=2, label="zeroed")
        if shuffle:
            bottom.plot(
                shuffle_epochs, shuffle, color=VAL_COLOR, linewidth=2, linestyle="--",
                label="shuffled",
            )
        bottom.axhline(0, color=GRID, linewidth=1, zorder=0)
        bottom.set_ylabel("emotion delta", color=INK)
        bottom.legend(frameon=False, labelcolor=INK)

    if mark_epoch:
        for panel in panels:
            panel.axvline(mark_epoch, color=INK_MUTED, linewidth=1.2, zorder=0)
        panels[0].annotate(
            f"adapters unfrozen (ep{mark_epoch})",
            xy=(mark_epoch, 0.97), xycoords=("data", "axes fraction"),
            xytext=(5, 0), textcoords="offset points",
            ha="left", va="top", color=INK_MUTED, fontsize=9,
        )

    for panel in panels:
        panel.grid(color=GRID, linewidth=0.6, alpha=0.5)
        panel.set_axisbelow(True)
        panel.tick_params(colors=INK_MUTED)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color(GRID)
    panels[-1].set_xlabel("epoch", color=INK)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def main():
    args = parse_args()
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv_path)), "loss_curve.png"
    )
    plot(args.csv_path, out_path, args.title, mark_epoch=args.mark_epoch)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
