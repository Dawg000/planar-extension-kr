"""PRL figure: perturbation size against Hencky strain for several runs.

  python prl_history.py RUN [RUN ...] [--measure velocity] [--relative]
      [--labels "D = 0.01" "D = 0.005" ...] [--xmax E] [--ylim LO:HI]
      [--legend-loc best] [--style markers|lines|both] [--marker-spacing S] [--width col] [--height 2.2]
      [--out path.pdf]

RUN is SWEEP_DIR:CASE, e.g. kr_match_s0:7; runs may come from different sweep folders.
--measure takes one or more of (one panel each, stacked, sharing the strain axis):
    velocity  rms|u - eps_dot (x, -y)|, the deviation from ideal planar extension, in units
              of eps_dot L (L the cell side); from the sweep's record, or reconstructed from
              saved fields for older sweeps
    stress    rms(C - <C>)
    stretch   rms(lambda - <lambda>), from saved fields (snapshot spacing)
These are the histories of viz_sweep.py (growth / movie --history), strain stage by stage
for scheduled runs. --relative divides each curve by its initial value, as in the movie
strips. All lines are black; --style sets how runs are told apart:
    markers  (default) a different open marker per run, solid lines
    lines    a different dash pattern per run (solid, dashed, dotted, dash-dot, ...)
    both     both of these
Markers sit at strains spaced --marker-spacing apart (default: a tenth of the strain range),
staggered from run to run so they do not overlap. Default labels are SWEEP_DIR #CASE. Writes PDF + PNG, by default
prl_history_MEASURES.pdf in the current folder.
"""

import argparse
import os

import numpy as np

import prl_style
from viz_common import INK
from viz_sweep import case_curve, load_sweep, stage_table, to_strain

import matplotlib.pyplot as plt

YLABELS = {
    "velocity": r"$\langle|\mathbf{v}|^2\rangle^{1/2}/(\dot\varepsilon L)$",
    "stress": r"$\langle|\mathbf{C}-\langle\mathbf{C}\rangle|^2\rangle^{1/2}$",
    "stretch": r"$\langle\delta\lambda^2\rangle^{1/2}$",
}
MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]
LINESTYLES = [
    "-",
    (0, (4, 1.5)),
    (0, (1, 1)),
    (0, (4, 1.5, 1, 1.5)),
    (0, (7, 2)),
    (0, (4, 1.5, 1, 1.5, 1, 1.5)),
    (0, (2, 2)),
    (0, (7, 2, 1, 2)),
]


def marker_indices(e, spacing, offset):
    """Indices of the samples nearest the strains offset, offset + spacing, ..."""
    targets = np.arange(offset, e[-1] + 1e-9, spacing)
    return sorted({int(np.argmin(np.abs(e - t))) for t in targets})


def parse_run(text):
    folder, _, case = text.rpartition(":")
    if not folder:
        raise SystemExit(f"run {text!r}: expected SWEEP_DIR:CASE")
    return folder, int(case)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=parse_run, help="SWEEP_DIR:CASE")
    ap.add_argument(
        "--measure", nargs="+", choices=list(YLABELS), default=["velocity"]
    )
    ap.add_argument("--relative", action="store_true", help="divide by initial value")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--xmax", type=float)
    ap.add_argument("--ylim", help="LO:HI, applied to every panel")
    ap.add_argument("--legend-loc", default="best")
    ap.add_argument(
        "--marker-spacing", type=float, help="strain between markers on each line"
    )
    ap.add_argument(
        "--style",
        choices=("markers", "lines", "both"),
        default="markers",
        help="tell runs apart by marker, dash pattern, or both (default markers)",
    )
    ap.add_argument("--width", default="col", help="col, double or inches")
    ap.add_argument("--height", type=float, default=2.2, help="inches per panel")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if len(args.runs) > len(MARKERS):  # as many dash patterns as markers
        raise SystemExit(f"at most {len(MARKERS)} runs")
    labels = args.labels or [f"{os.path.basename(os.path.normpath(f))} #{c}" for f, c in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("give one label per run")
    prl_style.use()

    sweeps = {f: load_sweep(f) for f in {f for f, _ in args.runs}}
    n = len(args.measure)
    W = prl_style.width(args.width)
    fig, axes = plt.subplots(
        n, 1, figsize=(W, args.height * n + 0.1 * (n == 1)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    n_runs = len(args.runs)
    for ax, m in zip(axes, args.measure):
        for i, ((folder, c), lab) in enumerate(zip(args.runs, labels)):
            config, rows, series = sweeps[folder]
            if c not in rows:
                raise SystemExit(f"{folder}: no case {c}")
            curve = case_curve(folder, config, rows[c], series, m)
            if curve is None:
                raise SystemExit(f"{folder} case {c}: no {m} history available")
            t, y = curve[:2]
            e = to_strain(t, stage_table(config, rows[c], series))
            curves = (e, y / y[0] if args.relative else y)
            spacing = args.marker_spacing or (args.xmax or e[-1]) / 10
            marks = dict(
                marker=MARKERS[i], ms=3.5, mfc="white", mec=INK, mew=0.7,
                markevery=marker_indices(e, spacing, spacing * (i + 0.5) / n_runs),
            ) if args.style != "lines" else {}
            ls = LINESTYLES[i] if args.style != "markers" else "-"
            ax.semilogy(*curves, color=INK, lw=0.8, ls=ls, label=lab, **marks)
        ylab = YLABELS[m]
        ax.set_ylabel(ylab + (" (rel. to initial)" if args.relative else ""))
        if args.ylim:
            lo, hi = (float(v) for v in args.ylim.split(":"))
            ax.set_ylim(lo, hi)
        ax.set_xlim(0, args.xmax) if args.xmax else ax.set_xlim(left=0)
    axes[0].legend(loc=args.legend_loc, handlelength=None if args.style == "markers" else 2.6)
    axes[-1].set_xlabel(r"Hencky strain $\varepsilon = \dot\varepsilon t$")
    fig.tight_layout(pad=0.3, h_pad=0.3)
    out = args.out or f"prl_history_{'-'.join(args.measure)}.pdf"
    prl_style.save(fig, out)


if __name__ == "__main__":
    main()
