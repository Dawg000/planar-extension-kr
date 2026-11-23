"""Per-frame cutoff length and orientation of the structures, for every saved run in a set
of sweep folders (e.g. one folder per seed), cached so reruns only process new runs.

  python summary_data.py kr_match_s0 kr_match_s1 kr_match_s2 z300_N256_s0 z300_N256_s1 \
      [--field txx] [--window 0.5:1] [--workers 4] [--out summary_data.pkl]

For each run (a case with fields/case_NNNN.npz), over the frames in the strain window
(fractions of the run's final strain, so runs stopped early use their own second half):
  per frame   lobe_theta_deg, taylor/cutoff lengths (structure_kr.gradient_lengths) and
              the large-scale ridge angle and lattice-vector angles (ridge_and_lattice)
  pooled      gradient_lengths over all frames of the window (the structure_kr numbers)
  run info    D, N, eps_dot, Lbox, lattice, status, seed, final strain, number of frames,
              and 'growth': rms at the end of the window over rms at its start (about 1
              once the run has saturated; summary_figure.py leaves out runs above 1.5)
The output maps (folder, case) to that record. A run is recomputed only when its field
file, the field or the window changes.
"""

import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np

from fourier_kr import load_frames, parse_window
from kr_solver import kr_lattice
from structure_kr import field_frames, gradient_lengths, physical_acf, ridge_and_lattice
from viz_sweep import case_params, load_sweep


def analyse_run(job):
    folder, row, config, field, window = job
    c = int(row["case"])
    p = case_params(config, row)
    fr = load_frames(folder, c)
    e = fr["strain"]
    sel = np.where((e >= window[0] * e[-1] - 1e-9) & (e <= window[1] * e[-1] + 1e-9))[0]
    f = field_frames(fr, p, field)
    L = p.Lbox
    h = max(0.005, 0.8 / p.N) * L
    frames = []
    for i in sel:
        A = fr["lattice"][i]
        cen, C = physical_acf(f[i : i + 1], A[None], half_width=0.6 * L, h=h)
        frames.append(
            dict(
                strain=float(e[i]),
                **gradient_lengths(f[i : i + 1], A[None]),
                **ridge_and_lattice(cen, C, A[None], L),
            )
        )
    rms = f[sel].std(axis=(1, 2))
    q = max(1, len(rms) // 4)
    return dict(
        folder=folder,
        case=c,
        D=p.D,
        N=p.N,
        eps_dot=p.eps_dot,
        Lbox=L,
        lattice=kr_lattice(p.kr_lattice).name,
        model=p.model,
        status=row["status"],
        seed=config.get("seed"),
        final_strain=float(e[-1]),
        n_frames=len(sel),
        growth=float(rms[-q:].mean() / rms[:q].mean()),
        frames=frames,
        pooled=gradient_lengths(f[sel], fr["lattice"][sel]),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--field", default="txx")
    ap.add_argument("--window", default="0.5:1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="summary_data.pkl")
    args = ap.parse_args(argv)
    window = parse_window(args.window)
    old = pickle.load(open(args.out, "rb")) if os.path.exists(args.out) else {}
    jobs, keep = [], {}
    for folder in args.folders:
        config, rows, _ = load_sweep(folder)
        for c, row in rows.items():
            path = os.path.join(folder, "fields", f"case_{c:04d}.npz")
            if not os.path.exists(path):
                continue
            key = (os.path.normpath(folder), c)
            stamp = (os.path.getmtime(path), os.path.getsize(path), args.field, window)
            if key in old and old[key].get("stamp") == stamp:
                keep[key] = old[key]
            else:
                jobs.append((key, stamp, (folder, row, config, args.field, window)))
    print(f"{len(keep)} runs cached, {len(jobs)} to analyse", flush=True)
    with Pool(args.workers) as pool:
        for (key, stamp, _), r in zip(
            jobs, pool.imap(analyse_run, [j[2] for j in jobs])
        ):
            keep[key] = {**r, "stamp": stamp, "field": args.field}
            print(
                f"  {key[0]} case {key[1]}: D={r['D']:.3g} N={r['N']} strain to "
                f"{r['final_strain']:.2f}, {r['n_frames']} frames, growth {r['growth']:.2f}, "
                f"cutoff {r['pooled']['cutoff_across']:.4f}",
                flush=True,
            )
    pickle.dump(keep, open(args.out, "wb"))
    print(f"wrote {args.out} ({len(keep)} runs)")


if __name__ == "__main__":
    main()
