"""Short-time spatial and temporal self-convergence study for kr_solver.

Runs one case at several grid sizes N (fixed steps per period) and at several steps per
period (fixed N), from an initial condition that is the same at every N, and compares the
states at chosen strains against the finest run.

Error measure: relative L2 norm of the fluctuation of the state (C, or Psi = log C in log
mode; components xx, xy, yy; the k = 0 mean excluded), computed from the Fourier
coefficients on the coarser grid's dealiased lattice modes. Lattice mode n is the same
physical wavevector at every N (Lbox fixed), so no interpolation is needed.

Observed order is taken from successive differences, p = log2(|u_h - u_h/2| / |u_h/2 - u_h/4|),
which does not suffer from the finest run's own error. For space the ratios are per
refinement step of the N list, so p is only an order if N doubles each time; for a
spectral method the error ratio itself (or the drop per refinement) is the useful number.

Usage (from the folder containing kr_solver.py and sweep_kr.py):
    python convergence_kr.py                          # Charlie's default Rolie-Poly case
    python convergence_kr.py --mode time --workers 4
    python convergence_kr.py --set D=1e-4 --N 64,128,256 --periods 0.5,1,2
Outputs: a printed table, <out>/results.json and <out>/convergence.png.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from kr_solver import KRSolver, Params
from sweep_kr import linear_rate, steady_state

DEFAULTS = dict(
    model="rolie_poly",
    eps_dot=10.0,
    tau_R=0.0067,
    eta_s=1e-3,
    D=1e-3,
    beta=0.045,
    L_max=0.0,
    log_conformation=True,
)


def parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def lowmode_noise(amp, nmax, seed, n_ref=64):
    """Random real noise on lattice modes |n1|, |n2| <= nmax, the same at every N.

    Returns the mode numbers n and coefficients c[comp, i, j] for exp(2 pi i n.xi) with
    the numpy fft convention (unnormalised: field = sum c exp(...) / N^2 * N^2). The peak
    of the xx, xy, yy fields, measured on an n_ref x n_ref grid, is amp, as in
    KRSolver.perturb (whose noise, by contrast, depends on N for a fixed seed).
    """
    rng = np.random.default_rng(seed)
    m = 2 * nmax + 1
    c = rng.standard_normal((3, m, m)) + 1j * rng.standard_normal((3, m, m))
    c = 0.5 * (c + np.conj(c[:, ::-1, ::-1]))  # Hermitian: real field
    c[:, nmax, nmax] = 0  # zero mean
    n = np.arange(-nmax, nmax + 1)
    G = np.zeros((3, n_ref, n_ref), complex)
    G[:, n[:, None] % n_ref, n[None, :] % n_ref] = c
    peak = np.abs(np.fft.ifft2(G).real).max()
    return n, c * amp / peak / n_ref**2  # coefficients of the field itself


def make_solver(p, steps, amp, nmax, seed):
    s = KRSolver(p, steps)
    C0 = steady_state(p)
    if C0 is None:
        raise SystemExit("no homogeneous steady state for these parameters")
    s.set_C(np.broadcast_to(C0[:, None, None], (4, p.N, p.N)).copy())
    n, c = lowmode_noise(amp, nmax, seed)
    N = p.N
    s.Ch[:3][:, n[:, None] % N, n[None, :] % N] += c * N**2
    return s


def check_steps(periods, steps_list):
    """Each evaluation strain must be a whole number of steps for every step count."""
    for S in steps_list:
        for f in periods:
            k = f * S
            if abs(k - round(k)) > 1e-9:
                raise SystemExit(
                    f"{f} periods is {k:g} steps at {S} steps/period: choose periods "
                    "that are whole numbers of steps for every step count"
                )


def run_case(job):
    """One run; returns {periods: spectral state / N^2} plus timing and remap losses."""
    pdict, N, steps, periods, amp, nmax, seed = job
    p = Params(**{**pdict, "N": N})
    s = make_solver(p, steps, amp, nmax, seed)
    t0, k, out = time.time(), 0, {}
    unstable = False
    with np.errstate(all="ignore"):  # a blow-up is detected below, not warned about
        for f in sorted(periods):
            target = int(round(f * steps))
            while k < target:
                s.step()
                k += 1
            if not np.all(np.isfinite(s.Ch)):
                unstable = True
                break
            out[f] = s.Ch / N**2
    return dict(
        unstable=unstable,
        N=N,
        steps=steps,
        states=out,
        wall=time.time() - t0,
        max_remap_loss=max(s.remap_loss) if s.remap_loss else 0.0,
    )


def restrict(Ch, Nc):
    """xx, xy, yy coefficients on the dealiased modes of an Nc grid, from any grid >= Nc."""
    N, cut = Ch.shape[-1], Nc // 3
    n = np.arange(-cut, cut + 1)
    return Ch[:3][:, n[:, None] % N, n[None, :] % N]


def rel_err(a, b):
    """Relative L2 norm of a - b over the fluctuation of b (k = 0 excluded, Parseval)."""
    a, b, c = a.copy(), b.copy(), a.shape[-1] // 2
    a[:, c, c] = b[:, c, c] = 0
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def orders(diffs):
    return [
        float(np.log2(d0 / d1)) if d1 > 0 else np.nan
        for d0, d1 in zip(diffs, diffs[1:])
    ]


def analyse(runs, key, Nc_of):
    """Errors against the finest run and successive differences, per evaluation strain.

    runs: list of run results ordered coarse -> fine; Nc_of(r_coarse, r_fine) gives the
    grid on which to compare two runs."""
    if len(runs) < 2:
        raise SystemExit(
            f"fewer than two stable runs in the {key} study: nothing to compare"
        )
    fine = runs[-1]
    res = {}
    for f in fine["states"]:
        vs_ref = {
            r[key]: rel_err(
                restrict(r["states"][f], Nc_of(r, fine)),
                restrict(fine["states"][f], Nc_of(r, fine)),
            )
            for r in runs[:-1]
        }
        diffs = [
            rel_err(
                restrict(a["states"][f], Nc_of(a, b)),
                restrict(b["states"][f], Nc_of(a, b)),
            )
            for a, b in zip(runs, runs[1:])
        ]
        res[f] = dict(vs_finest=vs_ref, successive=diffs, p=orders(diffs))
    return res


def report(title, key, runs, res, eps_p):
    print(f"\n=== {title} ===")
    for r in runs:
        print(
            f"  {key}={r[key]:5d}  wall {r['wall']:7.1f} s  max remap loss {r['max_remap_loss']:.1e}"
        )
    for f, d in res.items():
        print(f"  {f:g} periods (Hencky strain {f * eps_p:.3f}):")
        print(
            "    vs finest:   "
            + "  ".join(f"{k}: {v:.2e}" for k, v in d["vs_finest"].items())
        )
        pairs = [f"{a[key]}->{b[key]}" for a, b in zip(runs, runs[1:])]
        print(
            "    successive:  "
            + "  ".join(f"{s}: {v:.2e}" for s, v in zip(pairs, d["successive"]))
        )
        print("    observed p:  " + "  ".join(f"{v:.2f}" for v in d["p"]))


def plot(results, eps_p, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(k, v) for k, v in results.items() if v]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4), squeeze=False
    )
    for ax, (name, d) in zip(axes[0], panels):
        for f, r in d["errors"].items():
            x = list(r["vs_finest"])
            ax.loglog(
                x,
                list(r["vs_finest"].values()),
                "o-",
                label=f"strain {float(f) * eps_p:.2f}",
            )
        if (
            name == "time"
        ):  # third-order guide through the first point of the last strain
            x = np.array(x, float)
            y0 = list(r["vs_finest"].values())[0]
            ax.loglog(x, y0 * (x / x[0]) ** -3, "k:", label="slope -3")
            ax.set_xlabel("steps per KR period")
            ax.set_title(f"time, N = {d['N']}")
        else:
            ax.set_xlabel("N")
            ax.set_title(f"space, {d['steps']} steps/period")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(int(v)) for v in x])
        ax.minorticks_off()
        ax.set_ylabel("relative L2 error vs finest run")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nfigure: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="override a Params field (repeatable)",
    )
    ap.add_argument("--mode", choices=("space", "time", "both"), default="both")
    ap.add_argument(
        "--N", default="32,64,128,256", help="grid sizes for the spatial study"
    )
    ap.add_argument(
        "--space-steps",
        type=int,
        default=1000,
        help="steps/period in the spatial study",
    )
    ap.add_argument(
        "--steps",
        default="500,1000,2000,4000",
        help="steps/period for the temporal study",
    )
    ap.add_argument(
        "--time-N", type=int, default=128, help="grid size in the temporal study"
    )
    ap.add_argument(
        "--periods",
        default="0.1,0.5,1",
        help="evaluation strains, in KR periods (whole numbers of steps)",
    )
    ap.add_argument(
        "--amp", type=float, default=0.1, help="peak initial noise amplitude"
    )
    ap.add_argument("--nmax", type=int, default=4, help="highest lattice mode seeded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel")
    ap.add_argument("--out", default="convergence_out")
    a = ap.parse_args(argv)

    pdict = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        pdict[k] = parse_value(v)
    pdict.pop("N", None)
    Ns = sorted(int(x) for x in a.N.split(","))
    steps = sorted(int(x) for x in a.steps.split(","))
    periods = [float(x) for x in a.periods.split(",")]
    if a.nmax > min(Ns + [a.time_N]) // 3:
        raise SystemExit(
            "nmax is not resolved on the coarsest grid (need nmax <= N // 3)"
        )

    jobs = []
    if a.mode in ("space", "both"):
        check_steps(periods, [a.space_steps])
        jobs += [("space", N, a.space_steps) for N in Ns]
    if a.mode in ("time", "both"):
        check_steps(periods, steps)
        jobs += [("time", a.time_N, S) for S in steps]
    unique = sorted(
        {(N, S) for _, N, S in jobs}, key=lambda x: -x[0] * x[0] * x[1]
    )  # longest first
    args = [(pdict, N, S, periods, a.amp, a.nmax, a.seed) for N, S in unique]
    print(f"params: {pdict}\n{len(unique)} runs, {a.workers} worker(s)")
    p0 = Params(**{**pdict, "N": 16})
    rate, t_p = linear_rate(p0, steady_state(p0)), KRSolver(p0, 1).t_p
    print(
        "stability, fastest linear rate x dt (explicit RK3 limit ~2.5): "
        + "  ".join(
            f"{S}: {rate * t_p / S:.2f}" for S in sorted({S for _, _, S in jobs})
        )
    )
    with ProcessPoolExecutor(a.workers) as ex:
        done = {(r["N"], r["steps"]): r for r in ex.map(run_case, args)}

    eps_p = KRSolver(Params(**{**pdict, "N": 16}), 1).lat.eps_p
    for key, r in sorted(done.items()):
        if r["unstable"]:
            print(
                f"N={key[0]}, {key[1]} steps/period went non-finite: left out of the analysis"
            )
    done = {k: r for k, r in done.items() if not r["unstable"]}
    results = {}
    if a.mode in ("space", "both"):
        runs = [done[(N, a.space_steps)] for N in Ns if (N, a.space_steps) in done]
        res = analyse(runs, "N", lambda r, fine: r["N"])
        report(
            f"space ({a.space_steps} steps/period, reference N = {Ns[-1]})",
            "N",
            runs,
            res,
            eps_p,
        )
        results["space"] = dict(
            steps=a.space_steps,
            errors=res,
            remap_loss={r["N"]: r["max_remap_loss"] for r in runs},
        )
    if a.mode in ("time", "both"):
        runs = [done[(a.time_N, S)] for S in steps if (a.time_N, S) in done]
        res = analyse(runs, "steps", lambda r, fine: a.time_N)
        report(
            f"time (N = {a.time_N}, reference {steps[-1]} steps/period)",
            "steps",
            runs,
            res,
            eps_p,
        )
        results["time"] = dict(N=a.time_N, errors=res)

    os.makedirs(a.out, exist_ok=True)
    meta = dict(params=pdict, amp=a.amp, nmax=a.nmax, seed=a.seed, eps_p=eps_p)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(dict(meta=meta, **results), fh, indent=1, default=str)
    plot(results, eps_p, os.path.join(a.out, "convergence.png"))


if __name__ == "__main__":
    main()
