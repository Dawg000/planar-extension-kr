"""Print the sweep_kr.py command that reruns one case of an existing sweep, with some
parameters changed and everything else (noise, forcing, strain, time step, seed, ...)
copied from the sweep's config.json.

  python rerun_case.py REF_SWEEP_DIR --case 7 --set Lbox=0.5 --set N=128 --out box_L0.5

The new run is a one-case sweep: the reference case's swept values become --set values.
Case numbers follow sweep_kr.py's order (itertools.product over the sweeps, in order).
Checks printed alongside: grid spacing Lbox/N against the reference, and a warning if the
initial noise or the forcing is defined in lattice modes, whose physical wavelength
(Lbox/nmax) changes with Lbox.
"""

import argparse
import itertools
import json
import os
import shlex

CFG_FLAGS = (  # sweep_kr.CONFIG_KEYS except out; flag = --key-with-dashes
    "strain",
    "steps_per_period",
    "dt_frac",
    "stiff_cfl",
    "max_cfl",
    "samples_per_period",
    "amp",
    "nmax",
    "noise_length",
    "seed",
    "sigma_tol",
    "saturation",
    "forcing_length",
    "fields_per_period",
)


def text(v):
    return ("true" if v else "false") if isinstance(v, bool) else repr(v).strip("'")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ref")
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(os.path.join(args.ref, "config.json")) as f:
        cfg = json.load(f)
    combos = list(itertools.product(*cfg["sweeps"].values()))
    if not 0 <= args.case < len(combos):
        raise SystemExit(f"case {args.case} not in this sweep (0-{len(combos) - 1})")
    params = {**cfg["fixed"], **dict(zip(cfg["sweeps"], combos[args.case]))}
    new = dict(params)
    for item in args.set:
        name, _, value = item.partition("=")
        new[name] = value  # sweep_kr.py casts it

    cmd = ["python", "sweep_kr.py", "--model", cfg["model"]]
    cmd += [f"--set={k}={text(v)}" for k, v in new.items()]
    cmd += [f"--{k.replace('_', '-')}={text(cfg[k])}" for k in CFG_FLAGS]
    if cfg["save_fields"]:
        cmd.append("--save-fields")
    cmd.append(f"--out={args.out}")
    print(" ".join(shlex.quote(c) for c in cmd))

    def dx(p):
        return float(p.get("Lbox", 1.0)) / float(p.get("N", 64))

    print(
        f"\n# reference case {args.case}: "
        + ", ".join(f"{k}={text(v)}" for k, v in zip(cfg["sweeps"], combos[args.case]))
    )
    print(f"# grid spacing Lbox/N: reference {dx(params):.4g}, new {dx(new):.4g}")
    L_new = float(new.get("Lbox", 1.0))
    if cfg["noise_length"] == "modes":
        print(
            f"# note: initial noise is in lattice modes |n| <= {cfg['nmax']}: physical "
            f"wavelength >= {L_new / cfg['nmax']:.3g} here (transient only)"
        )
    if (
        float(new.get("forcing_amp", 0))
        and cfg["forcing_length"] == "modes"
        and not float(new.get("forcing_length", 0))
    ):
        nmax = int(new.get("forcing_nmax", 4))
        print(
            f"# WARNING: forcing is in lattice modes |n| <= {nmax}, so its physical "
            f"wavelength (>= {L_new / nmax:.3g}) changes with Lbox. To keep it fixed, "
            f"scale forcing_nmax with Lbox (and decide whether that is the test you want)"
        )


if __name__ == "__main__":
    main()
