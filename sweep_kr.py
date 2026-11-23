"""Parameter sweeps for the KR planar-extension solver, to search for instabilities.

For each parameter set the script:
  1. finds the homogeneous steady state of the 0D model in planar extension
     (cases with no steady state, e.g. Oldroyd-B at Wi >= 0.5, are skipped),
  2. starts the 2D solver from that uniform state plus small random noise,
  3. records the rms fluctuation of C (xx, xy, yy) over time,
  4. fits an exponential growth rate sigma over the second half of the record
     (or up to saturation / decay to roundoff, if that comes first).

Time is in the same units as the Params time scales (lam, tau_R), so
sigma * lam is the dimensionless growth rate and Wi = eps_dot * lam.

Example:
  python sweep_kr.py --model giesekus \\
      --sweep eps_dot=0.5:5:7:log --sweep alpha=0.1,0.3,0.5 \\
      --set eta_s=0.05 --strain 12 --workers 4 --out giesekus_sweep

Initial noise (--noise-length): white noise in lattice modes |n| <= nmax (default), or
noise smoothed by a Gaussian of width sqrt(D/eps_dot) or sqrt(D*lam).

Stochastic forcing: --set forcing_amp=A adds A*sqrt(dt)*xi every step (xi: unit-rms
Gaussian noise, zero cell average), independent of the initial noise; --forcing-length
picks its spatial structure. With forcing on, a stable case settles at a noise floor
instead of decaying, so its fitted sigma is ~0 and fluctuates: raise --sigma-tol.

Sweep specs:  name=a,b,c (list)   name=lo:hi:n (linspace)   name=lo:hi:n:log (geomspace)
Any Params field can be swept or set. Sweep defaults (override with --set):
log_conformation=true, N=64, D=1e-4. The initial noise amplitude can also be swept or
set this way (--sweep amp=1e-6:1e-2:5:log); it then overrides --amp and is a column of
results.csv.

Schedules: --schedule NAME=V0,V1,... --remaps-per-stage K changes a Params field during
each run: stage 0 uses V0 for K KR periods, then the field switches to V1 (right after a
remap) and the run carries on from the current state for K more periods, and so on. The
run length is (number of stages) x K periods; --strain is not used. Several --schedule
options change their fields together (same number of values each). The value specs are
the same as for --sweep, so tau_R=1e-4:1e-2:3:log works. Every case of the sweep follows
the schedule. Stage 0's values appear in results.csv as fixed values. The model, N, Lbox,
kr_lattice, log_conformation and amp cannot be scheduled.
At each switch the steps per period are re-planned for the new values (the same rules as
above, at both the current cell average and the new homogeneous steady state, if there is
one), and the saturation and decay tests are re-referenced to that steady state (or to
the current average if there is none). Decay to roundoff does not end a scheduled
run (a later stage can still amplify what is left): every stage runs in full, and the
status is stable_decayed if the run ends at roundoff. A stage that starts at roundoff
fits its growth rate to roundoff noise, so read sigma_lam_sN together with the series. status, sigma and sigma_lam describe the last
stage reached (fit over its second half); stage_reached, sigma_lam_sN and
steps_per_period_sN give the stage reached and each stage's own values. The series files
gain 'stage' (stage of each sample) and 't_switch' (start time of each stage).

Time step: steps per KR period = max(--steps-per-period, period / (--dt-frac * shortest
relaxation time), period * rate / --stiff-cfl), where rate is the largest linear rate of
the base state (relaxation, retraction, and the solvent mode ~ G|tau|/eta_s, which is the
stiff one at small eta_s). During a run, a case stops with status cfl_exceeded if the
advective number max|v| k_max dt passes --max-cfl, rather than blowing up.

Safety for long sweeps:
  --dry-run   plan every case (steps, cost, disk, warnings) without running anything
  --order     run cases in index order (default), shortest estimated run time first
              (results arrive soonest), or longest first (the whole sweep finishes soonest)
  --resume    continue an interrupted sweep in the same --out: finished cases are skipped,
              cases that ended in an error are rerun. Without --resume an existing
              results.csv is never overwritten.
A case that raises an error is recorded as status "error: ..." (traceback in errors/)
and the sweep continues.

Run-time estimates (--dry-run, --order): unless --ms-per-step is given, the cost of a time
step is measured first, for each grid size of the sweep, with as many cases running at
once as the sweep will run (--workers, or fewer if there are fewer cases): each case slows
down as more share the machine, so a fixed per-step cost underestimates busy sweeps.

Outputs in --out: results.csv (one row per case, written as cases finish),
series/case_NNNN.npz (time series per case, written as cases finish), config.json (the
full sweep definition), and with --save-fields, fields/case_NNNN.npz (snapshots of the
stress fluctuation, tau/G - <tau/G>, for viz_sweep.py). The dtxx_* columns give
max|delta tau_xx| / G at the end of the run and its peak over the run.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process

import argparse
import csv
import dataclasses
import heapq
import itertools
import json
import math
import time
import traceback
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

from kr_solver import (
    MODELS,
    XX,
    YY,
    ZZ,
    KRSolver,
    Params,
    StretchLimitError,
    effective_gradient,
    fene_factor,
    homogeneous_rhs,
    kr_lattice,
    model_source,
    stokes_velocity_hat,
    stress_conformation,
    upper_convected,
)

FIELDS = {f.name: f.type for f in dataclasses.fields(Params) if f.name != "model"}
CASE_OPTIONS = {"amp": float}  # per-case run options that are not Params fields
SWEEP_DEFAULTS = dict(log_conformation=True, N=64, D=1e-4)
RESULT_COLUMNS = [
    "status",
    "sigma",
    "sigma_lam",
    "sigma_v_lam",
    "N1_base",
    "rms_initial",
    "rms_final",
    "dtxx_final",
    "dtxx_peak",
    "max_remap_loss",
    "diff_len_dx",
    "steps_per_period",
    "stiff_rate_dt",
    "max_cfl",
    "runtime_s",
]
CONFIG_KEYS = (
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
    "save_fields",
    "forcing_length",
    "fields_per_period",
    "out",
    "schedule",
    "remaps_per_stage",
)
UNSCHEDULABLE = ("model", "N", "Lbox", "kr_lattice", "log_conformation", "amp")


# --- command-line parsing -------------------------------------------------------
def cast(name, text):
    types = {**FIELDS, **CASE_OPTIONS}
    if name not in types:
        raise SystemExit(f"unknown parameter {name!r}; choose from {sorted(types)}")
    t = types[name]
    if t is bool:
        if str(text).lower() in ("1", "true", "yes", "on"):
            return True
        if str(text).lower() in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"{name} must be true or false, got {text!r}")
    return int(float(text)) if t is int else t(text)


def parse_sweep(spec):
    name, _, values = spec.partition("=")
    if ":" in values:
        parts = values.split(":")
        lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
        grid = (
            np.geomspace(lo, hi, n) if parts[3:] == ["log"] else np.linspace(lo, hi, n)
        )
        return name, [cast(name, v) for v in grid]
    return name, [cast(name, v) for v in values.split(",")]


# --- physics ------------------------------------------------------------------------
def steady_state(p):
    """Homogeneous steady state of the 0D model in planar extension, or None."""
    rhs = lambda t, c: homogeneous_rhs(c, p)
    blowup = lambda t, c: c[XX] - 1e8
    blowup.terminal = True
    events = [blowup]
    if p.L_max:  # a finite-extensibility steady state never reaches L_max
        limit = lambda t, c: (c[XX] + c[YY] + c[ZZ]) / 3 - p.L_max**2 * (1 - 1e-9)
        limit.terminal = True
        events.append(limit)
    C = np.array([1.0, 0.0, 1.0, 1.0])
    for t_end in (200 * p.lam, 2000 * p.lam, 20000 * p.lam):
        sol = solve_ivp(
            rhs, (0, t_end), C, method="LSODA", rtol=1e-10, atol=1e-12, events=events
        )
        if sol.status == 1 or not np.all(np.isfinite(sol.y[:, -1])):
            return None  # unbounded growth
        C = sol.y[:, -1]
        if np.linalg.norm(rhs(0, C)) * p.lam < 1e-9 * (1 + np.linalg.norm(C)):
            return C
    return None


def linear_rate(p, C0, n_dir=36):
    """Largest |eigenvalue| of the linearized dynamics of one Fourier mode about the uniform
    state C0, over wavevector directions. At Re = 0 the Stokes coupling does not depend on
    |k|, so this covers relaxation, retraction (with finite extensibility), extension, and
    the solvent mode ~ G|tau|/eta_s. Stress diffusion is integrated exactly and is excluded.
    """
    L = (p.eps_dot, 0.0, 0.0, -p.eps_dot)
    h = 1e-7 * max(1.0, np.abs(C0).max())
    E = np.eye(4)
    Js, Jt = np.empty((4, 4)), np.empty((4, 4))
    for j in range(4):
        Js[:, j] = (
            model_source(C0 + h * E[j], L, p) - model_source(C0 - h * E[j], L, p)
        ) / (2 * h)
        Jt[:, j] = (
            stress_conformation(C0 + h * E[j], p)
            - stress_conformation(C0 - h * E[j], p)
        ) / (2 * h)
    worst = 0.0
    for th in np.linspace(0, np.pi, n_dir, endpoint=False):
        kx, ky = np.cos(th), np.sin(th)
        KX, KY = np.array([[0.0, kx]]), np.array([[0.0, ky]])
        J = Js.astype(complex)
        for j in range(4):
            Ch = np.zeros((4, 1, 2), complex)
            Ch[:, 0, 1] = Jt[:, j]
            vx, vy = (v[0, 1] for v in stokes_velocity_hat(Ch, KX, KY, p.G, p.eta_s))
            dk = (1j * kx * vx, 1j * ky * vx, 1j * kx * vy, 1j * ky * vy)
            J[:, j] += upper_convected(C0, effective_gradient(dk, p))
        worst = max(worst, float(np.abs(np.linalg.eigvals(J)).max()))
    return worst


def advective_cfl(s):
    """max|v| * k_max * dt for the current state (RK3 is unstable above about 1.7)."""
    S = np.fft.ifft2(s.Ch).real
    kx, ky = s.wavevectors(s.tau)
    vxh, vyh = s.velocity_hat(s.stress_hat(S, s.Ch), kx, ky)
    vmax = np.sqrt(np.fft.ifft2(vxh).real ** 2 + np.fft.ifft2(vyh).real ** 2).max()
    return float(vmax * np.sqrt(kx**2 + ky**2)[s.keep].max() * s.dt)


def case_params(pdict):
    """Params of a case, leaving out the per-case run options (CASE_OPTIONS)."""
    return Params(**{k: v for k, v in pdict.items() if k not in CASE_OPTIONS})


def stages(cfg):
    """Per-stage parameter overrides: [{}] without a schedule."""
    sched = cfg["schedule"]
    if not sched:
        return [{}]
    n = len(next(iter(sched.values())))
    return [{k: v[i] for k, v in sched.items()} for i in range(n)]


def periods_per_stage(cfg, eps_p):
    """KR periods per stage: --remaps-per-stage with a schedule, else --strain."""
    if cfg["schedule"]:
        return cfg["remaps_per_stage"]
    return math.ceil(cfg["strain"] / eps_p)


def stage_params(pdict, override, cfg):
    """Params of a case with a stage's overrides applied."""
    p = case_params({**pdict, **override})
    if cfg["forcing_length"] != "modes":
        p = dataclasses.replace(
            p, forcing_length=noise_length(p, cfg["forcing_length"])
        )
    return p


def plan_steps(p, C, cfg):
    """(steps per KR period, largest linear rate * dt) for parameters p about state C."""
    # Rolie-Poly retraction rate ~ f / tau_R (f = 1 without finite extensibility)
    timescale = (
        min(p.lam, p.tau_R / fene_factor(C, p)) if p.model == "rolie_poly" else p.lam
    )
    t_p = kr_lattice(p.kr_lattice).eps_p / p.eps_dot
    rate = linear_rate(p, C)
    steps = max(
        cfg["steps_per_period"],
        math.ceil(t_p / (cfg["dt_frac"] * timescale)),
        math.ceil(t_p * rate / cfg["stiff_cfl"]),
    )
    return steps, rate * t_p / steps


def plan_case(pdict, cfg):
    """Params, base state C0 (None if there is no steady state) and the time-step plan.
    With a schedule, later stages are planned at their own steady state (or, if they
    have none, at the previous stage's); the run re-plans them from the actual state."""
    p = stage_params(pdict, {}, cfg)
    C0 = steady_state(p)
    if C0 is None:
        return p, None, None
    steps, stiff = plan_steps(p, C0, cfg)
    stage_steps, C = [steps], C0
    for override in stages(cfg)[1:]:
        pi = stage_params(pdict, override, cfg)
        Ci = steady_state(pi)
        C = C if Ci is None else Ci
        stage_steps.append(plan_steps(pi, C, cfg)[0])
    periods = periods_per_stage(cfg, kr_lattice(p.kr_lattice).eps_p)
    return (
        p,
        C0,
        {
            "steps": steps,
            "stiff_rate_dt": stiff,
            "stage_steps": stage_steps,
            "n_steps": periods * sum(stage_steps),
        },
    )


def switch_stage(s, pdict, override, cfg):
    """Apply a stage's overrides to the running solver, re-planning the steps per period
    at the current cell average and at the new steady state. Returns (Params, reference
    state for the saturation / decay tests, steps per period)."""
    p = stage_params(pdict, override, cfg)
    C_ss = steady_state(p)
    C_now = s.mean_C()
    refs = [C_now] if C_ss is None else [C_now, C_ss]
    steps = max(plan_steps(p, C, cfg)[0] for C in refs)
    s.set_params(p, steps)
    return p, (C_now if C_ss is None else C_ss), steps


def diffusion_length_dx(p):
    """sqrt(D / eps_dot) in grid spacings: the width that stretched filaments thin to."""
    return math.sqrt(p.D / p.eps_dot) * p.N / p.Lbox


def noise_length(p, kind):
    """Smoothing length of the initial noise, or None for low-mode white noise."""
    if kind == "modes":
        return None
    return math.sqrt(p.D / p.eps_dot) if kind == "extension" else math.sqrt(p.D * p.lam)


def fluctuation_rms(s):
    C = s.C()[:3]
    return float(np.sqrt(np.mean((C - C.mean(axis=(1, 2), keepdims=True)) ** 2)))


def velocity_rms(dC, A, G, eta_s, eps_dot):
    """rms of the perturbation velocity v = u - eps_dot (x, -y), in units of eps_dot * Lbox,
    from a snapshot of C - <C> (components xx, xy, yy) on the lattice A (columns = lattice
    vectors). Exact: the Stokes velocity depends linearly on the stress fluctuation."""
    N = dC.shape[-1]
    n = np.fft.fftfreq(N, 1.0 / N)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    B = 2 * np.pi * np.linalg.inv(A).T  # k = 2 pi A^-T n
    kx, ky = B[0, 0] * n1 + B[0, 1] * n2, B[1, 0] * n1 + B[1, 1] * n2
    vx, vy = stokes_velocity_hat(np.fft.fft2(dC), kx, ky, G, eta_s)
    rms = np.sqrt(np.sum(np.abs(vx) ** 2 + np.abs(vy) ** 2)) / N**2
    return float(rms / (eps_dot * math.sqrt(abs(np.linalg.det(A)))))


def fit_window(t, r):
    """Samples used for the growth-rate fit: the second half of the record."""
    t, r = np.asarray(t), np.asarray(r)
    sel = (t >= t[-1] / 2) & (r > 0)
    return sel if sel.sum() >= 3 else r > 0


def fit_growth(t, r):
    """Least-squares exponential growth rate over fit_window."""
    t, r = np.asarray(t), np.asarray(r)
    sel = fit_window(t, r)
    if sel.sum() < 2:
        return float("nan")
    return float(np.polyfit(t[sel], np.log(r[sel]), 1)[0])


def snapshot(s):
    """Fluctuation of tau/G (xx, xy, yy, zz; equal to C - <C> without finite
    extensibility) about its cell average, plus the lattice, the time and the cell
    average itself (so full fields such as the chain stretch can be rebuilt)."""
    C = stress_conformation(s.C(), s.p)
    mean = C.mean(axis=(1, 2))
    return (C - mean[:, None, None]).astype(np.float32), s.lattice(), s.t, mean


def save_frames(path, frames, p):
    dC, A, t, mean = zip(*frames)
    np.savez_compressed(
        path,
        dC=np.array(dC),
        lattice=np.array(A),
        t=np.array(t),
        mean=np.array(mean),
        eps_dot=p.eps_dot,
        G=p.G,
    )


def run_case_safe(job):
    """run_case, but an error becomes a result row (and a traceback file), not a crash."""
    try:
        return run_case(job)
    except Exception as e:
        idx, pdict, cfg = job
        try:
            os.makedirs(os.path.join(cfg["out"], "errors"), exist_ok=True)
            with open(
                os.path.join(cfg["out"], "errors", f"case_{idx:04d}.txt"), "w"
            ) as f:
                f.write(traceback.format_exc())
        except OSError:
            pass
        msg = f"error: {type(e).__name__}: {e}".replace("\n", " ")[:300]
        return {"case": idx, **pdict, "status": msg}, None


def run_case(job):
    idx, pdict, cfg = job
    start = time.time()
    row = {"case": idx, **pdict}
    p, C0, plan = plan_case(pdict, cfg)
    if C0 is None:
        row.update(status="no_steady_state")
        return row, None
    S0 = stress_conformation(C0, p)
    row["N1_base"] = p.G * (S0[XX] - S0[YY])
    steps = plan["steps"]
    s = KRSolver(p, steps)
    s.set_C(np.broadcast_to(C0[:, None, None], (4, p.N, p.N)).copy())
    s.perturb(
        pdict.get("amp", cfg["amp"]),
        nmax=cfg["nmax"],
        seed=cfg["seed"],
        smooth_length=noise_length(p, cfg["noise_length"]),
    )

    periods = periods_per_stage(cfg, s.lat.eps_p)

    def sample():
        snap = snapshot(s)[0]
        ts.append(s.t)
        rs.append(fluctuation_rms(s))
        vs.append(velocity_rms(snap, s.lattice(), p.G, p.eta_s, p.eps_dot))
        txx.append(float(np.abs(snap[0]).max()))  # max |delta tau_xx| / G
        stage_of.append(i_stage)
        return rs[-1]

    ts, rs, vs, txx, stage_of = [], [], [], [], []
    i_stage = 0
    sample()
    max_cfl = advective_cfl(s)
    frames = [snapshot(s)] if cfg["save_fields"] else None
    status = None
    stage_info = []  # (start time, steps per period, lam) of each stage reached
    for i_stage, override in enumerate(stages(cfg)):
        if i_stage:
            p, C0, steps = switch_stage(s, pdict, override, cfg)
        stage_info.append((s.t, steps, p.lam))
        scale = np.linalg.norm(C0[:3])
        record_every = max(1, steps // cfg["samples_per_period"])
        fpp = min(cfg["fields_per_period"], steps)
        n_steps = periods * steps
        for k in range(1, n_steps + 1):
            try:
                s.step()
                # fpp snapshots per period, evenly spaced and aligned with period boundaries
                if frames is not None and (k * fpp) // steps > ((k - 1) * fpp) // steps:
                    frames.append(snapshot(s))
                if k % record_every and k != n_steps:
                    continue
                r = sample()
                if not np.isfinite(r):
                    status = "blew_up"
                    break
                max_cfl = max(max_cfl, advective_cfl(s))
            except StretchLimitError:
                status = "exceeded_L_max"
                break
            if max_cfl > cfg["max_cfl"]:
                status = "cfl_exceeded"
                break
            if r > cfg["saturation"] * scale:
                status = "unstable_saturated"
                break
            # with a schedule every stage runs in full: a later stage can still amplify
            # what is left, and the decay test is applied once at the end
            if r < 1e-12 * scale and not cfg["schedule"]:
                status = "stable_decayed"
                break
        if status is not None:
            break

    if (
        frames is not None
        and status in ("unstable_saturated", "stable_decayed", "cfl_exceeded")
        and frames[-1][2] != s.t
    ):
        frames.append(snapshot(s))  # last state before an early stop
    if frames is not None:
        save_frames(
            os.path.join(cfg["out"], "fields", f"case_{idx:04d}.npz"), frames, p
        )
    good = np.isfinite(rs)
    ts, rs, vs, txx, stage_of = (np.array(a)[good] for a in (ts, rs, vs, txx, stage_of))

    def stage_fit(y, i):
        """Growth rate of y over the second half of stage i (time from its start)."""
        sel = stage_of == i
        return fit_growth(ts[sel] - stage_info[i][0], y[sel])

    last = int(stage_of[-1])
    sigma = stage_fit(rs, last)
    if status is None and cfg["schedule"] and rs[-1] < 1e-12 * scale:
        status = "stable_decayed"
    if status is None:
        status = "unstable" if sigma * p.lam > cfg["sigma_tol"] else "stable"
    # Remap loss only signals under-resolution while a fluctuation persists; once it has
    # decayed to roundoff the lost fraction is measured on noise.
    loss = max(s.remap_loss, default=0.0)
    if loss > 1e-6 and status != "stable_decayed":
        status += "_underresolved"
    row.update(
        status=status,
        sigma=sigma,
        sigma_lam=sigma * p.lam,
        sigma_v_lam=stage_fit(vs, last) * p.lam,
        rms_initial=rs[0],
        rms_final=rs[-1],
        dtxx_final=txx[-1],
        dtxx_peak=txx.max(),
        max_remap_loss=loss,
        diff_len_dx=diffusion_length_dx(p),
        steps_per_period=stage_info[0][1],
        stiff_rate_dt=plan["stiff_rate_dt"],
        max_cfl=max_cfl,
        runtime_s=round(time.time() - start, 1),
    )
    out = {"t": ts, "rms": rs, "vrms": vs, "dtxx": txx}
    if cfg["schedule"]:
        row["stage_reached"] = last
        for i, (_, steps_i, lam_i) in enumerate(stage_info):
            row[f"sigma_lam_s{i}"] = stage_fit(rs, i) * lam_i
            row[f"steps_per_period_s{i}"] = steps_i
        out.update(stage=stage_of, t_switch=np.array([t for t, _, _ in stage_info]))
    return row, out


# --- driver -------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--sweep", action="append", default=[], metavar="NAME=SPEC")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument(
        "--strain", type=float, default=10.0, help="total Hencky strain per case"
    )
    ap.add_argument(
        "--steps-per-period", type=int, default=200, help="minimum steps per KR period"
    )
    ap.add_argument(
        "--dt-frac",
        type=float,
        default=0.1,
        help="max time step as a fraction of the shortest relaxation time",
    )
    ap.add_argument("--samples-per-period", type=int, default=20)
    ap.add_argument("--amp", type=float, default=1e-4, help="initial noise amplitude")
    ap.add_argument(
        "--nmax",
        type=int,
        default=4,
        help="highest lattice mode seeded with noise (--noise-length modes only)",
    )
    ap.add_argument(
        "--noise-length",
        choices=("modes", "extension", "relaxation"),
        default="modes",
        help="initial noise: white in lattice modes |n| <= nmax (modes), or smoothed "
        "by a Gaussian of width sqrt(D/eps_dot) (extension) or sqrt(D*lam) "
        "(relaxation; lam = tau_d for Rolie-Poly)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sigma-tol",
        type=float,
        default=0.01,
        help="sigma * lam above which a case counts as unstable",
    )
    ap.add_argument(
        "--saturation",
        type=float,
        default=0.1,
        help="stop when rms fluctuation exceeds this fraction of |C_base|",
    )
    ap.add_argument(
        "--forcing-length",
        choices=("modes", "extension", "relaxation"),
        default="modes",
        help="spatial structure of the stochastic forcing (turned on with "
        "--set forcing_amp=A): lattice modes |n| <= forcing_nmax, or smoothed "
        "by sqrt(D/eps_dot) or sqrt(D*lam). An explicit --set forcing_length=L "
        "is used as given with the default 'modes'",
    )
    ap.add_argument(
        "--save-fields",
        action="store_true",
        help="save field snapshots per case to OUT/fields/ (for viz_sweep.py movie)",
    )
    ap.add_argument("--fields-per-period", type=int, default=10)
    ap.add_argument(
        "--stiff-cfl",
        type=float,
        default=1.0,
        help="max (largest linear rate of the base state) * dt; RK3 is unstable "
        "above about 2.5",
    )
    ap.add_argument(
        "--max-cfl",
        type=float,
        default=1.0,
        help="stop a case (status cfl_exceeded) when max|v| k_max dt exceeds this",
    )
    ap.add_argument(
        "--schedule",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="change a Params field during each run, one value per stage (same specs "
        "as --sweep); repeat to change several fields together",
    )
    ap.add_argument(
        "--remaps-per-stage",
        type=int,
        default=None,
        help="KR periods (remaps) per stage of a --schedule",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted sweep in --out (same command otherwise)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="plan all cases and print cost, disk and warnings; run nothing",
    )
    ap.add_argument(
        "--order",
        choices=ORDERS,
        default="index",
        help="order cases by index, or by estimated run time: shortest first "
        "(early results) or longest first (shortest total wall time)",
    )
    ap.add_argument(
        "--ms-per-step",
        type=float,
        default=None,
        help="cost per step at N = 128 for the run-time estimates (scaled by N^2); "
        "default: measure it with a short benchmark of each grid size, run with as "
        "many cases at once as the sweep will run",
    )
    args = ap.parse_args(argv)

    fixed = dict(SWEEP_DEFAULTS)
    for item in args.set:
        name, _, value = item.partition("=")
        fixed[name] = cast(name, value)
    sweeps = dict(parse_sweep(spec) for spec in args.sweep)
    schedule = dict(parse_sweep(spec) for spec in args.schedule)
    set_names = {item.partition("=")[0] for item in args.set}
    for name, values in schedule.items():
        if name in UNSCHEDULABLE:
            raise SystemExit(f"--schedule: {name} cannot change during a run")
        if name in sweeps or name in set_names:
            raise SystemExit(
                f"--schedule: {name} is also given with --sweep or --set; "
                "the schedule sets it for every stage"
            )
        if len(values) < 2:
            raise SystemExit(f"--schedule: {name} needs at least two values")
    if len({len(v) for v in schedule.values()}) > 1:
        raise SystemExit(
            "--schedule: every scheduled field needs the same number of values"
        )
    if schedule and not (args.remaps_per_stage and args.remaps_per_stage > 0):
        raise SystemExit("--schedule needs --remaps-per-stage K (K >= 1)")
    if args.remaps_per_stage is not None and not schedule:
        raise SystemExit("--remaps-per-stage is only used with --schedule")
    for name, values in schedule.items():
        fixed[name] = values[0]  # stage 0
    args.schedule = schedule or None  # None keeps config.json unchanged without one
    if schedule:
        n = len(next(iter(schedule.values())))
        print(
            f"schedule: {n} stages of {args.remaps_per_stage} KR periods "
            f"({', '.join(schedule)}); --strain is not used"
        )
    if "eps_dot" not in sweeps and "eps_dot" not in fixed:
        raise SystemExit("set or sweep eps_dot (Wi = eps_dot * lam)")

    names = list(sweeps)
    cases = [
        dict(fixed, model=args.model, **dict(zip(names, combo)))
        for combo in itertools.product(*sweeps.values())
    ]
    cfg = {k: getattr(args, k) for k in CONFIG_KEYS}
    if "modes" != args.noise_length or "modes" != args.forcing_length:
        if any(c["D"] <= 0 for c in cases):
            raise SystemExit("extension/relaxation noise or forcing lengths need D > 0")
    if args.forcing_length != "modes" and not any(
        c.get("forcing_amp", 0) for c in cases
    ):
        print(
            "note: --forcing-length has no effect without --set forcing_amp=A (A > 0)"
        )
    print(f"{len(cases)} cases -> {args.out}/results.csv")
    thin = [c for c in cases if diffusion_length_dx(case_params(c)) < 2]
    if thin:
        print(
            f"note: {len(thin)} cases have sqrt(D/eps_dot) < 2 grid spacings; filaments "
            "will thin below the grid (raise N or D if growth rates near zero matter)"
        )
    jobs = [(i, c, cfg) for i, c in enumerate(cases)]
    if args.dry_run:
        return dry_run(jobs, cfg, args)

    config = json.loads(
        json.dumps({"model": args.model, "fixed": fixed, "sweeps": sweeps, **cfg})
    )
    columns = [
        "case",
        "model",
        *names,
        *[k for k in fixed if k not in names],
        *RESULT_COLUMNS,
    ]
    if schedule:
        n = len(next(iter(schedule.values())))
        columns += [
            "stage_reached",
            *[f"sigma_lam_s{i}" for i in range(n)],
            *[f"steps_per_period_s{i}" for i in range(n)],
        ]
    results_path = os.path.join(args.out, "results.csv")
    config_path = os.path.join(args.out, "config.json")
    previous = []
    if args.resume:
        if not os.path.exists(results_path):
            raise SystemExit(f"--resume: {results_path} not found")
        with open(config_path) as f:
            old = json.load(f)
        diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
        if diff:
            raise SystemExit(
                f"--resume: the command differs from the original sweep in "
                f"{diff}; rerun with the original arguments"
            )
        with open(results_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != columns:
                raise SystemExit(
                    "--resume: results.csv columns differ from this version of "
                    "the script; start a new --out"
                )
            previous = list(reader)
        # later rows win, so a rerun error case is replaced by its new result
        latest = {int(r["case"]): r for r in previous}
        done = {c for c, r in latest.items() if not r["status"].startswith("error")}
        jobs = [j for j in jobs if j[0] not in done]
        print(f"--resume: {len(done)} cases already finished, {len(jobs)} to run")
    elif os.path.exists(results_path):
        raise SystemExit(
            f"{results_path} exists: use --resume to continue that sweep, or "
            "choose a new --out"
        )

    if args.order != "index" and jobs:
        print(
            f"planning {len(jobs)} cases to order them by estimated run time ...",
            flush=True,
        )
        planned = plan_all(jobs, args.workers)
        est = estimated_costs(planned, step_costs(planned, args))
        jobs = order_jobs(jobs, est, args.order)
        print(
            f"order: {args.order} first; estimated wall time "
            f"{makespan([est[j[0]] for j in jobs], args.workers) / 3600:.1f} h"
        )

    os.makedirs(os.path.join(args.out, "series"), exist_ok=True)
    if args.save_fields:
        os.makedirs(os.path.join(args.out, "fields"), exist_ok=True)
    if not args.resume:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    rows, n_errors = [], 0
    with (
        open(results_path, "a" if args.resume else "w", newline="") as f,
        Pool(args.workers) as pool,
    ):
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        if not args.resume:
            writer.writeheader()
        for row, ts in pool.imap_unordered(run_case_safe, jobs):
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if ts is not None:
                np.savez_compressed(
                    os.path.join(args.out, "series", f"case_{row['case']:04d}.npz"),
                    **ts,
                )
            n_errors += row["status"].startswith("error")
            swept = ", ".join(
                f"{n}={row[n]:.4g}" if isinstance(row[n], float) else f"{n}={row[n]}"
                for n in names
            )
            sig = f"sigma*lam={row['sigma_lam']:+.4f}" if "sigma_lam" in row else ""
            print(
                f"[{len(rows)}/{len(jobs)}] {swept}  {row['status']}  {sig}", flush=True
            )
    if n_errors:
        print(
            f"\n{n_errors} cases ended in an error (tracebacks in {args.out}/errors/); "
            "rerun them with --resume after fixing the cause"
        )

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    final = {int(r["case"]): r for r in previous + rows}
    ranked = sorted(
        (r for r in final.values() if np.isfinite(num(r.get("sigma_lam")))),
        key=lambda r: -num(r["sigma_lam"]),
    )
    print("\nLargest growth rates:")
    for r in ranked[:10]:
        swept = ", ".join(f"{n}={r[n]}" for n in names)
        print(f"  sigma*lam={num(r['sigma_lam']):+.4f}  {r['status']:<24} {swept}")


def plan_summary(job):
    idx, pdict, cfg = job
    try:
        p, C0, plan = plan_case(pdict, cfg)
    except Exception as e:
        return idx, pdict, None, f"{type(e).__name__}: {e}"
    return idx, pdict, plan, None


ORDERS = ("index", "shortest", "longest")


def plan_all(jobs, workers):
    """{case: (pdict, plan or None, error or None)} for every job, planned in parallel."""
    with Pool(workers) as pool:
        return {
            i: (c, pl, err) for i, c, pl, err in pool.imap_unordered(plan_summary, jobs)
        }


def estimated_costs(plans, cost):
    """Estimated seconds per case (0 for cases that end at once: no steady state, error).
    cost: {N: seconds per step}, or a number, the ms per step at N = 128 scaled by N^2.
    """
    per_step = lambda N: (
        cost[N] if isinstance(cost, dict) else cost * (N / 128) ** 2 / 1000
    )
    return {
        i: 0.0 if pl is None else pl["n_steps"] * per_step(c["N"])
        for i, (c, pl, err) in plans.items()
    }


def _benchmark_case(pdict, warmup=2, steps=8):
    """Seconds per time step of one case's solver (state: rest plus small noise)."""
    s = KRSolver(case_params(pdict), 1000)
    s.perturb(1e-4, seed=0)
    for _ in range(warmup):
        s.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        s.step()
    return (time.perf_counter() - t0) / steps


def benchmark_step_costs(cases, concurrent):
    """{N: seconds per step} measured with `concurrent` copies of one case of each grid
    size running at once, as in the sweep: on a shared machine each case slows down as
    more run together (memory bandwidth as well as cores), which a fixed per-step cost
    misses. Median over the copies."""
    out = {}
    for N in sorted({c["N"] for c in cases}):
        pdict = next(c for c in cases if c["N"] == N)
        with Pool(concurrent) as pool:
            out[N] = float(np.median(pool.map(_benchmark_case, [pdict] * concurrent)))
    return out


def step_costs(planned, args):
    """--ms-per-step if given, else a benchmark of the planned cases (printed)."""
    if args.ms_per_step is not None:
        return args.ms_per_step
    cases = [c for c, pl, err in planned.values() if pl]
    if not cases:
        return {}
    n = max(1, min(args.workers, len(cases)))
    print(f"benchmarking the time step with {n} cases running at once ...", flush=True)
    cost = benchmark_step_costs(cases, n)
    print(
        "  measured: "
        + ", ".join(f"N = {N}: {1000 * v:.0f} ms/step" for N, v in cost.items())
    )
    return cost


def order_jobs(jobs, est, order):
    if order == "index":
        return list(jobs)
    sign = 1 if order == "shortest" else -1
    return sorted(jobs, key=lambda j: (sign * est[j[0]], j[0]))


def makespan(costs, workers):
    """Wall time of running jobs in this order, each going to the first free worker."""
    free = [0.0] * max(1, workers)
    for c in costs:
        heapq.heappush(free, heapq.heappop(free) + c)
    return max(free)


def dry_run(jobs, cfg, args):
    """Plan every case without running: cost, disk and warnings."""
    planned = plan_all(jobs, args.workers)
    plans = [(i, *planned[i]) for i in sorted(planned)]
    ok = [(i, c, pl) for i, c, pl, err in plans if pl]
    errors = [(i, err) for i, c, pl, err in plans if err]
    no_ss = len(plans) - len(ok) - len(errors)
    cores = os.cpu_count() or 1
    if args.workers > cores:
        print(
            f"note: --workers {args.workers} is more than the {cores} cores here; cases "
            "then share cores and the sweep is not faster"
        )
    est = estimated_costs(planned, step_costs(planned, args))
    secs = np.array([est[i] for i, c, pl in ok])
    steps = np.array([pl["steps"] for i, c, pl in ok])
    print(
        f"\nDry run: {len(plans)} cases, {no_ss} without a steady state (skipped at once), "
        f"{len(errors)} that would raise an error"
    )
    for i, err in errors[:5]:
        print(f"  case {i}: {err}")
    if not ok:
        return
    total = secs.sum()
    print(
        f"steps per period: min {steps.min()}, median {int(np.median(steps))}, max {steps.max()}"
    )
    print(
        f"estimated run time: {total / 3600:.1f} core-hours, about "
        f"{total / max(1, min(args.workers, len(ok))) / 3600:.1f} h on {args.workers} workers "
        f"({cores} cores here; full-length runs: cases stopped early take less)"
    )
    print(f"longest case: {secs.max() / 3600:.2f} h")
    for i in np.argsort(-secs)[:5]:
        idx, c, pl = ok[i]
        swept = ", ".join(f"{n}={c[n]}" for n in cfg_sweep_names(jobs))
        print(
            f"  case {idx}: {secs[i] / 60:7.1f} min, {pl['steps']} steps/period  ({swept})"
        )
    for order in ORDERS:
        wall = makespan([est[j[0]] for j in order_jobs(jobs, est, order)], args.workers)
        mark = "   <- --order " + order if order == args.order else ""
        print(
            f"wall time, {order + ' first' if order != 'index' else 'index order':>14}: "
            f"{wall / 3600:5.1f} h{mark}"
        )
    if cfg["save_fields"]:
        frames = sum(
            cfg["fields_per_period"]
            * periods_per_stage(cfg, kr_lattice(c.get("kr_lattice", "golden")).eps_p)
            * len(stages(cfg))
            + 1
            for _, c, _ in ok
        )
        n = ok[0][1]["N"]
        print(
            f"--save-fields: about {frames * 3 * n * n * 4 / 1e9:.1f} GB of field snapshots"
        )
    print(
        "This is one sweep: sweeps started one after another (e.g. one per seed in a "
        "shell loop) add their wall times."
    )
    print("No files were written.")


def cfg_sweep_names(jobs):
    """Parameters that vary across the cases (for labelling)."""
    keys = jobs[0][1].keys()
    return [k for k in keys if len({repr(j[1][k]) for j in jobs}) > 1]


if __name__ == "__main__":
    main()
