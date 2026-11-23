"""Verification tests for kr_solver. Run with `pytest test_kr.py` or `python test_kr.py`."""

import dataclasses

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm

from kr_solver import (
    EPS_P,
    M_INV_T,
    XX,
    XY,
    YY,
    ZZ,
    KRSolver,
    M,
    Params,
    R,
    conformation_to_log,
    homogeneous_rhs,
    log_source,
    log_to_conformation,
    model_source,
    upper_convected,
)


def run_periods(s, n_periods):
    for _ in range(int(round(n_periods * s.steps_per_period))):
        s.step()


def test_lattice_maps_onto_itself():
    Lam = np.diag([np.exp(EPS_P), np.exp(-EPS_P)])
    assert np.allclose(Lam @ R, R @ M, atol=1e-12)


def test_remap_preserves_field_at_physical_points():
    p = Params("oldroyd_b", eps_dot=1.0, N=32)
    s = KRSolver(p, steps_per_period=10)
    s.perturb(1.0, nmax=3, seed=1)
    pts = np.random.default_rng(2).uniform(-2, 2, size=(50, 2))

    def evaluate(tau):
        kx, ky = s.wavevectors(tau)
        nz = np.abs(s.Ch[XX]) > 1e-12
        phase = np.exp(1j * (np.outer(pts[:, 0], kx[nz]) + np.outer(pts[:, 1], ky[nz])))
        return (phase @ s.Ch[XX][nz]).real / p.N**2

    before = evaluate(s.t_p)
    s._remap()
    after = evaluate(0.0)
    assert s.remap_loss[-1] == 0.0
    assert np.allclose(before, after, atol=1e-12)


CASES = {
    "oldroyd_b": dict(eps_dot=0.3),
    "giesekus": dict(eps_dot=1.0, alpha=0.3),
    "johnson_segalman": dict(eps_dot=0.6, a=0.5),
    "rolie_poly": dict(eps_dot=1.0, tau_R=0.1, beta=0.5, delta=-0.5),
}


def test_log_source_matches_matrix_log_derivative():
    """d/dt log C from the decomposition equals the derivative of logm(C(t))."""
    rng = np.random.default_rng(3)
    h = 1e-6
    to_mat = lambda c: np.array([[c[XX], c[XY]], [c[XY], c[YY]]])
    for model, kw in CASES.items():
        p = Params(model, **kw)
        for _ in range(5):
            A = rng.standard_normal((2, 2))
            C = np.array(
                [
                    *(lambda S: (S[0, 0], S[0, 1], S[1, 1]))(A @ A.T + 0.5 * np.eye(2)),
                    rng.uniform(0.5, 2),
                ]
            )
            kap = tuple(rng.standard_normal(4))
            f = model_source(C, kap, p)
            dlog = (logm(to_mat(C + h * f)) - logm(to_mat(C - h * f))).real / (2 * h)
            want = np.array([dlog[0, 0], dlog[0, 1], dlog[1, 1], f[ZZ] / C[ZZ]])
            got = log_source(conformation_to_log(C), kap, p)
            assert np.allclose(got, want, rtol=1e-6, atol=1e-6), (model, got, want)
    # equal eigenvalues (C = I) must not produce NaNs
    got = log_source(
        np.zeros(4), (0.3, 0.2, -0.1, -0.3), Params("oldroyd_b", eps_dot=1.0)
    )
    assert np.all(np.isfinite(got))
    C = np.array([2.0, 0.3, 0.7, 1.4])
    assert np.allclose(log_to_conformation(conformation_to_log(C)), C)


def test_homogeneous_matches_0d_ode_all_models():
    for (model, kw), log in [(c, log) for c in CASES.items() for log in (False, True)]:
        p = Params(model, N=16, D=1e-3, log_conformation=log, **kw)
        s = KRSolver(p, steps_per_period=200)
        run_periods(s, 3)
        ref = solve_ivp(
            lambda t, c: homogeneous_rhs(c, p),
            (0, s.t),
            [1.0, 0.0, 1.0, 1.0],
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
        )
        err = np.max(np.abs(s.mean_C() - ref.y[:, -1]) / np.abs(ref.y[:, -1]).max())
        assert err < 1e-7, (model, log, err)
        print(f"  {model}{' (log)' if log else ''}: homogeneous rel. error {err:.2e}")
        assert (
            np.abs(s.Ch[:, 1:, :]).max() < 1e-8 and np.abs(s.Ch[:, 0, 1:]).max() < 1e-8
        )


def test_oldroyd_b_steady_state_analytic():
    Wi = 0.2
    s = KRSolver(Params("oldroyd_b", eps_dot=Wi, N=16), steps_per_period=200)
    run_periods(s, 6)
    assert np.isclose(s.mean_C()[XX], 1 / (1 - 2 * Wi), rtol=1e-6)
    assert np.isclose(s.mean_C()[YY], 1 / (1 + 2 * Wi), rtol=1e-6)


def _linear_single_mode(log, lattice="golden", N=32, n_periods=2.5, steps=400):
    """Small single-mode perturbation (Oldroyd-B) vs the exact per-mode linear ODE in
    fixed physical coordinates. Stress diffusion is only included in conformation mode,
    because in log mode it acts on Psi, which is not equivalent to diffusing C."""
    p = Params(
        "oldroyd_b",
        eps_dot=0.4,
        lam=1.0,
        G=1.0,
        eta_s=0.2,
        D=0.0 if log else 1e-3,
        N=N,
        log_conformation=log,
        kr_lattice=lattice,
    )
    s = KRSolver(p, steps_per_period=steps)
    amp, n0 = 1e-7, np.array([1, 0])
    pattern = np.array([1.0, 0.5, -0.7, 0.0])
    xi = np.arange(p.N) / p.N
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    s.set_C(
        np.array([1.0, 0.0, 1.0, 1.0])[:, None, None]
        + pattern[:, None, None] * amp * np.cos(2 * np.pi * (n0[0] * X1 + n0[1] * X2))
    )
    kx0, ky0 = s.kx0[n0[0], n0[1]], s.ky0[n0[0], n0[1]]
    L = (p.eps_dot, 0.0, 0.0, -p.eps_dot)

    def ref_rhs(t, y):
        C0, d = y[:4], y[4:]
        kx, ky = kx0 * np.exp(-p.eps_dot * t), ky0 * np.exp(p.eps_dot * t)
        fx = 1j * p.G * (kx * d[0] + ky * d[1])
        fy = 1j * p.G * (kx * d[1] + ky * d[2])
        k2 = kx**2 + ky**2
        kf = (kx * fx + ky * fy) / k2
        vx, vy = (fx - kx * kf) / (p.eta_s * k2), (fy - ky * kf) / (p.eta_s * k2)
        dk = (1j * kx * vx, 1j * ky * vx, 1j * kx * vy, 1j * ky * vy)
        dd = upper_convected(d, L) + upper_convected(C0, dk) - d / p.lam - p.D * k2 * d
        return np.concatenate([homogeneous_rhs(C0.real, p), dd])

    run_periods(s, n_periods)
    y0 = np.concatenate([[1, 0, 1, 1], pattern * amp / 2]).astype(complex)
    ref = solve_ivp(ref_rhs, (0, s.t), y0, method="DOP853", rtol=1e-11, atol=1e-20)

    n = n0.copy()
    for _ in range(int(n_periods)):
        n = s.lat.M_inv_T @ n
    got = np.fft.fft2(s.C())[:3, n[0] % p.N, n[1] % p.N] / p.N**2
    want = ref.y[4:7, -1]
    err = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert err < 1e-5, err
    assert max(s.remap_loss) < 1e-12  # roundoff-level only
    print(
        f"  linear-mode{' (log)' if log else ''} [{lattice}] rel. error {err:.2e}, "
        f"remap losses {s.remap_loss}"
    )


def test_linear_single_mode_matches_mode_ode_across_remaps():
    _linear_single_mode(log=False)


def test_linear_single_mode_log_conformation():
    _linear_single_mode(log=True)


def test_other_kr_lattices():
    """Each lattice maps onto itself after one period, its remap is exact at physical points,
    the cell at half a period is the mirror image, and a linear mode is exact across remaps."""
    from kr_solver import LATTICES, kr_lattice

    g = kr_lattice("golden")
    assert (
        np.allclose(g.R, R)
        and np.isclose(g.eps_p, EPS_P)
        and (g.M_inv_T == M_INV_T).all()
    )
    for name in LATTICES:
        lat = kr_lattice(name)
        Lam = np.diag([np.exp(lat.eps_p), np.exp(-lat.eps_p)])
        assert np.allclose(Lam @ lat.R, lat.R @ lat.M, atol=1e-10)
        assert np.isclose(lat.cycle, lat.eps_p / 2)
        p = Params("oldroyd_b", eps_dot=1.0, N=48, kr_lattice=name)
        s = KRSolver(p, steps_per_period=10)
        s.perturb(1.0, nmax=2, seed=1)
        pts = np.random.default_rng(2).uniform(-2, 2, size=(50, 2))

        def evaluate(tau):
            kx, ky = s.wavevectors(tau)
            nz = np.abs(s.Ch[XX]) > 1e-12
            ph = np.exp(
                1j * (np.outer(pts[:, 0], kx[nz]) + np.outer(pts[:, 1], ky[nz]))
            )
            return (ph @ s.Ch[XX][nz]).real / p.N**2

        before = evaluate(s.t_p)
        s._remap()
        assert s.remap_loss[-1] == 0.0 and np.allclose(
            before, evaluate(0.0), atol=1e-12
        ), name
    mirror = kr_lattice("golden_mirror")
    assert np.isclose(np.degrees(mirror.theta), 90 - np.degrees(g.theta))
    assert np.isclose(np.degrees(kr_lattice("silver").theta), 22.5)
    # with diffusion the mode decays below the reference ODE's tolerance after 2 silver periods,
    # so check one remap with diffusion and two without
    _linear_single_mode(log=False, lattice="silver", N=48, n_periods=1.5, steps=800)
    _linear_single_mode(log=True, lattice="silver", N=48, n_periods=2, steps=800)


def test_sweep_helpers():
    from sweep_kr import fit_growth, steady_state

    Wi = 0.2
    C0 = steady_state(Params("oldroyd_b", eps_dot=Wi))
    assert np.allclose(C0[[XX, YY]], [1 / (1 - 2 * Wi), 1 / (1 + 2 * Wi)], rtol=1e-8)
    assert (
        steady_state(Params("oldroyd_b", eps_dot=0.6)) is None
    )  # unbounded above Wi = 0.5
    t = np.linspace(0, 10, 50)
    assert np.isclose(fit_growth(t, 1e-4 * np.exp(0.37 * t)), 0.37)


def test_smoothed_noise_has_requested_length():
    """Gaussian-smoothed white noise in 2D has <|grad f|^2> / <f^2> = 1 / l^2."""
    p = Params("oldroyd_b", eps_dot=1.0, N=128)
    for l, tau in [(0.04, 0.0), (0.06, 0.5)]:  # also on a deformed cell (tau > 0)
        s = KRSolver(p, steps_per_period=10)
        s.tau = tau
        s.perturb(1e-3, seed=4, smooth_length=l)
        kx, ky = s.wavevectors(tau)
        F = s.Ch[:3].copy()
        F[:, 0, 0] = 0
        ratio = np.sum((kx**2 + ky**2) * np.abs(F) ** 2) / np.sum(np.abs(F) ** 2)
        assert abs(ratio * l**2 - 1) < 0.1, (l, tau, ratio * l**2)


def test_velocity_rms_matches_solver_velocity():
    """sweep_kr.velocity_rms (lattice-based wavevectors) equals the rms of the solver's
    own Stokes velocity, on a deformed cell."""
    from sweep_kr import velocity_rms

    p = Params("giesekus", eps_dot=0.7, alpha=0.3, eta_s=0.08, G=1.3, Lbox=1.7, N=48)
    s = KRSolver(p, steps_per_period=50)
    s.perturb(1e-2, seed=5)
    for _ in range(17):
        s.step()  # tau > 0: the cell is sheared
    vx, vy = s.velocity_hat(s.Ch, *s.wavevectors(s.tau))
    vx, vy = np.fft.ifft2(vx).real, np.fft.ifft2(vy).real
    want = np.sqrt(np.mean(vx**2 + vy**2)) / (p.eps_dot * p.Lbox)
    C = s.C()[:3]
    got = velocity_rms(
        C - C.mean(axis=(1, 2), keepdims=True), s.lattice(), p.G, p.eta_s, p.eps_dot
    )
    assert np.isclose(got, want, rtol=1e-10), (got, want)


def test_forcing_increment_statistics():
    """Each forcing increment has zero cell average and rms forcing_amp * sqrt(dt)."""
    for length in (0.0, 0.05):
        p = Params(
            "oldroyd_b", eps_dot=0.5, N=64, forcing_amp=0.3, forcing_length=length
        )
        s = KRSolver(p, steps_per_period=40)
        rms = []
        for _ in range(200):
            before = s.Ch.copy()
            s._force()
            inc = np.fft.ifft2(s.Ch - before).real[:3]
            assert np.abs(inc.mean(axis=(1, 2))).max() < 1e-12
            rms.append(np.sqrt(np.mean(inc**2)))
        assert abs(np.sqrt(np.mean(np.square(rms))) / (0.3 * np.sqrt(s.dt)) - 1) < 0.05


def test_forcing_off_is_unchanged_and_level_is_dt_independent():
    p = Params("giesekus", eps_dot=0.5, alpha=0.3, eta_s=0.1, D=1e-3, N=32)
    s = KRSolver(p, 50)
    s.perturb(1e-3, seed=2)
    base = s.Ch.copy()
    s.step()
    s2 = KRSolver(dataclasses.replace(p, forcing_amp=0.0), 50)
    s2.perturb(1e-3, seed=2)
    s2.step()
    assert np.array_equal(s.Ch, s2.Ch) and not np.array_equal(s.Ch, base)

    # Statistically steady noise floor: same level for two time steps (within sampling error)
    pf = dataclasses.replace(p, forcing_amp=1e-3, forcing_length=0.05)
    levels = []
    for steps in (100, 400):
        s = KRSolver(pf, steps)
        vals = []
        for k in range(8 * steps):
            s.step()
            if k >= 4 * steps and k % (steps // 20) == 0:
                C = s.C()[:3]
                vals.append(np.mean((C - C.mean(axis=(1, 2), keepdims=True)) ** 2))
        levels.append(np.sqrt(np.mean(vals)))
    print(
        f"  forced noise floor rms: {levels[0]:.3e} (100 steps/period), "
        f"{levels[1]:.3e} (400 steps/period)"
    )
    assert abs(levels[0] / levels[1] - 1) < 0.15


# --- finite-extensibility Rolie-Poly -------------------------------------------------------
FENE = dict(tau_R=0.05, delta=-0.5, L_max=5.0)


def test_fene_rest_state_and_off_switch():
    from kr_solver import relaxation, stress_conformation

    I = np.array([1.0, 0.0, 1.0, 1.0])
    p = Params("rolie_poly", eps_dot=1.0, beta=0.5, **FENE)
    assert np.allclose(relaxation(I, p), 0, atol=1e-15)  # C = I is the rest state
    assert np.allclose(stress_conformation(I, p), I)  # with zero stress
    C = np.array([2.0, 0.4, 0.8, 1.1])
    p0 = dataclasses.replace(p, L_max=0.0)
    assert np.array_equal(
        relaxation(C, p0),
        relaxation(
            C, Params("rolie_poly", eps_dot=1.0, beta=0.5, tau_R=0.05, delta=-0.5)
        ),
    )
    for bad in (dict(L_max=1.0), dict(model="giesekus")):
        try:
            relaxation(C, dataclasses.replace(p, **bad))
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_fene_steady_extension_matches_independent_prototype():
    """Values from an independent implementation of the same equations (L_max = 5)."""
    from kr_solver import stress_conformation
    from sweep_kr import steady_state

    for beta, WiR, lam_ref, N1_ref in [
        (0.0, 10.0, 4.800, 846.34),
        (0.5, 1.05, 2.622, 26.85),
        (0.5, 3.0, 4.080, 143.01),
    ]:
        p = Params("rolie_poly", eps_dot=WiR / 0.05, beta=beta, **FENE)
        C = steady_state(p)
        S = stress_conformation(C, p)
        lam = np.sqrt((C[XX] + C[YY] + C[ZZ]) / 3)
        # references are rounded to 3 and 2 decimals: agree to half a unit in the last digit
        assert abs(lam - lam_ref) <= 5e-4 and abs(S[XX] - S[YY] - N1_ref) <= 5e-3, (
            beta,
            WiR,
            lam,
            S[XX] - S[YY],
        )
    # plain Rolie-Poly has no steady state at eps_dot * tau_R = 3; finite extensibility does
    assert steady_state(Params("rolie_poly", eps_dot=60.0, tau_R=0.05)) is None


def test_fene_log_source_and_homogeneous_runs():
    to_mat = lambda c: np.array([[c[XX], c[XY]], [c[XY], c[YY]]])
    rng = np.random.default_rng(7)
    p = Params("rolie_poly", eps_dot=20.0, beta=0.5, **FENE)
    for _ in range(10):
        A = rng.standard_normal((2, 2))
        S = A @ A.T * 3 + 0.3 * np.eye(2)
        C = np.array([S[0, 0], S[0, 1], S[1, 1], rng.uniform(0.3, 2)])
        if (C[XX] + C[YY] + C[ZZ]) / 3 > 0.9 * p.L_max**2:
            continue
        kap = tuple(rng.standard_normal(4))
        f = model_source(C, kap, p)
        h = 1e-6
        d = (logm(to_mat(C + h * f)) - logm(to_mat(C - h * f))).real / (2 * h)
        want = np.array([d[0, 0], d[0, 1], d[1, 1], f[ZZ] / C[ZZ]])
        assert np.allclose(
            log_source(conformation_to_log(C), kap, p), want, rtol=1e-6, atol=1e-6
        )
    for log in (False, True):  # 2D uniform state vs the 0D ODE, at eps_dot * tau_R = 3
        p = Params(
            "rolie_poly", eps_dot=60.0, beta=0.5, N=16, log_conformation=log, **FENE
        )
        s = KRSolver(p, steps_per_period=300)
        run_periods(s, 2)
        ref = solve_ivp(
            lambda t, c: homogeneous_rhs(c, p),
            (0, s.t),
            [1.0, 0.0, 1.0, 1.0],
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
        )
        err = np.max(np.abs(s.mean_C() - ref.y[:, -1]) / np.abs(ref.y[:, -1]).max())
        assert err < 1e-7, (log, err)


def test_fene_stokes_uses_spring_stress_and_guard():
    from kr_solver import StretchLimitError, fene_factor
    from sweep_kr import velocity_rms

    p = Params(
        "rolie_poly",
        eps_dot=2.0,
        beta=0.5,
        eta_s=0.1,
        N=32,
        log_conformation=True,
        **FENE,
    )
    s = KRSolver(p, steps_per_period=50)
    C = np.broadcast_to(
        np.array([8.0, 0.0, 0.5, 1.0])[:, None, None], (4, 32, 32)
    ).copy()
    s.set_C(C)
    s.perturb(0.05, seed=3)
    C = s.C()
    want = np.fft.fft2(fene_factor(C, p) * C)
    S = np.fft.ifft2(s.Ch).real
    assert np.allclose(s.stress_hat(S, s.Ch), want)
    # velocity reconstructed from a stress snapshot equals the solver's own velocity
    vx, vy = (np.fft.ifft2(v).real for v in s.velocity_hat(want, *s.wavevectors(s.tau)))
    fC = (fene_factor(C, p) * C)[:3]
    got = velocity_rms(
        fC - fC.mean(axis=(1, 2), keepdims=True), s.lattice(), p.G, p.eta_s, p.eps_dot
    )
    assert np.isclose(got, np.sqrt(np.mean(vx**2 + vy**2)) / (p.eps_dot * p.Lbox))
    # guard: a state stretched past L_max stops the step with a clear error
    C[XX] += 3 * p.L_max**2
    s.set_C(C)
    try:
        s.step()
        raise AssertionError("expected StretchLimitError")
    except StretchLimitError as e:
        assert "L_max" in str(e)


# --- sweep safety -----------------------------------------------------------------------
def test_linear_rate_known_values():
    from sweep_kr import linear_rate, steady_state

    p = Params(
        "oldroyd_b", eps_dot=1e-6, eta_s=0.01
    )  # solvent mode at rest: 1/lam + G/eta_s
    assert abs(linear_rate(p, steady_state(p)) - 101.0) < 1e-3
    p = Params("rolie_poly", eps_dot=10.0, tau_R=0.0167, eta_s=0.001)
    assert abs(linear_rate(p, steady_state(p)) / 4204.2 - 1) < 1e-4


def test_sweep_errors_resume_and_dry_run():
    import contextlib
    import csv
    import io
    import os
    import tempfile

    from sweep_kr import main

    base = [
        "--model",
        "rolie_poly",
        "--set",
        "N=16",
        "--set",
        "tau_R=0.1",
        "--strain",
        "1",
        "--sweep",
        "eps_dot=1,2",
        "--sweep",
        "L_max=0,1.0",
    ]  # L_max = 1 raises an error
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sw")
        with quiet():
            main(base + ["--out", out])
        rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
        status = {int(r["case"]): r["status"] for r in rows}
        assert (
            len(rows) == 4
            and sum(s.startswith("error: ValueError") for s in status.values()) == 2
        )
        assert sorted(os.listdir(os.path.join(out, "errors"))) == [
            "case_0001.txt",
            "case_0003.txt",
        ]
        assert sorted(os.listdir(os.path.join(out, "series"))) == [
            "case_0000.npz",
            "case_0002.npz",
        ]
        assert all(
            float(r["dtxx_peak"]) > 0
            for r in rows
            if not r["status"].startswith("error")
        )
        for extra, msg in ([], "exists"), (["--resume", "--strain", "2"], "differs"):
            try:
                with quiet():
                    main(base + ["--out", out] + extra)
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert msg in str(e), e
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(base + ["--out", out, "--resume"])
        assert "2 cases already finished, 2 to run" in buf.getvalue()
        assert len(list(csv.DictReader(open(os.path.join(out, "results.csv"))))) == 6
        dry = os.path.join(tmp, "dry")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(base + ["--out", dry, "--dry-run", "--save-fields"])
        assert "2 that would raise an error" in buf.getvalue() and not os.path.exists(
            dry
        )


def test_job_ordering_and_makespan():
    from sweep_kr import makespan, order_jobs

    jobs = [(i, {}, {}) for i in range(5)]
    est = {0: 3.0, 1: 1.0, 2: 0.0, 3: 5.0, 4: 1.0}
    assert [j[0] for j in order_jobs(jobs, est, "shortest")] == [2, 1, 4, 0, 3]
    assert [j[0] for j in order_jobs(jobs, est, "longest")] == [3, 0, 1, 4, 2]
    assert [j[0] for j in order_jobs(jobs, est, "index")] == [0, 1, 2, 3, 4]
    assert makespan([5, 3, 1, 1, 0], 2) == 5 and makespan([0, 1, 1, 3, 5], 2) == 6


def test_rank_classifies_synthetic_sweep():
    """Synthetic sweep: flow-driven plateau (same level for both forcings), noise floor
    (level proportional to forcing), still growing, and transient."""
    import contextlib
    import csv
    import io
    import json
    import os
    import tempfile

    from viz_sweep import rank_cases

    e = np.linspace(0, 10, 101)
    shapes = {  # eps_dot -> max|dtau_xx| history as a function of strain and forcing amp
        1.0: lambda A: np.minimum(1e-4 * np.exp(2 * e), 1.0),  # saturates at 1 G
        2.0: lambda A: np.full_like(e, 50 * A) + 1e-4 * np.exp(-e),  # noise floor ~ A
        3.0: lambda A: 1e-4 * np.exp(0.8 * e),  # still growing
        4.0: lambda A: 1e-4 * np.exp(5 * e * np.exp(-e / 2)),  # transient, ~40x peak
    }
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "series"))
        json.dump(
            {
                "model": "rolie_poly",
                "fixed": {},
                "sweeps": {"eps_dot": list(shapes), "forcing_amp": [1e-5, 1e-3]},
                "sigma_tol": 0.1,
            },
            open(os.path.join(d, "config.json"), "w"),
        )
        rows, c = [], 0
        for ed, f in shapes.items():
            for A in (1e-5, 1e-3):
                y = f(A)
                np.savez(
                    os.path.join(d, "series", f"case_{c:04d}.npz"),
                    t=e / ed,
                    rms=y,
                    vrms=y,
                    dtxx=y,
                )
                rows.append(
                    dict(
                        case=c,
                        eps_dot=ed,
                        forcing_amp=A,
                        status="stable",
                        sigma_lam=0.0,
                    )
                )
                c += 1
        with open(os.path.join(d, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            rank_cases(d)
        got = {
            (float(r["eps_dot"]), float(r["forcing_amp"])): r["behaviour"]
            for r in csv.DictReader(open(os.path.join(d, "triage.csv")))
        }
    want = {1.0: "plateau", 2.0: "noise_floor", 3.0: "growing_at_end", 4.0: "transient"}
    assert got == {(ed, A): want[ed] for ed in want for A in (1e-5, 1e-3)}, got


# --- Fourier analysis -------------------------------------------------------------------
def test_fourier_single_mode_on_deformed_lattice():
    """A single lattice mode on a sheared cell lands at its physical |k| and angle, and the
    binned powers sum to the field variance (Parseval)."""
    from fourier_kr import bin_polar, find_peaks, frame_power, k_edges_for

    p = Params("oldroyd_b", eps_dot=1.0, N=48)
    s = KRSolver(p, steps_per_period=40)
    for _ in range(23):
        s.step()  # sheared cell, tau > 0
    A = s.lattice()
    xi = np.arange(48) / 48
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    m = np.array([3, -5])
    field = 0.7 * np.cos(2 * np.pi * (m[0] * X1 + m[1] * X2))
    k_true = 2 * np.pi * np.linalg.inv(A).T @ m
    kx, ky, P = frame_power(field, A)
    assert np.isclose(P.sum(), np.var(field))
    i = np.argmax(P)
    assert np.allclose([kx[i], ky[i]], k_true) or np.allclose([kx[i], ky[i]], -k_true)
    edges = k_edges_for([A], 48, 64)
    H = bin_polar(kx, ky, P, edges, 72)
    pk = find_peaks(H, edges, 72)[0]
    th_true = np.degrees(np.mod(np.arctan2(k_true[1], k_true[0]), np.pi))
    assert abs(np.log(pk["k"] / np.hypot(*k_true))) < np.log(edges[1] / edges[0])
    assert (
        min(abs(pk["theta_deg"] - th_true), 180 - abs(pk["theta_deg"] - th_true)) < 2.5
    )
    assert pk["power_fraction"] > 0.99


def test_fourier_peak_merges_across_theta_wrap():
    from fourier_kr import find_peaks

    edges = np.geomspace(1, 100, 21)
    P = np.full((20, 36), 1e-6)
    P[9:11, :2] = 1.0  # theta just above 0 ...
    P[9:11, -2:] = 1.0  # ... and just below 180: the same orientation
    pk = find_peaks(P, edges, 36, percentile=95)
    assert len(pk) == 1 and min(pk[0]["theta_deg"], 180 - pk[0]["theta_deg"]) < 1e-6
    from fourier_kr import overall

    o = overall(P, edges, 36)
    assert (
        min(o["theta_mean_deg"], 180 - o["theta_mean_deg"]) < 1e-6
        and o["anisotropy"] > 0.9
    )
    assert abs(pk[0]["k"] / 10 - 1) < 0.15


def test_stretch_reconstruction_inverts_finite_extensibility():
    from fourier_kr import stretch_fluctuation
    from kr_solver import stress_conformation

    p = Params("rolie_poly", eps_dot=1.0, tau_R=0.05, L_max=5.0)
    rng = np.random.default_rng(1)
    C = np.empty((2, 4, 8, 8))
    C[:, XX] = 4 + rng.uniform(0, 3, (2, 8, 8))
    C[:, YY] = 0.3 + rng.uniform(0, 0.2, (2, 8, 8))
    C[:, XY] = 0.1
    C[:, ZZ] = 1 + rng.uniform(0, 0.3, (2, 8, 8))
    S = np.stack([stress_conformation(Cf, p) for Cf in C])
    mean = S.mean(axis=(2, 3))
    frames = dict(dC=S - mean[:, :, None, None], mean=mean)
    lam = np.sqrt((C[:, XX] + C[:, YY] + C[:, ZZ]) / 3)
    assert np.allclose(
        stretch_fluctuation(frames, p), lam - lam.mean(axis=(1, 2), keepdims=True)
    )


# --- flow type ----------------------------------------------------------------------------
def test_flow_type_known_flows():
    from flowtype_kr import flow_type

    one = np.ones(3)
    assert np.allclose(
        flow_type(2 * one, 0 * one, 0 * one, -2 * one), 1
    )  # planar extension
    assert np.allclose(flow_type(0 * one, 3 * one, 0 * one, 0 * one), 0)  # simple shear
    assert np.allclose(flow_type(0 * one, 1 * one, -1 * one, 0 * one), -1)  # rotation


def test_streamfunction_gives_velocity_and_rotation_sense():
    """psi' reproduces v = (d psi'/dy, -d psi'/dx) on a sheared cell, and the vorticity at its
    maximum is positive (solid contours = counter-clockwise, as the overlay legend says)."""
    from flowtype_kr import perturbation_velocity_hat, streamfunction

    rng = np.random.default_rng(1)
    N, A = 32, np.array([[1.3, 0.4], [-0.2, 0.8]])
    low = (
        np.abs(np.fft.fftfreq(N, 1 / N)) < 5
    )  # smooth random stress, no Nyquist content
    dS = np.fft.ifft2(np.fft.fft2(rng.normal(size=(3, N, N))) * np.outer(low, low)).real
    kx, ky, vx, vy = perturbation_velocity_hat(dS, A, 1.0, 0.1)
    psi = streamfunction(kx, ky, vx, vy)
    ph = np.fft.fft2(psi)
    real = lambda h: np.fft.ifft2(h).real  # real-space fields, as plotted
    ux, uy = real(vx), real(vy)
    assert np.allclose(real(1j * ky * ph), ux, atol=1e-10 * np.abs(ux).max())
    assert np.allclose(real(-1j * kx * ph), uy, atol=1e-10 * np.abs(uy).max())
    omega = real(1j * kx * vy - 1j * ky * vx)
    assert omega.flat[np.argmax(psi)] > 0


def test_sample_periodic_maps_lattice_to_physical():
    """Sampling at x = A xi_node returns the node values, also from a periodic copy."""
    from flowtype_kr import sample_periodic

    rng = np.random.default_rng(2)
    N, A = 16, np.array([[1.3, 0.4], [-0.2, 0.8]])
    field = rng.normal(size=(N, N))
    i1, i2 = rng.integers(0, N, 20), rng.integers(0, N, 20)
    xi = np.stack([i1, i2]) / N + np.array([[2], [-1]])  # shifted by a lattice vector
    X, Y = A @ xi
    assert np.allclose(sample_periodic(field, A, X, Y), field[i1, i2])


def test_velocity_gradient_from_snapshot_matches_solver():
    """kappa rebuilt from a saved stress fluctuation equals L + grad v from the solver's own
    Stokes velocity (finite extensibility, log mode, sheared cell)."""
    from flowtype_kr import flow_type, velocity_gradient
    from kr_solver import stress_conformation

    p = Params(
        "rolie_poly",
        eps_dot=2.0,
        tau_R=0.05,
        beta=0.5,
        eta_s=0.1,
        L_max=5.0,
        N=32,
        log_conformation=True,
    )
    s = KRSolver(p, steps_per_period=50)
    s.set_C(
        np.broadcast_to(
            np.array([6.0, 0.0, 0.4, 1.0])[:, None, None], (4, 32, 32)
        ).copy()
    )
    s.perturb(0.05, seed=4)
    for _ in range(13):
        s.step()
    S = stress_conformation(s.C(), p)
    dS = S - S.mean(axis=(1, 2), keepdims=True)
    got = velocity_gradient(dS, s.lattice(), p.G, p.eta_s, p.eps_dot)
    kx, ky = s.wavevectors(s.tau)
    vx, vy = s.velocity_hat(s.stress_hat(np.fft.ifft2(s.Ch).real, s.Ch), kx, ky)
    d = lambda k, v: np.fft.ifft2(1j * k * v).real
    want = (p.eps_dot + d(kx, vx), d(ky, vx), d(kx, vy), -p.eps_dot + d(ky, vy))
    for g, w in zip(got, want):
        assert np.allclose(g, w, atol=1e-10 * p.eps_dot)
    assert np.allclose(got[0] + got[3], 0, atol=1e-10)  # incompressible
    xi = flow_type(*got)
    assert xi.max() <= 1 + 1e-12 and xi.min() < 1 - 1e-6  # perturbed, but bounded
    zero = velocity_gradient(np.zeros_like(dS), s.lattice(), p.G, p.eta_s, p.eps_dot)
    assert np.allclose(flow_type(*zero), 1)  # base flow: pure extension


# --- structure analysis -------------------------------------------------------------------
def test_structure_measures_on_known_fields():
    """A single mode gives C = cos(k.r): repeat at 2 pi/|k| in its direction; an amplitude
    modulated on the half-period cell cycle is detected, random amplitudes are not."""
    from structure_kr import (
        HALF,
        axis_cut,
        cell_cycle_test,
        cut_measures,
        physical_acf,
        secondary_peak,
    )

    p = Params("oldroyd_b", eps_dot=1.0, N=64)
    s = KRSolver(p, steps_per_period=20)
    xi = np.arange(64) / 64
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    fields, lattices = [], []
    for f in range(
        20
    ):  # one full KR period, deforming cell; mode chosen as (0, 1) in y
        A = s.lattice()
        k_target = np.array(
            [0.0, 2 * np.pi * 3]
        )  # wavelength 1/3 across the extension axis
        m = np.round(A.T @ k_target / (2 * np.pi)).astype(int)
        fields.append(np.cos(2 * np.pi * (m[0] * X1 + m[1] * X2)))
        lattices.append(A)
        s.step()
    cen, C = physical_acf(np.array(fields), np.array(lattices), h=0.8 / 64)
    r, v = axis_cut(cen, C, 1)
    m = cut_measures(r, v, 0.5)
    assert abs(m["min_r"] - 1 / 6) < 0.03 and abs(m["repeat_r"] - 1 / 3) < 0.03, m
    pk = secondary_peak(cen, C, np.array(lattices), 0.5)
    assert (
        abs(pk["peak_r"] - 1 / 3) < 0.05
        and min(pk["peak_theta_deg"], 180 - pk["peak_theta_deg"]) > 60
    ), pk
    e = np.arange(200) * HALF / 5
    rng = np.random.default_rng(0)
    locked = 1 + 0.1 * np.cos(2 * np.pi * e / HALF) + 0.02 * rng.standard_normal(200)
    assert cell_cycle_test(e, locked)["cell_locked_fraction"] > 0.8
    free = 1 + 0.1 * rng.standard_normal(200)
    res = cell_cycle_test(e, free)
    assert res["cell_locked_fraction"] < 4 * res["cell_locked_chance"]


def _field_from_spectrum(A, N, S):
    """Real lattice-grid field whose dealiased mode powers are S(kx, ky) (zero phases)."""
    n = np.fft.fftfreq(N, 1.0 / N)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    B = 2 * np.pi * np.linalg.inv(A).T
    kx, ky = B[0, 0] * n1 + B[0, 1] * n2, B[1, 0] * n1 + B[1, 1] * n2
    F = np.sqrt(S(kx, ky)) * ((np.abs(n1) <= N // 3) & (np.abs(n2) <= N // 3))
    F[0, 0] = 0
    return np.fft.ifft2(F).real


def _deformed_cells(L, strains=(0.0, 0.3, 0.6, 0.9)):
    """Golden-lattice cells of side L at several strains within one KR period."""
    return [np.diag([np.exp(e), np.exp(-e)]) @ (L * R) for e in strains]


def test_gradient_lengths_tilted_gaussian():
    """A Gaussian autocorrelation of widths l_long (axis at 28 deg) and l_across, drawn on
    deforming cells: tilt, Taylor and cutoff (= l / sqrt 3) lengths are exact, and the
    mirror image has the opposite tilt and the same lengths."""
    from structure_kr import gradient_lengths

    th, l_long, l_across, N = np.radians(28.0), 0.1, 0.04, 192
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    S = lambda kx, ky: np.exp(
        -0.5
        * (
            (kx * u[0] + ky * u[1]) ** 2 * l_long**2
            + (kx * v[0] + ky * v[1]) ** 2 * l_across**2
        )
    )
    lats = _deformed_cells(1.0)
    fields = np.array([_field_from_spectrum(A, N, S) for A in lats])
    g = gradient_lengths(fields, lats)
    assert abs(g["lobe_theta_deg"] - 28.0) < 0.1, g
    # a zero-mean field lacks the k = 0 mode, a fraction f of a Gaussian's variance
    f = 2 * np.pi * l_long * l_across
    for key, want in [
        ("taylor_across", l_across * np.sqrt(1 - f)),
        ("taylor_along", l_long * np.sqrt(1 - f)),
        ("cutoff_across", l_across / np.sqrt(3)),
        ("cutoff_along", l_long / np.sqrt(3)),
    ]:
        assert abs(g[key] / want - 1) < 1e-4, (key, g[key], want)
    S_m = lambda kx, ky: S(kx, -ky)
    m = gradient_lengths(
        np.array([_field_from_spectrum(A, N, S_m) for A in lats]), lats
    )
    assert abs(m["lobe_theta_deg"] + 28.0) < 0.1 and np.isclose(
        m["cutoff_across"], g["cutoff_across"], rtol=1e-4
    ), m


def test_cutoff_length_ignores_box_but_half_width_does_not():
    """Band flat per ln k from the box up to an exp(-k^2 l^2) cutoff, on cells of side 1 and
    0.5 at the same grid spacing: the cutoff length is 2 l / sqrt 3 in both, while the
    autocorrelation half-width and the Taylor length shrink with the box."""
    from structure_kr import axis_cut, cut_measures, gradient_lengths, physical_acf

    l = 0.02
    S = lambda kx, ky: (
        np.exp(-(kx**2 + ky**2) * l**2) / np.maximum(kx**2 + ky**2, 1e-12)
    )
    out = {}
    for L, N in ((1.0, 256), (0.5, 128)):
        lats = _deformed_cells(L)
        fields = np.array([_field_from_spectrum(A, N, S) for A in lats])
        g = gradient_lengths(fields, lats)
        cen, C = physical_acf(fields, np.array(lats), half_width=0.6 * L, h=0.8 / 256)
        g["y_half"] = cut_measures(*axis_cut(cen, C, 1), 0.5 * L)["half"]
        out[L] = g
    for L in out:
        assert abs(out[L]["cutoff_across"] / (2 * l / np.sqrt(3)) - 1) < 0.02, out[L]
    assert abs(out[1.0]["cutoff_across"] / out[0.5]["cutoff_across"] - 1) < 0.01
    assert out[1.0]["y_half"] / out[0.5]["y_half"] > 1.2, out
    assert out[1.0]["taylor_across"] / out[0.5]["taylor_across"] > 1.1, out


def _random_band_frames(L, N, l, a, n_frames, seed):
    """Random-phase fields on golden cells across one KR period, with mode powers flat per
    ln k up to a Gaussian cutoff of length l and elongated along x by a factor a."""
    rng = np.random.default_rng(seed)
    n = np.fft.fftfreq(N, 1.0 / N)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    dealias = (np.abs(n1) <= N // 3) & (np.abs(n2) <= N // 3)
    fields, lats = [], []
    for e in np.linspace(0, EPS_P, n_frames, endpoint=False):
        A = np.diag([np.exp(e), np.exp(-e)]) @ (L * R)
        B = 2 * np.pi * np.linalg.inv(A).T
        kx, ky = B[0, 0] * n1 + B[0, 1] * n2, B[1, 0] * n1 + B[1, 1] * n2
        q2 = (a * kx) ** 2 + ky**2
        S = np.where(q2 > 0, np.exp(-q2 * l**2) / np.maximum(q2, 1e-12), 0) * dealias
        g = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        fields.append(np.fft.ifft2(np.sqrt(S) * g).real)
        lats.append(A)
    return np.array(fields), lats


def test_across_spectrum_background_and_injected_stripe():
    """A featureless band elongated along x: the across direction is y and nothing stands
    above the background. A stripe of 0.1 x the field rms with spacing 0.1 is found at
    that spacing."""
    from fourier_kr import across_analysis

    runs = [_random_band_frames(1.0, 128, 0.03, 3.0, 20, s) for s in range(3)]
    r = across_analysis(runs, blocks=5)
    assert abs(r["theta_across_deg"] - 90) < 5, r["theta_across_deg"]
    assert r["peaks"] == [], r["peaks"]
    r = across_analysis(runs, blocks=5, inject=(0.1, 0.1))
    assert len(r["peaks"]) == 1 and abs(r["peaks"][0]["wavelength"] / 0.1 - 1) < 0.05, (
        r["peaks"]
    )


def test_phase_resolved_repeat_bins_and_mirror():
    """Frames over two golden KR periods split evenly into phase bins; the cell's lattice
    vector a1 starts at -31.7 deg and turns toward x; mirroring the cell (y -> -y) flips
    the sign of every angle and keeps the correlation values."""
    from structure_kr import phase_resolved_repeat

    N = 64
    xi = np.arange(N) / N
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    strain = np.linspace(0, 2 * EPS_P, 40, endpoint=False)
    k = 2 * np.pi * 3 * np.array([np.cos(1.2), np.sin(1.2)])
    fields, lats = [], []
    for e in strain:
        s = e % EPS_P
        A = np.diag([np.exp(s), np.exp(-s)]) @ R
        m = np.round(A.T @ k / (2 * np.pi))
        fields.append(np.cos(2 * np.pi * (m[0] * X1 + m[1] * X2)))
        lats.append(A)
    fields, lats = np.array(fields), np.array(lats)
    rows, _ = phase_resolved_repeat(fields, lats, strain, EPS_P, bins=5)
    assert [r["n_frames"] for r in rows] == [8] * 5
    a1 = [r["a1_theta_deg"] for r in rows]
    assert -31.8 < a1[0] < -25 and all(np.diff(a1) > 0) and a1[-1] < 0, a1
    mirrored, _ = phase_resolved_repeat(
        fields, np.array([np.diag([1, -1]) @ A for A in lats]), strain, EPS_P, bins=5
    )
    for r, m in zip(rows, mirrored):
        for key in ("ridge_theta_deg", "a1_theta_deg", "a2_theta_deg"):
            assert np.isclose(m[key], -r[key]) or abs(abs(r[key]) - 90) < 2, (key, r, m)
        assert np.isclose(m["ridge_C"], r["ridge_C"])


def test_run_time_estimates_measured_or_scaled():
    """Per-case run time from measured {N: s/step} or from ms/step at N = 128 scaled by
    N^2; the benchmark returns a positive cost for each grid size."""
    from sweep_kr import benchmark_step_costs, estimated_costs

    plans = {
        0: ({"N": 128}, {"n_steps": 1000}, None),
        1: ({"N": 256}, {"n_steps": 1000}, None),
        2: ({"N": 256}, None, "no steady state"),
    }
    assert estimated_costs(plans, {128: 0.1, 256: 0.5}) == {0: 100.0, 1: 500.0, 2: 0.0}
    assert estimated_costs(plans, 20.0) == {0: 20.0, 1: 80.0, 2: 0.0}
    cases = [dict(model="oldroyd_b", eps_dot=0.5, N=n) for n in (16, 32)]
    cost = benchmark_step_costs(cases, 2)
    assert set(cost) == {16, 32} and all(v > 0 for v in cost.values())


def test_set_params_at_period_boundary():
    """set_params changes the rates and the time step only at a KR period boundary, and
    refuses to change the grid, cell or state variable."""
    p = Params("oldroyd_b", eps_dot=0.5, N=16)
    s = KRSolver(p, steps_per_period=20)
    s.set_params(dataclasses.replace(p, eps_dot=1.0, lam=0.5), steps_per_period=40)
    assert s.p.lam == 0.5 and s.steps_per_period == 40
    assert np.isclose(s.dt, EPS_P / 1.0 / 40)
    s.step()
    for bad, err in ((dataclasses.replace(s.p, lam=2.0), RuntimeError),):
        try:
            s.set_params(bad)
            raise AssertionError("expected an error mid-period")
        except err:
            pass
    for _ in range(39):
        s.step()  # back on a boundary (remapped)
    try:
        s.set_params(dataclasses.replace(s.p, N=32))
        raise AssertionError("expected a structural-change error")
    except ValueError as e:
        assert "N" in str(e)


def test_schedule_runs_every_stage_and_dry_run():
    """A two-stage schedule runs both stages (re-planned steps, stage columns, series
    stage markers); the dry run plans it with a measured step cost; unschedulable fields
    are refused."""
    import contextlib
    import csv
    import io
    import os
    import tempfile

    from sweep_kr import main

    base = [
        "--model", "oldroyd_b", "--set", "N=16", "--set", "eps_dot=0.3",
        "--schedule", "lam=1,0.5", "--remaps-per-stage", "1",
    ]  # fmt: skip
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sch")
        with quiet():
            main(base + ["--out", out])
        (row,) = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
        assert row["stage_reached"] == "1", row
        assert (
            int(row["steps_per_period_s0"]) > 0 and int(row["steps_per_period_s1"]) > 0
        )
        ser = np.load(os.path.join(out, "series", "case_0000.npz"))
        assert set(np.unique(ser["stage"])) == {0, 1} and len(ser["t_switch"]) == 2
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(base + ["--out", os.path.join(tmp, "dry"), "--dry-run"])
        assert "measured: N = 16" in buf.getvalue(), buf.getvalue()
    try:
        with quiet():
            main(
                base[:6]
                + ["--schedule", "N=16,32", "--remaps-per-stage", "1", "--dry-run"]
            )
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "cannot change" in str(e)


def test_stretch_history_from_saved_fields():
    """viz_sweep's stretch measure is the rms of lambda - <lambda> of each saved snapshot
    (finite extensibility inverted), with one point per snapshot."""
    import contextlib
    import io
    import os
    import shutil
    import tempfile

    from fourier_kr import stretch_field
    from sweep_kr import main
    from viz_sweep import MEASURES, case_curve, case_params, load_sweep, movie

    assert {"stress", "velocity", "stretch"} <= set(MEASURES)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "st")
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                ["--model", "rolie_poly", "--set", "N=16", "--set", "eps_dot=20"]
                + ["--set", "tau_R=0.05", "--set", "L_max=5", "--strain", "1"]
                + ["--save-fields", "--out", out]
            )
        config, rows, series = load_sweep(out)
        t, y, sigma_lam, _ = case_curve(out, config, rows[0], series, "stretch")
        d = dict(np.load(os.path.join(out, "fields", "case_0000.npz")))
        lam = stretch_field(d, case_params(config, rows[0]))
        want = np.sqrt(
            np.mean((lam - lam.mean(axis=(1, 2), keepdims=True)) ** 2, axis=(1, 2))
        )
        assert np.allclose(t, d["t"]) and np.allclose(y, want) and np.all(y > 0)
        assert np.isfinite(sigma_lam)
        if shutil.which("ffmpeg"):  # the stretch field as the movie quantity
            with contextlib.redirect_stdout(io.StringIO()):
                movie(out, [0], "stretch", 12, "all")
            assert os.path.exists(
                os.path.join(out, "figures", "movie_stretch_all_cases_0.mp4")
            )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
