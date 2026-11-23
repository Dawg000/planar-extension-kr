"""2D pseudospectral Stokes (Re = 0) solver for planar extensional flow of
polymer models with Kraynik-Reinelt (KR) periodic boundary conditions.

Formulation
-----------
  base flow  u = L x + v,   L = eps_dot * diag(1, -1)    (x = extension axis)
  lattice    x = A(t) xi,   A(t) = Lambda(t) * Lbox * R,  xi in [0, 1)^2
  Fields are Fourier series in xi; mode n has physical wavevector
      k = 2 pi A(t)^-T n = (2 pi / Lbox) diag(e^{-eps_dot t}, e^{eps_dot t}) R n.
  In xi-coordinates the base-flow advection L x . grad cancels exactly.
  Every KR period t_p = eps_p / eps_dot the lattice maps onto itself
  (Lambda(t_p) R = R M) and mode n is relabelled to M^-T n. This is exact;
  modes that land outside the dealiased box are dropped and the dropped
  fraction is recorded in `remap_loss`.
  Params.kr_lattice picks the integer matrix M (see LATTICES). All are square
  cells of side Lbox (M symmetric, so R is a rotation) that differ in their
  angle to the extension axis and their period eps_p = ln(largest eigenvalue).

Equations
---------
  0 = -grad p + eta_s lap v + div tau,   div v = 0,   tau = G (C - I)
  dC/dt + v . grad C = kap C + C kap^T + relax(C) + D lap C
  kap_ij = d u_i / d x_j = L_ij + d v_i / d x_j
  (Johnson-Segalman replaces kap by W + a D_rate, the Gordon-Schowalter form.)
C carries xx, xy, yy, zz. With in-plane flow, zz only relaxes; it matters
only for Rolie-Poly, through tr C.

Finite extensibility for Rolie-Poly (Params.L_max > 0)
-------------------------------------------------------
With stretch lam = sqrt(tr C / 3) and the normalized Warner spring factor
  f = (1 - 1/L_max^2) / (1 - lam^2/L_max^2)          (f = 1 at lam = 1),
the relaxation and the stress become
  relax(C) = -(f C - I)/tau_d - (2/tau_R) f (1 - 1/lam) (C + beta lam^(2 delta) (f C - I))
  tau      = G (f C - I)
so f C = I is the zero-stress state throughout. L_max = 0 is plain Rolie-Poly.
The run stops with StretchLimitError if lam reaches L_max.

Log-conformation (Params.log_conformation = True)
-------------------------------------------------
The state is Psi = log C (Fattal & Kupferman 2004), which keeps C positive
definite at high Wi:
  dPsi/dt + v . grad Psi = (Omega Psi - Psi Omega) + 2 B + relax(C) C^-1 + D lap Psi
with kap_eff = Omega + B + N C^-1 decomposed in the eigenbasis of Psi (see
log_source). Stress diffusion then acts on Psi rather than on C.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

PHI = (1 + 5**0.5) / 2
EPS_P = 2 * np.log(PHI)  # Hencky strain per KR period (~0.9624)
THETA = np.arctan(1 / PHI)  # lattice angle to the extension axis (~31.72 deg)
M = np.array([[2, 1], [1, 1]])
M_INV_T = np.array([[1, -1], [-1, 2]])
R = np.array(
    [[np.cos(THETA), np.sin(THETA)], [-np.sin(THETA), np.cos(THETA)]]
)  # columns = unit lattice vectors

# Admissible KR matrices: symmetric integer, det 1, trace > 2. "golden_mirror" is the mirror
# image of "golden" (same period, lattice angle 58.28 deg); "silver" has angle 22.50 deg and
# period ln(3 + 2 sqrt 2) = 1.763.
LATTICES = {
    "golden": ((2, 1), (1, 1)),
    "golden_mirror": ((1, 1), (1, 2)),
    "silver": ((5, 2), (2, 1)),
}


class KRLattice(NamedTuple):
    name: str
    M: np.ndarray
    M_inv_T: np.ndarray
    R: np.ndarray  # rows: eigenvectors of M; columns: unit lattice vectors at tau = 0
    eps_p: float  # Hencky strain per KR period
    theta: float  # angle of the stretching eigenvector to the extension axis
    cycle: float  # strain after which the cell repeats up to the mirror y -> -y


def kr_lattice(name="golden"):
    """Geometry of a named KR lattice. Solves Lambda(eps_p) R = R M for symmetric M."""
    if name not in LATTICES:
        raise ValueError(f"unknown kr_lattice {name!r}; choose from {tuple(LATTICES)}")
    Mi = np.array(LATTICES[name])
    w, V = np.linalg.eigh(Mi.astype(float))
    v = V[:, 1] * np.sign(V[0, 1])  # stretching eigenvector, x component > 0
    theta = float(np.arctan2(v[1], v[0]))
    R_ = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    eps_p = float(np.log(w[1]))
    assert np.allclose(np.diag([w[1], 1 / w[1]]) @ R_, R_ @ Mi, atol=1e-10)
    # half-period mirror: is F Lambda(eps_p / 2) R the same lattice as R, i.e. integer in R?
    h = np.exp(eps_p / 2)
    U = R_.T @ np.diag([1, -1]) @ np.diag([h, 1 / h]) @ R_
    mirror = np.allclose(U, np.round(U), atol=1e-9)
    Minv_T = np.round(np.linalg.inv(Mi).T).astype(int)
    return KRLattice(name, Mi, Minv_T, R_, eps_p, theta, eps_p / 2 if mirror else eps_p)


XX, XY, YY, ZZ = range(4)
MODELS = ("oldroyd_b", "giesekus", "johnson_segalman", "rolie_poly")


@dataclass
class Params:
    model: str
    eps_dot: float  # extension rate
    lam: float = 1.0  # relaxation time (reptation time tau_d for Rolie-Poly)
    G: float = 1.0  # modulus
    eta_s: float = 0.1  # solvent viscosity (> 0 required at Re = 0)
    alpha: float = 0.0  # Giesekus mobility
    a: float = 1.0  # Johnson-Segalman slip parameter
    tau_R: float = 0.1  # Rolie-Poly Rouse time
    beta: float = 0.0  # Rolie-Poly CCR coefficient
    delta: float = -0.5  # Rolie-Poly CCR exponent
    D: float = 0.0  # stress diffusivity
    Lbox: float = 1.0  # lattice cell side length
    N: int = 64  # grid points per lattice direction
    log_conformation: bool = False  # evolve Psi = log C instead of C
    # Stochastic forcing: every step adds forcing_amp * sqrt(dt) * xi to the xx, xy, yy
    # components of the state, with xi a Gaussian random field of unit rms per component
    # and zero cell average. forcing_amp is in state units per sqrt(time), so the injected
    # noise does not depend on the time step. 0 turns forcing off.
    forcing_amp: float = 0.0
    forcing_length: float = (
        0.0  # Gaussian smoothing width of xi; 0 = lattice modes <= nmax
    )
    forcing_nmax: int = 4  # highest lattice mode forced when forcing_length = 0
    forcing_seed: int = 1  # independent of the initial-noise seed
    L_max: float = 0.0  # Rolie-Poly maximum stretch (finite extensibility); 0 = off
    kr_lattice: str = "golden"  # KR matrix, a key of LATTICES


def upper_convected(C, kap):
    """kap C + C kap^T for in-plane kap = (kxx, kxy, kyx, kyy); zz part is 0."""
    kxx, kxy, kyx, kyy = kap
    return np.stack(
        [
            2 * (kxx * C[XX] + kxy * C[XY]),
            kxx * C[XY] + kxy * C[YY] + kyx * C[XX] + kyy * C[XY],
            2 * (kyx * C[XY] + kyy * C[YY]),
            np.zeros_like(C[ZZ]),
        ]
    )


def effective_gradient(kap, p):
    """Velocity gradient seen by the convected derivative: kap itself, or
    W + a D (Gordon-Schowalter) for Johnson-Segalman."""
    if p.model != "johnson_segalman":
        return kap
    kxx, kxy, kyx, kyy = kap
    return (
        p.a * kxx,
        0.5 * ((1 + p.a) * kxy + (p.a - 1) * kyx),
        0.5 * ((1 + p.a) * kyx + (p.a - 1) * kxy),
        p.a * kyy,
    )


class StretchLimitError(FloatingPointError):
    """The chain stretch reached L_max (the spring factor would become singular)."""


def fene_factor(C, p):
    """Normalized Warner spring factor f(lam) for Rolie-Poly with L_max > 0, else 1."""
    if not p.L_max:
        return 1.0
    if p.model != "rolie_poly":
        raise ValueError(
            "L_max (finite extensibility) is only implemented for rolie_poly"
        )
    if p.L_max <= 1:
        raise ValueError("L_max must be > 1 (or 0 to turn finite extensibility off)")
    L2 = p.L_max**2
    return (1 - 1 / L2) / (1 - (C[XX] + C[YY] + C[ZZ]) / (3 * L2))


def stress_conformation(C, p):
    """tau / G + I: the tensor whose divergence drives the flow (f C, or C)."""
    return fene_factor(C, p) * C if p.L_max else C


def relaxation(C, p):
    """Relaxation part of dC/dt, a function of C alone (so it commutes with C)."""
    if p.L_max and p.model != "rolie_poly":
        raise ValueError(
            "L_max (finite extensibility) is only implemented for rolie_poly"
        )
    I = np.array([1.0, 0.0, 1.0, 1.0]).reshape((4,) + (1,) * (np.ndim(C) - 1))
    B = C - I
    if p.model in ("oldroyd_b", "johnson_segalman"):
        return -B / p.lam
    if p.model == "giesekus":
        B2 = np.stack(
            [
                B[XX] ** 2 + B[XY] ** 2,
                B[XY] * (B[XX] + B[YY]),
                B[XY] ** 2 + B[YY] ** 2,
                B[ZZ] ** 2,
            ]
        )
        return -(B + p.alpha * B2) / p.lam
    if p.model == "rolie_poly":
        trC = C[XX] + C[YY] + C[ZZ]
        stretch = 1 - np.sqrt(3 / trC)
        reg = np.tanh(
            100 * p.lam / (3 * p.tau_R) * stretch
        )  # Regularization term used in our work
        if p.L_max:  # finite extensibility: relax f C - I, retraction sped up by f
            f = fene_factor(C, p)
            S = f * C - I
            return -S / p.lam - (2 / p.tau_R) * f * stretch * (
                C + p.beta * reg * (trC / 3) ** p.delta * S
            )
        return -B / p.lam - (2 / p.tau_R) * stretch * (
            C + p.beta * reg * (trC / 3) ** p.delta * B
        )
    raise ValueError(f"unknown model {p.model!r}; choose from {MODELS}")


def model_source(C, kap, p):
    """Right-hand side of dC/dt (excluding advection and diffusion)."""
    return upper_convected(C, effective_gradient(kap, p)) + relaxation(C, p)


# --- log-conformation (Fattal & Kupferman 2004) ----------------------------------
def _sym2(xx, xy, yy):
    """Stack symmetric 2x2 components into an array of shape (..., 2, 2)."""
    xx, xy, yy = np.broadcast_arrays(xx, xy, yy)
    return np.stack([np.stack([xx, xy], -1), np.stack([xy, yy], -1)], -2)


def _eigh2(xx, xy, yy):
    """Closed-form eigendecomposition of symmetric 2x2 fields: eigenvalues ascending
    (shape (..., 2)) and eigenvectors as columns of Q (shape (..., 2, 2))."""
    xx, xy, yy = np.broadcast_arrays(xx, xy, yy)
    m, h = 0.5 * (xx + yy), 0.5 * (xx - yy)
    r = np.hypot(h, xy)
    theta = 0.5 * np.arctan2(xy, h)  # direction of the larger eigenvalue
    c, s = np.cos(theta), np.sin(theta)
    Q = np.stack([np.stack([-s, c], -1), np.stack([c, s], -1)], -2)
    return np.stack([m - r, m + r], -1), Q


def log_to_conformation(Psi):
    """C = exp(Psi) for components (xx, xy, yy, zz)."""
    psi, Q = _eigh2(Psi[XX], Psi[XY], Psi[YY])
    C2 = (Q * np.exp(psi)[..., None, :]) @ np.swapaxes(Q, -1, -2)
    return np.stack([C2[..., 0, 0], C2[..., 0, 1], C2[..., 1, 1], np.exp(Psi[ZZ])])


def conformation_to_log(C):
    """Psi = log(C) for components (xx, xy, yy, zz); C must be positive definite."""
    lam, Q = _eigh2(C[XX], C[XY], C[YY])
    P2 = (Q * np.log(lam)[..., None, :]) @ np.swapaxes(Q, -1, -2)
    return np.stack([P2[..., 0, 0], P2[..., 0, 1], P2[..., 1, 1], np.log(C[ZZ])])


def log_source(Psi, kap, p):
    """Right-hand side of dPsi/dt, Psi = log C (excluding advection and diffusion).

    In the eigenbasis of Psi (eigenvalues psi_i, lambda_i = exp(psi_i)), with
    K = Q^T kap_eff Q and G = Q^T relax(C) Q (diagonal, since relax commutes with C):
        dPsi_ii = 2 K_ii + G_ii / lambda_i
        dPsi_12 = (lambda_2 K_12 + lambda_1 K_21) (psi_2 - psi_1) / (lambda_2 - lambda_1)
    The last factor is evaluated as exp(-psi_1) x / expm1(x), x = psi_2 - psi_1,
    which stays finite when the eigenvalues coincide.
    """
    psi, Q = _eigh2(Psi[XX], Psi[XY], Psi[YY])
    lam = np.exp(psi)
    QT = np.swapaxes(Q, -1, -2)
    C2 = (Q * lam[..., None, :]) @ QT
    C = np.stack([C2[..., 0, 0], C2[..., 0, 1], C2[..., 1, 1], np.exp(Psi[ZZ])])

    kxx, kxy, kyx, kyy = np.broadcast_arrays(*effective_gradient(kap, p), Psi[XX])[:4]
    K = QT @ np.stack([np.stack([kxx, kxy], -1), np.stack([kyx, kyy], -1)], -2) @ Q
    g = relaxation(C, p)
    G = QT @ _sym2(g[XX], g[XY], g[YY]) @ Q

    x = psi[..., 1] - psi[..., 0]
    small = np.abs(x) < 1e-8
    q = np.exp(-psi[..., 0]) * np.where(
        small, 1.0, x / np.where(small, 1.0, np.expm1(x))
    )
    off = (lam[..., 1] * K[..., 0, 1] + lam[..., 0] * K[..., 1, 0]) * q

    dt = np.zeros_like(K)
    dt[..., 0, 0] = 2 * K[..., 0, 0] + G[..., 0, 0] / lam[..., 0]
    dt[..., 1, 1] = 2 * K[..., 1, 1] + G[..., 1, 1] / lam[..., 1]
    dt[..., 0, 1] = dt[..., 1, 0] = off
    d = Q @ dt @ QT
    return np.stack([d[..., 0, 0], d[..., 0, 1], d[..., 1, 1], g[ZZ] / C[ZZ]])


def homogeneous_rhs(C, p):
    """0D constitutive ODE in steady planar extension (reference for tests)."""
    return model_source(np.asarray(C), (p.eps_dot, 0.0, 0.0, -p.eps_dot), p)


def stokes_velocity_hat(Ch, kx, ky, G, eta_s):
    """Perturbation velocity from the spectral conformation (xx, xy, yy, ...):
    eta_s |k|^2 v = P(k) (i k . tau), tau = G (C - I); the k = 0 mode is zero."""
    fx = 1j * G * (kx * Ch[XX] + ky * Ch[XY])
    fy = 1j * G * (kx * Ch[XY] + ky * Ch[YY])
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    kf = (kx * fx + ky * fy) / k2
    vx = (fx - kx * kf) / (eta_s * k2)
    vy = (fy - ky * kf) / (eta_s * k2)
    vx[0, 0] = vy[0, 0] = 0
    return vx, vy


class KRSolver:
    def __init__(self, p: Params, steps_per_period: int):
        if p.model not in MODELS:
            raise ValueError(f"unknown model {p.model!r}; choose from {MODELS}")
        self.p = p
        self.lat = kr_lattice(p.kr_lattice)
        R_, N = self.lat.R, p.N
        n = np.fft.fftfreq(N, 1.0 / N).astype(int)
        self.n1, self.n2 = np.meshgrid(n, n, indexing="ij")
        cut = N // 3  # 2/3-rule dealiasing
        self.keep = (np.abs(self.n1) <= cut) & (np.abs(self.n2) <= cut)

        s = 2 * np.pi / p.Lbox  # wavevector components at tau = 0
        self.kx0 = s * (R_[0, 0] * self.n1 + R_[0, 1] * self.n2)
        self.ky0 = s * (R_[1, 0] * self.n1 + R_[1, 1] * self.n2)

        Mt = self.lat.M_inv_T
        m1 = Mt[0, 0] * self.n1 + Mt[0, 1] * self.n2
        m2 = Mt[1, 0] * self.n1 + Mt[1, 1] * self.n2
        self._src = self.keep & (np.abs(m1) <= cut) & (np.abs(m2) <= cut)
        self._dst = (m1[self._src] % N, m2[self._src] % N)

        if p.forcing_amp < 0 or p.forcing_length < 0:
            raise ValueError("forcing_amp and forcing_length must be >= 0")
        self._forcing_rng = (
            np.random.default_rng(p.forcing_seed) if p.forcing_amp else None
        )

        self.t_p = self.lat.eps_p / p.eps_dot
        self.steps_per_period = steps_per_period
        self.dt = self.t_p / steps_per_period
        self.t = 0.0  # total time
        self.tau = 0.0  # time since the last remap
        self._step_in_period = 0
        self.remap_loss = []  # fraction of fluctuation energy dropped per remap

        # Spectral state: C, or Psi = log C when p.log_conformation is set.
        self.Ch = np.zeros((4, N, N), complex)
        if not p.log_conformation:
            self.Ch[[XX, YY, ZZ], 0, 0] = N * N  # C = I (Psi = 0 otherwise)

    # --- geometry ---------------------------------------------------------
    def wavevectors(self, tau):
        ed = self.p.eps_dot
        return self.kx0 * np.exp(-ed * tau), self.ky0 * np.exp(ed * tau)

    def lattice(self, tau=None):
        """Current lattice matrix A (columns = lattice vectors)."""
        tau = self.tau if tau is None else tau
        ed = self.p.eps_dot
        return np.diag([np.exp(ed * tau), np.exp(-ed * tau)]) @ (
            self.p.Lbox * self.lat.R
        )

    def grid_xy(self):
        """Physical coordinates (X, Y) of the grid nodes at the current time."""
        xi = np.arange(self.p.N) / self.p.N
        X1, X2 = np.meshgrid(xi, xi, indexing="ij")
        A = self.lattice()
        return A[0, 0] * X1 + A[0, 1] * X2, A[1, 0] * X1 + A[1, 1] * X2

    # --- fields -----------------------------------------------------------
    def set_C(self, C):
        self.Ch = np.fft.fft2(conformation_to_log(C) if self.p.log_conformation else C)
        self.Ch[:, ~self.keep] = 0

    def C(self):
        S = np.fft.ifft2(self.Ch).real
        return log_to_conformation(S) if self.p.log_conformation else S

    def mean_C(self):
        if self.p.log_conformation:
            return self.C().mean(axis=(1, 2))
        return self.Ch[:, 0, 0].real / self.p.N**2

    def velocity_hat(self, Ch, kx, ky):
        return stokes_velocity_hat(Ch, kx, ky, self.p.G, self.p.eta_s)

    def stress_hat(self, S, Ch):
        """Spectral tau / G + I from the real-space state S (C, or Psi in log mode) and its
        transform Ch. Raises StretchLimitError if the stretch has reached L_max."""
        p = self.p
        if not (p.log_conformation or p.L_max):
            return Ch
        C = log_to_conformation(S) if p.log_conformation else S
        if p.L_max:
            lam2 = np.max(C[XX] + C[YY] + C[ZZ]) / 3
            if not lam2 < p.L_max**2:
                raise StretchLimitError(
                    f"chain stretch reached L_max = {p.L_max} at t = {self.t:.6g} "
                    f"(max lambda = {np.sqrt(lam2):.6g}); reduce the time step"
                )
        return np.fft.fft2(stress_conformation(C, p))

    # --- time stepping ------------------------------------------------------
    def rhs(self, Ch, tau):
        ifft = lambda f: np.fft.ifft2(f).real
        kx, ky = self.wavevectors(tau)
        S = ifft(Ch)  # C, or Psi in log mode
        log = self.p.log_conformation
        vxh, vyh = self.velocity_hat(self.stress_hat(S, Ch), kx, ky)
        vx, vy = ifft(vxh), ifft(vyh)
        ed = self.p.eps_dot
        kap = (
            ed + ifft(1j * kx * vxh),
            ifft(1j * ky * vxh),
            ifft(1j * kx * vyh),
            -ed + ifft(1j * ky * vyh),
        )
        adv = vx * ifft(1j * kx * Ch) + vy * ifft(1j * ky * Ch)
        source = log_source(S, kap, self.p) if log else model_source(S, kap, self.p)
        out = np.fft.fft2(source - adv)
        out[:, ~self.keep] = 0
        return out

    def _diffuse(self, a, b):
        """Exact stress-diffusion factor exp(-D int_a^b |k(s)|^2 ds)."""
        if self.p.D == 0:
            return
        ed = self.p.eps_dot
        integral = (
            self.kx0**2 * (np.exp(-2 * ed * a) - np.exp(-2 * ed * b))
            + self.ky0**2 * (np.exp(2 * ed * b) - np.exp(2 * ed * a))
        ) / (2 * ed)
        self.Ch *= np.exp(-self.p.D * integral)

    def step(self):
        """One Strang step (half diffusion, SSP-RK3, half diffusion), the remap if the period
        is complete, then the stochastic forcing increment if forcing is on."""
        dt, tau = self.dt, self.tau
        self._diffuse(tau, tau + dt / 2)
        C0 = self.Ch
        C1 = C0 + dt * self.rhs(C0, tau)
        C2 = 0.75 * C0 + 0.25 * (C1 + dt * self.rhs(C1, tau + dt))
        self.Ch = C0 / 3 + (2 / 3) * (C2 + dt * self.rhs(C2, tau + dt / 2))
        self._diffuse(tau + dt / 2, tau + dt)

        self.t += dt
        self.tau += dt
        self._step_in_period += 1
        if self._step_in_period == self.steps_per_period:
            self._remap()
        if self._forcing_rng is not None and self.p.forcing_amp:
            self._force()

    STRUCTURAL = ("model", "N", "Lbox", "kr_lattice", "log_conformation")

    def set_params(self, p, steps_per_period=None):
        """Change the parameters at a KR period boundary (before the first step or right
        after a remap) and optionally the number of steps per period; the state is kept.
        The fields in STRUCTURAL define the grid, cell and state variable and cannot change.
        A forcing stream turned on here starts from forcing_seed; one already running
        continues (a later change of forcing_seed has no effect)."""
        changed = [f for f in self.STRUCTURAL if getattr(p, f) != getattr(self.p, f)]
        if changed:
            raise ValueError(f"cannot change {changed} during a run")
        if self._step_in_period:
            raise RuntimeError("parameters can only change at a KR period boundary")
        if p.forcing_amp < 0 or p.forcing_length < 0:
            raise ValueError("forcing_amp and forcing_length must be >= 0")
        if p.forcing_amp and self._forcing_rng is None:
            self._forcing_rng = np.random.default_rng(p.forcing_seed)
        self.p = p
        if steps_per_period is not None:
            self.steps_per_period = steps_per_period
        self.t_p = self.lat.eps_p / p.eps_dot
        self.dt = self.t_p / self.steps_per_period

    def _force(self):
        """Euler-Maruyama increment forcing_amp * sqrt(dt) * xi (additive noise)."""
        p = self.p
        noise, filt = self._noise_field(
            self._forcing_rng, p.forcing_nmax, p.forcing_length or None
        )
        expected_rms = (
            np.sqrt(np.sum(filt**2)) / p.N
        )  # per component, over realizations
        self.Ch[:3] += (p.forcing_amp * np.sqrt(self.dt) / expected_rms) * noise

    def _remap(self):
        old = self.Ch
        fluct = self.keep.copy()
        fluct[0, 0] = False
        total = np.sum(np.abs(old[:, fluct]) ** 2)
        lost = np.sum(np.abs(old[:, self.keep & ~self._src]) ** 2)
        self.remap_loss.append(lost / total if total > 0 else 0.0)

        self.Ch = np.zeros_like(old)
        self.Ch[:, self._dst[0], self._dst[1]] = old[:, self._src]
        self.tau = 0.0
        self._step_in_period = 0

    def perturb(self, amp, nmax=4, seed=0, smooth_length=None):
        """Add random real noise of peak amplitude amp to the xx, xy, yy components of
        the state (C, or Psi = log C in log mode).

        smooth_length=None: white noise restricted to lattice modes |n| <= nmax.
        smooth_length=l:    white noise over all resolved modes, smoothed by a Gaussian of
                            width l in physical space (multiplier exp(-|k|^2 l^2 / 2) with k
                            the current physical wavevector); nmax is not used.
        """
        noise, _ = self._noise_field(np.random.default_rng(seed), nmax, smooth_length)
        noise *= amp / np.abs(np.fft.ifft2(noise).real).max()
        self.Ch[:3] += noise

    def _noise_field(self, rng, nmax, smooth_length):
        """Spectral white noise (3 components) passed through a filter; returns the noise
        and the real filter (zero at k = 0 and outside the dealiased set)."""
        N = self.p.N
        noise = np.fft.fft2(rng.standard_normal((3, N, N)))
        if smooth_length is None:
            filt = ((np.abs(self.n1) <= nmax) & (np.abs(self.n2) <= nmax)).astype(float)
        else:
            if smooth_length <= 0:
                raise ValueError("smooth_length must be positive")
            kx, ky = self.wavevectors(self.tau)
            filt = np.exp(-0.5 * (kx**2 + ky**2) * smooth_length**2) * self.keep
        filt[0, 0] = 0
        if filt.max() < 1e-12:
            raise ValueError(
                "smoothing removed all resolved modes; smooth_length is too "
                "large for the cell"
            )
        return noise * filt, filt


if __name__ == "__main__":
    # Demo: Rolie-Poly startup of planar extension with small noise.
    p = Params(
        "rolie_poly",
        eps_dot=0.5,
        lam=1.0,
        tau_R=0.1,
        beta=0.0,
        eta_s=0.05,
        D=1e-4,
        N=64,
    )
    s = KRSolver(p, steps_per_period=400)
    s.perturb(1e-3)
    for period in range(5):
        for _ in range(s.steps_per_period):
            s.step()
        C = s.C()
        fluct = np.sqrt(np.mean((C[:3] - C[:3].mean(axis=(1, 2), keepdims=True)) ** 2))
        print(
            f"strain {s.t * p.eps_dot:5.2f}  <Cxx>={s.mean_C()[XX]:.4f}  "
            f"<Cyy>={s.mean_C()[YY]:.4f}  rms fluct={fluct:.3e}  "
            f"remap loss={s.remap_loss[-1]:.1e}"
        )
