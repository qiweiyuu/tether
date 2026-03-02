"""curved_pde_sim.py  —
====================================================
Finite volume solver for coupled bulk and surface phase fields

* φ‑dynamics use cell‑volume and face‑area fractions near the curved wall so
  mass is conserved even in cut cells.
* ψ evolves along the wall with a non‑uniform, arc‑length‑aware Laplacian
  (segment lengths `seg_left/seg_right/seg_center` match the notebook).
* The Robin wetting boundary couples φ ↔ ψ exactly as in the notebook via
  ghost‑cell gradients.

* in this simulation, we only impose the tether Dirichlet boundary condition on one side, so that the chemical potentials do not have to be equal

Quick start
-----------
```python
import jax.numpy as jnp
from curved_pde_sim import PhaseFieldSimulator

sim = PhaseFieldSimulator(nx=96, nz=48)      # tune params as you like
phi0 = jnp.zeros((sim.nx, sim.nz))            # initial bulk field
psi0 = jnp.full(sim.nx, 0.05)                 # initial wall field
history = sim.run(phi0, psi0, num_epochs=10)  # 10×1000 steps
```
Each entry of **history** is a `(phi, psi)` snapshot you can plot.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm.auto import trange


# -----------------------------------------------------------------------------
#  Flory-Huggins stuff  - chemical potential, free energy, convex hull
# -----------------------------------------------------------------------------
def convex_hull_1d(points):
    points = points[np.argsort(points[:, 0])]  # Sort by x-coordinates

    def cross(o, a, b):
        """2D cross product of OA and OB vectors (z-component)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    return np.array(lower)


def calc_binodal_spinodal(psi_mesh, f_mesh):
    xdata, ydata = psi_mesh, f_mesh

    # find the lower hull of the free energy
    points = np.column_stack((xdata, ydata))
    lower_hull = convex_hull_1d(points)

    # find binodal by comparing  free energy
    xmesh = np.linspace(psi_mesh[0], psi_mesh[-1], 1000)
    yinterp = np.interp(xmesh, xdata, ydata)
    yinterp_hull = np.interp(xmesh, lower_hull[:, 0], lower_hull[:, 1])

    df_threshold = 1e-8
    binodal_region = yinterp_hull < yinterp - df_threshold
    # find first and last non-zero region
    binodal_region_idx = np.where(binodal_region)[0]
    if len(binodal_region_idx) == 0:
        return [np.nan, np.nan], [np.nan, np.nan]

    psi_binodal = [xmesh[binodal_region_idx[0]], xmesh[binodal_region_idx[-1]]]

    # to determine spinodal, we need to compute the inflection point
    free_energy_derivative2 = CubicSpline(
        xdata,
        savgol_filter(ydata, window_length=5, polyorder=3, deriv=2)
        / (xdata[1] - xdata[0]) ** 2,
    )
    xmesh = np.linspace(psi_binodal[0], psi_binodal[1], 100)[1:-1]
    spinodal_region = free_energy_derivative2(xmesh) < 0
    spinodal_region_idx = np.where(spinodal_region)[0]
    if len(spinodal_region_idx) == 0:
        psi_spinodal = [np.nan, np.nan]
    psi_spinodal = [xmesh[spinodal_region_idx[0]], xmesh[spinodal_region_idx[-1]]]
    return psi_binodal, psi_spinodal


phi_tol = 1e-8


def calc_f_entropy(psi):
    return psi * jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) + (1 - psi) * jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_f_uniform(psi, chi_psi):
    return calc_f_entropy(psi) + chi_psi * psi * (1 - psi)


def calc_mu_entropy(psi):
    return jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) - jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_mu_uniform(psi, chi_psi):
    return calc_mu_entropy(psi) + chi_psi * (1 - 2 * psi)


# -----------------------------------------------------------------------------
#  Geometry helpers  —  verbatim notebook logic, lightly tidied
# -----------------------------------------------------------------------------


def signed_distance_lower_wall(
    xg: jnp.ndarray,
    yg: jnp.ndarray,
    y_b: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    newton_iter: int = 15,
):
    """
    Compute a signed‑distance field ϕ for a domain bounded below by y_b(x).

    Parameters
    ----------
    xg, yg : 2‑D jax.numpy arrays
        Mesh‑grid of physical coordinates.
    y_b : callable
        y_b(x)  →  wall height at position x.
    newton_iter : int
        Iterations of Newton search used to find the closest point
        on the curved wall.  10–20 is plenty for smooth curves.

    Returns
    -------
    ϕ : 2‑D jax.numpy array
        Signed‑distance field.  ϕ > 0  ⇒  (x,y) is above / inside the fluid,
        ϕ < 0  ⇒  (x,y) is below / inside the solid.
    """

    # ── helper: distance from one point (xp,yp) to the curve y_b(x) ──
    def _point_signed_distance(xp, yp):
        """
        Find the closest point x* on y_b that minimises
        D²(x*) = (xp - x*)² + (yp - y_b(x*))²
        and return ±√D² with the correct sign.
        """

        # Newton iteration on g(x) = (x - xp) + y_b'(x) * (y_b(x) - yp) = 0
        def newton(xk, _):
            yk = y_b(xk)
            dydx = jax.grad(y_b)(xk)
            g = (xk - xp) + dydx * (yk - yp)
            dg = 1.0 + dydx**2  # derivative of g
            return xk - g / dg, None

        x_star, _ = lax.scan(newton, init=xp, xs=None, length=newton_iter)
        y_star = y_b(x_star)

        dx = xp - x_star
        dy = yp - y_star
        dist = jnp.sqrt(dx * dx + dy * dy)

        # Sign: positive if point is above the wall (yp > y_b(xp) is a cheap test)
        return jnp.where(yp >= y_b(xp), dist, -dist), jnp.stack((x_star, y_star))

    # vmap over rows and columns of the mesh grid
    signed_dist, normal_contact = vmap(vmap(_point_signed_distance, in_axes=(0, 0)))(
        xg, yg
    )
    return signed_dist, normal_contact


def cutcell_fractions(
    x_c: jnp.ndarray,
    z_c: jnp.ndarray,
    dx: float,
    dz: float,
    y_b: Callable[[jnp.ndarray], jnp.ndarray],
    mask_inside: jnp.ndarray = None,
):
    """Return *vol_frac* and *face_frac* (west/east/south/north) for FVM."""
    Xc, Zc = jnp.meshgrid(x_c, z_c, indexing="ij")
    yb_c = y_b(Xc)

    vol_tot = (Zc + dz / 2) ** 2 - (Zc - dz / 2) ** 2
    vol_frac = (
        jnp.clip((Zc + dz / 2) ** 2 - jnp.maximum(yb_c, Zc - dz / 2) ** 2, 0.0, vol_tot)
        / vol_tot
    )

    # (a) vertical faces: x = x_face, spanning the full Δz column
    x_face = jnp.concatenate([x_c - dx / 2, x_c[-1:] + dx / 2])
    Xf, Zf = jnp.meshgrid(x_face, z_c, indexing="ij")
    yb_f = y_b(Xf)
    h_face = jnp.clip((Zf + dz / 2) - jnp.maximum(yb_f, Zf - dz / 2), 0.0, dz)
    vert_frac = h_face / dz  # shape (nx+1, nz)

    # (b) horizontal faces: z = z_face, extend full Δx in x
    z_face = jnp.concatenate([z_c - dz / 2, z_c[-1:] + dz / 2])  # (nz+1,)
    Xh, Zh = jnp.meshgrid(x_c, z_face, indexing="ij")  # (nx, nz+1)
    yb_h = y_b(Xh)

    horiz_frac = (Zh >= yb_h).astype(Zc.dtype)  # 1 if open, 0 if blocked

    # 3. map the face arrays back onto each cell ------------------------------
    face_frac = {
        "west": vert_frac[:-1, :],  # west  face of cell (i,j)
        "east": vert_frac[1:, :],  # east  face of cell (i,j)
        "south": horiz_frac[:, :-1],  # south (lower‑z) face
        "north": horiz_frac[:, 1:],  # north (upper‑z) face
    }
    # set area to zero for cells outside the domain
    face_frac["west"] = jnp.where(
        mask_inside
        & np.pad(
            mask_inside[:-1, :],
            ((1, 0), (0, 0)),
            mode="constant",
            constant_values=False,
        ),
        face_frac["west"],
        0.0,
    )
    face_frac["east"] = jnp.where(
        mask_inside
        & np.pad(
            mask_inside[1:, :],
            ((0, 1), (0, 0)),
            mode="constant",
            constant_values=False,
        ),
        face_frac["east"],
        0.0,
    )
    face_frac["south"] = jnp.where(
        mask_inside
        & np.pad(
            mask_inside[:, :-1],
            ((0, 0), (1, 0)),
            mode="constant",
            constant_values=False,
        ),
        face_frac["south"],
        0.0,
    )
    face_frac["north"] = jnp.where(
        mask_inside
        & np.pad(
            mask_inside[1:, :],
            ((0, 1), (0, 0)),
            mode="constant",
            constant_values=False,
        ),
        face_frac["north"],
        0.0,
    )
    return vol_frac, face_frac


def plot_profile(phi, psi, Lx, Lr1, Lr2, x_center, r_center, axes=None):
    if axes is None:
        fig = plt.figure(figsize=(6, 5), tight_layout=True)

        # Create a GridSpec with 2 rows and 1 column
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[4, 1]
        )  # Adjust height_ratios to make axs[1] shorter

        # Create the first subplot
        ax1 = fig.add_subplot(gs[0])  # For the first plot (phi)
        # Create the second subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share x-axis with the first plot
    else:
        fig, ax1, ax2 = axes

    c = ax1.imshow(
        phi.T,
        origin="lower",
        aspect="auto",
        extent=(0, Lx, Lr1, Lr2),
        cmap=plt.get_cmap("plasma"),
    )
    ax1.set_ylabel("$r$")
    ax2.plot(x_center, psi)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$\psi$")
    # Add the colorbar to the top of the first subplot
    cb = fig.colorbar(
        c,
        ax=ax1,
        orientation="horizontal",
        location="top",
        label="$\phi$",
        aspect=75,
    )
    # ax1.set_aspect(1)
    return fig, ax1, ax2


# -----------------------------------------------------------------------------
#  Simulator
# -----------------------------------------------------------------------------
class PhaseFieldSimulator:
    """Notebook‑faithful curved‑wall phase‑field solver."""

    # -------- default wall --------------------------------------------------
    @staticmethod
    def _default_wall(Lx):
        return (
            lambda x: 2.0 * (jnp.tanh((jnp.abs(x - Lx / 2) - Lx / 4) / 5) + 1.0) + 0.5
        )

    # -------- init ----------------------------------------------------------
    def __init__(
        self,
        *,
        Lx: float = 30.0,
        Lr1: float = 5.0,
        Lr2: float = 15.0,
        nx: int = 128,
        nr: int = 64,
        chi_phi: float = 2.5,
        chi_psi: float = 0.0,
        lmda_phi: float = 1.0,
        lmda_psi: float = 1.0,
        h0: float = 0.0,
        h1: float = 1.0,
        M_phi: float = 1.0,
        M_psi: float = 1.0,
        psi_boundary: float = 0.05,
        dt: float = 1e-4 / 4,
        wall: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ):
        # geometry ----------------------------------------------------------------
        self.Lx, self.Lr1, self.Lr2, self.nx, self.nr = Lx, Lr1, Lr2, nx, nr
        self.dx, self.dr = Lx / nx, (Lr2 - Lr1) / nr
        self.x_c = jnp.arange(nx) * self.dx + 0.5 * self.dx
        self.r_c = jnp.arange(nr) * self.dr + 0.5 * self.dr + Lr1
        self.wall = wall or self._default_wall(Lx)

        # arc‑length segments along the wall
        self.wall_r = self.wall(self.x_c)
        seg = jnp.sqrt(jnp.diff(self.x_c) ** 2 + jnp.diff(self.wall_r) ** 2)
        self.seg_l = jnp.pad(seg, (1, 0), mode="edge")
        self.seg_r = jnp.pad(seg, (0, 1), mode="edge")
        self.seg_c = 0.5 * (self.seg_l + self.seg_r)

        # find the bottom of the wall
        self.wall_r_id = jnp.clip(
            jnp.searchsorted(self.r_c, self.wall_r), 0, self.nr - 1
        )
        self.wall_r_weight = (self.wall_r - self.r_c[self.wall_r_id - 1]) / self.dr

        # masks, signed distance for debugging/visualisation
        Xc, Rc = jnp.meshgrid(self.x_c, self.r_c, indexing="ij")
        self.signed_dist, self.normal_contact = signed_distance_lower_wall(
            Xc, Rc, self.wall, newton_iter=15
        )
        self.mask_inside = self.signed_dist > 0.0

        # cut‑cell fractions
        self.vol_frac, self.face_frac = cutcell_fractions(
            self.x_c, self.r_c, self.dx, self.dr, self.wall, self.mask_inside
        )
        self._prepare_ghosts(self.normal_contact)

        # physical params ------------------------------------------------------
        self.chi_phi, self.chi_psi = chi_phi, chi_psi
        self.lmda_phi, self.lmda_psi = lmda_phi, lmda_psi
        self.h0, self.h1 = h0, h1
        self.M_phi, self.M_psi = M_phi, M_psi
        self.dt = dt

        # boundary conditions -----------------------------------------------
        # φ: no flux at the wall, so we use ghost points to compute the gradient
        # ψ: Dirichlet boundary condition
        self.psi_boundary = psi_boundary

        # JIT kernels ----------------------------------------------------------
        self._laplace_2d = jit(self._laplace_2d)
        self._step = self._make_step_kernel()

        # ───────────────────── ghost‑cell preparation ──────────────────────

    def _prepare_ghosts(self, contact):
        neigh = (
            jnp.pad(self.mask_inside[:-1, :], ((1, 0), (0, 0)))
            | jnp.pad(self.mask_inside[1:, :], ((0, 1), (0, 0)))
            | jnp.pad(self.mask_inside[:, :-1], ((0, 0), (1, 0)))
            | jnp.pad(self.mask_inside[:, 1:], ((0, 0), (0, 1)))
        )
        ghost_mask = (~self.mask_inside) & neigh
        self.ghost_points_id = jnp.argwhere(ghost_mask)  # (Ng, 2)

        x_np, z_np = np.array(self.x_c), np.array(self.r_c)
        inside_idx = np.argwhere(np.array(self.mask_inside))
        ghost_in = np.zeros((self.ghost_points_id.shape[0], 2, 2), dtype=int)
        coeff = np.zeros((self.ghost_points_id.shape[0], 3))

        for g, (i, j) in enumerate(np.array(self.ghost_points_id)):
            gx, gz = x_np[i], z_np[j]
            d2 = (x_np[inside_idx[:, 0]] - gx) ** 2 + (z_np[inside_idx[:, 1]] - gz) ** 2
            # sel = inside_idx[np.argsort(d2)[:2]]
            # ghost_in[g] = sel
            cand = inside_idx[np.argsort(d2)]
            ghost_in0 = cand[0]
            v0 = np.array([x_np[ghost_in0[0]] - gx, z_np[ghost_in0[1]] - gz])
            v0 /= np.linalg.norm(v0) + 1e-12

            # find the second point that is not colinear with the first one
            for c in cand[1:]:
                v1 = np.array([x_np[c[0]] - gx, z_np[c[1]] - gz])
                v1 /= np.linalg.norm(v1) + 1e-12
                if abs(v0 @ v1) < 0.9:
                    ghost_in1 = c
                    break

            ghost_in[g] = (ghost_in0, ghost_in1)

            nc = np.array(contact[i, j])
            nvec = -np.array([gx - nc[0], gz - nc[1]])
            nvec /= np.linalg.norm(nvec) + 1e-12
            tvec = np.array([nvec[1], -nvec[0]])
            r = np.array([[x_np[p] - gx, z_np[q] - gz] for p, q in ghost_in[g]])
            a = nvec @ r.T
            b = tvec @ r.T
            coeff[g] = np.array([b[1], -b[0], a[1] * b[0] - a[0] * b[1]]) / (
                b[1] - b[0] + 1e-12
            )
        self.ghost_inside_id = jnp.array(ghost_in)
        self.ghost_coef = jnp.array(coeff)

        ghost_contact = contact[self.ghost_points_id[:, 0], self.ghost_points_id[:, 1]]
        j0 = jnp.clip(
            ((ghost_contact[..., 0] - self.dx / 2) // self.dx).astype(int),
            0,
            self.nx - 2,
        )
        w = (ghost_contact[..., 0] - self.x_c[j0]) / self.dx
        self.ghost_j0, self.ghost_w = j0, w

    # -------------------------------------------------------------------------
    #  helpers
    # -------------------------------------------------------------------------
    def _laplace_2d(self, f: jnp.ndarray):
        """Standard centred‑difference Laplacian."""
        fpad = jnp.pad(f, ((1, 1), (1, 1)), mode="edge")
        d2x = (fpad[2:, 1:-1] - 2.0 * f + fpad[:-2, 1:-1]) / self.dx**2
        d2r = (fpad[1:-1, 2:] - 2.0 * f + fpad[1:-1, :-2]) / self.dr**2
        d2r_cyl = d2r + (fpad[1:-1, 2:] - fpad[1:-1, :-2]) / (
            2 * self.r_c[None, :] * self.dr
        )
        return d2x + d2r_cyl

    # @jit
    def _laplace_nonuniform_1d(self, f: jnp.ndarray):
        # centred finite‑difference with non‑uniform spacing (seg_left/right)
        f_fwd = jnp.roll(f, -1)
        f_back = jnp.roll(f, 1)
        return (
            2
            * (
                f_back / self.seg_l
                - f / self.seg_l
                - f / self.seg_r
                + f_fwd / self.seg_r
            )
            / (self.seg_l + self.seg_r)
        )

    def _eval_phi_ghosts(self, phi, psi):
        """Evaluate φ at ghost points."""
        psi_ghost = (
            self.ghost_w * psi[self.ghost_j0]
            + (1 - self.ghost_w) * psi[self.ghost_j0 + 1]
        )
        dphi_dn = -(self.h0 + self.h1 * psi_ghost) / self.lmda_phi
        phi_ghost = (
            self.ghost_coef[:, 0]
            * phi[self.ghost_inside_id[:, 0, 0], self.ghost_inside_id[:, 0, 1]]
            + self.ghost_coef[:, 1]
            * phi[self.ghost_inside_id[:, 1, 0], self.ghost_inside_id[:, 1, 1]]
            + self.ghost_coef[:, 2] * dphi_dn
        )
        return phi_ghost

    # ───────────────────────────── step kernel ──────────────────────────────
    def _make_step_kernel(self):
        dx, dr = self.dx, self.dr
        ff, vf = self.face_frac, self.vol_frac
        seg_l, seg_r, seg_c = self.seg_l, self.seg_r, self.seg_c
        nx, nr = self.nx, self.nr
        x_c, r_c, wall_fn = self.x_c, self.r_c, self.wall
        wall_r = wall_fn(x_c)  # wall height at each x_c
        wall_r_l = wall_fn(x_c - dx / 2)  # left edge of each cell
        chi_phi, chi_psi = self.chi_phi, self.chi_psi
        lmda_phi, lmda_psi = self.lmda_phi, self.lmda_psi
        h0, h1 = self.h0, self.h1
        M_phi, M_psi, dt = self.M_phi, self.M_psi, self.dt
        psi_boundary = self.psi_boundary
        eval_ghost = self._eval_phi_ghosts
        lap2d = self._laplace_2d
        lap1d_nonuniform = self._laplace_nonuniform_1d

        ghost_points_id = self.ghost_points_id
        ff_west, ff_south = ff["west"], ff["south"]
        ff_west_mask, ff_south_mask = ff_west > 1e-9, ff_south > 1e-9

        @jit
        def _step(phi, psi):
            # 1) impose ghost values for φ
            phi_ghost = eval_ghost(phi, psi)
            phi = phi.at[ghost_points_id[:, 0], ghost_points_id[:, 1]].set(phi_ghost)

            # 2) chemical potentials
            mu_phi = calc_mu_uniform(phi, chi_phi) - lmda_phi * lap2d(phi)

            phi_wall = (1 - self.wall_r_weight) * phi[
                np.arange(nx), self.wall_r_id - 1
            ] + self.wall_r_weight * phi[np.arange(nx), self.wall_r_id]
            mu_psi = (
                calc_mu_uniform(psi, chi_psi)
                - h1 * phi_wall
                - lmda_psi * lap1d_nonuniform(psi)
            )

            # 3) fluxes
            j_phi_x = jnp.where(
                ff_west_mask,
                -M_phi * jnp.diff(mu_phi, prepend=mu_phi[:1, :], axis=0) / dx * ff_west,
                0.0,
            )
            j_phi_r = jnp.where(
                ff_south_mask,
                -M_phi
                * jnp.diff(mu_phi, prepend=mu_phi[:, :1], axis=1)
                / dr
                * ff_south,
                0.0,
            )
            div_j_phi = (
                jnp.diff(j_phi_x, append=0, axis=0) / dx
                # + jnp.diff(j_phi_r, append=0, axis=1) / dr
                # + j_phi_r / r_c[None, :]  # cylindrical coords
                + jnp.diff(j_phi_r * r_c[None, :], append=0, axis=1) / dr / r_c[None, :]
            ) / vf
            phi_new = phi - dt * div_j_phi

            j_psi = -M_psi * jnp.diff(mu_psi, prepend=mu_psi[:1]) / seg_l
            div_j_psi = jnp.diff(j_psi * wall_r_l, append=0) / seg_c / wall_r
            psi_new = psi - dt * div_j_psi

            # 4) boundary conditions
            # psi_new = psi_new.at[0].set(psi_boundary).at[-1].set(psi_boundary)
            # only set Dirichlet boundary condition on the right side; left side has natural Neumann boundary from "prepend" in j_psi
            psi_new = psi_new.at[-1].set(psi_boundary)
            # phi_new = phi_new.at[:, -1].set(phi_boundary)

            return phi_new, psi_new

        return _step

    # ───────────────────────────── public API ───────────────────────────────
    def step(self, phi: jnp.ndarray, psi: jnp.ndarray):
        """Advance a single time step (in‑place stable)."""
        return self._step(phi, psi)

    def run(
        self,
        phi0: jnp.ndarray,
        psi0: jnp.ndarray,
        *,
        num_epochs: int = 20,
        steps_per_epoch: int = 1000,
        progress: bool = True,
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run simulation, returning snapshots after each epoch."""

        phi, psi = jnp.array(phi0), jnp.array(psi0)
        history: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
        for _ in trange(num_epochs, disable=not progress):
            (phi, psi), _ = lax.scan(
                lambda state, _: (self._step(*state), None),
                (phi, psi),
                None,
                length=steps_per_epoch,
                unroll=8,
            )
            history.append((phi, psi))
        return history
