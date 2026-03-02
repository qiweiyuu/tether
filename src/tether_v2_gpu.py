from jax import config

config.update("jax_enable_x64", True)

# reserve GPU memory 90% for JAX; for visualization nodes, request less GPU RAM.
import os, sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax.numpy as jnp
from jax import jit, random, lax, vmap
import argparse
from tqdm import tqdm

from scipy.optimize import fsolve, minimize, curve_fit
from scipy.interpolate import CubicSpline, splrep, splev, UnivariateSpline
from scipy.signal import savgol_filter


plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "savefig.format": "pdf",
        "legend.edgecolor": "0.0",
        "legend.framealpha": 1.0,
    }
)


plasmamap = plt.get_cmap("plasma")

parser = argparse.ArgumentParser(description="ZTS_radial_v1_debug")

# parameters for the grid
parser.add_argument("--Lr", type=float, default=30, help="radial domain size")
parser.add_argument("--Lz", type=float, default=15, help="vertical domain size")
parser.add_argument(
    "--nr", type=int, default=64, help="number of points in the radial direction"
)
parser.add_argument(
    "--nz", type=int, default=32, help="number of points in the vertical direction"
)

# interaction parameters
parser.add_argument(
    "--chi-psi", type=float, default=0.0, help="chi parameter for psi (membrane)"
)
parser.add_argument(
    "--chi-phi", type=float, default=2.5, help="chi parameter for phi (bulk)"
)
parser.add_argument(
    "--lmda-phi", type=float, default=1.0, help="gradient coefficient for phi (bulk)"
)
parser.add_argument(
    "--lmda-psi",
    type=float,
    default=1.0,
    help="gradient coefficient for psi (membrane)",
)
parser.add_argument(
    "--h0",
    type=float,
    default=0.0,
    help="coupling coefficient between phi and the bare membrane",
)
parser.add_argument(
    "--h-phi-psi",
    type=float,
    default=1.0,
    help="coupling coefficient between phi and psi",
)

# mobility coefficients
parser.add_argument(
    "--m-phi", type=float, default=1.0, help="mobility coefficient for phi"
)
parser.add_argument(
    "--m-psi", type=float, default=1.0, help="mobility coefficient for psi"
)
parser.add_argument("--dt", type=float, default=1e-4, help="time step size")


# bulk tether concentration
parser.add_argument(
    "--psi-bulk", type=float, default=0.1, help="bulk tether concentration"
)

## number of steps
parser.add_argument("--n-steps", type=int, default=2000000, help="number of steps")
parser.add_argument("--n-steps-relax", type=int, default=500000, help="number of steps")

parser.add_argument(
    "--init-profile",
    type=str,
    default=None,
    help="initial profile for phi and psi",
)

parser.add_argument(
    "--outputdir",
    type=str,
    default="data_tether_v2/",
    help="output directory",
)
flags = parser.parse_args()

chi_psi, chi_phi = flags.chi_psi, flags.chi_phi
lmda_phi, lmda_psi = flags.lmda_phi, flags.lmda_psi
h0, h_phi_psi = flags.h0, flags.h_phi_psi
M_phi, M_psi = flags.m_phi, flags.m_psi
psi_boundary = flags.psi_bulk


output_foldername = (
    flags.outputdir
    + f"/chi_psi_{chi_psi}_chi_phi_{chi_phi}_h0_{h0}_h1_{h_phi_psi}/"
    + f"psi_{psi_boundary:.3f}/"
)

os.system(f"mkdir -p {output_foldername}")

# redirect the output of this script to a file
output_file = f"{output_foldername}/output.txt"
sys.stdout = open(output_file, "w")


########################################################################################
# construct the grid

dz = flags.Lz / flags.nz
dr = flags.Lr / flags.nr

r_left = jnp.arange(flags.nr) * dr
r_center = r_left + dr / 2
z_center = dz / 2 + jnp.arange(flags.nz) * dz


def plot_profile(phi, psi):
    fig = plt.figure(figsize=(6, 5), tight_layout=True)

    # Create a GridSpec with 2 rows and 1 column
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[4, 1]
    )  # Adjust height_ratios to make axs[1] shorter

    # Create the first subplot
    ax1 = fig.add_subplot(gs[0])  # For the first plot (phi)
    c = ax1.imshow(
        phi,
        origin="lower",
        aspect="auto",
        extent=(0, flags.Lr, 0, flags.Lz),
        cmap=plasmamap,
    )
    ax1.set_ylabel("$z$")

    # Create the second subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share x-axis with the first plot
    ax2.plot(r_center, psi)
    ax2.set_xlabel("$r$")
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
    return fig, ax1, ax2


def calc_1d_grad(psi, dr):
    return jnp.pad(
        psi[2:] - psi[:-2],
        ((1, 1)),
        mode="constant",
        constant_values=(
            -3 * psi[0] + 4 * psi[1] - psi[2],
            3 * psi[-1] - 4 * psi[-2] + psi[-3],
        ),
    ) / (2 * dr)


def calc_laplacian_1d(psi):
    # psi'' using central stencil; using one sided stencil at r=0 and r=R
    d2_psi = jnp.pad(
        (psi[2:] + psi[:-2] - 2.0 * psi[1:-1]) / dr**2,
        ((1, 1)),
        mode="constant",
        constant_values=(
            (2 * psi[0] - 5 * psi[1] + 4 * psi[2] - psi[3]) / dr**2,
            (2 * psi[-1] - 5 * psi[-2] + 4 * psi[-3] - psi[-4]) / dr**2,
        ),
    )
    # psi' using central stencil; use one-side stencil at r=0 and r=R
    d1_psi = calc_1d_grad(psi, dr)
    return d2_psi + d1_psi / r_center


def calc_laplacian_2d(phi, psi):
    # in the r direction, use the same code as above, but apply to the last axis
    laplacian_r = jnp.apply_along_axis(calc_laplacian_1d, 1, phi)
    # in the z direction, use central stencil, with ghost points that fulfill the wetting boundary conditions
    phi_z_pad = jnp.concatenate(
        [
            (phi[0] + (h0 + h_phi_psi * psi) / lmda_phi * dz)[jnp.newaxis, :],
            phi,
            phi[-1][jnp.newaxis, :],
        ],
        axis=0,
    )
    laplacian_z = (phi_z_pad[2:] + phi_z_pad[:-2] - 2.0 * phi_z_pad[1:-1]) / dz**2

    return laplacian_r + laplacian_z


phi_tol = 1e-8


def calc_mu_entropy(psi):
    return jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) - jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_mu_uniform(psi, chi_psi):
    return calc_mu_entropy(psi) + chi_psi * (1 - 2 * psi)


def calc_mu_phi(phi, psi):
    return (
        calc_mu_entropy(phi)
        + chi_phi * (1 - 2 * phi)
        - lmda_phi * calc_laplacian_2d(phi, psi)
    )


def calc_mu_psi(psi, phi):
    return (
        calc_mu_entropy(psi)
        + chi_psi * (1 - 2 * psi)
        - lmda_psi * calc_laplacian_1d(psi)
        - (h0 + h_phi_psi * phi[0])
    )


# @jit
def calc_step(phi, psi, dt, phi_boundary, psi_boundary):
    mu_phi = calc_mu_phi(phi, psi)
    mu_psi = calc_mu_psi(psi, phi)

    # compute the fluxes, always evaluate at the left boundary
    j_psi = -M_psi * jnp.diff(mu_psi, prepend=mu_psi[-1]) / dr
    j_phi_r = -M_phi * jnp.diff(mu_phi, prepend=mu_phi[:, -1:], axis=1) / dr
    j_phi_z = -M_phi * jnp.diff(mu_phi, prepend=mu_phi[:1, :], axis=0) / dz

    # compute the divergence of the fluxes
    psi_new = psi - dt * jnp.diff(j_psi * r_left, append=0) / (dr * r_center)
    phi_div = (
        jnp.diff(j_phi_r * r_left, append=0, axis=1) / (dr * r_center)
        + jnp.diff(j_phi_z, append=0, axis=0) / dz
    )
    phi_new = phi - dt * phi_div

    # impose the boundary condition (bulk concentration)
    # for contact angle measurements, use Neuemann boundary condition for phi
    # phi_new = phi_new.at[-1, :].set(phi_boundary)  # at z=zmax
    psi_new = psi_new.at[-1].set(psi_boundary)  # at r=rmax

    # phi_new = phi_new.at[:, -1].set(phi_boundary)  # at r=rmax
    return phi_new, psi_new


def calc_step_wrapper(carry, _):
    phi, psi, dt, phi_boundary, psi_boundary = carry
    phi_new, psi_new = calc_step(phi, psi, dt, phi_boundary, psi_boundary)
    return (phi_new, psi_new, dt, phi_boundary, psi_boundary), None


def calc_f_entropy(psi):
    return psi * jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) + (1 - psi) * jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_f_uniform(psi, chi_psi):
    return calc_f_entropy(psi) + chi_psi * psi * (1 - psi)


@jit
def calc_f_psi(psi, phi, psi_boundary):
    psi_grad = jnp.apply_along_axis(calc_1d_grad, 0, psi, dr=dr)
    mu_psi_boundary = calc_mu_uniform(psi_boundary, chi_psi)
    return jnp.sum(
        (
            calc_f_uniform(psi, chi_psi)
            - (h0 + h_phi_psi * psi) * phi[0]
            + lmda_psi / 2 * psi_grad**2
            - mu_psi_boundary * psi
        )
        * dr
        * 2
        * np.pi
        * r_center
    )


@jit
def calc_f_phi(phi, phi_boundary):
    phi_grad_r = jnp.apply_along_axis(calc_1d_grad, 1, phi, dr=dr)
    phi_grad_z = jnp.apply_along_axis(calc_1d_grad, 0, phi, dr=dz)  # this is incorrect

    # don't worry about the phi chemical potential since we use Neuemann boundary condition
    # mu_phi_boundary = calc_mu_uniform(phi_boundary, chi_phi)
    return jnp.sum(
        (
            calc_f_uniform(phi, chi_phi)
            + lmda_phi / 2 * (phi_grad_r**2 + phi_grad_z**2)
            # - mu_phi_boundary * phi
        )
        * dr
        * 2
        * np.pi
        * r_center
        * dz
    )


calc_phi_tot = lambda phi: jnp.sum(phi * dr * 2 * np.pi * r_center * dz)
calc_phi_mean = lambda phi: calc_phi_tot(phi) / (np.pi * flags.Lr**2 * flags.Lz)


def measure_contact_angle_fit(
    phi, phi_threshold, order=2, spherical_cap=False, psi=None
):
    # use the contour generated by the contour function; also does not work super well...
    if psi is None:
        psi = np.zeros_like(phi[0])

    fig, ax1, ax2 = plot_profile(phi, psi)
    c = ax1.contour(
        r_center,
        z_center,
        phi,
        levels=[phi_threshold],
        colors="k",
        linestyles="--",
    )

    c_path = c.allsegs[0][0]
    # sort along the radial direction
    c_path = c_path[np.argsort(-c_path[:, 0])]
    # fit a polynomial to the last few points to determine the contact line
    # fit_end = np.max([np.argmax(c_path[:, 1] > 1), 3])

    if spherical_cap:
        # fit a spherical cap
        r_sphere_fun = lambda z, R0, z0: R0**2 - (z - z0) ** 2
        fit_mask = c_path[:, 1] > 1
        fitres_sphere = curve_fit(
            r_sphere_fun,
            c_path[fit_mask, 1],
            c_path[fit_mask, 0] ** 2,
            p0=[c_path.max() + 2, -c_path.max()],
        )
        cos_theta = -fitres_sphere[0][1] / fitres_sphere[0][0]
        r_contact = np.sqrt(r_sphere_fun(0, *fitres_sphere[0]))

        # plot the quality of the fit
        z_fit = np.linspace(0, c_path[:, 1].max(), 100)
        r_fit = np.sqrt(r_sphere_fun(z_fit, *fitres_sphere[0]))
        ax1.plot(r_fit, z_fit, "r--")
        return r_contact, cos_theta, (fig, ax1, ax2)
    else:
        # fit a polynomial
        plt.close(fig)

        fit_end = 4
        fitres = np.polyfit(c_path[:fit_end, 1], c_path[:fit_end, 0], order)
        # fit using higher order polynomials tend to diverge

        # compute the slope at z=0
        fitres_slope = -1 / fitres[-2]
        contact_line = np.poly1d(fitres)
        r_contact = contact_line(0)
        cos_theta = 1 / np.sqrt(1 + fitres_slope**2)
        return r_contact, cos_theta


########################################################################################
# determine binodal concentration
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


########################################################################################
phi_mesh = np.linspace(0, 1, 100)
f_phi_mesh = calc_f_uniform(phi_mesh, chi_phi)
phi_binodal, phi_spinodal = calc_binodal_spinodal(phi_mesh, f_phi_mesh)
phi_boundary = min(phi_binodal)
phi_binodal_center = (phi_binodal[0] + phi_binodal[1]) / 2
########################################################################################

## initialize the profile
if flags.init_profile is not None:
    data = np.load(flags.init_profile, allow_pickle=True)

    psi_init = jnp.interp(r_center, data["r_center"], data["psi"])
    psi_init -= psi_init[-1] - psi_boundary
    psi_init = jnp.clip(psi_init, phi_tol, 1 - phi_tol)

    phi_data = data["phi"]
    phi_interp_r = jnp.stack(
        [
            jnp.interp(r_center, data["r_center"], phi_data[i, :])
            for i in range(phi_data.shape[0])
        ],
        axis=0,
    )
    phi_init = jnp.stack(
        [
            jnp.interp(z_center, data["z_center"], phi_interp_r[:, i])
            for i in range(phi_interp_r.shape[1])
        ],
        axis=1,
    )
else:
    psi_init = np.ones(flags.nr) * psi_boundary
    phi_init = np.ones((flags.nz, flags.nr)) * phi_boundary
    tanh_fun = lambda x, xcenter, width: 0.5 * (1 + np.tanh((x - xcenter) / width))
    R_init = 12
    z_init = 1
    r_init = np.sqrt(r_center**2 + (z_center[:, np.newaxis] + z_init) ** 2)
    phi_init += (phi_binodal[1] - phi_binodal[0]) * tanh_fun(R_init, r_init, 1)

fig, ax1, ax2 = plot_profile(phi_init, psi_init)
fig.savefig(f"{output_foldername}/initial_profile.pdf")


# relax the initial condition
n_steps = flags.n_steps_relax
dt = flags.dt
# phi, psi = phi_init.copy(), psi_init.copy()
# for _ in tqdm(range(n_steps)):
#     phi, psi = calc_step(phi, psi, dt, phi_boundary, psi_boundary)
(phi, psi, dt, phi_boundary, psi_boundary), _ = lax.scan(
    calc_step_wrapper, (phi_init, psi_init, dt, phi_boundary, psi_boundary), None, length=n_steps
)
phi_init_relaxed, psi_init_relaxed = phi.copy(), psi.copy()
fig, ax1, ax2 = plot_profile(phi_init_relaxed, psi_init_relaxed)
fig.savefig(f"{output_foldername}/initial_profile_relaxed.pdf")

print("Initial relaxation complete, starting the main simulation.")
## start the main simulation
# phi, psi = phi_init.copy(), psi_init.copy()
phi, psi = phi_init_relaxed.copy(), psi_init_relaxed.copy()

# adaptive time stepping
phi_tol = 0.001
# psi_tol = 1e-4
n_steps = flags.n_steps
err_tol = 1e-2
# dt_max = flags.dt * 4
t_save = n_steps * flags.dt / 25
dt = flags.dt * 4
n_step_save = int(t_save / dt)
n_epoch = n_steps // n_step_save

t = 0
t_remaining = t_save

f_trace = [calc_f_phi(phi, phi_boundary) + calc_f_psi(psi, phi, psi_boundary)]
phi_trace = [phi.copy()]
psi_trace = [psi.copy()]
t_trace = [t]

for epoch in range(n_epoch):
    (phi, psi, dt, phi_boundary, psi_boundary), _ = lax.scan(
        calc_step_wrapper, (phi, psi, dt, phi_boundary, psi_boundary), None, length=n_step_save
    )
    t += n_step_save * dt
    f_trace.append(
        calc_f_phi(phi, phi_boundary) + calc_f_psi(psi, phi, psi_boundary)
    )
    phi_trace.append(phi.copy())
    psi_trace.append(psi.copy())
    t_trace.append(t)


# for i in tqdm(range(n_steps)):
#     phi_new, psi_new = calc_step(phi, psi, dt, phi_boundary, psi_boundary)
#     err = jnp.max(jnp.abs(phi_new - phi))
#     if err < err_tol:
#         phi, psi = phi_new, psi_new
#         t += dt
#         t_remaining -= dt
#         dt = min(dt_max, dt * 1.05)
#         if t_remaining < 0:
#             f_trace.append(
#                 calc_f_phi(phi, phi_boundary) + calc_f_psi(psi, phi, psi_boundary)
#             )
#             phi_trace.append(phi.copy())
#             psi_trace.append(psi.copy())
#             t_trace.append(t)
#             if jnp.linalg.norm(phi_trace[-1] - phi_trace[-2]) < phi_tol:
#                 # if jnp.linalg.norm(psi_trace[-1] - psi_trace[-2]) < psi_tol:
#                 break
#             t_remaining += t_save
#     else:
#         dt /= 2

print(f"The simulation has completed at t={t:.2e}.")
print(f"final free energy change: {f_trace[-1] - f_trace[-2]:.2e}")
print(f"Average time step: {t / n_steps:.2e}")

t_trace = jnp.array(t_trace)
f_trace = jnp.array(f_trace)

np.savez(
    f"{output_foldername}/final_profile.npz",
    phi=phi,
    psi=psi,
    r_center=r_center,
    z_center=z_center,
)

### visualize the final profile
t_id = -1
fig, ax1, ax2 = plot_profile(phi_trace[t_id], psi_trace[t_id])
ax1.text(
    flags.Lr * 0.98,
    flags.Lz * 0.9,
    f"$t={t_trace[t_id]:.2f}$",
    color="white",
    ha="right",
    fontsize=16,
)
fig.savefig(f"{output_foldername}/final_profile.pdf")

## plot the convergence of various quantities
fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, f_trace - f_trace[0], "o-", clip_on=False)
ax.set_xlabel("Time $t$")
ax.set_ylabel("Free energy $f$")
ax.set_xlim(0, None)
fig.savefig(f"{output_foldername}/free_energy_trace.pdf")

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
phi_trace = np.stack(phi_trace)
phi_diff = np.linalg.norm(np.diff(phi_trace, axis=0), axis=(1, 2))
ax.plot(t_trace[1:], phi_diff, "o-")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"$\| \phi^{(n)} - \phi^{(n-1)} \|$")
ax.set_xlim(0, None)
ax.set_yscale("log")
fig.savefig(f"{output_foldername}/phi_diff_trace.pdf")

### measure the contact angle
phi_binodal_center = (phi_binodal[0] + phi_binodal[1]) / 2
cos_theta_trace = np.zeros_like(t_trace)
r_contact_trace = np.zeros_like(t_trace)
for t_id in tqdm(range(len(t_trace))):
    r_contact_trace[t_id], cos_theta_trace[t_id], (fig, ax1, ax2) = (
        measure_contact_angle_fit(
            phi_trace[t_id],
            phi_threshold=phi_binodal_center,
            spherical_cap=True,
            psi=psi_trace[t_id],
        )
    )
np.savez(
    f"{output_foldername}/trace.npz",
    t_trace=t_trace,
    f_trace=f_trace,
    r_contact_trace=r_contact_trace,
    cos_theta_trace=cos_theta_trace,
)

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, r_contact_trace, "o-")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"$r_{\mathrm{contact}}$")
ax.set_xlim(0, None)
fig.savefig(f"{output_foldername}/r_contact_trace.pdf")

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, cos_theta_trace, "o-", clip_on=False, label="data")
expfun = lambda x, y0, y1, x0: y0 - y1 * np.exp(-x / x0)
# fit_len = min(20, len(t_trace))
# fitres = curve_fit(
#     expfun,
#     t_trace[-fit_len:],
#     cos_theta_trace[-fit_len:],
#     p0=[cos_theta_trace[-1], -cos_theta_trace[-20] + cos_theta_trace[-1], t_trace[-1]],
# )
# ax.plot(
#     t_trace,
#     expfun(t_trace, *fitres[0]),
#     "--",
#     color="tab:red",
#     label=r"fit, $\cos\theta\to" + f"{fitres[0][0]:.2f}$",
# )
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"$\cos(\theta)$")
ax.set_xlim(0, None)
ax.legend(frameon=False, loc="lower right")
fig.savefig(f"{output_foldername}/cos_theta_trace.pdf")

fig, ax1, ax2 = plot_profile(phi_trace[-1], psi_trace[-1])

contact_line = lambda r: (r_contact_trace[-1] - r) * np.tan(
    np.arccos(cos_theta_trace[-1])
)
rmesh = np.linspace(0, r_contact_trace[-1], 100)
ax1.plot(rmesh, contact_line(rmesh), "--", color="white")
ax1.set_ylim(0, 5)
ax1.set_xlim(
    np.max([0, r_contact_trace[-1] - 5]), np.min([r_contact_trace[-1] + 5, flags.Lr])
)
fig.savefig(f"{output_foldername}/final_profile_contact_line.pdf")


# output_file.close()
