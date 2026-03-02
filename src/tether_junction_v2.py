from jax import config

config.update("jax_enable_x64", True)

# reserve GPU memory 90% for JAX; for visualization nodes, request less GPU RAM.
import os

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "legend.fontsize": 14,  # this is the font size in legends
        "xtick.labelsize": 14,  # this and next are the font of ticks
        "ytick.labelsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 18,  # this is the fonx of axes labels
        "savefig.format": "pdf",  # how figures should be saved
        "legend.edgecolor": "0.0",
        "legend.framealpha": 1.0,
        "text.usetex": True,
    }
)


plasmamap = plt.get_cmap("plasma")
virimap = plt.cm.viridis


parser = argparse.ArgumentParser(description="ZTS_radial_v1_debug")

# parameters for the grid
parser.add_argument("--Lx", type=float, default=30, help="radial domain size")
parser.add_argument("--Lz", type=float, default=30, help="vertical domain size")
parser.add_argument(
    "--nx", type=int, default=64, help="number of points in the radial direction"
)
parser.add_argument(
    "--nz", type=int, default=64, help="number of points in the vertical direction"
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
    default=-0.2,
    help="coupling coefficient between phi and the bare membrane",
)
parser.add_argument(
    "--h-phi-psi",
    type=float,
    default=2.0,
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
    "--psi-bulk", type=float, default=0.05, help="bulk tether concentration"
)

# bulk condensate concentration, measured in units of (spinodal-binodal)
parser.add_argument(
    "--phi-supersaturation",
    type=float,
    default=0.5,
    help="supersaturation of the condensate",
)
parser.add_argument(
    "--phi-cover-depth",
    type=float,
    default=2.0,
    help="depth of the condensate covering the membrane",
)
parser.add_argument(
    "--boundary-repel",
    type=float,
    default=5.0,
    help="distance from the boundary where the condensate is not present initially",
)

## number of steps
parser.add_argument("--n-epoch", type=int, default=200, help="number of epochs")
parser.add_argument(
    "--n-steps-per-epoch", type=int, default=200000, help="number of steps per epoch"
)

parser.add_argument(
    "--init-profile",
    type=str,
    default=None,
    help="initial profile for phi and psi",
)

parser.add_argument(
    "--outputdir",
    type=str,
    default="data_tether_junction_v2_test/",
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
    + f"psi_{psi_boundary:.3f}_m_psi_{M_psi:.3f}/"
)
print(f"Output folder: {output_foldername}")
os.system(f"mkdir -p {output_foldername}")


########################################################################################
# construct the grid

dz = flags.Lz / flags.nz
dx = flags.Lx / flags.nx

x_center = dx / 2 + jnp.arange(flags.nx) * dx
z_center = dz / 2 + jnp.arange(flags.nz) * dz


def plot_profile(phi, psi_x, psi_z, axes=None, vmin=None, vmax=None):
    fig, ax_img = plt.subplots(figsize=(6, 5), tight_layout=True)
    c = ax_img.imshow(
        phi,
        origin="lower",
        # aspect="auto",
        extent=(0, flags.Lx, 0, flags.Lz),
        cmap=plasmamap,
        vmin=vmin,
        vmax=vmax,
    )

    # create side / bottom axes that hug the image axes exactly
    divider = make_axes_locatable(ax_img)
    ax_x = divider.append_axes("bottom", size="20%", pad=0.5, sharex=ax_img)
    ax_z = divider.append_axes("left", size="20%", pad=0.5, sharey=ax_img)

    # Create the second subplot
    ax_x.plot(x_center, psi_x)
    ax_x.set_xlabel("$x$")
    ax_x.set_ylabel("$\psi$")
    ax_x.set_ylim(0, None)

    # Create the second subplot
    ax_z.plot(psi_z, z_center)
    ax_z.set_ylabel("$z$")
    ax_z.set_xlabel("$\psi$")
    ax_z.set_xlim(0, None)

    # Add the colorbar to the top of the first subplot
    cb = fig.colorbar(
        c,
        ax=ax_img,
        orientation="vertical",
        location="right",
        label="$\phi$",
        aspect=50,
    )

    return fig, (ax_img, ax_x, ax_z)


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


def calc_laplacian_1d(psi, dr):
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
    return d2_psi


def calc_laplacian_2d(phi, psi_x, psi_z):

    phi_z_pad = jnp.concatenate(
        [
            (phi[0] + (h0 + h_phi_psi * psi_x) / lmda_phi * dz)[jnp.newaxis, :],
            phi,
            phi[-1][jnp.newaxis, :],
        ],
        axis=0,
    )
    laplacian_z = (phi_z_pad[2:] + phi_z_pad[:-2] - 2.0 * phi_z_pad[1:-1]) / dz**2

    phi_x_pad = jnp.concatenate(
        [
            (phi[:, 0] + (h0 + h_phi_psi * psi_z) / lmda_phi * dx)[:, jnp.newaxis],
            phi,
            phi[-1][:, jnp.newaxis],
        ],
        axis=1,
    )
    laplacian_x = (
        phi_x_pad[:, 2:] + phi_x_pad[:, :-2] - 2.0 * phi_x_pad[:, 1:-1]
    ) / dx**2
    return laplacian_x + laplacian_z


phi_tol = 1e-8


def calc_mu_entropy(psi):
    return jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) - jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_mu_uniform(psi, chi_psi):
    return calc_mu_entropy(psi) + chi_psi * (1 - 2 * psi)


def calc_mu_phi(phi, psi_x, psi_z):
    return (
        calc_mu_entropy(phi)
        + chi_phi * (1 - 2 * phi)
        - lmda_phi * calc_laplacian_2d(phi, psi_x, psi_z)
    )


def calc_mu_psi(psi, phi, dr):
    return (
        calc_mu_entropy(psi)
        + chi_psi * (1 - 2 * psi)
        - lmda_psi * calc_laplacian_1d(psi, dr)
        - (h0 + h_phi_psi * phi)
    )


@jit
def calc_step(phi, psi_x, psi_z, dt, phi_boundary, psi_boundary):
    # compute chemical potentials, evaluted at the center of the grid (dx/2, 3dx/2, ..., Lx-dx/2)
    mu_phi = calc_mu_phi(phi, psi_x, psi_z)
    mu_psi_x = calc_mu_psi(psi_x, phi[0], dx)
    mu_psi_z = calc_mu_psi(psi_z, phi[:, 0], dz)

    # compute the fluxes, always evaluate at the left boundary (0, dx, 2dx, ..., Lx-dx)
    mu_psi = jnp.concatenate([mu_psi_z[::-1], mu_psi_x])
    assert (dx - dz) < 1e-8, "dx and dz should be equal, cannot handel psi flux for now"
    j_psi = (
        -M_psi * jnp.diff(mu_psi, prepend=mu_psi[:1], append=mu_psi[-1:], axis=0) / dz
    )
    j_phi_x = -M_phi * jnp.diff(mu_phi, prepend=mu_phi[:, :1], axis=1) / dx
    j_phi_z = -M_phi * jnp.diff(mu_phi, prepend=mu_phi[:1, :], axis=0) / dz

    # compute the divergence of the fluxes
    psi_div = jnp.diff(j_psi) / dz
    psi_x_new = psi_x - dt * psi_div[-psi_x.shape[0] :]
    psi_z_new = psi_z - dt * psi_div[: psi_z.shape[0]][::-1]

    # ATTN: need to figure out how the divergence is handeled at the junction.

    phi_div = (
        jnp.diff(j_phi_x, append=0, axis=1) / dx
        + jnp.diff(j_phi_z, append=0, axis=0) / dz
    )
    phi_new = phi - dt * phi_div

    # impose the boundary condition (bulk concentration)
    # for contact angle measurements, use Neuemann boundary condition for phi
    # phi_new = phi_new.at[-1, :].set(phi_boundary)  # at z=zmax
    psi_x_new = psi_x_new.at[-1].set(psi_boundary)  # at r=rmax
    psi_z_new = psi_z_new.at[-1].set(psi_boundary)  # at z=zmax

    # phi_new = phi_new.at[:, -1].set(phi_boundary)  # at r=rmax
    return phi_new, psi_x_new, psi_z_new


def calc_f_entropy(psi):
    return psi * jnp.log(jnp.clip(psi, phi_tol, 1.0 - phi_tol)) + (1 - psi) * jnp.log(
        jnp.clip(1.0 - psi, phi_tol, 1.0 - phi_tol)
    )


def calc_f_uniform(psi, chi_psi):
    return calc_f_entropy(psi) + chi_psi * psi * (1 - psi)


def loss_sphere(c_path, R0, cos_theta):
    z0 = -R0 * cos_theta * np.sign(1 / np.sqrt(2) - cos_theta)
    return np.sum(((c_path[:, 0] - z0) ** 2 + (c_path[:, 1] - z0) ** 2 - R0**2) ** 2)


def measure_contact_angle_fit(phi, phi_threshold, psi_x=None, psi_z=None):
    # use the contour generated by the contour function; also does not work super well...
    if psi_x is None:
        psi_x = np.zeros(phi.shape[1])
    if psi_z is None:
        psi_z = np.zeros(phi.shape[0])

    fig, (ax_img, ax_x, ax_z) = plot_profile(phi, psi_x, psi_z)
    c = ax_img.contour(
        x_center,
        z_center,
        phi,
        levels=[phi_threshold],
        colors="k",
        linestyles="--",
    )
    c_path = c.allsegs[0][0]
    c_path = c_path[np.argsort(-c_path[:, 0])]

    fitres_sphere = minimize(
        lambda x: loss_sphere(c_path, x[0], x[1]),
        x0=[flags.Lz, 0],
        method="Nelder-Mead",
        bounds=((0, flags.Lz * 50), (-1, 1 / np.sqrt(2))),
    )
    fitres_sphere_2 = minimize(
        lambda x: loss_sphere(c_path, x[0], x[1]),
        x0=[flags.Lz, 1],
        method="Nelder-Mead",
        bounds=((0, flags.Lz * 50), (1 / np.sqrt(2), 1)),
    )
    if fitres_sphere.fun > fitres_sphere_2.fun:
        fitres_sphere = fitres_sphere_2

    R0 = fitres_sphere.x[0]
    cos_theta = fitres_sphere.x[1]
    z0 = -R0 * cos_theta * np.sign(1 / np.sqrt(2) - cos_theta)
    # print(z0, R0)
    # if there are two intersections, take the larger one
    if z0 > R0 / np.sqrt(2):
        # print(cos_theta)
        cos_theta = -np.abs(cos_theta)
        # print("Flipping cos_theta to", cos_theta)

    x_fit = np.linspace(0, flags.Lx, 100)
    z_fit = z0 + np.sqrt(R0**2 - (x_fit - z0) ** 2) * np.sign(
        1 / np.sqrt(2) - cos_theta
    )
    plot_mask = np.bitwise_and(z_fit >= 0, z_fit <= flags.Lz)
    x_fit = x_fit[plot_mask]
    z_fit = z_fit[plot_mask]
    ax_img.plot(x_fit, z_fit, "r--")
    r_contact = z0 + R0 * np.sin(np.arccos(cos_theta)) * np.sign(
        1 / np.sqrt(2) - cos_theta
    )

    return r_contact, cos_theta, (fig, (ax_img, ax_x, ax_z))


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


if __name__ == "__main__":
    phi_mesh = np.linspace(0, 1, 100)
    f_phi_mesh = calc_f_uniform(phi_mesh, chi_phi)
    phi_binodal, phi_spinodal = calc_binodal_spinodal(phi_mesh, f_phi_mesh)
    phi_boundary = min(phi_binodal)
    phi_binodal_center = (phi_binodal[0] + phi_binodal[1]) / 2

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    ax.plot(phi_mesh, f_phi_mesh)
    ax.plot(
        phi_binodal,
        [
            calc_f_uniform(phi_binodal[0], chi_phi),
            calc_f_uniform(phi_binodal[1], chi_phi),
        ],
        "o-",
        label="binodal",
    )
    ax.plot(
        phi_spinodal,
        [
            calc_f_uniform(phi_spinodal[0], chi_phi),
            calc_f_uniform(phi_spinodal[1], chi_phi),
        ],
        "o",
        color="tab:red",
        label="spinodal",
    )
    ax.legend(frameon=False, loc="upper center", ncols=2)
    ax.set_title("$\chi_{\phi}=$" + f"{chi_phi:.2f}")
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$f_{\phi}$")
    fig.savefig(output_foldername + "binodal_spinodal_phi.pdf")

    ##########################################################
    # initial conditions: condensate coating the surface
    psi_x_init = np.ones(flags.nx) * psi_boundary
    psi_z_init = np.ones(flags.nz) * psi_boundary
    phi_bulk = (
        phi_binodal[0] + (phi_spinodal[0] - phi_binodal[0]) * flags.phi_supersaturation
    )
    R_init = 5
    z_init = 0

    phi_init = np.ones((flags.nz, flags.nx)) * phi_bulk
    tanh_fun = lambda x, xcenter, width: 0.5 * (1 + np.tanh((x - xcenter) / width))
    cover_depth = flags.phi_cover_depth
    boundary_repel = flags.boundary_repel
    membrane_dist = np.minimum(z_center[:, np.newaxis], x_center[np.newaxis, :])
    # avoid boundary effects
    membrane_dist = np.sqrt(
        membrane_dist**2
        + np.clip(
            boundary_repel
            - np.minimum(
                flags.Lz - z_center[:, np.newaxis], flags.Lx - x_center[np.newaxis, :]
            ),
            0,
            None,
        )
        ** 2
    )
    phi_init += (phi_binodal[1] - phi_bulk) * tanh_fun(cover_depth, membrane_dist, 1)

    fig, (ax_img, ax_x, ax_z) = plot_profile(phi_init, psi_x_init, psi_z_init)
    fig.savefig(f"{output_foldername}/initial_profile.pdf")
    np.savez(
        output_foldername + "initial_profile.npz",
        phi=phi_init,
        psi_x=psi_x_init,
        psi_z=psi_z_init,
        x_center=x_center,
        z_center=z_center,
    )
    print("Initial profile saved.")

    # relax the initial condition
    phi, psi_x, psi_z = phi_init.copy(), psi_x_init.copy(), psi_z_init.copy()
    # n_steps = flags.n_steps_relax
    n_epoch = flags.n_epoch
    n_steps_per_epoch = flags.n_steps_per_epoch
    dt = flags.dt
    t_trace = np.arange(n_epoch + 1) * n_steps_per_epoch * dt
    sol_trace = [(phi.copy(), psi_x.copy(), psi_z.copy())]
    for epoch in tqdm(range(n_epoch)):
        for _ in range(n_steps_per_epoch):
            phi, psi_x, psi_z = calc_step(
                phi, psi_x, psi_z, dt, phi_boundary, psi_boundary
            )
        sol_trace.append((phi.copy(), psi_x.copy(), psi_z.copy()))

    fig, (ax_img, ax_x, ax_z) = plot_profile(*sol_trace[-1])
    fig.savefig(f"{output_foldername}/final_profile.pdf")
    np.savez(
        output_foldername + "final_profile.npz",
        phi=sol_trace[-1][0],
        psi_x=sol_trace[-1][1],
        psi_z=sol_trace[-1][2],
        x_center=x_center,
        z_center=z_center,
    )

    np.savez(
        output_foldername + "sol_trace.npz",
        t_trace=t_trace,
        phi=np.array([s[0] for s in sol_trace]),
        psi_x=np.array([s[1] for s in sol_trace]),
        psi_z=np.array([s[2] for s in sol_trace]),
        x_center=x_center,
        z_center=z_center,
    )

    print("Relaxed profile saved.")

    def calc_mean_r2(phi):
        r_center = np.sqrt(x_center**2 + z_center[:, np.newaxis] ** 2)
        return np.sum(phi * r_center**2 * dx * dz) / np.sum(phi * dx * dz)

    def calc_mean_r(phi):
        r_center = np.sqrt(x_center**2 + z_center[:, np.newaxis] ** 2)
        return np.sum(phi * r_center * dx * dz) / np.sum(phi * dx * dz)

    r2_trace = []
    r_trace = []
    for step_id, (phi, psi_x, psi_z) in enumerate(tqdm(sol_trace)):
        r2_trace.append(calc_mean_r2(phi))
        r_trace.append(calc_mean_r(phi))
    r_trace = np.array(r_trace)
    r2_trace = np.array(r2_trace)

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    ax.plot(t_trace, r_trace)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\\langle r \\rangle$")
    fig.savefig(f"{output_foldername}/mean_r_trace.pdf")

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    ax.plot(t_trace, np.sqrt(r2_trace))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\sqrt{\\langle r^2 \\rangle}$")
    fig.savefig(f"{output_foldername}/mean_r2_trace.pdf")

    np.savez(
        output_foldername + "r_trace.npz",
        t_trace=t_trace,
        r_trace=r_trace,
        r2_trace=r2_trace,
    )

    # plot snapshots
    print("Plotting snapshots...")
    output_foldername_snaps = output_foldername + "snaps/"
    os.system(f"mkdir -p {output_foldername_snaps}")
    print(output_foldername_snaps)

    vmin = np.min([sol_trace[t_id][0].min() for t_id in range(len(sol_trace))])
    vmax = np.max([sol_trace[t_id][0].max() for t_id in range(len(sol_trace))])
    for t_id in range(len(sol_trace)):
        fig, (ax_img, ax_x, ax_z) = plot_profile(*sol_trace[t_id], vmin=vmin, vmax=vmax)
        ax_img.text(
            0.95,
            0.9,
            f"$t={t_trace[t_id]:.2f}$",
            transform=ax_img.transAxes,
            fontsize=16,
            ha="right",
            color="white",
        )
        fig.savefig(output_foldername_snaps + f"snap_{t_id:05d}.jpg", dpi=300)
        plt.close(fig)

    command = (
        "magick -delay 10 -loop 0 "
        + output_foldername_snaps
        + "snap_*.jpg "
        + output_foldername
        + "snap_movie.mp4"
    )
    print(command)
    os.system(command)
