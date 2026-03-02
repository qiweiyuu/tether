from jax import config

config.update("jax_enable_x64", True)

# reserve GPU memory 90% for JAX; for visualization nodes, request less GPU RAM.
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp
from jax import jit, random, lax, vmap, grad
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
        # "text.usetex": True,  # this is weird on the cluster for some reason...
    }
)

# %%
import curved_tubule_one_side_v1 as curve_pde
import importlib

importlib.reload(curve_pde)

# %%
parser = argparse.ArgumentParser(description="ZTS_radial_v1_debug")

# parameters for the grid
parser.add_argument("--Lx", type=float, default=30, help="radial domain size")
parser.add_argument("--Lr1", type=float, default=5, help="vertical domain size")
parser.add_argument("--Lr2", type=float, default=15, help="vertical domain size")
parser.add_argument(
    "--nx", type=int, default=128, help="number of points in the radial direction"
)
parser.add_argument(
    "--nr", type=int, default=64, help="number of points in the vertical direction"
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
parser.add_argument("--dt", type=float, default=1e-4 / 4, help="time step size")


# bulk tether concentration
parser.add_argument(
    "--psi-bulk", type=float, default=0.05, help="bulk tether concentration"
)

# initial conditions
parser.add_argument(
    "--init-x-center",
    type=float,
    default=20,
    help="x center of the initial droplet",
)
parser.add_argument(
    "--init-z-center",
    type=float,
    default=4,
    help="z center of the initial droplet",
)
parser.add_argument(
    "--init-radius", type=float, default=3, help="radius of the initial droplet"
)

# shape of the lower wall to use
parser.add_argument(
    "--lower-wall-fun",
    type=int,
    default=2,
    help="which lower wall function to use",
)
parser.add_argument(
    "--z0", type=float, default=0.5, help="z0 parameter for lower wall function"
)
parser.add_argument(
    "--z1", type=float, default=4.0, help="z1 parameter for lower wall function"
)
parser.add_argument(
    "--w", type=float, default=5.0, help="w parameter for lower wall function"
)

## number of steps
parser.add_argument("--num-epoch", type=int, default=300, help="number of epochs")
parser.add_argument(
    "--steps-per-epoch", type=int, default=100000, help="number of steps per epoch"
)

parser.add_argument(
    "--init-profile",
    type=str,
    default=None,
    help="initial profile for phi",
)

parser.add_argument(
    "--outputdir",
    type=str,
    default="data_curved_tubule_v1_test/",
    help="output directory",
)
flags = parser.parse_args()

chi_psi, chi_phi = flags.chi_psi, flags.chi_phi
lmda_phi, lmda_psi = flags.lmda_phi, flags.lmda_psi
h0, h_phi_psi = flags.h0, flags.h_phi_psi
M_phi, M_psi = flags.m_phi, flags.m_psi
psi_boundary = flags.psi_bulk

if M_psi < 0.001:
    decimal_M_psi = int(np.ceil(-np.log10(M_psi + 1e-12)))
    output_foldername = (
        flags.outputdir
        + f"/chi_psi_{chi_psi}_chi_phi_{chi_phi}_h0_{h0}_h1_{h_phi_psi}/"
        + f"psi_{psi_boundary:.3f}_mpsi_{M_psi:.0{decimal_M_psi}f}/"
    )
else:
    output_foldername = (
        flags.outputdir
        + f"/chi_psi_{chi_psi}_chi_phi_{chi_phi}_h0_{h0}_h1_{h_phi_psi}/"
        + f"psi_{psi_boundary:.3f}_mpsi_{M_psi:.3f}/"
    )
print(f"Output folder: {output_foldername}")
os.system(f"mkdir -p {output_foldername}")

# %% [markdown]
# ## Determine binodal and spinodal from free energy

# %%
phi_mesh = np.linspace(0, 1, 100)
f_phi_mesh = curve_pde.calc_f_uniform(phi_mesh, chi_phi)
phi_binodal, phi_spinodal = curve_pde.calc_binodal_spinodal(phi_mesh, f_phi_mesh)
phi_boundary = min(phi_binodal)
phi_binodal_center = (phi_binodal[0] + phi_binodal[1]) / 2

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(phi_mesh, f_phi_mesh)
ax.plot(
    phi_binodal,
    [
        curve_pde.calc_f_uniform(phi_binodal[0], chi_phi),
        curve_pde.calc_f_uniform(phi_binodal[1], chi_phi),
    ],
    "o-",
    label="binodal",
)
ax.plot(
    phi_spinodal,
    [
        curve_pde.calc_f_uniform(phi_spinodal[0], chi_phi),
        curve_pde.calc_f_uniform(phi_spinodal[1], chi_phi),
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

# %% [markdown]
# ### Tether concentration in the condensate

psi_l_fun = (
    lambda psi_g: psi_g
    * np.exp(h_phi_psi * (phi_binodal[1] - phi_binodal[0]))
    / (1 + psi_g * (np.exp(h_phi_psi * (phi_binodal[1] - phi_binodal[0])) - 1))
)

psi_g = flags.psi_bulk
psi_l = psi_l_fun(psi_g)

### define lower wall and the solver


def lower_wall_1(x, z0=0.5, z1=4.0):
    return z0 + z1 * ((x - flags.Lx / 2) / (flags.Lx / 2)) ** 2


def lower_wall_2(x, z0=0.5, z1=4.0):
    return z0 + z1 * x / flags.Lx


lower_wall_all = [lower_wall_1, lower_wall_2]
lower_wall = lambda x: lower_wall_all[flags.lower_wall_fun - 1](
    x, z0=flags.z0, z1=flags.z1
)

PhaseFieldSys = curve_pde.PhaseFieldSimulator(
    Lx=flags.Lx,
    Lr1=flags.Lr1,
    Lr2=flags.Lr2,
    nx=flags.nx,
    nr=flags.nr,
    chi_psi=chi_psi,
    chi_phi=chi_phi,
    lmda_phi=lmda_phi,
    lmda_psi=lmda_psi,
    h0=h0,
    h1=h_phi_psi,
    M_phi=M_phi,
    M_psi=M_psi,
    psi_boundary=psi_boundary,
    dt=flags.dt,
    wall=lower_wall,
)

# %%
fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(PhaseFieldSys.x_c, lower_wall(PhaseFieldSys.x_c), label="lower wall")
ax.set_xlim(0, flags.Lx)
ax.set_xlabel("$x$")
ax.set_ylabel("$r$")

# %%
nx, nr = PhaseFieldSys.nx, PhaseFieldSys.nr
x_center, r_center = PhaseFieldSys.x_c, PhaseFieldSys.r_c
x_grid, r_grid = jnp.meshgrid(x_center, r_center, indexing="ij")
mask_inside = PhaseFieldSys.mask_inside

if flags.init_profile is not None:
    data_init = np.load(flags.init_profile)
    phi_init = data_init["phi"]
    psi_init = data_init["psi"]
else:
    phi_init = jnp.ones((nx, nr)) * phi_binodal[0]
    # initial condition
    xc, rc = 20, 6
    r_droplet = 5
    mask_droplet_init = (
        np.tanh((r_droplet - np.sqrt((x_grid - xc) ** 2 + (r_grid - rc) ** 2)) / 1) + 1
    ) / 2
    phi_init = phi_init + mask_droplet_init * (phi_binodal[1] - phi_binodal[0])
    phi_init = jnp.where(mask_inside, phi_init, jnp.nan)
    psi_init = jnp.ones((nx)) * psi_boundary

    dist_boundary = np.sqrt((x_center - xc) ** 2 + (lower_wall(x_center) - rc) ** 2)
    mask_psi_inside = (np.tanh((r_droplet - dist_boundary) / flags.lmda_psi) + 1) / 2
    psi_init = psi_init + mask_psi_inside * (psi_l_fun(psi_g) - psi_boundary)

fig, ax1, ax2 = curve_pde.plot_profile(
    phi_init,
    psi_init,
    Lx=flags.Lx,
    Lr1=flags.Lr1,
    Lr2=flags.Lr2,
    x_center=x_center,
    r_center=r_center,
)
# ax2.set_ylim(0.0, 0.1)
ax1.plot(x_center, lower_wall(x_center), "--", color="black", lw=2)
ax1.set_aspect(1)
fig.savefig(output_foldername + "init_profile.pdf")

num_epoch = flags.num_epoch
step_per_epoch = flags.steps_per_epoch
phi_psi_trace = PhaseFieldSys.run(
    phi0=phi_init,
    psi0=psi_init,
    num_epochs=num_epoch,
    steps_per_epoch=step_per_epoch,
    progress=True,
)

phi_final, psi_final = phi_psi_trace[-1]

fig, ax1, ax2 = curve_pde.plot_profile(
    phi_final,
    psi_final,
    Lx=flags.Lx,
    Lr1=flags.Lr1,
    Lr2=flags.Lr2,
    x_center=x_center,
    r_center=r_center,
)
# ax2.set_ylim(0.0, 0.1)
ax1.plot(x_center, lower_wall(x_center), "--", color="black", lw=2)
ax1.set_aspect(1)

fig.savefig(output_foldername + "final_profile.pdf")

# save the trace
t_trace = np.arange(num_epoch) * step_per_epoch * flags.dt

np.savez(
    output_foldername + "trace.npz",
    t_trace=t_trace,
    phi_trace=[phi_psi_trace[i][0] for i in range(num_epoch)],
    psi_trace=[phi_psi_trace[i][1] for i in range(num_epoch)],
)

# %%
phi_mean_trace = np.zeros((num_epoch))
psi_mean_trace = np.zeros((num_epoch))
phi_diff_trace = np.zeros((num_epoch))
for i in range(num_epoch):
    phi_mean_trace[i] = jnp.nanmean(phi_psi_trace[i][0] * PhaseFieldSys.vol_frac)
    psi_mean_trace[i] = jnp.mean(phi_psi_trace[i][1] * PhaseFieldSys.seg_c)
    if i == 0:
        phi_diff_trace[i] = (
            jnp.linalg.norm(np.where(mask_inside, phi_psi_trace[i][0] - phi_init, 0.0))
            / t_trace[0]
        )
    else:
        phi_diff_trace[i] = jnp.linalg.norm(
            np.where(mask_inside, phi_psi_trace[i][0] - phi_psi_trace[i - 1][0], 0.0)
        ) / (t_trace[i] - t_trace[i - 1])


fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, phi_diff_trace, "o-", clip_on=False)
ax.set_ylim(0, None)
ax.set_xlim(0, None)
ax.set_xlabel("$t$")
ax.set_ylabel("$\|\Delta \phi \|/ \Delta t$")
fig.savefig(output_foldername + "phi_diff_trace.pdf")
ax.set_yscale("log")
ax.set_ylim(None, None)
fig.savefig(output_foldername + "phi_diff_trace_log.pdf")

# %% [markdown]
# Compute the center of mass


# %%
def calc_com_cylindrical(phi, PhaseFieldSys):
    xsum = np.nansum(
        phi
        * PhaseFieldSys.vol_frac
        * PhaseFieldSys.x_c[:, None]
        * PhaseFieldSys.mask_inside
        * PhaseFieldSys.r_c[None, :]  # weight by r in cylindrical coordinates
    )
    rsum = np.nansum(
        phi
        * PhaseFieldSys.vol_frac
        * PhaseFieldSys.r_c[None, :]
        * PhaseFieldSys.mask_inside
        * PhaseFieldSys.r_c[None, :]  # weight by r in cylindrical coordinates
    )
    phisum = np.nansum(
        phi
        * PhaseFieldSys.vol_frac
        * PhaseFieldSys.mask_inside
        * PhaseFieldSys.r_c[None, :]
    )
    return xsum / phisum, rsum / phisum


# %%
phi_com_threshold = (phi_binodal[0] + phi_binodal[1]) / 2
phi_trace = [phi_psi_trace[i][0] for i in range(num_epoch)]
com_trace = np.array(
    # [calc_com_cylindrical(phi - phi_binodal[0], PhaseFieldSys) for phi in phi_trace]
    [
        calc_com_cylindrical(np.maximum(phi - phi_com_threshold, 0), PhaseFieldSys)
        for phi in phi_trace
    ]
)

# %%
fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, com_trace[:, 0], "-o")
# ax.set_ylim(0, None)
ax.legend(frameon=False, ncol=2)
ax.set_xlim(0, None)
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Center of mass $\langle x \rangle_{\delta \phi}$")
fig.savefig(output_foldername + f"phi_xc_trace.pdf")

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
ax.plot(t_trace, com_trace[:, 1], "-o")
# ax.set_ylim(0, None)
ax.legend(frameon=False, ncol=2)
ax.set_xlim(0, None)
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Center of mass $\langle r \rangle_{\delta \phi}$")
fig.savefig(output_foldername + f"phi_rc_trace.pdf")

# %%
# Make snapshots and a movie

psi_max = np.max([res[1].max() for res in phi_psi_trace])
psi_plot_max = 0.05 * (1 + psi_max // 0.05)

output_foldername_snaps = output_foldername + "snaps/"
os.system(f"mkdir -p {output_foldername_snaps}")
print(output_foldername_snaps)
for t_id in range(num_epoch):
    fig, ax1, ax2 = curve_pde.plot_profile(
        phi_psi_trace[t_id][0],
        phi_psi_trace[t_id][1],
        Lx=flags.Lx,
        Lr1=flags.Lr1,
        Lr2=flags.Lr2,
        x_center=x_center,
        r_center=r_center,
    )
    # ax2.set_ylim(0.0, 0.1)
    ax1.plot(x_center, lower_wall(x_center), "--", color="black", lw=2)
    ax1.set_aspect(1)
    ax2.set_ylim(0, psi_plot_max)
    fig.savefig(output_foldername_snaps + f"relax_{t_id:05d}.jpg", dpi=300)
    plt.close(fig)

command = (
    "magick -delay 10 -loop 0 "
    + output_foldername_snaps
    + "relax_*.jpg "
    + output_foldername
    + "relax.mp4"
)
print(command)
