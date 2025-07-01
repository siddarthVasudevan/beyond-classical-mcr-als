"""
Visualization of NMF results for Rxn2 dataset with 0% noise for a specific combination
of known concentrations.

The visualizations include:

1. Initial guess of concentration profiles from SNPA compared to ground truth
2. Initial guess of pure component spectra from SNPA compared to ground truth
3. Predicted concentration profiles from three different NMF algorithms (FroALS,
   FroFPGM, MinVol) compared to ground truth, with relative MSE values
4. Predicted pure component spectra from all three NMF algorithms compared to
   ground truth, organized in a 3×3 grid with relative MSE values
"""

import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from mcrnmf import FroALS, SNPA, FroFPGM, MinVol

import figure_templates

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


def get_W_truth_data():
    """
    Load the ground truth pure component spectra.

    Returns
    -------
    intensity : numpy.ndarray
        Pure component spectra intensity values.
    wavenumber : numpy.ndarray
        Wavenumber values corresponding to the intensity data.
    """
    fname = "pure_spectra.csv"
    fpath = osp.join(os.getcwd(), "datasets", "rxn", fname)
    data_arr = np.loadtxt(fpath, delimiter=",", skiprows=1)
    intensity = data_arr[:, 1:]
    wavenumber = data_arr[:, 0]

    return intensity, wavenumber


dataset = "rxn2"
plots_save_dir_path = osp.join(os.getcwd(), "rxn-figures")
noise_perc = "0"
x_fname = f"X_100dPt_{dataset}_{noise_perc}PercNoise.csv"
h_fname = f"H_100dPt_{dataset}.csv"
data_path = osp.join(os.getcwd(), "datasets", "rxn")
X = np.loadtxt(osp.join(data_path, x_fname), delimiter=",")
H = np.loadtxt(osp.join(data_path, h_fname), delimiter=",")
W, wavenumbers = get_W_truth_data()
time = np.arange(H.shape[1])

# timepoints for equality constraint
idx_ls = [0, -1, 45, 30]
known_H = np.full_like(H, fill_value=np.nan)
known_H[:, idx_ls] = H[:, idx_ls]

# Initial guess and plot of it
rank = 3
snpa = SNPA(rank=rank)
snpa.fit(X)
# get initial guess in the same order as ground truth
sort_order = np.argsort(snpa.col_indices_ls)
Wi = snpa.W[:, sort_order]
Hi = snpa.H[sort_order, :]

# labels
comp_label_ls = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$"]
time_label = "Time [a.u.]"
conc_label = "Mole fraction"
intensity_label = "Intensity [a.u.]"
wavenumber_label = "Wavenumber [1/cm]"

# plot of the initial guess of concentration profiles
fig_obj = figure_templates.SingleColumn()
fig, axes = fig_obj.create_figure()
color_ls = ["C0", "C1", "C2"]
for i in range(rank):
    axes.plot(
        time,
        Hi[i, :],
        "--",
        linewidth=1,
        color=color_ls[i],
        label=comp_label_ls[i] + " (SNPA)",
    )
    axes.plot(
        time,
        H[i, :],
        color=color_ls[i],
        linewidth=2.5,
        alpha=0.5,
        label=comp_label_ls[i] + " (Ground truth)",
    )
axes.set_xlabel(time_label)
axes.set_ylabel(conc_label)
fig.suptitle("Initial guess of Concentration profiles from SNPA")
fig_obj.style_axes(axes)
save_fname = "example_snpa_conc_res_rxn2.pdf"
fig_obj.finalize(fig=fig, filename=osp.join(plots_save_dir_path, save_fname))

# plot of initial guess of pure spectra
fig_obj = figure_templates.GridMxN(rows=1, cols=3, height=2, legend_height_ratio=0.1)
fig, axes = fig_obj.create_figure()
for i in range(rank):
    axes[i].plot(
        wavenumbers,
        Wi[:, i],
        "--",
        linewidth=1,
        color="C0",
        label="SNPA",
    )
    axes[i].plot(
        wavenumbers,
        W[:, i],
        color="C3",
        linewidth=2,
        alpha=0.4,
        label="Ground truth",
    )
    axes[i].set_xlabel(wavenumber_label)
    if i == 0:
        axes[i].set_ylabel(intensity_label)
    axes[i].set_title(comp_label_ls[i])
fig.suptitle("Initial guess of Pure spectra from SNPA")
fig_obj.style_axes(axes)
save_fname = "example_snpa_spectra_res_rxn2.pdf"
fig_obj.finalize(fig=fig, filename=osp.join(plots_save_dir_path, save_fname))

# decomposition with different NMF models
iter_max = 2000
tol = 1e-4
constraint_kind = 4
als = FroALS(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)
als.fit(X, Wi, Hi, known_H=known_H)
fpgm = FroFPGM(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    inner_iter_max=20,
    tol=tol,
)
fpgm.fit(X, Wi, Hi, known_H=known_H, preprocess_scale_WH=False)
mvol = MinVol(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    inner_iter_max=20,
    tol=tol,
)
mvol.fit(X, Wi, Hi, known_H=known_H, preprocess_scale_WH=False)


# some functions to return the relative MSE
def get_rel_W(method_obj, W_truth):
    return np.linalg.norm(W_truth - method_obj.W) / np.linalg.norm(W_truth) * 100


def get_rel_H(method_obj, H_truth):
    return np.linalg.norm(H_truth - method_obj.H) / np.linalg.norm(H_truth) * 100


# plot of the predicted concentration profiles
fig_obj = figure_templates.GridMxN(
    rows=1, cols=3, height=2.75, legend_height_ratio=0.12
)
fig, axs = fig_obj.create_figure()
for i in range(rank):
    axs[0].plot(
        als.H[i, :],
        "--",
        linewidth=1,
        color=color_ls[i],
        label=comp_label_ls[i] + " (Prediction)",
    )
    axs[0].plot(
        H[i, :],
        "-",
        color=color_ls[i],
        linewidth=2,
        alpha=0.4,
        label=comp_label_ls[i] + " (Ground truth)",
    )
axs[0].set_ylabel(conc_label)
axs[0].set_title(
    f"FroALS\n\nRel-MSE(H): {get_rel_H(method_obj=als, H_truth=H):.2f} \\%"
)
for i in range(rank):
    axs[1].plot(
        fpgm.H[i, :],
        "--",
        linewidth=1,
        color=color_ls[i],
    )
    axs[1].plot(H[i, :], "-", color=color_ls[i], linewidth=2, alpha=0.5)
axs[1].set_title(
    f"FroFPGM\n\nRel-MSE(H): {get_rel_H(method_obj=fpgm, H_truth=H):.2f} \\%"
)
for i in range(rank):
    axs[2].plot(
        mvol.H[i, :],
        "--",
        color=color_ls[i],
        linewidth=1,
    )
    axs[2].plot(H[i, :], "-", color=color_ls[i], linewidth=2, alpha=0.5)
axs[2].set_title(
    f"MinVol\n\nRel-MSE(H): {get_rel_H(method_obj=mvol, H_truth=H):.2f} \\%"
)
for i in range(3):
    axs[i].set_xlabel(time_label)
    if i > 0:
        axs[i].set_yticklabels([])
fig_obj.style_axes(axes=axs)
save_fname = "an_example_pred_conc_from_nmf_rxn2.pdf"
fig_obj.finalize(
    fig=fig,
    filename=osp.join(plots_save_dir_path, save_fname),
    legend_ncol=3,
)

# plot of the predicted pure spectra
fig_obj = figure_templates.GridMxN(
    rows=3, cols=3, width=6.3, height=4.5, legend_height_ratio=0.07
)
fig, axs = fig_obj.create_figure()

method_labels = ["FroALS", "FroFPGM", "MinVol"]
for col, method in enumerate([als, fpgm, mvol]):
    for comp in range(rank):
        idx_axes = comp * 3 + col
        axs[idx_axes].plot(
            wavenumbers,
            method.W[:, comp],
            "--",
            color="C0",
            label="Prediction",
            linewidth=1,
        )
        axs[idx_axes].plot(
            wavenumbers,
            W[:, comp],
            "-",
            color="C3",
            alpha=0.4,
            label="Ground truth",
            linewidth=2,
        )
        if comp == 0:
            axs[idx_axes].set_title(
                f"{method_labels[col]}\n\nRel-MSE(W): "
                f"{get_rel_W(method_obj=method, W_truth=W):.2f}\\%",
                fontsize=fig_obj.font_sizes["title"],
            )
        if comp == 2:
            axs[idx_axes].set_xlabel(
                wavenumber_label, fontsize=fig_obj.font_sizes["xlabel"]
            )
        if col == 0:
            axs[idx_axes].set_ylabel(
                intensity_label, fontsize=fig_obj.font_sizes["ylabel"]
            )
            axs[idx_axes].text(
                -0.35,
                0.5,
                comp_label_ls[comp],
                fontsize=fig_obj.font_sizes["title"],
                rotation=90,
                va="center",
                ha="center",
                transform=axs[idx_axes].transAxes,
            )

fig_obj.style_axes(axes=axs)
save_fname = "an_example_pred_spectra_from_nmf_rxn2.pdf"
fig_obj.finalize(
    fig=fig, filename=osp.join(plots_save_dir_path, save_fname), legend_ncol=2
)
