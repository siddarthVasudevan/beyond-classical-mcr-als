"""
Raman dataset visualization and analysis module.

This module processes and visualizes Raman dataset, applying FroALS, FroFPGM, and MinVol
NMF approaches to extract component concentrations. It compares results from different
algorithms with and without various constraints (closure, unimodality, equality).

The module also generates visualizations of raw and corrected Raman spectra over time.
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from mcrnmf import FroALS, FroFPGM, MinVol, SNPA
from figure_templates import SingleColumn, GridMxN

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


def load_rxn_spectra():
    DATA_DIR_PATH = osp.join(os.getcwd(), "datasets", "raman")

    fpath_corr = osp.join(DATA_DIR_PATH, "raman.csv")
    data_corr = np.genfromtxt(fpath_corr, delimiter=",", skip_header=1)

    time = np.genfromtxt(
        fpath_corr, delimiter=",", max_rows=1, missing_values="", filling_values=np.nan
    )[1:]
    wv = data_corr[:, 0].astype(dtype=np.float64)
    X_corr = data_corr[:, 1:].astype(dtype=np.float64)

    fpath_raw = osp.join(DATA_DIR_PATH, "raman_raw.csv")
    data_raw = np.genfromtxt(fpath_raw, delimiter=",", skip_header=1)
    X_raw = data_raw[:, 1:].astype(dtype=np.float64)
    return X_corr, X_raw, wv, time


plot_dpath = osp.join(os.getcwd(), "raman-figures")

if not osp.exists(plot_dpath):
    os.makedirs(plot_dpath)

# load the corrected raman data
X_corr, X_raw, wavenumber, time = load_rxn_spectra()
known_H_df = pd.read_csv(
    osp.join(os.getcwd(), "datasets", "raman", "known_H_raman.csv"),
    header=None,
    index_col=0,
)
known_H = known_H_df.to_numpy()

# verify that the corrected data and raw data have the same shape
assert X_raw.shape[0] == X_corr.shape[0]
assert X_raw.shape[1] == X_corr.shape[1]
assert wavenumber.size == X_corr.shape[0]
assert time.size == X_corr.shape[1]

norm = Normalize(vmin=time.min(), vmax=time.max())
wavenumber_label = "Raman shift [1/cm]"
time_label = "Time [h]"
intensity_label = "Intensity [a.u.]"
comp_label_ls = known_H_df.index.to_list()
###################### Plot the corrected raman data ###################################
fig_X_corr = SingleColumn(height=1.8, width=3.4)
fig, ax = fig_X_corr.create_figure()
for i in range(X_corr.shape[1]):
    ax.plot(
        wavenumber, X_corr[:, i], color=plt.cm.viridis(norm(time[i])), linewidth=0.75
    )
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_xlabel(wavenumber_label, fontsize=fig_X_corr.font_sizes["xlabel"])
ax.set_ylabel(intensity_label, fontsize=fig_X_corr.font_sizes["ylabel"])
fig_X_corr.style_axes(ax=ax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(time_label, fontsize=fig_X_corr.font_sizes["xlabel"])
if ax.get_legend():
    ax.get_legend().remove()
fig_X_corr.finalize(fig=fig, filename=osp.join(plot_dpath, "corrected_spectra.pdf"))

#################### Plot the raw raman data ###########################################
fig_X_raw = SingleColumn(height=2.8, width=5)
fig, ax = fig_X_raw.create_figure()
for i in range(X_raw.shape[1]):
    ax.plot(
        wavenumber, X_raw[:, i], color=plt.cm.viridis(norm(time[i])), linewidth=0.75
    )
ax.set_xlabel(wavenumber_label, fontsize=fig_X_corr.font_sizes["xlabel"])
ax.set_ylabel(intensity_label, fontsize=fig_X_corr.font_sizes["ylabel"])
fig_X_corr.style_axes(ax=ax)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(time_label, fontsize=fig_X_corr.font_sizes["xlabel"])
if ax.get_legend():
    ax.get_legend().remove()
fig_X_raw.finalize(fig=fig, filename=osp.join(plot_dpath, "raw_spectra.pdf"))

######################## Perform the NMF ###############################################
rank = 4
iter_max = 2000
tol = 1e-4
constraint_kind = 1
# generate initial guess
snpa = SNPA(rank=rank)
snpa.fit(X=X_corr)
Wi = snpa.W.copy()
Hi = snpa.H.copy()

# unimodal False, closure active
als_only_closure = FroALS(
    rank=rank,
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)
fpgm_only_closure = FroFPGM(
    rank=rank,
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)
mvol_only_closure = MinVol(
    rank=rank,
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)

# fit without equality constraints on H
als_only_closure.fit(X=X_corr, Wi=Wi, Hi=Hi)
fpgm_only_closure.fit(X=X_corr, Wi=Wi, Hi=Hi)
mvol_only_closure.fit(X=X_corr, Wi=Wi, Hi=Hi)

# unimodal True, closure active

als_all_con = FroALS(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)
fpgm_all_con = FroFPGM(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)
mvol_all_con = MinVol(
    rank=rank,
    unimodal={"H": True},
    constraint_kind=constraint_kind,
    iter_max=iter_max,
    tol=tol,
)

# fit with equality constraints on H
als_all_con.fit(X=X_corr, Wi=Wi, Hi=Hi, known_H=known_H)
fpgm_all_con.fit(X=X_corr, Wi=Wi, Hi=Hi, known_H=known_H)
mvol_all_con.fit(X=X_corr, Wi=Wi, Hi=Hi, known_H=known_H)


# plot settings
color_ls = ["C0", "C1", "C2", "C3"]
title_ls = [r"$\mathcal{S}$", r"$\mathcal{I}1$", r"$\mathcal{I}2$", r"$\mathcal{P}$"]
######################### A single plot of both results ########################
fig_obj = GridMxN(rows=2, cols=4, height=3.5, width=6.3, legend_height_ratio=0.08)
fig, axs = fig_obj.create_figure()
axs_index = [2, 3, 0, 1, 6, 7, 4, 5]
for i in range(len(axs_index)):
    if i <= 3:
        als_H = als_only_closure.H
        fpgm_H = fpgm_only_closure.H
        mvol_H = mvol_only_closure.H
        comp_i = i
    else:
        als_H = als_all_con.H
        fpgm_H = fpgm_all_con.H
        mvol_H = mvol_all_con.H
        comp_i = i - 4
    axs[axs_index[i]].plot(
        time, als_H[comp_i, :], "-", color="g", zorder=1, linewidth=0.75, label="FroALS"
    )
    axs[axs_index[i]].plot(
        time,
        fpgm_H[comp_i, :],
        color="b",
        zorder=0,
        linewidth=3,
        alpha=0.3,
        label="FroFPGM",
    )
    axs[axs_index[i]].plot(
        time,
        mvol_H[comp_i, :],
        "--",
        color="r",
        zorder=2,
        linewidth=0.75,
        alpha=0.75,
        label="MinVol",
    )
    axs[axs_index[i]].plot(
        time,
        known_H[comp_i, :],
        "o",
        markersize=3,
        markerfacecolor="none",
        color="k",
        label="Offline",
    )
    axs[axs_index[i]].set_ylim(ymin=-0.02, ymax=1.02)
    axs[axs_index[i]].grid(True, alpha=0.5, zorder=--1)
    if axs_index[i] % 4 == 0:
        if axs_index[i] == 0:
            axs[axs_index[i]].set_ylabel(
                f"\\textbf{{Closure constraint}}\n\nMole fraction",
                fontsize=fig_obj.font_sizes["ylabel"],
            )
        elif axs_index[i] == 4:
            axs[axs_index[i]].set_ylabel(
                f"\\textbf{{All constraints}}\n\nMole fraction",
                fontsize=fig_obj.font_sizes["ylabel"],
            )
    else:
        axs[axs_index[i]].set_yticklabels([])
    if i >= 4:
        axs[axs_index[i]].set_xlabel("Time [h]", fontsize=fig_obj.font_sizes["xlabel"])
        axs[axs_index[i]].set_xticks([0, 2, 4, 6, 8])
        axs[axs_index[i]].set_xticklabels([f"{int(x)}" for x in [0, 2, 4, 6, 8]])
    else:
        axs[axs_index[i]].set_xticks([0, 2, 4, 6, 8])
        axs[axs_index[i]].set_xticklabels([])
        axs[axs_index[i]].set_title(
            "Compound: " + title_ls[axs_index[i]], fontsize=fig_obj.font_sizes["title"]
        )
fig_obj.set_subplot_labels(
    axes=axs,
    labels=["a", "b", "c", "d", "e", "f", "g", "h"],
    xpos=0.4,
    ypos=0.93,
)
fig_obj.style_axes(axes=axs)
save_fname = "raman_exp_comparison_of_constraints.pdf"
fig_obj.finalize(
    fig=fig,
    filename=osp.join(plot_dpath, save_fname),
    legend_style="shared",
    legend_ncol=4,
)
