"""
Visualization module for results from rxn dataset with varying noise levels.

This module provides a class for visualizing and analyzing rxn data with different noise
levels. It includes methods for plotting pure spectra, principal component analysis,
time series spectra, and error heatmaps for different NMF models.
"""

import ast
import os
import os.path as osp

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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


class RxnAbcWithNoisePlotter:
    """
    Class for plotting and analyzing results from rxn dataset with varying noise levels.

    This class provides methods for visualizing pure spectra, principal component
    analysis, time series data, and error heatmaps for different NMF models at different
    noise levels.

    Attributes
    ----------
    dpath : str
        Path to the reaction datasets directory.
    combination_res_dpath : str
        Path to the combined results directory.
    plots_save_dir_path : str
        Path to save generated plots.
    tol_str : str
        Tolerance string used in filenames.
    combination_res_df : pandas.DataFrame
        Combined results from all models.
    pspectra : numpy.ndarray
        Pure component spectra intensity values.
    wavenumbers : numpy.ndarray
        Wavenumber values corresponding to the intensity data.
    datasets : list
        List of dataset names.
    dataset_labels : dict
        Mapping from dataset names to formatted labels.
    pspectra_labels : list
        Labels for pure component spectra.
    noise_perc_levels : list
        Noise percentage levels.
    X_dict : dict
        Dictionary containing spectroscopic data for each dataset and noise level.
    """

    def __init__(self, plots_save_dir_name="rxn-figures", tol_str="tol_1e-04"):
        """
        Initialize the plotter with paths and load data.

        Parameters
        ----------
        plots_save_dir_name : str, optional
            Name of directory to save plots, default is "rxn-figures".
        tol_str : str, optional
            Tolerance string used in filenames, default is "tol_1e-04".
        """
        self.dpath = osp.join(os.getcwd(), "datasets", "rxn")
        self.combination_res_dpath = osp.join(os.getcwd(), "rxn-comb-results")
        self.plots_save_dir_path = osp.join(os.getcwd(), plots_save_dir_name)
        if osp.exists(self.plots_save_dir_path) is False:
            os.makedirs(self.plots_save_dir_path)

        self.tol_str = tol_str
        self.combination_res_df = self._load_combination_results()
        self.pspectra, self.wavenumbers = get_W_truth_data()
        self.datasets = ["rxn1", "rxn2"]
        self.dataset_labels = {"rxn1": r"$\mathtt{Rxn1}$", "rxn2": r"$\mathtt{Rxn2}$"}
        self.pspectra_labels = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$"]
        self.noise_perc_levels = [0, 2, 5]  # in percentage
        self.X_dict = self._load_X_data()

    def _load_combination_results(self):
        """
        Load data from all result files and combine into a single DataFrame.

        Returns
        -------
        pandas.DataFrame
            Combined results from all models.
        """
        df_ls = []
        for fi in os.listdir(self.combination_res_dpath):
            if fi.endswith(".csv") and (self.tol_str in fi):
                fpath = osp.join(self.combination_res_dpath, fi)
                df_ls.append(pd.read_csv(fpath, header=0))

        combined_df = pd.concat(df_ls, ignore_index=True)
        combined_df["points"] = combined_df["points"].apply(ast.literal_eval)

        return combined_df

    def _load_X_data(self):
        """
        Load spectroscopic data for all datasets and noise levels.

        Creates a nested dictionary with dataset names as first-level keys and noise
        percentage values as second-level keys.

        Returns
        -------
        dict
            Dictionary containing spectroscopic data for each dataset and noise level.
        """
        X_dict = {}
        for dataset in self.datasets:
            X_dict[dataset] = {}
            for noise in self.noise_perc_levels:
                fname = f"X_100dPt_{dataset}_{noise}PercNoise.csv"
                X_dict[dataset][noise] = np.loadtxt(
                    osp.join(self.dpath, fname), delimiter=",", dtype=np.float64
                )

        return X_dict

    def plot_pure_spectra(self):
        """
        Plot the pure component spectra.

        Creates a figure showing the pure component spectra for all components
        with different line styles.
        """
        fig_obj = figure_templates.SingleColumn()
        fig, ax = fig_obj.create_figure()

        linestyle_ls = [":", "--", "-"]

        for i in range(self.pspectra.shape[1]):
            ax.plot(
                self.wavenumbers,
                self.pspectra[:, i],
                linestyle=linestyle_ls[i],
                label=self.pspectra_labels[i],
                linewidth=fig_obj.linewidth,
            )

        ax.set_xlabel("Wavenumber [1/cm]", fontsize=fig_obj.font_sizes["xlabel"])
        ax.set_ylabel("Intensity [a.u.]", fontsize=fig_obj.font_sizes["ylabel"])
        fig_obj.style_axes(ax=ax)
        save_fname = "pure_spectra.pdf"
        fig_obj.finalize(
            fig=fig,
            filename=osp.join(self.plots_save_dir_path, save_fname),
        )

    def plot_pc_at_noise(self, noise: int = 0):
        """
        Plot the first two principal components of the reaction datasets at a given noise level.

        Parameters
        ----------
        noise : int, optional
            Noise percentage level to plot, default is 0.
        """
        # the loading vectors from PCA is obtained by fitting the data to the pure
        # spectra
        pca = PCA(n_components=2)
        pspectra_pc = pca.fit_transform(X=self.pspectra.T)

        # create an array to plot the convex hull
        hull = np.vstack([pspectra_pc, pspectra_pc[0, :][np.newaxis, :]])

        # perform PCA on the X data
        X_pc_dict = {}
        for dataset, noise_dict in self.X_dict.items():
            X_pc_dict[dataset] = {}
            for noise_lvl, X in noise_dict.items():
                X_pc_dict[dataset][noise_lvl] = pca.transform(X=X.T)

        fig_obj = figure_templates.SingleColumn(
            width=2.4, height=2.4, legend_position="bottom", legend_height_ratio=0.18
        )
        fig, ax = fig_obj.create_figure()
        marker_dict = {"rxn1": "o", "rxn2": "^"}

        # plot the vertices of the convex hull
        ax.plot(
            pspectra_pc[:, 0],
            pspectra_pc[:, 1],
            marker="s",
            color="k",
            label="Vertex",
            linestyle="None",
            markersize=9,
            markerfacecolor="None",
            zorder=-1,
        )

        # annotate the pspectra vertex
        xytext_dict = {
            r"$\mathcal{A}$": (15, 0),
            r"$\mathcal{B}$": (-15, 0),
            r"$\mathcal{C}$": (-15, 0),
        }
        for i, label in enumerate(self.pspectra_labels):
            ax.annotate(
                label,
                (pspectra_pc[i, 0], pspectra_pc[i, 1]),
                xytext=xytext_dict[label],
                textcoords="offset points",
                fontsize=fig_obj.font_sizes["xlabel"],
                ha="left" if i in [1, 2] else "right",
                va="center",
            )

        # draw the hull
        ax.plot(
            hull[:, 0],
            hull[:, 1],
            color="k",
            linestyle="-",
            linewidth=0.75,
            label="Hull",
            zorder=1,
        )
        for dataset in X_pc_dict:
            X_pc = X_pc_dict[dataset][noise]
            ax.plot(
                X_pc[:, 0],
                X_pc[:, 1],
                linestyle="None",
                marker=marker_dict[dataset],
                markersize=3,
                markerfacecolor="None",
                markeredgewidth=0.5,
                zorder=0,
                label=rf"{self.dataset_labels[dataset]} ({noise}\% noise)",
            )

        ax.set_xlabel("PC-1", fontsize=fig_obj.font_sizes["xlabel"])
        ax.set_ylabel("PC-2", fontsize=fig_obj.font_sizes["ylabel"])
        ax.set_xlim(xmin=-3.9, xmax=3.1)
        ax.set_ylim(ymin=-3.0, ymax=2.5)
        fig_obj.style_axes(ax=ax, frameon=False)
        save_fname = f"pc_abc_rxn_with_{noise}_perc_noise.pdf"
        fig_obj.finalize(
            fig=fig,
            filename=osp.join(self.plots_save_dir_path, save_fname),
            legend_ncols=2,
        )

    def plot_time_series_spectra(
        self, dataset: str = "rxn2", noise_levels: list[int] = [0, 5]
    ):
        """
        Plot time series spectra of a given dataset at different noise levels.

        Parameters
        ----------
        dataset : str, optional
            Dataset to plot, default is "rxn2".
        noise_levels : list[int], optional
            List of noise percentage levels to plot, default is [0, 5].
        """
        nrows = len(noise_levels)

        fig_obj = figure_templates.GridMxN(rows=nrows, cols=1, width=3, height=3.75)
        fig, axs = fig_obj.create_figure()

        temp_X = self.X_dict[dataset][noise_levels[0]]
        ntime_pts = temp_X.shape[1]
        time = np.arange(100)
        norm = Normalize(vmin=0, vmax=ntime_pts - 1)
        for idx_ax, noise_level in enumerate(noise_levels):
            X = self.X_dict[dataset][noise_level]
            for idx_time in range(ntime_pts):
                axs[idx_ax].plot(
                    self.wavenumbers,
                    X[:, idx_time],
                    color=plt.cm.viridis(norm(time[idx_time])),
                    linewidth=0.75,
                )
            axs[idx_ax].set_ylabel(
                "Intensity [a.u.]", fontsize=fig_obj.font_sizes["xlabel"]
            )
            axs[idx_ax].set_title(
                rf"{self.dataset_labels[dataset]} ({noise_level}\% noise)"
            )
        axs[idx_ax].set_xlabel(
            "Wavenumber [1/cm]", fontsize=fig_obj.font_sizes["ylabel"]
        )
        fig.subplots_adjust(
            right=0.85, left=0.15, top=0.9, bottom=0.1, wspace=0.1, hspace=0.6
        )
        cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.55])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        cbar = fig.colorbar(mappable=sm, cax=cbar_ax)
        cbar.set_label("Time [a.u.]", fontsize=fig_obj.font_sizes["xlabel"])
        if dataset == "rxn2":
            fig_obj.set_subplot_labels(axes=axs, labels=["a", "b"])
        elif dataset == "rxn1":
            fig_obj.set_subplot_labels(axes=axs, labels=["a", "b", "c"])

        fig_obj.style_axes(axes=axs)

        save_fname = f"{dataset}_time_series_spectra.pdf"
        fig.savefig(
            osp.join(self.plots_save_dir_path, save_fname),
            dpi=fig_obj.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_error_hmap_single_noise(self, lambda_val: float = 1e-4):
        """
        Plot error heatmaps for all models at a single noise level.

        Creates heatmaps showing X, W, and H errors for each model and dataset
        at each noise level.

        Parameters
        ----------
        lambda_val : float, optional
            Lambda value for MinVol model, default is 1e-4.
        """
        model_ls = ["als", "fpgm", "mvol"]
        error_type_ls = ["% Rel-MSE X", "% Rel-MSE W", "% Rel-MSE H"]
        num_cols = len(self.datasets)
        num_rows = len(error_type_ls)

        for model in model_ls:
            df_model = self.combination_res_df[
                self.combination_res_df["model"] == model
            ]
            if model == "mvol":
                df_model = df_model[df_model["lambda"] == lambda_val]
            for unimodal in [True]:
                df_unimodal = df_model[df_model["unimodal"] == unimodal]
                for noise_level in self.noise_perc_levels:
                    df_noise = df_unimodal[df_unimodal["noise [%]"] == noise_level]
                    fig_obj = figure_templates.GridMxN(
                        rows=num_rows, cols=num_cols, width=2.5, height=2.5
                    )
                    fig, axes = fig_obj.create_figure()
                    vmin = df_noise[error_type_ls].min().min()
                    vmax = df_noise[error_type_ls].max().max()
                    for row, error_type in enumerate(error_type_ls):
                        for col, dataset in enumerate(self.datasets):
                            df = df_noise[df_noise["dataset"] == dataset]
                            ax = axes[row * num_cols + col]
                            error_arr = self._create_error_array(df, error_type)
                            im = ax.imshow(
                                error_arr,
                                cmap="viridis",
                                norm=Normalize(vmin=0, vmax=vmax),
                            )
                            ax.set_xlim(xmin=1, xmax=98)
                            ax.set_ylim(ymin=98, ymax=1)
                            ax.set_yticks([1, 49, 98])
                            ax.set_yticklabels([1, 49, 98])
                            ax.set_xticks([1, 49, 98])
                            ax.set_xticklabels([1, 49, 98])
                            if row == 0:
                                ax.set_title(
                                    rf"{self.dataset_labels[dataset]}"
                                    + f"\n{noise_level}\\% noise",
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                )
                            if row == num_rows - 1:
                                ax.set_xlabel(
                                    r"$t_2$", fontsize=fig_obj.font_sizes["xlabel"]
                                )
                            else:
                                ax.set_xticklabels([])
                            if col == 0:
                                ax.set_ylabel(
                                    r"$t_1$", fontsize=fig_obj.font_sizes["ylabel"]
                                )
                                ax.text(
                                    -0.9,
                                    0.5,
                                    rf"{error_type[-1]}",
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                    rotation=90,
                                    va="center",
                                    ha="center",
                                    transform=ax.transAxes,
                                )
                            else:
                                ax.set_yticklabels([])
                    fig.subplots_adjust(
                        right=0.85,
                        left=0.15,
                        top=0.9,
                        bottom=0.1,
                        wspace=0.2,
                        hspace=0.3,
                    )

                    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.55])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(
                        f"\\% Rel-RMSE",
                        fontsize=fig_obj.font_sizes["xlabel"],
                    )

                    fig_obj.set_subplot_labels(
                        axes=axes,
                        labels=["a", "b", "c", "d", "e", "f"],
                        xpos=0.5,
                        ypos=0.3,
                    )
                    fig_obj.style_axes(axes=axes)
                    save_fname = (
                        f"{model}_all_error_heatmap_{noise_level}_noisePerc_unimodal"
                        f"_{unimodal}_all_datasets.pdf"
                    )

                    fig.savefig(
                        osp.join(self.plots_save_dir_path, save_fname),
                        dpi=fig_obj.dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

    def plot_error_hmap_single_model(self, lambda_val: float = 1e-4):
        """
        Plot error heatmaps for a single model at all noise levels.

        Creates heatmaps showing X, W, and H errors for a given dataset at all three noise
        levels for a single model.

        Parameters
        ----------
        lambda_val : float, optional
            Lambda value for MinVol model, default is 1e-4.
        """
        model_ls = ["als", "fpgm", "mvol"]
        error_type_ls = ["% Rel-MSE X", "% Rel-MSE W", "% Rel-MSE H"]
        num_cols = len(error_type_ls)
        num_rows = len(self.noise_perc_levels)

        for model in model_ls:
            df_model = self.combination_res_df[
                self.combination_res_df["model"] == model
            ]
            if model == "mvol":
                df_model = df_model[df_model["lambda"] == lambda_val]
            for unimodal in [True]:
                df_unimodal = df_model[df_model["unimodal"] == unimodal]
                for dataset in self.datasets:
                    df_dataset = df_unimodal[df_unimodal["dataset"] == dataset]
                    fig_obj = figure_templates.GridMxN(
                        rows=num_rows, cols=num_cols, width=3.5, height=3
                    )
                    fig, axes = fig_obj.create_figure()
                    vmin = df_dataset[error_type_ls].min().min()
                    vmax = df_dataset[error_type_ls].max().max()
                    for row, noise_level in enumerate(self.noise_perc_levels):
                        df = df_dataset[df_dataset["noise [%]"] == noise_level]
                        for col, error_type in enumerate(error_type_ls):
                            ax = axes[row * num_cols + col]
                            error_arr = self._create_error_array(df, error_type)
                            im = ax.imshow(
                                error_arr,
                                cmap="viridis",
                                norm=Normalize(vmin=0, vmax=vmax),
                            )
                            ax.set_xlim(xmin=1, xmax=98)
                            ax.set_ylim(ymin=98, ymax=1)
                            ax.set_yticks([1, 49, 98])
                            ax.set_yticklabels([1, 49, 98])
                            ax.set_xticks([1, 49, 98])
                            ax.set_xticklabels([1, 49, 98])
                            if row == 0:
                                ax.set_title(
                                    rf"{error_type[-1]}",
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                )
                            if row == num_rows - 1:
                                ax.set_xlabel(
                                    r"$t_2$", fontsize=fig_obj.font_sizes["xlabel"]
                                )
                            else:
                                ax.set_xticklabels([])
                            if col == 0:
                                ax.set_ylabel(
                                    r"$t_1$", fontsize=fig_obj.font_sizes["ylabel"]
                                )
                                ax.text(
                                    -0.9,
                                    0.5,
                                    rf"{noise_level}\% noise",
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                    rotation=90,
                                    va="center",
                                    ha="center",
                                    transform=ax.transAxes,
                                )
                            else:
                                ax.set_yticklabels([])
                        # add a single colorbar
                    fig.subplots_adjust(right=0.9)
                    cbar_ax = fig.add_axes([1, 0.25, 0.02, 0.55])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(
                        f"\\% Rel-RMSE",
                        fontsize=fig_obj.font_sizes["xlabel"],
                    )
                    fig.supylabel(
                        rf"{self.dataset_labels[dataset]} dataset",
                        x=0.07,
                        fontsize=fig_obj.font_sizes["title"] - 1,
                    )
                    fig_obj.set_subplot_labels(
                        axes=axes,
                        labels=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                        xpos=0.5,
                        ypos=0.3,
                    )
                    fig_obj.style_axes(axes=axes)
                    save_fname = (
                        f"{model}_all_error_heatmap_{dataset}_unimodal"
                        f"_{unimodal}_all_noise.pdf"
                    )
                    fig_obj.finalize(
                        fig=fig,
                        filename=osp.join(self.plots_save_dir_path, save_fname),
                        legend_style=None,
                    )

    def plot_error_hmap_all_noise(
        self, lambda_val: float = 1e-4, rel_error_thresh: bool = False
    ):
        """
        Plot error heatmaps for all noise levels.

        Creates heatmaps showing X, W, or H errors for a given dataset at all three noise
        levels (0%, 2%, and 5%).

        Parameters
        ----------
        lambda_val : float, optional
            Lambda value for MinVol model, default is 1e-4.
        rel_error_thresh : bool, optional
            Whether to show threshold contours on the heatmaps, default is False.
        """
        error_type_ls = ["% Rel-MSE X", "% Rel-MSE W", "% Rel-MSE H"]
        model_ls = ["als", "fpgm", "mvol"]
        model_label_ls = ["FroALS", "FroFPGM", "MinVol"]
        num_rows = len(self.noise_perc_levels)
        num_cols = len(model_ls)
        for error_type in error_type_ls:
            for dataset in self.datasets:
                df_dataset = self.combination_res_df[
                    self.combination_res_df["dataset"] == dataset
                ]
                for unimodal in [True]:
                    df_unimodal = df_dataset[df_dataset["unimodal"] == unimodal]
                    fig_obj = figure_templates.GridMxN(
                        rows=num_rows, cols=num_cols, width=2.8, height=2.8
                    )
                    fig, axes = fig_obj.create_figure()
                    vmin = min(
                        df_unimodal[df_unimodal["model"] == "als"][error_type].min(),
                        df_unimodal[df_unimodal["model"] == "fpgm"][error_type].min(),
                        df_unimodal[
                            (df_unimodal["model"] == "mvol")
                            & (df_unimodal["lambda"] == lambda_val)
                        ][error_type].min(),
                    )
                    vmax = max(
                        df_unimodal[df_unimodal["model"] == "als"][error_type].max(),
                        df_unimodal[df_unimodal["model"] == "fpgm"][error_type].max(),
                        df_unimodal[
                            (df_unimodal["model"] == "mvol")
                            & (df_unimodal["lambda"] == lambda_val)
                        ][error_type].max(),
                    )
                    for row, noise_level in enumerate(self.noise_perc_levels):
                        df = df_unimodal[df_unimodal["noise [%]"] == noise_level]
                        for col, model in enumerate(model_ls):
                            ax = axes[row * num_cols + col]
                            df_model = df[df["model"] == model]
                            if model == "mvol":
                                df_model = df_model[df_model["lambda"] == lambda_val]
                            error_arr = self._create_error_array(df_model, error_type)
                            im = ax.imshow(
                                error_arr,
                                cmap="viridis",
                                norm=Normalize(vmin=0, vmax=vmax),
                            )
                            # mask for threshold
                            if rel_error_thresh is True:
                                if noise_level == 0:
                                    upper_noise = 1
                                elif noise_level == 2:
                                    upper_noise = 3
                                elif noise_level == 5:
                                    upper_noise = 6
                                mask_thresh = error_arr <= upper_noise
                                ax.contour(
                                    mask_thresh,
                                    levels=[0.5],
                                    colors="red",
                                    linewidths=1,
                                )
                            ax.set_xlim(xmin=1, xmax=98)
                            ax.set_ylim(ymin=98, ymax=1)
                            ax.set_yticks([1, 49, 98])
                            ax.set_yticklabels([1, 49, 98])
                            ax.set_xticks([1, 49, 98])
                            ax.set_xticklabels([1, 49, 98])
                            if row == 0:
                                ax.set_title(
                                    model_label_ls[col],
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                )
                            if row == num_rows - 1:
                                ax.set_xlabel(
                                    r"$t_2$", fontsize=fig_obj.font_sizes["xlabel"]
                                )
                            else:
                                ax.set_xticklabels([])
                            if col == 0:
                                ax.set_ylabel(
                                    r"$t_1$", fontsize=fig_obj.font_sizes["ylabel"]
                                )
                                ax.text(
                                    -0.8,
                                    0.5,
                                    rf"{noise_level}\% noise",
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                    rotation=90,
                                    va="center",
                                    ha="center",
                                    transform=ax.transAxes,
                                )
                            else:
                                ax.set_yticklabels([])
                        # add a single colorbar
                    fig.suptitle(
                        rf"{self.dataset_labels[dataset]} dataset",
                        y=1.05,
                        fontsize=fig_obj.font_sizes["title"],
                    )
                    fig.subplots_adjust(
                        right=0.85,
                        left=0.15,
                        top=0.9,
                        bottom=0.1,
                        wspace=0.2,
                        hspace=0.3,
                    )
                    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.55])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(
                        f"\\% Rel-RMSE({error_type[-1]})",
                        fontsize=fig_obj.font_sizes["xlabel"],
                    )
                    fig_obj.set_subplot_labels(
                        axes=axes,
                        labels=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                        xpos=0.5,
                        ypos=0.3,
                    )
                    fig_obj.style_axes(axes=axes)
                    save_fname = (
                        f"{error_type[-1]}error_heatmap_{dataset}_unimodal"
                        f"_{unimodal}_all_models.pdf"
                    )
                    fig.savefig(
                        osp.join(self.plots_save_dir_path, save_fname),
                        dpi=fig_obj.dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig=fig)

    def plot_error_hmap_all_models(
        self, lambda_val: float = 1e-4, rel_error_thresh: float | None = None
    ):
        """
        Plot error heatmaps for all models.

        Creates heatmaps showing X, W, or H errors for both datasets at a given noise level
        for all models.

        Parameters
        ----------
        lambda_val : float, optional
            Lambda value for MinVol model, default is 1e-4.
        rel_error_thresh : float or None, optional
            Threshold value for error contours, default is None.
        """
        error_type_ls = ["% Rel-MSE X", "% Rel-MSE W", "% Rel-MSE H"]
        model_ls = ["als", "fpgm", "mvol"]
        model_label_ls = ["FroALS", "FroFPGM", "MinVol"]
        num_rows = len(self.datasets)
        num_cols = len(model_ls)
        for error_type in error_type_ls:
            for noise_perc in self.noise_perc_levels:
                df_noise = self.combination_res_df[
                    self.combination_res_df["noise [%]"] == noise_perc
                ]
                for unimodal in [True]:
                    df_unimodal = df_noise[df_noise["unimodal"] == unimodal]
                    fig_obj = figure_templates.GridMxN(
                        rows=num_rows, cols=num_cols, width=3, height=2
                    )
                    fig, axes = fig_obj.create_figure()
                    vmin = min(
                        df_unimodal[df_unimodal["model"] == "als"][error_type].min(),
                        df_unimodal[df_unimodal["model"] == "fpgm"][error_type].min(),
                        df_unimodal[
                            (df_unimodal["model"] == "mvol")
                            & (df_unimodal["lambda"] == lambda_val)
                        ][error_type].min(),
                    )
                    vmax = max(
                        df_unimodal[df_unimodal["model"] == "als"][error_type].max(),
                        df_unimodal[df_unimodal["model"] == "fpgm"][error_type].max(),
                        df_unimodal[
                            (df_unimodal["model"] == "mvol")
                            & (df_unimodal["lambda"] == lambda_val)
                        ][error_type].max(),
                    )
                    for row, dataset in enumerate(self.datasets):
                        df = df_unimodal[df_unimodal["dataset"] == dataset]
                        for col, model in enumerate(model_ls):
                            ax = axes[row * num_cols + col]
                            df_model = df[df["model"] == model]
                            if col == 2:
                                df_model = df_model[df_model["lambda"] == lambda_val]
                            error_arr = self._create_error_array(df_model, error_type)
                            im = ax.imshow(
                                error_arr,
                                cmap="viridis",
                                norm=Normalize(vmin=0, vmax=vmax),
                            )
                            # mask for threshold
                            if rel_error_thresh is not None:
                                mask_thresh = error_arr <= rel_error_thresh
                                ax.contour(
                                    mask_thresh,
                                    levels=[0.5],
                                    colors="red",
                                    linewidths=1,
                                )
                            ax.set_xlim(xmin=1, xmax=98)
                            ax.set_ylim(ymin=98, ymax=1)
                            ax.set_yticks([1, 49, 98])
                            ax.set_yticklabels([1, 49, 98])
                            ax.set_xticks([1, 49, 98])
                            ax.set_xticklabels([1, 49, 98])
                            if row == 0:
                                ax.set_title(
                                    model_label_ls[col],
                                    fontsize=fig_obj.font_sizes["title"] - 1,
                                )
                            if row == num_rows - 1:
                                ax.set_xlabel(
                                    r"$t_2$", fontsize=fig_obj.font_sizes["xlabel"]
                                )
                            else:
                                ax.set_xticklabels([])
                            if col == 0:
                                ax.set_ylabel(
                                    r"$t_1$", fontsize=fig_obj.font_sizes["ylabel"]
                                )
                                ax.text(
                                    -0.95,
                                    0.5,
                                    self.dataset_labels[dataset],
                                    fontsize=fig_obj.font_sizes["title"],
                                    rotation=90,
                                    va="center",
                                    ha="center",
                                    transform=ax.transAxes,
                                )
                            else:
                                ax.set_yticklabels([])
                    # add a single colorbar
                    fig.subplots_adjust(
                        right=0.85,
                        left=0.15,
                        top=0.9,
                        bottom=0.1,
                        wspace=0.2,
                        hspace=0.3,
                    )
                    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.55])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(
                        f"\\% Rel-RMSE({error_type[-1]})",
                        fontsize=fig_obj.font_sizes["xlabel"],
                    )
                    fig_obj.set_subplot_labels(
                        axes=axes,
                        labels=["a", "b", "c", "d", "e", "f"],
                        xpos=0.5,
                        ypos=0.3,
                    )
                    fig_obj.style_axes(axes=axes)
                    save_fname = (
                        f"{error_type[-1]}error_heatmap_{noise_perc}noisePerc_unimodal"
                        f"_{unimodal}_all_models.pdf"
                    )
                    fig.savefig(
                        osp.join(self.plots_save_dir_path, save_fname),
                        dpi=fig_obj.dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig=fig)

    def plot_error_cum_dist(self):
        """
        Plot cumulative distribution of errors.

        Creates plots showing the cumulative distribution of X, W, and H errors
        for all models, datasets, and noise levels.
        """
        error_type_ls = ["% Rel-MSE W", "% Rel-MSE H", "% Rel-MSE X"]
        # cols and rows for the figure
        num_cols = len(self.noise_perc_levels)
        num_rows = len(self.datasets)
        lambda_val_ls = np.sort(self.combination_res_df["lambda"].dropna().unique())
        model_ls = self.combination_res_df["model"].unique()
        alpha_dict = {"als": 0.5, "fpgm": 1, "mvol": 1}
        color_dict = {"als": "g", "fpgm": "b", "mvol": "r"}
        linewidth_delta_dict = {"als": 2, "fpgm": -0.5, "mvol": 0}
        linestyle_dict = {"als": "solid", "fpgm": "dashdot", "mvol": "dotted"}
        label_dict = {"als": "FroALS", "fpgm": "FroFPGM", "mvol": "MinVol"}
        for unimodal in [True]:
            df_uni = self.combination_res_df[
                self.combination_res_df["unimodal"] == unimodal
            ]
            for error_type in error_type_ls:
                for lambda_val in lambda_val_ls:
                    fig_obj = figure_templates.GridMxN(
                        rows=num_rows,
                        cols=num_cols,
                        width=4.4,
                        height=2.5,
                        legend_height_ratio=0.1,
                    )
                    fig, axes = fig_obj.create_figure()
                    xmax = np.max(
                        [
                            df_uni[df_uni["model"] == "als"][error_type],
                            df_uni[df_uni["model"] == "fpgm"][error_type],
                            df_uni[
                                (df_uni["model"] == "mvol")
                                & (df_uni["lambda"] == lambda_val)
                            ][error_type],
                        ]
                    )
                    for row, dataset in enumerate(self.datasets):
                        for col, noise_perc in enumerate(self.noise_perc_levels):
                            df = df_uni[df_uni["noise [%]"] == noise_perc]
                            ax = axes[row * num_cols + col]
                            for model in model_ls:
                                mask = (df["model"] == model) & (
                                    df["dataset"] == dataset
                                )
                                if model == "mvol":
                                    mask = mask & (df["lambda"] == lambda_val)
                                ax.ecdf(
                                    df[mask][error_type],
                                    linewidth=fig_obj.linewidth
                                    + linewidth_delta_dict[model],
                                    linestyle=linestyle_dict[model],
                                    alpha=alpha_dict[model],
                                    label=label_dict[model],
                                    color=color_dict[model],
                                )
                                ax.set_xlim(xmin=-2, xmax=xmax)
                                if row == 0:
                                    ax.set_title(
                                        f"{noise_perc}\\% Noise",
                                        fontsize=fig_obj.font_sizes["title"],
                                    )
                                if row == (num_rows - 1):
                                    ax.set_xlabel(
                                        f"Rel-RMSE({error_type[-1]})\\ [\\%]",
                                        fontsize=fig_obj.font_sizes["xlabel"],
                                    )
                                else:
                                    ax.set_xticklabels([])

                                if col == 0:
                                    ax.set_ylabel(
                                        f"$\\eta$",
                                        fontsize=fig_obj.font_sizes["ylabel"],
                                    )
                                    ax.text(
                                        -0.55,
                                        0.5,
                                        self.dataset_labels[dataset],
                                        fontsize=fig_obj.font_sizes["title"],
                                        rotation=90,
                                        va="center",
                                        ha="center",
                                        transform=ax.transAxes,
                                    )
                                else:
                                    ax.set_yticklabels([])
                                ax.grid(True, alpha=0.5, zorder=--1)

                    fig_obj.style_axes(axes=axes)
                    save_fname = (
                        f"{error_type[-1]}error_cum_dist_lambda_{lambda_val:.0e}_"
                        f"unimodal_{unimodal}.pdf"
                    )
                    fig_obj.finalize(
                        fig=fig,
                        filename=osp.join(self.plots_save_dir_path, save_fname),
                        legend_style="shared",
                        legend_ncol=3,
                        plot_center_corr=0.07,
                    )

    def plot_lambda_dependence(self):
        """
        Plot lambda dependence for MinVol model.

        Creates plots showing how the errors in the MinVol model depend on the
        regularization parameter lambda.
        """
        lambda_val_arr = np.sort(self.combination_res_df["lambda"].dropna().unique())
        # colors for plotting
        norm = Normalize(
            vmin=np.log10(lambda_val_arr.min()), vmax=np.log10(lambda_val_arr.max())
        )
        error_type_ls = ["% Rel-MSE W", "% Rel-MSE H", "% Rel-MSE X"]
        num_cols = len(self.noise_perc_levels)
        num_rows = len(self.datasets)
        # get all the data corresponding to min vol approach
        df = self.combination_res_df[self.combination_res_df["model"] == "mvol"]
        for unimodal in [True]:
            df_uni = df[df["unimodal"] == unimodal]
            for error_type in error_type_ls:
                fig_obj = figure_templates.GridMxN(
                    rows=num_rows,
                    cols=num_cols,
                    width=5,
                    height=3,
                    legend_height_ratio=0.12,
                )
                fig, axes = fig_obj.create_figure()
                xmax = df_uni[error_type].max()
                for row, dataset in enumerate(self.datasets):
                    df_dataset = df_uni[df_uni["dataset"] == dataset]
                    for col, noise_perc in enumerate(self.noise_perc_levels):
                        df_noise = df_dataset[df_dataset["noise [%]"] == noise_perc]
                        ax = axes[row * num_cols + col]
                        for lambda_val in lambda_val_arr:
                            color = plt.cm.viridis(norm(np.log10(lambda_val)))
                            mask = df_noise["lambda"] == lambda_val
                            ax.ecdf(
                                df_noise[mask][error_type],
                                linewidth=fig_obj.linewidth,
                                label=rf"$\lambda = ${lambda_val:.0e}",
                                color=color,
                                # alpha=0.9,
                            )
                        if error_type != "% Rel-MSE X":
                            ax.set_xlim(xmin=-2, xmax=xmax)
                        if row == 0:
                            ax.set_title(
                                f"{noise_perc}\\% Noise",
                                fontsize=fig_obj.font_sizes["title"],
                            )
                        if row == (num_rows - 1):
                            ax.set_xlabel(
                                f"Rel-RMSE({error_type[-1]})\\ [\\%]",
                                fontsize=fig_obj.font_sizes["xlabel"],
                            )
                        else:
                            if error_type != "% Rel-MSE X":
                                ax.set_xticklabels([])
                        if col == 0:
                            ax.set_ylabel(
                                f"$\\eta$",
                                fontsize=fig_obj.font_sizes["ylabel"],
                            )
                            ax.text(
                                -0.55,
                                0.5,
                                self.dataset_labels[dataset],
                                fontsize=fig_obj.font_sizes["title"],
                                rotation=90,
                                va="center",
                                ha="center",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.set_yticklabels([])
                        ax.grid(True, alpha=0.5, zorder=--1)
                fig_obj.style_axes(axes=axes)
                save_fname = (
                    f"minvol_{error_type[-1]}error_dependence_on_lambda_unimodal_"
                    f"{unimodal}.pdf"
                )
                fig_obj.finalize(
                    fig=fig,
                    filename=osp.join(self.plots_save_dir_path, save_fname),
                    legend_style="shared",
                    legend_ncol=3,
                    plot_center_corr=0.05,
                )

    def _create_error_array(self, df, error_type):
        """
        Create a 2D array of errors from DataFrame rows.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing error data.
        error_type : str
            Type of error to extract.

        Returns
        -------
        numpy.ndarray
            2D array of errors.
        """
        error_arr = np.full((100, 100), fill_value=np.nan)
        for i in df.index:
            row, col = df.loc[i, "points"]
            error_arr[row, col] = df.loc[i, error_type]

        return error_arr


if __name__ == "__main__":
    plotter = RxnAbcWithNoisePlotter()
    plotter.plot_pure_spectra()
    plotter.plot_pc_at_noise()
    plotter.plot_time_series_spectra(dataset="rxn1", noise_levels=[0, 2, 5])
    plotter.plot_time_series_spectra(dataset="rxn2", noise_levels=[0, 2, 5])
    plotter.plot_error_hmap_all_models()
    plotter.plot_error_cum_dist()
    plotter.plot_error_hmap_all_noise(rel_error_thresh=True)
    plotter.plot_error_hmap_single_model()
    plotter.plot_error_hmap_single_noise()
    plotter.plot_lambda_dependence()
