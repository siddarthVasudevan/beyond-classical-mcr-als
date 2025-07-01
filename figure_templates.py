"""
Matplotlib figure creation and styling utilities.

This module provides classes for creating and styling matplotlib figures with consistent
formatting for publication-quality graphics.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


class BaseFigure:
    """
    Base class for figure creation with standard styling.

    This class defines common properties and methods for figure styling
    that can be inherited by more specific figure classes.

    Parameters
    ----------
    dpi : int
        Resolution in dots per inch for saved figures.
    font_sizes : dict
        Dictionary of font sizes for different figure elements.
    linewidth : float
        Default line width for plots.
    """

    def __init__(self):
        """Initialize with default styling parameters."""
        self.dpi = 300
        self.font_sizes = {
            "xlabel": 10,
            "ylabel": 10,
            "title": 12,
            "tick": 8,
            "legend": 8,
            "subplot_label": 12,
        }
        self.linewidth = 1.5
        plt.rcParams.update({"font.size": self.font_sizes["tick"]})

    def style_axes(self, ax):
        """
        Apply standard styling to axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to style.
        """
        ax.tick_params(axis="both", which="major", labelsize=self.font_sizes["tick"])

    def finalize(self, fig, filename):
        """
        Finalize and save the figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        filename : str
            Path where the figure will be saved.
        """
        plt.tight_layout()
        fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)


class SingleColumn(BaseFigure):
    """
    Single column figure with customizable legend position.

    This class creates a single-panel figure suitable for publication
    in single-column format with options for legend placement.

    Parameters
    ----------
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.
    legend_position : str
        Position of the legend ('best', 'bottom', etc.).
    legend_height_ratio : float
        Ratio of legend height to figure height when legend_position is 'bottom'.
    """

    def __init__(
        self, width=3.5, height=2.625, legend_position="best", legend_height_ratio=0.15
    ):
        """Initialize the single column figure."""
        super().__init__()
        self.width = width
        self.height = height
        self.legend_position = legend_position
        self.legend_height_ratio = legend_height_ratio

    def create_figure(self):
        """
        Create a new figure with a single axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The axes in the figure.
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        return fig, ax

    def style_axes(self, ax, frameon=True, facecolor="none"):
        """
        Apply styling to axes including legend.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to style.
        frameon : bool, optional
            Whether to draw a frame around the legend, default is True.
        facecolor : str, optional
            Background color of the legend, default is 'none'.
        """
        super().style_axes(ax)
        self._set_legend(ax, frameon, facecolor)

    def finalize(self, fig, filename, legend_ncols=3):
        """
        Finalize and save the figure with proper legend placement.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        filename : str
            Path where the figure will be saved.
        legend_ncols : int, optional
            Number of columns in the legend when legend_position is 'bottom',
            default is 3.
        """
        if self.legend_position == "bottom":
            self._adjust_figure_size(fig)
            plt.tight_layout(rect=[0, self.legend_height_ratio, 1, 1])
            ax = fig.gca()
            bbox = ax.get_position()  # get the position of the axis
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                bbox_to_anchor=(bbox.x0 + (bbox.width / 2), 0.05),
                loc="lower center",
                fontsize=self.font_sizes["legend"],
                ncol=legend_ncols,
            )
        else:
            plt.tight_layout()
        fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _set_legend(self, ax, frameon, facecolor):
        """
        Set the legend on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to add the legend to.
        frameon : bool
            Whether to draw a frame around the legend.
        facecolor : str
            Background color of the legend.
        """
        if self.legend_position == "bottom":
            # legend will be set in finalize method
            pass
        else:
            ax.legend(
                loc=self.legend_position,
                fontsize=self.font_sizes["legend"],
                frameon=frameon,
                facecolor=facecolor,
            )

    def _adjust_figure_size(self, fig):
        """
        Adjust figure size to accommodate bottom legend.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to resize.
        """
        fig.set_size_inches(self.width, self.height * (1 + self.legend_height_ratio))


class GridMxN(BaseFigure):
    """
    Multi-panel figure arranged in a grid.

    This class creates a figure with multiple panels arranged in a grid
    with options for shared or individual legends.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.
    legend_height_ratio : float
        Ratio of legend height to figure height for shared legends.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        width: float = 6.3,
        height: float = 7.8,
        legend_height_ratio: float | None = None,
    ):
        """Initialize the grid figure."""
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.legend_height_ratio = (
            0 if legend_height_ratio is None else legend_height_ratio
        )

    def create_figure(self):
        """
        Create a new figure with a grid of axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axes : list
            List of matplotlib.axes.Axes objects in the grid.
        """
        fig = plt.figure(figsize=(self.width, self.height))
        gs = gridspec.GridSpec(self.rows, self.cols, figure=fig)
        axes = []
        for i in range(self.rows):
            for j in range(self.cols):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
        return fig, axes

    def style_axes(self, axes):
        """
        Apply styling to all axes in the grid.

        Parameters
        ----------
        axes : list
            List of matplotlib.axes.Axes objects to style.
        """
        for ax in axes:
            super().style_axes(ax)

    def finalize(
        self,
        fig,
        filename,
        legend_style="shared",
        legend_loc="best",
        legend_ncol=3,
        plot_center_corr: float = 0,
        apply_tight_layout: bool = True,
    ):
        """
        Finalize and save the figure with proper legend placement.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        filename : str
            Path where the figure will be saved.
        legend_style : str, optional
            Style of legend: 'shared', 'individual', or None, default is 'shared'.
        legend_loc : str, optional
            Location of individual legends, default is 'best'.
        legend_ncol : int, optional
            Number of columns in shared legend, default is 3.
        plot_center_corr : float, optional
            Correction to the horizontal position of the shared legend,
            default is 0.
        apply_tight_layout : bool, optional
            Whether to apply tight_layout to the figure, default is True.
        """
        if legend_style == "individual":
            self.set_subplot_legends(fig.axes, loc=legend_loc)
            if apply_tight_layout:
                plt.tight_layout()
        elif legend_style == "shared":
            self.add_shared_legend(
                fig, fig.axes, ncol=legend_ncol, plot_center_corr=plot_center_corr
            )
            if apply_tight_layout:
                plt.tight_layout(rect=[0, self.legend_height_ratio, 1, 1])
        else:
            if apply_tight_layout:
                plt.tight_layout()
        fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def set_subplot_titles(self, axes, titles):
        """
        Set titles for each subplot.

        Parameters
        ----------
        axes : list
            List of matplotlib.axes.Axes objects.
        titles : list
            List of title strings corresponding to each axes.
        """
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=self.font_sizes["title"])

    def set_subplot_labels(self, axes, labels, xpos=0.5, ypos=0.97):
        """
        Add labels (e.g., (a), (b), etc.) to subplots.

        Parameters
        ----------
        axes : list
            List of matplotlib.axes.Axes objects.
        labels : list
            List of label strings corresponding to each axes.
        xpos : float, optional
            Horizontal position of label in axes coordinates, default is 0.5.
        ypos : float, optional
            Vertical position of label in axes coordinates, default is 0.97.
        """
        for ax, label in zip(axes, labels):
            ax.text(
                xpos,
                ypos,
                f"\\textbf{{({label})}}",
                transform=ax.transAxes,
                fontsize=self.font_sizes["subplot_label"] - 1,
                va="top",
                ha="center",
            )

    def set_subplot_legends(self, axes, loc="best"):
        """
        Add individual legends to each subplot.

        Parameters
        ----------
        axes : list
            List of matplotlib.axes.Axes objects.
        loc : str, optional
            Location of legends, default is 'best'.
        """
        for ax in axes:
            ax.legend(loc=loc, fontsize=self.font_sizes["legend"])

    def add_shared_legend(self, fig, axes, ncol=3, plot_center_corr=0):
        """
        Add a shared legend at the bottom of the figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to add the legend to.
        axes : list
            List of matplotlib.axes.Axes objects to collect legend entries from.
        ncol : int, optional
            Number of columns in the legend, default is 3.
        plot_center_corr : float, optional
            Correction to the horizontal position of the legend, default is 0.
        """
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)

        for ax in axes:
            ax.get_legend().remove() if ax.get_legend() else None

        fig.set_size_inches(self.width, self.height * (1 + self.legend_height_ratio))

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5 + plot_center_corr, 0),
            ncol=ncol,
            fontsize=self.font_sizes["legend"],
        )
