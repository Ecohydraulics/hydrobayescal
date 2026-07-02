"""
Base class for the BayesianPlotter: output folder handling and the global
LaTeX matplotlib style.
"""

from pathlib import Path

import matplotlib.pyplot as plt


class PlotterBase:
    def __init__(
            self,
            results_folder_path='',
            plots_subfolder='plots',
            variable_name=''
    ):
        """
        Constructor of BayesianPlotter class, which is used to create and save various plots related to Bayesian calibration.

        Parameters
        ----------
        results_folder_path : str
            Path to the folder where results (including plots) will be saved. Usually 'auto-saved-results'.
        plots_subfolder : str, optional
            Name of the subfolder within the results folder where plots will be saved. Default: 'plots'.
        variable_name : str, optional
            Name of the variable for which plots will be saved (used as a subfolder name).
        """
        # Define paths
        self.results_folder_path = Path(results_folder_path)
        self.save_folder = self.results_folder_path / plots_subfolder / variable_name

        # Create folder if it doesn't exist
        self.save_folder.mkdir(parents=True, exist_ok=True)

        # Matplotlib LaTeX style
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times'],
            'axes.labelsize': 26,
            'axes.titlesize': 26,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 50,
            'lines.linewidth': 1.5,
            'lines.markersize': 8,
            'axes.linewidth': 0.8,
            'svg.fonttype': 'none'
        })

    def _set_latex_format(self, ax):
        """
        Sets LaTeX formatting for the text in the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to set LaTeX formatting.
        """
        ax.set_xlabel(ax.get_xlabel(), fontsize=28, family='serif')
        ax.set_ylabel(ax.get_ylabel(), fontsize=28, family='serif')
        ax.legend(fontsize=28)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=28)
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
