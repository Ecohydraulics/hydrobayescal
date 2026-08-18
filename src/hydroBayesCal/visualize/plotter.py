"""
Code for plotting results in the context of Bayesian Calibration with GPE.

The BayesianPlotter groups thematic plot mixins; the individual plot
implementations live in the sibling modules of this package.
"""

from hydroBayesCal.visualize.agreement_plots import AgreementPlots
from hydroBayesCal.visualize.bal_plots import BALPlots
from hydroBayesCal.visualize.base import PlotterBase
from hydroBayesCal.visualize.calibration_assessment import CalibrationAssessment
from hydroBayesCal.visualize.metrics_plots import MetricsPlots
from hydroBayesCal.visualize.model_comparison_plots import ModelComparisonPlots
from hydroBayesCal.visualize.posterior_plots import PosteriorPlots


class BayesianPlotter(
    PosteriorPlots,
    BALPlots,
    ModelComparisonPlots,
    MetricsPlots,
    CalibrationAssessment,
    AgreementPlots,
    PlotterBase,
):
    pass
