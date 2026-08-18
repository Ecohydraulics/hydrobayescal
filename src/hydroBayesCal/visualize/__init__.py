"""Plot routines for surrogate-assisted Bayesian calibration results."""

__all__ = ["BayesianPlotter"]


def __getattr__(name):
    """Import :class:`~hydroBayesCal.visualize.plotter.BayesianPlotter` on first use.

    The plotter aggregates every plotting mixin of the package, and therefore every
    plotting dependency, into one class. Importing it eagerly here made a single
    submodule unusable without the dependencies of all the others, which is the wrong
    trade for the figures that a driver writes when a calibration finishes.
    """
    if name == "BayesianPlotter":
        from hydroBayesCal.visualize.plotter import BayesianPlotter

        return BayesianPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
