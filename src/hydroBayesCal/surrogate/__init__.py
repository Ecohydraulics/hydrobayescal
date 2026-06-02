"""Gaussian process surrogate models and Bayesian active learning.

This package contains hydroBayesCal's own surrogate-model and Bayesian-active-
learning implementations (single- and multi-output GP emulators via scikit-learn
and GPyTorch, Bayesian inference, and the sequential-design / exploration logic).

Upstream ``bayesvalidrox`` is used only for experimental design / parameter
sampling (``bayesvalidrox.Input`` and ``bayesvalidrox.ExpDesigns``); the GP/BAL
math here is maintained in-tree.
"""
