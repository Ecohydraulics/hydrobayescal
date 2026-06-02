References
==========

HydroBayesCal implements and builds on the following work. Each item is
available from its publisher via the DOI/link below. (For developer convenience
the project keeps local copies in a git-ignored ``ExportedItems/`` folder; these
are not redistributed with the package.)

Core methodology
----------------

* **Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020).**
  *Bayesian3 Active Learning for the Gaussian Process Emulator Using Information
  Theory.* Entropy, 22(8), 890.
  `doi:10.3390/e22080890 <https://doi.org/10.3390/e22080890>`_

  The Bayesian active-learning strategy (Bayesian model evidence and relative
  entropy as training-point selection criteria) at the heart of HydroBayesCal.

* **Rasmussen, C. E., & Williams, C. K. I. (2006).**
  *Gaussian Processes for Machine Learning.* MIT Press.
  Freely available at `gaussianprocess.org/gpml
  <http://gaussianprocess.org/gpml/>`_.

  The reference text for the Gaussian-process regression used to build the
  surrogate models.

Applications
------------

* **Mouris, K., Acuña Espinoza, E., Schwindt, S., Mohammadi, F., Haun, S.,
  Wieprecht, S., & Oladyshkin, S. (2023).**
  *Stability Criteria for Bayesian Calibration of Reservoir Sedimentation
  Models.* Modeling Earth Systems and Environment, 9, 3643–3661.
  `doi:10.1007/s40808-023-01712-7 <https://doi.org/10.1007/s40808-023-01712-7>`_

  Surrogate-assisted Bayesian calibration of a 2D hydro-morphodynamic reservoir
  sedimentation model.

* **Schwindt, S., Callau Medrano, S., Mouris, K., Beckers, F., Haun, S.,
  Nowak, W., Wieprecht, S., & Oladyshkin, S. (2023).**
  *Bayesian Calibration Points to Misconceptions in Three-Dimensional
  Hydrodynamic Reservoir Modeling.* Water Resources Research, 59(3),
  e2022WR033660.
  `doi:10.1029/2022WR033660 <https://doi.org/10.1029/2022WR033660>`_

  Bayesian calibration of a 3D reservoir hydrodynamic model, showing how
  posterior geometry reveals faulty model assumptions.

Software dependency
-------------------

* **BayesValidRox** — `documentation
  <https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/>`_.
  Used by HydroBayesCal for the experimental design and parameter sampling
  (``Input`` and ``ExpDesigns``).
