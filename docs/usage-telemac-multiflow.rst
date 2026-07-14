.. _usage-telemac-multiflow:

Multi-discharge (multi-flow) calibration with TELEMAC
=====================================================

The standard :doc:`usage-telemac` workflow calibrates a model against ground
truth measured at **one** hydraulic state. Field campaigns, however, are often
repeated at **several steady discharges**. Calibrating a single shared parameter
set (e.g. bed-roughness zones) against all of them at once constrains the model
far better than any single flow, because a good parameter value must reproduce
the measurements across the whole range of states.

HydroBayesCal supports this through an **additive** multi-flow extension that
builds on the stock TELEMAC binding without changing it - existing single-flow
configs and workflows are unaffected.

.. contents::
   :local:
   :depth: 2

How it works
------------

Every collocation point (one parameter sample) is evaluated by running the
full-complexity model **once per flow**. Each flow has:

* its own steering ``.cas`` file (same mesh, boundary and friction files; the
  flows differ only in the hydraulic forcing - discharge, downstream stage,
  simulated duration - and in the results file name), and
* its own calibration-points CSV (the measurements taken at that discharge).

The per-flow observations and model outputs are concatenated along the location
axis into one combined observation/output space, so the Gaussian Process
Emulator learns ``quantity(location, flow; parameters)`` and the Bayesian
inference uses every flow's measurements simultaneously.

Components
----------

``hydroBayesCal.telemac.multiflow_telemac.MultiflowTelemacModel``
    Wraps **one stock**
    :class:`~hydroBayesCal.telemac.control_telemac.TelemacModel` **per flow**
    (composition, not subclassing), each with its own results subtree
    ``<res_dir>/flow-<name>/``. It concatenates ``observations`` /
    ``variances`` / ``measurement_errors`` / ``model_evaluations`` across flows,
    runs the flows sequentially per collocation-point set, and mirrors the
    bookkeeping attributes the driver reads. Because each flow is a plain
    ``TelemacModel``, all single-flow behaviour (steering/friction updates,
    solver launch, extraction, output processing, restart files) is exactly the
    code single-flow users run. A fail-fast check rejects a degenerate per-flow
    evaluation matrix (a run of all zeros, or initial-design rows that are
    identical across every run) so a silent extraction failure surfaces in
    minutes instead of after days of compute.

``templates/bal_telemac_multiflow.py``
    The driver. It **imports and reuses** ``bal_telemac.py`` (the experiment
    design, the initial-runs loop and the Bayesian-Active-Learning loop) and
    only swaps the single-flow model for a ``MultiflowTelemacModel``. Run it
    exactly like ``bal_telemac.py``::

        python bal_telemac_multiflow.py --config config_Telemac_multiflow.py

Configuration
-------------

Use a standard ``config_Telemac.py`` (``paths`` / ``hydrodynamic_simulation`` /
``morphodynamic_simulation`` / ``calibration`` / ``sampling`` / ``execution``)
**plus** a ``multiflow`` block. The single-flow
``paths['calibration_pts_file_path']`` and
``hydrodynamic_simulation['control_file']`` are ignored in favour of the
per-flow entries; everything else keeps its single-flow meaning.

.. code-block:: python

   multiflow = {
       'flows': [
           {'name': 'q47',
            'control_file': 'steady2d-q47.cas',
            'results_filename_base': 'r2d-q47',
            'calibration_pts_file_path': '/path/measurements-q47.csv'},
           {'name': 'q168',
            'control_file': 'steady2d-q168.cas',
            'results_filename_base': 'r2d-q168',
            'calibration_pts_file_path': '/path/measurements-q168.csv'},
       ],
   }

Requirements on the flows:

* identical ``calibration_parameters`` / ``param_values`` (the whole point - one
  shared parameter vector),
* identical ``calibration_quantities`` / ``extraction_quantities``,
* all steering files in the same ``model_dir`` (same mesh/boundary/friction),
* a **distinct** ``results_filename_base`` per flow (otherwise the per-run
  result files overwrite each other).

Practical guidance
------------------

**Cost scales with the number of flows.** One collocation point is now *N*
solver runs. Budget accordingly, and prefer the fewest flows that span the
states of interest.

**Not every campaign is a valid velocity target.** Compare each campaign's mean
velocity, water depth and spatial extent. A high-discharge campaign with a
*low* mean velocity and *shallow* depth is usually sampling slow, wadeable
margins rather than the fast channel the model routes the flow through - a gap
no roughness value can close. Calibrate velocity on the states where the
measurements sample the conveyance, and bring a high flow in through a different
observable (water level / flood extent).

**Down-weight a spatially clustered campaign.** A single dense cross-section is
not *N* independent samples of the flow field. Inflate its per-point error (the
``<QTY>_ERROR`` column) so its many low-variance points inform, but do not
dominate, the joint likelihood (HydroBayesCal weights each point by
``1 / variance``).

**Memory.** The surrogate is predicted at ``sampling['prior_samples']`` points
by building a dense ``prior_samples x prior_samples`` covariance **per
location** in gpytorch. Large values (e.g. 25000) can exhaust RAM on modest
machines; a low-dimensional parameter posterior needs only a few thousand
samples.

**Restarting.** Each flow writes its completed initial design to
``flow-<name>/auto-saved-results-HydroBayesCal/restart_data/``. Set
``execution['only_bal_mode'] = True`` to reuse those and skip straight to
surrogate training and BAL - useful after adding, removing or re-weighting a
flow (the reused model outputs are independent of the measurement error, so
changing a ``<QTY>_ERROR`` column and resuming applies the new variance while
reusing the completed runs).

hydromate integration
---------------------

The `hydromate <https://github.com/Ecohydraulics/hydromate>`_ pre/post-processor
generates all of the above automatically from a single case configuration: the
per-flow steering files (derived from the built base case), the per-campaign
calibration CSVs (from FlowTracker exports), ``config_Telemac_multiflow.py``,
and it stages and launches this driver. See its ``run_Bayes_cal_multiflow.py``
and ``hydromate.bayescal`` / ``hydromate.campaigns``.
