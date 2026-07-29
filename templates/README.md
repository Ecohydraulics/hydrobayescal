# Templates

Runnable driver scripts and configuration templates for HydroBayesCal. Copy
and adapt them to your project; they are not part of the installable
`hydroBayesCal` package.

| Script | Purpose |
|---|---|
| `bal_telemac.py` | Main driver: GPE training and Bayesian Active Learning with TELEMAC |
| `bal_openfoam.py` | Main driver: GPE training and Bayesian Active Learning with OpenFOAM |
| `config_Telemac.py` | Configuration template consumed by `bal_telemac.py --config` |
| `config_OpenFOAM.py` | Configuration template consumed by `bal_openfoam.py --config` |
| `prebal_telemac_error_analysis.py` | Pre-calibration error analysis for TELEMAC setups |
| `derive_calibrated_parameters.py` | Post-calibration: per-parameter marginal optima, joint posterior optimum, equifinality diagnostic, and the candidate parameter sets for the final full-complexity runs |
| `telemac_extract.py` | Extract model outputs from a TELEMAC result file (.slf) at calibration points |
| `assess_calibration.py` | Post-calibration assessment of surrogate and full-complexity model outputs |
| `main_plots.py` | Plot BAL posterior results (BME/RE evolution, posterior histograms) |
| `main_validate.py` | Validate calibrated models against observations |
| `vectrino_postprocess.py` | Post-process Vectrino ADV measurements (despiking, TKE, velocity profiles) |

Run a driver with the repository root or any working directory of your choice,
for example:

```bash
python templates/bal_telemac.py --config templates/config_Telemac.py
```

All scripts import from the installed `hydroBayesCal` package
(`pip install -e .` from the repository root).
