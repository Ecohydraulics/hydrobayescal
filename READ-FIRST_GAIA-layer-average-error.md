# Read first: the GAIA layer average on this branch read sand classes, not bed layers

This note exists on the branch `hbc-telemac-andres` only. It concerns every result that used a layer-averaged sand ratio (`gaia_layer_average`, the target `LAY1 SAND RAT`) and was extracted with the code of this branch before it was merged for release 1.10.0. Delete the file with `git rm READ-FIRST_GAIA-layer-average-error.md` once the points below are settled.

## What was wrong

GAIA writes the ratio of every sand class in every bed layer as its own result variable and names it `LAY<layer> SAND RAT<class>`. The number after `LAY` is the bed layer, and the trailing number is the sand class. The subroutine `nomvar_gaia.f` states it in a comment: "LAY 2 SAND RAT 3" corresponds to the sand ratio of sand class 3 in the 2nd layer.

The extraction added in commit `17c6a49` built the variable names by appending the layer number to the configured name. With

```python
'gaia_layer_average': {"LAY1 SAND RAT": {"layers": [1, 2], "thicknesses": [0.08, 0.60]}}
```

it read `LAY1 SAND RAT1` and `LAY1 SAND RAT2`. These are the sand classes 1 and 2 of bed layer 1, not sand class 1 in the bed layers 1 and 2. The value stored as `LAY1 SAND RAT` was therefore

```
(0.08 * ratio(layer 1, class 1) + 0.60 * ratio(layer 1, class 2)) / 0.68
```

which is a mix of two sand classes of the top layer, weighted with the thicknesses of two layers. For a result file that holds class 1 at ratios of 0.20 and 0.50 in the layers 1 and 2, and class 2 at 0.80 in layer 1, the code returned 0.73. The ratio of class 1 averaged over both layers is 0.46.

The value lies between 0 and 1 in either case, so nothing in the results points to the error. Every calibration, surrogate and posterior obtained with this target on this branch refers to the mixed quantity.

## How to check it yourself

List the variables of one of your GAIA result files:

```python
from hydroBayesCal.telemac.pputils.ppmodules.selafin_io_pp import ppSELAFIN

slf = ppSELAFIN("results_hydromorphodynamics_gaia_1.slf")
slf.readHeader()
print([" ".join(name.split()) for name in slf.getVarNames()])
```

A model with N sand classes and M bed layers lists `LAY1 SAND RAT1` to `LAYM SAND RATN`. The GAIA example `yen-t2d` of the TELEMAC installation, with five sand classes in two bed layers, lists `LAY1 SAND RAT1`, `LAY2 SAND RAT1`, `LAY1 SAND RAT2`, and so on up to `LAY2 SAND RAT5`.

## What release 1.10.0 does instead

An entry of `gaia_layer_average` names the sand classes and the bed layers:

```python
'gaia_layer_average': {"LAY1 SAND RAT": {"sand_classes": [1], "layers": [1, 2], "thicknesses": [0.08, 0.60]}}
```

The ratios of the listed classes are summed in each layer and averaged over the layers with the thicknesses as weights. An entry without `sand_classes` stops the model at construction with an error that states the naming rule, so a configuration in the old form cannot run unnoticed. The setting is described in `docs/usage-telemac.rst` under "Sand ratio averaged over GAIA bed layers".

## What needs to be done

1. **Confirm the sand classes.** `examples/Telemac/Hydromorphodynamic/Ering/config_Ering.py` was converted with `"sand_classes": [1]`. That is an assumption, because the GAIA steering file of the case is not in the repository. List every class that counts as fine sediment in the measurements. The class numbers follow the order of `CLASSES SEDIMENT DIAMETERS` in the GAIA steering file.
2. **Extract the target again.** The values stored under `auto-saved-results-HydroBayesCal` (`restart_data/initial-model-outputs.json`, the `*-detailed.json` files, `BAL_dictionary.pkl` and the pickled surrogates) were computed with the old variable names. A restart with `only_bal_mode = True` reuses them without any message, and `config_Ering.py` is currently set to `only_bal_mode = True`. The stored values can be replaced only from the GAIA result files of the runs. Where those were deleted (`delete_complex_outputs = True`), the runs have to be repeated.
3. **Convert your own configuration files** that are not in the repository to the new form.
4. **Reproduce figures and numbers that were derived from this target** with the corrected extraction before they are used further.

The other corrections made to this branch during the merge are listed in the notes of release 1.10.0 under "Integration notes for the contributors": https://github.com/Ecohydraulics/hydrobayescal/releases/tag/v1.10.0
