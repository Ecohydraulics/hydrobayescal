
Usage
=====

Regular Usage
-------------

TKEanalyst requires meta data (i.e. data about your data) defined in an input workbook. Therefore, `download input.xlsx <https://github.com/sschwindt/TKEanalyst/raw/main/input.xlsx>`_) and save it on your computer.
Next, with Python installed and the code living on your computer:

- Save your data in a folder and make sure the files are named with ``XX_YY_ZZ_something.FILEENDING`` where ``XX``, ``YY``, and ``ZZ`` are streamwise (x), perpendicular (y), and vertical (z) coordinates in CENTIMETERS, respectively. ``FILEENDING`` could be, for example, ``.vna``.
- Complete the required information on the experimental setup in ``input.xlsx`` (see figure below). **IMPORTANT: Never modify column A or any list in the sourcetables sheet (unless you also modify** ``load_input_defs`` **in line 25ff of** ``profile_analyst.py`` **).**  The code uses the text provided in these areas of *input.xlsx* to identify setups. If useful, consider substituting the *Wood* wording in your mind and with a note in column C with your characteristic turbulence objects, but do not modify column A. Ultimately, you can also save the input file under a different name and call the code with a different input file name.

.. figure:: https://github.com/sschwindt/TKEanalyst/raw/main/docs/img/input-xlsx.jpg
   :alt: input turbulent tke experiment setup parameters

   *The interface of the input.xlsx workbook for entering experiment parameters and specifying a despiking method.*

- Implement the following code in a Python script and run that Python script:

.. code-block::

    import TKEanalyst
    input_file = r"C:\\my\\project\\adv\\input.xlsx"
    TKEanalyst.process_adv_files(input_file)

- Alternatively:
    + run the code: ``python profile_analyst.py "C:/dir/to/input.xlsx``)
- Wait until the code finished with ``-- DONE -- ALL TASKS FINISHED --``
- After a successful run, the code will have produced the following files in ``...\your-data\``:
    + ``.xlsx`` files of full-time series data, with spikes and despiked.
    + ``.xlsx`` files of statistic summaries (i.e., average, standard deviation *std*, TKE) of velocity parameters with x, y, and z positions, with spikes and despiked (see workbook example in the figure below).
    + Two plots (``norm-tke-x.png`` and ``norm-tke-x-despiked.png``) showing normalized TKE plotted against normalized x, with spikes and despiked, respectively (see plot example in the figure below).

.. figure:: https://github.com/sschwindt/tke-calculator/raw/main/docs/img/output-example.jpg
   :alt: example output tke-calculator

    *Exemplary output workbook of despiked statistics, such as averages, standard deviations, and standard errors of u, v, w, shear stresses (tau) and TKE.*

.. figure:: https://github.com/sschwindt/tke-calculator/raw/main/docs/img/norm-tke-x-despiked.png
   :alt: example output normalized tke plot

    *Exemplary outputof normalized TKE vs. normalized x coordinates.*

Usage Example
-------------

For example, consider your data lives in a folder called ``C:\my-project\TKEanalysis\test01``. To analyze ``*.vna`` files in ``test01`` save the following code to a Python script named ``tke_analysis.py`` along with definitions in an ``input.xlsx`` workbook :


.. code-block::

    import TKEanalyst
    input_file = r"C:\\my-project\\TKEanalysis\\test01\\input.xlsx"
    TKEanalyst.process_adv_files(input_file)

The definitions in the above-shown ``input.xlsx`` define x-normalization as a function of a wood log length, for example, a wood log diameter of 0.114 m.

Cell ``B2`` containing **Input folder directory** in ``input.xlsx`` defines that the input data for ``test01``.

.. important::

    The data directory of the subfolder definition in cell ``B2`` may not end on any ``\`` or  ``/`` . Also, make sure to **use the** ``/`` **sign for folder name separation** (do not use ``\``).

To run the code with the example data, open Anaconda Prompt (or any other Python-able Terminal) and:
    + ``cd`` into the code directory (e.g., ``cd "C:\my-project\TKEanalysis\test01"``
    + run the code: ``python tke_analysis.py``
    + wait until the code finished with ``-- DONE -- ALL TASKS FINISHED --``
- After a successful run, the code will have produced the following files in ``C:\my-project\TKEanalysis\test01``:
    + ``.xlsx`` files of full-time series data, with spikes and despiked.
    + ``.xlsx`` files of statistic summaries (i.e., average, standard deviation *std*, TKE) of velocity parameters with x, y, and z positions, with spikes and despiked.
    + Two plots (``norm-tke-x.png`` and ``norm-tke-x-despiked.png``) showing normalized TKE plotted against normalized x, with spikes and despiked, respectively.







