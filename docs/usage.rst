
Usage
=====

Regular Usage
-------------

Comming soon

.. figure:: https://github.com/sschwindt/stochatis-surrogate/raw/main/docs/img/brower-icon-large.jpg
   :alt: calibrate surrogate bayesian gaussian bal gpe

   *Intro figure.*

Implement the following code in a Python script and run that Python script:

.. code-block::

    import stochastic_surrogate as sur
    model_dir = r"C:\telemac\\v8\\models\\training-example"
    sur.optimize(model_dir)


.. important::

    The model directory may not end on any ``\`` or  ``/`` .

- After a successful run, the code will have produced the following files in ``...\your-data\``:
    + ``files`` das

Usage Example
-------------

For example, consider your model lives in a folder called ``C:\telemac\models\reservoir2d``.



