.. code docs head file

Developer Docs
==============

The following sections provide details of functions, their arguments, and outputs to help tweaking the code for individual purposes.

.. important::

   All modules that should be available in HyBayesCal have to be imported in ``HyBayesCal.__init__.py``. Thus, all top-level packages and modules (e.g., `model_structure`) do never need to be imported relatively. That is, use `import module_structure` in any script, never `import .module_structure` even though your programming IDE (e.g., PyCharm) says that it cannot find the import.

   Only use relative imports inside of packages, where the ``__init__.py`` file do not import anything. This is why, for example, the script ``telemac.usr_def_telemac.py`` uses a relative import for ``from .config_telemac import *``.

Update Requirements
-------------------

Activate the ``HBCenv`` Python environment:

.. code-block::

   source HBCenv/bin/activate

Make sure all additionally required packages for updated software are pip-installed, not more and not less (to avoid garbage requirements). Then, update pip/pip3 and re-build ``requirements.txt`` with pip or pip3:

.. code-block::

   pip install --upgrade pip
   pip freeze > requirements.txt

.. note::

   This builds a requirements.txt file for all libraries and their versions installed in your local HBCenv environment. So, if you have unnecessary or stale libraries installed, it will write them to the new requirements.txt.

A future release containing design of experiments-based construction of an initial response surface additionally needs the following packages:

- pydoe (``pip install pydoe``)
- diversipy (``pip install diversipy``)

action_logger.py
----------------

.. automodule:: HyBayesCal.action_logger
    :members:

BAL_core.py
-----------

.. automodule:: HyBayesCal.BAL_core
    :members:


basic_functions.py
------------------

.. automodule:: HyBayesCal.basic_functions
    :members:

config.py
---------

.. automodule:: HyBayesCal.config
    :members:


GPE_BAL_telemac.py
------------------

.. automodule:: HyBayesCal.GPE_BAL_telemac
    :members:

telemac_core.py
------------------

.. automodule:: HyBayesCal.telemac_core
    :members:

usr_defs.py
-----------

.. automodule:: HyBayesCal.usr_defs
    :members:



