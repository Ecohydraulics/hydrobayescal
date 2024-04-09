## Basics

Make sure to understand the basics of building a PyPI package ([example tutorial](https://towardsdatascience.com/build-your-first-open-source-python-project-53471c9942a7)).

## Requirements (PyPI)

* Create a [TestPyPI](https://test.pypi.org/) account
* Create a [PyPI](https://pypi.org/) account
* Install requirements for developers (in *Terminal*)</br>`pip install -r requirements-dev.txt`

## Build and push test version

SHORT VERSION:

```
python setup.py develop
twine upload dist/*
```

TEST:

```
python -m venv test_env
source test_env/bin/activate
pip install stochastic_surrogate
import stochastic_surrogate
```


Before adding a new version of *stochastic_surrogate*, please inform about the severity and version numbering semantics on [python.org](https://www.python.org/dev/peps/pep-0440/).

1. `cd` to your local *stochastic_surrogate* folder (in *Terminal*)
1. Create *stochastic_surrogate* locally 
	* Linux (in Terminal): `sudo python setup.py sdist bdist_wheel`
	* Windows (in Anaconda Prompt with flussenv): `python setup.py sdist bdist_wheel`
1. Upload the (new version) to TestPyPI (with your TestPyPI account):
	* `twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*`
	* If any error occurs, fix it and rebuild the package (previous step).
1. Create a new environment and activate it to test if the upload and installation work
    * On *Linux*:</br>`python -m venv test_env`</br>`source test_env/bin/activate`
    * On *Windows* (with Anaconda):</br>`conda activate stochastic_surrogate-test`
1. Install the new version of *stochastic_surrogate* in the environment:
	* `pip install -i https://test.pypi.org/simple/ stochastic_surrogate`
1. Launch python and import *stochastic_surrogate*:
	* `python`
	* `>>> import stochastic_surrogate`

## Push to PyPI

If you could build and install the test version successfully, you can push the new version to PyPI. **Make sure to increase the `VERSION="major.minor.micro" in *ROOT/setup.py***. Then push to PyPI (with your PyPI account):

`twine upload dist/*`

## Create a new release on GitHub

Please note that we are currently still in the *growing* phase of *stochastic_surrogate*. Since *version 0.2*, login at github.com and create a new *release* after merging branches.
