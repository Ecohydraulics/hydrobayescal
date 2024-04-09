"""
Head for sampling parameter spaces according to a user-defined experimental design.

Note::
    The initial code design only allows for the following designes:
        * MIN - equal interval - MAX: use interval minima and maxima and sample n calibration param values
            between them with equal interval distances
        * MIN - random - MAX: use interval minima and maxima and randomly sample n calibration param values
            between them
        * Random: randomly sample n initial calibration parameter values
"""
from random import choice
import numpy as _np
import pandas as _pd
# import DOE_functions as doef  # for later implementation


class DesignOfExperiment:
    def __init__(self):
        """Head class for an experimental design for the initial value sets for model parameters

        For a multi-parameter space, use
        DesignOfExperiment.generate_multi_parameter_space(parameter_dict, method, total_number_of_samples)

            * Make sure the method is provided by DesignOfExperiment.DESIGN_METHODS, otherwise
              generate_interval raises an error
            * Structure for parameter_dict: ``{par_name: {"bounds": (min, max)}``

        """
        self.__DESIGN_METHODS = [
            "MIN - equal interval - MAX"
            "MIN - random - MAX"
            "Random"
            "DOE functions (doef) not yet implemented -- 2DO"
        ]

        self.df_parameter_spaces = _pd.DataFrame([])

    @property
    def DESIGN_METHODS(self):
        return self.__DESIGN_METHODS

    def generate_interval(self, method, minimum, maximum, n):
        """
        Head method to generate an interval according to a selected method between minimum and maximum
            bounds and with n samples. The provided method must be one of DesignOfExperiments.DESIGN_METHODS.

        :param str method: must be listed in DesignOfExperiments.DESIGN_METHODS
        :param float minimum: minimum of the parameter space
        :param float maximum: maximum of the parameter space
        :param int n: number of equally distanced samples
        :return: numpy.array (1d): series of parameter values
        """
        if not (method in self.__DESIGN_METHODS):
            print("ERROR: invalid design of experiments method. Valid methods are:\n" + ", ".join(self.__DESIGN_METHODS))
            raise NameError

        if method.lower() == "min - equal interval - max":
            return self.min_equalInterval_max(minimum, maximum, n)
        if method.lower() == "min - random - max":
            return self.min_randomInterval_max(minimum, maximum, n)
        if method.lower() == "interval":
            return self.randomInterval(minimum, maximum, n)
        if "DOE functions (doef)" in method:
            print("SORRY: I cannot yet deal with Box-DoE methods")
            return _np.array([])

    def generate_multi_parameter_space(self, parameter_dict, method, total_number_of_samples):
        """ Create a multi parameter space (combination) for a set of parameters.

        :param dict parameter_dict: nested dictionary with parameter names and their minimum and maximum values
                The parameter dict has to contain a sub-dict with a boundary entry.
                Construction: {par_name: {"bounds": (min, max)}
                Example dict: {par_name: {"bounds": (0.1, 1.5)}
        :param str method: must be listed in DesignOfExperiments.DESIGN_METHODS
        :param int total_number_of_samples: total number of samples, which should be a multiple of the
                            number of parameters
        :return None: assigns the internal pandas.DataFrame where column heads are parameter names,
                                                                rows correspond to parameter combinations
        """

        dict4parameter_file = {}  # required for pd.DataFrame and textfile with parameter combinations
        # build parameter combination (PC) index dict
        index_dict = {}
        [index_dict.update({i: "PC{}".format(i+1)}) for i in range(total_number_of_samples)]
        # generate n samples for every calibration parameter
        for par in parameter_dict.keys():
            parameter_dict[par]["value array"] = self.generate_interval(
                minimum=parameter_dict[par]["bounds"][0],
                maximum=parameter_dict[par]["bounds"][1],
                n=total_number_of_samples,
                method=method,
            )
            dict4parameter_file.update({par: parameter_dict[par]["value array"]})

        self.df_parameter_spaces = _pd.DataFrame.from_dict(data=dict4parameter_file)
        self.df_parameter_spaces.rename(index=index_dict, inplace=True)

    @staticmethod
    def min_equalInterval_max(minimum, maximum, n):
        """ Generate n equally distanced values for a parameter between a minimum and
            a maximum value

        :param float minimum: minimum of the parameter space
        :param float maximum: maximum of the parameter space
        :param int n: number of equally distanced samples
        :return numpy.array (1d): series of parameter values
        """
        try:
            return _np.linspace(start=minimum, stop=maximum, num=n)
        except Exception as e:
            print("ERROR: could not generate equally distanced parameter space:\n " + str(e))
            return _np.array([])

    def min_randomInterval_max(self, minimum, maximum, n):
        """ Generate n random values for a parameter between a minimum and a maximum value where the
            smallest and largest values correspond to the interval minimum and maximum

        :param float minimum: minimum of the parameter space
        :param float maximum: maximum of the parameter space
        :param int n: number of equally distanced samples
        :return numpy.array (1d): series of parameter values
        """
        try:
            rand_array = self.randomInterval(minimum, maximum, n)
            rand_array[0] = min
            rand_array[-1] = max
            return rand_array
        except Exception as e:
            print("ERROR: could not generate equally distanced parameter space:\n " + str(e))
            return _np.array([])

    @staticmethod
    def randomInterval(minimum, maximum, n):
        """ Generate n random values for a parameter between a minimum and a maximum value

        :param float minimum: minimum of the parameter space
        :param float maximum: maximum of the parameter space
        :param int n: number of equally distanced samples
        :return numpy.array (1d): series of parameter values
        """
        try:
            base_space = _np.linspace(start=minimum, stop=maximum, num=n * 100)
            return _np.array([choice(base_space) for _ in range(n)])
        except Exception as e:
            print("ERROR: could not generate equally distanced parameter space:\n " + str(e))
            return _np.array([])



