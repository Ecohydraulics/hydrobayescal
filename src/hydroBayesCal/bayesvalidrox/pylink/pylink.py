#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calls to the model and evaluations
"""

from dataclasses import dataclass

import os
import shutil
import h5py
import numpy as np
import time
import zipfile
import pandas as pd
import multiprocessing
from functools import partial
import tqdm

from multiprocessing import get_context
#from multiprocess import get_context



def within_range(out, minout, maxout):
    """
    Checks if all the values in out lie between minout and maxout

    Parameters
    ----------
    out : array or list
        Data to check against range
    minout : int
        Lower bound of the range
    maxout : int
        Upper bound of the range

    Returns
    -------
    inside : bool
        True if all values in out are in the specified range

    """
    try:
        out = np.array(out)
    except:
        raise AttributeError('The given values should be a 1D array, but are not')
    if out.ndim != 1:
            raise AttributeError('The given values should be a 1D array, but are not')
        
    if minout > maxout:
        raise ValueError('The lower and upper bounds do not form a valid range, they might be switched')
    
    inside = False
    if (out > minout).all() and (out < maxout).all():
        inside = True
    return inside


class PyLinkForwardModel(object):
    """
    A forward model binder

    This calss serves as a code wrapper. This wrapper allows the execution of
    a third-party software/solver within the scope of BayesValidRox.

    Attributes
    ----------
    link_type : str
        The type of the wrapper. The default is `'pylink'`. This runs the
        third-party software or an executable using a shell command with given
        input files.
        Second option is `'function'` which assumed that model can be run using
        a function written separately in a Python script.
    name : str
        Name of the model.
    py_file : str
        Python file name without `.py` extension to be run for the `'function'`
        wrapper. Note that the name of the python file and that of the function
        must be simillar. This function must recieve the parameters in an array
        of shape `(n_samples, n_params)` and returns a dictionary with the
        x_values and output arrays for given output names.
    func_args : dict
        Additional arguments for the python file. The default is `{}`.
    shell_command : str
        Shell command to be executed for the `'pylink'` wrapper.
    input_file : str or list
        The input file to be passed to the `'pylink'` wrapper.
    input_template : str or list
        A template input file to be passed to the `'pylink'` wrapper. This file
        must be a copy of `input_file` with `<Xi>` place holder for the input
        parameters defined using `inputs` class, with i being the number of
        parameter. The file name ending should include `.tpl` before the actual
        extension of the input file, for example, `params.tpl.input`.
    aux_file : str or list
        The list of auxiliary files needed for the `'pylink'` wrapper.
    exe_path : str
        Execution path if you wish to run the model for the `'pylink'` wrapper
        in another directory. The default is `None`, which corresponds to the
        currecnt working directory.
    output_file_names : list of str
        List of the name of the model output text files for the `'pylink'`
        wrapper.
    output_names : list of str
        List of the model outputs to be used for the analysis.
    output_parser : str
        Name of the model parser file (without `.py` extension) that recieves
        the `output_file_names` and returns a 2d-array with the first row being
        the x_values, e.g. x coordinates or time and the rest of raws pass the
        simulation output for each model output defined in `output_names`. Note
        that again here the name of the file and that of the function must be
        the same.
    multi_process: bool
        Whether the model runs to be executed in parallel for the `'pylink'`
        wrapper. The default is `True`.
    n_cpus: int
        The number of cpus to be used for the parallel model execution for the
        `'pylink'` wrapper. The default is `None`, which corresponds to all
        available cpus.
    meas_file : str
        The name of the measurement text-based file. This file must contain
        x_values as the first column and one column for each model output. The
        default is `None`. Only needed for the Bayesian Inference.
    meas_file_valid : str
        The name of the measurement text-based file for the validation. The
        default is `None`. Only needed for the validation with Bayesian
        Inference.
    mc_ref_file : str
        The name of the text file for the Monte-Carlo reference (mean and
        standard deviation) values. It must contain `x_values` as the first
        column, `mean` as the second column and `std` as the third. It can be
        used to compare the estimated moments using meta-model in the post-
        processing step. This is only available for one output.
    obs_dict : dict
        A dictionary containing the measurement text-based file. It must
        contain `x_values` as the first item and one item for each model output
        . The default is `{}`. Only needed for the Bayesian Inference.
    obs_dict_valid : dict
        A dictionary containing the validation measurement text-based file. It
        must contain `x_values` as the first item and one item for each model
        output. The default is `{}`.
    mc_ref_dict : dict
        A dictionary containing the Monte-Carlo reference (mean and standard
        deviation) values. It must contain `x_values` as the first item and
        `mean` as the second item and `std` as the third. The default is `{}`.
        This is only available for one output.
    """

    # Nested class
    @dataclass
    class OutputData(object):
        parser: str = ""
        names: list = None
        file_names: list = None

    def __init__(self, link_type='pylink', name=None, py_file=None,
                 func_args={}, shell_command='', input_file=None,
                 input_template=None, aux_file=None, exe_path='',
                 output_file_names=[], output_names=[], output_parser='',
                 multi_process=True, n_cpus=None, meas_file=None,
                 meas_file_valid=None, mc_ref_file=None, obs_dict={},
                 obs_dict_valid={}, mc_ref_dict={}):
        self.link_type = link_type
        self.name = name
        self.shell_command = shell_command
        self.py_file = py_file
        self.func_args = func_args
        self.input_file = input_file
        self.input_template = input_template
        self.aux_file = aux_file
        self.exe_path = exe_path
        self.multi_process = multi_process
        self.n_cpus = n_cpus
        self.Output = self.OutputData(
            parser=output_parser,
            names=output_names,
            file_names=output_file_names,
        )
        self.n_outputs = len(self.Output.names)
        self.meas_file = meas_file
        self.meas_file_valid = meas_file_valid
        self.mc_ref_file = mc_ref_file
        self.observations = obs_dict
        self.observations_valid = obs_dict_valid
        self.mc_reference = mc_ref_dict

    # -------------------------------------------------------------------------
    def read_observation(self, case='calib'):
        """
        Reads/prepare the observation/measurement data for
        calibration.
        
        Parameters
        ----------
        case : str
            The type of observation to read in. Can be either 'calib',
            'valid' or 'mc_ref'

        Returns
        -------
        DataFrame
            A dataframe with the calibration data.

        """
        # TOOD: check that what is read in/transformed matches the expected form of data/reference
        if case.lower() == 'calib':
            if isinstance(self.observations, dict) and bool(self.observations):
                self.observations = pd.DataFrame.from_dict(self.observations)
            elif self.meas_file is not None:
                file_path = os.path.join(os.getcwd(), self.meas_file)
                self.observations = pd.read_csv(file_path, delimiter=',')
            elif isinstance(self.observations, pd.DataFrame):
                self.observations = self.observations
            else:
                raise Exception("Please provide the observation data as a "
                                "dictionary via observations attribute or pass"
                                " the csv-file path to MeasurementFile "
                                "attribute")
            # Compute the number of observation
            self.n_obs = self.observations[self.Output.names].notnull().sum().values.sum()
            return self.observations
            
        elif case.lower() == 'valid':
            if isinstance(self.observations_valid, dict) and \
              bool(self.observations_valid):
                self.observations_valid = pd.DataFrame.from_dict(self.observations_valid)
            elif self.meas_file_valid is not None:
                file_path = os.path.join(os.getcwd(), self.meas_file_valid)
                self.observations_valid = pd.read_csv(file_path, delimiter=',')
            elif isinstance(self.observations_valid, pd.DataFrame):
                self.observations_valid = self.observations_valid
            else:
                raise Exception("Please provide the observation data as a "
                                "dictionary via observations attribute or pass"
                                " the csv-file path to MeasurementFile "
                                "attribute")
            # Compute the number of observation
            self.n_obs_valid = self.observations_valid[self.Output.names].notnull().sum().values.sum()
            return self.observations_valid
                
        elif case.lower() == 'mc_ref':
            if self.mc_ref_file is None and \
               isinstance(self.mc_reference, pd.DataFrame):
                return self.mc_reference
            elif isinstance(self.mc_reference, dict) and bool(self.mc_reference):
                self.mc_reference = pd.DataFrame.from_dict(self.mc_reference)
            elif self.mc_ref_file is not None:
                file_path = os.path.join(os.getcwd(), self.mc_ref_file)
                self.mc_reference = pd.read_csv(file_path, delimiter=',')
            else:
                self.mc_reference = None
            return self.mc_reference


    # -------------------------------------------------------------------------
    def read_output(self):
        """
        Reads the the parser output file and returns it as an
         executable function. It is required when the models returns the
         simulation outputs in csv files.

        Returns
        -------
        Output : func
            Output parser function.

        """
        output_func_name = self.Output.parser

        output_func = getattr(__import__(output_func_name), output_func_name)

        file_names = []
        for File in self.Output.file_names:
            file_names.append(os.path.join(self.exe_path, File))
        try:
            output = output_func(self.name, file_names)
        except TypeError:
            output = output_func(file_names)
        return output

    # -------------------------------------------------------------------------
    def update_input_params(self, new_input_file, param_set):
        """
        Finds this pattern with <X1> in the new_input_file and replace it with
         the new value from the array param_sets.

        Parameters
        ----------
        new_input_file : list
            List of the input files with the adapted names.
        param_set : array of shape (n_params)
            Parameter set.

        Returns
        -------
        None.

        """
        NofPa = param_set.shape[0]
        text_to_search_list = [f'<X{i+1}>' for i in range(NofPa)]

        for filename in new_input_file:
            # Read in the file
            with open(filename, 'r') as file:
                filedata = file.read()

            # Replace the target string
            for text_to_search, params in zip(text_to_search_list, param_set):
                filedata = filedata.replace(text_to_search, f'{params:0.4e}')

            # Write the file out again
            with open(filename, 'w') as file:
                file.write(filedata)

    # -------------------------------------------------------------------------
    def run_command(self, command, output_file_names):
        """
        Runs the execution command given by the user to run the given model.
        It checks if the output files have been generated. If yes, the jobe is
        done and it extracts and returns the requested output(s). Otherwise,
        it executes the command again.

        Parameters
        ----------
        command : str
            The shell command to be executed.
        output_file_names : list
            Name of the output file names.

        Returns
        -------
        simulation_outputs : array of shape (n_obs, n_outputs)
            Simulation outputs.

        """

        # Check if simulation is finished
        while True:
            time.sleep(3)
            files = os.listdir(".")
            if all(elem in files for elem in output_file_names):
                break
            else:
                # Run command
                Process = os.system(f'./../{command}')
                if Process != 0:
                    print('\nMessage 1:')
                    print(f'\tIf the value of \'{Process}\' is a non-zero value'
                          ', then compilation problems occur \n' % Process)          
        os.chdir("..")

        # Read the output
        simulation_outputs = self.read_output()

        return simulation_outputs

    # -------------------------------------------------------------------------
    def run_forwardmodel(self, xx):
        """
        This function creates subdirectory for the current run and copies the
        necessary files to this directory and renames them. Next, it executes
        the given command.

        Parameters
        ----------
        xx : tuple
            A tuple including parameter set, simulation number and key string.

        Returns
        -------
        output : array of shape (n_outputs+1, n_obs)
            An array passed by the output paraser containing the x_values as
            the first row and the simulations results stored in the the rest of
            the array.

        """
        c_points, run_no, key_str = xx

        # Handle if only one imput file is provided
        if not isinstance(self.input_template, list):
            self.input_template = [self.input_template]
        if not isinstance(self.input_file, list):
            self.input_file = [self.input_file]

        new_input_file = []
        # Loop over the InputTemplates:
        for in_temp in self.input_template:
            if '/' in in_temp:
                in_temp = in_temp.split('/')[-1]
            new_input_file.append(in_temp.split('.tpl')[0] + key_str +
                                  f"_{run_no+1}" + in_temp.split('.tpl')[1])

        # Create directories
        newpath = self.name + key_str + f'_{run_no+1}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Copy the necessary files to the directories
        print(self.input_template)
        for in_temp in self.input_template:
            # Input file(s) of the model
            shutil.copy2(in_temp, newpath)
        # Auxiliary file
        if self.aux_file is not None:
            shutil.copy2(self.aux_file, newpath)  # Auxiliary file

        # Rename the Inputfile and/or auxiliary file
        os.chdir(newpath)
        for input_tem, input_file in zip(self.input_template, new_input_file):
            if '/' in input_tem:
                input_tem = input_tem.split('/')[-1]
            os.rename(input_tem, input_file)

        # Update the parametrs in Input file
        self.update_input_params(new_input_file, c_points)

        # Update the user defined command and the execution path
        try:
            new_command = self.shell_command.replace(self.input_file[0],
                                                     new_input_file[0])
            new_command = new_command.replace(self.input_file[1],
                                              new_input_file[1])
        except:
            new_command = self.shell_command.replace(self.input_file[0],
                                                     new_input_file[0])
        # Set the exe path if not provided
        if not bool(self.exe_path):
            self.exe_path = os.getcwd()

        # Run the model
        print(new_command)
        output = self.run_command(new_command, self.Output.file_names)

        return output

    # -------------------------------------------------------------------------
    def run_model_parallel(self, c_points, prevRun_No=0, key_str='',
                           mp=True, verbose=True):
        """
        Runs model simulations. If mp is true (default), then the simulations
         are started in parallel.

        Parameters
        ----------
        c_points : array of shape (n_samples, n_params)
            Collocation points (training set).
        prevRun_No : int, optional
            Previous run number, in case the sequential design is selected.
            The default is `0`.
        key_str : str, optional
            A descriptive string for validation runs. The default is `''`.
        mp : bool, optional
            Multiprocessing. The default is `True`.
        verbose: bool, optional
            Verbosity. The default is `True`.

        Returns
        -------
        all_outputs : dict
            A dictionary with x values (time step or point id) and all outputs.
            Each key contains an array of the shape `(n_samples, n_obs)`.
        new_c_points : array
            Updated collocation points (training set). If a simulation does not
            executed successfully, the parameter set is removed.

        """

        # Initilization
        n_c_points = len(c_points)
        all_outputs = {}
        
        # If the link type is UM-Bridge, then no parallel needs to be started from here
        if self.link_type.lower() == 'umbridge':
            import umbridge 
            if not hasattr(self, 'x_values'):
                raise AttributeError('For model type `umbridge` the attribute `x_values` needs to be set for the model!')
            # Init model
            #model = umbridge.HTTPModel('http://localhost:4242', 'forward')
            self.model = umbridge.HTTPModel(self.host, 'forward') # TODO: is this always forward?
            Function = self.uMBridge_model

        # Extract the function
        if self.link_type.lower() == 'function':
            # Prepare the function
            Function = getattr(__import__(self.py_file), self.py_file)
        # ---------------------------------------------------------------
        # -------------- Multiprocessing with Pool Class ----------------
        # ---------------------------------------------------------------
        # Start a pool with the number of CPUs
        if self.n_cpus is None:
            n_cpus = multiprocessing.cpu_count()
        else:
            n_cpus = self.n_cpus

        # Run forward model
        if n_c_points == 1 or not mp:
            if n_c_points== 1:
                if self.link_type.lower() == 'function' or self.link_type.lower() == 'umbridge':
                    group_results = Function(c_points, **self.func_args)
                else:
                    group_results = self.run_forwardmodel(
                        (c_points[0], prevRun_No, key_str)
                        )
            else:
                for i in range(c_points.shape[0]):
                    if i == 0:
                        if self.link_type.lower() == 'function' or self.link_type.lower() == 'umbridge':
                            group_results = Function(np.array([c_points[0]]), **self.func_args)
                        else:
                            group_results = self.run_forwardmodel(
                                (c_points[0], prevRun_No, key_str)
                                )
                        for key in group_results:
                            if key != 'x_values':
                                group_results[key] = [group_results[key]]
                    else: 
                        if self.link_type.lower() == 'function' or self.link_type.lower() == 'umbridge':
                            res = Function(np.array([c_points[i]]), **self.func_args)
                        else:
                            res = self.run_forwardmodel(
                                (c_points[i], prevRun_No, key_str)
                                )
                        for key in res:
                            if key != 'x_values':
                                group_results[key].append(res[key])
        
                for key in group_results:
                    if key != 'x_values':
                        group_results[key]= np.array(group_results[key])

        elif self.multi_process or mp:
            with get_context('spawn').Pool(n_cpus) as p:
            #with multiprocessing.Pool(n_cpus) as p:
                
                if self.link_type.lower() == 'function' or self.link_type.lower() == 'umbridge':
                    imap_var = p.imap(partial(Function, **self.func_args),
                                      c_points[:, np.newaxis])
                else:
                    args = zip(c_points,
                               [prevRun_No+i for i in range(n_c_points)],
                               [key_str]*n_c_points)
                    imap_var = p.imap(self.run_forwardmodel, args)

                if verbose:
                    desc = f'Running forward model {key_str}'
                    group_results = list(tqdm.tqdm(imap_var, total=n_c_points,
                                                   desc=desc))
                else:
                    group_results = list(imap_var)

        # Check for NaN
        for var_i, var in enumerate(self.Output.names):
            # If results are given as one dictionary
            if isinstance(group_results, dict):
                Outputs = np.asarray(group_results[var])
            # If results are given as list of dictionaries
            elif isinstance(group_results, list):
                Outputs = np.asarray([item[var] for item in group_results],
                                     dtype=np.float64)
            NaN_idx = np.unique(np.argwhere(np.isnan(Outputs))[:, 0])
            new_c_points = np.delete(c_points, NaN_idx, axis=0)
            all_outputs[var] = np.atleast_2d(
                np.delete(Outputs, NaN_idx, axis=0)
                )

        # Print the collocation points whose simulations crashed
        if len(NaN_idx) != 0:
            print('\n')
            print('*'*20)
            print("\nThe following parameter sets have been removed:\n",
                  c_points[NaN_idx])
            print("\n")
            print('*'*20)

        # Save time steps or x-values
        if isinstance(group_results, dict):
            all_outputs["x_values"] = group_results["x_values"]
        elif any(isinstance(i, dict) for i in group_results):
            all_outputs["x_values"] = group_results[0]["x_values"]

        # Store simulations in a hdf5 file
        self._store_simulations(
            c_points, all_outputs, NaN_idx, key_str, prevRun_No
            )

        return all_outputs, new_c_points
    
    def uMBridge_model(self, params):
        """
        Function that calls a UMBridge model and transforms its output into the 
        shape expected for the surrogate.
    
        Parameters
        ----------
        params : 2d np.array, shape (#samples, #params)
            The parameter values for which the model is run.
    
        Returns
        -------
        dict
            The transformed model outputs.
    
        """
        # Run the model
        #out = np.array(model(np.ndarray.tolist(params), {'level':0}))
        out = np.array(self.model(np.ndarray.tolist(params), self.modelparams))
        
        # Sort into dict
        out_dict = {}
        cnt = 0
        for key in self.Output.names:
        #    # If needed resort into single-value outputs
        #    if self.output_type == 'single-valued':
        #        if out.shape[1]>1:  # TODO: this doesn't fully seem correct??
        #            for i in range(out[:,key]): # TODO: this doesn't fully seem correct??
        #                new_key = key+str(i)
        #                if new_key not in self.Output.names:
        #                    self.Output.names.append(new_key)
        #                    if i == 0:
        #                        self.Ouptut.names.remove(key)
        #                out_dict[new_key] = out[:,cnt,i] # TODO: not sure about this, need to test
        #        else: 
        #            out_dict[key] = out[:,cnt]
        #            
        #        
        #    else:
            out_dict[key] = out[:,cnt]
            cnt += 1
        
            
        ## TODO: how to deal with the x-values?
        #if self.output_type == 'single-valued':
        #    out_dict['x_values'] = [0]
        #else:
        #    out_dict['x_values'] = np.arange(0,out[:,0].shape[0],1)
        out_dict['x_values'] = self.x_values
        
        #return {'T1':out[:,0], 'T2':out[:,1], 'H1':out[:,2], 'H2':out[:,3], 
       #         'x_values':[0]}
        return out_dict

    # -------------------------------------------------------------------------
    def _store_simulations(self, c_points, all_outputs, NaN_idx, key_str,
                           prevRun_No):
        """
        

        Parameters
        ----------
        c_points : TYPE
            DESCRIPTION.
        all_outputs : TYPE
            DESCRIPTION.
        NaN_idx : TYPE
            DESCRIPTION.
        key_str : TYPE
            DESCRIPTION.
        prevRun_No : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # Create hdf5 metadata
        if key_str == '':
            hdf5file = f'ExpDesign_{self.name}.hdf5'
        else:
            hdf5file = f'ValidSet_{self.name}.hdf5'
        hdf5_exist = os.path.exists(hdf5file)
        file = h5py.File(hdf5file, 'a')

        # ---------- Save time steps or x-values ----------
        if not hdf5_exist:
            if type(all_outputs["x_values"]) is dict:
                grp_x_values = file.create_group("x_values/")
                for varIdx, var in enumerate(self.Output.names):
                    grp_x_values.create_dataset(
                        var, data=all_outputs["x_values"][var]
                        )
            else:
                file.create_dataset("x_values", data=all_outputs["x_values"])

        # ---------- Save outputs ----------
        for varIdx, var in enumerate(self.Output.names):
            if not hdf5_exist:
                grpY = file.create_group("EDY/"+var)
            else:
                grpY = file.get("EDY/"+var)

            if prevRun_No == 0 and key_str == '':
                grpY.create_dataset(f'init_{key_str}', data=all_outputs[var])
            else:
                try:
                    oldEDY = np.array(file[f'EDY/{var}/adaptive_{key_str}'])
                    del file[f'EDY/{var}/adaptive_{key_str}']
                    data = np.vstack((oldEDY, all_outputs[var]))
                except KeyError:
                    data = all_outputs[var]
                grpY.create_dataset('adaptive_'+key_str, data=data)

            if prevRun_No == 0 and key_str == '':
                grpY.create_dataset(f"New_init_{key_str}",
                                    data=all_outputs[var])
            else:
                try:
                    name = f'EDY/{var}/New_adaptive_{key_str}'
                    oldEDY = np.array(file[name])
                    del file[f'EDY/{var}/New_adaptive_{key_str}']
                    data = np.vstack((oldEDY, all_outputs[var]))
                except KeyError:
                    data = all_outputs[var]
                grpY.create_dataset(f'New_adaptive_{key_str}', data=data)

        # ---------- Save CollocationPoints ----------
        new_c_points = np.delete(c_points, NaN_idx, axis=0)
        grpX = file.create_group("EDX") if not hdf5_exist else file.get("EDX")
        if prevRun_No == 0 and key_str == '':
            grpX.create_dataset("init_"+key_str, data=c_points)
            if len(NaN_idx) != 0:
                grpX.create_dataset("New_init_"+key_str, data=new_c_points)

        else:
            try:
                name = f'EDX/adaptive_{key_str}'
                oldCollocationPoints = np.array(file[name])
                del file[f'EDX/adaptive_{key_str}']
                data = np.vstack((oldCollocationPoints, new_c_points))
            except KeyError:
                data = new_c_points
            grpX.create_dataset('adaptive_'+key_str, data=data)

            if len(NaN_idx) != 0:
                try:
                    name = f'EDX/New_adaptive_{key_str}'
                    oldCollocationPoints = np.array(file[name])
                    del file[f'EDX/New_adaptive_{key_str}']
                    data = np.vstack((oldCollocationPoints, new_c_points))
                except KeyError:
                    data = new_c_points
                grpX.create_dataset('New_adaptive_'+key_str, data=data)

        # Close h5py file
        file.close()

    # -------------------------------------------------------------------------
    def zip_subdirs(self, dir_name, key):
        """
        Zips all the files containing the key(word).

        Parameters
        ----------
        dir_name : str
            Directory name.
        key : str
            Keyword to search for.

        Returns
        -------
        None.

        """
        # setup file paths variable
        dir_list = []
        file_paths = []

        # Read all directory, subdirectories and file lists
        dir_path = os.getcwd()

        for root, directories, files in os.walk(dir_path):
            for directory in directories:
                # Create the full filepath by using os module.
                if key in directory:
                    folderPath = os.path.join(dir_path, directory)
                    dir_list.append(folderPath)

        # Loop over the identified directories to store the file paths
        for direct_name in dir_list:
            for root, directories, files in os.walk(direct_name):
                for filename in files:
                    # Create the full filepath by using os module.
                    filePath = os.path.join(root, filename)
                    file_paths.append('.'+filePath.split(dir_path)[1])

        # writing files to a zipfile
        if len(file_paths) != 0:
            zip_file = zipfile.ZipFile(dir_name+'.zip', 'w')
            with zip_file:
                # writing each file one by one
                for file in file_paths:
                    zip_file.write(file)

            file_paths = [path for path in os.listdir('.') if key in path]

            for path in file_paths:
                shutil.rmtree(path)

            print("\n")
            print(f'{dir_name}.zip has been created successfully!\n')

        return
