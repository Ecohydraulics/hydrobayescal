#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input space built from set prior distributions
"""

import numpy as np
import chaospy
import scipy.stats as st


class InputSpace:
    """
    This class generates the input space for the metamodel from the
    distributions provided using the `Input` object.

    Attributes
    ----------
    Input : obj
        Input object containing the parameter marginals, i.e. name,
        distribution type and distribution parameters or available raw data.
    meta_Model_type : str
        Type of the meta_Model_type.

    """

    def __init__(self, Input, meta_Model_type='pce'):
        self.InputObj = Input
        self.meta_Model_type = meta_Model_type
        
        # Other 
        self.apce = None
        self.ndim = None
        
        # Init 
        self.check_valid_inputs()
        
        
    def check_valid_inputs(self)-> None:
        """
        Check if the given InputObj is valid to use for further calculations:
            Has some Marginals
            Marginals have valid priors
            All Marginals given as the same type (samples vs dist)

        Returns
        -------
        None

        """
        Inputs = self.InputObj
        self.ndim = len(Inputs.Marginals)
        
        # Check if PCE or aPCE metamodel is selected.
        # TODO: test also for 'pce'??
        if self.meta_Model_type.lower() == 'apce':
            self.apce = True
        else:
            self.apce = False

        # check if marginals given 
        if not self.ndim >=1:
            raise AssertionError('Cannot build distributions if no marginals are given')
            
        # check that each marginal is valid
        for marginals in Inputs.Marginals:
            if len(marginals.input_data) == 0:
                if marginals.dist_type == None:
                    raise AssertionError('Not all marginals were provided priors')
                    break
            if np.array(marginals.input_data).shape[0] and (marginals.dist_type != None):
                raise AssertionError('Both samples and distribution type are given. Please choose only one.')
                break
                
        # Check if input is given as dist or input_data.
        self.input_data_given = -1
        for marg in Inputs.Marginals:
            #print(self.input_data_given)
            size = np.array(marg.input_data).shape[0]
            #print(f'Size: {size}')
            if size and abs(self.input_data_given) !=1:
                self.input_data_given = 2
                break
            if (not size) and self.input_data_given > 0:
                self.input_data_given = 2
                break
            if not size:
                self.input_data_given = 0
            if size:
                self.input_data_given = 1
                
        if self.input_data_given == 2:
            raise AssertionError('Distributions cannot be built as the priors have different types')
            
    
        # Get the bounds if input_data are directly defined by user:
        if self.input_data_given:
            for i in range(self.ndim):
                low_bound = np.min(Inputs.Marginals[i].input_data)
                up_bound = np.max(Inputs.Marginals[i].input_data)
                Inputs.Marginals[i].parameters = [low_bound, up_bound]

  

    # -------------------------------------------------------------------------
    def init_param_space(self, max_deg=None):
        """
        Initializes parameter space.

        Parameters
        ----------
        max_deg : int, optional
            Maximum degree. The default is `None`.

        Creates
        -------
        raw_data : array of shape (n_params, n_samples)
            Raw data.
        bound_tuples : list of tuples
            A list containing lower and upper bounds of parameters.

        """
        # Recheck all before running!
        self.check_valid_inputs()
        
        Inputs = self.InputObj
        ndim = self.ndim
        rosenblatt_flag = Inputs.Rosenblatt
        mc_size = 50000

        # Save parameter names
        self.par_names = []
        for parIdx in range(ndim):
            self.par_names.append(Inputs.Marginals[parIdx].name)

        # Create a multivariate probability distribution
        # TODO: change this to make max_deg obligatory? at least in some specific cases?
        if max_deg is not None:
            JDist, poly_types = self.build_polytypes(rosenblatt=rosenblatt_flag)
            self.JDist, self.poly_types = JDist, poly_types

        if self.input_data_given:
            self.MCSize = len(Inputs.Marginals[0].input_data)
            self.raw_data = np.zeros((ndim, self.MCSize))

            for parIdx in range(ndim):
                # Save parameter names
                try:
                    self.raw_data[parIdx] = np.array(
                        Inputs.Marginals[parIdx].input_data)
                except:
                    self.raw_data[parIdx] = self.JDist[parIdx].sample(mc_size)

        else:
            # Generate random samples based on parameter distributions
            self.raw_data = chaospy.generate_samples(mc_size,
                                                     domain=self.JDist)

        # Extract moments
        for parIdx in range(ndim):
            mu = np.mean(self.raw_data[parIdx])
            std = np.std(self.raw_data[parIdx])
            self.InputObj.Marginals[parIdx].moments = [mu, std]

        # Generate the bounds based on given inputs for marginals
        bound_tuples = []
        for i in range(ndim):
            if Inputs.Marginals[i].dist_type == 'unif':
                low_bound = Inputs.Marginals[i].parameters[0]
                up_bound = Inputs.Marginals[i].parameters[1]
            else:
                low_bound = np.min(self.raw_data[i])
                up_bound = np.max(self.raw_data[i])

            bound_tuples.append((low_bound, up_bound))

        self.bound_tuples = tuple(bound_tuples)

    # -------------------------------------------------------------------------
    def build_polytypes(self, rosenblatt):
        """
        Creates the polynomial types to be passed to univ_basis_vals method of
        the MetaModel object.

        Parameters
        ----------
        rosenblatt : bool
            Rosenblatt transformation flag.

        Returns
        -------
        orig_space_dist : object
            A chaospy JDist object or a gaussian_kde object.
        poly_types : list
            List of polynomial types for the parameters.

        """
        Inputs = self.InputObj
        
        all_data = []
        all_dist_types = []
        orig_joints = []
        poly_types = []
        
        for parIdx in range(self.ndim):

            if Inputs.Marginals[parIdx].dist_type is None:
                data = Inputs.Marginals[parIdx].input_data
                all_data.append(data)
                dist_type = None
            else:
                dist_type = Inputs.Marginals[parIdx].dist_type
                params = Inputs.Marginals[parIdx].parameters

            if rosenblatt:
                polytype = 'hermite'
                dist = chaospy.Normal()

            elif dist_type is None:
                polytype = 'arbitrary'
                dist = None

            elif 'unif' in dist_type.lower():
                polytype = 'legendre'
                if not np.array(params).shape[0]>=2:
                    raise AssertionError('Distribution has too few parameters!')
                dist = chaospy.Uniform(lower=params[0], upper=params[1])

            elif 'norm' in dist_type.lower() and \
                 'log' not in dist_type.lower():
                if not np.array(params).shape[0]>=2:
                    raise AssertionError('Distribution has too few parameters!')
                polytype = 'hermite'
                dist = chaospy.Normal(mu=params[0], sigma=params[1])

            elif 'gamma' in dist_type.lower():
                polytype = 'laguerre'
                if not np.array(params).shape[0]>=3:
                    raise AssertionError('Distribution has too few parameters!')
                dist = chaospy.Gamma(shape=params[0],
                                     scale=params[1],
                                     shift=params[2])

            elif 'beta' in dist_type.lower():
                if not np.array(params).shape[0]>=4:
                    raise AssertionError('Distribution has too few parameters!')
                polytype = 'jacobi'
                dist = chaospy.Beta(alpha=params[0], beta=params[1],
                                    lower=params[2], upper=params[3])

            elif 'lognorm' in dist_type.lower():
                polytype = 'hermite'
                if not np.array(params).shape[0]>=2:
                    raise AssertionError('Distribution has too few parameters!')
                mu = np.log(params[0]**2/np.sqrt(params[0]**2 + params[1]**2))
                sigma = np.sqrt(np.log(1 + params[1]**2 / params[0]**2))
                dist = chaospy.LogNormal(mu, sigma)
                # dist = chaospy.LogNormal(mu=params[0], sigma=params[1])

            elif 'expon' in dist_type.lower():
                polytype = 'exponential'
                if not np.array(params).shape[0]>=2:
                    raise AssertionError('Distribution has too few parameters!')
                dist = chaospy.Exponential(scale=params[0], shift=params[1])

            elif 'weibull' in dist_type.lower():
                polytype = 'weibull'
                if not np.array(params).shape[0]>=3:
                    raise AssertionError('Distribution has too few parameters!')
                dist = chaospy.Weibull(shape=params[0], scale=params[1],
                                       shift=params[2])

            else:
                message = (f"DistType {dist_type} for parameter"
                           f"{parIdx+1} is not available.")
                raise ValueError(message)

            if self.input_data_given or self.apce:
                polytype = 'arbitrary'

            # Store dists and poly_types
            orig_joints.append(dist)
            poly_types.append(polytype)
            all_dist_types.append(dist_type)

        # Prepare final output to return
        if None in all_dist_types:
            # Naive approach: Fit a gaussian kernel to the provided data
            Data = np.asarray(all_data)
            try:
                orig_space_dist = st.gaussian_kde(Data)
            except:
                raise ValueError('The samples provided to the Marginals should be 1D only')
            self.prior_space = orig_space_dist
        else:
            orig_space_dist = chaospy.J(*orig_joints)
            try:
                self.prior_space = st.gaussian_kde(orig_space_dist.sample(10000))
            except:
                raise ValueError('Parameter values are not valid, please set differently')

        return orig_space_dist, poly_types

    # -------------------------------------------------------------------------
    def transform(self, X, params=None, method=None):
        """
        Transforms the samples via either a Rosenblatt or an isoprobabilistic
        transformation.

        Parameters
        ----------
        X : array of shape (n_samples,n_params)
            Samples to be transformed.
        method : string
            If transformation method is 'user' transform X, else just pass X.

        Returns
        -------
        tr_X: array of shape (n_samples,n_params)
            Transformed samples.

        """
        # Check for built JDist
        if not hasattr(self, 'JDist'):
            raise AttributeError('Call function init_param_space first to create JDist')
            
        # Check if X is 2d
        if X.ndim != 2:
            raise AttributeError('X should have two dimensions')
            
        # Check if size of X matches Marginals
        if X.shape[1]!= self.ndim:
            raise AttributeError('The second dimension of X should be the same size as the number of marginals in the InputObj')
        
        if self.InputObj.Rosenblatt:
            self.origJDist, _ = self.build_polytypes(False)
            if method == 'user':
                tr_X = self.JDist.inv(self.origJDist.fwd(X.T)).T
            else:
                # Inverse to original spcace -- generate sample ED
                tr_X = self.origJDist.inv(self.JDist.fwd(X.T)).T
        else:
            # Transform samples via an isoprobabilistic transformation
            n_samples, n_params = X.shape
            Inputs = self.InputObj
            origJDist = self.JDist
            poly_types = self.poly_types

            disttypes = []
            for par_i in range(n_params):
                disttypes.append(Inputs.Marginals[par_i].dist_type)

            # Pass non-transformed X, if arbitrary PCE is selected.
            if None in disttypes or self.input_data_given or self.apce:
                return X

            cdfx = np.zeros((X.shape))
            tr_X = np.zeros((X.shape))

            for par_i in range(n_params):

                # Extract the parameters of the original space
                disttype = disttypes[par_i]
                if disttype is not None:
                    dist = origJDist[par_i]
                else:
                    dist = None
                polytype = poly_types[par_i]
                cdf = np.vectorize(lambda x: dist.cdf(x))

                # Extract the parameters of the transformation space based on
                # polyType
                if polytype == 'legendre' or disttype == 'uniform':
                    # Generate Y_Dists based
                    params_Y = [-1, 1]
                    dist_Y = st.uniform(loc=params_Y[0],
                                        scale=params_Y[1]-params_Y[0])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                elif polytype == 'hermite' or disttype == 'norm':
                    params_Y = [0, 1]
                    dist_Y = st.norm(loc=params_Y[0], scale=params_Y[1])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                elif polytype == 'laguerre' or disttype == 'gamma':
                    if params == None:
                        raise AttributeError('Additional parameters have to be set for the gamma distribution!')
                    params_Y = [1, params[1]]
                    dist_Y = st.gamma(loc=params_Y[0], scale=params_Y[1])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                # Compute CDF_x(X)
                cdfx[:, par_i] = cdf(X[:, par_i])

                # Compute invCDF_y(cdfx)
                tr_X[:, par_i] = inv_cdf(cdfx[:, par_i])

        return tr_X
