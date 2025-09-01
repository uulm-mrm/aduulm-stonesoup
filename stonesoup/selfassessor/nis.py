import numpy as np
from typing import List
from numpy.typing import ArrayLike
from scipy.stats.distributions import chi2
from copy import deepcopy
import math


class NIS:
    """Class for self-assessment using the normalized innovation squared (NIS).

    Attributes
    ----------
    window_length : int
        window used for averaging
        size of the window over which the mean is calculated to smooth the output.
    alpha : float
        alpha used for the confidence interval.
    sigma_cutoff: float
        cutoff distance for association.
    mapping: List[int]
        List of measurement entries considered for the NIS calculation.
    symmetric: bool
        decision if the confidence interval should be symmetric or not.
    dim : int
        dimension of the measurement space.
    nis : float
        current value of the normalized innovation squared.
    nis_window : array_like, float
        memory of NIS values of past window_length time steps.
        Oldest value gets deleted when a new value is inserted or no NIS could be
        calculated.
    nis_averaged : float
        current value of the averaged nis.
    lower_bound : float
        lower bound of confidence interval. Only calculated when nis_window has at least
        one value. 1-alpha of all values should lie inside the confidence interval.
    upper_bound : float
        upper bound of confidence interval. Only calculated when nis_window has at least
        one value. 1-alpha of all values should lie inside the confidence interval.

    """

    def __init__(self, window_length: int = 35, alpha: float = 0.05, dim: int = 2,
                 sigma_cutoff: float = None, mapping: List[int] = None, symmetric_border: int = 7,
                 probability_detected: float = None, clutter_exp_number: int = None,
                 fov: list = [], association_algorithm: str = None):
        """Construct NIS object.

        This NIS version also estimates the PDA-NIS based on the method and table presented in
        "Multitarget-Multisensor Tracking: Principles and Techniques" (1995), Page 284.
        The corresponding table can be found in the paper:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1103935&tag=1
        Note: This is an approximation, as the parameters are derived from a graph.
        Therefore, the accuracy of the estimated parameters is not guaranteed.

        Parameters
        ----------
        window_length : int
            window used for averaging
            size of the window over which the mean is calculated to smooth the output
        alpha : float , optional
            alpha used for the confidence interval. The default is 0.05.
        dim : int , optional
            dimension of the measurement space. The default is 2. Not needed if a mapping is provided.
        sigma_cutoff : float , optional
            cutoff distance for association. The default is None.
        mapping: List, optional
            List of the measurement vector entries that should contribute to the NIS, others are ignored.
        symmetric_border: int , optional
            For dim < symmetric_border an asymmetric confidence interval will be used, whereas a symmetric one
            is used for dim >= symmetric_border.
            The default is 7.
        probability_detected: float
            target Detection Probability
        clutter_exp_number: int
            expected number of clutter measurements (false detection)
        fov: List [fov_x, fov_v_x, fov_y, fov_v_y], optional
            sensor's field of view (FOV)
        association_algorithm: String
            indicates which association algorithm is used

        Returns
        -------
        NIS.

        """
        self.none_nis = False
        self.association_algorithm = association_algorithm if association_algorithm is not None else 'nn'

        if self.association_algorithm == 'pda':
            self.probability_detected = round(probability_detected, 1)
        else:
            self.probability_detected = probability_detected

        if self.association_algorithm.lower() == 'pda':
            self.dict_parameter_q = {
                1.0: {0: 1.00, 1: 0.70, 2: 0.60, 3: 0.51, 4: 0.45, 5: 0.38},
                0.9: {0: 0.85, 1: 0.61, 2: 0.46, 3: 0.36, 4: 0.29, 5: 0.24},
                0.8: {0: 0.78, 1: 0.50, 2: 0.38, 3: 0.31, 4: 0.26, 5: 0.22},
                0.7: {0: 0.68, 1: 0.39, 2: 0.28, 3: 0.22, 4: 0.19, 5: 0.16},
                0.6: {0: 0.58, 1: 0.30, 2: 0.21, 3: 0.17, 4: 0.13, 5: 0.12},
                0.5: {0: 0.48, 1: 0.23, 2: 0.15, 3: 0.11, 4: 0.09, 5: 0.085},
                0.4: {0: 0.38, 1: 0.15, 2: 0.10, 3: 0.08, 4: 0.06, 5: 0.06},
                0.3: {0: 0.28, 1: 0.09, 2: 0.06, 3: 0.05, 4: 0.05, 5: 0.05},
                0.2: {0: 0.18, 1: 0.05, 2: 0.04, 3: 0.03, 4: 0.03, 5: 0.03},
                0.1: {0: 0.08, 1: 0.015, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01}
            }
            self.fov = fov
            self.clutter_spatial_density = clutter_exp_number

        self.pD_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self.clutter_values = [0, 1, 2, 3, 4, 5]
        self.window_length = window_length
        self.alpha = alpha
        self.sigma_cutoff = sigma_cutoff
        self.mapping = mapping
        self.symmetric_border = symmetric_border

        if mapping is None:
            self.dim = dim
        else:
            self.dim = len(mapping)

        self.symmetric = False if self.dim < symmetric_border else True

        self.nis = 0
        self.nis_window = []
        self.nis_averaged = 0
        self.lower_bound = 0
        self.upper_bound = 0

    def assess(self, z_pred: ArrayLike, S_pred: ArrayLike, z: ArrayLike, volume_validation_gate=0):
        """Calculate NIS value.

        Parameters
        ----------
        z_pred : array_like, float
            predicted measurement as given by the Kalman filter.
        S_pred : array_like, float
            innovation covariance matrix.
        z : array_like, float
            measurement provided by the simulator or sensor.
            or the Combined Innovation for the PDA.
        volume_validation_gate: float
            The parameter is needed to calculate the expected number of false measurements
            inside the validation gate to choose the correct parameter q.

        Returns
        -------
        nis_averaged : float
            averaged value of the NIS. Only returned when provided measurement is not
            None.
        lower_bound : float
            lower bound of confidence interval. Only returned when nis_window has at
            least one value. 1-alpha of all values should lie inside the confidence
            interval.
        upper_bound : float
            upper bound of confidence interval. Only returned when nis_window has at
            least one value. 1-alpha of all values should lie inside the confidence
            interval.
        n : int
            number of nis values used for nis_averaged calculation
        """

        if (self.sigma_cutoff is not None) and (S_pred is None):
            ValueError('Covariance has to be provided, when NIS has to be calculated for\
                       a missing associated measurement')

        # if there is no associated measurement
        if z is None:
            self.none_nis = True
            if self.sigma_cutoff is None:
                # return None as NIS value
                self.nis = None
            elif self.window_length == 1:
                self.nis = self.sigma_cutoff**2
            else:
                # the normalized innovation is given by the parameter sigma_cutoff
                # this states the distance in the measurement space which is allowed
                # as the maximum before there is no association and thus no value for z
                # self.nis = self.sigma_cutoff**2
                self.nis = None
        else:
            assert not hasattr(z, '__len__') or not hasattr(z_pred, '__len__') or (len(z) == len(z_pred)), \
                "Measurement and predicted measurement need to have the same dimension"

            # filter based on mapping (if provided)
            self.none_nis = False

            if self.association_algorithm.lower() == 'pda':
                if self.dim == 1:
                    c = 2
                elif self.dim == 2:
                    c = np.pi
                else:
                    c = (4 * np.pi) / 3

                volume_validation_gate = c * np.power(self.sigma_cutoff, self.dim) * np.sqrt(np.linalg.det(S_pred))
                # number of false measurements inside the validation gate according to
                # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1103935&tag=1
                clutter_in_gate = self.clutter_spatial_density / (self.fov[0] * self.fov[2]) * volume_validation_gate
                lower_bound = math.floor(clutter_in_gate)
                upper_bound = math.ceil(clutter_in_gate)

                if upper_bound > 5:
                    parameter_q = self.dict_parameter_q[self.probability_detected][5]
                    print(f"Warning: Estimation of reduction parameter q = {parameter_q} was inaccurate "
                          f"for at least one time step. More parameters: "
                          f"volume_validation_gate = {volume_validation_gate}, "
                          f"clutter_in_gate = {clutter_in_gate}, "
                          f"lower_bound = {lower_bound}, "
                          f"and upper_bound = {upper_bound}.")
                else:
                    parameter_q_down = self.dict_parameter_q[self.probability_detected][lower_bound]
                    parameter_q_up = self.dict_parameter_q[self.probability_detected][upper_bound]
                    parameter_q = parameter_q_down + (parameter_q_up - parameter_q_down) * (
                                clutter_in_gate - lower_bound)
                S_pred = parameter_q * deepcopy(S_pred)
            else:
                S_pred = S_pred

            if self.mapping is not None:
                assert 0 < len(self.mapping) <= len(z_pred), 'WARNING: Invalid mapping specified in init'
                assert np.amax(self.mapping) < len(z_pred), 'WARNING: Invalid mapping specified in init'
                assert np.amin(self.mapping) >= 0, 'WARNING: Invalid mapping specified in init'

                z_pred = z_pred[self.mapping]
                z = z[self.mapping]
                S_pred = S_pred[tuple([self.mapping])][:, self.mapping]

            # PDA-NIS has as input the combined innovation, which already calculates the residual.
            if self.association_algorithm.lower() == 'pda':
                residual = z
            else:
                residual = z-z_pred
            self.nis = np.ndarray.item(
                np.dot(np.transpose(residual), np.dot(np.linalg.inv(S_pred), residual))
            )

        # remove the oldest value if the window size is reached
        if len(self.nis_window) >= self.window_length:
            del self.nis_window[0]
        # append new value to memory
        self.nis_window.append(self.nis)
        # get nis_window without the None-elements
        nis_values = [x for x in self.nis_window if x is not None]
        n = len(nis_values)
        # if there is at least one NIS value in the window
        if n > 0:
            # average over window
            self.nis_averaged = sum(nis_values) / n
            # calculate bounds depending on symmetric or asymmetric bounds based on
            # Bar-Shalom "Estimation with Applications to Tracking and Navigation"
            # Section: 5.4.3 Examples of Filter Consistency Testing (page 237-244)
            if n*self.dim > self.symmetric_border:
                self.lower_bound = chi2.ppf(self.alpha/2, n * self.dim) / n
                self.upper_bound = chi2.ppf(1 - self.alpha/2, n * self.dim) / n
            else:
                self.lower_bound = 0
                self.upper_bound = chi2.ppf(1-self.alpha, n * self.dim) / n
        # if there is no NIS value in the whole window
        else:
            # all values are None
            self.nis_averaged = None
            if n*self.dim > self.symmetric_border:
                self.lower_bound = chi2.ppf(self.alpha/2, self.dim) / 1
                self.upper_bound = chi2.ppf(1 - self.alpha/2, self.dim) / 1
            else:
                self.lower_bound = 0
                self.upper_bound = chi2.ppf(1-self.alpha, self.dim) / 1

        return [self.nis_averaged, [self.lower_bound, self.upper_bound], n]
