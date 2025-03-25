from copy import deepcopy
import numpy as np
import numpy.typing as ArrayLike
from typing import List
from scipy.linalg import sqrtm
from scipy.special import gammaincinv
from stonesoup.subjective_logic import subjective_logic as sl
from stonesoup.selfassessor._gauss_variable import GaussVariable1D, GaussVariableND
from stonesoup.selfassessor._threshold import calc_threshold_op_diff as calc_tsh
from stonesoup.selfassessor._threshold import calc_threshold_n_diff as calc_tsh_n


class KalmanSelfAssessor:
    """Class for self assessment in Kalman filtering using subjective logic.

    Attributes
    ----------
    n_st : int
        Length of the region of the short term opinion.
    prob_detect : float, optional
        Probability of detection for each measurement. The default is 0.9.
    n_c : int
        Length of window between comparisons between long and short term opinion.
    threshold_ltst : float
        Threshold up to which the degree of conflict between long and short term
        opinion can go until long term opinion is reset.
    _threshold_dc : float
        DC-threshold above which two opinions are considered different.
    alpha_threshold_dc : float
        Alpha used for the confidence for the DC-threshold.
    trust_discount : float
        Trust discount probability of long term opinion.
    _delta : float
        DC between complete opinion and reference.
    _u_delta : float
        Uncertainty of the complete opinion.
    _counter_lt_op : int
        Counter for the number of evidence accumulated in the long term opinion.
    _counter_st_op : int
        Counter for the number of evidence accumulated in the short term opinion.
    _counter_compar : int
        Counter for the number of steps taken since last comparison between long and short
        term opinion.
    _X : GaussVariable1D or GaussVariableND
        One of the stated types to represent the reference distribution used in generating
        the opinion of the incoming measurement.
    _op_0 : MultiOpinion
        Vague opinion with base rate according to the reference distribution.
    _op_G : MultiOpinion
        Dogmatic opinion with belief and base rate according to the
        reference distribution.
    _op_st : MultiOpinion
        Opinion for the short term evaluation of the tracking performance.
    _op_lt : MultiOpinion
        Opinion for the long term evaluation of the tracking performance.
    _op_X : MultiOpinion
        Opinion for the evaluation of the tracking performance.
    _op_z : MultiOpinion
        Opinion for the current measurement.
    _op_z_memory : array_like, MultiOpinion
        Storage of the last n_st opinions of the measurement.

    """

    def __init__(self, num_X: int, n_st: int, n_c: int = 1, prob_detect: float = 1, dim_meas: int = 2,
                 threshold_ltst: float = None, alpha_threshold_dc: float = 0.1, sigma_cutoff: float = None,
                 trust_discount: float = 0.9, type_compar: str = 'dimensional', mapping: List[int] = None):
        """Construct KalmanSelfAssessor object.

        Parameters
        ----------
        num_X : int
            Number of bins in which a distribution is divided.
        n_st : int
            Length of the region of the short term opinion.
        n_c : int, optional
            Length of window between comparisons between long and short term opinion.
            The default is 1.
        prob_detect : float, optional
            Probability of detection for each measurement. The default is 0.9.
        dim_meas : int, optional
            Number of dimensions of the measurement. The default is 2.
        threshold_ltst : float, optional
            Threshold up to which the degree of conflict between long and short term
            opinion can go until long term opinion is reset. The default is 0.2.
        alpha_threshold_dc : float, optional
            Alpha used for the confidence for the DC-threshold. The default is 0.1.
        sigma_cutoff : float, optional
            Distance after which only one bin is considered. The default is 3.
        trust_discount : float, optional
            Trust discount probability of long term opinion. The default is 0.9.
        type_compar : string, optional
            Type on how the Opinion regarding the measurement should be generated.
            'elementwise': each element in the measurement is compared to a one
                dimensional normal distribution.
            'dimensional': the whole vector is classified using the distance to the mean
            The default is 'dimensional'.
        mapping: List, optional
            List of the measurement vector entries that should contribute to the SL-SAS, others are ignored.

        Raises
        ------
        ValueError
            If parameter for type_compar has a value different from the two possible values.

        Returns
        -------
        KalmanSelfAssessor.

        """
        self.n_st = n_st
        self.n_c = n_c
        self.prob_detect = prob_detect
        if mapping is None:
            self.dim_meas = dim_meas
        else:
            self.dim_meas = len(mapping)
        self.alpha_threshold_dc = alpha_threshold_dc
        self.trust_discount = trust_discount

        # counters
        self._counter_lt_op = 0
        self._counter_st_op = 0
        self._counter_compar = 0

        # storage for auxiliary variables
        if sigma_cutoff is None:
            sigma_cutoff = np.sqrt(2 * gammaincinv(dim_meas / 2, 1 - alpha_threshold_dc))

        if type_compar == 'elementwise':
            self._X = GaussVariable1D(num_X, sigma_cutoff)
        elif type_compar == 'dimensional':
            self._X = GaussVariableND(self.dim_meas, num_X, sigma_cutoff)
        else:
            raise ValueError("type_compar has to be either 'elementwise' or 'dimensional'")
        self._op_0 = sl.MultiOpinion(np.zeros(num_X), self._X.probability_bins, 1)
        self._op_G = sl.MultiOpinion(self._X.probability_bins,
                                     self._X.probability_bins, 0)
        self._op_st = deepcopy(self._op_0)
        self._op_lt = deepcopy(self._op_0)
        self._op_z = deepcopy(self._op_0)
        self._op_z_memory = []

        # storage for values
        self._threshold_dc = 0
        self._delta = 0
        self._u_delta = 1
        self._op_X = deepcopy(self._op_0)
        # component filtering during assess
        self.mapping = mapping

        # calculate the threshold for _op_st and _op_lt fusion
        if threshold_ltst is None:
            self.threshold_ltst = calc_tsh_n(num_X, self.n_st, self.alpha_threshold_dc)
        else:
            self.threshold_ltst = threshold_ltst

    def generate_opinion_over_X(self, z: ArrayLike):
        """Generate opinion of measurement regarding reference distribution.

        Parameters
        ----------
        z : array_like, float
            Current measurement.

        Raises
        ------
        TypeError
            X has to be of type GaussVariable in 1 or N dimensions.

        Returns
        -------
        op : MultiOpinion
            Opinion for the current measurement.

        """
        evidence_meas = np.zeros(self._X.number_bins)
        if z is None:
            # Generate evidence using different methods depending on the reference Gauss
            # distribution provided.
            # Here only a simple case is considered. If the object was not detected,
            # it is assumed, that the object lies outside of the sigma interval used for
            # association. This is a good assumption if the detection probability is high.
            if isinstance(self._X, GaussVariable1D):
                # inside with missed detection
                evidence_meas[1:-1] = 0
                # outside
                evidence_meas[-1] = 0.5
                evidence_meas[0] = 0.5
            elif isinstance(self._X, GaussVariableND):
                # inside with missed detection
                evidence_meas[0:-1] = 0
                # outside
                evidence_meas[-1] = 1
            else:
                raise TypeError('X has to be of type GaussVariable in 1 or N dimensions')
        else:
            # Generate evidence using different methods depending on the reference Gauss
            # distribution provided
            if isinstance(self._X, GaussVariable1D):
                # separate for each dimension instead as measure for higher dimension
                # separation should work because the residual in each direction is
                # independent from each other
                for z_i in z:
                    evidence_meas += self._X.check_value_for_bin(z_i)
            elif isinstance(self._X, GaussVariableND):
                # using the distance to the mean value (due to the normalization this is 0)
                # evidence states, that value lies inside certain radius around 0
                evidence_meas = self._X.check_value_for_bin(np.linalg.norm(z))
            else:
                raise TypeError('X has to be of type GaussVariable in 1 or N dimensions')
        # generate opinion using the evidence
        op = sl.MultiOpinion(evidence_meas, self._X.probability_bins, 1, evidence=True)
        return op

    def update_opinion(self, z_pred: ArrayLike, S: ArrayLike, z: ArrayLike):
        """Generate updated short term opinion using measurement.

        Parameters
        ----------
        z_pred : array_like, float
            Predicted measurement as given by the Kalman Filter.
        S : array_like, float
            Measurement covariance matrix.
        z : array_like, float
            Measurement provided by the simulator or sensor.

        Returns
        -------
        list
            op_up : MultiOpinion
                updated short term opinion.
            op_z : MultiOpinion
                opinion generated from the measurement.

        """
        if z is None:
            z_norm = z
        else:
            # normalize the residuals
            z_norm = np.dot(sqrtm(np.linalg.inv(S)), (z-z_pred))
        op_z = self.generate_opinion_over_X(z_norm)

        # fuse opinion of the measurement into the short term opinion
        op_up = sl.fusion_acbf(self._op_st, op_z)
        return [op_up, op_z]

    def get_complete_opinion(self):
        """Output the current value for the complete opinion.

        Complete opinion constitutes of short and long term opinion fused together.

        Returns
        -------
        _op_X : MultiOpinion
            Opinion fused from st and lt opinion.

        """
        return deepcopy(self._op_X)

    def get_gauss_var(self):
        """Output the GaussVariable.

        Returns
        -------
        _X : GaussVariable1D or GaussVariableND
            One of the stated types to represent the reference distribution used
            in generating the opinion of the incoming measurement.

        """
        return deepcopy(self._X)

    def get_sas_measures(self):
        """Output the current values of the self-assessment values.

        Values used for self assessment are the DC to the reference,
        its uncertainty and the threshold

        Returns
        -------
        list
            _delta : float
                DC between complete opinion and reference.
            _u_delta : float
                Uncertainty of the complete opinion.
            _threshold_dc : float
                DC-threshold above which two opinions are considered different.

        """
        return deepcopy([self._delta, self._u_delta, self._threshold_dc])

    def assess(self, z_pred: ArrayLike, S_pred: ArrayLike, z: ArrayLike):
        """Calculate self assessment measure.

        Parameters
        ----------
        z_pred : array_like, float
            Predicted measurement as given by the Kalman Filter.
        S : array_like, float
            Measurement covariance matrix.
        z : array_like, float
            Measurement provided by the simulator or sensor.

        Returns
        -------
        list
            op_X : MultiOpinion
                the current opinion of the tracking.
            delta : float
                the degree of conflict between opinion about the tracking and
                the assumptions.
            u_delta : float
                uncertainty in the above measures.

        """
        # based on the mapping, only certain components are considered in the self-assessment

        if self.mapping is not None:
            assert 0 < len(self.mapping) <= len(z_pred), 'WARNING: Invalid mapping specified in init'
            assert np.amax(self.mapping) < len(z_pred), 'WARNING: Invalid mapping specified in init'
            assert np.amin(self.mapping) >= 0,  'WARNING: Invalid mapping specified in init'

            if z is not None:
                z = z[self.mapping]
            z_pred = z_pred[self.mapping]
            S_pred = S_pred[tuple([self.mapping])][:, self.mapping]


        # update short term opinion and generate opinion about measurement
        [self._op_st, self._op_z] = self.update_opinion(z_pred, S_pred, z)
        # append measurement to memory for later use
        self._op_z_memory.append(self._op_z)
        # if short term opinion is not generated over a long enough interval, use
        # incoming measurements to increase this interval and only update the short term
        # opinion
        if self._counter_st_op < self.n_st:
            # make deep copy to avoid errors
            self._op_X = deepcopy(self._op_st)
            # increase counter for the length of the interval
            self._counter_st_op += 1
        # if short term opinion is made over long enough interval
        else:
            # extract opinion about measurement n_st+1 steps ago
            op_st_to_lt = self._op_z_memory[0]
            # unfuse this opinion out of the short term opinion
            self._op_st = sl.unfusion_cu(self._op_st, op_st_to_lt)
            # fuse it into the long term opinion
            self._op_lt = sl.fusion_acbf(self._op_lt, op_st_to_lt)
            # delete it from the memory so that the memory only contains the opinions
            # about measurements which also make up the short term opinion
            del self._op_z_memory[0]
            # fuse long and short term to generate opinion about complete tracking
            self._op_X = sl.fusion_acbf(self._op_lt, self._op_st)
            # increase counters for length of long term opinion and window used for
            # comparison
            self._counter_lt_op += 1
            self._counter_compar += 1
            # if the length for window for comparison is reached and the long term
            # opinion has gathered at least the evidence of the short term opinion
            if (self._counter_compar >= self.n_c) and (self._counter_lt_op >= self.n_st):
                self._counter_compar = 0
                # if degree of conflict between long and short term opinion exceeds
                # the set threshold
                if sl.dc(self._op_st, self._op_lt) > self.threshold_ltst:
                    # reset long term opinion and the counter of its length
                    self._op_lt = deepcopy(self._op_0)
                    self._counter_lt_op = 0
                else:
                    # apply trust discount to the long term opinion
                    self._op_lt = self._op_lt.trust_discount(self.trust_discount)
        # calculate degree of conflict between tracking opinion and the assumptions
        # and save values to memory
        self._delta = sl.dc(self._op_X, self._op_G)
        self._u_delta = self._op_X.uncertainty

        # calculate threshold for DC comparison with reference
        self._threshold_dc = calc_tsh(self._op_X, self.alpha_threshold_dc)

    def assess_fusion(self, op_fused):
        self._delta = sl.dc(op_fused, self._op_G)
        self._u_delta = op_fused.uncertainty

        # calculate threshold for DC comparison with reference
        self._threshold_dc = calc_tsh(op_fused, self.alpha_threshold_dc)
