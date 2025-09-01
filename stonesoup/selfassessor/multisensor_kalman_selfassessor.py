from copy import deepcopy
from numpy.typing import ArrayLike
from typing import List
from stonesoup.subjective_logic import subjective_logic as sl
from stonesoup.selfassessor import kalman_selfassessor


class MultiSensorKalmanSelfAssessor:
    """Class for overall self-assessment in multi-sensor Kalman filtering
    using subjective logic with only CBF fusion."""

    def __init__(self, num_X: int, n_st: int, n_c: int = 1, prob_detect: float = 1, dim_meas: int = 2,
                 threshold_ltst: float = None, alpha_threshold_dc: float = 0.1, sigma_cutoff: float = None,
                 trust_discount: float = 0.99, type_compar: str = 'dimensional', mapping: List[int] = None):
        """Construct MultiSensorKalmanSelfAssessor (CBF-only version).

        Parameters
        ----------
        num_X : int
            Number of bins in which a distribution is divided.
        n_st : int
            Length of the region of the short term opinion.
        n_c : int, optional
            Length of window between comparisons between long and short term opinion.
        prob_detect : float, optional
            Probability of detection for each measurement. Default is 1.
        dim_meas : int, optional
            Number of dimensions of the measurement. Default is 2.
        threshold_ltst : float, optional
            Threshold up to which the degree of conflict between long and short term
            opinion can go until long term opinion is reset.
        alpha_threshold_dc : float, optional
            Alpha used for the confidence for the DC-threshold. Default is 0.1.
        sigma_cutoff : float, optional
            Distance after which only one bin is considered.
        trust_discount : float, optional
            Trust discount probability of long term opinion. Default is 0.99.
        type_compar : str, optional
            Type on how the Opinion regarding the measurement should be generated.
            'elementwise': each element in the measurement is compared to a one
                dimensional normal distribution.
            'dimensional': the whole vector is classified using the distance to the mean
            The default is 'dimensional'.
        mapping: List[int], optional
            List of the measurement vector entries that should contribute to SL-SAS,, others are ignored.
        """
        self.sl_multisensor_kalman_selfassessor_cbf = kalman_selfassessor.KalmanSelfAssessor(
            num_X=num_X, n_st=n_st, n_c=n_c, dim_meas=dim_meas,
            threshold_ltst=threshold_ltst, prob_detect=prob_detect,
            alpha_threshold_dc=alpha_threshold_dc, sigma_cutoff=sigma_cutoff,
            trust_discount=trust_discount, type_compar=type_compar, mapping=mapping
        )

        # Initialize attributes that will be filled in assess()
        self.op_X_fused_cbf = None
        self.delta_fused_cbf = None
        self.u_delta_fused_cbf = None
        self.theta_fused_cbf = None

    def assess(self, op_all: ArrayLike):
        """Calculate overall self-assessment measure for the multi-sensor case using only CBF fusion.

        Parameters
        ----------
        op_all : array_like of MultiOpinion
            Array of all considered opinions.

        Returns
        -------
        None
        """
        op_cbf = sl.fusion_acbf_expanded(op_all)
        self.op_X_fused_cbf = op_cbf
        self.sl_multisensor_kalman_selfassessor_cbf.get_sas_measures_fused_opinion(op_cbf)
        (self.delta_fused_cbf,
         self.u_delta_fused_cbf,
         self.theta_fused_cbf) = self.sl_multisensor_kalman_selfassessor_cbf.get_sas_measures()

    def get_sas_measures(self):
        """Return the current self-assessment values (CBF only).

        Returns
        -------
        list of float
            [delta, u_delta, theta] in that order.
        """
        return deepcopy([self.delta_fused_cbf, self.u_delta_fused_cbf, self.theta_fused_cbf])

    def get_complete_opinion(self):
        """Return the current complete fused opinion (CBF only)."""
        return deepcopy(self.op_X_fused_cbf)
