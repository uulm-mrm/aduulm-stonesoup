from copy import deepcopy
from math import floor
from typing import List
import math
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import poisson
from stonesoup.subjective_logic import subjective_logic as sl
from stonesoup.selfassessor._gauss_variable import GaussVariableND
from stonesoup.selfassessor._threshold import calc_threshold_op_diff as calc_tsh
from stonesoup.selfassessor._threshold import calc_threshold_n_diff as calc_tsh_n


class SOTClutterSelfAssessor:
    """Self-assessment class for single-object tracking (SOT) in clutter
    using subjective logic for NN or PDA tracking.

    The set of opinions depends on the `association_algorithm`:
        - 'pda': uses a full set of opinions including clutter, innovation, gate, detection, etc.
        - 'nn': uses a reduced set of opinions (subset of PDA).

    Attributes
    ----------
    n_st : list[int]
        Length of each region of the short-term opinion.
    n_c : int
        Length of the window between comparisons of long- and short-term opinions.
    prob_detect : float
        Probability of detection for each measurement.
        Rounded to one decimal for PDA; exact value for NN.
    dim_meas : int
        Number of measurement dimensions (length of mapping if provided).
    threshold_ltst : dict[str, float]
        Threshold values for conflict between long- and short-term opinions before
        resetting the long-term opinion. Defaults to 0.2 for each opinion component.
    alpha_threshold_dc : dict[str, float]
        Alpha values used to compute DC thresholds for each opinion. Automatically
        determined by number of bins: 2 → 0.3, 3–5 → 0.2, >5 → 0.1.
    trust_discount : float
        Trust discount probability applied to long-term opinions.
    exp_clutter : float
        Expected number of clutter measurements in the field of view.
    fov_area : float
        Field of view size in which measurements lie.
    mapping : list[int], optional
        Indices of measurement vector entries to consider for opinion calculation; others are ignored.
    prob_gate : float
        Probability that a detection lies inside the gate.
    association_algorithm : str
        Association algorithm ('nn' or 'pda'). Determines which set of opinions is used.

    Internal Attributes
    -------------------
    _X : GaussVariableND
        Reference Gaussian variable used to generate measurement opinions.
    _op_0 : dict[str, Union[MultiOpinion, BiOpinion]]
        Initial vague opinions for each component.
    _op_G : dict[str, Union[MultiOpinion, BiOpinion]]
        Reference dogmatic opinions.
    _op_st : dict[str, Union[MultiOpinion, BiOpinion]]
        Short-term opinions.
    _op_lt : dict[str, Union[MultiOpinion, BiOpinion]]
        Long-term opinions.
    _op_X : dict[str, Union[MultiOpinion, BiOpinion]]
        Overall fused opinions.
    _op_z_memory : dict[str, list[Union[MultiOpinion, BiOpinion]]]
        Memory of the last n_st opinions for each component.
    _counter_lt_op : np.ndarray
        Evidence counters for long-term opinions.
    _counter_st_op : np.ndarray
        Evidence counters for short-term opinions.
    _counter_compar : np.ndarray
        Step counters since last comparison.
    _threshold_dc : np.ndarray
        DC thresholds for each opinion component.
    _delta : np.ndarray
        DC values between fused and reference opinions.
    _u_delta : np.ndarray
        Uncertainty of fused opinions.
    _clutter_bin_borders : np.ndarray
        Bin borders for clutter-related opinions.
    _association_situation_borders : np.ndarray
        Bin borders for entropy-related opinions.
    association_situation_proj_prob : optional
        Projected probability for the association situation.
    association_situation_value : optional
        Value for the association situation.
    threshold_proj_prob : optional
        Projected threshold probability.
    """

    def __init__(self, num_X, n_st, n_c=1, prob_detect=1.0, dim_meas=2, threshold_ltst=None,
                 alpha_threshold_dc=None, sigma_cutoff=4, trust_discount=0.99, exp_clutter=4,
                 fov_area=400, mapping: List[int] = None, prob_gate=1.0, association_algorithm='nn'):
        """Initialize a self-assessor (NN or PDA) based on association_algorithm.

        Parameters
        ----------
        num_X : int
            Number of bins used for opinion distributions.
        n_st : dict[str, int]
            Length of each region of the short-term opinion.
        n_c : int, optional
            Length of the window between comparisons of long- and short-term opinions.
            Default is 1.
        prob_detect : float, optional
            Probability of detection for each measurement. Default is 1.0.
        dim_meas : int, optional
            Number of measurement dimensions. Default is 2.
        threshold_ltst : dict[str, float], optional
            Threshold values for conflict between long- and short-term opinions before
            resetting the long-term opinion. If None, defaults to 0.2 for all components.
        alpha_threshold_dc : list[float], optional
            Alpha values used to calculate DC thresholds. Default is a predefined list.
        sigma_cutoff : float, optional
            Cutoff distance beyond which only one bin is considered. Default is 4.
        trust_discount : float, optional
            Trust discount probability applied to long-term opinions. Default is 0.99.
        exp_clutter : float, optional
            Expected number of clutter measurements in the FOV. Default is 4.
        fov_area : float, optional
            Field of view size in which measurements lie. Default is 400.
        mapping : list[int], optional
            Indices of measurement vector entries to consider for NIS; others are ignored.
        prob_gate : float, optional
            Probability that a detection lies inside the gate. Default is 1.0.
        association_algorithm : str, optional
            Association algorithm to use (e.g., 'nn', 'pda'). Default is 'nn'.
        """

        # ---------- Core parameters ----------
        self.association_algorithm = association_algorithm.lower()
        self.n_st = n_st
        self.n_c = n_c
        self.sigma_cutoff = sigma_cutoff
        self.trust_discount = trust_discount
        self.exp_clutter = exp_clutter
        self.fov_area = fov_area
        self.prob_gate = prob_gate

        # Detection probability logic depending on algorithm
        if self.association_algorithm == 'pda':
            self.prob_detect = round(prob_detect, 1)
        elif self.association_algorithm == 'nn':
            self.prob_detect = prob_detect
        else:
            raise ValueError(f"Unknown association_algorithm '{association_algorithm}'")

        # Measurement dimension
        self.dim_meas = len(mapping) if mapping is not None else dim_meas
        self.mapping = mapping

        # ---------- Determine opinion specification ----------
        if self.association_algorithm == "pda":
            self.opinion_spec = {
                "innovation": num_X, "detection": 2, "gate": num_X - 1,
                "clutter": 3,
                "combined_innovation": num_X, "association_situation": 2,
                "combined": 2, "threshold": 2,
                "binomial_clutter": 2, "binomial_combined_innovation": 2
            }
        elif self.association_algorithm == "nn":
            self.opinion_spec = {
                "innovation": num_X, "detection": 2, "gate": num_X - 1,
                "clutter": 3,
                "combined": 2, "threshold": 2,
                "binomial_clutter": 2, "binomial_combined_innovation": 2
            }

        # ---------- Alpha thresholds ----------
        if alpha_threshold_dc is None:
            self.alpha_threshold_dc = {}
            for key, val in self.opinion_spec.items():
                if val == 2:
                    self.alpha_threshold_dc[key] = 0.3
                elif 3 <= val <= 5:
                    self.alpha_threshold_dc[key] = 0.2
                else:  # val > 5
                    self.alpha_threshold_dc[key] = 0.1
        else:
            # If provided as dict, use it
            self.alpha_threshold_dc = alpha_threshold_dc

        # ---------- Threshold for long-term vs short-term ----------
        if threshold_ltst is None:
            # num_X_all = {key: val for key, val in self.opinion_spec.items()}
            # self.threshold_ltst = calc_tsh_n(num_X_all, self.n_st, self.alpha_threshold_dc)
            self.threshold_ltst = {key: 0.2 for key in self.opinion_spec.keys()}
        else:
            self.threshold_ltst = threshold_ltst

        # ---------- Initialize Gaussian variable ----------
        self._X = GaussVariableND(self.dim_meas, num_X, self.sigma_cutoff)

        # ---------- Generate distributions ----------
        dist_innovation = self.generate_innovation_distribution(exp_clutter, fov_area)
        dist_gate = self.generate_gate_distribution(exp_clutter, fov_area)
        dist_detection = [sum(dist_innovation[:-1]), dist_innovation[-1]]
        [self._clutter_bin_borders, dist_clutter] = self.generate_clutter_distribution(exp_clutter)

        # ---------- Initialize opinions ----------
        if self.association_algorithm == "pda":
            [self._association_situation_borders, dist_association_situation] = \
                self.generate_association_situation_distribution()
            # evidence-based vague opinions
            self._op_0 = {
                "innovation": sl.MultiOpinion(np.zeros(num_X), dist_innovation, 1),
                "detection": sl.BiOpinion(0, 0, dist_detection[0], 1),
                "gate": sl.MultiOpinion(np.zeros(num_X - 1), dist_gate, 1),
                "clutter": sl.MultiOpinion(np.zeros(3), dist_clutter, 1),
                "combined_innovation": sl.MultiOpinion(np.zeros(num_X), dist_innovation, 1),
                "association_situation": sl.BiOpinion(0, 0, 0.8, 1),
                "combined": sl.BiOpinion(0, 0, 0.5, 1),
                "threshold": sl.BiOpinion(0, 0, 0.9, 1),
                "binomial_clutter": sl.BiOpinion(0, 0, dist_clutter[1], 1),
                "binomial_combined_innovation": sl.BiOpinion(0, 0, dist_detection[0], 1),
            }
            # reference dogmatic opinions
            self._op_G = {
                "innovation": sl.MultiOpinion(dist_innovation, dist_innovation, 0),
                "detection": sl.BiOpinion(dist_detection[0], dist_detection[1], dist_detection[0], 0),
                "gate": sl.MultiOpinion(dist_gate, dist_gate, 0),
                "clutter": sl.MultiOpinion(dist_clutter, dist_clutter, 0),
                "combined_innovation": sl.MultiOpinion(dist_innovation, dist_innovation, 0),
                "association_situation": sl.BiOpinion(0.8, 0.2, 0.8, 0),
                "combined": sl.BiOpinion(0.5, 0.5, 0.5, 0),
                "threshold": sl.BiOpinion(0.9, 0.1, 0.9, 0),
                "binomial_clutter": sl.BiOpinion(dist_clutter[1], dist_clutter[0] + dist_clutter[2],
                                                 dist_clutter[1], 0),
                "binomial_combined_innovation": sl.BiOpinion(dist_detection[0], dist_detection[1], dist_detection[0], 0)
            }
        elif self.association_algorithm == "nn":
            # evidence-based vague opinions
            self._op_0 = {
                "innovation": sl.MultiOpinion(np.zeros(num_X), dist_innovation, 1),
                "detection": sl.BiOpinion(0, 0, dist_detection[0], 1),
                "gate": sl.MultiOpinion(np.zeros(num_X - 1), dist_gate, 1),
                "clutter": sl.MultiOpinion(np.zeros(3), dist_clutter, 1),
                "combined": sl.BiOpinion(0, 0, 0.5, 1),
                "threshold": sl.BiOpinion(0, 0, 0.9, 1),
                "binomial_clutter": sl.BiOpinion(0, 0, dist_clutter[1], 1),
                "binomial_combined_innovation": sl.BiOpinion(0, 0, dist_detection[0], 1)
            }
            # reference dogmatic opinions
            self._op_G = {
                "innovation": sl.MultiOpinion(dist_innovation, dist_innovation, 0),
                "detection": sl.BiOpinion(dist_detection[0], dist_detection[1], dist_detection[0], 0),
                "gate": sl.MultiOpinion(dist_gate, dist_gate, 0),
                "clutter": sl.MultiOpinion(dist_clutter, dist_clutter, 0),
                "combined": sl.BiOpinion(0.5, 0.5, 0.5, 0),
                "threshold": sl.BiOpinion(0.9, 0.1, 0.9, 0),
                "binomial_clutter": sl.BiOpinion(dist_clutter[1], dist_clutter[0] + dist_clutter[2],
                                                 dist_clutter[1], 0),
                "binomial_combined_innovation": sl.BiOpinion(dist_detection[0], dist_detection[1], dist_detection[0], 0)
            }

        # ---------- Generate distributions continue (combined opinion) ----------
        self.generate_combined_opinion()

        # ---------- Initial short-term and long-term opinions ----------
        self._op_st = deepcopy(self._op_0)
        self._op_lt = deepcopy(self._op_0)
        self._op_z_memory = {key: [] for key in self._op_G.keys()}

        # ---------- Counters ----------
        self._counter_lt_op = np.zeros(len(self._op_0))
        self._counter_st_op = np.zeros(len(self._op_0))
        self._counter_compar = np.zeros(len(self._op_0))

        # ---------- Storage ----------
        self._threshold_dc = np.zeros(len(self._op_0))
        self._delta = np.zeros(len(self._op_0))
        self._u_delta = np.ones(len(self._op_0))
        self._op_X = deepcopy(self._op_0)

        self.association_situation_proj_prob = None
        self.association_situation_value = None
        self.threshold_proj_prob = None

    # ---------------------------------------------------- #
    # ---------- Generate distributions methods ---------- #
    # ---------------------------------------------------- #

    def generate_innovation_distribution(self, exp_clutter, fov_area):
        """Generate the innovation distribution for data association.

        Computes the distribution of evidence across bins, considering:
        - Detection probability
        - Expected clutter
        - Field of view
        - Missed detections

        Parameters
        ----------
        exp_clutter : float
            Expected number of clutter measurements in the FOV.
        fov_area : float
            Field of view in which the measurements lie.

        Returns
        -------
        dist : np.ndarray
            Probability for each bin.

        """
        num_bins = self._X.number_bins
        intervals = np.zeros(num_bins)
        intervals[0] = self._X.borders_bins[0]
        intervals[1:-1] = self._X.borders_bins[1:] - self._X.borders_bins[:-1]
        intervals[-1] = np.sqrt(fov_area) / 2 - self._X.borders_bins[-1]

        if exp_clutter < 0:
            raise ValueError("Expected clutter must be >= 0")

        # Handle edge case with zero clutter
        if exp_clutter == 0:
            c0 = 1 - self.prob_detect
            c1 = self.prob_detect
        else:
            # continuous extension of factorial with Gamma function (n! = Gamma(n+1))
            c0 = (1 - self.prob_detect) * (
                        math.exp(-exp_clutter) / math.gamma((exp_clutter + self.prob_detect) + 1)) * (
                             exp_clutter / fov_area) ** (exp_clutter + self.prob_detect)
            # continuous extension of factorial with Gamma function (n! = Gamma(n+1))
            c1 = self.prob_detect * (
                        math.exp(-exp_clutter) / math.gamma((exp_clutter + self.prob_detect) + 1)) * math.pow(
                exp_clutter / fov_area, exp_clutter + self.prob_detect - 1)

        # Compute distribution per intervals
        dist_c0 = c0 * intervals
        dist_c1 = c1 * intervals
        dist = dist_c0 + dist_c1 * (exp_clutter + self.prob_detect) * self._X.probability_bins

        # normalize, the resulting distribution sums up to 1 in theoretical view, but to
        # ensure exact additivity and avoid the build-up of numerical artifacts it is
        # normalized nevertheless
        dist /= sum(dist)
        return dist

    def generate_gate_distribution(self, exp_clutter, fov_area):
        """Generate the distribution of measurements after gating.

        This distribution represents the probabilities of measurements in each bin
        that can be associated with a target (i.e., excluding missed detections and
        the last bin representing out-of-gate measurements).

        Parameters
        ----------
        exp_clutter : float
            Expected number of clutter measurements in the FOV.
        fov_area : float
            Field of view in which the measurements lie.

        Returns
        -------
        dist : np.ndarray
            Probability distribution for measurements inside the gate.
        """
        # Generate complete innovation distribution
        dist_innovation = self.generate_innovation_distribution(exp_clutter, fov_area)

        # Exclude the last bin (out-of-gate measurements)
        dist_in_gate = dist_innovation[:-1]

        if dist_in_gate.sum() == 0:
            # Avoid division by zero
            return dist_in_gate

        # Normalize to get a proper probability distribution
        dist_in_gate /= dist_in_gate.sum()

        return dist_in_gate

    def generate_clutter_distribution(self, exp_clutter):
        """Generate clutter distribution in 3 intervals: too few, about average, too many.

        The distribution is based on a Poisson model of the expected number of clutter
        measurements (exp_clutter). The bin borders are set to capture roughly low,
        medium, and high clutter probabilities.

        Parameters
        ----------
        exp_clutter : float
            Expected number of clutter measurements in the FOV.

        Returns
        -------
        tuple
            borders : np.ndarray
                Borders for the 3 bins (low, medium, high).
            dist_clutter : np.ndarray
                Probability for the number of clutter measurements in each bin.
        """
        # Determine bin borders using Poisson percent point function
        # Roughly captures the lower and upper extremes (2.5% and 97.5%)
        borders = poisson.ppf([0.025, 0.975], exp_clutter)
        # alternative
        # if exp_clutter <= 0.8:
        #     borders = [1, exp_clutter + 1]
        # elif exp_clutter <= 2:
        #     borders = [0, exp_clutter + 1]
        # elif exp_clutter <= 3:
        #     borders = [0, exp_clutter + 2]
        # elif exp_clutter <= 8:
        #     borders = [exp_clutter - 4, exp_clutter + 3]
        # elif exp_clutter <= 9:
        #     borders = [exp_clutter - 5, exp_clutter + 4]
        # else:
        #     borders = [exp_clutter - 6, exp_clutter + 5]
        # borders = np.round(borders)

        # Initialize distribution for 3 bins
        dist_clutter = np.zeros(3)

        # Probability of too few clutter measurements
        dist_clutter[0] = poisson.cdf(borders[0], exp_clutter)

        # Probability of average clutter measurements (between the borders)
        dist_clutter[1] = poisson.cdf(borders[1], exp_clutter) - dist_clutter[0]

        # Probability of too many clutter measurements (above upper border)
        dist_clutter[2] = poisson.sf(borders[1], exp_clutter)

        return borders, dist_clutter

    def generate_association_situation_distribution(self):
        """
        Generate a heuristic distribution for the association situation based on entropy.

        This is currently based on experimental simulations and is primarily heuristic.
        The distribution is defined over two bins:
            1. Low entropy (high certainty)
            2. High entropy (uncertain / everything else)

        Returns
        -------
        tuple
            borders : np.ndarray
                Bin borders. First bin corresponds to near-certainty, second bin contains all
                remaining values up to 1. Currently, set heuristically.
            dist : np.ndarray
                Probability associated with each bin, based on experimental simulation
                (expected clutter ~4, averaged over 200 Monte Carlo runs). Currently, static
                and needs adjustment for other clutter values.
        """
        # ToDo: make this function more meaningful, so far it is only heuristic
        # Heuristic bin borders: [low certainty threshold, max value]
        borders = np.array([0.2, 1.0])

        # Heuristic probabilities for the two bins
        dist = np.array([0.8, 0.2])

        return borders, dist

    def generate_combined_opinion(self):
        """
        Update the combined short-term opinion by fusing relevant opinions
        into a binomial opinion representation.

        The fused opinion is stored in:
            - self._op_G["combined"] : reference (dogmatic) opinion
            - self._op_0["combined"] : initial vague opinion
        """
        # Select dogmatic opinions for fusion
        dogmatic_detection = self._op_G["detection"]
        dogmatic_clutter = self._op_G["binomial_clutter"]

        if self.association_algorithm == "pda":
            dogmatic_combined_innovation = self._op_G["binomial_combined_innovation"]
            fused_opinion = sl.fusion_acbf_expanded(
                [dogmatic_detection, dogmatic_clutter, dogmatic_combined_innovation]
            )
        else:  # NN
            fused_opinion = sl.fusion_acbf_expanded(
                [dogmatic_detection, dogmatic_clutter]
            )

        # Update combined dogmatic and initial vague opinions
        self._op_G["combined"] = sl.BiOpinion(
            fused_opinion.baseRate[0],
            fused_opinion.baseRate[1],
            fused_opinion.baseRate[0],
            0
        )

        self._op_0["combined"] = sl.BiOpinion(
            0,
            0,
            fused_opinion.baseRate[0],
            1
        )

    # ---------------------------------------------------- #
    # ----------------- Assess SA Method ----------------- #
    # ---------------------------------------------------- #

    def assess(self, z_pred, S_pred, z_array, num_measurements,
               weight_array=1.0, combinedInnovation=None):
        """Calculate self assessment measures for all opinions.

        Parameters
        ----------
        z_pred : array_like, float
            Predicted measurement as given by the Kalman filter.
        S_pred : array_like, float
            Measurement covariance matrix.
        z_array : array_like, float
            Either measurement provided by the simulator or sensor,
            or array containing all measurements as well as the corresponding hypothesis.
        num_measurements : int
            Number of measurements in the mahalanobis_distance_cutoff distance.
        weight_array: array_like, float
            Array containing all hypothesis.probabilities. If len != num_meas error is thrown.
            Entropy can only be calculated once for each timestep not for each measurement.
        combinedInnovation: array_like, float
            Array containing the combinedInnovation of the PDA.
        Returns
        -------
        list
            op_X : array_like, MultiOpinion
                Current opinion of the tracking.
            delta : array_like, float
                Degree of conflict between opinion about the tracking and
                the assumptions.
            u_delta : array_like, float
                Uncertainty in the above measures.

        """

        # ---------- Input preparation ----------
        if isinstance(weight_array, float):
            iterator_length = 1
        else:
            iterator_length = len(weight_array)

        for i in range(iterator_length):
            # --- Get measurement & weight ---
            if self.association_algorithm == "pda":
                z = z_array[i] if z_array is not None else None
                weight = weight_array[i] if z_array is not None else None
            else:
                z, weight = z_array, weight_array

            # --- Apply mapping if given ---
            if self.mapping is not None:
                assert len(self.mapping) <= len(z_pred), "Invalid mapping specified in init"
                assert any(self.mapping) < len(z_pred), "Invalid mapping specified in init"

                if z is not None:
                    z = z[self.mapping]
                z_pred = z_pred[self.mapping]
                S_pred = S_pred[tuple([self.mapping])][:, self.mapping]

            # ---------- Opinion initialization ----------
            op_z = deepcopy(self._op_0)

            # ---------- Category: Measurements ----------
            self._op_st["innovation"], op_z["innovation"] = self.update_innovation_opinion(
                z_pred, S_pred, z, weight=weight
            )
            self._op_st["detection"], op_z["detection"] = self.update_detection_opinion(
                z, opinion_innovation=op_z["innovation"], weight=weight
            )
            self._op_st["gate"], op_z["gate"] = self.update_gate_opinion(
                z_pred, S_pred, z, weight=weight
            )

            # ---------- Category: Clutter ----------
            self._op_st["clutter"], op_z["clutter"] = self.update_clutter_opinion(
                z, num_measurements, weight=weight
            )
            self._op_st["binomial_clutter"], op_z["binomial_clutter"] = (
                self.update_binomial_clutter_opinion(deepcopy(op_z["clutter"]))
            )

            # ---------- Sliding window update for base opinions ----------
            for index, value in enumerate(self._op_0):
                if (value == 'combined_innovation' or value == "association_situation" or
                        value == "combined" or value == "threshold" or
                        value == 'binomial_combined_innovation'):
                    continue
                else:
                    # append measurement to memory for later use
                    self._op_z_memory[value].append(op_z[value])
                    if self._counter_st_op[index] < self.n_st[value]:
                        self._op_X[value] = deepcopy(self._op_st[value])
                        # increase counter for the length of the interval
                        self._counter_st_op[index] += float(weight)
                    # if short-term opinion is made over long enough interval
                    else:
                        # extract opinion about measurement n_st+1 steps ago
                        op_st_to_lt = self._op_z_memory[value][0]
                        # unfuse this opinion out of the short-term opinion
                        self._op_st[value] = sl.unfusion_cu(self._op_st[value], op_st_to_lt)
                        # fuse it into the long term opinion
                        self._op_lt[value] = sl.fusion_acbf(self._op_lt[value], op_st_to_lt)
                        # delete it from the memory so that the memory only contains the opinions
                        # about measurements which also make up the short term opinion
                        del self._op_z_memory[value][0]
                        # fuse long and short-term to generate the resulting SA opinion
                        self._op_X[value] = sl.fusion_acbf(self._op_lt[value], self._op_st[value])
                        # increase counters for length of long term opinion
                        # and window used for comparison
                        self._counter_lt_op[index] += float(weight)
                        self._counter_compar[index] += float(weight)
                        # if the length for window for comparison is reached and the long term
                        # opinion has gathered at least the evidence of the short term opinion
                        if (self._counter_compar[index] >= self.n_c[value]) and \
                                (self._counter_lt_op[index] >= self.n_st[value]):
                            self._counter_compar[index] = 0
                            # if degree of conflict between long and short term opinion
                            # exceeds the calculated threshold
                            if sl.dc(self._op_st[value], self._op_lt[value]) > self.threshold_ltst[value]:
                                # reset long term opinion and the counter of its length
                                self._op_lt[value] = deepcopy(self._op_0[value])
                                self._counter_lt_op[index] = 0
                            else:
                                # apply trust discount to the long term opinion
                                self._op_lt[value] = self._op_lt[value].trust_discount(self.trust_discount[value])

        # ---------- Special handling for PDA ----------
        if self.association_algorithm == 'pda':
            if combinedInnovation is not None and self.mapping is not None:
                combinedInnovation = combinedInnovation[self.mapping]

            [self._op_st["combined_innovation"], op_z["combined_innovation"]] = \
                self.update_combinedInnovation_opinion(S_pred, combinedInnovation,
                                                       weight_measurements=weight_array)
            [self._op_st["binomial_combined_innovation"], op_z["binomial_combined_innovation"]] = \
                self.update_binomial_combinedInnovation_opinion(deepcopy(op_z["combined_innovation"]))
            [self._op_st["association_situation"], op_z["association_situation"]] = \
                self.update_association_situation_opinion(weights=weight_array)

            op_z_comb = deepcopy(self._op_0)
            op_z_comb["combined_innovation"] = deepcopy(op_z["combined_innovation"])
            op_z_comb["association_situation"] = deepcopy(op_z["association_situation"])
            op_z_comb["clutter"] = sl.fusion_acbf_expanded(self._op_z_memory["clutter"][-iterator_length:])
            op_z_comb["detection"] = sl.fusion_acbf_expanded(self._op_z_memory["detection"][-iterator_length:])

            self._op_st["combined"], op_z["combined"], evidence_combined = (
                self.update_combined_opinion(op_z_comb)
            )
        else:
            self._op_st["combined"], op_z["combined"], evidence_combined = (
                self.update_combined_opinion(op_z)
            )

        # ---------- Sliding window update for combined opinions ----------
        for value in {'combined_innovation', 'combined', 'binomial_combined_innovation'}:
            if value not in self._op_z_memory or value not in self._op_0:
                continue  # skip if not relevant in current mode (e.g. when NN is used)

            self._op_z_memory[value].append(op_z[value])

            idx = list(self._op_0).index(value)
            if self._counter_st_op[idx] < self.n_st[value]:
                self._op_X[value] = deepcopy(self._op_st[value])
                if value == 'combined':
                    self._counter_st_op[idx] += evidence_combined
                else:
                    self._counter_st_op[idx] += 1
            else:
                op_st_to_lt = self._op_z_memory[value][0]
                # unfuse this opinion out of the short-term opinion
                self._op_st[value] = sl.unfusion_cu(self._op_st[value], op_st_to_lt)
                # fuse it into the long term opinion
                self._op_lt[value] = sl.fusion_acbf(self._op_lt[value], op_st_to_lt)
                # delete it from the memory so that the memory only contains the opinions
                # about measurements which also make up the short-term opinion
                del self._op_z_memory[value][0]
                # fuse long and short-term to generate the resulting SA opinion
                self._op_X[value] = sl.fusion_acbf(self._op_lt[value], self._op_st[value])
                # increase counters for length of long-term opinion
                # and window used for comparison
                self._counter_lt_op[idx] += 1
                self._counter_compar[idx] += 1
                # if the length for window for comparison is reached and the long-term
                # opinion has gathered at least the evidence of the short-term opinion
                if (self._counter_compar[idx] >= self.n_c[value]) and \
                        (self._counter_lt_op[idx] >= self.n_st[value]):
                    self._counter_compar[idx] = 0
                    # if degree of conflict between long and short-term opinion
                    # exceeds the calculated threshold
                    if sl.dc(self._op_st[value], self._op_lt[value]) > self.threshold_ltst[value]:
                        # reset long-term opinion and the counter of its length
                        self._op_lt[value] = deepcopy(self._op_0[value])
                        self._counter_lt_op[idx] = 0
                    else:
                        # apply trust discount to the long-term opinion
                        self._op_lt[value] = self._op_lt[value].trust_discount(self.trust_discount[value])

        # combined opinion: version 2 - just fusion of single resulting SA opinions
        # self._op_X["combined"] = self.obtained_combined_opinion(self._op_X)

        # ---------- Global comparison ----------
        for index, value in enumerate(self._op_X):
            # calculate degree of conflict between resulting SA opinion
            # and the assumptions and save values to memory
            self._delta[index] = sl.dc(self._op_X[value], self._op_G[value])
            self._u_delta[index] = self._op_X[value].uncertainty
        # calculate threshold for DC comparison with reference
        self._threshold_dc = calc_tsh(self._op_X, list(self.alpha_threshold_dc.values()))

        # ---------- Threshold + association situation ----------
        [self._op_st["threshold"], op_z["threshold"]] = self.update_overall_threshold_opinion()
        update_keys = {"threshold", "association_situation"}
        for key in update_keys:
            if key not in self._op_z_memory or key not in self._op_0:
                continue  # skip keys not available in current mode
            self._op_z_memory[key].append(op_z[key])
            idx = list(self._op_0).index(key)
            if self._counter_st_op[idx] < self.n_st[key]:
                self._op_X[key] = deepcopy(self._op_st[key])
                self._counter_st_op[idx] += 1
            else:
                op_st_to_lt = self._op_z_memory[key][0]
                self._op_st[key] = sl.unfusion_cu(self._op_st[key], op_st_to_lt)
                del self._op_z_memory[key][0]
                self._op_X[key] = deepcopy(self._op_st[key])

        if "threshold" in self._op_X:
            self.threshold_proj_prob = self._op_X["threshold"].get_projected_prob()
        if "association_situation" in self._op_X:
            self.association_situation_proj_prob = self._op_X["association_situation"].get_projected_prob()

    # ---------------------------------------------------- #
    # -------------- Update opinions methods ------------- #
    # ---------------------------------------------------- #

    def update_innovation_opinion(self, z_pred, S_pred, z, weight=1.0):
        """Update the short-term innovation opinion.

        Parameters
        ----------
        z_pred : array_like of float
            Predicted measurement from the Kalman filter.
        S_pred : array_like of float
            Measurement prediction covariance matrix.
        z : array_like of float or None
            Set of measurements from the simulator or sensor. If None, no update is performed.
        weight : float, optional
            Weight of the measurement, typically provided by the PDA algorithm (default: 1.0).

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term innovation opinion.
            op_z : MultiOpinion
                Measurement opinion generated from the residual.
        """
        if z is None:
            z_norm = None
        else:
            # Normalize the innovation (residual)
            z_norm = np.dot(sqrtm(np.linalg.inv(S_pred)), (z - z_pred))

        # Generate innovation opinion from the normalized residual
        op_z = self.generate_innovation_opinion_over_X(z_norm, weight=weight)

        # Fuse with existing short-term opinion
        op_up = sl.fusion_acbf(self._op_st["innovation"], op_z)

        return [op_up, op_z]

    def generate_innovation_opinion_over_X(self, z, weight=1.0):
        """Generate an innovation opinion over the reference distribution.

        This constructs the opinion based on the normalized measurement residual.
        If no measurement is provided, it assumes a missed detection.

        Parameters
        ----------
        z : array_like of float or None
            Normalized measurement residual. If None, a missed detection is assumed.
        weight : float, optional
            Weight of the measurement, typically from the PDA algorithm (default: 1.0).

        Returns
        -------
        op : MultiOpinion
            Opinion corresponding to the measurement residual.
        """
        evidence_meas = np.zeros(self._X.number_bins)

        if z is None:
            # Case: missed detection
            # Assumption: the object lies outside the gating region.
            evidence_meas[-1] = weight
        else:
            # Case: measurement available
            # Evidence assigned based on the distance of the residual from zero.
            # (Due to normalization, the mean is always at 0.)
            distance = np.linalg.norm(z)
            evidence_meas = self._X.check_value_for_bin(distance, weight=weight)

            # Debug option (kept as comment for future reference):
            # if evidence_meas[-1] != 0:
            #     print("Residual outside last border (not a missed detection).")

        # Construct opinion from evidence
        op = sl.MultiOpinion(
            evidence_meas,
            self._op_0["innovation"].baseRate,
            1,
            evidence=True
        )

        return op

    def update_detection_opinion(self, z, opinion_innovation=None, weight=1.0):
        """Update the short-term detection opinion.

        Detection is modeled as a binary opinion: either the object is detected
        or it is not. A missed detection is assumed if no measurement is given
        or if the innovation opinion indicates the residual is outside the last bin.

        Parameters
        ----------
        z : array_like of float or None
            Measurement provided by the simulator or sensor. If None, a missed
            detection is assumed.
        opinion_innovation : MultiOpinion, optional
            Opinion on the innovation (measurement residual). Used to determine
            missed detections when residuals fall outside the gating region.
        weight : float, optional
            Weight of the measurement, typically from the PDA algorithm
            (default: 1.0).

        Returns
        -------
        list
            op_up : BiOpinion
                Updated short-term detection opinion after fusing with new evidence.
            op_z : BiOpinion
                Opinion generated from the current measurement.
        """
        # Initialize evidence for two bins: [detected, missed]
        evidence = np.zeros(2)

        # Missed detection: either no measurement or innovation outside last border
        if z is None or (opinion_innovation is not None and opinion_innovation.belief[-1] != 0):
            evidence[1] = weight
        else:
            evidence[0] = weight

        # Create binomial opinion for this measurement
        op_z = sl.BiOpinion(
            evidence[0],
            evidence[1],
            self._op_0["detection"].baseRate[0],
            1,
            evidence=True
        )

        # Fuse with running short-term detection opinion
        op_up = sl.fusion_acbf(self._op_st["detection"], op_z)

        return [op_up, op_z]

    def update_gate_opinion(self, z_pred, S_pred, z, weight=1.0):
        """Update the short-term gate opinion.

        Parameters
        ----------
        z_pred : array_like of float
            Predicted measurement from the Kalman Filter.
        S_pred : array_like of float
            Measurement prediction covariance matrix.
        z : array_like of float or None
            Actual measurement from the simulator or sensor. If None, no
            measurement is considered inside the gate.
        weight : float, optional
            Weight of the measurement, typically from the PDA algorithm
            (default: 1.0).

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term gate opinion after fusing with the new evidence.
            op_z : MultiOpinion
                Opinion generated from the current measurement.
        """
        if z is None:
            # No measurement inside the gate → use vacuous opinion
            op_z = deepcopy(self._op_0["gate"])
        else:
            # Normalize residuals: (z - z_pred) transformed by inverse sqrt of covariance
            z_norm = np.dot(sqrtm(np.linalg.inv(S_pred)), (z - z_pred))

            # Evidence: assign weight based on how far the measurement is from the mean (0 after normalization)
            evidence_meas = self._X.check_value_for_bin(np.linalg.norm(z_norm), weight=weight)

            # Create opinion excluding the last bin (outside region)
            op_z = sl.MultiOpinion(
                evidence_meas[:-1],
                self._op_0["gate"].baseRate,
                1,
                evidence=True
            )

        # Fuse measurement opinion into short-term running opinion
        op_up = sl.fusion_acbf(self._op_st["gate"], op_z)

        return [op_up, op_z]

    def update_clutter_opinion(self, z, num_measurements, weight=1.0):
        """Update the short-term clutter opinion.

        Parameters
        ----------
        z : array_like of float or None
            Measurement provided by the simulator or sensor. If None, all
            measurements are assumed to be clutter.
        num_measurements : int
            Number of measurements in the FOV.
        weight : float, optional
            Weight of the measurement (default: 1.0). For PDA this is the
            association weight, for NN this is always 1.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term clutter opinion after fusion.
            op_z : MultiOpinion
                Opinion generated from the current clutter evidence.
        """
        # Estimate number of clutter measurements
        if z is None:
            # No detection assigned, that means all measurements treated as clutter
            num_clutter = num_measurements
        else:
            # Subtract expected true detections, then floor to ensure integer bins
            num_clutter = floor(num_measurements - self.prob_detect)

        # Initialize evidence for 3 bins: [too few, about average, too many]
        evidence = np.zeros(3)

        # Assign evidence to the corresponding bin
        if num_clutter <= self._clutter_bin_borders[0]:
            evidence[0] = weight
        elif num_clutter <= self._clutter_bin_borders[1]:
            evidence[1] = weight
        else:
            evidence[2] = weight

        # Create the opinion for this measurement
        op_z = sl.MultiOpinion(evidence, self._op_0["clutter"].baseRate, 1, evidence=True)

        # Fuse measurement opinion into the short-term running opinion
        op_up = sl.fusion_acbf(self._op_st["clutter"], op_z)

        return [op_up, op_z]

    def update_binomial_clutter_opinion(self, op_z_clutter_original):
        """Update the short-term binomial clutter opinion.

        Converts the 3-bin clutter opinion into a binomial form
        and fuses it into the short-term binomial clutter opinion.

        Parameters
        ----------
        op_z_clutter_original : MultiOpinion
            Original clutter opinion with 3 bins: [too few, about average, too many].

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term binomial clutter opinion after fusion.
            op_z : MultiOpinion
                Binomial clutter opinion generated from the current measurement.
        """
        # Collapse the 3-bin clutter opinion into a binomial form
        if self.exp_clutter <= 0.8:
            # Low expected clutter -> treat "too few" as the positive evidence
            op_z_bi_clutter = sl.create_binomial_opinion(op_z_clutter_original, [0])
        else:
            # Higher expected clutter -> treat "about average" as the positive evidence
            op_z_bi_clutter = sl.create_binomial_opinion(op_z_clutter_original, [1])

        # Fuse into the running binomial clutter opinion
        op_up = sl.fusion_acbf(self._op_st["binomial_clutter"], op_z_bi_clutter)

        return [op_up, op_z_bi_clutter]

    def update_combinedInnovation_opinion(self, S_pred, combinedInnovation=None, weight_measurements=1.0):
        """Update the short-term combined innovation opinion.

        Uses a reduction parameter (q) to correct the covariance of the predicted
        measurement, following the PDA filter approach. The parameter values are
        interpolated from a lookup table based on detection probability and
        expected clutter.

        References
        ----------
        - "Detection thresholds for tracking in clutter? A connection between estimation
          and signal processing," IEEE (https://ieeexplore.ieee.org/abstract/document/1103935/)

        Parameters
        ----------
        S_pred : array_like, float
            Measurement prediction covariance matrix (from the Kalman filter).
        combinedInnovation : array_like, float, optional
            Combined innovation from the PDA filter. If None, only prediction weight
            contributes to the opinion.
        weight_measurements : array_like, float
            PDA-calculated weights for measurements (first entry is prediction weight).

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term combined innovation opinion after fusion.
            op_z : MultiOpinion
                Combined innovation opinion generated from the current measurement.
        """
        # Lookup table for reduction parameters q (from literature / offline computation)
        parameter_q_lookup_table = {
            1.0: {0: 1.00,     1: 0.70,    2: 0.60,     3: 0.51,    4: 0.45,    5: 0.38},
            0.9: {0: 0.85,     1: 0.61,    2: 0.46,     3: 0.36,    4: 0.29,    5: 0.24},
            0.8: {0: 0.78,     1: 0.50,    2: 0.38,     3: 0.31,    4: 0.26,    5: 0.22},
            0.7: {0: 0.68,     1: 0.39,    2: 0.28,     3: 0.22,    4: 0.19,    5: 0.16},
            0.6: {0: 0.58,     1: 0.30,    2: 0.21,     3: 0.17,    4: 0.13,    5: 0.12},
            0.5: {0: 0.48,     1: 0.23,    2: 0.15,     3: 0.11,    4: 0.09,    5: 0.085},
            0.4: {0: 0.38,     1: 0.15,    2: 0.10,     3: 0.08,    4: 0.06,    5: 0.06},
            0.3: {0: 0.28,     1: 0.09,    2: 0.06,     3: 0.05,    4: 0.05,    5: 0.05},
            0.2: {0: 0.18,     1: 0.05,    2: 0.04,     3: 0.03,    4: 0.03,    5: 0.03},
            0.1: {0: 0.08,     1: 0.015,   2: 0.01,     3: 0.01,    4: 0.01,    5: 0.01}
        }

        # Normalizing constant depending on measurement dimension
        if self.dim_meas == 1:
            c = 2
        elif self.dim_meas == 2:
            c = np.pi
        else:  # 3D
            c = (4 * np.pi) / 3

        # Volume of the validation gate (ellipsoid in measurement space)
        volume_gate = c * np.power(self.sigma_cutoff, self.dim_meas) * np.sqrt(np.linalg.det(S_pred))

        # Expected number of clutter measurements inside the gate
        clutter_in_gate = (self.exp_clutter / self.fov_area) * volume_gate
        lower_bound = math.floor(clutter_in_gate)
        upper_bound = math.ceil(clutter_in_gate)

        # Interpolate reduction parameter q from table
        if upper_bound > 5:
            parameter_q = parameter_q_lookup_table[self.prob_detect][5]
            print(f"Warning: Reduction parameter q={parameter_q} estimated inaccurately "
                  f"(upper bound exceeded). Details: volume_gate={volume_gate}, "
                  f"clutter_in_gate={clutter_in_gate}, bounds=({lower_bound}, {upper_bound}).")
        else:
            q_low = parameter_q_lookup_table[self.prob_detect][lower_bound]
            q_up = parameter_q_lookup_table[self.prob_detect][upper_bound]
            parameter_q = q_low + (q_up - q_low) * (clutter_in_gate - lower_bound)

        # Scale covariance with reduction parameter
        S_pred = parameter_q * S_pred

        # Allocate evidence array
        evidence_meas = np.zeros(self._X.number_bins)

        # Prediction weight (missed detection case)
        evidence_meas[-1] = weight_measurements[0]

        # Remaining weights (associated measurements)
        weight_without_prediction = sum(weight_measurements) - weight_measurements[0]

        # If an innovation is available, map it to evidence
        if combinedInnovation is not None:
            z_norm = np.dot(sqrtm(np.linalg.inv(S_pred)), combinedInnovation)
            distance = np.linalg.norm(z_norm)

            # Evidence based on normalized distance
            evidence_meas += self._X.check_value_for_bin(distance, weight=weight_without_prediction)

        # Generate opinion from evidence
        op_z = sl.MultiOpinion(
            evidence_meas,
            self._op_0["combined_innovation"].baseRate,
            1,
            evidence=True
        )

        # Fuse into short-term combined innovation opinion
        op_up = sl.fusion_acbf(self._op_st["combined_innovation"], op_z)

        return [op_up, op_z]

    def update_binomial_combinedInnovation_opinion(self, op_z_original):
        """Update the short-term binomial combined innovation opinion.

        Converts the original combined innovation opinion into a binomial form
        (by collapsing over specified event indices), then fuses it into the
        short-term opinion state.

        Parameters
        ----------
        op_z_original : MultiOpinion
            Original combined innovation opinion.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term binomial combined innovation opinion after fusion.
            op_z : MultiOpinion
                Binomial combined innovation opinion derived from the original.
        """
        # Convert to binomial form (collapse over all events [0–5])
        op_z_binomial = sl.create_binomial_opinion(op_z_original, [0, 1, 2, 3, 4, 5])

        # Fuse into the short-term binomial combined innovation opinion
        op_up = sl.fusion_acbf(self._op_st["binomial_combined_innovation"], op_z_binomial)

        return [op_up, op_z_binomial]

    def update_association_situation_opinion(self, weights=None):
        """Update the short-term association situation (entropy-based) opinion.

        The association situation is measured using entropy of the measurement
        weights (from PDA). A low entropy indicates a confident association
        (one weight dominates), while a high entropy indicates uncertainty
        (weights are more evenly distributed).

        Parameters
        ----------
        weights : array_like of float, optional
            Measurement weights provided by the PDA algorithm. If None, defaults
            to an empty list.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term entropy opinion.
            op_z : MultiOpinion
                Entropy opinion generated from the measurement weights.
        """
        if weights is None:
            weights = []

        # Default: prediction-only case (weights[0] == 1 -> no measurements)
        if len(weights) > 0 and weights[0] != 1:
            weight_prediction = weights[0]

            # Filter small weights to avoid skewing entropy with near-zero values
            weights = [w for w in weights if w > 1e-3]
            num_valid = len(weights)

            # Compute entropy
            entropy = sum(w * np.log2(w) for w in weights)
            entropy_max = np.log2(num_valid) if num_valid > 0 else 0
            entropy = np.array(entropy)

            # Normalize entropy to [0, 1]
            if entropy_max > 0:
                entropy_scaled = -entropy / entropy_max
            elif num_valid == 1:
                # Single measurement with high confidence -> low entropy
                entropy_scaled = 0
            else:
                # Edge case: should rarely happen
                entropy_scaled = 1
        else:
            # Prediction-only scenario (no real measurements)
            entropy_scaled = 0
            weight_prediction = 1

        # Save for diagnostics/visualization
        self.association_situation_value = entropy_scaled

        # Map entropy into discrete evidence bins
        evidence = np.zeros(2)
        if entropy_scaled <= self._association_situation_borders[0]:
            evidence[0] = 1
        else:
            evidence[1] = 1

        # Construct opinion (binary case: low vs. high entropy)
        op_z = sl.BiOpinion(
            evidence[0],
            evidence[1],
            self._op_0["association_situation"].baseRate[0],
            1,
            evidence=True
        )

        # Fuse into short-term association_situation opinion
        op_up = sl.fusion_acbf(self._op_st["association_situation"], op_z)

        return [op_up, op_z]

    def update_combined_opinion(self, op_z):
        """Update short-term combined opinion.

        Parameters
        ----------
        op_z : MultiOpinion
            Self-assessment opinions generated from the current measurements.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term combined opinion.
            op_z : BiOpinion
                Combined opinion generated from the current opinions.
            evidence_sum : float
                Total evidence used for normalization.

        """
        # Collect relevant opinions
        op_z_detection = op_z["detection"]
        op_z_clutter = op_z["binomial_clutter"]

        if self.association_algorithm == 'pda':
            op_z_combined_inno = op_z["binomial_combined_innovation"]
            evidence = op_z_detection.get_evidence() + op_z_clutter.get_evidence() + op_z_combined_inno.get_evidence()
        else:  # NN
            evidence = op_z_detection.get_evidence() + op_z_clutter.get_evidence()

        # Normalize evidence to ensure it sums to 1
        evidence_sum = np.sum(evidence)
        if evidence_sum > 0:
            evidence /= evidence_sum

        # Create current combined opinion
        op_z_combined = sl.BiOpinion(
            evidence[0],
            evidence[1],
            self._op_0["combined"].baseRate[0],
            1,
            evidence=True
        )
        # Fuse with short-term combined opinion
        op_up = sl.fusion_acbf(self._op_st["combined"], op_z_combined)

        return [op_up, op_z_combined, evidence_sum]

    def obtained_combined_opinion(self, op_X):
        """Obtain the current combined opinion for NN or PDA algorithms.

        Parameters
        ----------
        op_X : MultiOpinion
            Resulting short-term self-assessment (SA) opinion of the entire SA module.

        Returns
        -------
        MultiOpinion
            Combined opinion after fusing relevant sub-opinions.
        """
        if self.association_algorithm == 'nn':
            # For nearest-neighbor, fuse detection and binomialized clutter opinions
            op_X_detection = op_X["detection"]
            op_X_clutter = sl.create_binomial_opinion(op_X["clutter"], [1])
            op_X = sl.fusion_acbf_expanded([op_X_detection, op_X_clutter])

        elif self.association_algorithm == 'pda':
            # For PDA, fuse detection, combined innovation, and binomialized clutter opinions
            op_X_detection = op_X["detection"]
            op_X_clutter = sl.create_binomial_opinion(op_X["clutter"], [1])

            combined_innovation = deepcopy(op_X["combined_innovation"])
            # Optionally, apply trust discount from association_situation if needed:
            # combined_innovation = combined_innovation.trust_discount(op_X["association_situation"].get_proj_prob()[0])

            op_X_combined_inno = sl.create_binomial_opinion(combined_innovation, [0, 1, 2, 3, 4, 5])
            op_X = sl.fusion_acbf_expanded([op_X_detection, op_X_combined_inno, op_X_clutter])

        else:
            raise ValueError(f"Association algorithm '{self.association_algorithm}' is not implemented.")

        return op_X

    def update_overall_threshold_opinion(self):
        """Update the short-term overall threshold opinion based on threshold conflicts.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term threshold conflict opinion.
            op_z : MultiOpinion
                Threshold conflict opinion generated from the current SA measures (deltas) and thresholds.
        """
        # Select relevant opinion keys based on the association algorithm
        if self.association_algorithm == 'pda':
            relevant_keys = ["detection", "binomial_clutter", "binomial_combined_innovation"]
        else:  # 'nn'
            relevant_keys = ["detection", "binomial_clutter"]

        # Initialize equal-weighted evidence for the two bins: [no conflict, conflict]
        evidence = np.zeros(2)
        weight_per_opinion = 1 / len(relevant_keys)

        # Accumulate evidence based on delta vs threshold comparison
        for key in relevant_keys:
            delta = self._delta[list(self._op_0).index(key)]
            threshold = self._threshold_dc[list(self._op_0).index(key)]
            if delta >= threshold > 0:
                evidence[1] += weight_per_opinion  # conflict
            else:
                evidence[0] += weight_per_opinion  # no conflict

        # Create the short-term threshold opinion
        op_z = sl.BiOpinion(
            evidence[0], evidence[1],
            self._op_0["threshold"].baseRate[0],
            1, evidence=True
        )
        # Fuse the new opinion into the existing short-term opinion
        op_up = sl.fusion_acbf(self._op_st["threshold"], op_z)

        return [op_up, op_z]

    # ---------------------------------------------------- #
    # ------------------- Getter methods ----------------- #
    # ---------------------------------------------------- #

    def get_sas_measures(self, index_or_key=None, projected_prob=False):
        """Return the current self-assessment measures (DC, uncertainty, threshold).

        Parameters
        ----------
        index_or_key : int, str, list of int or list of str, optional
            If provided, return measures for the specified opinion(s) by index or key.
        projected_prob : bool, optional
            If True, return projected probabilities instead of [_delta, _u_delta, _threshold_dc].

        Returns
        -------
        list
            Deep copy of [_delta, _u_delta, _threshold_dc] or projected probabilities
            for all opinions, or for the specified index/key(s).
        """

        def resolve_index(idx_or_key):
            if isinstance(idx_or_key, str):
                return list(self._op_0).index(idx_or_key)
            return idx_or_key

        if projected_prob:
            probs = []
            if index_or_key is None:
                for key in ["threshold", "association_situation"]:
                    if key in self._op_X:
                        probs.append(self._op_X[key].get_projected_prob())
                return deepcopy(probs)

            if isinstance(index_or_key, (int, str)):
                idx = resolve_index(index_or_key)
                key = list(self._op_0)[idx]
                if key in self._op_X:
                    return deepcopy([self._op_X[key].get_projected_prob()])
                return []

            if isinstance(index_or_key, list):
                for idx_or_key in index_or_key:
                    idx = resolve_index(idx_or_key)
                    key = list(self._op_0)[idx]
                    if key in self._op_X:
                        probs.append(self._op_X[key].get_projected_prob())
                return deepcopy(probs)

        else:
            if index_or_key is None:
                return deepcopy([self._delta, self._u_delta, self._threshold_dc])

            if isinstance(index_or_key, (int, str)):
                idx = resolve_index(index_or_key)
                return deepcopy([self._delta[idx], self._u_delta[idx], self._threshold_dc[idx]])

            if isinstance(index_or_key, list):
                measures = []
                for idx_or_key in index_or_key:
                    idx = resolve_index(idx_or_key)
                    measures.append([self._delta[idx], self._u_delta[idx], self._threshold_dc[idx]])
                return deepcopy(measures)

        raise ValueError("index_or_key must be None, int, str, or list of int/str.")

    def get_last_opZ_fused(self):
        """Return the last fused opinion (op_Z) used for generating the overall tracking opinion.

        Returns
        -------
        MultiOpinion
            Deep copy of the most recent fused opinion (_op_z_memory["combined"][-1]).
        """
        return deepcopy(self._op_z_memory["combined"][-1])
        # combined opinion: version 2 - just fusion of single resulting SA opinions
        # return deepcopy(self._op_X["combined"])

    def get_last_opX(self):
        """Return the last combined opinion (op_X) for the overall assessment.

        Returns
        -------
        MultiOpinion
            Deep copy of the most recent combined opinion (_op_X["combined"]).
        """
        return deepcopy(self._op_X["combined"])

    def get_op_G(self):
        """Return the dogmatic opinion of the self-assessment components.

        Returns
        -------
        _op_G : float, array_like
            dogmatic opinion
        """
        return deepcopy(self._op_G)
