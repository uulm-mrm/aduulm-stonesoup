from copy import deepcopy
import numpy as np
from scipy.special import gammaincinv
from stonesoup.subjective_logic import subjective_logic as sl
from stonesoup.selfassessor._threshold import calc_threshold_op_diff as calc_tsh
from stonesoup.selfassessor import sot_clutter_selfassessor


class MultiSensorSOTClutterSelfAssessor:
    """Class for overall self-assessment in multi-sensor single-object tracking (SOT) in clutter
    using subjective logic based on single-sensor SOTClutterSelfAssessors.
    """
    def __init__(self, sa_configs, scenario_prob_det_configs, scenario_clutter_configs,
                 association_algorithm=None):

        sa_sensor = sa_configs['sensors']
        track_combined_opinion_params = sa_configs['track_combined_opinion']
        track_threshold_opinion_config = sa_configs['track_threshold_opinion']
        self.association = association_algorithm

        # Initialize self-assessors for each sensor
        self.selfassessors = {
            sens_name: sot_clutter_selfassessor.SOTClutterSelfAssessor(
                sa_sensor[sens_name]['num_X'],
                sa_sensor[sens_name]['n_st'],
                sa_sensor[sens_name]['n_c'],
                prob_detect=scenario_prob_det_configs[sens_name]['init_prob_det'],
                dim_meas=sa_sensor[sens_name]['dim_meas'],
                alpha_threshold_dc=sa_sensor[sens_name]['alpha_threshold_dc'],
                sigma_cutoff=np.sqrt(2 * gammaincinv(sa_sensor[sens_name]["dim_meas"] / 2,
                                                     1 - sa_sensor[sens_name]["sigma_cutoff_alpha"])),
                trust_discount=sa_sensor[sens_name]['trust_discount'],
                exp_clutter=scenario_clutter_configs[sens_name]["init_exp_num_clutter"],
                fov_area=sa_sensor[sens_name]['fov'][0] * sa_sensor[sens_name]['fov'][2],
                association_algorithm=association_algorithm
            )
            for n, sens_name in enumerate(sa_sensor)
        }

        # Initialize default opinions
        self._op_0 = {
            "track_combined": sl.BiOpinion(0, 0, 0.5, 1),
            "track_threshold": sl.BiOpinion(0, 0, 0.9, 1),
        }
        self._op_G = {
            "track_combined": sl.BiOpinion(0.5, 0.5, 0.5, 0),
            "track_threshold": sl.BiOpinion(0.9, 0.1, 0.9, 0),
        }

        self.n_st = {
            "track_combined": track_combined_opinion_params['n_st'],
            "track_threshold": track_threshold_opinion_config['n_st'],
        }

        self.alpha_threshold_dc = {
            "track_combined": track_combined_opinion_params['alpha_threshold_dc'],
            "track_threshold": track_threshold_opinion_config['alpha_threshold_dc'],
        }

        self.num_sensors = len(sa_sensor)

        # Memory and counters
        self._op_z_memory = {key: [] for key in self._op_G.keys()}
        self._counter_st_op = np.zeros(len(self._op_0))
        self._threshold_dc = np.zeros(len(self._op_0))
        self._delta = np.zeros(len(self._op_0))
        self._u_delta = np.ones(len(self._op_0))

        self.get_last_opZ_fused_sensor_memory = []
        self.get_last_opX_ss_sa_memory = []
        self.current_ss_sa_measures_memory = []
        self.current_ss_sa_multiple_measures_memory = []

        self._op_G_0_sensor_memory = [
            self.selfassessors[sens_name].get_op_G() for sens_name in sa_sensor
        ]

        # Generate initial track opinion
        self.generate_track_combined_opinion()
        self._op_st = deepcopy(self._op_0)
        self._op_X = deepcopy(self._op_0)
        self.track_threshold_proj_prob = None

    # ---------------------------------------------------- #
    # ------------- Generate opinion methods ------------- #
    # ---------------------------------------------------- #

    def generate_track_combined_opinion(self):
        """Generate the initial reference distribution for the overall track combined opinion.

        This method fuses the dogmatic combined opinions from all sensors to create a
        starting reference for the whole tracking module.

        Returns
        -------
        None
        """
        # Fuse dogmatic opinions from all sensors (averaged approach)
        dogmatic_fused = sl.fusion_averaged_expanded(
            [self._op_G_0_sensor_memory[idx]['combined'] for idx in range(len(self.selfassessors))]
        )

        # Initialize dogmatic opinion for overall track combined opinion
        self._op_G["track_combined"] = sl.BiOpinion(
            dogmatic_fused.baseRate[0],  # belief
            dogmatic_fused.baseRate[1],  # disbelief
            dogmatic_fused.baseRate[0],  # base rate
            0  # dogmatic (zero uncertainty)
        )

        # Initialize vague (uninformed) initial short-term opinion
        self._op_0["track_combined"] = sl.BiOpinion(
            0,  # zero belief
            0,  # zero disbelief
            dogmatic_fused.baseRate[0],  # base rate inherited from fused dogmatic
            1  # full uncertainty
        )

    # ---------------------------------------------------- #
    # ------------------ Getter methods ------------------ #
    # ---------------------------------------------------- #

    def get_sas_measures(self, sensor_name, index_or_key=None, projected_prob=False):
        """Return the current self-assessment measures (DC, uncertainty, threshold).

        Parameters
        ----------
        sensor_name : str
            Sensor for what the self-assessors are performed.
        index_or_key : int, str, list of int or list of str, optional
            If provided, return measures for the specified opinion(s) by index or key.
        projected_prob : bool, optional
            If True, return projected probabilities instead of [_delta, _u_delta, _threshold_dc].

        Returns
        -------
        list
            Self-assessment measures [_delta, _u_delta, _threshold_dc] or projected probabilities
            for all opinions, or for the specified index/key(s).
        """
        return self.selfassessors[sensor_name].get_sas_measures(index_or_key=index_or_key,
                                                                projected_prob=projected_prob)

    def get_track_opinion_measures(self):
        """Return the track overall combined self-assessment measures."""

        return deepcopy([self._delta[list(self._op_0).index("track_combined")],
                         self._u_delta[list(self._op_0).index("track_combined")],
                         self._threshold_dc[list(self._op_0).index("track_combined")]])

    # ---------------------------------------------------- #
    # ---------- Assess self-assessor methods ------------ #
    # ---------------------------------------------------- #

    def assess(self, sensor_name, z_p, S_p, meas_array, num_meas,
               weight_array=1.0, combined_innovation=None):
        """Perform self-assessment for single sensors and update overall track opinion
        if all sensors have been assessed.

        Parameters
        ----------
        sensor_name : str
            Name of the sensor to assess.
        z_p, S_p, meas_array, num_meas : array_like
            Inputs required by the sensor's self-assessor.
        weight_array : array_like, float, optional
            PDA-specific weights of the measurements.
        combined_innovation : array_like, optional
            Combined innovation for PDA filter.
        """

        # Determine which self-assessor indices are relevant based on association algorithm
        if self.association == 'pda':
            track_overall_sa_keys = ["detection", "binomial_clutter", "binomial_combined_innovation"]
            self.selfassessors[sensor_name].assess(
                z_p, S_p, meas_array, num_meas,
                weight_array=weight_array,
                combinedInnovation=combined_innovation
            )
        elif self.association == 'nn':
            track_overall_sa_keys = ["detection", "binomial_clutter"]
            self.selfassessors[sensor_name].assess(
                z_p, S_p, meas_array, num_meas
            )
        else:
            raise Exception(f"Association algorithm {self.association} not supported for assessment.")

        # Add the new sensor's self-assessment results first
        self.current_ss_sa_multiple_measures_memory.append(
            self.selfassessors[sensor_name].get_sas_measures(index_or_key=track_overall_sa_keys)
        )
        self.get_last_opZ_fused_sensor_memory.append(
            self.selfassessors[sensor_name].get_last_opZ_fused()
        )
        self.get_last_opX_ss_sa_memory.append(
            self.selfassessors[sensor_name].get_last_opX()
        )
        self.current_ss_sa_measures_memory.append(
            self.selfassessors[sensor_name].get_sas_measures(index_or_key=["combined"])
        )

        # Now check if all sensors are ready
        if len(self.get_last_opZ_fused_sensor_memory) == self.num_sensors:
            self.assess_overall_track()
            # Clear memory for next round
            self.get_last_opZ_fused_sensor_memory.clear()
            self.get_last_opX_ss_sa_memory.clear()
            self.current_ss_sa_measures_memory.clear()
            self.current_ss_sa_multiple_measures_memory.clear()

    def assess_overall_track(self):
        """Update the overall self-assessment track opinions based on the current assessments of all sensors."""

        # Obtain fused combined track opinion
        self._op_X["track_combined"] = self.obtain_track_combined_opinion()

        # Compute degree of conflict and uncertainty
        self._delta[list(self._op_0).index("track_combined")] = sl.dc(
            self._op_X["track_combined"],
            self._op_G["track_combined"]
        )
        self._u_delta[list(self._op_0).index("track_combined")] = self._op_X["track_combined"].uncertainty

        # Compute threshold for DC comparison with reference
        self._threshold_dc = calc_tsh(
            self._op_X, list(self.alpha_threshold_dc.values())
        )

        # Update short-term opinion for track threshold
        op_z = deepcopy(self._op_0)
        self._op_st["track_threshold"], op_z["track_threshold"] = self.update_track_threshold_opinion()

        # Update memory and short-term/long-term fusion
        key = "track_threshold"
        idx = list(self._op_0).index(key)  # map key to index
        self._op_z_memory[key].append(op_z[key])

        if self._counter_st_op[idx] < self.n_st[key]:
            self._op_X[key] = deepcopy(self._op_st[key])
            self._counter_st_op[idx] += 1.0
        else:
            # Unfuse the oldest opinion from short-term opinion
            op_st_to_lt = self._op_z_memory[key].pop(0)
            self._op_st[key] = sl.unfusion_cu(self._op_st[key], op_st_to_lt)
            self._op_X[key] = deepcopy(self._op_st[key])

        # Project the probability for monitoring or downstream usage
        self.track_threshold_proj_prob = self._op_X[key].get_projected_prob()

    # ---------------------------------------------------- #
    # ------------- Update opinion methods --------------- #
    # ---------------------------------------------------- #

    def obtain_track_combined_opinion(self):
        """Get the overall track combined opinion at the current time step.

        Returns
        -------
        MultiOpinion
            The resulting fused track combined opinion across all sensors.
        """
        # Fuse all short-term sensor opinions using an averaged approach
        op_X = sl.fusion_averaged_expanded(
            [self.get_last_opX_ss_sa_memory[idx] for idx in range(len(self.selfassessors))]
        )

        return op_X

    def update_track_threshold_opinion(self):
        """Update the short-term overall track threshold opinion.

        Returns
        -------
        list
            op_up : MultiOpinion
                Updated short-term track threshold opinion.
            op_z : MultiOpinion
                Current track threshold opinion generated from the multiple assessments.
        """
        # Determine equal weight per evidence contribution
        num_sensors = len(self.selfassessors)
        num_opinions_per_sensor = len(self.current_ss_sa_multiple_measures_memory[0])
        weight = 1 / (num_sensors * num_opinions_per_sensor)

        evidence = np.zeros(2)

        # Accumulate evidence across all sensors and opinions
        for sensor_idx in range(num_sensors):
            for opinion_idx in range(num_opinions_per_sensor):
                delta = self.current_ss_sa_multiple_measures_memory[sensor_idx][opinion_idx][0]
                threshold = self.current_ss_sa_multiple_measures_memory[sensor_idx][opinion_idx][2]

                if delta >= threshold > 0:
                    evidence[1] += weight
                else:
                    evidence[0] += weight

        # Create the short-term opinion
        op_z = sl.BiOpinion(
            evidence[0], evidence[1],
            self._op_0["track_threshold"].baseRate[0],
            1,
            evidence=True
        )

        # Fuse with the existing short-term opinion
        op_up = sl.fusion_acbf(self._op_st["track_threshold"], op_z)

        return [op_up, op_z]
