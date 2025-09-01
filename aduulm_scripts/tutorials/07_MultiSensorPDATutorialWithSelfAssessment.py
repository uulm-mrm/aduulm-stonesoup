#!/usr/bin/env python

"""
===========================================
7 - Probabilistic data association tutorial
===========================================
"""

# %%
# Making an assignment between a single track and a single measurement can be problematic. In the
# previous tutorials you may have encountered the phenomenon of *track seduction*. This occurs
# when clutter, or other track, points are mis-associated with a prediction. If this happens
# repeatedly (as can be the case in high-clutter or low-:math:`p_d` situations) the track can
# deviate significantly from the truth.
#
# Rather than make a firm assignment at each time-step, we could work out the probability that each
# measurement should be assigned to a particular target. We could then propagate a measure of
# these collective probabilities to mitigate the effect of track seduction.
#
# Pictorially:
#
# - Calculate a posterior for each hypothesis;
#
# .. image:: ../_static/PDA_Hypothesis_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# - Weight each posterior state according to the probability that its corresponding hypothesis
#   was true (including the probability of missed-detection);
#
# .. image:: ../_static/PDA_Weighting_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# - Merge the resulting estimate states in to a single posterior approximation.
#
# .. image:: ../_static/PDA_Merge_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# This results in a more robust approximation to the posterior state covariances that incorporates
# not only the uncertainty in state, but also in the association.

# %%
# A PDA filter example
# --------------------
#
# Ground truth
# ^^^^^^^^^^^^
#
# So, as before, we'll first begin by simulating some ground truth.
import numpy as np

from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

np.random.seed(1991)

start_time = datetime.now().replace(microsecond=0)
q_x = 0.005
q_y = 0.005
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])

# Import the disturbance method for the transition model
from aduulm_scripts.utils.add_disturbance import disturbance_transition_model
# Disturbance configurations for ground truth generation
gt_transition_configs = {
    'noise_diff_coeff': [[q_x, q_y]],  # for transition model gt
    'disturbance_mode': ['jump'],
    'parameters': [[[1000, 16.0], [1200, 0.0625]]]
}
process_noise_coeff_memory = [[], []]

num_steps = 1300
for k in range(1, num_steps):
    timesteps.append(start_time+timedelta(seconds=k))

    # Disturb the transition model based on the disturbance modes
    transition_model = disturbance_transition_model(transition_model, gt_transition_configs, k)
    # Save the noise coefficients (process noise memory)
    for i, model in enumerate(transition_model.model_list):
        # Collect covariance matrices for each model component
        process_noise_coeff_memory[i].append(model.noise_diff_coeff)

    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

# %%
# Add clutter.
from scipy.stats import uniform

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]])
    )

prob_detect = 0.9  # 90% chance of detection.

# %%
# Multi-sensor setup
from copy import deepcopy, copy

sensors = {"Sensor_01", "Sensor_02", "Sensor_03"}
measurement_models = {}
for sensor_name in sensors:
    measurement_models[sensor_name] = deepcopy(measurement_model)

# %%
# Import the disturbance method for the measurement model
from aduulm_scripts.utils.add_disturbance import disturbance_measurement_noise, disturbance_prob_det_and_clutter
from scipy.stats import poisson

disturbed_sensor = {"Sensor_01"}

# Disturbance configurations for measurement generation
gt_measurement_noise_configs_disturbed = {
    'disturbance_mode': ['jump'],
    'parameters': [[[100, 4], [300, 0.25]]]
}
gt_prob_det_configs_disturbed = {
    'init_prob_det': prob_detect,
    'disturbance_mode': ['jump'],
    'parameters': [[[700, 0.5], [900, 2.0]]]
}
gt_clutter_configs_disturbed = {
    'init_exp_num_clutter': 4,
    'disturbance_mode': ['jump'],
    'parameters': [[[400, 4], [600, 0.25]]]
}
gt_measurement_noise_configs_normal = {
    'disturbance_mode': ['none'],
    'parameters': []
}
gt_prob_det_configs_normal = {
    'init_prob_det': prob_detect,
    'disturbance_mode': ['none'],
    'parameters': []
}
gt_clutter_configs_normal = {
    'init_exp_num_clutter': 4,
    'disturbance_mode': ['none'],
    'parameters': []
}
gt_measurement_noise_configs = {}
gt_prob_det_configs = {}
gt_clutter_configs = {}
for sensor_name in sensors:
    if sensor_name in disturbed_sensor:
        gt_measurement_noise_configs[sensor_name] = deepcopy(gt_measurement_noise_configs_disturbed)
        gt_prob_det_configs[sensor_name] = deepcopy(gt_prob_det_configs_disturbed)
        gt_clutter_configs[sensor_name] = deepcopy(gt_clutter_configs_disturbed)
    else:
        gt_measurement_noise_configs[sensor_name] = deepcopy(gt_measurement_noise_configs_normal)
        gt_prob_det_configs[sensor_name] = deepcopy(gt_prob_det_configs_normal)
        gt_clutter_configs[sensor_name] = deepcopy(gt_clutter_configs_normal)

# field of view of all sensors
fov_x = fov_y = 20
fov_v_x = fov_v_y = 10
fov = [fov_x, fov_v_x, fov_y, fov_v_y]

all_meas_std_dev_memory = []
all_prob_det_memory = []
all_exp_num_clutter_memory = []
all_measurements = []
measurements_sensor = {"Sensor_01": [], "Sensor_02": [], "Sensor_03": []}

for sensor_name in sensors:
    measurement_model = measurement_models[sensor_name]
    gt_measurement_noise_config = gt_measurement_noise_configs[sensor_name]
    gt_prob_det_config = gt_prob_det_configs[sensor_name]
    gt_clutter_config = gt_clutter_configs[sensor_name]

    all_sensor_measurements = []
    sensor_meas_std_dev_memory = []
    sensor_prob_det_memory = []
    sensor_exp_num_clutter_memory = []
    prob_detect = gt_prob_det_config['init_prob_det']
    exp_num_clutter = gt_clutter_config['init_exp_num_clutter']

    for k, state in enumerate(truth):
        # Disturb the measurement model based on the disturbance modes
        measurement_model = disturbance_measurement_noise(measurement_model, gt_measurement_noise_config, k)
        # Disturb the detection probability
        prob_detect = disturbance_prob_det_and_clutter(gt_prob_det_config, k, prob_detect)

        sensor_measurement_set = set()
        # Generate actual detection from the state with a 1-p_d chance that no detection is received.
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(state, noise=True)
            sensor_measurement_set.add(TrueDetection(state_vector=measurement,
                                                     groundtruth_path=truth,
                                                     timestamp=state.timestamp,
                                                     measurement_model=measurement_model))

        # Disturb the clutter rate
        exp_num_clutter = disturbance_prob_det_and_clutter(gt_clutter_config, k, exp_num_clutter)
        # typically number of clutter measurements is Poisson distributed
        # with the expected number of clutter per time step
        clutter_range = poisson.rvs(exp_num_clutter)

        # Generate clutter at this time-step
        truth_x = state.state_vector[0]
        truth_y = state.state_vector[2]
        for _ in range(clutter_range):
            x = uniform.rvs(truth_x - fov_x / 2, fov_x)
            y = uniform.rvs(truth_y - fov_y / 2, fov_y)
            sensor_measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=state.timestamp,
                                               measurement_model=measurement_model))

        sensor_meas_std_dev_memory.append(np.sqrt(measurement_model.noise_covar))
        sensor_prob_det_memory.append(prob_detect)
        sensor_exp_num_clutter_memory.append(exp_num_clutter)
        all_sensor_measurements.append(sensor_measurement_set)

    all_measurements.append(all_sensor_measurements)
    measurements_sensor[sensor_name] = all_sensor_measurements

# %%
# Plot the ground truth and measurements with clutter.

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])

# Plot true detections and clutter.
# plotter.plot_measurements(all_measurements, [0, 2])
for sensor_name in sensors:
    plotter.plot_measurements(measurements_sensor[sensor_name], [0, 2], label=sensor_name)
plotter.fig


# %%
# Create the predictor and updater
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
# Multi-sensor updaters
updaters = {sensor_name: KalmanUpdater(measurement_models[sensor_name]) for sensor_name in sensors}

# %%
# Construct a Self-Assessor for the Multi-Sensor Single-Object Tracking in Clutter
# Using Probabilistic Data Association
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We're now ready to construct a self-assessor to monitor the assumptions of the tracking algorithm.

from stonesoup.selfassessor.multisensor_sot_clutter_selfassessor import MultiSensorSOTClutterSelfAssessor
from scipy.special import gammaincinv

# Self-assessor settings
sa_settings = {
    "sensors": {
        "Sensor_01": {
            "num_X": 7,
            "n_st": {
                "innovation": 35, "detection": 10, "gate": 35,
                "clutter": 15,
                "combined_innovation": 35, "association_situation": 15,
                "combined": 10, "threshold": 10,
                "binomial_clutter": 10, "binomial_combined_innovation": 10
            },
            "n_c": {
                "innovation": 1, "detection": 1, "gate": 1,
                "clutter": 1,
                "combined_innovation": 1, "association_situation": 1,
                "combined": 1, "threshold": 1,
                "binomial_clutter": 1, "binomial_combined_innovation": 1
            },
            "dim_meas": measurement_models["Sensor_01"].ndim_meas,
            "alpha_threshold_dc": {
                "innovation": 0.1, "detection": 0.3, "gate": 0.1,
                "clutter": 0.2,
                "combined_innovation": 0.1, "association_situation": 0.3,
                "combined": 0.3, "threshold": 0.3,
                "binomial_clutter": 0.3, "binomial_combined_innovation": 0.3
            },
            "sigma_cutoff_alpha": 0.01,
            "trust_discount": {
                "innovation": 0.99, "detection": 0.99, "gate": 0.99,
                "clutter": 0.99,
                "combined_innovation": 0.99, "association_situation": 0.99,
                "combined": 0.99, "threshold": 0.99,
                "binomial_clutter": 0.99, "binomial_combined_innovation": 0.99
            },
            "fov": [20, 10, 20, 10]
        },
        "Sensor_02": {
            "num_X": 7,
            "n_st": {
                "innovation": 35, "detection": 10, "gate": 35,
                "clutter": 15,
                "combined_innovation": 35, "association_situation": 15,
                "combined": 10, "threshold": 10,
                "binomial_clutter": 10, "binomial_combined_innovation": 10
            },
            "n_c": {
                "innovation": 1, "detection": 1, "gate": 1,
                "clutter": 1,
                "combined_innovation": 1, "association_situation": 1,
                "combined": 1, "threshold": 1,
                "binomial_clutter": 1, "binomial_combined_innovation": 1
            },
            "dim_meas": measurement_models["Sensor_02"].ndim_meas,
            "alpha_threshold_dc": {
                "innovation": 0.1, "detection": 0.3, "gate": 0.1,
                "clutter": 0.2,
                "combined_innovation": 0.1, "association_situation": 0.3,
                "combined": 0.3, "threshold": 0.3,
                "binomial_clutter": 0.3, "binomial_combined_innovation": 0.3
            },
            "sigma_cutoff_alpha": 0.01,
            "trust_discount": {
                "innovation": 0.99, "detection": 0.99, "gate": 0.99,
                "clutter": 0.99,
                "combined_innovation": 0.99, "association_situation": 0.99,
                "combined": 0.99, "threshold": 0.99,
                "binomial_clutter": 0.99, "binomial_combined_innovation": 0.99
            },
            "fov": [20, 10, 20, 10]
        },
        "Sensor_03": {
            "num_X": 7,
            "n_st": {
                "innovation": 35, "detection": 10, "gate": 35,
                "clutter": 15,
                "combined_innovation": 35, "association_situation": 15,
                "combined": 10, "threshold": 10,
                "binomial_clutter": 10, "binomial_combined_innovation": 10
            },
            "n_c": {
                "innovation": 1, "detection": 1, "gate": 1,
                "clutter": 1,
                "combined_innovation": 1, "association_situation": 1,
                "combined": 1, "threshold": 1,
                "binomial_clutter": 1, "binomial_combined_innovation": 1
            },
            "dim_meas": measurement_models["Sensor_03"].ndim_meas,
            "alpha_threshold_dc": {
                "innovation": 0.1, "detection": 0.3, "gate": 0.1,
                "clutter": 0.2,
                "combined_innovation": 0.1, "association_situation": 0.3,
                "combined": 0.3, "threshold": 0.3,
                "binomial_clutter": 0.3, "binomial_combined_innovation": 0.3
            },
            "sigma_cutoff_alpha": 0.01,
            "trust_discount": {
                "innovation": 0.99, "detection": 0.99, "gate": 0.99,
                "clutter": 0.99,
                "combined_innovation": 0.99, "association_situation": 0.99,
                "combined": 0.99, "threshold": 0.99,
                "binomial_clutter": 0.99, "binomial_combined_innovation": 0.99
            },
            "fov": [20, 10, 20, 10]
        }
    },
    'track_combined_opinion': {
        "n_st": 15,
        "alpha_threshold_dc": 0.3,
    },
    'track_threshold_opinion': {
        "n_st": 15,
        "alpha_threshold_dc": 0.3
    }
}
# Multi-sensor single-object tracking in clutter self-assessor
multisensor_selfassessors = MultiSensorSOTClutterSelfAssessor(sa_settings,
                                                              gt_prob_det_configs,
                                                              gt_clutter_configs,
                                                              association_algorithm='pda')
selfassessors_measures_combined_histories = {sensor_name: [] for sensor_name in sensors}
multisensor_measures_combined_history = []

from stonesoup.selfassessor.nis import NIS
# NIS settings
nis_settings = {
    "window_length": 1,  # window size of the NIS averaging
    "alpha": 0.01,  # significance level
    "dim_meas": measurement_models["Sensor_01"].ndim_meas,
    "sigma_cutoff_alpha": 0.0001  # gating significance level
}
sigma_cutoff_gating = np.sqrt(2 * gammaincinv(nis_settings["dim_meas"] / 2, 1 - nis_settings["sigma_cutoff_alpha"]))
nis_assessors = {sensor_name: NIS(window_length=nis_settings["window_length"],
                                  alpha=nis_settings["alpha"],
                                  dim=nis_settings["dim_meas"],
                                  sigma_cutoff=sigma_cutoff_gating,
                                  probability_detected=gt_prob_det_configs[sensor_name]["init_prob_det"],
                                  clutter_exp_number=gt_clutter_configs[sensor_name]["init_exp_num_clutter"],
                                  fov=fov,
                                  association_algorithm="pda")
                 for sensor_name in sensors}
nis_measures_histories = {sensor_name: [] for sensor_name in sensors}

# %%
# Initialise Probabilistic Data Associator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :class:`~.PDAHypothesiser` and :class:`~.PDA` associator generate track predictions and
# calculate probabilities for all prediction-detection pairs for a single prediction and multiple
# detections.
# The :class:`~.PDAHypothesiser` returns a collection of :class:`~.SingleProbabilityHypothesis`
# types. The :class:`~.PDA` takes these hypotheses and returns a dictionary of key-value pairings
# of each track and detection which it is to be associated with.
from stonesoup.hypothesiser.probability import PDAHypothesiser
# Note: Self-assessment results depend on the PDAHypothesiser settings,
# so ensure they match the self-assessment configuration.
hypothesiser = {sensor_name: PDAHypothesiser(predictor=predictor,
                                             updater=updaters[sensor_name],
                                             clutter_spatial_density=gt_clutter_configs[sensor_name][
                                                                         'init_exp_num_clutter'] / (fov_x * fov_y),
                                             prob_detect=gt_prob_det_configs[sensor_name]["init_prob_det"],
                                             prob_gate=1.0 - nis_settings['sigma_cutoff_alpha'])
                for sensor_name in sensors}

from stonesoup.dataassociator.probability import PDA
data_associators = {sensor_name: PDA(hypothesiser=hypothesiser[sensor_name]) for sensor_name in sensors}

# %%
# Run the PDA Filter
# ^^^^^^^^^^^^^^^^^^
#
# With these components, we can run the simulated data and clutter through the Kalman filter.

# Create prior
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# Loop through the predict, hypothesise, associate and update steps.
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors  # For storing state vectors during association
from stonesoup.functions import gm_reduce_single  # For merging states to get posterior estimate
from stonesoup.types.update import GaussianStateUpdate  # To store posterior estimate

prediction_memory = []

track = Track([prior])
# iterate over all time steps
for i in range(num_steps-1):
    # get measurements from all sensors for current time step
    measurements = [all_measurements[n][i] for n in range(len(sensors))]
    # prediction
    prediction = predictor.predict(prior, timestamp=start_time + timedelta(seconds=i))
    prediction_memory.append(prediction)

    track_sens = prediction
    # loop over all sensor measurements
    for n, sensor_name in enumerate(sensors):
        hypotheses = data_associators[sensor_name].associate({track_sens},
                                                             measurements[n],
                                                             start_time + timedelta(seconds=i))
        hypotheses_track = hypotheses[track_sens]

        H = measurement_models[sensor_name].matrix()
        z_p = H @ prediction_memory[i].mean
        S_p = H @ prediction_memory[i].covar @ H.T + measurement_models[sensor_name].noise_covar

        # Loop through each hypothesis, creating posterior states for each, and merge to calculate
        # approximation to actual posterior state mean and covariance.
        posterior_states = []
        posterior_state_weights = []
        # values used for self-assessment
        pda_associated_measurements = []
        pda_weights = []
        combined_innovation = None
        for hypothesis in hypotheses_track:
            pda_weight = float(hypothesis.probability)

            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
                # values used for self-assessment
                meas = None
            else:
                posterior_state = updaters[sensor_name].update(hypothesis)
                posterior_states.append(posterior_state)
                # values used for self-assessment
                meas = hypothesis.measurement.state_vector
                innovation = meas - z_p
                combined_innovation = pda_weight * innovation if combined_innovation is None \
                    else combined_innovation + pda_weight * innovation

            posterior_state_weights.append(
                hypothesis.probability)
            pda_associated_measurements.append(meas)
            pda_weights.append(pda_weight)

        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)

        # Reduce mixture of states to one posterior estimate Gaussian.
        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        # Add a Gaussian state approximation to the track.
        post = GaussianStateUpdate(
            post_mean, post_covar,
            hypotheses_track,
            hypotheses_track[0].measurement.timestamp)

        track_sens = post

        # self-assessment
        pda_weights = np.asarray(pda_weights)
        if np.sum(pda_weights) > 0:
            pda_weights /= np.sum(pda_weights)
        num_meas = len(measurements[n])
        multisensor_selfassessors.assess(sensor_name, z_p, S_p, pda_associated_measurements, num_meas,
                                         weight_array=pda_weights, combined_innovation=combined_innovation)
        nis_measures = nis_assessors[sensor_name].assess(z_p, S_p, combined_innovation)
        # save self-assessment results for plotting
        selfassessor_measures = multisensor_selfassessors.get_sas_measures(sensor_name, index_or_key="combined")
        selfassessors_measures_combined_histories[sensor_name].append(selfassessor_measures)
        nis_measures_histories[sensor_name].append(nis_measures)

    # save self-assessment results for plotting
    selfassessor_measures_track_combined = multisensor_selfassessors.get_track_opinion_measures()
    multisensor_measures_combined_history.append(selfassessor_measures_track_combined)

    track.append(post)
    prior = post

# %%
# Plot the resulting track

#plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show(renderer="browser")
plotter.show()

# %%
# Plot the self-assessment measures
from aduulm_scripts.utils.plotting import plot_selfassessment_with_nis, plot_multisensor_selfassessment
for sensor_name in sensors:
    plot_selfassessment_with_nis(selfassessors_measures_combined_histories[sensor_name],
                                 nis_measures_histories[sensor_name],
                                 nis_settings["alpha"], sensor_name=sensor_name, opinion_type='Combined Opinion',
                                 assessor_type='PDASelfAssessor')

plot_multisensor_selfassessment(multisensor_measures_combined_history,
                                assessor_type='Multi-Sensor PDASelfAssessor: Combined Opinion')

# %%
# Ground-truth based evaluation
from aduulm_scripts.utils.plotting import plot_gospa
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.measures import Euclidean
measure = Euclidean([0, 2])
gospa_generator = GOSPAMetric(c=20, p=1, measure=measure)
gospa_values = gospa_generator.compute_over_time(track[1:], [0] * len(track),
                                                 truth[:-1], [0] * len(truth))
plot_gospa(gospa_values)

# %%
# References
# ----------
# 1. Bar-Shalom Y, Daum F, Huang F 2009, The Probabilistic Data Association Filter, IEEE Control
# Systems Magazine

# %%
# Self-assessment references
# ----------
# .. [#] T. Griebel, J. Mueller, M. Buchholz, and K. Dietmayer, “Kalman filter meets subjective logic:
#        A self-assessing Kalman filter using subjective logic,” in 2020 IEEE 23rd International
#        Conference on Information Fusion (FUSION). IEEE, 2020.
# .. [#] T. Griebel, J. Mueller, P. Geisler, C. Hermann, M. Herrmann, M. Buchholz, and K. Dietmayer,
#        “Self-assessment for single-object tracking in clutter using subjective logic,”
#        in 2022 25th International Conference on Information Fusion (FUSION). IEEE, 2022.
# .. [#] T. Griebel, J. Heinzler, M. Buchholz and K. Dietmayer, "Online Performance Assessment
#        of Multi-Sensor Kalman Filters Based on Subjective Logic," 2023 26th International
#        Conference on Information Fusion (FUSION), Charleston, SC, USA, 2023, pp. 1-8.

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/Tutorial_7.PNG'
