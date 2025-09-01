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
    'parameters': [[[1000, 8.0], [1200, 0.125]]]
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

# Import the disturbance method for the measurement model
from aduulm_scripts.utils.add_disturbance import disturbance_measurement_noise, disturbance_prob_det_and_clutter
from scipy.stats import poisson

# Disturbance configurations for measurement generation
gt_measurement_noise_configs = {
    'disturbance_mode': ['jump'],
    'parameters': [[[100, 2], [300, 0.5]]]
}
gt_prob_det_configs = {
    'init_prob_det': prob_detect,
    'disturbance_mode': ['jump'],
    'parameters': [[[700, 0.6667], [900, 1.5]]]
}
gt_clutter_configs = {
    'init_exp_num_clutter': 4,
    'disturbance_mode': ['jump'],
    'parameters': [[[400, 2], [600, 0.5]]]
}
exp_num_clutter = gt_clutter_configs['init_exp_num_clutter']

fov_x = fov_y = 20
fov_v_x = fov_v_y = 10
fov = [fov_x, fov_v_x, fov_y, fov_v_y]

meas_std_dev_memory = []
prob_det_memory = []
exp_num_clutter_memory = []

all_measurements = []
for k, state in enumerate(truth):
    # Disturb the measurement model based on the disturbance modes
    measurement_model = disturbance_measurement_noise(measurement_model, gt_measurement_noise_configs, k)
    # Disturb the detection probability
    prob_det = disturbance_prob_det_and_clutter(gt_prob_det_configs, k, prob_detect)

    measurement_set = set()
    # Generate detection.
    if np.random.rand() <= prob_detect:
        measurement = measurement_model.function(state, noise=True)
        measurement_set.add(TrueDetection(state_vector=measurement,
                                          groundtruth_path=truth,
                                          timestamp=state.timestamp,
                                          measurement_model=measurement_model))

    # Disturb the clutter rate
    exp_num_clutter = disturbance_prob_det_and_clutter(gt_clutter_configs, k, exp_num_clutter)
    # typically number of clutter measurements is Poisson distributed
    # with the expected number of clutter per time step
    clutter_range = poisson.rvs(exp_num_clutter)
    # Generate clutter.
    truth_x = state.state_vector[0]
    truth_y = state.state_vector[2]
    for _ in range(clutter_range):
        x = uniform.rvs(truth_x - fov_x/2, fov_x)
        y = uniform.rvs(truth_y - fov_y/2, fov_y)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=state.timestamp,
                                    measurement_model=measurement_model))

    meas_std_dev_memory.append(np.sqrt(measurement_model.noise_covar))
    prob_det_memory.append(prob_det)
    exp_num_clutter_memory.append(exp_num_clutter)

    all_measurements.append(measurement_set)

# %%
# Plot the ground truth and measurements with clutter.

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])

# Plot true detections and clutter.
plotter.plot_measurements(all_measurements, [0, 2])
plotter.fig


# %%
# Create the predictor and updater
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Construct a Self-Assessor for the Single-Object Tracking in Clutter
# Using Probabilistic Data Association
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We're now ready to construct a self-assessor to monitor the assumptions of the tracking algorithm.

from stonesoup.selfassessor.sot_clutter_selfassessor import SOTClutterSelfAssessor
from scipy.special import gammaincinv

# Self-assessor settings
sa_settings = {
    "num_X": 7,
    "n_st": {"innovation": 35, "detection": 10, "gate": 35,
             "clutter": 15,
             "combined_innovation": 35, "association_situation": 15,
             "combined": 10, "threshold": 10,
             "binomial_clutter": 10, "binomial_combined_innovation": 10
             },
    "n_c": {"innovation": 1, "detection": 1, "gate": 1,
             "clutter": 1,
             "combined_innovation": 1, "association_situation": 1,
             "combined": 1, "threshold": 1,
             "binomial_clutter": 1, "binomial_combined_innovation": 1
             },
    "dim_meas": measurement_model.ndim_meas,
    "alpha_threshold_dc": {"innovation": 0.1, "detection": 0.3, "gate": 0.1,
                           "clutter": 0.2,
                           "combined_innovation": 0.1, "association_situation": 0.3,
                           "combined": 0.3, "threshold": 0.3,
                           "binomial_clutter": 0.3, "binomial_combined_innovation": 0.3
                           },
    "sigma_cutoff_alpha": 0.01,
    "trust_discount": {"innovation": 0.99, "detection": 0.99, "gate": 0.99,
                       "clutter": 0.99,
                       "combined_innovation": 0.99, "association_situation": 0.99,
                       "combined": 0.99, "threshold": 0.99,
                       "binomial_clutter": 0.99, "binomial_combined_innovation": 0.99
                       }
}
selfassessor = SOTClutterSelfAssessor(num_X=sa_settings["num_X"],
                                      n_st=sa_settings["n_st"],
                                      n_c=sa_settings["n_c"],
                                      prob_detect=gt_prob_det_configs['init_prob_det'],
                                      dim_meas=sa_settings["dim_meas"],
                                      alpha_threshold_dc=sa_settings["alpha_threshold_dc"],
                                      sigma_cutoff=np.sqrt(2 * gammaincinv(sa_settings["dim_meas"] / 2,
                                                                           1 - sa_settings["sigma_cutoff_alpha"])),
                                      trust_discount=sa_settings["trust_discount"],
                                      exp_clutter=gt_clutter_configs["init_exp_num_clutter"],
                                      fov_area=fov_x*fov_y,
                                      association_algorithm="pda")
selfassessor_measures_combined_history = []
multiple_selfassessor_measures_history = {
    'detection': [],
    'clutter': [],
    'combined_innovation': [],
    'association_situation': []
}

from stonesoup.selfassessor.nis import NIS
# NIS settings
nis_settings = {
    "window_length": 1,  # window size of the NIS averaging
    "alpha": 0.01,  # significance level
    "dim_meas": measurement_model.ndim_meas,
    "sigma_cutoff_alpha": 0.0001  # gating significance level
}
nis = NIS(window_length=nis_settings["window_length"],
          alpha=nis_settings["alpha"],
          dim=nis_settings["dim_meas"],
          sigma_cutoff=np.sqrt(2 * gammaincinv(nis_settings["dim_meas"] / 2, 1 - nis_settings["sigma_cutoff_alpha"])),
          probability_detected=gt_prob_det_configs["init_prob_det"],
          clutter_exp_number=gt_clutter_configs["init_exp_num_clutter"],
          fov=fov,
          association_algorithm="pda")
nis_measures_history = []

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
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=gt_clutter_configs['init_exp_num_clutter'] / (fov_x*fov_y),
                               prob_detect=gt_prob_det_configs["init_prob_det"],
                               prob_gate=1.0 - nis_settings['sigma_cutoff_alpha'])

from stonesoup.dataassociator.probability import PDA
data_associator = PDA(hypothesiser=hypothesiser)

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

track = Track([prior])
for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate({track},
                                           measurements,
                                           start_time + timedelta(seconds=n))

    hypotheses = hypotheses[track]

    H = measurement_model.matrix()
    z_p_pred = H @ hypotheses.single_hypotheses[0].prediction.mean
    S_p_pred = H @ hypotheses.single_hypotheses[0].prediction.covar @ H.T + measurement_model.noise_covar

    # Loop through each hypothesis, creating posterior states for each, and merge to calculate
    # approximation to actual posterior state mean and covariance.
    posterior_states = []
    posterior_state_weights = []
    # values used for self-assessment
    pda_associated_measurements = []
    pda_weights = []
    combined_innovation = None
    for hypothesis in hypotheses:
        pda_weight = float(hypothesis.probability)

        if not hypothesis:
            posterior_states.append(hypothesis.prediction)
            # values used for self-assessment
            meas = None
            z_p, S_p = z_p_pred, S_p_pred
        else:
            posterior_state = updater.update(hypothesis)
            posterior_states.append(posterior_state)
            # values used for self-assessment
            meas = hypothesis.measurement.state_vector
            z_p, S_p = hypothesis.measurement_prediction.mean, hypothesis.measurement_prediction.covar
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
    track.append(GaussianStateUpdate(
        post_mean, post_covar,
        hypotheses,
        hypotheses[0].measurement.timestamp))

    # self-assessment
    pda_weights = np.asarray(pda_weights)
    if np.sum(pda_weights) > 0:
        pda_weights /= np.sum(pda_weights)
    num_meas = len(measurements)
    selfassessor.assess(z_p_pred, S_p_pred, pda_associated_measurements, num_meas,
                        weight_array=pda_weights, combinedInnovation=combined_innovation)
    nis_measures = nis.assess(z_p_pred, S_p_pred, combined_innovation)
    # save self-assessment results for plotting
    selfassessor_measures_combined = selfassessor.get_sas_measures(index_or_key='combined')
    selfassessor_measures_combined_history.append(selfassessor_measures_combined)
    nis_measures_history.append(nis_measures)
    # SA components
    selfassessor_measures_detection = selfassessor.get_sas_measures(index_or_key='detection')
    multiple_selfassessor_measures_history['detection'].append(selfassessor_measures_detection)
    selfassessor_measures_clutter = selfassessor.get_sas_measures(index_or_key='clutter')
    multiple_selfassessor_measures_history['clutter'].append(selfassessor_measures_clutter)
    selfassessor_measures_combined_innovation = selfassessor.get_sas_measures(index_or_key='combined_innovation')
    multiple_selfassessor_measures_history['combined_innovation'].append(selfassessor_measures_combined_innovation)
    selfassessor_measures_association_situation = selfassessor.get_sas_measures(index_or_key='association_situation',
                                                                                projected_prob=True)
    multiple_selfassessor_measures_history['association_situation'].append(selfassessor_measures_association_situation)
# %%
# Plot the resulting track

plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show(renderer="browser")
plotter.show()

# %%
# Plot the self-assessment measures
from aduulm_scripts.utils.plotting import plot_selfassessment_with_nis, plot_multiple_selfassessment_measures
plot_selfassessment_with_nis(selfassessor_measures_combined_history, nis_measures_history, nis_settings["alpha"],
                             opinion_type='Combined Opinion', assessor_type='PDASelfAssessor')
plot_multiple_selfassessment_measures(multiple_selfassessor_measures_history, assessor_type='PDASelfAssessor')

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
