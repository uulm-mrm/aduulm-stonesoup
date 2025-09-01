#!/usr/bin/env python

"""
==============================================
3 - Non-linear models: unscented Kalman filter
==============================================
"""

# %%
# The previous tutorial showed how the extended Kalman filter propagates estimates using a
# first-order linearisation of the transition and/or sensor models. Clearly there are limits to
# such an approximation, and in situations where models deviate significantly from linearity,
# performance can suffer.
#
# In such situations it can be beneficial to seek alternative approximations. One such comes via
# the so-called *unscented transform* (UT). In this we characterise a Gaussian distribution using a
# series of weighted samples, *sigma points*, and propagate these through the non-linear function.
# A transformed Gaussian is then reconstructed from the new sigma points. This forms the basis for
# the unscented Kalman filter (UKF).
#
# This tutorial will first run a simulation in an entirely equivalent fashion to the previous
# (EKF) tutorial. We'll then look into more precise details concerning the UT and try and develop
# some intuition into the reasons for its effectiveness.

# %%
# Background
# ----------
# Limited detail on how Stone Soup does the UKF is provided below. See Julier et al. (2000) [#]_
# for fuller, better details of the UKF.
#
# For dimension :math:`D`, a set of :math:`2 D + 1` sigma points are calculated at:
#
# .. math::
#           \mathbf{s}_j &= \mathbf{x}, \ \ j = 0 \\
#           \mathbf{s}_j &= \mathbf{x} + \alpha \sqrt{\kappa} A_j, \ \ j = 1, ..., D \\
#           \mathbf{s}_j &= \mathbf{x} - \alpha \sqrt{\kappa} A_j, \ \ j = D + 1, ..., 2 D
#
# where :math:`A_j` is the :math:`j` th column of :math:`A`, a *square root matrix* of the
# covariance, :math:`P = AA^T`, of the state to be approximated, and :math:`\mathbf{x}` is its
# mean.
#
# Two sets of weights, mean and covariance, are calculated:
#
# .. math::
#           W^m_0 &= \frac{\lambda}{c} \\
#           W^c_0 &= \frac{\lambda}{c} + (1 - \alpha^2 + \beta) \\
#           W^m_j &= W^c_j = \frac{1}{2 c}
#
# where :math:`c = \alpha^2 (D + \kappa)`, :math:`\lambda = c - D`. The parameters
# :math:`\alpha, \ \beta, \ \kappa` are user-selectable parameters with default values of
# :math:`0.5, \ 2, \ 3 - D`.
#
# After the sigma points are transformed :math:`\mathbf{s^{\prime}} = f( \mathbf{s} )`, the
# distribution is reconstructed as:
#
# .. math::
#           \mathbf{x}^\prime &= \sum\limits^{2 D}_{0} W^{m}_j \mathbf{s}^{\prime}_j \\
#           P^\prime &= (\mathbf{s}^{\prime} - \mathbf{x}^\prime) \, diag(W^c) \,
#           (\mathbf{s}^{\prime} - \mathbf{x}^\prime)^T + Q
#
# The posterior mean and covariance are accurate to the 2nd order Taylor expansion for any
# non-linear model. [#]_

# %%
# Nearly-constant velocity example
# --------------------------------
# This example is equivalent to that in the previous (EKF) tutorial. As with that one, you are
# invited to play with the parameters and watch what happens.

# Some general imports and initialise time
import numpy as np

from datetime import datetime, timedelta
start_time = datetime.now().replace(microsecond=0)

# %%

np.random.seed(1991)

# %%
# Create ground truth
# ^^^^^^^^^^^^^^^^^^^
#
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

q_x = 0.05
q_y = 0.05
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
    'parameters': [[[1100, 16.0], [1300, 0.0625]]]
}
process_noise_coeff_memory = [[], []]

num_steps = 1400
for k in range(1, num_steps + 1):
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
# Set-up plot to render ground truth, as before.

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Simulate the measurement
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
# Sensor position
sensor_x = 50
sensor_y = 0

# Make noisy measurement (with bearing variance = 0.002 degrees).
measurement_model = CartesianToBearingRange(ndim_state=4,
                                            mapping=(0, 2),
                                            noise_covar=np.diag([np.radians(0.002), 0.1]),
                                            translation_offset=np.array([[sensor_x], [sensor_y]]))

# %%
from stonesoup.types.detection import Detection

# Import the disturbance method for the measurement model
from aduulm_scripts.utils.add_disturbance import disturbance_measurement_noise
# Disturbance configurations for measurement generation
gt_measurement_configs = {
    'disturbance_mode': ['jump', 'drift', 'outliers'],
    'parameters': [[[100, 2], [300, 0.5]], [[400, 500, 2.5], [600, 700, 0.4]], [[800, 1000, 30, 3]]]
}
meas_std_dev_memory = []

# Make sensor that produces the noisy measurements.
measurements = []
for k, state in enumerate(truth):
    # Disturb the measurement model based on the disturbance modes
    measurement_model = disturbance_measurement_noise(measurement_model, gt_measurement_configs, k)

    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model))

    meas_std_dev_memory.append(np.sqrt(measurement_model.noise_covar))

# Plot the measurements
# Where the model is nonlinear the plotting function uses the inverse function to get coordinates

plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Create unscented Kalman filter components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that the transition of the target state is linear, so we have no real need for a
# :class:`~.UnscentedKalmanPredictor`. But we'll use one anyway, if nothing else to demonstrate
# that a linear model won't break anything.
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
predictor = UnscentedKalmanPredictor(transition_model)
# Create :class:`~.UnscentedKalmanUpdater`
from stonesoup.updater.kalman import UnscentedKalmanUpdater
unscented_updater = UnscentedKalmanUpdater(measurement_model)  # Keep alpha as default = 0.5

# %%
# Construct a Self-Assessor for the Unscented Kalman Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We're now ready to construct a self-assessor to monitor the assumptions of the unscented Kalman filter.

from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
# Self-assessor settings
sa_settings = {
    "num_X": 7,
    "n_st": 35,
    "n_c": 1,
    "dim_meas": measurement_model.ndim_meas,
    "alpha_threshold_dc": 0.1,
    "trust_discount": 0.99,
}
selfassessor = KalmanSelfAssessor(num_X=sa_settings["num_X"],
                                  n_st=sa_settings["n_st"],
                                  n_c=sa_settings["n_c"],
                                  dim_meas=sa_settings["dim_meas"],
                                  alpha_threshold_dc=sa_settings["alpha_threshold_dc"],
                                  trust_discount=sa_settings["trust_discount"])
selfassessor_measures_history = []

from stonesoup.selfassessor.nis import NIS
# NIS settings
nis_settings = {
    "window_length": 35,  # window size of the NIS averaging
    "alpha": 0.01,  # significance level
    "dim_meas": measurement_model.ndim_meas,
}
nis = NIS(window_length=nis_settings["window_length"],
          alpha=nis_settings["alpha"],
          dim=nis_settings["dim_meas"])
nis_measures_history = []

# %%
# Run the Unscented Kalman Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a prior
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Populate the track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = unscented_updater.update(hypothesis)
    track.append(post)
    prior = track[-1]
    # self-assessment
    z_p = hypothesis.measurement_prediction.mean
    S_p = hypothesis.measurement_prediction.covar
    selfassessor.assess(z_p, S_p, measurement.state_vector)
    selfassessor_measures = selfassessor.get_sas_measures()
    nis_measures = nis.assess(z_p, S_p, measurement.state_vector)
    selfassessor_measures_history.append(selfassessor_measures)
    nis_measures_history.append(nis_measures)

# %%
# And plot

plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show(renderer="browser")
plotter.fig

# %%
# Plot the self-assessment measures
from aduulm_scripts.utils.plotting import plot_selfassessment_with_nis
plot_selfassessment_with_nis(selfassessor_measures_history, nis_measures_history, nis_settings["alpha"],
                             assessor_type='KalmanSelfAssessor')

# %%
# The UT in slightly more depth
# -----------------------------
# We will skip this part of the original tutorial in Stone Soup.
# Please find this part in *03_UnscentedKalmanFilterTutorial.py*.

# ...

# %%
# You may have to spend some time fiddling with the parameters to see major differences between the
# EKF and UKF. Indeed, the point to make is not that there is any great magic about the UKF. Its
# power is that it harnesses some extra free parameters to give a more flexible description of the
# transformed distribution.

# %%
# Key points
# ----------
# 1. The unscented Kalman filter offers a powerful alternative to the EKF when undertaking tracking
#    in non-linear regimes.

# %%
# References
# ----------
# .. [#] Julier S., Uhlmann J., Durrant-Whyte H.F. 2000, A new method for the nonlinear
#        transformation of means and covariances in filters and estimators, in IEEE Transactions
#        on Automatic Control, vol. 45, no. 3, pp. 477-482, doi: 10.1109/9.847726.
# .. [#] Julier S.J. 2002, The scaled unscented transformation, Proceedings of the 2002 American
#        Control Conference (IEEE Cat. No.CH37301), Anchorage, AK, USA, 2002, pp. 4555-4559 vol.6,
#        doi: 10.1109/ACC.2002.1025369.

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
# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/Tutorial_3.PNG'
