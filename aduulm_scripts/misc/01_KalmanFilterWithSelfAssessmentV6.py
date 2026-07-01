#!/usr/bin/env python

"""
==========================================================
1 - An introduction to Stone Soup: using the Kalman filter
==========================================================
"""

# %%
import numpy as np
from datetime import datetime, timedelta

from matplotlib.pyplot import ylabel

import subjective_logic as sl
import matplotlib
from copy import deepcopy
from stonesoup.types.prediction import GaussianStatePrediction, MeasurementPrediction
from collections import deque
from dataclasses import dataclass

matplotlib.use('TkAgg')
def sample_uniform_noise_from_cov(R, rng, spread_factor=(1.0, 1.0)):
    R = np.asarray(R, dtype=float)

    sigma = np.sqrt(np.diag(R))
    spread_factor = np.asarray(spread_factor, dtype=float)

    half_width = spread_factor * np.sqrt(3.0) * sigma

    noise = rng.uniform(
        low=-half_width,
        high=half_width,
        size=R.shape[0]
    )

    return noise.reshape(-1, 1)

def sample_laplace_noise_from_cov(R, rng, spread_factor=1.0):
    """
    Zero-mean Laplace noise with same marginal variances as R
    if spread_factor = 1.

    spread_factor > 1 increases variance by spread_factor**2.
    Assumes diagonal R.
    """
    R = np.asarray(R, dtype=float)

    sigma = np.sqrt(np.diag(R))
    scale = spread_factor * sigma / np.sqrt(2.0)

    noise = rng.laplace(
        loc=0.0,
        scale=scale,
        size=R.shape[0],
    )

    return noise.reshape(-1, 1)

def sample_student_t_noise_from_cov(R, rng, df=3.0, spread_factor=1.0):
    """
    Zero-mean Student-t noise with same marginal variances as R
    if spread_factor = 1 and df > 2.

    Smaller df => heavier tails.
    Typical choices:
        df = 3: very heavy-tailed
        df = 5: moderately heavy-tailed
        df = 10: close to Gaussian
    """
    if df <= 2:
        raise ValueError("df must be > 2 to have finite variance.")

    R = np.asarray(R, dtype=float)

    sigma = np.sqrt(np.diag(R))
    scale = spread_factor * sigma * np.sqrt((df - 2.0) / df)

    noise = rng.standard_t(df=df, size=R.shape[0]) * scale

    return noise.reshape(-1, 1)

def sample_bimodal_noise_from_cov(
    R,
    rng,
    mode_distance=2.0,
    mode_prob=0.5,
    within_scale=0.3,
    axis=0,
):
    """
    Bimodal measurement noise.

    Creates two modes along one measurement axis.

    mode_distance is measured in sigma units.
    within_scale controls the Gaussian spread within each mode.
    """
    R = np.asarray(R, dtype=float)

    m = R.shape[0]
    sigma = np.sqrt(np.diag(R))

    noise = rng.normal(
        loc=0.0,
        scale=within_scale * sigma,
        size=m,
    )

    sign = 1.0 if rng.uniform() < mode_prob else -1.0
    noise[axis] += sign * mode_distance * sigma[axis]

    return noise.reshape(-1, 1)

def sample_truncated_gaussian_noise_from_cov(
    R,
    rng,
    truncation_sigma=1.0,
    max_tries=10_000,
):
    """
    Samples from N(0, R) conditioned component-wise on
    |v_i| <= truncation_sigma * sigma_i.

    This keeps one measurement per timestep.
    """
    R = np.asarray(R, dtype=float)
    sigma = np.sqrt(np.diag(R))
    m = R.shape[0]

    noise = np.zeros(m)

    for dim in range(m):
        for _ in range(max_tries):
            candidate = rng.normal(loc=0.0, scale=sigma[dim])

            if abs(candidate) <= truncation_sigma * sigma[dim]:
                noise[dim] = candidate
                break
        else:
            raise RuntimeError(
                f"Could not sample truncated Gaussian for dim={dim}. "
                f"Try larger truncation_sigma."
            )

    return noise.reshape(-1, 1)

def sample_correlated_gaussian_noise_from_cov(
    R,
    rng,
    rho=0.8,
):
    """
    Samples zero-mean Gaussian measurement noise with the same marginal
    variances as R, but with correlation rho between x and y.

    The filter may still assume diagonal R, so the marginal variances are
    correct but the joint covariance structure is wrong.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (2, 2):
        raise ValueError("This function assumes a 2D measurement covariance.")

    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1).")

    sigma = np.sqrt(np.diag(R))

    R_corr = np.array([
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ])

    noise = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=R_corr,
    )

    return noise.reshape(-1, 1)

def sample_diagonal_mixture_noise_from_cov(
    R,
    rng,
    spread_factor=1.0,
):
    """
    Measurement-level diagonal-mixture noise.

    Construction:
        z ~ N(0, 1)
        s in {-1, +1} with equal probability

        v_x = sigma_x * z
        v_y = sigma_y * s * z

    Marginals:
        v_x ~ N(0, R_xx)
        v_y ~ N(0, R_yy)

    Joint distribution:
        not bivariate Gaussian; samples lie on two diagonals.

    If spread_factor = 1:
        marginal variances match R.

    If spread_factor > 1:
        marginal variances are increased, so components may also react.
        For the radial-only demonstration, keep spread_factor = 1.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (2, 2):
        raise ValueError("This function assumes a 2D measurement covariance.")

    sigma = np.sqrt(np.diag(R))

    z = rng.normal(loc=0.0, scale=1.0)
    s = 1.0 if rng.uniform() < 0.5 else -1.0

    noise = np.array([
        sigma[0] * z,
        sigma[1] * s * z,
    ])

    noise = spread_factor * noise

    return noise.reshape(-1, 1)


def sample_diagonal_mixture_noise_from_cov(
    R,
    rng,
    spread_factor=1.0,
):
    """
    Measurement-level diagonal-mixture noise.

    Construction:
        z ~ N(0, 1)
        s in {-1, +1} with equal probability

        v_x = sigma_x * z
        v_y = sigma_y * s * z

    Marginals:
        v_x ~ N(0, R_xx)
        v_y ~ N(0, R_yy)

    Joint distribution:
        not bivariate Gaussian; samples lie on two diagonals.

    If spread_factor = 1:
        marginal variances match R.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (2, 2):
        raise ValueError("This function assumes a 2D measurement covariance.")

    sigma = np.sqrt(np.diag(R))

    z = rng.normal(loc=0.0, scale=1.0)
    s = 1.0 if rng.uniform() < 0.5 else -1.0

    noise = np.array([
        sigma[0] * z,
        sigma[1] * s * z,
    ])

    noise = spread_factor * noise

    return noise.reshape(-1, 1)
# %%
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, KnownTurnRate, CombinedGaussianTransitionModel
from stonesoup.models.transition.nonlinear import ConstantTurn
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import State, GaussianState

start_time = datetime.now()
np.random.seed(0) #2

# %%
use_ct_model = False
activate_disturbances = True
q_x   = 0.25   # Geschwindigkeitsrauschen (σ_vvel)
q_y   = 0.25
if use_ct_model:
    q_phi = 0.01   # Drehraten-Rauschen (σ_omega)
    transition_model = ConstantTurn([q_x, q_y], q_phi)
else:
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), ConstantVelocity(q_y)])

# %%
# Ground truth: CV until k=400, then realistic right turn, then CV again.
# Compatible with the old manual truth-generation loop.
#
# State convention:
#   x = [x, vx, y, vy]^T

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.array import StateVector
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, KnownTurnRate
from datetime import timedelta
import numpy as np

# ------------------------------------------------------------------
# Ground-truth model parameters
# ------------------------------------------------------------------
dt = timedelta(seconds=0.1)
num_steps = 1400

# CV model before and after the turn
gt_cv_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(q_x),
    ConstantVelocity(q_y)
])

# Right-turn model
# Stone Soup KnownTurnRate expects turn rate in rad/s.
# Negative = right turn.
#
# Example:
#   -10 deg/s for 6 seconds => about 60 degree right turn.
turn_rate = np.radians(-20.0)
gt_right_turn_model = KnownTurnRate([q_x, q_y], turn_rate)

# Turn interval
turn_start = 900
turn_end = 1000   # 60 steps at dt=0.1s -> 6 seconds

# ------------------------------------------------------------------
# Initial ground truth
# ------------------------------------------------------------------
timesteps = [start_time]

truth = GroundTruthPath([
    GroundTruthState(
        StateVector([[0.0], [5.0], [0.0], [5.0]]),
        timestamp=timesteps[0]
    )
])

# For plotting / debugging
model_indices = []
process_noise_coeff_memory = [[], []]

# Import the disturbance method for the transition model
from aduulm_scripts.utils.add_disturbance import disturbance_transition_model
disturbance_factor_process = 32 if activate_disturbances else 1 #16
# Disturbance configurations for ground truth generation
gt_transition_configs = {
    'noise_diff_coeff': [[q_x, q_y]],  # for transition model gt
    'disturb_noise_coeff': [True, False],
    'disturbance_mode': ['jump'],
    # 'parameters': [[[150, disturbance_factor_process], [200, 1/disturbance_factor_process], [250, disturbance_factor_process], [300, 1/disturbance_factor_process]]] #, [[99, 1/100]]]
    'parameters': [[[1100, disturbance_factor_process], [1300, 1/disturbance_factor_process]]], #, [800, 1/disturbance_factor_process], [850, disturbance_factor_process]]]
}

# ------------------------------------------------------------------
# Generate GT trajectory
# ------------------------------------------------------------------
for k in range(1, num_steps + 1):

    timesteps.append(start_time + k * dt)

    ###################################################################################################
    try:
        # # Disturb the transition model based on the disturbance modes
        # if turn_start <= k < turn_end:
        #     gt_right_turn_model = disturbance_transition_model(gt_right_turn_model, gt_transition_configs, k)
        # else:
        #     gt_cv_model = disturbance_transition_model(gt_cv_model, gt_transition_configs, k)
        gt_cv_model = disturbance_transition_model(gt_cv_model, gt_transition_configs, k)

        if turn_start <= k < turn_end:
            gt_model = gt_right_turn_model
            model_indices.append(1)  # right turn
        else:
            gt_model = gt_cv_model
            model_indices.append(0)  # CV
        # Save the noise coefficients (process noise memory)
        # print(k, ":\t", gt_model.model_list)
        if len(gt_model.model_list) == 0:
            # KnownTurnRate may not expose model_list in the same way
            process_noise_coeff_memory[0].append(q_x)
            process_noise_coeff_memory[1].append(q_y)
        else:
            for i, model in enumerate(gt_model.model_list):
                try:
                    # Collect covariance matrices for each model component
                    process_noise_coeff_memory[i].append(model.noise_diff_coeff)
                except Exception as e:
                    print(e)
                    # KnownTurnRate may not expose model_list in the same way
                    process_noise_coeff_memory[0].append(q_x)
                    process_noise_coeff_memory[1].append(q_y)
    except:
        print("Disturbance generation failed, trying to continue")
    ###################################################################################################

    # Debug prints
    # if k in [699, 700, 701, 799, 800, 801]:
    #     print("k =", k)
    #     print("model:", type(gt_model))
    #     print("covar:\n", gt_model.covar(time_interval=dt))
    #     try:
    #         print("noise coeffs:", [m.noise_diff_coeff for m in gt_model.model_list])
    #     except Exception as e:
    #         print("no model_list/noise_diff_coeff:", e)

    truth.append(
        GroundTruthState(
            gt_model.function(
                truth[k - 1],
                noise=True,
                time_interval=dt
            ),
            timestamp=timesteps[k]
        )
    )

truths = set([truth])

# # %%
from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=1) #, height=300) #, height=1000)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig
# %%
# We can check the :math:`F_k` and :math:`Q_k` matrices (generated over a 1s period).
# transition_model.matrix(time_interval=timedelta(seconds=.1))

# %%
transition_model.covar(time_interval=timedelta(seconds=.1))

# %%
# We're going to need a :class:`~.Detection` type to
# store the detections, and a :class:`~.LinearGaussian` measurement model.
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
import numpy as np

# %%
# GT-Messmodell: 4D [x, vx, y, vy] – GT läuft mit KnownTurnRate (4D)
meas_cov = 1
gt_measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[meas_cov, 0],
                          [0, meas_cov]])
)
# Filter-Messmodell: 5D [x, vx, y, vy, omega] – CTRV-State
# mapping=(0, 2): misst x (Index 0) und y (Index 2)
measurement_model = LinearGaussian(
    ndim_state=5 if use_ct_model else 4,
    mapping=(0, 2),
    noise_covar=np.array([[meas_cov, 0],
                          [0, meas_cov]])
)

# %%
# Check the output is as we expect
measurement_model.matrix()

# %%
measurement_model.covar()
#%%
stationary_measurement_model = deepcopy(measurement_model)

# %%
# Generate the measurements

# Import the disturbance method for the measurement model
from aduulm_scripts.utils.add_disturbance import disturbance_measurement_noise
# Disturbance configurations for measurement generation
disturbance_factor_meas = 2 if activate_disturbances else 1 #4
gt_measurement_configs = {
    # 'disturbance_mode': ['jump', 'drift', 'outliers'],
    # 'parameters': [[[50, 4], [100, 0.25]], [[200, 300, 2.5], [300, 400, 0.4]], [[500, 550, 5, 5]]]
    'disturbance_mode': ['jump', 'outliers'],
    'parameters': [[[300, disturbance_factor_meas, [1, 0]], [400, 1/disturbance_factor_meas, [1, 0]], [500, 1/disturbance_factor_meas, [1, 1]], [600, disturbance_factor_meas, [1, 1]]], [[100, 200, 10, 8]]], #[400, 1/disturbance_factor_meas, [1, 1]], [500, disturbance_factor_meas, [1, 1]]]],
    # 'disturb_noise_coeff': [[1, 0], [1, 1]], #[x, y], # 0= no disturbance, 1= disturbance_factor_meas, 2= 1/disturbance_factor_meas
}
meas_std_dev_memory = []
meas_correlated_gaussian_memory = []
meas_truncated_gaussian_memory = []

measurements = []
rng = np.random.default_rng(1)
correlated_noise_interval = [] #list(range(400, 500))
alternative_noise_interval = list(range(700, 800))

for truth in truths:
    for k, state in enumerate(truth):

        # Disturb the measurement model based on the disturbance modes
        gt_measurement_model = disturbance_measurement_noise(gt_measurement_model, gt_measurement_configs, k)
        trunc_sigma = 0.0
        if k in alternative_noise_interval and activate_disturbances:
            measurement = gt_measurement_model.function(state, noise=False)
            R_true = np.asarray(gt_measurement_model.noise_covar, dtype=float)
            # v = sample_uniform_noise_from_cov(R_true, rng, spread_factor=(1, 1))
            # v = sample_laplace_noise_from_cov(R_true, rng, spread_factor=1)
            # v = sample_student_t_noise_from_cov(
            #     R_true,
            #     rng,
            #     df=3.0,
            #     spread_factor=2.0,
            # )
            trunc_sigma = 1.0
            v = sample_truncated_gaussian_noise_from_cov(
                R_true,
                rng,
                truncation_sigma=trunc_sigma,
            )

            measurement += v
            meas_truncated_gaussian_memory.append(trunc_sigma)
        elif k in correlated_noise_interval and activate_disturbances:
            measurement = gt_measurement_model.function(state, noise=False)

            R_true = np.asarray(gt_measurement_model.noise_covar, dtype=float)

            v = sample_diagonal_mixture_noise_from_cov(
                R_true,
                rng,
            )

            measurement = measurement + v
            meas_truncated_gaussian_memory.append(trunc_sigma)
        else:
            measurement = gt_measurement_model.function(state, noise=True)
            meas_truncated_gaussian_memory.append(trunc_sigma)
            meas_correlated_gaussian_memory.append(0)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))  # Filter-Messmodell für Update
        meas_std_dev_memory.append(np.sqrt(gt_measurement_model.noise_covar))

# %%
meas_bias_memory = []
x_bias = -10
y_bias = -10
# for i, measurement in enumerate(measurements):
#     if 600 <= i <= 650 and activate_disturbances:
#         measurement.state_vector += np.array([[x_bias], [y_bias]])
#         meas_bias_memory.append((x_bias, y_bias))
#     else:
#         meas_bias_memory.append((0, 0))
# %%
# Plot the result, again mapping the x and y position values
plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# ConstantTurn ist nichtlinear → UKF (Unscented Kalman Filter) verwenden
# entspricht Griebel Table 4.1: "Filter type: UKF"
from stonesoup.predictor.kalman import UnscentedKalmanPredictor, KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater, KalmanUpdater, ExtendedKalmanUpdater

use_unscented = True
if use_ct_model:
    if use_unscented:
        predictor = UnscentedKalmanPredictor(transition_model)
        updater   = UnscentedKalmanUpdater(measurement_model)
    else:
        predictor = ExtendedKalmanPredictor(transition_model)
        updater   = ExtendedKalmanUpdater(measurement_model)
else:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

# %%
# Prior für CTRV/UKF: Zustandsvektor [x, vx, y, vy, omega]  (ndim=5)
# omega=0: Startet mit Annahme Geradeausfahrt, wird mitgeschätzt
from stonesoup.types.state import GaussianState

prior = GaussianState(
    StateVector([[0], [5], [0], [5], [0]]) if use_ct_model else StateVector([[0], [5], [0], [5]]),         # omega initial = 0
    CovarianceMatrix(np.diag([0.5, 1, 0.5, 1, 0.1])) if use_ct_model else CovarianceMatrix(np.diag([0.5, 1, 0.5, 1])),
    timestamp=start_time
)


# %%
# Construct a Self-Assessor for the Kalman Filter

from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
from stonesoup.subjective_logic.subjective_logic import BiOpinion, fusion_weighted_belief
from stonesoup.selfassessor._threshold import calc_threshold_op_diff, calc_threshold_n_diff
# Self-assessor settings
sa_settings = {
    "num_X": 7,
    "n_st": 35,
    "n_c": 1,
    "dim_meas": measurement_model.ndim_meas,
    "alpha_threshold_dc": 0.01,
    "trust_discount": 0.99,
    # "type_compar" : 'elementwise'
}
selfassessor = KalmanSelfAssessor(num_X=sa_settings["num_X"],
                                  n_st=sa_settings["n_st"],
                                  n_c=sa_settings["n_c"],
                                  dim_meas=sa_settings["dim_meas"],
                                  alpha_threshold_dc=sa_settings["alpha_threshold_dc"],
                                  trust_discount=sa_settings["trust_discount"],
                                  # type_compar="elementwise"
                                  # threshold_ltst=0.267,
                                  )
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

def calibrate_entropy_threshold(W, n_s, alpha=0.05, N=10000):
    vals = []
    for _ in range(N):
        # sample uniform multinomial
        counts = np.random.multinomial(n_s, [1/W]*W)
        p = counts / np.sum(counts)

        H = -np.sum(p * np.log2(p + 1e-15))
        H /= np.log2(W)
        E = 1 - H

        vals.append(E)

    return np.quantile(vals, 1 - alpha)

def calibrate_opinion_threshold(W, n_s, alpha=0.05, N=100000):
    vals = []
    op_ref = eval(f"sl.Opinion{W}d")(*([1/W]*W))
    for _ in range(N):
        # sample uniform multinomial
        counts = np.random.multinomial(n_s, [1/W]*W)
        dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(counts)
        op = dist.as_opinion()
        dc = op.degree_of_conflict(op_ref)

        vals.append(dc)
    return np.quantile(vals, 1 - alpha)

def calculate_lt_evidence(W: int, alpha=0.99):
    return int((-(W - 1) + np.sqrt((W-1)**2 + ((4 * W) / (1 - alpha + 1e-12)))) / (2))


# eSLIM++ LTST Buffer
SHORT_WINDOW_SIZE = 35
SHORT_WINDOW_SIZE_RADIAL = 20
W = 7
DISCOUNT = 0.99
DISCOUNT_RADIAL = 0.9958

import json
with open("entropy_threshold.json", 'r') as f:
    entropy_thresholds = json.load(f)
with open("opinion_threshold_smoothed.json", 'r') as f:
    opinion_thresholds = json.load(f)
with open("adjusted_opinion_threshold_smoothed.json", 'r') as f:
    adjusted_opinion_thresholds = json.load(f)

griebel_threshold = calc_threshold_n_diff(W, SHORT_WINDOW_SIZE, 0.1)
print("Griebels threshold:", griebel_threshold)
# THRESHOLD = calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT), 0.01)
THRESHOLD = opinion_thresholds[f"{W}, {calculate_lt_evidence(W, DISCOUNT)}, 0.01"] #calibrate_opinion_threshold(W, calculate_lt_evidence(W, DISCOUNT), 0.005)
THRESHOLD_RADIAL = opinion_thresholds[f"{W}, {calculate_lt_evidence(W, DISCOUNT_RADIAL)}, 0.01"]
print("Threshold for LTST:", THRESHOLD)
print("Threshold for RADIAL:", THRESHOLD_RADIAL)
# FUSION_TYPE = sl.FusionType.AVERAGE
FUSION_TYPE = sl.FusionType.CUMULATIVE
# FUSION_TYPE = sl.FusionType.WEIGHTED
HANDLE_ST_CONFLICT = False
AVG_DC_CONFLICT_HANDLING = False

ltst = eval(f"sl.LongShortTermMemory{W}d")(
    SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, AVG_DC_CONFLICT_HANDLING
)
ltst_component_x = eval(f"sl.LongShortTermMemory{W}d")(
    SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, AVG_DC_CONFLICT_HANDLING
)
ltst_component_y = eval(f"sl.LongShortTermMemory{W}d")(
    SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, AVG_DC_CONFLICT_HANDLING
)
component_x_buffered = []
component_y_buffered = []

ltst_whiteness_x = eval(f"sl.LongShortTermMemory{W}d")(
    SHORT_WINDOW_SIZE,
    THRESHOLD,
    DISCOUNT,
    FUSION_TYPE,
    HANDLE_ST_CONFLICT,
    AVG_DC_CONFLICT_HANDLING
)

ltst_whiteness_y = eval(f"sl.LongShortTermMemory{W}d")(
    SHORT_WINDOW_SIZE,
    THRESHOLD,
    DISCOUNT,
    FUSION_TYPE,
    HANDLE_ST_CONFLICT,
    AVG_DC_CONFLICT_HANDLING
)

whiteness_x_buffered = []
whiteness_y_buffered = []

ops = []
ops_per_timestep = []

# Without LTST Buffer
radial_window = []

import scipy.linalg
from datetime import timedelta
def compute_stationary_kf_quantities(transition_model, measurement_model, prior):
    """
    Berechnet stationäre KF-Größen für lineares zeitinvariantes Modell:
      P_inf, K_inf, S_inf, Sigma_eta_inf, mu_R, sigma_R

    Erwartet:
      transition_model.matrix()
      transition_model.covar()
      measurement_model.matrix()
      measurement_model.covar()
    """


    dt = timedelta(seconds=.1)
    F = np.asarray(transition_model.matrix(time_interval = dt), dtype=float)
    Q = np.asarray(transition_model.covar(time_interval = dt), dtype=float)
    H = np.asarray(measurement_model.matrix(time_interval = dt), dtype=float)
    R = np.asarray(measurement_model.covar(time_interval = dt), dtype=float)

    n = F.shape[0]
    m = H.shape[0]

    # Diskrete algebraische Riccati-Gleichung:
    # P = FPF^T - FPH^T(HPH^T+R)^-1HPF^T + Q
    #
    # scipy solve_discrete_are löst:
    # A^T X A - X - A^T X B (R + B^T X B)^-1 B^T X A + Q = 0
    #
    # Dazu setzen wir:
    # A = F^T, B = H^T, Q_dare = Q, R_dare = R
    #
    # Praktisch funktioniert für Kalman oft direkt:
    P_inf = scipy.linalg.solve_discrete_are(
        a=F.T,
        b=H.T,
        q=Q,
        r=R
    )

    S_inf = H @ P_inf @ H.T + R
    K_inf = P_inf @ H.T @ np.linalg.inv(S_inf)

    I_m = np.eye(m)
    Sigma_eta_inf = (I_m - H @ K_inf) @ S_inf @ (I_m - H @ K_inf).T

    R_inv = np.linalg.inv(R)
    B = R_inv @ Sigma_eta_inf

    mu_R = np.trace(B)
    sigma_R = np.sqrt(2.0 * np.trace(B @ B) + 1e-12)

    return {
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "P_inf": P_inf,
        "S_inf": S_inf,
        "K_inf": K_inf,
        "Sigma_eta_inf": Sigma_eta_inf,
        "R_inv": R_inv,
        "mu_R": mu_R,
        "sigma_R": sigma_R,
    }

def scalar_u_to_opinion(u: float, W: int, scale=1):
    """
    Map a scalar u in [0,1] to a W-dimensional one-hot evidence opinion.
    """
    if u == -1:
        return eval(f"sl.Opinion{W}d")(*([0]*W))

    u = float(np.clip(u, 0.0, 1.0))
    evidence = np.zeros(W, dtype=float)

    # Bin index in {0, ..., W-1}
    idx = min(int(np.floor(u * W)), W - 1)
    evidence[idx] = 1.0 * scale

    dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(evidence)
    return dist.as_opinion()

def multinomial_opinion_to_binomial_ok_opinion(op, W, prior_ok=0.5, eps=1e-12):
    """
    Maps a W-dimensional multinomial opinion to a binomial OK opinion.

    H = "component/model is consistent"

    Returns:
        op_ok: binomial opinion with
            b = OK belief
            d = alarm / inconsistency disbelief
            u = original uncertainty
    """
    u = op.uncertainty()
    c = 1.0 - u

    if c <= eps:
        op_ok = sl.Opinion2d(0.0, 0.0)
        op_ok.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
        return op_ok

    a = np.ones(W) / W
    b = np.asarray(op.belief_masses, dtype=float)

    # evidential distribution
    b_tilde = b / c

    # normalized TV distance
    tv = 0.5 * np.sum(np.abs(b_tilde - a))
    tv_max = 1.0 - 1.0 / W

    d_alarm = c * tv / tv_max
    d_alarm = np.clip(d_alarm, 0.0, c)

    b_ok = c - d_alarm

    op_ok = sl.Opinion2d(b_ok, d_alarm)
    op_ok.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
    return op_ok

def bar_shalom_realtime_whiteness_pit(
    innovation_buffer,
    lag=1,
    eps=1e-12,
):
    """
    Bar-Shalom real-time single-run whiteness statistic,
    mapped via PIT to Uniform(0,1).

    For component ell and lag j:

        rho_ell(j) =
            sum_k nu_ell(k) nu_ell(k+j)
            /
            sqrt(
                sum_k nu_ell(k)^2
                *
                sum_k nu_ell(k+j)^2
            )

    Under H0:

        rho_ell(j) ~ N(0, 1/K)

    Therefore:

        u_ell(j) = Phi(rho_ell(j); 0, 1/K)
                 = Phi(sqrt(K) * rho_ell(j))

    is approximately Uniform(0,1).
    """
    innovation_buffer = np.asarray(innovation_buffer, dtype=float)

    if innovation_buffer.ndim != 2:
        raise ValueError("innovation_buffer must have shape (K + lag, m).")

    if lag <= 0:
        raise ValueError("lag must be positive.")

    L, m = innovation_buffer.shape

    if L <= lag:
        return {
            "rho": np.full(m, np.nan),
            "z": np.full(m, np.nan),
            "u": np.full(m, np.nan),
            "K": 0,
        }

    # Number of products
    K = L - lag

    # nu(k), k = 1,...,K
    x0 = innovation_buffer[:K, :]

    # nu(k+j), k = 1,...,K
    xj = innovation_buffer[lag:lag + K, :]

    numerator = np.sum(x0 * xj, axis=0)

    denominator = (
        np.sqrt(np.sum(x0**2, axis=0))
        * np.sqrt(np.sum(xj**2, axis=0))
        + eps
    )

    rho = numerator / denominator

    # Standardized statistic under H0
    z = np.sqrt(K) * rho

    # PIT under z ~ N(0,1)
    u = norm.cdf(z)
    u = np.clip(u, 0.0, 1.0)

    return {
        "rho": rho,
        "z": z,
        "u": u,
        "K": K,
    }
# %%
from stonesoup.types.hypothesis import SingleHypothesis

# %%
import subjective_logic as sl
from subjective_logic.draw_sl_opinions import *

# %%
from stonesoup.types.track import Track
from stonesoup.types.groundtruth import GroundTruthPath
from scipy.stats import chi2, beta
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg

from math import sqrt, ceil

track = Track()
alphas = np.array([1.0, 1.0])

m = len(measurement_model.mapping)
assert m == 2
evidence_per_step = 1.0
opinions = []

op_griebel = []
dirichlet_pdfs = []
x_pdf = np.linspace(0.001, 0.999, 500)


M = W #int(ceil(sqrt(1/(1 - forget_param))))  # Anzahl Bins, 8
# print("# of Bins:", M)
bin_edges = np.linspace(0.0, 1.0, M + 1)
# counts = np.ones(M) * (1/M)
# counts = np.ones(2) * (1/2)
# print("counts:", counts)
# buffer für u_k
u_buffer = []
u_r_buffer = []
c_buffer = []
c_buffer_len = 10
max_buffer = SHORT_WINDOW_SIZE # M **2

# window_size = 50  # optional (rolling window)
u_history = []
C_history = []
K_history = []
eta_history = []
nu_white_history = []
nu_white_full_history = []
counts_history = []


kl_C_history = []

belief_history = []
disbelief_history = []
uncertainty_history = []

e_beta_history = []

fused_op_obj_history = []


dc_history = []

h3_ng_op_history = []
global_op_history = []
fused_2_op_obj_history = []


from ordered_set import OrderedSet
tracks = set([Track([])])
# truths = set([GroundTruthPath([truth])])
kl_opinion = sl.Opinion(0, 0)

# ------------------------------------------------------------------
# Additional diagnostic channels
# ------------------------------------------------------------------
from scipy.stats import norm, chi2
from collections import deque


# Per-timestep opinions for new channels
ops_component_x = []
ops_component_y = []

# ------------------------------------------------------------------
# Bar-Shalom Real-Time Single-Run Whiteness Channel
# ------------------------------------------------------------------
WHITENESS_K = 35       # number of products K in Eq. (5.4.2-15)
WHITENESS_LAG = 1      # j = 1

# Need K + lag samples to compute K products.
innovation_barshalom_buffer = []

ops_whiteness_x = []
ops_whiteness_y = []

whiteness_rho_x_history = []
whiteness_rho_y_history = []
whiteness_z_x_history = []
whiteness_z_y_history = []
whiteness_p_x_history = []
whiteness_p_y_history = []
whiteness_accept_x_history = []
whiteness_accept_y_history = []
whiteness_bound_history = []

from griebels_methods.binomial_hypothesis import GriebelBinomialOpinion, GriebelInnovationTest
griebel_inno = GriebelInnovationTest(
    dim_meas=measurement_model.ndim_meas,
    alpha=0.05,
    window_length=35,
    two_sided=True,
    mapping=measurement_model.mapping,
)
griebel_inno_window = []
griebel_p_ok_history = []

if not use_ct_model:
    stationary = compute_stationary_kf_quantities(
        transition_model=transition_model,
        measurement_model=stationary_measurement_model,
        prior=prior
    )

    F_h2 = stationary["F"]
    Q_h2 = stationary["Q"]
    H_h2 = stationary["H"]
    R_h2 = stationary["R"]
    m_h2 = H_h2.shape[0]
    K_inf = stationary["K_inf"]
    print("Stationary Kalman Gain:\n", K_inf)

for i, measurement in enumerate(measurements):
    prediction: GaussianStatePrediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    # if i == 1:
    #     print(post)
    prior = track[-1]
    for t in tracks:
        t.append(post)

    dx_update = (post.state_vector - prediction.state_vector).reshape(-1)

    # self-assessment
    z_p = hypothesis.measurement_prediction.mean
    S_p = hypothesis.measurement_prediction.covar
    selfassessor.assess(z_p, S_p, measurement.state_vector)
    selfassessor_measures = selfassessor.get_sas_measures()
    nis_measures = nis.assess(z_p, S_p, measurement.state_vector)
    selfassessor_measures_history.append(selfassessor_measures)
    nis_measures_history.append(nis_measures)
    op_griebel.append(selfassessor._op_st)

    # -----------------------------
    # Likelihood → Evidence
    # -----------------------------
    meas_pred = hypothesis.measurement_prediction.mean
    delta = (measurement.state_vector - meas_pred).copy()
    S = hypothesis.measurement_prediction.covar.copy()
    delta = delta.reshape(-1, 1)            # Innovation

    P = hypothesis.prediction.covar
    H = measurement_model.matrix()
    K = P @ H.T @ np.linalg.inv(S)
    K_history.append(K)

    z = measurement.state_vector
    z_pred = hypothesis.measurement_prediction.mean
    S = hypothesis.measurement_prediction.covar

    griebel_inno_result = griebel_inno.assess(z, z_pred, S)
    griebel_inno_score = griebel_inno_result["score"]
    griebel_inno_accept = griebel_inno_result["accept_h0"]
    griebel_inno_belief = griebel_inno_result["opinion"]["belief"]
    griebel_inno_disbelief = griebel_inno_result["opinion"]["disbelief"]
    griebel_inno_uncertainty = griebel_inno_result["opinion"]["uncertainty"]
    griebel_evidence = np.array([int(griebel_inno_accept), 1-int(griebel_inno_accept)])
    griebel_dist = eval(f"sl.DirichletDistribution{2}d").from_evidences(griebel_evidence)
    griebel_op = griebel_dist.as_opinion()
    # griebel_op = sl.Opinion(griebel_inno_belief, griebel_inno_disbelief)
    griebel_op.prior_belief_masses =  [0.95, 0.05]
    griebel_inno_window.append(griebel_op)
    if len(griebel_inno_window) > griebel_inno.window_length:
        griebel_inno_window.pop(0)
    griebel_results_opinion = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, griebel_inno_window)
    # print(griebel_results_opinion)
    griebel_p_ok_history.append(griebel_results_opinion.getProjection()[0])

    fused_2_op_obj_history.append(griebel_results_opinion)

    eta = (measurement.state_vector.reshape(-1, 1) - H @ post.state_vector).flatten()
    eta_history.append(eta)
    if len(eta_history) > max_buffer:
        eta_history.pop(0)


    # -----------------------------
    # Whitening
    # -----------------------------
    # Robuste Cholesky-Zerlegung: beim EKF kann S durch die Jacobi-Approximation
    # numerisch nicht-positiv-definit werden. Fallback-Kette:
    #   1) Standard Cholesky
    #   2) Symmetrisierung + Jitter (Tikhonov-Regularisierung)
    #   3) Eigenwert-Clipping (alle negativen Eigenwerte → kleines ε)
    def robust_cholesky(M, max_jitter=1e-3):
        M = (M + M.T) / 2  # Symmetrisieren (floating-point drift)
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            pass
        # Jitter-Strategie: schrittweise epsilon aufaddieren
        eps = 1e-9
        while eps <= max_jitter:
            try:
                return np.linalg.cholesky(M + eps * np.eye(M.shape[0]))
            except np.linalg.LinAlgError:
                eps *= 10
        # Eigenwert-Clipping als letzter Fallback
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-8)
        M_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.cholesky(M_fixed)


    S_sqrt = robust_cholesky(S)
    nu_white = np.linalg.solve(S_sqrt, delta).flatten()
    nu_white_full_history.append(nu_white.copy())

    nu_white_history.append(nu_white.copy())
    if len(nu_white_history) > max_buffer:
        nu_white_history.pop(0)
    # ------------------------------------------------------------------
    # Bar-Shalom real-time single-run whiteness test
    #
    # Uses the non-whitened innovation nu_k = z_k - z_hat_{k|k-1},
    # not the whitened innovation.
    # ------------------------------------------------------------------
    innovation_barshalom_buffer.append(delta.flatten().copy())

    max_white_buffer_len = WHITENESS_K + WHITENESS_LAG

    if len(innovation_barshalom_buffer) > max_white_buffer_len:
        innovation_barshalom_buffer.pop(0)

    if len(innovation_barshalom_buffer) >= max_white_buffer_len:
        white_res = bar_shalom_realtime_whiteness_pit(
            np.asarray(innovation_barshalom_buffer),
            lag=WHITENESS_LAG,
        )

        rho_white = white_res["rho"]
        z_white = white_res["z"]
        u_white = white_res["u"]

        u_white_x = float(u_white[0])
        u_white_y = float(u_white[1])

        op_white_x = scalar_u_to_opinion(u_white_x, W)
        op_white_y = scalar_u_to_opinion(u_white_y, W)
        whiteness_p_x_history.append(u_white_x)
        whiteness_p_y_history.append(u_white_y)

        whiteness_rho_x_history.append(float(rho_white[0]))
        whiteness_rho_y_history.append(float(rho_white[1]))
        whiteness_z_x_history.append(float(z_white[0]))
        whiteness_z_y_history.append(float(z_white[1]))

    else:
        op_white_x = scalar_u_to_opinion(-1, W)
        op_white_y = scalar_u_to_opinion(-1, W)

        whiteness_rho_x_history.append(np.nan)
        whiteness_rho_y_history.append(np.nan)
        whiteness_z_x_history.append(np.nan)
        whiteness_z_y_history.append(np.nan)
        whiteness_p_x_history.append(np.nan)
        whiteness_p_y_history.append(np.nan)

    ops_whiteness_x.append(op_white_x)
    ops_whiteness_y.append(op_white_y)

    # # -----------------------------
    # # Whitening
    # # -----------------------------
    # S_sqrt = np.linalg.cholesky(S)
    # nu_white = np.linalg.solve(S_sqrt, delta).flatten()
    # nu_white_history.append(nu_white.copy())
    # if len(nu_white_history) > max_buffer:
    #     nu_white_history.pop(0)

    # # ------------------------------------------------------------------
    # # 1) Store current whitened innovation in bounded rolling buffers
    # # ------------------------------------------------------------------
    # nu_white_x_buffer.append(float(nu_white[0]))
    # nu_white_y_buffer.append(float(nu_white[1]))

    # ------------------------------------------------------------------
    # 2) Component-wise Gaussian consistency opinion
    # H0: each whitened component ~ N(0,1)
    # PIT: u = Phi(nu_white_j)
    # ------------------------------------------------------------------
    u_comp_x = norm.cdf(float(nu_white[0]))
    u_comp_y = norm.cdf(float(nu_white[1]))

    op_comp_x = scalar_u_to_opinion(u_comp_x, W)
    op_comp_y = scalar_u_to_opinion(u_comp_y, W)

    ops_component_x.append(op_comp_x)
    ops_component_y.append(op_comp_y)

    # # ------------------------------------------------------------------
    # # Q-like lag-1 dynamic residual correlation test
    # # direct statistical channel + gain-based impact
    # # ------------------------------------------------------------------
    # if len(nu_white_history) >= QTEST_WINDOW and (i % QTEST_STRIDE == 0):
    #     rho_q, T_q, p_q, _ = q_like_lag1_test(
    #         np.asarray(nu_white_history, dtype=float),
    #         ridge=1e-6
    #     )
    # elif len(rho_qtest_history) > 0:
    #     # hold last value between stride updates
    #     rho_q = rho_qtest_history[-1]
    #     T_q = T_qtest_history[-1]
    #     p_q = p_qtest_history[-1]
    # else:
    #     rho_q, T_q, p_q = 0.0, 0.0, -1
    #
    # rho_qtest_history.append(rho_q)
    # T_qtest_history.append(T_q)
    # p_qtest_history.append(p_q)
    #
    # # ------------------------------------------------------------------
    # # Q-like PIT -> multinomial one-step opinion
    # # p_q is already approximately uniform under H0.
    # # ------------------------------------------------------------------
    # op_qtest = scalar_u_to_opinion(p_q, W)#, scale=len(nu_white_history)//3)
    # ops_qtest.append(op_qtest)

    d2 = (delta.T @ np.linalg.inv(S) @ delta).item()
    u = chi2.cdf(d2, df=m)
    u_history.append(u)
    u_buffer.append(u)

    if len(u_buffer) > max_buffer:
        u_buffer.pop(0)
        # u_buffer = [] #.pop(0)
        # u_buffer.append(u)

    # print(len(u_buffer))

    counts = np.zeros(M) #* (1/M)
    evidence = np.zeros(M)


    for u_i in u_buffer:
        idx = np.searchsorted(bin_edges, u_i, side='right') - 1
        idx = np.clip(idx, 0, M - 1)
        counts[idx] += 1

    evidence_idx = np.searchsorted(bin_edges, u, side='right') - 1
    evidence_idx = np.clip(evidence_idx, 0, M - 1)
    evidence[evidence_idx] += 1

    one_time_dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(evidence)
    one_time_op = one_time_dist.as_opinion()
    radial_window.append(one_time_op)
    ops_per_timestep.append(one_time_op)

    counts_history.append(list(counts))

    opinions.append((kl_opinion.belief(), kl_opinion.disbelief(), kl_opinion.uncertainty()))

    counts_R_history = counts_history.copy()

    # ---------------------------------
    # H3: non-Gaussian innovation statistics
    # ---------------------------------
    # h3_opinion = sl.Fusion.fuse_opinions(
    #     sl.FusionType.AVERAGE,
    #     opinion,
    #     ad_opinion
    # )
    # h3_opinion = kl_opinion.wb_fuse(ad_opinion)
    h3_opinion = kl_opinion
    h3_ng_op_history.append(h3_opinion)

    fused_op_obj_history.append(h3_opinion)   # optional: keep old history for plotting


    # ---------------------------------
    # Global opinion from five hypotheses
    # ---------------------------------
    # global_opinion = sl.Fusion.fuse_opinions(
    #     sl.FusionType.AVERAGE,
    #     h1_opinion,   # H1: R wrong
    #     h2_opinion,   # H2: Q / dynamics wrong
    #     h3_opinion,   # H3: non-Gaussian
    #     h4_opinion,   # H4: bias
    #     h5_opinion    # H5: whiteness violated
    # )
    # global_op_history.append(global_opinion)

    h_opinions = [
                  # h1_opinion,
                  # h2_opinion,
                  h3_opinion,
                  # h4_opinion,
                  # h5_opinion
    ]
    w_all = []
    for h in h_opinions:
        w_all.append(BiOpinion(h.belief(), h.disbelief(), h.prior_belief(), h.uncertainty()))

    # global_opinion_ss = fusion_weighted_belief(w_all)
    # global_opinion = sl.Opinion(global_opinion_ss.belief[0], global_opinion_ss.belief[1])
    # global_opinion = sl.Fusion.fuse_opinions(sl.FusionType.AVERAGE, h_opinions)
    # global_opinion = h_opinions[0]
    # global_op_history.append(global_opinion)

    # belief_history.append(global_opinion.belief())
    # disbelief_history.append(global_opinion.disbelief())
    # uncertainty_history.append(global_opinion.uncertainty())
    belief_history.append(kl_opinion.belief())
    disbelief_history.append(kl_opinion.disbelief())
    uncertainty_history.append(kl_opinion.uncertainty())


def compute_position_error_single_run(track, truth, mapping=(0, 2)):
    truth_by_time = {
        state.timestamp: state
        for state in truth
    }

    errors = []
    timestamps = []

    for estimate in track:
        gt_state = truth_by_time.get(estimate.timestamp)
        if gt_state is None:
            continue

        x_est = np.asarray(estimate.state_vector, dtype=float).reshape(-1)
        x_gt = np.asarray(gt_state.state_vector, dtype=float).reshape(-1)

        e = x_est[list(mapping)] - x_gt[list(mapping)]

        errors.append(e)
        timestamps.append(estimate.timestamp)

    errors = np.asarray(errors)

    pos_error = np.sqrt(np.sum(errors**2, axis=1))

    return {
        "timestamps": timestamps,
        "errors_xy": errors,
        "position_error": pos_error,
    }


rmse_single = compute_position_error_single_run(
    track=track,
    truth=truth,
    mapping=(0, 2),
)

plt.figure()
plt.plot(rmse_single["position_error"], label=r"$\| \hat p_k - p_k \|_2$")
plt.grid()
plt.legend()
plt.xlabel("time step")
plt.ylabel("position error [m]")
plt.title("Single-run position error")
# %%
import tqdm
ops = ops_per_timestep
inp = np.zeros((len(ops), 2))
buffer = np.zeros((len(ops), 2))
st_buffer = np.zeros((len(ops), 2))
lt_buffer = np.zeros((len(ops), 2))
resets = np.zeros(len(ops))
dc_ref = np.zeros(len(ops))
dc_ltst = np.zeros(len(ops))
th_dc = np.zeros(len(ops))

op_ref = eval(f"sl.Opinion{W}d")(*([1/W]*W))
radial_window_op = eval(f"sl.Opinion{W}d")(list(np.zeros(W)))

radial_window_dc_history = []
p_ok_exp = []
entropy_opinions = []
direct_entropy_opinions = []
conceivability_cross_entropy = []
sums_of_evidence = []
sums_of_evidence_comp = []
sums_of_evidence_x = []
sums_of_evidence_y = []
evidences = []
buffer_uncertainties =[]
buffered_ops = []

for idx, op_obs in tqdm.tqdm(enumerate(ops_per_timestep), total=len(ops_per_timestep)):
    ltst.add(op_obs)
    op_buffer = ltst.get_opinion()
    # if 500 <= idx <=550:
    #     buffer_u = op_buffer.uncertainty()
    #     op_buffer = sl.Opinion(*([(1 - buffer_u)/W]*W))
    st_op = ltst.get_short_opinion()
    lt_op = ltst.get_long_opinion()
    conflict = ltst.get_conflicted_pair()
    if ltst.is_last_conflicted():
        _, lt_op_prior = conflict
    else:
        lt_op_prior = lt_op
    # print("idx:", idx)
    # print("len of st:", ltst.get_short_size())
    # print("len of lt:", ltst.get_long_size())

    # write results to buffers
    # inp[idx, :] = (op_obs.getBinomialProjection(), op_obs.uncertainty())
    buffer[idx, :] = (op_buffer.getProjection()[0], op_buffer.uncertainty())
    st_buffer[idx, :] = (st_op.getProjection()[0], st_op.uncertainty())
    lt_buffer[idx, :] = (lt_op.getProjection()[0], lt_op.uncertainty())
    resets[idx] = ltst.is_last_conflicted()
    dc_ref[idx] = op_buffer.degree_of_conflict(op_ref)
    dc_ltst[idx] = st_op.degree_of_conflict(lt_op_prior)
    buffered_ops.append(op_buffer)

    th_dc[idx] = calc_threshold_n_diff(W, sum(op_buffer.as_dirichlet().evidences), 0.1)
    # print("Sum of Evidence:", sum(op_buffer.as_dirichlet().evidences))
    sums_of_evidence.append(int(sum(op_buffer.as_dirichlet().evidences)))
    evidences.append(list(op_buffer.as_dirichlet().evidences))

    buffer_uncertainties.append(op_buffer.uncertainty())

    entropy = -1 * np.sum(op_buffer.getProjection() * np.log2(op_buffer.getProjection()))
    entropy /= np.log2(W)
    entropy = 1 - entropy
    entropy_opinions.append(entropy)

    eps = 1e-12
    con_cross_ent = -1 * np.sum(
        op_ref.getProjection()
        * np.log2(op_buffer.belief_masses +eps) # + op_buffer.uncertainty())
    )
    Hb_max = -((W - 1) / W) * np.log2(eps)
    con_cross_ent /= Hb_max

    u_min = W / (W + SHORT_WINDOW_SIZE + calculate_lt_evidence(W, 0.99))

    Hc_max = -((W - 1) / W) * np.log2(u_min)

    con_cross_ent_norm = 1 - con_cross_ent / Hc_max

    R = sum(op_buffer.as_dirichlet().evidences)
    u_current = W / (W + R)
    Hc_max_current = -((W - 1) / W) * np.log2(u_current)
    compatibility = con_cross_ent / Hc_max_current

    conceivability_cross_entropy.append(con_cross_ent)
    # con_cross_ent = -1 * np.sum(op_ref.getProjection()  * (np.log2(op_buffer.belief_masses + op_buffer.uncertainty())))
    # u_min = W / (W + SHORT_WINDOW_SIZE + calculate_lt_evidence(W, DISCOUNT))
    # Hc_max = -((W - 1) / W) * np.log2(u_min)
    # print(Hc_max)
    # con_cross_ent /= Hc_max
    # con_cross_ent = 1 - con_cross_ent
    # conceivability_cross_entropy.append(con_cross_ent)

    evid = op_buffer.as_dirichlet().evidences + 1e-12
    evid_sum = sum(evid)
    evid /= evid_sum
    direct_entropy = -1 * np.sum(evid * np.log2(evid))
    direct_entropy /= np.log2(len(evid))
    direct_entropy = 1 - direct_entropy
    direct_entropy_opinions.append(direct_entropy)

    # without any buffer
    radial_window_op = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, [radial_window_op, radial_window[idx]])
    radial_window_op = radial_window_op.trust_discount(0.9976)
    radial_window_dc = radial_window_op.degree_of_conflict(op_ref)

    u = radial_window_op.uncertainty()
    c = 1.0 - u
    a = np.ones(W) / W

    if c <= 1e-9:
        radial_window_dc_history.append(np.nan)
        continue

    b = np.array(radial_window_op.belief_masses)
    b_tilde = b / c

    # L1 / adjusted TV
    tv = 0.5 * np.sum(np.abs(b_tilde - a))
    tv_max = 1.0 - 1.0 / W

    radial_window_op_norm = c * tv / tv_max
    radial_window_dc_history.append(radial_window_op_norm)
    op_exp_bin = multinomial_opinion_to_binomial_ok_opinion(radial_window_op, W)
    p_ok_exp.append(op_exp_bin.getProjection()[0])
    # print(radial_window_op)
    # print(sum(radial_window_op.as_dirichlet().evidences))


for opx, opy in zip(
    ops_component_x, ops_component_y,
):
    ltst_component_x.add(opx)
    ltst_component_y.add(opy)


    component_x_buffered.append(ltst_component_x.get_opinion())
    component_y_buffered.append(ltst_component_y.get_opinion())
    sums_of_evidence_x.append(int(sum(ltst_component_x.get_opinion().as_dirichlet().evidences)))
    sums_of_evidence_y.append(int(sum(ltst_component_y.get_opinion().as_dirichlet().evidences)))

for opwx, opwy in zip(ops_whiteness_x, ops_whiteness_y):
    ltst_whiteness_x.add(opwx)
    ltst_whiteness_y.add(opwy)

    whiteness_x_buffered.append(ltst_whiteness_x.get_opinion())
    whiteness_y_buffered.append(ltst_whiteness_y.get_opinion())

# dc_white_x = [white_x.degree_of_conflict(op_ref) for white_x in whiteness_x_buffered]
# dc_white_y = [white_y.degree_of_conflict(op_ref) for white_y in whiteness_y_buffered]
dc_comp_x = [comp_x.degree_of_conflict(op_ref) for comp_x in component_x_buffered]
dc_comp_y = [comp_y.degree_of_conflict(op_ref) for comp_y in component_y_buffered]
component_buffered = []


for opx, opy in zip(component_x_buffered, component_y_buffered):
    fused_comp = sl.Fusion.fuse_opinions(sl.FusionType.BELIEF_CONSTRAINT, [opx, opy])
    component_buffered.append(fused_comp)
    sums_of_evidence_comp.append(int(sum(fused_comp.as_dirichlet().evidences)))

dc_comp = [comp.degree_of_conflict(op_ref) for comp in component_buffered]


eps = 1e-12
a = np.ones(W) / W

dc_adj_l1 = []
dc_adj_l2 = []
dc_adj_kl = []

for i, op in enumerate(buffered_ops):  # z.B. op_buffer history speichern
    u = op.uncertainty()
    c = 1.0 - u

    if c <= 1e-9:
        dc_adj_l1.append(np.nan)
        dc_adj_l2.append(np.nan)
        dc_adj_kl.append(np.nan)
        global_op_history.append(sl.Opinion2d(0, 0))
        # fused_2_op_obj_history.append(sl.Opinion2d(1 - u - dc_ref[i], dc_ref[i]))
        continue

    b = np.array(op.belief_masses)
    b_tilde = b / c

    # L1 / adjusted TV
    # dc_adj_l1.append(c * 0.5 * np.sum(np.abs(b_tilde - a)))
    tv = 0.5 * np.sum(np.abs(b_tilde - a))
    tv_max = 1.0 - 1.0 / W

    dc_l1_norm = c * tv / tv_max
    # dc_l1_norm = np.clip(dc_l1_norm, 0.0, c)

    dc_adj_l1.append(dc_l1_norm)

    op_l1 = sl.Opinion2d(1.0 - u - dc_l1_norm, dc_l1_norm)
    # op_l1.prior_belief_masses = [0.99, 0.01]
    # L2
    d2_max = np.sqrt(1.0 - 1.0 / W)
    dc_adj_l2.append(c * np.linalg.norm(b_tilde - a) / d2_max)

    # KL
    bt = np.clip(b_tilde, eps, 1.0)
    dc_adj_kl.append(c * np.sum(bt * np.log(bt / a)) / np.log(W))

    # op_l1 = sl.Opinion2d(1 - u - dc_adj_l1[-1], dc_adj_l1[-1])
    # op_l1.prior_belief_masses = [0.99, 0.01]
    global_op_history.append(op_l1)

    # op_dc = sl.Opinion2d(1 - u - dc_ref[i], dc_ref[i])
    # fused_2_op_obj_history.append(op_dc)


component_x_binomial = []
component_y_binomial = []
component_binomial = []
overall_binomial = []

p_ok_x = []
p_ok_y = []
p_ok_comp = []
p_ok_overall = []

whiteness_x_binomial = []
whiteness_y_binomial = []
whiteness_binomial = []

p_ok_white_x = []
p_ok_white_y = []
p_ok_whiteness = []
u_whiteness = []

prior_ok = float(np.sqrt(0.5))

for i, ops in enumerate(zip(component_x_buffered, component_y_buffered)):
    opx, opy = ops
    opx_bin= multinomial_opinion_to_binomial_ok_opinion(opx, W, prior_ok=prior_ok)
    opy_bin = multinomial_opinion_to_binomial_ok_opinion(opy, W, prior_ok=prior_ok)

    component_x_binomial.append(opx_bin)
    component_y_binomial.append(opy_bin)
    component_op = opx_bin.multiply(opy_bin)
    # component_op.prior_belief_masses = [0.5, 0.5]
    # component_binomial.append(sl.Fusion.fuse_opinions(sl.FusionType.WEIGHTED, [opx_bin, opy_bin]))
    component_binomial.append(component_op)
    # print(i, "Compon:", component_op)
    # print(i, "Radial:", global_op_history[i])
    # overall_op = component_op.multiply(global_op_history[i])
    overall_op = sl.Fusion.fuse_opinions(sl.FusionType.WEIGHTED, [component_op, global_op_history[i]])
    overall_binomial.append(overall_op)
    # print(i, "Overall:", overall_op)

    p_ok_x.append(opx_bin.getProjection()[0]) #opx_bin.belief() + prior_ok * opx_bin.uncertainty())
    p_ok_y.append(opy_bin.getProjection()[0]) #opy_bin.belief() + prior_ok * opy_bin.uncertainty())
    p_ok_comp.append(component_binomial[-1].getProjection()[0])
    p_ok_overall.append(overall_op.getProjection()[0])

prior_white = float(np.sqrt(0.5))

for opwx, opwy in zip(whiteness_x_buffered, whiteness_y_buffered):
    opwx_bin = multinomial_opinion_to_binomial_ok_opinion(
        opwx,
        W,
        prior_ok=prior_white,
    )

    opwy_bin = multinomial_opinion_to_binomial_ok_opinion(
        opwy,
        W,
        prior_ok=prior_white,
    )

    # H_white = H_white_x AND H_white_y
    white_op = opwx_bin.multiply(opwy_bin)

    whiteness_x_binomial.append(opwx_bin)
    whiteness_y_binomial.append(opwy_bin)
    whiteness_binomial.append(white_op)

    p_ok_white_x.append(opwx_bin.getProjection()[0])
    p_ok_white_y.append(opwy_bin.getProjection()[0])
    p_ok_whiteness.append(white_op.getProjection()[0])
    u_whiteness.append(white_op.uncertainty())

# for i, _ in enumerate(dc_st_lt):
#     assert np.isclose(dc_ltst[i], dc_st_lt[i]), f"{i}, {dc_ltst[i]}, {dc_st_lt[i]}"

# plt.figure()
# plt.plot(dc_adj_l1, label="dc_adj_l1")
# plt.plot(dc_adj_l2, label="dc_adj_l2")
# plt.plot(dc_adj_kl, label="dc_adj_kl")
# plt.legend()
# plt.grid()
# plt.title("DC Adjusted")

plt.figure()
plt.plot(model_indices, label="Chosen Model")

plt.yticks(
    [0, 1, 2],
    ["Straight", "Turn right", "Turn left"]
)
plt.grid()
plt.legend()
plt.title("Chosen Model")

plt.figure()
plt.plot([meas_std_dev_memory[i][0, 0] for i, mat in enumerate(meas_std_dev_memory)], label="meas_cov")
plt.plot(meas_truncated_gaussian_memory, label="Truncate Gaussian")
plt.plot([meas_bias_memory[i][0] for i in range(len(meas_bias_memory))], label="bias x")
plt.plot([meas_bias_memory[i][1] for i in range(len(meas_bias_memory))], label="bias y")
plt.plot(model_indices, label="Turn Right")
plt.plot(process_noise_coeff_memory[0], label="q_x")
plt.plot(process_noise_coeff_memory[1], label="q_y")
plt.legend()
plt.grid()
plt.title("Disturbances")

# print(p_ok_whiteness)
plt.figure()
plt.plot(griebel_p_ok_history, label="Griebel Innovation SA")
# plt.plot(griebel_bias.score_history, label="Griebel Bias SA")
# plt.plot(buffer[:, 0], label="Your PIT/LTST Projection")
plt.axhline(0.95, label="0.95")
plt.plot([global_op_history[i].getProjection()[0] for i in range(len(global_op_history))], label="P_OK Radial")
# plt.plot(p_ok_x, label="P_OK Comp X")
# plt.plot(p_ok_y, label="P_OK Comp Y")
plt.plot(p_ok_comp, label="P_OK Comp Fused")
plt.plot(p_ok_overall, label="P_OK Overall")
# plt.plot(p_ok_whiteness, label="P_OK Whiteness")
plt.plot([overall_binomial[i].uncertainty() for i in range(len(overall_binomial))], label="P_OK Uncertainty")
plt.plot([fused_2_op_obj_history[i].uncertainty() for i in range(len(fused_2_op_obj_history))], label="Griebel Uncertainty")
plt.plot([component_binomial[i].uncertainty() for i in range(len(component_binomial))], label="Comp Uncertainty")
# plt.plot(p_ok_exp, label="P_OK Exp")
plt.legend()
plt.grid()
plt.title("Griebel Baseline vs Own Method")

# plt.figure()
# plt.plot(p_ok_white_x, label=r"$P_{OK,\mathrm{white},x}$")
# plt.plot(p_ok_white_y, label=r"$P_{OK,\mathrm{white},y}$")
# plt.plot(p_ok_whiteness, label=r"$P_{OK,\mathrm{white}}$")
# plt.plot(u_whiteness, label=r"$u_{\mathrm{white}}$")
# plt.grid()
# plt.legend()
# plt.title("Bar-Shalom Real-Time Whiteness Channel")
#
# plt.figure()
# plt.plot(whiteness_p_x_history, label=r"$u_{\mathrm{white},x}$")
# plt.plot(whiteness_p_y_history, label=r"$u_{\mathrm{white},y}$")
# plt.grid()
# plt.legend()
# plt.title("Bar-Shalom Whiteness PIT values")
#
# plt.figure()
# plt.plot(whiteness_z_x_history, label=r"$z_{\mathrm{white},x}$")
# plt.plot(whiteness_z_y_history, label=r"$z_{\mathrm{white},y}$")
# plt.axhline(0.0, linestyle="--")
# plt.grid()
# plt.legend()
# plt.title(r"Standardized Bar-Shalom statistic $z=\sqrt{K}\rho$")
#
# plt.figure()
# plt.plot(whiteness_rho_x_history, label="rho_x")
# plt.plot(whiteness_rho_y_history, label="rho_y")
# plt.axhline(0.0, linestyle="--")
# plt.grid()
# plt.legend()
# plt.title(r"Standardized Bar-Shalom statistic rho values")

# plt.figure()
# plt.plot(list(np.linspace(0, 0.5, 500)), list(map(lambda x: calc_threshold_n_diff(5, 58.62, x), list(np.linspace(0, 0.5, 500)))), label='th')
# plt.legend()
# plt.title("Threshold evaluation")
# plt.figure()
# plt.plot(T_qtest_history, label="T_qtest")
# plt.legend()
# plt.grid()
# plt.title("Q-like chi-square statistic")
#
# plt.figure()
# plt.plot(p_qtest_history, label="p_qtest")
# plt.axhline(0.05, color='k', linestyle='--', label='p=0.05')
# plt.axhline(0.01, color='r', linestyle='--', label='p=0.01')
# plt.legend()
# plt.grid()
# plt.title("Q-like p-values")
#
# plt.figure()
# plt.plot(dc_qtest, label="DC Q-like")
# plt.plot(
#     [opinion_thresholds[f"{W}, {min(max(s, 1), 150)}, 0.005"] for s in sums_of_evidence_q],
#     label="Th Q-like"
# )
# plt.legend()
# plt.grid()
# plt.title("Q-like DC")
#
# plt.figure()
# plt.plot(dc_ref, label="DC Radial")
# plt.plot(dc_qtest, label="DC Q-like")
# plt.plot(
#     [opinion_thresholds[f"{W}, {s}, 0.005"] for s in sums_of_evidence],
#     label="Th Radial",
#     alpha=0.7
# )
# plt.plot(
#     [opinion_thresholds[f"{W}, {min(max(s, 1), 150)}, 0.005"] for s in sums_of_evidence_q],
#     label="Th Q-like",
#     alpha=0.7
# )
# plt.legend()
# plt.grid()
# plt.title("Radial vs Q-like DC")

# plt.figure()
# plt.plot(eta_res_ref_history, label="eta ref")
# plt.plot(eta_st_lt_history, label="eta ltst")
# plt.legend()
# plt.title("Dynamic Threshold")

# plt.figure()
# # plt.plot(inp[:, 0], label="Inp")
# # plt.plot(inp[:, 1], label="Inp u")
# plt.plot(buffer[:, 0], label="PP")
# plt.plot(st_buffer[:, 0], label="st")
# plt.plot(lt_buffer[:, 0], label="lt")
# plt.legend()
# plt.title("LTST Buffer")

plt.figure()
plt.plot(buffer[:, 1], label="u")
plt.plot(st_buffer[:, 1], label="st u")
plt.plot(lt_buffer[:, 1], label="lt u")
plt.legend()
plt.title("LTST Uncertainties")


plt.figure()

# plt.plot(resets, label="Reset")
plt.plot(dc_ref, label=r"$DC_J$")
# plt.plot([dc_ref[i]/(1 - buffer_uncertainties[i]) for i in range(len(dc_ref))], label="DC Radial/(1-u)")
dc_adj = []
for dc, u in zip(dc_ref, buffer_uncertainties):
    c = (1.0 - u)**2
    dc_adj.append(dc / c if c > 1e-6 else np.nan)

# plt.plot(dc_adj, label="Adjusted DC Radial")
# plt.plot(buffer_uncertainties, label="Buffer Uncertainty")
# plt.plot(dc_ltst, label="DC LTST")
# plt.plot(entropy_opinions, label="Entropy")
# plt.plot(direct_entropy_opinions, label="Direct Entropy")
# plt.plot(conceivability_cross_entropy, label="Conceivability Cross Entropy")
# plt.plot([calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT)+SHORT_WINDOW_SIZE, alpha=0.01)]*len(entropy_opinions), label = "Entropy Th LT+ST")
# plt.plot([calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT), alpha=0.01)]*len(entropy_opinions), label = "Entropy Th LT")
# plt.plot([entropy_thresholds[f"{W}, {s}, 0.01"] for s in sums_of_evidence], label = "Entropy Th Dynamic")
# plt.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.01"] for s in sums_of_evidence], label = "Th Radial")
plt.plot(dc_adj_l1, label=r"$DC_{\mathrm{norm}}$")
plt.plot(np.array(selfassessor_measures_history)[:, 0], label="Griebel SA measure")
plt.plot(np.array(selfassessor_measures_history)[:, 2], label="Griebel SA Threshold")
# plt.plot([adjusted_opinion_thresholds[f"{W}, {min(s, 150)}, 0.01"] for s in sums_of_evidence], label = "Th L1 DC")
# plt.plot([radial_window_dc_history[i] for i in range(len(radial_window_dc_history))], label="dc exp Opinion")
# plt.plot(dc_adj_l2, label="dc_adj_l2")
# plt.plot(dc_adj_kl, label="dc_adj_kl")
# plt.plot([calc_threshold_n_diff(W, s, 0.1) for s in sums_of_evidence], label = "Griebel Th Dynamic")
# plt.plot([adjusted_opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for i, s in enumerate(sums_of_evidence)], label="Th Norm")
# plt.plot(kl_C_history, label="KL C")
plt.yticks(np.linspace(0, 1, 11))
plt.grid()
plt.legend()
plt.title("DCs")

# plt.figure()
# plt.plot([dc_ref[i] / opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for i, s in enumerate(sums_of_evidence)], label="Radial")
# plt.plot([dc_adj_l1[i] / adjusted_opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for i, s in enumerate(sums_of_evidence)], label="L1 DC")
# plt.legend()
# plt.grid()
# plt.title("Signal")

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# # --- Radial ---
# ax1.plot(dc_ref, label="DC Radial")
# ax1.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for s in sums_of_evidence],
#          label="Th Radial")
# ax1.set_title("Radial")
# ax1.grid()
# ax1.legend()
# # --- Comp ---
# ax2.plot(dc_comp, label="DC Comp")
# ax2.plot([opinion_thresholds[f"{W}, {min(ceil(s/m), 150)}, 0.005"] for s in sums_of_evidence_comp],
#          label="Th Comp")
# ax2.set_title("Comp")
# ax2.grid()
# ax2.legend()
# # Gemeinsamer Titel
# fig.suptitle("DC Global")
# plt.tight_layout()

# print(calculate_lt_evidence(W, DISCOUNT)+SHORT_WINDOW_SIZE)
# plt.figure()
# # plt.plot(dc_ltst, label="DC LTST")
# plt.plot(dc_ref, label="DC Radial")
# plt.plot(dc_comp, label="DC Comp")
# plt.plot([opinion_thresholds[f"{W}, {s}, 0.005"] for s in sums_of_evidence], label = "Th Radial")
# plt.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for s in sums_of_evidence_comp], label="Th Comp")
# # plt.plot(dc_whiteness, label="DC Whiteness")
# plt.legend()
# plt.grid()
# plt.title("DC Global")

plt.figure()
# plt.plot(dc_ltst, label="DC LTST")
# plt.plot(dc_white_x, label="DC White X")
# plt.plot(dc_white_y, label="DC White Y")
# plt.plot(dc_comp_x, label="DC Comp X")
# plt.plot(dc_comp_y, label="DC Comp Y")
# plt.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.01"] for s in sums_of_evidence_x], label = "Th x", color="blue", alpha=0.5)
# plt.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.01"] for s in sums_of_evidence_y], label = "Th y", color="orange", alpha=0.5)
plt.plot(p_ok_x, label="P_OK Comp X")
plt.plot(p_ok_y, label="P_OK Comp Y")
plt.plot([component_x_binomial[i].uncertainty() for i in range(len(component_x_binomial))], label="u_comp X")
plt.plot([component_y_binomial[i].uncertainty() for i in range(len(component_y_binomial))], label="u_comp Y")
plt.legend()
plt.grid()
plt.title("P_OK Local")

# plt.figure()
# plt.plot([chisquare_uniform_test(e, 0.01)['reject_H0'] for e in evidences], label="Reject H0")
# plt.legend()
# plt.title("Chi-Square")

import numpy as np

def pp_bounds_single(u, a):
    p_min = a * u
    p_max = 1.0 - (1.0 - a) * u
    return p_min, p_max


def pp_bounds_component(u_x, u_y, a_x=np.sqrt(0.5), a_y=np.sqrt(0.5)):
    px_min, px_max = pp_bounds_single(u_x, a_x)
    py_min, py_max = pp_bounds_single(u_y, a_y)

    p_comp_min = px_min * py_min
    p_comp_max = px_max * py_max

    return p_comp_min, p_comp_max


def pp_bounds_overall_average(u_x, u_y, u_r,
                              a_x=np.sqrt(0.5),
                              a_y=np.sqrt(0.5),
                              a_r=0.5):
    p_comp_min, p_comp_max = pp_bounds_component(u_x, u_y, a_x, a_y)
    p_r_min, p_r_max = pp_bounds_single(u_r, a_r)

    p_overall_min = 0.5 * (p_comp_min + p_r_min)
    p_overall_max = 0.5 * (p_comp_max + p_r_max)

    return p_overall_min, p_overall_max


def pp_bounds_overall_abf(u_x, u_y, u_comp, u_r,
                          a_x=np.sqrt(0.5),
                          a_y=np.sqrt(0.5),
                          a_r=0.5,
                          eps=1e-12):
    p_comp_min, p_comp_max = pp_bounds_component(u_x, u_y, a_x, a_y)
    p_r_min, p_r_max = pp_bounds_single(u_r, a_r)

    denom = u_comp + u_r

    if denom <= eps:
        # both dogmatic: ABF degenerates; projection lies between channel limits
        return min(p_comp_min, p_r_min), max(p_comp_max, p_r_max)

    p_overall_min = (u_r * p_comp_min + u_comp * p_r_min) / denom
    p_overall_max = (u_r * p_comp_max + u_comp * p_r_max) / denom

    return p_overall_min, p_overall_max

p_min_avg = []
p_max_avg = []

p_min_abf = []
p_max_abf = []

for opx, opy, op_comp, op_r in zip(
    component_x_binomial,
    component_y_binomial,
    component_binomial,
    global_op_history
):
    u_x = opx.uncertainty()
    u_y = opy.uncertainty()
    u_comp = op_comp.uncertainty()
    u_r = op_r.uncertainty()

    lo_avg, hi_avg = pp_bounds_overall_average(u_x, u_y, u_r)
    lo_abf, hi_abf = pp_bounds_overall_abf(u_x, u_y, u_comp, u_r)

    p_min_avg.append(lo_avg)
    p_max_avg.append(hi_avg)

    p_min_abf.append(lo_abf)
    p_max_abf.append(hi_abf)

plt.figure()
plt.plot(p_ok_overall, label="P_OK Overall")
plt.plot(p_min_avg, "--", label="min PP avg")
plt.plot(p_max_avg, "--", label="max PP avg")
plt.grid()
plt.legend()
plt.title("Analytical PP bounds")
# %%
"""
==============
GOSPA PLOTTING
==============
"""
from matplotlib.dates import num2date, SecondLocator, MicrosecondLocator
from matplotlib.ticker import FuncFormatter, MultipleLocator
def plot_gospa(gospa_metrics, gospa_gen_name: str | list[str], plot_switching=False):

    if type(gospa_gen_name) != list:
        gospa_gen_name = [gospa_gen_name]
    for gen_name in gospa_gen_name:
        plt.figure()
        gospa_timesteps = []
        gospa_distances = []
        gospa_localisation = []
        gospa_missed = []
        gospa_false = []
        gospa_switching = []

        gospa_time_range_metrics = gospa_metrics[gen_name]['GOSPA Metrics']
        for single_time_metric in gospa_time_range_metrics.value:
            gospa_timesteps.append(single_time_metric.timestamp)
            gospa_distances.append(single_time_metric.value['distance'])
            gospa_localisation.append(single_time_metric.value['localisation'])
            gospa_missed.append(single_time_metric.value['missed'])
            gospa_false.append(single_time_metric.value['false'])
            gospa_switching.append(single_time_metric.value['switching'])

        if plot_switching:
            ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
            ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
            ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
            ax4 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
            ax5 = plt.subplot2grid((3, 4), (2, 1), colspan=2)
        else:
            ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2)
            ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
            ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
            ax4 = plt.subplot2grid((2, 4), (1, 0), colspan=2)


        def format_date(a, b):
            t=num2date(a)
            ms = str(t.microsecond)[:1]
            res = f"{t.second}.{ms}"
            return res

        plt.suptitle(f"GOSPA Metrics {gen_name}")

        ax1.set_title("Distance")
        # ax1.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax1.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax1.set_xlabel("t in seconds")
        ax1.plot(gospa_distances)
        # plt.xticks(np.arange(start=start_time, stop=start_time + timedelta(seconds=stop), step=timedelta(seconds=1)))

        ax2.set_title("Localisation")
        # ax2.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax2.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax2.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax2.set_xlabel("t in seconds")
        ax2.plot(gospa_localisation)

        ax3.set_title("Missed")
        # ax4.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax4.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax4.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax3.set_xlabel("t in seconds")
        ax3.plot(gospa_missed)

        ax4.set_title("False")
        # ax5.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax5.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax5.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax4.set_xlabel("t in seconds")
        ax4.plot(gospa_false)

        if plot_switching:
            ax5.set_title("Switching")
            # ax3.xaxis.set_major_locator(SecondLocator(interval=10))
            # ax3.xaxis.set_major_formatter(FuncFormatter(format_date))
            # ax3.xaxis.set_minor_locator(MicrosecondLocator(100000))
            ax5.set_xlabel("t in seconds")
            ax5.plot(gospa_switching)

        plt.tight_layout()

    return plt
# %%
"""
======================
DEFINE METRICS MANAGER 
======================
"""
# load a multi-metric manager
from stonesoup.metricgenerator.manager import MultiManager
# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.measures.state import Euclidean

stone_soup_tracker = True

c=10
p=2

# Euclidean-Distanz nur über Positions-Indizes x=0, y=2
# Track ist 5D [x,vx,y,vy,omega], GT ist 4D [x,vx,y,vy] → mapping nötig
from stonesoup.measures.state import Euclidean
position_measure = Euclidean(mapping=[0, 2])

from stonesoup.metricgenerator.ospametric import GOSPAMetric
# GOSPA Stone Soup
if stone_soup_tracker: gospa_kalman = GOSPAMetric(c=c, p=p, generator_name='GOSPA',
                            tracks_key='tracks', truths_key='truths',
                            switching_penalty=1, measure=position_measure)

# Use the track associator
associator = TrackToTruth(association_threshold=30, measure=position_measure)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([gospa_kalman],
                              associator)

# for tru in truths:
metric_manager.add_data({'truths' : {truth}}, overwrite=False)
metric_manager.add_data({'tracks' : {track}}, overwrite=False)
# print("Number of Stone Soup Tracks:", len(tracks))

# %%
metrics = metric_manager.generate_metrics()

plt = plot_gospa(metrics, gospa_gen_name=["GOSPA"])
# %%
# Plot the resulting track, including uncertainty ellipses
# EKF-Kovarianzmatrizen können durch Jacobi-Drift komplexe Eigenwerte haben →
# arctan2 in _generate_ellipse_points schlägt fehl. Kovarianz vor dem Plotten
# auf reell-symmetrisch-positiv-definit projizieren.
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track as _Track
def sanitize_track_covar(trk):
    clean = _Track()
    for state in trk:
        P = np.array(state.covar, dtype=float)  # komplexe Anteile verwerfen
        P = (P + P.T) / 2                        # symmetrisieren
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals.real, 1e-8) # negative EW clipping
        P_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        clean.append(GaussianState(state.state_vector, P_clean,
                                   timestamp=state.timestamp))
    return clean

plotter.plot_tracks(sanitize_track_covar(track), [0, 2], uncertainty=True)
# plotter.fig.show(renderer="browser")
# plotter.show()
# plt.show()

from aduulm_scripts.utils.plotting import plot_selfassessment_with_nis


fig = plot_selfassessment_with_nis(selfassessor_measures_history, nis_measures_history, nis_settings["alpha"],
                             assessor_type='KalmanSelfAssessor')

# plotter.fig.show(renderer="browser")
# import sys
# sys.exit(0)

# %%
import numpy as np
import plotly.graph_objects as go

def add_following_zoom_to_animated_plot(
    fig,
    states,
    mapping=(0, 2),
    half_width=20.0,
    half_height=20.0,
    frame_duration=8,
    transition_duration=0,
    xaxis_name="xaxis",
    yaxis_name="yaxis",
    keep_aspect=True,
):
    """
    Add a moving viewport to a Plotly animation.

    Important:
        Call this AFTER fig.set_subplots(...), AFTER all fig.add_trace(...),
        and AFTER all fig.update_layout(...).

    For your first subplot, the relevant layout axes are usually:
        xaxis_name="xaxis"
        yaxis_name="yaxis"
    """
    import numpy as np
    import plotly.graph_objects as go

    centers = []
    for state in states:
        x = np.asarray(state.state_vector, dtype=float).reshape(-1)
        centers.append((float(x[mapping[0]]), float(x[mapping[1]])))

    if not centers:
        raise ValueError("No states provided for following zoom.")

    n_frames = len(fig.frames)

    if n_frames == 0:
        raise ValueError("Figure has no animation frames.")

    if len(centers) < n_frames:
        centers += [centers[-1]] * (n_frames - len(centers))
    else:
        centers = centers[:n_frames]

    # Initial view
    cx0, cy0 = centers[0]

    fig.layout[xaxis_name].update(
        range=[cx0 - half_width, cx0 + half_width],
        autorange=False,
        fixedrange=False,
    )
    fig.layout[yaxis_name].update(
        range=[cy0 - half_height, cy0 + half_height],
        autorange=False,
        fixedrange=False,
    )

    if keep_aspect:
        fig.layout[yaxis_name].update(
            scaleanchor="x",
            scaleratio=1,
        )

    # Update every frame layout.
    new_frames = []

    for frame, (cx, cy) in zip(fig.frames, centers):
        # Preserve possible existing frame layout
        old_layout = frame.layout.to_plotly_json() if frame.layout is not None else {}

        old_layout[xaxis_name] = {
            **old_layout.get(xaxis_name, {}),
            "range": [cx - half_width, cx + half_width],
            "autorange": False,
            "fixedrange": False,
        }

        old_layout[yaxis_name] = {
            **old_layout.get(yaxis_name, {}),
            "range": [cy - half_height, cy + half_height],
            "autorange": False,
            "fixedrange": False,
        }

        if keep_aspect:
            old_layout[yaxis_name]["scaleanchor"] = "x"
            old_layout[yaxis_name]["scaleratio"] = 1

        new_frames.append(
            go.Frame(
                data=frame.data,
                name=frame.name,
                traces=frame.traces,
                layout=go.Layout(old_layout),
            )
        )

    fig.frames = tuple(new_frames)

    # Force animation to redraw layout changes.
    if fig.layout.updatemenus:
        for menu in fig.layout.updatemenus:
            for button in menu.buttons:
                if button.method == "animate":
                    button.args = [
                        button.args[0],
                        {
                            "frame": {
                                "duration": frame_duration,
                                "redraw": True,
                            },
                            "transition": {
                                "duration": transition_duration,
                            },
                            "fromcurrent": True,
                            "mode": "immediate",
                        },
                    ]

    if fig.layout.sliders:
        for slider in fig.layout.sliders:
            for step in slider.steps:
                step.args = [
                    step.args[0],
                    {
                        "frame": {
                            "duration": frame_duration,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": transition_duration,
                        },
                        "mode": "immediate",
                    },
                ]

    return fig

import numpy as np
import plotly.graph_objects as go


def make_follow_track_animation(
    truth,
    track,
    measurements=None,
    state_mapping=(0, 2),
    follow="track",
    half_width=20.0,
    half_height=20.0,
    tail_length=25,
    frame_stride=2,
    frame_duration=1,
    show_measurements=True,
):
    """
    Fast Plotly animation with fixed axes and moving data.

    Instead of moving xaxis/yaxis ranges, all positions are transformed
    into coordinates relative to the current follow center.

    This avoids Plotly's slow/inconsistent live relayout during animation.

    Parameters
    ----------
    truth : GroundTruthPath
    track : Track
    measurements : list[Detection] or None
    state_mapping : tuple[int, int]
        Position indices in state vector. In your setup: (0, 2).
    follow : {"track", "truth"}
        Defines the moving center.
    half_width, half_height : float
        Visible window size around the followed object.
    tail_length : int
        Number of previous samples shown as trail.
    frame_stride : int
        Use every n-th frame. Higher = faster animation.
    frame_duration : int
        Milliseconds per frame. Smaller = faster.
    show_measurements : bool
        Plot measurements if available.
    """

    def states_to_xy(states, mapping):
        xy = []
        for s in states:
            x = np.asarray(s.state_vector, dtype=float).reshape(-1)
            xy.append([float(x[mapping[0]]), float(x[mapping[1]])])
        return np.asarray(xy, dtype=float)

    def measurements_to_xy(detections):
        xy = []
        for d in detections:
            z = np.asarray(d.state_vector, dtype=float).reshape(-1)
            xy.append([float(z[0]), float(z[1])])
        return np.asarray(xy, dtype=float)

    truth_xy = states_to_xy(truth, state_mapping)
    track_xy = states_to_xy(track, state_mapping)

    n = min(len(truth_xy), len(track_xy))

    truth_xy = truth_xy[:n]
    track_xy = track_xy[:n]

    if measurements is not None:
        meas_xy = measurements_to_xy(measurements)[:n]
    else:
        meas_xy = None

    if follow == "track":
        centers = track_xy
    elif follow == "truth":
        centers = truth_xy
    else:
        raise ValueError("follow must be 'track' or 'truth'.")

    frame_indices = list(range(0, n, frame_stride))

    def rel_tail(xy, k):
        start = max(0, k - tail_length + 1)
        center = centers[k]
        return xy[start:k + 1] - center

    def rel_point(xy, k):
        return xy[k:k + 1] - centers[k]

    k0 = frame_indices[0]

    truth_rel0 = rel_tail(truth_xy, k0)
    track_rel0 = rel_tail(track_xy, k0)

    data = [
        go.Scatter(
            x=truth_rel0[:, 0],
            y=truth_rel0[:, 1],
            mode="lines+markers",
            name="Ground truth",
            marker=dict(size=5),
        ),
        go.Scatter(
            x=track_rel0[:, 0],
            y=track_rel0[:, 1],
            mode="lines+markers",
            name="Track",
            marker=dict(size=5),
        ),
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            name=f"Follow center ({follow})",
            marker=dict(size=10, symbol="cross"),
        ),
    ]

    if show_measurements and meas_xy is not None:
        meas_rel0 = rel_point(meas_xy, k0)
        data.append(
            go.Scatter(
                x=meas_rel0[:, 0],
                y=meas_rel0[:, 1],
                mode="markers",
                name="Measurement",
                marker=dict(size=6, symbol="x"),
            )
        )

    frames = []

    for k in frame_indices:
        truth_rel = rel_tail(truth_xy, k)
        track_rel = rel_tail(track_xy, k)

        frame_data = [
            go.Scatter(
                x=truth_rel[:, 0],
                y=truth_rel[:, 1],
                mode="lines+markers",
            ),
            go.Scatter(
                x=track_rel[:, 0],
                y=track_rel[:, 1],
                mode="lines+markers",
            ),
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
            ),
        ]

        if show_measurements and meas_xy is not None:
            meas_rel = rel_point(meas_xy, k)
            frame_data.append(
                go.Scatter(
                    x=meas_rel[:, 0],
                    y=meas_rel[:, 1],
                    mode="markers",
                )
            )

        cx, cy = centers[k]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(k),
                layout=go.Layout(
                    title=(
                        f"Follow-{follow} animation | "
                        f"k={k}, center=({cx:.1f}, {cy:.1f})"
                    )
                )
            )
        )

    fig = go.Figure(data=data, frames=frames)

    fig.update_layout(
        title=f"Follow-{follow} animation",
        xaxis=dict(
            range=[-half_width, half_width],
            autorange=False,
            title="relative x [m]",
            zeroline=True,
        ),
        yaxis=dict(
            range=[-half_height, half_height],
            autorange=False,
            title="relative y [m]",
            zeroline=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        width=900,
        height=800,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=frame_duration,
                                    redraw=False,
                                ),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(k)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(k),
                    )
                    for k in frame_indices
                ],
                currentvalue=dict(prefix="k = "),
            )
        ],
    )

    return fig

# fig_follow = make_follow_track_animation(
#     truth=truth,
#     track=track,
#     measurements=measurements,
#     state_mapping=(0, 2),
#     follow="track",          # oder "truth"
#     half_width=12.0,         # näher dran
#     half_height=12.0,
#     tail_length=25,
#     frame_stride=3,          # schneller: 3 oder 5
#     frame_duration=1,        # sehr schnell
#     show_measurements=True,
# )
#
# fig_follow.show(renderer="browser")

def get_xy_centers_from_states(states, mapping=(0, 2)):
    centers = []
    for state in states:
        x = np.asarray(state.state_vector, dtype=float).reshape(-1)
        centers.append([float(x[mapping[0]]), float(x[mapping[1]])])
    return np.asarray(centers, dtype=float)


def shift_xy_trace_to_follow_center(trace, center):
    """
    Shifts one xy trace into coordinates relative to center.

    Only applies to traces with x/y data.
    Ternary, bar etc. are left untouched by caller.
    """
    if not hasattr(trace, "x") or not hasattr(trace, "y"):
        return trace

    if trace.x is None or trace.y is None:
        return trace

    try:
        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
    except Exception:
        return trace

    trace.x = x - center[0]
    trace.y = y - center[1]
    trace.xaxis = "x"
    trace.yaxis = "y"

    return trace


def apply_follow_view_to_existing_animation(
    fig,
    follow_states,
    n_tracking_traces,
    mapping=(0, 2),
    half_width=18.0,
    half_height=18.0,
    frame_duration=5,
    frame_stride=1,
    redraw=False,
):
    """
    Converts only the tracking subplot of an existing Plotly animation
    to a relative follow-view.

    Important:
        This does NOT move xaxis/yaxis per frame.
        It shifts the tracking trace data per frame.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Your combined subplot figure.
    follow_states : Track or GroundTruthPath
        States used as moving center.
    n_tracking_traces : int
        Number of original Stone Soup tracking traces.
        Capture this before adding ternary/bar/etc. traces.
    mapping : tuple
        Position state indices, usually (0, 2).
    half_width, half_height : float
        Fixed visible follow window.
    frame_duration : int
        ms per frame.
    frame_stride : int
        Keep every n-th frame. Use 2, 3, or 5 for faster playback.
    """
    centers = get_xy_centers_from_states(follow_states, mapping=mapping)

    if len(centers) == 0:
        raise ValueError("No follow centers available.")

    # Optional speed-up: reduce number of frames.
    if frame_stride > 1:
        kept_frames = []
        kept_centers = []

        for idx, frame in enumerate(fig.frames):
            if idx % frame_stride == 0:
                kept_frames.append(frame)
                center_idx = min(idx, len(centers) - 1)
                kept_centers.append(centers[center_idx])

        fig.frames = tuple(kept_frames)
        centers_for_frames = np.asarray(kept_centers)
    else:
        centers_for_frames = centers[:len(fig.frames)]

    if len(centers_for_frames) < len(fig.frames):
        pad = np.repeat(centers_for_frames[-1][None, :],
                        len(fig.frames) - len(centers_for_frames),
                        axis=0)
        centers_for_frames = np.vstack([centers_for_frames, pad])

    # Shift initial/base tracking traces using first center.
    center0 = centers_for_frames[0]

    for trace_idx in range(min(n_tracking_traces, len(fig.data))):
        tr = fig.data[trace_idx]
        if getattr(tr, "type", None) == "scatter":
            shift_xy_trace_to_follow_center(tr, center0)

    # Shift tracking traces inside each frame.
    new_frames = []

    for frame_idx, frame in enumerate(fig.frames):
        center = centers_for_frames[frame_idx]
        new_data = list(frame.data)

        # Only the original Stone Soup xy traces are shifted.
        for trace_idx in range(min(n_tracking_traces, len(new_data))):
            tr = new_data[trace_idx]
            if getattr(tr, "type", None) == "scatter":
                shift_xy_trace_to_follow_center(tr, center)

        new_frames.append(
            go.Frame(
                data=new_data,
                name=str(frame_idx),
                traces=frame.traces,
                layout=frame.layout,
            )
        )

    fig.frames = tuple(new_frames)

    # Fixed tracking subplot axes.
    fig.update_xaxes(
        range=[-half_width, half_width],
        autorange=False,
        title_text="relative x [m]",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        range=[-half_height, half_height],
        autorange=False,
        title_text="relative y [m]",
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )

    # Fast animation controls.
    play_args = [
        None,
        dict(
            frame=dict(duration=frame_duration, redraw=redraw),
            transition=dict(duration=0),
            fromcurrent=True,
            mode="immediate",
        ),
    ]

    stop_args = [
        [None],
        dict(
            frame=dict(duration=0, redraw=redraw),
            transition=dict(duration=0),
            mode="immediate",
        ),
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=0.0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=play_args),
                    dict(label="Stop", method="animate", args=stop_args),
                ],
            )
        ]
    )

    # Rebuild slider for possibly reduced frame list.
    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=redraw),
                    transition=dict(duration=0),
                ),
            ],
            label=str(i),
        )
        for i in range(len(fig.frames))
    ]

    fig.update_layout(
        sliders=[
            dict(
                steps=slider_steps,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=30),
            )
        ]
    )

    return fig

# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
show_all_opinion = False
fig = plotter.fig
n_tracking_traces = len(fig.data)

# add_following_zoom_to_animated_plot(
#     fig,
#     states=track,          # folgt dem geschätzten Track
#     mapping=(0, 2),
#     half_width=30.0,       # näher dran
#     half_height=30.0,
#     frame_duration=15,     # schneller
#     transition_duration=0,
#     keep_aspect=True,
# )


fig.set_subplots(
    rows=4, cols=5,
    specs=[
        [{"colspan": 3, "rowspan": 3},  None,                   None,                   {"colspan": 2, "rowspan": 2, "type": "ternary"}, None],              # Zeile 1
        [None,                          None,                   None,                   None,                   None                ],        # Zeile 2
        [None,                          None,                   None,                   {"type": "ternary"},    {"type": "ternary"} ],        # Zeile 3
        [None,                          None,                   None,                   {"type": "xy"},         {"type": "xy"}      ]
    ],
    subplot_titles=[
        "Track", "Overall Opinion vs Griebel",
        # "H1 Opinion", "H2 Opinion",
        "Radial and Comps",
        "X and Y Comp",# "H5 Opinion",
        # "H2 Histogram",
        "Short Term Bins", "Radial Buffer Bins"#, "H5 Histogram"
    ],
    vertical_spacing=0.1 ,
    horizontal_spacing=0.01
)

for trace in fig.data:
    trace.update(xaxis="x1", yaxis="y1")

b0_f, d0_f, u0_f = overall_binomial[0].belief(), overall_binomial[0].disbelief(), overall_binomial[0].uncertainty()
b0_f2, d0_f2, u0_f2 = fused_2_op_obj_history[0].belief(), fused_2_op_obj_history[0].disbelief(), fused_2_op_obj_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u0_f, u0_f2],
        b=[d0_f, d0_f2],
        c=[b0_f, b0_f2],

        mode='markers',
        marker=dict(size=[14, 14], color=['purple', 'cyan']),
        hovertemplate=["Overall<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "Griebel<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="Overall/Griebel"
    ),
    row=1, col=4
)

prior = overall_binomial[0].prior_belief_masses[0]

P0 = b0_f + prior * u0_f

fig.add_trace(
    go.Scatterternary(
        a=[0],
        b=[1 - P0],
        c=[P0],
        mode='markers',
        marker=dict(size=10, color='purple'),
        name='PP Overall',
        hovertemplate="P: %{c:.2f}<extra></extra>"
    ),
    row=1, col=4
)

fig.add_trace(
    go.Scatterternary(
                a=[u0_f, 0],
                b=[d0_f, 1 - P0],
                c=[b0_f, P0],
                mode='lines',
                line=dict(color='purple', dash='dot'),
                showlegend=False,
            ),
    row=1, col=4
)

prior_2 = fused_2_op_obj_history[0].prior_belief_masses[0]
P0_2 = b0_f2 + prior_2 * u0_f2

fig.add_trace(
    go.Scatterternary(
        a=[0],
        b=[1 - P0_2],
        c=[P0_2],
        mode='markers',
        marker=dict(size=10, color='cyan'),
        name='PP Griebel',
        hovertemplate="P: %{c:.2f}<extra></extra>"
    ),
    row=1, col=4
)

fig.add_trace(
    go.Scatterternary(
                a=[u0_f2, 0],
                b=[d0_f2, 1 - P0_2],
                c=[b0_f2, P0_2],
                mode='lines',
                line=dict(color='cyan', dash='dot'),
                showlegend=False
            ),
    row=1, col=4
)

# b_h1, d_h1, u_h1 = h1_r_op_history[0].belief(), h1_r_op_history[0].disbelief(), h1_r_op_history[0].uncertainty()
# b_h11, d_h11, u_h11 = h1_1_op_history[0].belief(), h1_1_op_history[0].disbelief(), h1_1_op_history[0].uncertainty()
# b_h12, d_h12, u_h12 = h1_2_op_history[0].belief(), h1_2_op_history[0].disbelief(), h1_2_op_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u_h1, u_h11, u_h12] if show_all_opinion else [u_h1],
#         b=[d_h1, d_h11, d_h12]if show_all_opinion else [d_h1],
#         c=[b_h1, b_h11, b_h12]if show_all_opinion else [b_h1],
#         mode='markers',
#         marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
#         hovertemplate=["H1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H1.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H1.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
#         name="H1"
#     ),
#     row=3, col=1
# )

# b_h2, d_h2, u_h2 = h2_q_op_history[0].belief(), h2_q_op_history[0].disbelief(), h2_q_op_history[0].uncertainty()
# # b_h21, d_h21, u_h21 = h2_1_op_history[0].belief(), h2_1_op_history[0].disbelief(), h2_1_op_history[0].uncertainty()
# # b_h22, d_h22, u_h22 = h2_2_op_history[0].belief(), h2_2_op_history[0].disbelief(), h2_2_op_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u_h2], #, u_h21, u_h22],
#         b=[d_h2], #, d_h21, d_h22],
#         c=[b_h2], #, b_h21, b_h22],
#         mode='markers',
#         marker=dict(size=14, color=['green']), #, 'yellow', 'cyan']),
#         hovertemplate=["H2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"], #, "H2.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "H2.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
#         name="H2"
#     ),
#     row=3, col=2
# )

b0, d0, u0 = global_op_history[0].belief(), global_op_history[0].disbelief(), global_op_history[0].uncertainty()
# b0_ks, d0_ks, u0_ks = ks_op_obj_history[0].belief(), ks_op_obj_history[0].disbelief(), ks_op_obj_history[0].uncertainty()
# b0_ad, d0_ad, u0_ad = opinions_ad[0]
b_h3, d_h3, u_h3 = component_binomial[0].belief(), component_binomial[0].disbelief(), component_binomial[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h3, u0,], # if show_all_opinion else [u_h3],
        b=[d_h3, d0,], # if show_all_opinion else [d_h3],
        c=[b_h3, b0,], # if show_all_opinion else [b_h3],
        mode='markers',
        marker=dict(size=14, color=['green', 'cyan', 'yellow'] if show_all_opinion else ['green', 'cyan']),
        hovertemplate=["H3<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "KL<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "AD<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"]  if show_all_opinion else ["Components<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "Radial<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="Components/Radial"
    ),
    row=3, col=4
)

# fig.add_trace(
#     go.Scatter(
#         x=x_pdf,
#         y=dirichlet_pdfs[0],
#         mode='lines',
#         line=dict(color='blue'),
#         name='Beta'
#     ),
#     row=2, col=2
# )




b_h4, d_h4, u_h4 = component_x_binomial[0].belief(), component_x_binomial[0].disbelief(), component_x_binomial[0].uncertainty()
b_h41, d_h41, u_h41 = component_y_binomial[0].belief(), component_y_binomial[0].disbelief(), component_y_binomial[0].uncertainty()
# b_h42, d_h42, u_h42 = h4_2_op_history[0].belief(), h4_2_op_history[0].disbelief(), h4_2_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h4, u_h41], #, u_h42] if show_all_opinion else [u_h4],
        b=[d_h4, d_h41], #, d_h42] if show_all_opinion else [d_h4],
        c=[b_h4, b_h41], #, b_h42] if show_all_opinion else [b_h4],
        mode='markers',
        marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['red', 'blue']),
        hovertemplate=["H4<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H4.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H4.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["X<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "Y<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="x-/y-Comp"
    ),
    row=3, col=5
)


# b_h5, d_h5, u_h5 = h5_white_op_history[0].belief(), h5_white_op_history[0].disbelief(), h5_white_op_history[0].uncertainty()
# b_h51, d_h51, u_h51 = h5_1_op_history[0].belief(), h5_1_op_history[0].disbelief(), h5_1_op_history[0].uncertainty()
# b_h52, d_h52, u_h52 = h5_2_op_history[0].belief(), h5_2_op_history[0].disbelief(), h5_2_op_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u_h5, u_h51, u_h52] if show_all_opinion else [u_h5],
#         b=[d_h5, d_h51, d_h52] if show_all_opinion else [d_h5],
#         c=[b_h5, b_h51, b_h52] if show_all_opinion else [b_h5],
#         mode='markers',
#         marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
#         hovertemplate=["H5<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H5.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H5.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H5<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
#         name="H5"
#     ),
#     row=3, col=5
# )

# fig.add_trace(
#     go.Bar(
#         x=list(range(M)),
#         y=h2_count_history[0],
#         name="H2 counts",
#         marker=dict(color='green')
#
#     ),
#     row=4, col=2
# )

fig.add_trace(
    go.Bar(
        x=list(range(M)),
        y=counts_history[0],
        name="Short Term Buffer Radial",
        marker=dict(color='cyan')

    ),
    row=4, col=4
)

fig.add_trace(
    go.Bar(
        x=list(range(W)),
        y=evidences[0],
        name="LTST Buffer Radial",
        marker=dict(color='cyan')

    ),
    row=4, col=5
)
#
# fig.add_trace(
#     go.Bar(
#         x=list(range(M)),
#         y=h5_count_history[0],
#         name="H5 counts",
#         marker=dict(color='cyan')
#
#     ),
#     row=4, col=5
# )

# fig.update_yaxes(range=[0, M**2], row=4, col=2)
fig.update_yaxes(range=[0, SHORT_WINDOW_SIZE*2], row=4, col=4)
fig.update_yaxes(range=[0, SHORT_WINDOW_SIZE*2], row=4, col=5)
# fig.update_yaxes(range=[0, M**2], row=4, col=5)

n_base = len(fig.data)
# print(n_base)
idx_opinion     = len(fig.data) - 6
idx_beta        = len(fig.data) - 5
idx_proj        = len(fig.data) - 4
idx_line        = len(fig.data) - 3
idx_hist        = len(fig.data) - 2
idx_opinion_ad  = len(fig.data) - 1
new_frames = []


for i, frame in enumerate(fig.frames):
    frame.name = str(i)
    b_f, d_f, u_f = overall_binomial[i].belief(), overall_binomial[i].disbelief(), overall_binomial[
        i].uncertainty()
    b_f2, d_f2, u_f2 = fused_2_op_obj_history[i].belief(), fused_2_op_obj_history[i].disbelief(), fused_2_op_obj_history[i].uncertainty()
    # b_ks, d_ks, u_ks = b0_ks, d0_ks, u0_ks#ks_op_obj_history[i].belief(), ks_op_obj_history[i].disbelief(), ks_op_obj_history[i].uncertainty()
    # b_q, d_q, u_q = q_op_obj_history[i].belief(), q_op_obj_history[i].disbelief(), q_op_obj_history[i].uncertainty()
    # b_q2, d_q2, u_q2 = q2_op_obj_history[i].belief(), q2_op_obj_history[i].disbelief(), q2_op_obj_history[i].uncertainty()
    # b_r, d_r, u_r = r_op_obj_history[i].belief(), r_op_obj_history[i].disbelief(), r_op_obj_history[i].uncertainty()
    b, d, u = global_op_history[i].belief(), global_op_history[i].disbelief(), global_op_history[i].uncertainty()
    # b_ad, d_ad, u_ad = opinions_ad[i]
    # y_i = dirichlet_pdfs[i]
    P = b_f + prior * u_f
    P2 = b_f2 + prior_2 * u_f2
    counts_i = counts_history[i]
    evidences_i = evidences[i]
    # counts_h2 = h2_count_history[i]
    # counts_h4 = h4_count_history[i]
    # counts_h5 = h5_count_history[i]
    # counts_R_i = counts_R_history[i]

    # b_h1, d_h1, u_h1 = h1_r_op_history[i].belief(), h1_r_op_history[i].disbelief(), h1_r_op_history[i].uncertainty()
    # b_h11, d_h11, u_h11 = h1_1_op_history[i].belief(), h1_1_op_history[i].disbelief(), h1_1_op_history[i].uncertainty()
    # b_h12, d_h12, u_h12 = h1_2_op_history[i].belief(), h1_2_op_history[i].disbelief(), h1_2_op_history[i].uncertainty()
    # b_h2, d_h2, u_h2 = h2_q_op_history[i].belief(), h2_q_op_history[i].disbelief(), h2_q_op_history[i].uncertainty()
    # b_h21, d_h21, u_h21 = h2_1_op_history[i].belief(), h2_1_op_history[i].disbelief(), h2_1_op_history[i].uncertainty()
    # b_h22, d_h22, u_h22 = h2_2_op_history[i].belief(), h2_2_op_history[i].disbelief(), h2_2_op_history[i].uncertainty()
    b_h3, d_h3, u_h3 = component_binomial[i].belief(), component_binomial[i].disbelief(), component_binomial[i].uncertainty()
    b_h4, d_h4, u_h4 = component_x_binomial[i].belief(), component_x_binomial[i].disbelief(), component_x_binomial[i].uncertainty()
    b_h41, d_h41, u_h41 = component_y_binomial[i].belief(),  component_y_binomial[i].disbelief(), component_y_binomial[i].uncertainty()
    # b_h42, d_h42, u_h42 = h4_2_op_history[i].belief(), h4_2_op_history[i].disbelief(), h4_2_op_history[i].uncertainty()
    # b_h5, d_h5, u_h5 = h5_white_op_history[i].belief(), h5_white_op_history[i].disbelief(), h5_white_op_history[
    #     i].uncertainty()
    # b_h51, d_h51, u_h51 = h5_1_op_history[i].belief(), h5_1_op_history[i].disbelief(), h5_1_op_history[i].uncertainty()
    # b_h52, d_h52, u_h52 = h5_2_op_history[i].belief(), h5_2_op_history[i].disbelief(), h5_2_op_history[i].uncertainty()

    # b_pq, d_pq, u_pq = q_paper_op_obj_history[i].belief(), q_paper_op_obj_history[i].disbelief(), \
    # q_paper_op_obj_history[i].uncertainty()
    # b_pr, d_pr, u_pr = r_paper_op_obj_history[i].belief(), r_paper_op_obj_history[i].disbelief(), \
    # r_paper_op_obj_history[i].uncertainty()

    # Bestehende Daten behalten + erweitern
    new_data = list(frame.data)

    new_data.append(
        go.Scatterternary(a=[u_f, u_f2],
                          b=[d_f, d_f2],
                          c=[b_f, b_f2],
                          cliponaxis=False)
    )

    new_data.append(go.Scatterternary(a=[0], b=[1 - P], c=[P], cliponaxis=False))

    new_data.append(
        go.Scatterternary(a=[u_f, 0], b=[d_f, 1 - P], c=[b_f, P], mode='lines', line=dict(color='purple', dash='dot'),
                          showlegend=False, cliponaxis=False))

    new_data.append(go.Scatterternary(a=[0], b=[1 - P2], c=[P2], cliponaxis=False))

    new_data.append(
        go.Scatterternary(a=[u_f2, 0], b=[d_f2, 1 - P2], c=[b_f2, P2], mode='lines', line=dict(color='cyan', dash='dot'),
                          showlegend=False, cliponaxis=False))

    # new_data.append(
    #     go.Scatterternary(a=[u_h1, u_h11, u_h12] if show_all_opinion else [u_h1],
    #                       b=[d_h1, d_h11, d_h12] if show_all_opinion else [d_h1],
    #                       c=[b_h1, b_h11, b_h12] if show_all_opinion else [b_h1],
    #                       cliponaxis=False)
    # )
    # new_data.append(
    #     go.Scatterternary(a=[u_h2], #, u_h21, u_h22],
    #                       b=[d_h2], #, d_h21, d_h22],
    #                       c=[b_h2], #, b_h21, b_h22],
    #                       cliponaxis=False)
    # )
    new_data.append(
        go.Scatterternary(a=[u_h3, u],# if show_all_opinion else [u_h3],
                          b=[d_h3, d],# if show_all_opinion else [d_h3],
                          c=[b_h3, b],# if show_all_opinion else [b_h3],
                          cliponaxis=False)
    )

    # new_data.append(
    #     go.Scatter(x=x_pdf, y=y_i)
    # )

    # new_data.append(
    #     go.Scatterternary(a=[u_q, u_r], #, u_q2],
    #                       b=[d_q, d_r], #, d_q2],
    #                       c=[b_q, b_r], #, b_q2])
    #                       cliponaxis=False,
    #                       )
    # )
    new_data.append(
        go.Scatterternary(a=[u_h4, u_h41], #, u_h42] if show_all_opinion else [u_h4],
                          b=[d_h4, d_h41], #, d_h42] if show_all_opinion else [d_h4],
                          c=[b_h4, b_h41], #, b_h42] if show_all_opinion else [b_h4],
                          cliponaxis=False)
    )
    # new_data.append(
    #     go.Scatterternary(a=[u_h5, u_h51, u_h52] if show_all_opinion else [u_h5],
    #                       b=[d_h5, d_h51, d_h52] if show_all_opinion else [d_h5],
    #                       c=[b_h5, b_h51, b_h52] if show_all_opinion else [b_h5],
    #                       cliponaxis=False)
    # )

    # new_data.append(
    #     go.Scatterternary(a=[u_pq, u_pr], b=[d_pq, d_pr], c=[b_pq, b_pr], cliponaxis=False)
    # )

    # new_data.append(go.Bar(x=list(range(M)), y=counts_h2))
    new_data.append(go.Bar(x=list(range(M)), y=counts_i))
    new_data.append(go.Bar(x=list(range(M)), y=evidences_i))
    # new_data.append(go.Bar(x=list(range(M)), y=counts_h5))



    new_frames.append(go.Frame(data=new_data, name=frame.name))

fig.frames = new_frames

# sliders = [dict(
#     steps=[
#         dict(
#             method='animate',
#             args=[[str(i)],
#                   dict(mode='immediate',
#                        frame=dict(duration=1000, redraw=True),
#                        transition=dict(duration=0))],
#             label=str(i)
#         )
#         for i in range(len(fig.frames))
#     ],
#     currentvalue=dict(prefix="Step: "),
#     pad=dict(t=30),
# )]
#
# fig.update_layout(sliders=sliders)

fig.update_layout(
    height=1000,

    ternary=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)',  showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)',  showticklabels=False),

    ),
    ternary2=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)',  showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)',  showticklabels=False),

    ),
    ternary3=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
    ),
    ternary4=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
    ),
    ternary5=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
    ),
    ternary6=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', linecolor='rgba(0,0,0,0)', showticklabels=False),
    )

)
for ann in fig.layout.annotations:
    if "Comp" in ann.text:
        ann.update(x=ann.x - 0.1, y=ann.y - 0.05, xanchor='left', align='left')
    elif "Overall" in ann.text:
        ann.update(x=ann.x - 0.15, y=ann.y - 0.05, xanchor='left', align='left')

# add_following_zoom_to_animated_plot(
#     fig,
#     states=track,          # oder truth, wenn du Ground Truth folgen willst
#     mapping=(0, 2),
#     half_width=18.0,       # näher dran
#     half_height=18.0,
#     frame_duration=1,      # schneller
#     transition_duration=0,
#     xaxis_name="xaxis",
#     yaxis_name="yaxis",
#     keep_aspect=True,
# )

fig = apply_follow_view_to_existing_animation(
    fig,
    follow_states=track,          # oder truth
    n_tracking_traces=n_tracking_traces,
    mapping=(0, 2),
    half_width=12.0,              # näher dran
    half_height=12.0,
    frame_duration=100,             # schneller
    frame_stride=1,               # 2, 3 oder 5 für mehr Speed
    redraw= True,
)

fig.show(renderer="browser")
plt.show()

# plotter.fig.show(renderer="browser")
# plotter.show()

plt.show()