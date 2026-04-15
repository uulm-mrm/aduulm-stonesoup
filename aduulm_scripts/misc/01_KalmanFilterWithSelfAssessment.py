#!/usr/bin/env python

"""
==========================================================
1 - An introduction to Stone Soup: using the Kalman filter
==========================================================
"""

# %%
import numpy as np
from datetime import datetime, timedelta
# import subjective_logic as sl
# from subjective_logic.draw_sl_opinions import *
# import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
import matplotlib
from copy import deepcopy
from stonesoup.types.prediction import GaussianStatePrediction, MeasurementPrediction
from collections import deque
from dataclasses import dataclass

matplotlib.use('TkAgg')


# draw_flags_opinion = {
#     'draw_hypo_texts': True,
#     'draw_axis': False,
#     'draw_axis_label': True,
#     'draw_opinion': True,
#     'draw_opinion_label': True,
#     'draw_prior': False,
#     'draw_prior_label': True,
#     'draw_projection': False,
#     'draw_projection_label': True,
#     'belief_label_position': 0.5,
#     'disbelief_label_position': 0.7,
#     'uncertainty_label_position': 0.7,
# }

# op = sl.Opinion(0.4, 0.2)
# sl.create_triangle_plot()
# draw_flags_opinion = {
#     }
# draw_full_opinion_triangle(op, '_X^{B}', None, draw_flags_opinion)
# sl.draw_point(op)
# plt.show()

# %%
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now().replace(microsecond=0)

np.random.seed(1991)  # 1991

# %%
# factor = 100
q_x = .1 #0.1
q_y = .1 #0.1
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

stationary_transition_model = deepcopy(transition_model)
# %%
# A 'truth path' is created starting at (0,0) moving to the NE at one distance unit per (time)
# step in each dimension.
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])


# Import the disturbance method for the transition model
from aduulm_scripts.utils.add_disturbance import disturbance_transition_model
disturbance_factor_process = 10 #16
# Disturbance configurations for ground truth generation
gt_transition_configs = {
    'noise_diff_coeff': [[q_x, q_y]],  # for transition model gt
    'disturbance_mode': ['jump'],
    # 'parameters': [[[150, disturbance_factor_process], [200, 1/disturbance_factor_process], [250, disturbance_factor_process], [300, 1/disturbance_factor_process]]] #, [[99, 1/100]]]
    'parameters': [[[400, disturbance_factor_process], [450, 1/disturbance_factor_process]]]
}
process_noise_coeff_memory = [[], []]

num_steps = 500
# np.random.seed(1991)
for k in range(1, num_steps + 1):

    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps

    ###################################################################################################
    # Disturb the transition model based on the disturbance modes
    transition_model = disturbance_transition_model(transition_model, gt_transition_configs, k)
    # Save the noise coefficients (process noise memory)
    for i, model in enumerate(transition_model.model_list):
        # Collect covariance matrices for each model component
        process_noise_coeff_memory[i].append(model.noise_diff_coeff)
    ###################################################################################################

    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

# %%
from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=1) #, height=300) #, height=1000)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig


# %%
# We can check the :math:`F_k` and :math:`Q_k` matrices (generated over a 1s period).
transition_model.matrix(time_interval=timedelta(seconds=1))

# %%
transition_model.covar(time_interval=timedelta(seconds=1))

# %%
# We're going to need a :class:`~.Detection` type to
# store the detections, and a :class:`~.LinearGaussian` measurement model.
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
import numpy as np

# %%
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                          [0, 1]])
    )

# measurement_model_2 = LinearGaussian(
#     ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
#     mapping=(0, 1),  # Mapping measurement vector index to state index
#     noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
#                           [0, 1]])
#     )

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
disturbance_factor_meas = 1 #4
gt_measurement_configs = {
    'disturbance_mode': ['jump', 'drift', 'outliers'],
    'parameters': [[[50, 2], [100, 0.5]], [[150, 200, 2.5], [200, 250, 0.4]], [[300, 350, 10, 3]]]
    # 'disturbance_mode': ['jump'],
    # 'parameters': [[[50, disturbance_factor_meas], [100, 1/disturbance_factor_meas], [250, disturbance_factor_meas], [300, 1/disturbance_factor_meas]]]
}
meas_std_dev_memory = []

measurements = []
for k, state in enumerate(truth):

    # Disturb the measurement model based on the disturbance modes
    measurement_model = disturbance_measurement_noise(measurement_model, gt_measurement_configs, k)

    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))
    meas_std_dev_memory.append(np.sqrt(measurement_model.noise_covar))
# %%
# Generate the measurements
# measurements: list[Detection] = []
# for state in truth:
#     measurement = measurement_model.function(state, noise=True)
#     measurements.append(Detection(measurement,
#                                   timestamp=state.timestamp,
#                                   measurement_model=measurement_model_2
#                                   )
#                         )
# %%
# for i, measurement in enumerate(measurements):
#     if 350 <= i <= 400:
#         measurement.state_vector += np.array([[5], [-5]])
# %%
# Plot the result, again mapping the x and y position values
plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Construct a Self-Assessor for the Kalman Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We're now ready to construct a self-assessor to monitor the assumptions of the Kalman filter.

from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
from stonesoup.subjective_logic.subjective_logic import BiOpinion, fusion_weighted_belief
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


def anderson_darling_uniform(x):
    x = np.sort(np.asarray(x))
    n = len(x)

    if n < 2:
        return np.nan

    i = np.arange(1, n + 1)

    # clamp for numerical stability
    x = np.clip(x, 1e-12, 1 - 1e-12)

    term1 = (2*i - 1) * (np.log(x) + np.log(1 - x[::-1]))
    A2 = -n - np.sum(term1) / n

    return A2

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


    dt = timedelta(seconds=1)
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
# %%
# ============================================================
# Paper-based AKF: RTS smoother + exact weighting factors
# ============================================================

@dataclass
class PaperAKFConfig:
    N_smooth: int = 10
    L_update: int = 5
    eps: float = 1e-9
    gamma_q: float = 1.0
    gamma_r: float = 1.0


@dataclass
class PaperAKFState:
    Q_est: np.ndarray
    R_est: np.ndarray
    SSV: np.ndarray   # measurement-noise sum of squares
    SSW: np.ndarray   # process-noise sum of squares
    l_count: int = 0


def init_paper_akf_state(Q0: np.ndarray, R0: np.ndarray) -> PaperAKFState:
    n_x = Q0.shape[0]
    n_z = R0.shape[0]
    return PaperAKFState(
        Q_est=Q0.copy(),
        R_est=R0.copy(),
        SSV=np.zeros(n_z, dtype=float),
        SSW=np.zeros(n_x, dtype=float),
        l_count=0,
    )

def h2_multistep_nis_statistic(
    x0,
    P0,
    z_k,
    F,
    Q,
    H,
    R,
    h,
    ridge=1e-9
):
    """
    Compute h-step prediction consistency statistic.

    Parameters
    ----------
    x0 : ndarray, shape (n_x, 1)
        Posterior state at time k-h.
    P0 : ndarray, shape (n_x, n_x)
        Posterior covariance at time k-h.
    z_k : ndarray, shape (m, 1)
        Measurement at current time k.
    F, Q, H, R : ndarray
        Nominal linear Gaussian model matrices.
    h : int
        Prediction horizon.
    ridge : float
        Numerical stabilization for matrix inversion.

    Returns
    -------
    eps_h : float
        h-step NIS statistic.
    nu_h : ndarray, shape (m, 1)
        h-step innovation.
    S_h : ndarray, shape (m, m)
        h-step innovation covariance.
    """
    x_pred = np.asarray(x0, dtype=float).reshape(-1, 1)
    P_pred = np.asarray(P0, dtype=float)

    for _ in range(h):
        x_pred = F @ x_pred
        P_pred = F @ P_pred @ F.T + Q

    z_k = np.asarray(z_k, dtype=float).reshape(-1, 1)

    nu_h = z_k - H @ x_pred
    S_h = H @ P_pred @ H.T + R
    S_h = 0.5 * (S_h + S_h.T)

    S_h_reg = S_h + ridge * np.eye(S_h.shape[0])
    eps_h = float((nu_h.T @ np.linalg.inv(S_h_reg) @ nu_h).item())

    return eps_h, nu_h, S_h


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


# H1
def covariance_kl_divergence(S_emp, S_ref, ridge=1e-9):
    """
    KL-like covariance divergence:
    tr(S_ref^{-1} S_emp) - logdet(S_ref^{-1} S_emp) - m

    Returns 0 in the matched case, >0 otherwise.
    """
    S_emp = np.asarray(S_emp, dtype=float)
    S_ref = np.asarray(S_ref, dtype=float)

    m = S_emp.shape[0]
    S_emp = S_emp + ridge * np.eye(m)
    S_ref = S_ref + ridge * np.eye(m)

    A = np.linalg.solve(S_ref, S_emp)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return np.inf

    D = np.trace(A) - logdet - m
    return float(max(D, 0.0))

# H4
def h4_zero_mean_test_empirical(X, ridge=1e-6):
    """
    Wald/Hotelling-like zero-mean test with empirical covariance.
    X shape: (N, m)
    """
    X = np.asarray(X, dtype=float)
    N, m = X.shape

    if N < max(6, m + 2):
        return np.nan

    mu = np.mean(X, axis=0).reshape(-1, 1)
    Sigma_hat = np.cov(X, rowvar=False)
    Sigma_hat = np.atleast_2d(Sigma_hat)
    Sigma_hat += ridge * np.eye(m)

    T_mu = float(N * (mu.T @ np.linalg.inv(Sigma_hat) @ mu).item())
    return T_mu

# H5
def h5_portmanteau_test(X, max_lag=5, ridge=1e-10):
    """
    Multivariate portmanteau test for whiteness.

    Parameters
    ----------
    X : ndarray, shape (N, m)
        Window of whitened innovations.
    max_lag : int
        Number of lags to include.
    ridge : float
        Small numerical stabilizer (not really needed here, but kept for safety).

    Returns
    -------
    Q_h : float
        Portmanteau statistic.
    df_h : int
        Approximate chi-square degrees of freedom.
    """
    X = np.asarray(X, dtype=float)
    N, m = X.shape

    if N < max_lag + 3:
        return np.nan, 0

    # remove sample mean (robust against tiny mean offsets)
    Xc = X - np.mean(X, axis=0, keepdims=True)

    Q_h = 0.0
    for lag in range(1, max_lag + 1):
        R_lag = np.zeros((m, m), dtype=float)
        for k in range(lag, N):
            ek = Xc[k].reshape(-1, 1)
            ekm = Xc[k - lag].reshape(-1, 1)
            R_lag += ek @ ekm.T
        R_lag /= N

        Q_h += np.trace(R_lag.T @ R_lag) / max(N - lag, 1)

    Q_h *= N * (N + 2)

    # chi-square approximation
    df_h = m * m * max_lag
    return float(Q_h), int(df_h)
# %%
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([.5, 0.1, .5, 0.1]), timestamp=start_time)

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
from scipy.stats import gaussian_kde
from math import sqrt, ceil

track = Track()
alphas = np.array([1.0, 1.0])
ad_alphas = np.array([1.0, 1.0])
q_alphas = np.array([1.0, 1.0])
q2_alphas = np.array([1.0, 1.0])
ks_alphas = np.array([1.0, 1.0])
alphas_R = np.array([1.0, 1.0])

# -----------------------------
# New hypothesis-specific alpha states
# -----------------------------
h1_r_alphas = np.array([1.0, 1.0])     # H1: R false
h1_direct_alphas = np.array([1.0, 1.0])
h1_meta_alphas = np.array([1.0, 1.0])
h2_q_alphas = np.array([1.0, 1.0])     # H2: Q / dynamics false
h2_alphas = np.array([1.0, 1.0])
h2_direct_alphas = np.array([1.0, 1.0], dtype=float)
h2_meta_alphas = np.array([1.0, 1.0], dtype=float)
h4_bias_meta_alphas = np.array([1.0, 1.0])  # H4: bias / mean shift
h4_bias_direct_alphas = np.array([1.0, 1.0])
h5_white_alphas = np.array([1.0, 1.0]) # H5: lack of whiteness
h5_direct_alphas = np.array([1.0, 1.0])
h5_meta_alphas = np.array([1.0, 1.0])

m = len(measurement_model.mapping)
assert m == 2
evidence_per_step = 1.0
opinions = []
opinions_ad = []
op_griebel = []
dirichlet_pdfs = []
x_pdf = np.linspace(0.001, 0.999, 500)
forget_param = 0.9
cum_u = []
cum_lr = []

M = int(ceil(sqrt(1/(1 - forget_param))))  # Anzahl Bins, 8
# print("# of Bins:", M)
bin_edges = np.linspace(0.0, 1.0, M + 1)
counts = np.ones(M) * (1/M)
# counts = np.ones(2) * (1/2)
# print("counts:", counts)
# buffer für u_k
u_buffer = []
u_r_buffer = []
max_buffer = M **2

# window_size = 50  # optional (rolling window)
u_history = []
d2_history = []
C_history = []
K_history = []
eta_history = []
counts_history = []
counts_R = np.ones(M) * (1/M)
counts_R_history = []
nu_history = []
raw_nu_history = []
kl_C_history = []
kl_kde_history = []
belief_history = []
disbelief_history = []
uncertainty_history = []
ad_belief_history = []
ad_disbelief_history = []
ad_uncertainty_history = []
ks_history = []
ad_history= []
update_effort_history = []

e_beta_history = []
ad_e_beta_history = []
ks_e_beta_history = []
q_e_beta_history = []

op_obj_history = []
ad_op_obj_history = []
ks_op_obj_history = []
q_op_obj_history = []
r_op_obj_history = []
fused_op_obj_history = []
fused_2_op_obj_history = []

dc_history = []
p_r_history = []
p_q_history = []
p_s_history = []
q2_e_beta_history = []
q_comb_e_beta_history = []
q2_op_obj_history = []

# -----------------------------
# New hypothesis opinion histories
# -----------------------------
h1_r_op_history = []
h1_D_history = []
h1_1_op_history = []
h1_2_op_history = []
h2_q_op_history = []
h2_valid_count_history = []
h2_s_buffer = []
h2_u_history = []
h2_u_values = []
h2_count_history = []
h2_cov_trace_history = []
h2_entropy_history = []
h2_C_history = []
h2_D_history = []
h2_D_buffer = []
h2_e_beta_history = []
h2a_op_history = []
h2b_op_history = []
h2_logratio_history = []
h2_baseline_history = []
h2_baseline_steps = 25
h2_baseline_ready = False
h2_update_score_history = []
h2_ax_var_history = []
h2_ay_var_history = []
h3_ng_op_history = []
h4_bias_op_history = []
h4_1_op_history = []
h4_2_op_history = []
h5_white_op_history = []
h5_Q_history = []
h5_test_history = []
h5_1_op_history = []
h5_2_op_history = []
global_op_history = []

# ---------------------------------
# H2: multi-step dynamic consistency
# ---------------------------------
h2_horizon = 3               # h-step prediction horizon
h2_window = 16               # number of h-step NIS values in the test window

h2_eps_history = []          # raw h-step NIS values
h2_T_history = []            # windowed chi-square statistic
h2_p_history = []            # p-values
h2_1_op_history = []         # direct opinion
h2_2_op_history = []         # optional meta opinion
h2_multistep_score_history = []

# optional score histories
h1_r_score_history = []
h1_count_history = []
h2_q_score_history = []
h4_bias_score_history = []
h4_T_mu_history = []
h4_test_history=[]
h4_count_history=[]
h5_white_score_history = []
h5_count_history=[]

pi0_history = []
piQ_history = []
piR_history = []
piQR_history = []
decision_history = []

kl_tests_history= []

w_R = 0.45
w_Q = 0.55

# # ============================================================
# # Paper-based covariance estimation setup
# # ============================================================
#
paper_cfg = PaperAKFConfig(
    N_smooth=10,
    L_update=1,
    eps=1e-9,
    gamma_q=1.0,
    gamma_r=1.0,
)

# nominal covariances for diagnosis
dt = timedelta(seconds=1)
F_paper = np.asarray(stationary_transition_model.matrix(time_interval=dt), dtype=float)
Q_nom = np.asarray(stationary_transition_model.covar(time_interval=dt), dtype=float)
H_paper = np.asarray(stationary_measurement_model.matrix(), dtype=float)
R_nom = np.asarray(stationary_measurement_model.covar(), dtype=float)

paper_state = init_paper_akf_state(Q0=Q_nom, R0=R_nom)

# filtered_state_buffer = deque(maxlen=paper_cfg.N_smooth + 1)
# predicted_state_buffer = deque(maxlen=paper_cfg.N_smooth)
# measurement_buffer = deque(maxlen=paper_cfg.N_smooth)

N_h2 = 10
filtered_state_buffer = deque(maxlen=N_h2 + 1)
predicted_state_buffer = deque(maxlen=N_h2)
measurement_buffer = deque(maxlen=N_h2)   # optional für H2 nicht zwingend nötig
#
Q_hat_paper_history = []
R_hat_paper_history = []
p_q_paper_history = []
p_r_paper_history = []
q_paper_op_obj_history = []
r_paper_op_obj_history = []
dv_history_paper = []
dw_history_paper = []

# # tuning for covariance-to-probability mapping
# tau_q_paper_map = 5.0
# tau_r_paper_map = 5.0

stationary = compute_stationary_kf_quantities(
    transition_model=stationary_transition_model,
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

Sigma_eta_ref = stationary["Sigma_eta_inf"]

from ordered_set import OrderedSet
tracks = set([Track([])])
truths = set([GroundTruthPath([truth])])

for i, measurement in enumerate(measurements):
    prediction: GaussianStatePrediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]
    for t in tracks:
        t.append(post)

    filtered_state_buffer.append(post)
    predicted_state_buffer.append(prediction)
    measurement_buffer.append(np.asarray(measurement.state_vector).reshape(-1))

    dx_update = (post.state_vector - prediction.state_vector).reshape(-1)
    update_effort_history.append(dx_update)
    if len(update_effort_history) > max_buffer:
        update_effort_history.pop(0)

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

    raw_nu_history.append(delta.flatten())
    if len(raw_nu_history) > max_buffer:
        raw_nu_history.pop(0)

    P = hypothesis.prediction.covar
    H = measurement_model.matrix()
    K = P @ H.T @ np.linalg.inv(S)
    # print(K)
    K_history.append(K)

    eta = (measurement.state_vector.reshape(-1, 1) - H @ post.state_vector).flatten()
    eta_history.append(eta)
    if len(eta_history) > max_buffer:
        eta_history.pop(0)

    H_nom = measurement_model.matrix()
    R_nom = measurement_model.covar()
    R_inv = np.linalg.inv(R_nom)

    # Nominale stationäre Größen:
    P_nom = prior.covar.copy()
    S_nom = H_nom @ P_nom @ H_nom.T + R_nom
    K_nom = P_nom @ H_nom.T @ np.linalg.inv(S_nom)

    A_nom = np.eye(H_nom.shape[0]) - H_nom @ K_nom
    Sigma_eta_nom = A_nom @ S_nom @ A_nom.T

    B_nom = R_inv @ Sigma_eta_nom
    mu_R_nom = np.trace(B_nom)
    sigma_R_nom = np.sqrt(2.0 * np.trace(B_nom @ B_nom) + 1e-12)

    gamma_R = 1.0

    # -----------------------------
    # Whitening
    # -----------------------------
    S_sqrt = np.linalg.cholesky(S)
    nu_white = np.linalg.solve(S_sqrt, delta).flatten()

    nu_history.append(nu_white)
    if len(nu_history) > max_buffer:
        nu_history.pop(0)

    # ---------------------------------
    # H5: whiteness hypothesis on whitened innovations
    # Proposition: "innovations are white"
    # belief     -> white
    # disbelief  -> temporally correlated
    # ---------------------------------
    N_h5 = 16
    L_h5 = 2
    counts_h5 = np.ones(M) * (1.0 / M)

    if len(nu_history) >= max(N_h5, L_h5 + 3):
        X_h5 = np.asarray(nu_history[-N_h5:])

        Q_h, df_h = h5_portmanteau_test(X_h5, max_lag=L_h5)

        # direct test statistic to evidence
        tau_h5 = chi2.ppf(0.99, df=df_h)
        e_beta = 1.0 - np.exp(-Q_h / (tau_h5 + 1e-12))
        e_alpha = 1.0 - e_beta
        # tau_soft = chi2.ppf(0.9, df=df_h)
        # tau_hard = chi2.ppf(0.99, df=df_h)
        #
        # if Q_h <= tau_soft:
        #     e_beta = 0.0
        # elif Q_h >= tau_hard:
        #     e_beta = 1.0
        # else:
        #     e_beta = (Q_h - tau_soft) / (tau_hard - tau_soft)
        #
        # e_alpha = 1 - e_beta

        h5_Q_history.append(Q_h)
        h5_white_score_history.append(e_beta)

        h5_direct_alphas *= forget_param
        h5_direct_alphas += np.array([e_alpha, e_beta])
        h5_direct_alphas = np.maximum(h5_direct_alphas, 1.0)

        dist = sl.DirichletDistribution2d(h5_direct_alphas)
        opinion = dist.as_opinion()

        projected = opinion.getProjection()
        u_max = min(
            projected[0] / opinion.prior_belief_masses[0],
            projected[1] / opinion.prior_belief_masses[1]
        )
        b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
        h5_direct_opinion = sl.Opinion(b_max, d_max)

        # ---------------------------------
        # H5 meta test: PIT + entropy on the H5 test statistic
        # ---------------------------------
        h5_test = chi2.cdf(Q_h, df=df_h)
        h5_test_history.append(h5_test)
        if len(h5_test_history) > max_buffer:
            h5_test_history.pop(0)

        for u_i in h5_test_history:
            idx = np.searchsorted(bin_edges, u_i, side='right') - 1
            idx = np.clip(idx, 0, M - 1)
            counts_h5[idx] += 1

        h5_count_history.append(counts_h5.copy())
        N_hist = np.sum(counts_h5)
        p_hist = counts_h5 / N_hist
        p_safe = np.clip(p_hist, 1e-12, 1.0)

        H_h5 = -np.sum(p_safe * np.log(p_safe))
        H_max = np.log(M)

        C_h5 = H_h5 / H_max

        e_alpha_meta = C_h5
        e_beta_meta = 1.0 - C_h5

        h5_meta_alphas *= forget_param
        h5_meta_alphas += np.array([e_alpha_meta, e_beta_meta])
        h5_meta_alphas = np.maximum(h5_meta_alphas, 1.0)

        dist = sl.DirichletDistribution2d(h5_meta_alphas)
        opinion = dist.as_opinion()

        projected = opinion.getProjection()
        u_max = min(
            projected[0] / opinion.prior_belief_masses[0],
            projected[1] / opinion.prior_belief_masses[1]
        )
        b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
        h5_meta_opinion = sl.Opinion(b_max, d_max)

    else:
        h5_direct_opinion = sl.Opinion(0, 0)
        h5_meta_opinion = sl.Opinion(0, 0)
        h5_Q_history.append(0.0)
        h5_white_score_history.append(0.0)
        h5_count_history.append(counts_h5.copy())

    # for debugging first, keep both
    h5_1_op_history.append(h5_direct_opinion)
    h5_2_op_history.append(h5_meta_opinion)

    # start simple: use the direct opinion only
    # h5_opinion = h5_direct_opinion

    # later, if both behave well:
    h5_opinion = h5_direct_opinion #h5_direct_opinion.wb_fuse(h5_meta_opinion)

    h5_white_op_history.append(h5_opinion)

    # ---------------------------------
    # H4: zero-mean hypothesis on raw innovations
    # Proposition: "innovations are zero-mean"
    # belief     -> mean zero
    # disbelief  -> bias / mean shift present
    # ---------------------------------
    N_h4 = 16
    counts_h4 = np.ones(M) * (1 / M)
    if len(raw_nu_history) >= max(N_h4, m + 3):
        X_h4 = np.asarray(raw_nu_history[-N_h4:])

        T_mu = h4_zero_mean_test_empirical(X_h4, ridge=1e-6)

        # fault evidence
        # lambda_h4 = 1/sqrt(max_buffer)
        # e_beta = 1 - np.exp(-lambda_h4 * T_mu)

        tau_h4 = chi2.ppf(0.99, df=m)  # ca. "signifikant auffällig"
        e_beta = 1 - np.exp(-T_mu / tau_h4)
        # support for H0
        e_alpha = 1 - e_beta

        h4_T_mu_history.append(T_mu)
        h4_bias_score_history.append(e_beta)

        h4_bias_direct_alphas *= forget_param
        h4_bias_direct_alphas += np.array([e_alpha, e_beta])
        h4_bias_direct_alphas = np.maximum(h4_bias_direct_alphas, 1.0)

        dist = sl.DirichletDistribution2d(h4_bias_direct_alphas)
        opinion = dist.as_opinion()

        projected = opinion.getProjection()
        u_max = min(
            projected[0] / opinion.prior_belief_masses[0],
            projected[1] / opinion.prior_belief_masses[1]
        )
        b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
        h4_direct_opinion = sl.Opinion(b_max, d_max)

        # nu_stack = nu_history[-N:]
        # nu_mean = np.mean(nu_stack, axis=0)
        # T_nu = (N * (nu_mean @ nu_mean.T)).item()

        h4_test = chi2.cdf(T_mu, df=m)
        h4_test_history.append(h4_test)
        if len(h4_test_history) > max_buffer:
            h4_test_history.pop(0)

        for u_i in h4_test_history:
            idx = np.searchsorted(bin_edges, u_i, side='right') - 1
            idx = np.clip(idx, 0, M - 1)
            counts_h4[idx] += 1

        h4_count_history.append(counts_h4.copy())
        N = np.sum(counts_h4)
        p = counts_h4 / N
        # numerisch stabil
        p_safe = np.clip(p, 1e-12, 1.0)

        H = -1 * np.sum(p_safe * np.log(p_safe))
        H_max = np.log(M)

        C = H / H_max  # ∈ [0,1]
        #
        # N_eff = len(h4_test_history)
        #
        # D_KL = H_max - H
        # D_KL_bias = (M - 1) / (2.0 * max(N_eff, 1))
        # D_KL_corr = max(0.0, D_KL - D_KL_bias)
        #
        # kappa_kl = 1.0
        # p_kl = 1.0 - np.exp(-kappa_kl * N_eff * D_KL_corr)
        #
        # e_beta = p_kl
        # e_alpha = 1.0 - p_kl
        e_alpha = C
        e_beta = 1 - C
        # h4_bias_score_history.append(D_KL)

        h4_bias_meta_alphas *= forget_param
        h4_bias_meta_alphas += np.array([e_alpha, e_beta])
        h4_bias_meta_alphas = np.maximum(h4_bias_meta_alphas, 1)

        # -----------------------------
        # Subjective Logic Opinion
        # -----------------------------
        dist = sl.DirichletDistribution2d(h4_bias_meta_alphas)
        opinion = dist.as_opinion()

        # uncertainty maximized
        projected = opinion.getProjection()
        u_max = min(projected[0] / opinion.prior_belief_masses[0], projected[1] / opinion.prior_belief_masses[1])
        b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
        h4_meta_opinion = sl.Opinion(b_max, d_max)
        # h4_opinion = h4_direct_opinion
    else:
        h4_opinion = sl.Opinion(0, 0)
        h4_direct_opinion = sl.Opinion(0, 0)
        h4_meta_opinion = sl.Opinion(0, 0)
        h4_bias_score_history.append(0)
        h4_T_mu_history.append(0)
        h4_count_history.append(counts_h4.copy())

    h4_1_op_history.append(h4_direct_opinion)
    h4_2_op_history.append(h4_meta_opinion)
    # h4_bias_op_history.append(h4_opinion)

    h4_opinion = h4_direct_opinion #h4_direct_opinion.wb_fuse(h4_meta_opinion)  # weighted belief fusion
    h4_bias_op_history.append(h4_opinion)

    # OLD Q-TEST
    rho = 0
    if len(nu_history) >= 5:

        nu_stack = np.stack(nu_history, axis=0)

        # empirische Kovarianz
        Sigma = np.cov(nu_stack, rowvar=False)
        Sigma += 1e-6 * np.eye(Sigma.shape[0])
        try:
            L = np.linalg.cholesky(Sigma)
            L_inv = np.linalg.inv(L)

            nu_rewhite = (L_inv @ nu_stack.T).T  # shape: (N, dim)

        except np.linalg.LinAlgError:
            nu_rewhite = nu_stack  # fallback

        for k in range(1, len(nu_rewhite)):
            rho += np.dot(nu_rewhite[k], nu_rewhite[k - 1]) / nu_rewhite.shape[1]

        rho /= (len(nu_rewhite) - 1)
        # print("nu_stack: ", nu_stack)
        # print("rho: ", rho)
        # print("nu_rewhite: ", nu_rewhite)
        # print("nu_rewhite shape:", nu_rewhite.shape)
        # print("Sigma: ", Sigma)
        # print("L_inv: ", L_inv)
        # print("L:", L)

        N = len(nu_rewhite)
        T_Q = rho ** 2
        e_beta_Q = 1 - np.exp(-sqrt(N/m) * T_Q)
        e_alpha_Q = 1 - e_beta_Q

        # N = len(nu_rewhite)
        # T_Q = N * m * (rho ** 2)
        # e_beta_Q = 1.0 - chi2.cdf(T_Q, df=1)
        # e_alpha_Q = 1.0 - e_beta_Q

        q_e_beta_history.append(e_beta_Q)
        q_alphas *= forget_param
        q_alphas += np.array([e_alpha_Q, e_beta_Q])

        q_alphas = np.clip(q_alphas, 1, a_max=None)

        dist = sl.DirichletDistribution2d(q_alphas)
        q_opinion = dist.as_opinion()
        # uncertainty maximized
        projected = q_opinion.getProjection()
        u_max = min(projected[0] / q_opinion.prior_belief_masses[0], projected[1] / q_opinion.prior_belief_masses[1])
        b_max = projected[0] - q_opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - q_opinion.prior_belief_masses[1] * u_max
        q_opinion = sl.Opinion(b_max, d_max)
        q_op_obj_history.append(q_opinion)

    else:
        q_opinion = sl.Opinion(0, 0)
        q_op_obj_history.append(q_opinion)
    # current q_opinion is treated as an auxiliary whiteness opinion, not as H2
    aux_temporal_opinion = q_opinion


    d2 = (delta.T @ np.linalg.inv(S) @ delta).item()
    u = chi2.cdf(d2, df=m)
    cum_u.append(u)

    u_history.append(u)
    d2_history.append(d2)

    u_buffer.append(u)
    if len(u_buffer) > max_buffer:
        u_buffer.pop(0)

    A2 = anderson_darling_uniform(u_buffer)

    if not np.isnan(A2):
        ad_s = A2
        ad_history.append(ad_s)

        # -------------------------
        # Evidence mapping (SL)
        # -------------------------
        c = sqrt(max_buffer)
        N_eff = np.sum(u_buffer)
        e_beta = (1 - np.exp(- (ad_s/c)))    #ad_p_value(ad_s, N=len(u_buffer)) #np.exp(- (ad_s/c))
        e_alpha = (1 - e_beta)
        # e_beta = 1.0 - ad_p
        # e_alpha = ad_p
        ad_e_beta_history.append(e_beta)


        ad_alphas *= forget_param
        ad_alphas += np.array([e_alpha, e_beta])
        # print(np.sum(ad_alphas))
        # ad_alphas = np.array([1.0, 1.0]) + np.array([e_alpha, e_beta])
        ad_alphas = np.clip(ad_alphas, 1, a_max=None)

        dist = sl.DirichletDistribution2d(ad_alphas)

        ad_opinion = dist.as_opinion()

        # uncertainty maximized
        projected = ad_opinion.getProjection()
        u_max = min(projected[0] / ad_opinion.prior_belief_masses[0], projected[1] / ad_opinion.prior_belief_masses[1])
        b_max = projected[0] - ad_opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - ad_opinion.prior_belief_masses[1] * u_max
        opinion_max = sl.Opinion(b_max, d_max)
        r_paper_op_obj_history.append(opinion_max)
        q_paper_op_obj_history.append(opinion_max)
        ad_opinion = opinion_max
        ad_op_obj_history.append(ad_opinion)
        # opinion.prior_belief_masses = [1.0, 0]
        opinions_ad.append((ad_opinion.belief(), ad_opinion.disbelief(), ad_opinion.uncertainty()))
        ad_belief_history.append(ad_opinion.belief())
        ad_disbelief_history.append(ad_opinion.disbelief())
        ad_uncertainty_history.append(ad_opinion.uncertainty())



    else:
        ad_opinion = sl.Opinion(0, 0)
        ad_op_obj_history.append(ad_opinion)
        opinions_ad.append((0.0, 0.0, 1))

    # idx = np.searchsorted(bin_edges, u, side='right') - 1
    # idx = np.clip(idx, 0, M - 1)
    # counts *= forget_param
    # counts[idx] += 1  # 1
    # counts[idx] += 0.7
    # counts[max(idx - 1, 0)] += 0.15
    # counts[min(idx + 1, M - 1)] += 0.15

    counts = np.ones(M) * (1/M)
    for u_i in u_buffer:
        idx = np.searchsorted(bin_edges, u_i, side='right') - 1
        idx = np.clip(idx, 0, M - 1)
        counts[idx] += 1

    counts_history.append(list(counts))

    N = np.sum(counts)

    p = counts / N

    # numerisch stabil
    p_safe = np.clip(p, 1e-12, 1.0)

    H = -1 * np.sum(p_safe * np.log(p_safe))
    H_max = np.log(M)

    C = H / H_max  # ∈ [0,1]

    # KL-Divergence
    kl_C = -H + H_max
    kl_C_history.append(kl_C)


    C_history.append(C)
    N_eff = (np.sum(counts))
    # C-scale as transformation of KL-divergence
    e_alpha = C #*N_eff
    e_beta = (1 - C) #*N_eff

    # N_eff = len(u_buffer)
    #
    # D_KL = np.log(M) - H
    # D_KL_bias = (M - 1) / (2.0 * max(N_eff, 1))
    # D_KL_corr = max(0.0, D_KL - D_KL_bias)
    #
    # kappa_kl = 1.0
    # p_kl = 1.0 - np.exp(-kappa_kl * N_eff * D_KL_corr)

    # e_beta = p_kl
    # e_alpha = 1.0 - p_kl

    alphas *= forget_param
    alphas += np.array([e_alpha, e_beta])
    # alphas = np.array([1.0, 1.0]) + [e_alpha, e_beta]
    alphas = np.maximum(alphas, 1)
    # print(i, ": ", C, N_eff, e_alpha, e_beta)
    e_beta_history.append(e_beta)
    # -----------------------------
    # Subjective Logic Opinion
    # -----------------------------
    dist = sl.DirichletDistribution2d(alphas)

    pdf = list(map(dist.evaluate, x_pdf))
    # pdf = beta.pdf(x_pdf, alphas[0], alphas[1])
    dirichlet_pdfs.append(pdf)

    kl_opinion = dist.as_opinion()

    # uncertainty maximized
    projected = kl_opinion.getProjection()
    u_max = min(projected[0]/kl_opinion.prior_belief_masses[0], projected[1]/kl_opinion.prior_belief_masses[1])
    b_max = projected[0] - kl_opinion.prior_belief_masses[0] * u_max
    d_max = projected[1] - kl_opinion.prior_belief_masses[1] * u_max
    opinion_max = sl.Opinion(b_max, d_max)
    r_paper_op_obj_history.append(opinion_max)
    q_paper_op_obj_history.append(opinion_max)
    kl_opinion = opinion_max
    op_obj_history.append(kl_opinion)
    # opinion.prior_belief_masses = [1.0, 0]
    opinions.append((kl_opinion.belief(), kl_opinion.disbelief(), kl_opinion.uncertainty()))

    counts_R_history = counts_history.copy()

    # ---------------------------------
    # R-ISO TEST:
    # remove temporal structure first, then test covariance mismatch
    # ---------------------------------
    e_beta_R_iso = 0.0

    if len(nu_history) >= 8:
        nu_stack_r = np.stack(nu_history, axis=0)  # shape (N, m)
        N_r = len(nu_stack_r)

        # mean removal
        nu_stack_r = nu_stack_r - np.mean(nu_stack_r, axis=0, keepdims=True)

        # zero-lag covariance
        C0 = (nu_stack_r.T @ nu_stack_r) / N_r
        C0 += 1e-6 * np.eye(C0.shape[0])

        # lag-1 cross covariance
        C1 = np.zeros((m, m), dtype=float)
        for k in range(1, N_r):
            C1 += np.outer(nu_stack_r[k], nu_stack_r[k - 1])
        C1 /= (N_r - 1)

        # AR(1)-like coefficient
        A = C1 @ np.linalg.inv(C0)

        # de-temporalized residuals
        r_list = []
        for k in range(1, N_r):
            r_k = nu_stack_r[k] - A @ nu_stack_r[k - 1]
            r_list.append(r_k)

        r_stack = np.stack(r_list, axis=0)
        r_stack = r_stack - np.mean(r_stack, axis=0, keepdims=True)

        Sigma_r = np.cov(r_stack, rowvar=False)
        Sigma_r += 1e-6 * np.eye(Sigma_r.shape[0])

        # conservative R-score: diagonal mismatch only
        diag_err = np.diag(Sigma_r) - np.ones(m)
        S_R_iso = float(np.dot(diag_err, diag_err) / m)

        # finite-sample floor (simple online-safe approximation)
        bR_iso = (2.0 * m) / max(len(r_stack) - 1, 1)
        S_R_iso_eff = max(0.0, S_R_iso - bR_iso)

        # bounded raw probability
        tau_r_iso = 2.0 * bR_iso
        p_r_iso_raw = S_R_iso_eff / (S_R_iso_eff + tau_r_iso + 1e-12)
        p_r_iso_raw = float(np.clip(p_r_iso_raw, 0.0, 1.0))

        e_beta_R_iso = p_r_iso_raw
        e_alpha_R_iso = 1.0 - e_beta_R_iso

        alphas_R *= forget_param
        alphas_R += np.array([e_alpha_R_iso, e_beta_R_iso])
        alphas_R = np.clip(alphas_R, 1.0, None)

        r2_dist = sl.DirichletDistribution2d(alphas_R)
        r_opinion = r2_dist.as_opinion()

        # uncertainty maximized
        projected = r_opinion.getProjection()
        u_max = min(projected[0] / r_opinion.prior_belief_masses[0], projected[1] / r_opinion.prior_belief_masses[1])
        b_max = projected[0] - r_opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - r_opinion.prior_belief_masses[1] * u_max
        r_opinion = sl.Opinion(b_max, d_max)

        r_op_obj_history.append(r_opinion)
        # r_e_beta_history.append(e_beta_R_iso)

    else:
        r_opinion = sl.Opinion(0, 0)
        r_op_obj_history.append(r_opinion)


    # ---------------------------------
    # H1: measurement-noise consistency hypothesis
    # Proposition: "measurement noise model is consistent"
    # belief     -> R consistent
    # disbelief  -> R inconsistent
    # ---------------------------------
    N_h1 = 16

    if len(eta_history) >= max(N_h1, m + 3):
        X_h1 = np.asarray(eta_history[-N_h1:])
        # mu_eta = np.mean(X_h1, axis=0, keepdims=True)
        # Xc_h1 = X_h1 - mu_eta
        Sigma_eta_emp = np.cov(X_h1, rowvar=False)

        D_R = covariance_kl_divergence(Sigma_eta_emp, Sigma_eta_ref, ridge=1e-6)

        # map divergence to fault evidence
        # tau_h1 controls how large the mismatch must be before disbelief rises strongly
        tau_h1 = 4 #4
        e_beta = 1.0 - np.exp(-D_R / tau_h1)
        e_alpha = 1.0 - e_beta

        h1_D_history.append(D_R)
        h1_r_score_history.append(e_beta)

        h1_direct_alphas *= forget_param
        h1_direct_alphas += np.array([e_alpha, e_beta])
        h1_direct_alphas = np.maximum(h1_direct_alphas, 1.0)

        dist = sl.DirichletDistribution2d(h1_direct_alphas)
        opinion = dist.as_opinion()

        projected = opinion.getProjection()
        u_max = min(
            projected[0] / opinion.prior_belief_masses[0],
            projected[1] / opinion.prior_belief_masses[1]
        )
        b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
        d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
        h1_direct_opinion = sl.Opinion(b_max, d_max)

        # ---------------------------------
        # H1 meta test: use the old R-ISO branch as auxiliary evidence
        # ---------------------------------
        # assuming your current code already computes r_opinion
        # if not available at this point, move this block below the old R-ISO part
        h1_meta_opinion = r_opinion

    else:
        h1_direct_opinion = sl.Opinion(0, 0)
        h1_meta_opinion = sl.Opinion(0, 0)
        h1_D_history.append(0.0)
        h1_r_score_history.append(0.0)

    h1_1_op_history.append(h1_direct_opinion)
    h1_2_op_history.append(h1_meta_opinion)

    # final H1 fusion
    h1_opinion = h1_direct_opinion.wb_fuse(h1_meta_opinion)
    h1_r_op_history.append(h1_opinion)

    # ---------------------------------
    # H2: multi-step dynamic consistency hypothesis
    # Proposition: "process / dynamic model is consistent"
    # belief     -> dynamics consistent
    # disbelief  -> dynamics inconsistent
    # ---------------------------------

    h2_direct_opinion = sl.Opinion(0, 0)
    h2_meta_opinion = sl.Opinion(0, 0)
    counts_h2 = np.ones(M) * (1 / M)
    # Need a posterior from k-h and the current measurement at k
    if len(filtered_state_buffer) >= h2_horizon + 1:
        old_post = filtered_state_buffer[-(h2_horizon + 1)]

        x0_h2 = np.asarray(old_post.state_vector, dtype=float).reshape(-1, 1)
        P0_h2 = np.asarray(old_post.covar, dtype=float)
        z_now = np.asarray(measurement.state_vector, dtype=float).reshape(-1, 1)

        eps_h, nu_h, S_h = h2_multistep_nis_statistic(
            x0=x0_h2,
            P0=P0_h2,
            z_k=z_now,
            F=F_h2,
            Q=Q_h2,
            H=H_h2,
            R=R_h2,
            h=h2_horizon,
            ridge=1e-9
        )

        h2_eps_history.append(eps_h)
        if len(h2_eps_history) > max_buffer:
            h2_eps_history.pop(0)

        h2_multistep_score_history.append(eps_h)

        # direct windowed chi-square test
        if len(h2_eps_history) >= h2_window:
            eps_win = np.asarray(h2_eps_history[-h2_window:], dtype=float)

            T_h2 = float(np.sum(eps_win))
            df_h2 = int(m_h2 * h2_window)

            # one-sided upper-tail p-value
            p_h2 = float(1.0 - chi2.cdf(T_h2, df=df_h2))
            p_h2 = float(np.clip(p_h2, 0.0, 1.0))
            # T_h2 = 1 - p_h2
            # conservative soft-threshold mapping
            p_soft = 0.05
            p_hard = 0.005

            if p_h2 >= p_soft:
                e_beta = 0.0
            elif p_h2 <= p_hard:
                e_beta = 1.0
            else:
                e_beta = (p_soft - p_h2) / (p_soft - p_hard)

            e_alpha = 1.0 - e_beta

            h2_T_history.append(T_h2)
            h2_p_history.append(p_h2)
            h2_q_score_history.append(e_beta)

            h2_direct_alphas *= forget_param
            h2_direct_alphas += np.array([e_alpha, e_beta])
            h2_direct_alphas = np.maximum(h2_direct_alphas, 1.0)

            dist = sl.DirichletDistribution2d(h2_direct_alphas)
            opinion = dist.as_opinion()

            projected = opinion.getProjection()
            u_max = min(
                projected[0] / opinion.prior_belief_masses[0],
                projected[1] / opinion.prior_belief_masses[1]
            )
            b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
            d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
            h2_direct_opinion = sl.Opinion(b_max, d_max)

            # # ---------------------------------
            # # H2 meta test: PIT + entropy on the H2 test statistic
            # # ---------------------------------
            # h2_test = chi2.cdf(T_h2, df=df_h2)
            # h2_u_history.append(h2_test)
            # if len(h2_u_history) > max_buffer:
            #     h2_u_history.pop(0)
            #
            #
            # for u_i in h2_u_history:
            #     idx = np.searchsorted(bin_edges, u_i, side='right') - 1
            #     idx = np.clip(idx, 0, M - 1)
            #     counts_h2[idx] += 1
            #
            # h2_count_history.append(counts_h2.copy())
            #
            # N_hist = np.sum(counts_h2)
            # p_hist = counts_h2 / N_hist
            # p_safe = np.clip(p_hist, 1e-12, 1.0)
            #
            # H_h2_entropy = -np.sum(p_safe * np.log(p_safe))
            # H_h2_max = np.log(M)
            # C_h2 = H_h2_entropy / H_h2_max
            #
            # h2_entropy_history.append(H_h2_entropy)
            # h2_C_history.append(C_h2)
            #
            # e_alpha_meta = C_h2
            # e_beta_meta = 1.0 - C_h2
            #
            # h2_meta_alphas *= forget_param
            # h2_meta_alphas += np.array([e_alpha_meta, e_beta_meta])
            # h2_meta_alphas = np.maximum(h2_meta_alphas, 1.0)
            #
            # dist = sl.DirichletDistribution2d(h2_meta_alphas)
            # opinion = dist.as_opinion()
            #
            # projected = opinion.getProjection()
            # u_max = min(
            #     projected[0] / opinion.prior_belief_masses[0],
            #     projected[1] / opinion.prior_belief_masses[1]
            # )
            # b_max = projected[0] - opinion.prior_belief_masses[0] * u_max
            # d_max = projected[1] - opinion.prior_belief_masses[1] * u_max
            # h2_meta_opinion = sl.Opinion(b_max, d_max)

        else:
            h2_T_history.append(0.0)
            h2_p_history.append(1.0)
            h2_q_score_history.append(0.0)
            h2_count_history.append(counts_h2.copy())
    else:
        h2_T_history.append(0.0)
        h2_p_history.append(1.0)
        h2_q_score_history.append(0.0)
        h2_count_history.append(counts_h2.copy())

    h2_1_op_history.append(h2_direct_opinion)
    h2_2_op_history.append(h2_meta_opinion)

    # final H2 fusion
    h2_opinion = h2_direct_opinion #h2_direct_opinion.wb_fuse(h2_meta_opinion)
    h2_q_op_history.append(h2_opinion)

    # print(i, u_h2)
    # print(i, len(h2_D_buffer), len(u_buffer))
    # print("allclose buffers:",
    #       np.allclose(h2_u_history, u_buffer[:len(h2_u_history)]) if len(h2_u_history) <= len(u_buffer) else False)


    # # ---------------------------------
    # # H2: process-noise / dynamics consistency hypothesis
    # # Proposition: "process-noise / dynamics model is consistent"
    # # belief     -> Q / dynamics consistent
    # # disbelief  -> Q / dynamics inconsistent
    # # ---------------------------------
    #
    # # ---- H2.1: smoother-based process-consistency test ----
    # if len(filtered_state_buffer) == paper_cfg.N_smooth + 1 and len(predicted_state_buffer) == paper_cfg.N_smooth:
    #     z_window = np.stack(measurement_buffer, axis=0)
    #
    #     x_smooth_window, _ = rts_smoother_window(
    #         list(filtered_state_buffer),
    #         list(predicted_state_buffer),
    #         F_paper
    #     )
    #
    #     _, w_hat_seq = estimate_noise_sequences_from_smoothed_states(
    #         z_window=z_window,
    #         x_smooth_window=x_smooth_window,
    #         F=F_paper,
    #         H=H_paper
    #     )
    #
    #     T_q1 = h2_smoother_process_energy_test(
    #         w_hat_seq=w_hat_seq,
    #         Q_ref=Q_nom,
    #         ridge=1e-6
    #     )
    #
    #     # expected nominal scale is around state dimension if roughly consistent
    #     n_x = w_hat_seq.shape[1]
    #     T_q1_excess = max(0.0, T_q1 - n_x)
    #
    #     tau_q1 = n_x
    #     e_beta_1 = 1.0 - np.exp(-T_q1_excess / (tau_q1 + 1e-12))
    #     e_alpha_1 = 1.0 - e_beta_1
    #
    #     h2_logratio_history.append(T_q1)
    #     h2_ax_var_history.append(T_q1)   # optional: reuse history slot
    #     h2_ay_var_history.append(T_q1_excess)
    #
    #     h2a_alphas *= forget_param
    #     h2a_alphas += np.array([e_alpha_1, e_beta_1])
    #     h2a_alphas = np.maximum(h2a_alphas, 1.0)
    #
    #     h2_direct1_opinion = make_max_uncertainty_opinion(h2a_alphas)
    # else:
    #     h2_direct1_opinion = sl.Opinion(0, 0)
    #     h2_logratio_history.append(0.0)
    #     h2_ax_var_history.append(0.0)
    #     h2_ay_var_history.append(0.0)
    #
    # # ---- H2.2: update-effort test ----
    # N_h2b = 16
    # if len(update_effort_history) >= N_h2b:
    #     X_upd = np.asarray(update_effort_history[-N_h2b:])  # shape (N, 4)
    #     upd_energy = np.mean(np.sum(X_upd**2, axis=1))
    #     T_h2 = float(np.mean(X_upd))
    #
    #     tau_q2 = 0.25
    #     e_beta_2 = 1.0 - np.exp(-T_h2 / tau_q2)
    #     e_alpha_2 = 1.0 - e_beta_2
    #
    #     h2_update_score_history.append(upd_energy)
    #
    #     h2b_alphas *= forget_param
    #     h2b_alphas += np.array([e_alpha_2, e_beta_2])
    #     h2b_alphas = np.maximum(h2b_alphas, 1.0)
    #
    #     h2_direct2_opinion = make_max_uncertainty_opinion(h2b_alphas)
    # else:
    #     h2_direct2_opinion = sl.Opinion(0, 0)
    #     h2_update_score_history.append(0.0)
    #
    # # final H2 fusion: only H2-relevant branches
    # h2a_op_history.append(h2_direct1_opinion)
    # h2b_op_history.append(h2_direct2_opinion)
    #
    # h2_opinion = h2_direct1_opinion.wb_fuse(h2_direct2_opinion)
    # h2_q_op_history.append(h2_direct1_opinion)

    # # ============================================================
    # # PAPER-BASED Q/R ESTIMATION
    # # ============================================================
    # q_paper_opinion = sl.Opinion(0, 0)
    # r_paper_opinion = sl.Opinion(0, 0)
    #
    # if (
    #     len(filtered_state_buffer) == paper_cfg.N_smooth + 1
    #     and len(predicted_state_buffer) == paper_cfg.N_smooth
    #     and len(measurement_buffer) == paper_cfg.N_smooth
    # ):
    #     # RTS smoother on the current fixed window
    #     x_smooth_window, P_smooth_window = rts_smoother_window(
    #         filtered_states=list(filtered_state_buffer),
    #         predicted_states=list(predicted_state_buffer),
    #         F=F_paper,
    #     )
    #
    #     z_window = np.stack(list(measurement_buffer), axis=0)
    #
    #     # Reconstruct estimated measurement/process noise sequences
    #     v_hat_seq, w_hat_seq = estimate_noise_sequences_from_smoothed_states(
    #         z_window=z_window,
    #         x_smooth_window=x_smooth_window,
    #         F=F_paper,
    #         H=H_paper,
    #     )
    #
    #     # Exact paper weighting factors
    #     dv, dw = compute_weighting_factors_exact(
    #         F=F_paper,
    #         H=H_paper,
    #         Q=paper_state.Q_est,
    #         R=paper_state.R_est,
    #         N=paper_cfg.N_smooth,
    #         eps=paper_cfg.eps,
    #     )
    #
    #     dv_history_paper.append(dv.copy())
    #     dw_history_paper.append(dw.copy())
    #
    #     upd = update_paper_sums_and_covariances(
    #         paper_state=paper_state,
    #         v_hat_seq=v_hat_seq,
    #         w_hat_seq=w_hat_seq,
    #         dv=dv,
    #         dw=dw,
    #         cfg=paper_cfg,
    #     )
    #
    #     if upd is not None:
    #         sigma_w2_hat, sigma_v2_hat, Q_hat, R_hat = upd
    #
    #         Q_hat_paper_history.append(Q_hat.copy())
    #         R_hat_paper_history.append(R_hat.copy())
    #
    #         # ------------------------------------------------
    #         # YOUR APPLICATION: diagnostics -> SL opinions
    #         # ------------------------------------------------
    #         d_q_paper = covariance_deviation_score(Q_hat, Q_nom, eps=paper_cfg.eps)
    #         d_r_paper = covariance_deviation_score(R_hat, R_nom, eps=paper_cfg.eps)
    #
    #         # use rational mapping to avoid immediate saturation
    #         p_q_paper = deviation_to_probability_rational(d_q_paper, tau_q_paper_map)
    #         p_r_paper = deviation_to_probability_rational(d_r_paper, tau_r_paper_map)
    #
    #         p_q_paper_history.append(p_q_paper)
    #         p_r_paper_history.append(p_r_paper)
    #
    #         q_paper_alphas = np.array([1.0 + (1.0 - p_q_paper), 1.0 + p_q_paper])
    #         r_paper_alphas = np.array([1.0 + (1.0 - p_r_paper), 1.0 + p_r_paper])
    #
    #         q_paper_dist = sl.DirichletDistribution2d(q_paper_alphas)
    #         r_paper_dist = sl.DirichletDistribution2d(r_paper_alphas)
    #
    #         q_paper_opinion = q_paper_dist.as_opinion()
    #         r_paper_opinion = r_paper_dist.as_opinion()
    #
    # # q_paper_op_obj_history.append(q_paper_opinion)
    # # r_paper_op_obj_history.append(r_paper_opinion)
    #
    # if len(Q_hat_paper_history) > 0:
    #     last_q_paper = p_q_paper_history[-1]
    #     last_r_paper = p_r_paper_history[-1]
    #     # last_q_op = q_paper_op_obj_history[-1]
    #     # last_r_op = r_paper_op_obj_history[-1]
    # else:
    #     last_q_paper = 0.0
    #     last_r_paper = 0.0
    #     last_q_op = sl.Opinion(0, 0)
    #     last_r_op = sl.Opinion(0, 0)
    #
    # # if no fresh update was generated this step, repeat last value
    # if len(p_q_paper_history) < i + 1:
    #     p_q_paper_history.append(last_q_paper)
    #     p_r_paper_history.append(last_r_paper)
    #     # q_paper_op_obj_history.append(last_q_op)
    #     # r_paper_op_obj_history.append(last_r_op)



    # Fusion of Opinions
    # fused_kl_ad = sl.Fusion.fuse_opinions(sl.FusionType.AVERAGE, opinion, ad_opinion)
    # fused_kl_ad_2 = opinion.wb_fuse(ad_opinion) #sl.Fusion.fuse_opinions(sl.FusionType.BELIEF_CONSTRAINT, opinion, ad_opinion)
    # fused_overall_q = sl.Fusion.fuse_opinions(sl.FusionType.BELIEF_CONSTRAINT, fused_kl_ad, q_opinion)
    # fused_op_obj_history.append(fused_kl_ad)
    # fused_2_op_obj_history.append(fused_kl_ad_2)

    # ---------------------------------
    # H3: non-Gaussian innovation statistics
    # ---------------------------------
    # h3_opinion = sl.Fusion.fuse_opinions(
    #     sl.FusionType.AVERAGE,
    #     opinion,
    #     ad_opinion
    # )
    h3_opinion = kl_opinion.wb_fuse(ad_opinion)
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

    h_opinions = [h1_opinion,
                  h2_opinion,
                  h3_opinion,
                  h4_opinion,
                  h5_opinion]
    w_all = []
    for h in h_opinions:
        w_all.append(BiOpinion(h.belief(), h.disbelief(), h.prior_belief(), h.uncertainty()))

    global_opinion_ss = fusion_weighted_belief(w_all)
    global_opinion = sl.Opinion(global_opinion_ss.belief[0], global_opinion_ss.belief[1])
    global_op_history.append(global_opinion)

    belief_history.append(global_opinion.belief())
    disbelief_history.append(global_opinion.disbelief())
    uncertainty_history.append(global_opinion.uncertainty())


    # dc = r_opinion.degree_of_conflict(q_opinion)
    # dc_history.append(dc)
    dc = h1_opinion.degree_of_conflict(h2_opinion)
    dc_history.append(dc)

    # # --------------------------------------------------
    # # Four-hypothesis decision model
    # # H0  : no fault
    # # HQ  : Q-only fault
    # # HR  : R-only fault
    # # HQR : joint Q+R fault
    # # --------------------------------------------------
    #
    # # projected probability of the FAULT class
    # p_g = float(fused_kl_ad.getProjection()[1])
    # p_q_raw = float(q_opinion.getProjection()[1])
    #
    # # Q activation: raw Q evidence modulated by nominal system sensitivity
    # tau_q = 0.12
    # s_q = w_Q * p_q_raw
    # a_q = s_q / (s_q + tau_q + 1e-12)
    #
    # # R activation: residual global fault mass not explained by Q
    # p_q_expl = p_g * a_q
    # p_r_rest = max(0.0, p_g - p_q_expl)
    #
    # tau_r = 0.6
    # s_r = w_R * p_r_rest
    # a_r = s_r / (s_r + tau_r + 1e-12)
    #
    # # four hypothesis masses
    # pi_0  = max(0.0, 1.0 - p_g)
    # pi_Q  = p_g * a_q * (1.0 - a_r)
    # pi_R  = p_g * (1.0 - a_q) * a_r
    # pi_QR = p_g * a_q * a_r
    #
    # # normalize for numerical safety
    # pi_vec = np.array([pi_0, pi_Q, pi_R, pi_QR], dtype=float)
    # pi_sum = np.sum(pi_vec)
    # if pi_sum > 0:
    #     pi_vec /= pi_sum
    #
    # pi_0, pi_Q, pi_R, pi_QR = pi_vec
    #
    # decision_labels = ["none", "Q", "R", "Q+R"]
    # decision_idx = int(np.argmax(pi_vec))
    # decision = decision_labels[decision_idx]
    #
    # pi0_history.append(pi_0)
    # piQ_history.append(pi_Q)
    # piR_history.append(pi_R)
    # piQR_history.append(pi_QR)
    # decision_history.append(decision)


    # # --------------------------------------------------
    # # Four-hypothesis decision layer
    # # --------------------------------------------------
    #
    # # global fault probability = fault class
    # p_g = float(fused_kl_ad.getProjection()[1])
    #
    # # raw Q/R probabilities = fault class of respective opinions
    # p_q_raw = float(q_opinion.getProjection()[1])
    # p_r_raw = float(r_opinion.getProjection()[1])  # falls du jetzt r2_opinion nutzt: hier ersetzen
    #
    # # stationary sensitivity weighting
    # s_q = w_Q * p_q_raw
    # s_r = w_R * p_r_raw
    #
    # # saturating activations
    # tau_q = 0.15
    # tau_r = 0.25
    #
    # a_q = s_q / (s_q + tau_q + 1e-12)
    # a_r = s_r / (s_r + tau_r + 1e-12)
    #
    # # -------------------------
    # # attribution within fault space
    # # -------------------------
    # u_q = a_q * (1.0 - a_r)
    # u_r = (1.0 - a_q) * a_r
    # u_qr = a_q * a_r
    #
    # u_sum = u_q + u_r + u_qr
    #
    # if u_sum > 0:
    #     u_q /= u_sum
    #     u_r /= u_sum
    #     u_qr /= u_sum
    # else:
    #     u_q = 0.0
    #     u_r = 0.0
    #     u_qr = 0.0
    #
    # # -------------------------
    # # four hypothesis masses
    # # -------------------------
    # pi_0 = max(0.0, 1.0 - p_g)
    # pi_Q = p_g * u_q
    # pi_R = p_g * u_r
    # pi_QR = p_g * u_qr
    #
    # pi_vec = np.array([pi_0, pi_Q, pi_R, pi_QR], dtype=float)
    # pi_vec /= (np.sum(pi_vec) + 1e-12)
    #
    # pi_0, pi_Q, pi_R, pi_QR = pi_vec
    #
    # decision_labels = ["none", "Q", "R", "Q+R"]
    # decision_idx = int(np.argmax(pi_vec))
    # decision = decision_labels[decision_idx]
    #
    # # histories
    # pi0_history.append(pi_0)
    # piQ_history.append(pi_Q)
    # piR_history.append(pi_R)
    # piQR_history.append(pi_QR)
    # decision_history.append(decision)
    #
    # # convenience histories
    # p_s_history.append(p_g)
    # p_q_history.append(pi_Q + pi_QR)  # total Q involvement
    # p_r_history.append(pi_R + pi_QR)  # total R involvement

    # # --------------------------------------------------
    # # Four-hypothesis decision layer
    # # --------------------------------------------------
    #
    # # global fault probability = fault class
    # p_g = float(fused_kl_ad.getProjection()[1])
    #
    # # raw Q/R probabilities = fault class of respective opinions
    # p_q_raw = float(q_opinion.getProjection()[1])
    # p_r_raw = float(r_opinion.getProjection()[1])  # neue R-iso-opinion
    #
    # # stationary sensitivity weighting
    # s_q = w_Q * p_q_raw
    # s_r = w_R * p_r_raw
    #
    # # ---------------------------------
    # # fair half-activation thresholds
    # # tau_i = w_i * theta_i
    # # ---------------------------------
    # theta_q = 1
    # theta_r = 1
    #
    # tau_q = w_Q * theta_q
    # tau_r = w_R * theta_r
    #
    # a_q = s_q / (s_q + tau_q + 1e-12)
    # a_r = s_r / (s_r + tau_r + 1e-12)
    #
    # # ---------------------------------
    # # overlap-based attribution
    # # ---------------------------------
    # lambda_overlap = 0.5  # 0 -> conservative, 1 -> generous
    #
    # u_qr = lambda_overlap * min(a_q, a_r) + (1.0 - lambda_overlap) * (a_q * a_r)
    # u_q = max(0.0, a_q - u_qr)
    # u_r = max(0.0, a_r - u_qr)
    #
    # u_sum = u_q + u_r + u_qr
    #
    # if u_sum > 0:
    #     u_q /= u_sum
    #     u_r /= u_sum
    #     u_qr /= u_sum
    # else:
    #     u_q = 0.0
    #     u_r = 0.0
    #     u_qr = 0.0
    #
    # # ---------------------------------
    # # four hypothesis masses
    # # ---------------------------------
    # pi_0 = max(0.0, 1.0 - p_g)
    # pi_Q = p_g * u_q
    # pi_R = p_g * u_r
    # pi_QR = p_g * u_qr
    #
    # pi_vec = np.array([pi_0, pi_Q, pi_R, pi_QR], dtype=float)
    # pi_vec /= (np.sum(pi_vec) + 1e-12)
    #
    # pi_0, pi_Q, pi_R, pi_QR = pi_vec
    #
    # decision_labels = ["none", "Q", "R", "Q+R"]
    # decision_idx = int(np.argmax(pi_vec))
    # decision = decision_labels[decision_idx]
    #
    # # histories
    # pi0_history.append(pi_0)
    # piQ_history.append(pi_Q)
    # piR_history.append(pi_R)
    # piQR_history.append(pi_QR)
    # decision_history.append(decision)
    #
    # # convenience histories
    # p_s_history.append(p_g)
    # p_q_history.append(pi_Q + pi_QR)
    # p_r_history.append(pi_R + pi_QR)

    # --------------------------------------------------
    # Four-hypothesis decision layer
    # analytically parameterized, no GT pre-calibration
    # --------------------------------------------------

    # global fault probability = fault class
    p_g = float(global_opinion.getProjection()[0])

    # raw Q/R probabilities = fault class of respective opinions
    p_q_raw = float(q_opinion.getProjection()[1])
    p_r_raw = float(r_opinion.getProjection()[1])  # neue R-iso-opinion

    # effective window size for Q-null approximation
    N_q = max(len(nu_history), 1)

    # ---------------------------------
    # analytically corrected activations
    # ---------------------------------

    # Q-channel: subtract theoretical nominal floor
    p_q0 = 1.0 - np.exp(-1.0 / ((m ** 1.5) * np.sqrt(N_q) + 1e-12))
    a_q = (p_q_raw - p_q0) / (1.0 - p_q0 + 1e-12)
    a_q = float(np.clip(a_q, 0.0, 1.0))

    # R-channel: already floor-corrected in the R-iso test
    # optional tiny safety margin:
    eps_r = 0.0
    a_r = (p_r_raw - eps_r) / (1.0 - eps_r + 1e-12)
    a_r = float(np.clip(a_r, 0.0, 1.0))

    # ---------------------------------
    # attribution within fault space
    # stationary sensitivity weighting enters HERE
    # ---------------------------------
    u_q = w_Q * a_q * (1.0 - a_r)
    u_r = w_R * (1.0 - a_q) * a_r
    u_qr = np.sqrt(w_Q * w_R) * a_q * a_r

    u_sum = u_q + u_r + u_qr

    if u_sum > 0:
        u_q /= u_sum
        u_r /= u_sum
        u_qr /= u_sum
    else:
        u_q = 0.0
        u_r = 0.0
        u_qr = 0.0

    # ---------------------------------
    # four hypothesis masses
    # ---------------------------------
    pi_0 = max(0.0, 1.0 - p_g)
    pi_Q = p_g * u_q
    pi_R = p_g * u_r
    pi_QR = p_g * u_qr

    pi_vec = np.array([pi_0, pi_Q, pi_R, pi_QR], dtype=float)
    pi_vec /= (np.sum(pi_vec) + 1e-12)

    pi_0, pi_Q, pi_R, pi_QR = pi_vec

    decision_labels = ["none", "Q", "R", "Q+R"]
    decision_idx = int(np.argmax(pi_vec))
    decision = decision_labels[decision_idx]

    # histories
    pi0_history.append(pi_0)
    piQ_history.append(pi_Q)
    piR_history.append(pi_R)
    piQR_history.append(pi_QR)
    decision_history.append(decision)

    # convenience histories
    p_s_history.append(p_g)
    p_q_history.append(pi_Q + pi_QR)
    p_r_history.append(pi_R + pi_QR)


# plt.plot(list(range(len(cum_u))), cum_u)
# plt.title("cum_u")
# plt.figure()
# plt.plot(list(range(len(C_history))), C_history)
# plt.title("C_history")
# plt.figure()
# plt.plot(ad_history)
# plt.title("AD score")
# plt.figure()
# plt.plot(h2_u_values, label="H2 u history")
# plt.plot(u_history, label="H3 u history")
# plt.title("Score history")
# plt.legend()
# plt.figure()
# plt.plot(h2_C_history, label="H2 C-value")
# plt.plot(C_history, label="H3 C-value")
# plt.plot(h2_D_history, label="H2 D-value")
# # plt.plot(h2_T_history, label="H2 T-value")
# plt.plot(h2_valid_count_history, label="H2 Valid Count")
# plt.title("Test history")
# plt.legend()
plt.figure()
# plt.plot(belief_history, label="Belief")
# plt.plot(disbelief_history, label="Disbelief")
# plt.plot(uncertainty_history, label="Uncertainty")
plt.plot([0, len(belief_history)], [0.5, 0.5], "r--")
plt.plot(p_s_history, label="Projection (global)", linewidth=3, zorder=100)
plt.plot([h1.getProjection()[0] for h1 in h1_r_op_history], label="Projection (H1)")
plt.plot([h2.getProjection()[0] for h2 in h2_q_op_history], label="Projection (H2)")
plt.plot([h3.getProjection()[0] for h3 in h3_ng_op_history], label="Projection (H3)")
plt.plot([h4.getProjection()[0] for h4 in h4_bias_op_history], label="Projection (H4)")
plt.plot([h5.getProjection()[0] for h5 in h5_white_op_history], label="Projection (H5)")
plt.title("Opinions")
plt.legend()
# plt.figure()
# plt.plot(dc_history, label="Degree of Conflict (global vs Q)")
# plt.plot(p_s_history, label="Projection (global)")
# plt.plot(p_r_history, label="Projection (R)")
# plt.plot(p_q_history, label="Projection (Q)")
# plt.title("Projection comparison")
# plt.legend()
plt.figure()
plt.plot(decision_history, label="Decision history")
plt.legend()
plt.title("Hypotheses decision")
# plt.figure()
# plt.plot(pi0_history, label="Pi 0")
# plt.plot(piQ_history, label="Pi Q")
# plt.plot(piR_history, label="Pi R")
# plt.plot(piQR_history, label="Pi QR")
# plt.legend()
# plt.title("Pi comparison")

# plt.figure()
# plt.plot(kl_C_history, label="C (KL method)")
# plt.plot(ks_history, label="KS statistic")
# plt.plot(ad_history, label="AD statistic")
# plt.plot([0, len(ad_history)-1], [0.05, 0.05], 'r--')
# # plt.plot(cvm_history, label="CvM statistic")
# plt.legend()
# plt.title("Uniformity tests (statistics)")
#
# plt.figure()
# plt.plot(e_beta_history, label="KL")
# plt.plot(ad_e_beta_history, label="AD")
# # plt.plot(ks_e_beta_history, label="KS")
# plt.plot(q_e_beta_history, label="Q")
# plt.plot(q2_e_beta_history, label="Q2")
# plt.legend()
# plt.title("Comparison test-based Beta evidence")
"""
==============
GOSPA PLOTTING
==============
"""
from matplotlib.dates import num2date, SecondLocator, MicrosecondLocator
from matplotlib.ticker import FuncFormatter, MultipleLocator
def plot_gospa(gospa_metrics, gospa_gen_name: str | list[str]):

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

        ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((3, 4), (2, 1), colspan=2)
        ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)


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

        ax3.set_title("Switching")
        # ax3.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax3.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax3.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax3.set_xlabel("t in seconds")
        ax3.plot(gospa_switching)

        ax4.set_title("Missed")
        # ax4.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax4.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax4.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax4.set_xlabel("t in seconds")
        ax4.plot(gospa_missed)

        ax5.set_title("False")
        # ax5.xaxis.set_major_locator(SecondLocator(interval=10))
        # ax5.xaxis.set_major_formatter(FuncFormatter(format_date))
        # ax5.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax5.set_xlabel("t in seconds")
        ax5.plot(gospa_false)

        plt.tight_layout(pad=0, h_pad=-1.4)

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

stone_soup_tracker = True

c=10
p=2


from stonesoup.metricgenerator.ospametric import GOSPAMetric
# GOSPA Stone Soup
if stone_soup_tracker: gospa_kalman = GOSPAMetric(c=c, p=p, generator_name='GOSPA',
                            tracks_key='tracks',  truths_key='truths', switching_penalty=1)


# Use the track associator
associator = TrackToTruth(association_threshold=30)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([gospa_kalman],
                              associator)

metric_manager.add_data({'truths' : {truth}}, overwrite=False)
metric_manager.add_data({'tracks' : {track}}, overwrite=False)
# print("Number of Stone Soup Tracks:", len(tracks))

# %%
metrics = metric_manager.generate_metrics()

plt = plot_gospa(metrics, gospa_gen_name=["GOSPA"])
# %%
# Plot the resulting track, including uncertainty ellipses
plotter.plot_tracks(track, [0, 2], uncertainty=True)
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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
show_all_opinion = False
fig = plotter.fig

fig.set_subplots(
    rows=4, cols=5,
    specs=[
        [{"colspan": 2, "rowspan": 2},  None,                   None,                   {"colspan": 2, "rowspan": 2, "type": "ternary"}, None],              # Zeile 1
        [None,                          None,                   None,                   None,                   None                ],        # Zeile 2
        [{"type": "ternary"},           {"type": "ternary"},    {"type": "ternary"},    {"type": "ternary"},    {"type": "ternary"}],        # Zeile 3
        [None,                          None,                   {"type": "xy"},         {"type": "xy"},         {"type": "xy"}      ]
    ],
    subplot_titles=[
        "Track", "Global Opinion",
        "H1 Opinion", "H2 Opinion", "H3 Opinion", "H4 Opinion", "H5 Opinion",
        # "H2 Histogram",
        "H3 Histogram", #"H4 Histogram", "H5 Histogram"
    ],
    vertical_spacing=0.08 ,
    horizontal_spacing=0.01
)

for trace in fig.data:
    trace.update(xaxis="x1", yaxis="y1")

b0_f, d0_f, u0_f = global_op_history[0].belief(), global_op_history[0].disbelief(), global_op_history[0].uncertainty()
# b0_f2, d0_f2, u0_f2 = fused_2_op_obj_history[0].belief(), fused_2_op_obj_history[0].disbelief(), fused_2_op_obj_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u0_f], #, u0_f2],
        b=[d0_f], #, d0_f2],
        c=[b0_f], #, b0_f2],

        mode='markers',
        marker=dict(size=[14, 14], color=['purple', 'cyan']),
        hovertemplate=["F1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "F2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="Global Opinion"
    ),
    row=1, col=4
)

prior = 0.5  # oder: opinion.prior_belief_masses[0]

P0 = b0_f + prior * u0_f

fig.add_trace(
    go.Scatterternary(
        a=[0],
        b=[1 - P0],
        c=[P0],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Projected Probability',
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
                line=dict(color='green', dash='dot'),
                showlegend=False
            ),
    row=1, col=4
)

b_h1, d_h1, u_h1 = h1_r_op_history[0].belief(), h1_r_op_history[0].disbelief(), h1_r_op_history[0].uncertainty()
b_h11, d_h11, u_h11 = h1_1_op_history[0].belief(), h1_1_op_history[0].disbelief(), h1_1_op_history[0].uncertainty()
b_h12, d_h12, u_h12 = h1_2_op_history[0].belief(), h1_2_op_history[0].disbelief(), h1_2_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h1, u_h11, u_h12] if show_all_opinion else [u_h1],
        b=[d_h1, d_h11, d_h12]if show_all_opinion else [d_h1],
        c=[b_h1, b_h11, b_h12]if show_all_opinion else [b_h1],
        mode='markers',
        marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
        hovertemplate=["H1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H1.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H1.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H1"
    ),
    row=3, col=1
)

b_h2, d_h2, u_h2 = h2_q_op_history[0].belief(), h2_q_op_history[0].disbelief(), h2_q_op_history[0].uncertainty()
# b_h21, d_h21, u_h21 = h2_1_op_history[0].belief(), h2_1_op_history[0].disbelief(), h2_1_op_history[0].uncertainty()
# b_h22, d_h22, u_h22 = h2_2_op_history[0].belief(), h2_2_op_history[0].disbelief(), h2_2_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h2], #, u_h21, u_h22],
        b=[d_h2], #, d_h21, d_h22],
        c=[b_h2], #, b_h21, b_h22],
        mode='markers',
        marker=dict(size=14, color=['green']), #, 'yellow', 'cyan']),
        hovertemplate=["H2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"], #, "H2.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "H2.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H2"
    ),
    row=3, col=2
)

b0, d0, u0 = opinions[0]
# b0_ks, d0_ks, u0_ks = ks_op_obj_history[0].belief(), ks_op_obj_history[0].disbelief(), ks_op_obj_history[0].uncertainty()
b0_ad, d0_ad, u0_ad = opinions_ad[0]
b_h3, d_h3, u_h3 = h3_ng_op_history[0].belief(), h3_ng_op_history[0].disbelief(), h3_ng_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h3, u0, u0_ad] if show_all_opinion else [u_h3],
        b=[d_h3, d0, d0_ad] if show_all_opinion else [d_h3],
        c=[b_h3, b0, b0_ad] if show_all_opinion else [b_h3],
        mode='markers',
        marker=dict(size=14, color=['green', 'cyan', 'yellow'] if show_all_opinion else ['green']),
        hovertemplate=["H3<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "KL<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "AD<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"]  if show_all_opinion else ["H3<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H3"
    ),
    row=3, col=3
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




b_h4, d_h4, u_h4 = h4_bias_op_history[0].belief(), h4_bias_op_history[0].disbelief(), h4_bias_op_history[0].uncertainty()
b_h41, d_h41, u_h41 = h4_1_op_history[0].belief(), h4_1_op_history[0].disbelief(), h4_1_op_history[0].uncertainty()
b_h42, d_h42, u_h42 = h4_2_op_history[0].belief(), h4_2_op_history[0].disbelief(), h4_2_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h4, u_h41, u_h42] if show_all_opinion else [u_h4],
        b=[d_h4, d_h41, d_h42] if show_all_opinion else [d_h4],
        c=[b_h4, b_h41, b_h42] if show_all_opinion else [b_h4],
        mode='markers',
        marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
        hovertemplate=["H4<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H4.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H4.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H4<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H4"
    ),
    row=3, col=4
)


# b0, d0, u0 = q_op_obj_history[0].belief(), q_op_obj_history[0].disbelief(), q_op_obj_history[0].uncertainty()
# b0_r, d0_r, u0_r = r_op_obj_history[0].belief(), r_op_obj_history[0].disbelief(), r_op_obj_history[0].uncertainty()
# # b0_q2, d0_q2, u0_q2 = q2_op_obj_history[0].belief(), q2_op_obj_history[0].disbelief(), q2_op_obj_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u0, u0_r], #, u0_q2],
#         b=[d0, d0_r], #, d0_q2],
#         c=[b0, b0_r], #, b0_q2],
#         mode='markers',
#         #subplot="ternary",
#         marker=dict(size=14, color=['red', 'blue', 'purple']),
#         hovertemplate=["Q<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "R<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "Q2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"]
#     ),
#     row=3, col=4
# )

# b0_pq, d0_pq, u0_pq = q_paper_op_obj_history[0].belief(), q_paper_op_obj_history[0].disbelief(), q_paper_op_obj_history[0].uncertainty()
# b0_pr, d0_pr, u0_pr = r_paper_op_obj_history[0].belief(), r_paper_op_obj_history[0].disbelief(), r_paper_op_obj_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u0_pq, u0_pr], b=[d0_pq, d0_pr], c=[b0_pq, b0_pr],
#         mode='markers',
#         marker=dict(size=[14, 14], color=['red', 'blue']),
#         hovertemplate=["Paper Q<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>", "Paper R<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
#         name="Paper Fusion"
#     ),
#     row=3, col=5
# )

b_h5, d_h5, u_h5 = h5_white_op_history[0].belief(), h5_white_op_history[0].disbelief(), h5_white_op_history[0].uncertainty()
b_h51, d_h51, u_h51 = h5_1_op_history[0].belief(), h5_1_op_history[0].disbelief(), h5_1_op_history[0].uncertainty()
b_h52, d_h52, u_h52 = h5_2_op_history[0].belief(), h5_2_op_history[0].disbelief(), h5_2_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h5, u_h51, u_h52] if show_all_opinion else [u_h5],
        b=[d_h5, d_h51, d_h52] if show_all_opinion else [d_h5],
        c=[b_h5, b_h51, b_h52] if show_all_opinion else [b_h5],
        mode='markers',
        marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
        hovertemplate=["H5<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H5.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "H5.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H5<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H5"
    ),
    row=3, col=5
)

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
        name="H3 counts",
        marker=dict(color='cyan')

    ),
    row=4, col=3
)

# fig.add_trace(
#     go.Bar(
#         x=list(range(M)),
#         y=h4_count_history[0],
#         name="H4 counts",
#         marker=dict(color='cyan')
#
#     ),
#     row=4, col=4
# )
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
fig.update_yaxes(range=[0, M**2], row=4, col=3)
# fig.update_yaxes(range=[0, M**2], row=4, col=4)
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
    b_f, d_f, u_f = global_op_history[i].belief(), global_op_history[i].disbelief(), global_op_history[
        i].uncertainty()
    # b_f2, d_f2, u_f2 = fused_2_op_obj_history[i].belief(), fused_2_op_obj_history[i].disbelief(), fused_2_op_obj_history[i].uncertainty()
    # b_ks, d_ks, u_ks = b0_ks, d0_ks, u0_ks#ks_op_obj_history[i].belief(), ks_op_obj_history[i].disbelief(), ks_op_obj_history[i].uncertainty()
    b_q, d_q, u_q = q_op_obj_history[i].belief(), q_op_obj_history[i].disbelief(), q_op_obj_history[i].uncertainty()
    # b_q2, d_q2, u_q2 = q2_op_obj_history[i].belief(), q2_op_obj_history[i].disbelief(), q2_op_obj_history[i].uncertainty()
    b_r, d_r, u_r = r_op_obj_history[i].belief(), r_op_obj_history[i].disbelief(), r_op_obj_history[i].uncertainty()
    b, d, u = opinions[i]
    b_ad, d_ad, u_ad = opinions_ad[i]
    y_i = dirichlet_pdfs[i]
    P = b_f + prior * u_f
    counts_i = counts_history[i]
    # counts_h2 = h2_count_history[i]
    counts_h4 = h4_count_history[i]
    counts_h5 = h5_count_history[i]
    # counts_R_i = counts_R_history[i]

    b_h1, d_h1, u_h1 = h1_r_op_history[i].belief(), h1_r_op_history[i].disbelief(), h1_r_op_history[i].uncertainty()
    b_h11, d_h11, u_h11 = h1_1_op_history[i].belief(), h1_1_op_history[i].disbelief(), h1_1_op_history[i].uncertainty()
    b_h12, d_h12, u_h12 = h1_2_op_history[i].belief(), h1_2_op_history[i].disbelief(), h1_2_op_history[i].uncertainty()
    b_h2, d_h2, u_h2 = h2_q_op_history[i].belief(), h2_q_op_history[i].disbelief(), h2_q_op_history[i].uncertainty()
    # b_h21, d_h21, u_h21 = h2_1_op_history[i].belief(), h2_1_op_history[i].disbelief(), h2_1_op_history[i].uncertainty()
    # b_h22, d_h22, u_h22 = h2_2_op_history[i].belief(), h2_2_op_history[i].disbelief(), h2_2_op_history[i].uncertainty()
    b_h3, d_h3, u_h3 = h3_ng_op_history[i].belief(), h3_ng_op_history[i].disbelief(), h3_ng_op_history[i].uncertainty()
    b_h4, d_h4, u_h4 = h4_bias_op_history[i].belief(), h4_bias_op_history[i].disbelief(), h4_bias_op_history[i].uncertainty()
    b_h41, d_h41, u_h41 = h4_1_op_history[i].belief(),  h4_1_op_history[i].disbelief(), h4_1_op_history[i].uncertainty()
    b_h42, d_h42, u_h42 = h4_2_op_history[i].belief(), h4_2_op_history[i].disbelief(), h4_2_op_history[i].uncertainty()
    b_h5, d_h5, u_h5 = h5_white_op_history[i].belief(), h5_white_op_history[i].disbelief(), h5_white_op_history[
        i].uncertainty()
    b_h51, d_h51, u_h51 = h5_1_op_history[i].belief(), h5_1_op_history[i].disbelief(), h5_1_op_history[i].uncertainty()
    b_h52, d_h52, u_h52 = h5_2_op_history[i].belief(), h5_2_op_history[i].disbelief(), h5_2_op_history[i].uncertainty()

    b_pq, d_pq, u_pq = q_paper_op_obj_history[i].belief(), q_paper_op_obj_history[i].disbelief(), \
    q_paper_op_obj_history[i].uncertainty()
    b_pr, d_pr, u_pr = r_paper_op_obj_history[i].belief(), r_paper_op_obj_history[i].disbelief(), \
    r_paper_op_obj_history[i].uncertainty()

    # Bestehende Daten behalten + erweitern
    new_data = list(frame.data)

    new_data.append(
        go.Scatterternary(a=[u_f], #, u_f2],
                          b=[d_f], #, d_f2],
                          c=[b_f], #, b_f2],
                          cliponaxis=False)
    )

    new_data.append(go.Scatterternary(a=[0], b=[1 - P], c=[P], cliponaxis=False))

    new_data.append(
        go.Scatterternary(a=[u_f, 0], b=[d_f, 1 - P], c=[b_f, P], mode='lines', line=dict(color='green', dash='dot'),
                          showlegend=False, cliponaxis=False))

    new_data.append(
        go.Scatterternary(a=[u_h1, u_h11, u_h12] if show_all_opinion else [u_h1],
                          b=[d_h1, d_h11, d_h12] if show_all_opinion else [d_h1],
                          c=[b_h1, b_h11, b_h12] if show_all_opinion else [b_h1],
                          cliponaxis=False)
    )
    new_data.append(
        go.Scatterternary(a=[u_h2], #, u_h21, u_h22],
                          b=[d_h2], #, d_h21, d_h22],
                          c=[b_h2], #, b_h21, b_h22],
                          cliponaxis=False)
    )
    new_data.append(
        go.Scatterternary(a=[u_h3, u, u_ad] if show_all_opinion else [u_h3],
                          b=[d_h3, d, d_ad] if show_all_opinion else [d_h3],
                          c=[b_h3, b, b_ad] if show_all_opinion else [b_h3],
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
        go.Scatterternary(a=[u_h4, u_h41, u_h42] if show_all_opinion else [u_h4],
                          b=[d_h4, d_h41, d_h42] if show_all_opinion else [d_h4],
                          c=[b_h4, b_h41, b_h42] if show_all_opinion else [b_h4],
                          cliponaxis=False)
    )
    new_data.append(
        go.Scatterternary(a=[u_h5, u_h51, u_h52] if show_all_opinion else [u_h5],
                          b=[d_h5, d_h51, d_h52] if show_all_opinion else [d_h5],
                          c=[b_h5, b_h51, b_h52] if show_all_opinion else [b_h5],
                          cliponaxis=False)
    )

    # new_data.append(
    #     go.Scatterternary(a=[u_pq, u_pr], b=[d_pq, d_pr], c=[b_pq, b_pr], cliponaxis=False)
    # )

    # new_data.append(go.Bar(x=list(range(M)), y=counts_h2))
    new_data.append(go.Bar(x=list(range(M)), y=counts_i))
    # new_data.append(go.Bar(x=list(range(M)), y=counts_h4))
    # new_data.append(go.Bar(x=list(range(M)), y=counts_h5))



    new_frames.append(go.Frame(data=new_data, name=frame.name))

fig.frames = new_frames

sliders = [dict(
    steps=[
        dict(
            method='animate',
            args=[[str(i)],
                  dict(mode='immediate',
                       frame=dict(duration=1000, redraw=True),
                       transition=dict(duration=0))],
            label=str(i)
        )
        for i in range(len(fig.frames))
    ],
    currentvalue=dict(prefix="Step: "),
    pad=dict(t=30),
)]

fig.update_layout(sliders=sliders)

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
    if "Opinion" in ann.text:
        ann.update(x=ann.x - 0.1, y=ann.y - 0.05, xanchor='left', align='left')

plotter.fig.show(renderer="browser")
plotter.show()
plt.show()