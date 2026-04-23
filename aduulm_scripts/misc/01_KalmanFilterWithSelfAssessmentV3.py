#!/usr/bin/env python

"""
==========================================================
1 - An introduction to Stone Soup: using the Kalman filter
==========================================================
"""

# %%
import numpy as np
from datetime import datetime, timedelta
import subjective_logic as sl
import matplotlib
from copy import deepcopy
from stonesoup.types.prediction import GaussianStatePrediction, MeasurementPrediction
from collections import deque
from dataclasses import dataclass

matplotlib.use('TkAgg')

# %%
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now().replace(microsecond=0)

np.random.seed(1991+5)  # 1991

# %%
# factor = 100
q_x = 1 #0.1
q_y = 1 #0.1
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
disturbance_factor_process = 16 #16
# Disturbance configurations for ground truth generation
gt_transition_configs = {
    'noise_diff_coeff': [[q_x, q_y]],  # for transition model gt
    'disturbance_mode': ['jump'],
    # 'parameters': [[[150, disturbance_factor_process], [200, 1/disturbance_factor_process], [250, disturbance_factor_process], [300, 1/disturbance_factor_process]]] #, [[99, 1/100]]]
    'parameters': [[[500, disturbance_factor_process], [800, 1/disturbance_factor_process]]]
}
process_noise_coeff_memory = [[], []]

num_steps = 900
# np.random.seed(1991)
for k in range(1, num_steps + 1):

    timesteps.append(start_time+timedelta(seconds=k/10))  # add next timestep to list of timesteps

    ###################################################################################################
    # Disturb the transition model based on the disturbance modes
    transition_model = disturbance_transition_model(transition_model, gt_transition_configs, k)
    # Save the noise coefficients (process noise memory)
    for i, model in enumerate(transition_model.model_list):
        # Collect covariance matrices for each model component
        process_noise_coeff_memory[i].append(model.noise_diff_coeff)
    ###################################################################################################

    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=.1)),
        timestamp=timesteps[k]))

# %%
from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=1) #, height=300) #, height=1000)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig


# %%
# We can check the :math:`F_k` and :math:`Q_k` matrices (generated over a 1s period).
transition_model.matrix(time_interval=timedelta(seconds=.1))

# %%
transition_model.covar(time_interval=timedelta(seconds=.1))

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
    noise_covar=np.array([[.2, 0],  # Covariance matrix for Gaussian PDF
                          [0, .2]])
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
disturbance_factor_meas = 2 #4
gt_measurement_configs = {
    # 'disturbance_mode': ['jump', 'drift', 'outliers'],
    # 'parameters': [[[50, 2], [100, 0.5]], [[150, 250, 2.5], [250, 300, 0.4]], [[350, 400, 2, 5]]]
    'disturbance_mode': ['jump'],
    'parameters': [[[100, disturbance_factor_meas], [200, 1/disturbance_factor_meas], [300, 1/disturbance_factor_meas], [400, disturbance_factor_meas]]]
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

from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
from stonesoup.subjective_logic.subjective_logic import BiOpinion, fusion_weighted_belief
from stonesoup.selfassessor._threshold import calc_threshold_op_diff, calc_threshold_n_diff
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
                                  trust_discount=sa_settings["trust_discount"],
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

def calculate_lt_evidence(W: int, alpha=0.99):
    return int((-(W - 1) + np.sqrt((W-1)**2 + ((4 * W) / (1 - alpha)))) / (2))


# eSLIM++ LTST Buffer
SHORT_WINDOW_SIZE = 35
W = 7
DISCOUNT = 0.99

griebel_threshold = calc_threshold_n_diff(W, SHORT_WINDOW_SIZE, 0.01)
print("Griebels threshold:", griebel_threshold)
# THRESHOLD = calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT), 0.01)
THRESHOLD = calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT), 0.01)
print("Threshold for LTST:", THRESHOLD)
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

ops = []
ops_per_timestep = []

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

def scalar_u_to_opinion(u: float, W: int):
    """
    Map a scalar u in [0,1] to a W-dimensional one-hot evidence opinion.
    """
    u = float(np.clip(u, 0.0, 1.0))
    evidence = np.zeros(W, dtype=float)

    # Bin index in {0, ..., W-1}
    idx = min(int(np.floor(u * W)), W - 1)
    evidence[idx] = 1.0

    dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(evidence)
    return dist.as_opinion()
# %%
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([0.5, 1, 0.5, 1]), timestamp=start_time)

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
kl_opinion = sl.Opinion(0, 0)

# ------------------------------------------------------------------
# Additional diagnostic channels
# ------------------------------------------------------------------
from scipy.stats import norm, chi2
from collections import deque


# Per-timestep opinions for new channels
ops_component_x = []
ops_component_y = []


for i, measurement in enumerate(measurements):
    prediction: GaussianStatePrediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
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
    # print(selfassessor.threshold_ltst)
    # assert np.isclose(selfassessor.threshold_ltst, selfassessor._threshold_dc)

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
    # print(K)
    K_history.append(K)

    eta = (measurement.state_vector.reshape(-1, 1) - H @ post.state_vector).flatten()
    eta_history.append(eta)
    if len(eta_history) > max_buffer:
        eta_history.pop(0)

    # -----------------------------
    # Whitening
    # -----------------------------
    S_sqrt = np.linalg.cholesky(S)
    nu_white = np.linalg.solve(S_sqrt, delta).flatten()
    nu_white_history.append(nu_white.copy())
    if len(nu_white_history) > max_buffer:
        nu_white_history.pop(0)

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
    ops_per_timestep.append(one_time_op)

    counts_history.append(list(counts))
    #
    # N = np.sum(counts)
    #
    # p = counts / N
    #
    # # numerisch stabil
    # p_safe = np.clip(p, 1e-12, 1.0)
    #
    # H = -1 * np.sum(p_safe * np.log2(p_safe))
    # H_max = np.log2(M)
    #
    # C = H / H_max  # ∈ [0,1]

    # KL-Divergence
    # kl_C = -H + H_max
    # kl_C_history.append(1 - C)


    # C_history.append(C)
    # N_eff = (np.sum(counts))
    # # C-scale as transformation of KL-divergence
    #
    # p_safe = np.clip(p, 1e-12, 1.0)
    # H = -np.sum(p_safe * np.log2(p_safe))
    # C = H / np.log2(M)
    # kl_C_history.append(1 - C)
    #
    # s = 1.0 - C  # Inkonsistenzmaß
    # gamma = 1  # <1 macht sensitiver bei kleinen Abweichungen
    # s_tilde = s ** gamma
    #
    # e_beta = N_eff * s_tilde
    # e_alpha = N_eff * (1.0 - s_tilde)
    #
    # evidence2d = np.array([e_alpha, e_beta])
    #
    # alphas = np.array([1.0, 1.0]) + [e_alpha, e_beta]

    # e_alpha = C *N_eff
    # e_beta = (1 - C) *N_eff

    # N_eff = len(u_buffer)

    # alphas *= forget_param
    # alphas += np.array([e_alpha, e_beta])
    # alphas = np.array([1.0, 1.0]) + [e_alpha, e_beta]
    # alphas = np.maximum(alphas, 1)
    # print(i, ": ", C, N_eff, e_alpha, e_beta)
    # e_beta_history.append(e_beta)
    # # -----------------------------
    # # Subjective Logic Opinion
    # # -----------------------------
    # z_dist = sl.DirichletDistribution2d.from_evidences(evidence2d)
    # z_op = z_dist.as_opinion()
    # ops.append(z_op)
    # dist = sl.DirichletDistribution2d(alphas)

    # pdf = list(map(dist.evaluate, x_pdf))
    # # pdf = beta.pdf(x_pdf, alphas[0], alphas[1])
    # dirichlet_pdfs.append(pdf)
    #
    # # kl_opinion = dist.as_opinion()
    # if len(u_buffer) == max_buffer:
    #     kl_opinion = dist.as_opinion()
        # z_dist = sl.DirichletDistribution2d.from_evidences(evidence2d)
        # z_op = z_dist.as_opinion()
        # ops.append(z_op)
        # ops.append(kl_opinion)

    # uncertainty maximized
    # projected = kl_opinion.getProjection()
    # u_max = min(projected[0]/kl_opinion.prior_belief_masses[0], projected[1]/kl_opinion.prior_belief_masses[1])
    # b_max = projected[0] - kl_opinion.prior_belief_masses[0] * u_max
    # d_max = projected[1] - kl_opinion.prior_belief_masses[1] * u_max
    # opinion_max = sl.Opinion(b_max, d_max)
    # r_paper_op_obj_history.append(opinion_max)
    # q_paper_op_obj_history.append(opinion_max)
    # kl_opinion = opinion_max

    # opinion.prior_belief_masses = [1.0, 0]
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
    global_opinion = h_opinions[0]
    global_op_history.append(global_opinion)

    # belief_history.append(global_opinion.belief())
    # disbelief_history.append(global_opinion.disbelief())
    # uncertainty_history.append(global_opinion.uncertainty())
    belief_history.append(kl_opinion.belief())
    disbelief_history.append(kl_opinion.disbelief())
    uncertainty_history.append(kl_opinion.uncertainty())


    # dc = r_opinion.degree_of_conflict(q_opinion)
    # dc_history.append(dc)

    # dc_history.append(dc)


# plt.plot(list(range(len(C_history))), C_history)
# plt.title("C_history")
# plt.figure()
# plt.plot(belief_history, label="Belief")
# # plt.plot(disbelief_history, label="Disbelief")
# plt.plot(uncertainty_history, label="Uncertainty")
# plt.plot(p_s_history, label="Projection")
# plt.legend()
# plt.title("Global Opinion")
# plt.figure()
# plt.plot(decision_history, label="Decision")
# plt.figure()
# plt.plot([0, len(belief_history)], [0.5, 0.5], "r--")
# plt.plot(p_s_history, label="Projection (global)", linewidth=3, zorder=100)
# plt.plot([h1.getProjection()[0] for h1 in h1_r_op_history], label="Projection (H1)")
# plt.plot([h2.getProjection()[0] for h2 in h2_q_op_history], label="Projection (H2)")
# plt.plot([h3.getProjection()[0] for h3 in h3_ng_op_history], label="Projection (H3)")
# plt.plot([h4.getProjection()[0] for h4 in h4_bias_op_history], label="Projection (H4)")
# plt.plot([h5.getProjection()[0] for h5 in h5_white_op_history], label="Projection (H5)")
# plt.title("Opinions")
# plt.legend()


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

entropy_opinions = []
sums_of_evidence = []
sums_of_evidence_comp = []
evidences = []

for idx, op_obs in tqdm.tqdm(enumerate(ops_per_timestep), total=len(ops_per_timestep)):
    ltst.add(op_obs)
    op_buffer = ltst.get_opinion()
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

    th_dc[idx] = calc_threshold_n_diff(W, sum(op_buffer.as_dirichlet().evidences), 0.1)
    print("Sum of Evidence:", sum(op_buffer.as_dirichlet().evidences))
    sums_of_evidence.append(int(sum(op_buffer.as_dirichlet().evidences)))
    evidences.append(list(op_buffer.as_dirichlet().evidences))

    entropy = -1 * np.sum(op_buffer.getProjection() * np.log2(op_buffer.getProjection()))
    entropy /= np.log2(W)
    entropy = 1 - entropy
    entropy_opinions.append(entropy)

for opx, opy in zip(
    ops_component_x, ops_component_y,
):
    ltst_component_x.add(opx)
    ltst_component_y.add(opy)


    component_x_buffered.append(ltst_component_x.get_opinion())
    component_y_buffered.append(ltst_component_y.get_opinion())

# dc_white_x = [white_x.degree_of_conflict(op_ref) for white_x in whiteness_x_buffered]
# dc_white_y = [white_y.degree_of_conflict(op_ref) for white_y in whiteness_y_buffered]
dc_comp_x = [comp_x.degree_of_conflict(op_ref) for comp_x in component_x_buffered]
dc_comp_y = [comp_y.degree_of_conflict(op_ref) for comp_y in component_y_buffered]
component_buffered = []
whiteness_buffered = []

for opx, opy in zip(component_x_buffered, component_y_buffered):
    fused_comp = sl.Fusion.fuse_opinions(sl.FusionType.BELIEF_CONSTRAINT, [opx, opy])
    component_buffered.append(fused_comp)
    sums_of_evidence_comp.append(int(sum(fused_comp.as_dirichlet().evidences)))

# for opx, opy in zip(whiteness_x_buffered, whiteness_y_buffered):
#     fused_white = sl.Fusion.fuse_opinions(sl.FusionType.BELIEF_CONSTRAINT, [opx, opy])
#     whiteness_buffered.append(fused_white)

# dc_whiteness = [white.degree_of_conflict(op_ref) for white in whiteness_buffered]
dc_comp = [comp.degree_of_conflict(op_ref) for comp in component_buffered]

# for i, _ in enumerate(dc_st_lt):
#     assert np.isclose(dc_ltst[i], dc_st_lt[i]), f"{i}, {dc_ltst[i]}, {dc_st_lt[i]}"


# plt.figure()
# plt.plot(list(np.linspace(0, 0.5, 500)), list(map(lambda x: calc_threshold_n_diff(5, 58.62, x), list(np.linspace(0, 0.5, 500)))), label='th')
# plt.legend()
# plt.title("Threshold evaluation")

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

import json
with open("entropy_threshold.json", 'r') as f:
    entropy_thresholds = json.load(f)
with open("opinion_threshold_smoothed.json", 'r') as f:
    opinion_thresholds = json.load(f)

plt.figure()

# plt.plot(resets, label="Reset")
plt.plot(dc_ref, label="DC Radial")
# plt.plot(dc_ltst, label="DC LTST")
# plt.plot(entropy_opinions, label="Entropy")
# plt.plot([calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT)+SHORT_WINDOW_SIZE, alpha=0.01)]*len(entropy_opinions), label = "Entropy Th LT+ST")
# plt.plot([calibrate_entropy_threshold(W, calculate_lt_evidence(W, DISCOUNT), alpha=0.01)]*len(entropy_opinions), label = "Entropy Th LT")
# plt.plot([entropy_thresholds[f"{W}, {s}, 0.01"] for s in sums_of_evidence], label = "Entropy Th Dynamic")
plt.plot([opinion_thresholds[f"{W}, {s}, 0.005"] for s in sums_of_evidence], label = "Th Radial")
# plt.plot([calc_threshold_n_diff(W, s, 0.1) for s in sums_of_evidence], label = "Griebel Th Dynamic")
# plt.plot(kl_C_history, label="KL C")
plt.yticks(np.linspace(0, 1, 11))
plt.grid()
plt.legend()
plt.title("DCs")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# --- Radial ---
ax1.plot(dc_ref, label="DC Radial")
ax1.plot([opinion_thresholds[f"{W}, {s}, 0.005"] for s in sums_of_evidence],
         label="Th Radial")
ax1.set_title("Radial")
ax1.grid()
ax1.legend()
# --- Comp ---
ax2.plot(dc_comp, label="DC Comp")
ax2.plot([opinion_thresholds[f"{W}, {min(s, 150)}, 0.005"] for s in sums_of_evidence_comp],
         label="Th Comp")
ax2.set_title("Comp")
ax2.grid()
ax2.legend()
# Gemeinsamer Titel
fig.suptitle("DC Global")
plt.tight_layout()

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
plt.plot(dc_comp_x, label="DC Comp X")
plt.plot(dc_comp_y, label="DC Comp Y")
plt.legend()
plt.grid()
plt.title("DC Local")

# plt.figure()
# plt.plot([chisquare_uniform_test(e, 0.01)['reject_H0'] for e in evidences], label="Reject H0")
# plt.legend()
# plt.title("Chi-Square")
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
        [{"colspan": 3, "rowspan": 3},  None,                   None,                   {"colspan": 2, "rowspan": 2, "type": "ternary"}, None],              # Zeile 1
        [None,                          None,                   None,                   None,                   None                ],        # Zeile 2
        [None,                          None,                   None,                   {"type": "ternary"},    None                ],        # Zeile 3
        [None,                          None,                   None,                   {"type": "xy"},         {"type": "xy"}      ]
    ],
    subplot_titles=[
        "Track", "Global Opinion",
        # "H1 Opinion", "H2 Opinion",
        "H3 Opinion",
        # "H4 Opinion", "H5 Opinion",
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

b0, d0, u0 = opinions[0]
# b0_ks, d0_ks, u0_ks = ks_op_obj_history[0].belief(), ks_op_obj_history[0].disbelief(), ks_op_obj_history[0].uncertainty()
# b0_ad, d0_ad, u0_ad = opinions_ad[0]
b_h3, d_h3, u_h3 = h3_ng_op_history[0].belief(), h3_ng_op_history[0].disbelief(), h3_ng_op_history[0].uncertainty()
fig.add_trace(
    go.Scatterternary(
        a=[u_h3, u0,] if show_all_opinion else [u_h3],
        b=[d_h3, d0,] if show_all_opinion else [d_h3],
        c=[b_h3, b0,] if show_all_opinion else [b_h3],
        mode='markers',
        marker=dict(size=14, color=['green', 'cyan', 'yellow'] if show_all_opinion else ['green']),
        hovertemplate=["H3<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "KL<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
                       "AD<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"]  if show_all_opinion else ["H3<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
        name="H3"
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




# b_h4, d_h4, u_h4 = h4_bias_op_history[0].belief(), h4_bias_op_history[0].disbelief(), h4_bias_op_history[0].uncertainty()
# b_h41, d_h41, u_h41 = h4_1_op_history[0].belief(), h4_1_op_history[0].disbelief(), h4_1_op_history[0].uncertainty()
# b_h42, d_h42, u_h42 = h4_2_op_history[0].belief(), h4_2_op_history[0].disbelief(), h4_2_op_history[0].uncertainty()
# fig.add_trace(
#     go.Scatterternary(
#         a=[u_h4, u_h41, u_h42] if show_all_opinion else [u_h4],
#         b=[d_h4, d_h41, d_h42] if show_all_opinion else [d_h4],
#         c=[b_h4, b_h41, b_h42] if show_all_opinion else [b_h4],
#         mode='markers',
#         marker=dict(size=14, color=['green', 'yellow', 'cyan'] if show_all_opinion else ['green']),
#         hovertemplate=["H4<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H4.1<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>",
#                        "H4.2<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"] if show_all_opinion else ["H4<br>b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"],
#         name="H4"
#     ),
#     row=3, col=4
# )


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
        name="H3 counts",
        marker=dict(color='cyan')

    ),
    row=4, col=4
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
fig.update_yaxes(range=[0, SHORT_WINDOW_SIZE], row=4, col=4)
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
    # b_q, d_q, u_q = q_op_obj_history[i].belief(), q_op_obj_history[i].disbelief(), q_op_obj_history[i].uncertainty()
    # b_q2, d_q2, u_q2 = q2_op_obj_history[i].belief(), q2_op_obj_history[i].disbelief(), q2_op_obj_history[i].uncertainty()
    # b_r, d_r, u_r = r_op_obj_history[i].belief(), r_op_obj_history[i].disbelief(), r_op_obj_history[i].uncertainty()
    b, d, u = opinions[i]
    # b_ad, d_ad, u_ad = opinions_ad[i]
    # y_i = dirichlet_pdfs[i]
    P = b_f + prior * u_f
    counts_i = counts_history[i]
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
    b_h3, d_h3, u_h3 = h3_ng_op_history[i].belief(), h3_ng_op_history[i].disbelief(), h3_ng_op_history[i].uncertainty()
    # b_h4, d_h4, u_h4 = h4_bias_op_history[i].belief(), h4_bias_op_history[i].disbelief(), h4_bias_op_history[i].uncertainty()
    # b_h41, d_h41, u_h41 = h4_1_op_history[i].belief(),  h4_1_op_history[i].disbelief(), h4_1_op_history[i].uncertainty()
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
        go.Scatterternary(a=[u_f], #, u_f2],
                          b=[d_f], #, d_f2],
                          c=[b_f], #, b_f2],
                          cliponaxis=False)
    )

    new_data.append(go.Scatterternary(a=[0], b=[1 - P], c=[P], cliponaxis=False))

    new_data.append(
        go.Scatterternary(a=[u_f, 0], b=[d_f, 1 - P], c=[b_f, P], mode='lines', line=dict(color='green', dash='dot'),
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
        go.Scatterternary(a=[u_h3, u] if show_all_opinion else [u_h3],
                          b=[d_h3, d] if show_all_opinion else [d_h3],
                          c=[b_h3, b] if show_all_opinion else [b_h3],
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
    # new_data.append(
    #     go.Scatterternary(a=[u_h4, u_h41, u_h42] if show_all_opinion else [u_h4],
    #                       b=[d_h4, d_h41, d_h42] if show_all_opinion else [d_h4],
    #                       c=[b_h4, b_h41, b_h42] if show_all_opinion else [b_h4],
    #                       cliponaxis=False)
    # )
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