#!/usr/bin/env python3
"""
Monte-Carlo Kalman-filter self-assessment experiment.

This script is a non-interactive, paper-evaluation-oriented version of
01_KalmanFilterWithSelfAssessmentV6.py. It keeps the relevant disturbances from V6
and removes interactive plotting/Plotly/GOSPA code.

Outputs:
  - compressed NumPy archive with all runs and aggregate statistics
  - CSV files with mean and quantile time series
  - optional static matplotlib plot

Default disturbances, matching V6 semantics:
  1) Measurement-noise covariance jumps via disturbance_measurement_noise:
       k=200: factor 4, k=300: factor 1/4, k=400: factor 1/4, k=500: factor 4
  2) Truncated-Gaussian measurement noise:
       k in [600, 700)
  3) Motion-model mismatch:
       GT uses KnownTurnRate in k in [800, 875), filter stays CV unless use_ct_model=True
  4) Process-noise disturbance in GT transition:
       k=1000: factor 16, k=1200: factor 1/16, on y component as in V6
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg
from scipy.stats import chi2, norm
from tqdm.auto import tqdm

import subjective_logic as sl

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.models.transition.nonlinear import ConstantTurn
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor, ExtendedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater, ExtendedKalmanUpdater

from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
from stonesoup.selfassessor.nis import NIS
from stonesoup.selfassessor._threshold import calc_threshold_n_diff

from aduulm_scripts.utils.add_disturbance import (
    disturbance_transition_model,
    disturbance_measurement_noise,
)
from griebels_methods.binomial_hypothesis import GriebelInnovationTest


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Monte Carlo
    n_mc: int = 100
    seed0: int = 0

    # Time / model
    num_steps: int = 1300
    dt_seconds: float = 0.1
    use_ct_model: bool = False
    activate_disturbances: bool = True

    # Nominal model parameters, as in V6
    q_x: float = 0.25
    q_y: float = 0.25
    q_phi: float = 0.01
    meas_var_x: float = 1.0
    meas_var_y: float = 1.0

    # Motion-model mismatch disturbance
    turn_start: int = 800
    turn_end: int = 875
    turn_rate_deg_s: float = -20.0

    # GT process-noise disturbance, as in V6
    disturbance_factor_process: float = 32
    process_disturb_start: int = 1000
    process_disturb_end: int = 1200
    disturb_x: bool = True
    disturb_y: bool = False

    # Measurement covariance disturbance, as in V6
    disturbance_factor_meas: float = 2.0
    meas_disturb_k1: int = 200
    meas_disturb_k2: int = 300
    meas_disturb_k3: int = 400
    meas_disturb_k4: int = 500
    # V6 uses [1,1]; both dimensions are affected
    disturb_noise_coeff_x: int = 1
    disturb_noise_coeff_y: int = 1

    # Truncated-Gaussian measurement-noise disturbance, as in V6
    correlated_start: int = -1
    correlated_end: int = -1
    diagonal_start: int = -1
    diagonal_end: int = -1
    truncated_start: int = 600
    truncated_end: int = 700
    truncation_sigma: float = 1.0

    # Optional controlled counterexample: directly force whitened innovations
    # Set to True only if you want the signed-half-Gaussian counterexample.
    use_forced_half_innovation: bool = False
    forced_half_start: int = 1000
    forced_half_end: int = 1100
    forced_half_sign_x: float = -1.0
    forced_half_sign_y: float = +1.0
    forced_half_spread_factor: float = 1.0

    # Self-assessment
    W: int = 7
    short_window_size: int = 35
    discount: float = 0.99
    alpha_threshold_dc: float = 0.01
    griebel_alpha: float = 0.05
    griebel_window_length: int = 35
    griebel_two_sided: bool = False
    griebel_type_compar: str = "dimensional"  # V6 default is dimensional
    griebel_priors: tuple = (0.95, 0.05)

    # Output
    output_dir: str = "mc_results"
    output_prefix: str = "mc_kalman_sa"
    save_plot: bool = True
    show_plot: bool = False
    show_quantile_band: bool = True
    save_uncertainty_plots: bool = True
    overlay_uncertainty_in_pok_plots: bool = False


# -----------------------------------------------------------------------------
# Noise models / utility functions
# -----------------------------------------------------------------------------

from scipy.stats import norm
import numpy as np

def sample_shift_copula_noise_from_cov(
    R,
    rng,
    shift=0.5,
    eps=1e-12,
):
    """
    Measurement noise with exactly Gaussian marginals but a non-Gaussian
    deterministic copula.

    U ~ Uniform(0,1)
    w_x = Phi^{-1}(U)
    w_y = Phi^{-1}((U + shift) mod 1)

    Marginals:
        w_x ~ N(0,1)
        w_y ~ N(0,1)

    Joint:
        strongly non-Gaussian.

    For shift=0.5, extremes in one component are paired with central
    values in the other component.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (2, 2):
        raise ValueError("This function assumes a 2D measurement covariance.")

    sigma = np.sqrt(np.diag(R))

    u = rng.uniform(0.0, 1.0)
    v = (u + shift) % 1.0

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    w_x = norm.ppf(u)
    w_y = norm.ppf(v)

    noise = np.array([
        sigma[0] * w_x,
        sigma[1] * w_y,
    ])

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

def sample_truncated_gaussian_noise_from_cov(
    R: np.ndarray,
    rng: np.random.Generator,
    truncation_sigma: float = 1.0,
    max_tries: int = 10_000,
) -> np.ndarray:
    """
    Samples from N(0, R) conditioned component-wise on
    |v_i| <= truncation_sigma * sigma_i.
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


def sample_signed_half_whitened_innovation(
    dim: int,
    rng: np.random.Generator,
    signs: Tuple[float, ...] = (-1.0, +1.0),
    spread_factor: float = 1.0,
) -> np.ndarray:
    """
    Samples a signed half-normal vector in whitened innovation space.

    For signs=(-1,+1):
        w_x = -|N(0,1)|
        w_y = +|N(0,1)|
    """
    signs_arr = np.asarray(signs, dtype=float)
    if len(signs_arr) != dim:
        raise ValueError(f"len(signs)={len(signs_arr)} must match dim={dim}.")

    z = rng.normal(loc=0.0, scale=1.0, size=dim)
    w = signs_arr * np.abs(z) * spread_factor
    return w.reshape(-1, 1)


def robust_cholesky(M: np.ndarray, max_jitter: float = 1e-3) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = (M + M.T) / 2.0

    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        pass

    eps = 1e-9
    while eps <= max_jitter:
        try:
            return np.linalg.cholesky(M + eps * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            eps *= 10

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def scalar_u_to_opinion(u: float, W: int, scale: float = 1.0):
    """Map scalar u in [0,1] to a W-dimensional one-hot evidence opinion."""
    if u == -1:
        return eval(f"sl.Opinion{W}d")(*([0] * W))

    u = float(np.clip(u, 0.0, 1.0))
    evidence = np.zeros(W, dtype=float)
    idx = min(int(np.floor(u * W)), W - 1)
    evidence[idx] = scale

    dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(evidence)
    return dist.as_opinion()


def multinomial_opinion_to_binomial_ok_opinion(op, W: int, prior_ok: float = 0.5, eps: float = 1e-12):
    """
    Map W-dimensional multinomial opinion to binomial OK opinion using normalized TV.
    H = "component/model is consistent".
    """
    u = op.uncertainty()
    c = 1.0 - u

    if c <= eps:
        op_ok = sl.Opinion2d(0.0, 0.0)
        op_ok.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
        return op_ok

    a = np.ones(W) / W
    b = np.asarray(op.belief_masses, dtype=float)
    b_tilde = b / c

    tv = 0.5 * np.sum(np.abs(b_tilde - a))
    tv_max = 1.0 - 1.0 / W

    d_alarm = c * tv / tv_max
    d_alarm = float(np.clip(d_alarm, 0.0, c))
    b_ok = c - d_alarm

    op_ok = sl.Opinion2d(b_ok, d_alarm)
    op_ok.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
    return op_ok


def calculate_lt_evidence(W: int, alpha: float = 0.99) -> int:
    return int((-(W - 1) + np.sqrt((W - 1) ** 2 + ((4 * W) / (1 - alpha)))) / 2)


def load_or_calibrate_threshold(config: ExperimentConfig) -> float:
    """
    Load V6-style threshold if available. Otherwise fall back to calc_threshold_n_diff.
    """
    key = f"{config.W}, {calculate_lt_evidence(config.W, config.discount)}, {config.alpha_threshold_dc}"
    path = Path("opinion_threshold_smoothed.json")
    if path.exists():
        with path.open("r") as f:
            thresholds = json.load(f)
        if key in thresholds:
            return float(thresholds[key])

    return float(calc_threshold_n_diff(config.W, calculate_lt_evidence(config.W, config.discount), config.alpha_threshold_dc))


# -----------------------------------------------------------------------------
# Simulation primitives
# -----------------------------------------------------------------------------

def make_models(config: ExperimentConfig):
    dt = timedelta(seconds=config.dt_seconds)

    if config.use_ct_model:
        transition_model = ConstantTurn([config.q_x, config.q_y], config.q_phi)
    else:
        transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(config.q_x),
            ConstantVelocity(config.q_y),
        ])

    gt_cv_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(config.q_x),
        ConstantVelocity(config.q_y),
    ])
    gt_right_turn_model = KnownTurnRate(
        [config.q_x, config.q_y],
        np.radians(config.turn_rate_deg_s),
    )

    gt_measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[config.meas_var_x, 0.0], [0.0, config.meas_var_y]]),
    )
    measurement_model = LinearGaussian(
        ndim_state=5 if config.use_ct_model else 4,
        mapping=(0, 2),
        noise_covar=np.array([[config.meas_var_x, 0.0], [0.0, config.meas_var_y]]),
    )

    return dt, transition_model, gt_cv_model, gt_right_turn_model, gt_measurement_model, measurement_model


def generate_truth(config: ExperimentConfig, rng: np.random.Generator):
    dt, _, gt_cv_model, gt_right_turn_model, _, _ = make_models(config)

    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timesteps = [start_time]

    truth = GroundTruthPath([
        GroundTruthState(
            StateVector([[0.0], [5.0], [0.0], [5.0]]),
            timestamp=start_time,
        )
    ])

    model_indices = []
    process_noise_coeff_memory = [[], []]

    disturbance_factor_process = config.disturbance_factor_process if config.activate_disturbances else 1.0
    gt_transition_configs = {
        "noise_diff_coeff": [[config.q_x, config.q_y]],
        "disturb_noise_coeff": [config.disturb_x, config.disturb_y],
        "disturbance_mode": ["jump"],
        "parameters": [[
            [config.process_disturb_start, disturbance_factor_process],
            [config.process_disturb_end, 1.0 / disturbance_factor_process],
        ]],
    }

    for k in range(1, config.num_steps + 1):
        timesteps.append(start_time + k * dt)

        gt_cv_model = disturbance_transition_model(gt_cv_model, gt_transition_configs, k)

        if config.turn_start <= k < config.turn_end:
            gt_model = gt_right_turn_model
            model_indices.append(1)
        else:
            gt_model = gt_cv_model
            model_indices.append(0)

        try:
            if len(gt_model.model_list) == 0:
                process_noise_coeff_memory[0].append(config.q_x)
                process_noise_coeff_memory[1].append(config.q_y)
            else:
                for idx, model in enumerate(gt_model.model_list):
                    process_noise_coeff_memory[idx].append(getattr(model, "noise_diff_coeff", np.nan))
        except Exception:
            process_noise_coeff_memory[0].append(config.q_x)
            process_noise_coeff_memory[1].append(config.q_y)

        truth.append(
            GroundTruthState(
                gt_model.function(truth[k - 1], noise=True, time_interval=dt),
                timestamp=timesteps[k],
            )
        )

    return truth, timesteps, np.asarray(model_indices), process_noise_coeff_memory


def generate_measurements(config: ExperimentConfig, truth: GroundTruthPath, rng: np.random.Generator):
    _, _, _, _, gt_measurement_model, measurement_model = make_models(config)

    disturbance_factor_meas = config.disturbance_factor_meas if config.activate_disturbances else 1.0
    gt_measurement_configs = {
        "disturbance_mode": ["jump"],
        "parameters": [[
            [config.meas_disturb_k1, disturbance_factor_meas, [1, 0]],
            [config.meas_disturb_k2, 1.0 / disturbance_factor_meas, [1, 0]],
            [config.meas_disturb_k3, 1.0 / disturbance_factor_meas, [1, 1]],
            [config.meas_disturb_k4, disturbance_factor_meas, [1, 1]],
        ]],
        # "disturb_noise_coeff": [config.disturb_noise_coeff_x, config.disturb_noise_coeff_y],
    }

    measurements = []
    meas_std_dev_memory = []
    meas_correlated_gaussian_memory = []
    meas_truncated_gaussian_memory = []

    for k, state in enumerate(truth):
        gt_measurement_model = disturbance_measurement_noise(gt_measurement_model, gt_measurement_configs, k)

        trunc_sigma = 0.0
        correlation_activated = 0
        if config.truncated_start <= k < config.truncated_end and config.activate_disturbances:
            measurement = gt_measurement_model.function(state, noise=False)
            R_true = np.asarray(gt_measurement_model.noise_covar, dtype=float)
            trunc_sigma = config.truncation_sigma
            measurement += sample_truncated_gaussian_noise_from_cov(
                R_true,
                rng,
                truncation_sigma=trunc_sigma,
            )
        elif config.correlated_start <= k < config.correlated_end and config.activate_disturbances:
            measurement = gt_measurement_model.function(state, noise=False)
            R_true = np.asarray(gt_measurement_model.noise_covar, dtype=float)

            measurement += sample_correlated_gaussian_noise_from_cov(
                R_true,
                rng,
                rho=0.9999
            )
            correlation_activated = 1
        elif config.diagonal_start <= k < config.diagonal_end and config.activate_disturbances:
            measurement = gt_measurement_model.function(state, noise=False)
            R_true = np.asarray(gt_measurement_model.noise_covar, dtype=float)

            measurement += sample_shift_copula_noise_from_cov(
                R_true,
                rng,
            )
        else:
            measurement = gt_measurement_model.function(state, noise=True)

        measurements.append(
            Detection(
                measurement,
                timestamp=state.timestamp,
                measurement_model=measurement_model,
            )
        )
        meas_std_dev_memory.append(np.sqrt(gt_measurement_model.noise_covar))
        meas_correlated_gaussian_memory.append(correlation_activated)
        meas_truncated_gaussian_memory.append(trunc_sigma)

    return measurements, np.asarray(meas_truncated_gaussian_memory), meas_std_dev_memory


# -----------------------------------------------------------------------------
# Assessment execution
# -----------------------------------------------------------------------------

def run_single_simulation(seed: int, config: ExperimentConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    dt, transition_model, _, _, _, measurement_model = make_models(config)
    truth, timesteps, model_indices, process_noise_coeff_memory = generate_truth(config, rng)
    measurements, trunc_memory, meas_std_dev_memory = generate_measurements(config, truth, rng)

    if config.use_ct_model:
        if True:
            predictor = UnscentedKalmanPredictor(transition_model)
            updater = UnscentedKalmanUpdater(measurement_model)
        else:
            predictor = ExtendedKalmanPredictor(transition_model)
            updater = ExtendedKalmanUpdater(measurement_model)
    else:
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)

    prior = GaussianState(
        StateVector([[0.0], [5.0], [0.0], [5.0], [0.0]]) if config.use_ct_model else StateVector([[0.0], [5.0], [0.0], [5.0]]),
        CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0, 0.1])) if config.use_ct_model else CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0])),
        timestamp=timesteps[0],
    )

    W = config.W
    m = measurement_model.ndim_meas
    max_buffer = config.short_window_size
    bin_edges = np.linspace(0.0, 1.0, W + 1)

    # Griebel multinomial SA, V6 default: dimensional unless config changed
    selfassessor = KalmanSelfAssessor(
        num_X=W,
        n_st=config.short_window_size,
        n_c=1,
        dim_meas=m,
        alpha_threshold_dc=config.alpha_threshold_dc,
        trust_discount=config.discount,
        type_compar=config.griebel_type_compar,
    )
    selfassessor_measures_history = []

    # NIS baseline
    nis = NIS(window_length=config.short_window_size, alpha=config.alpha_threshold_dc, dim=m)
    nis_score_history = []

    # Griebel binomial innovation test
    griebel_inno = GriebelInnovationTest(
        dim_meas=m,
        alpha=config.griebel_alpha,
        window_length=config.griebel_window_length,
        two_sided=config.griebel_two_sided,
        mapping=measurement_model.mapping,
    )
    griebel_inno_window = []
    griebel_p_ok_history = []
    griebel_uncertainty_history = []

    # Proposed per-time opinions
    ops_per_timestep = []
    ops_component_x = []
    ops_component_y = []
    u_buffer = []

    track = Track()

    for i, measurement in enumerate(measurements):
        prediction: GaussianStatePrediction = predictor.predict(prior, timestamp=measurement.timestamp)

        # Optional controlled disturbance directly in innovation space.
        if config.use_forced_half_innovation and config.forced_half_start <= i < config.forced_half_end:
            H = np.asarray(measurement_model.matrix(), dtype=float)
            R_filter = np.asarray(measurement_model.covar(), dtype=float)
            P_pred = np.asarray(prediction.covar, dtype=float)
            z_pred_forced = H @ prediction.state_vector
            S_pred = H @ P_pred @ H.T + R_filter

            S_sqrt = scipy.linalg.sqrtm(S_pred)
            S_sqrt = np.real_if_close(S_sqrt).astype(float)
            w = sample_signed_half_whitened_innovation(
                dim=m,
                rng=rng,
                signs=(config.forced_half_sign_x, config.forced_half_sign_y),
                spread_factor=config.forced_half_spread_factor,
            )
            measurement = Detection(
                StateVector(z_pred_forced + S_sqrt @ w),
                timestamp=measurement.timestamp,
                measurement_model=measurement_model,
            )

        hypothesis = SingleHypothesis(prediction, measurement)
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

        z_p = hypothesis.measurement_prediction.mean
        S_p = hypothesis.measurement_prediction.covar
        z = measurement.state_vector

        selfassessor.assess(z_p, S_p, z)
        selfassessor_measures_history.append(selfassessor.get_sas_measures())
        nis_result = nis.assess(z_p, S_p, z)
        # Keep first element if NIS returns vector/list, otherwise scalar.
        try:
            nis_score_history.append(float(np.ravel(nis_result)[0]))
        except Exception:
            nis_score_history.append(np.nan)

        # Innovation quantities
        delta = (z - z_p).reshape(-1, 1)
        S = np.asarray(S_p, dtype=float)

        # Griebel binomial innovation opinion
        griebel_result = griebel_inno.assess(z, z_p, S)
        griebel_accept = bool(griebel_result["accept_h0"])
        griebel_evidence = np.array([int(griebel_accept), 1 - int(griebel_accept)])
        griebel_dist = eval("sl.DirichletDistribution2d").from_evidences(griebel_evidence)
        griebel_op = griebel_dist.as_opinion()
        griebel_op.prior_belief_masses = list(config.griebel_priors)
        griebel_inno_window.append(griebel_op)
        if len(griebel_inno_window) > config.griebel_window_length:
            griebel_inno_window.pop(0)
        griebel_fused = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, griebel_inno_window)
        griebel_p_ok_history.append(griebel_fused.getProjection()[0])
        griebel_uncertainty_history.append(griebel_fused.uncertainty())

        # Component-wise PIT opinions
        S_sqrt = robust_cholesky(S)
        nu_white = np.linalg.solve(S_sqrt, delta).flatten()
        u_comp_x = norm.cdf(float(nu_white[0]))
        u_comp_y = norm.cdf(float(nu_white[1]))
        ops_component_x.append(scalar_u_to_opinion(u_comp_x, W))
        ops_component_y.append(scalar_u_to_opinion(u_comp_y, W))

        # Radial PIT opinion
        d2 = float((delta.T @ np.linalg.inv(S) @ delta).item())
        u_rad = chi2.cdf(d2, df=m)
        u_buffer.append(u_rad)
        if len(u_buffer) > max_buffer:
            u_buffer.pop(0)

        evidence = np.zeros(W)
        evidence_idx = np.searchsorted(bin_edges, u_rad, side="right") - 1
        evidence_idx = int(np.clip(evidence_idx, 0, W - 1))
        evidence[evidence_idx] += 1
        one_time_dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(evidence)
        ops_per_timestep.append(one_time_dist.as_opinion())

    # Build proposed radial LTST opinion history
    threshold = load_or_calibrate_threshold(config)
    ltst = eval(f"sl.LongShortTermMemory{W}d")(
        config.short_window_size,
        threshold,
        config.discount,
        sl.FusionType.CUMULATIVE,
        False,
        False,
    )
    ltst_component_x = eval(f"sl.LongShortTermMemory{W}d")(
        config.short_window_size,
        threshold,
        config.discount,
        sl.FusionType.CUMULATIVE,
        False,
        False,
    )
    ltst_component_y = eval(f"sl.LongShortTermMemory{W}d")(
        config.short_window_size,
        threshold,
        config.discount,
        sl.FusionType.CUMULATIVE,
        False,
        False,
    )

    buffered_ops = []
    for op_obs in ops_per_timestep:
        ltst.add(op_obs)
        buffered_ops.append(ltst.get_opinion())

    component_x_buffered = []
    component_y_buffered = []
    for opx, opy in zip(ops_component_x, ops_component_y):
        ltst_component_x.add(opx)
        ltst_component_y.add(opy)
        component_x_buffered.append(ltst_component_x.get_opinion())
        component_y_buffered.append(ltst_component_y.get_opinion())

    # Radial binomial opinions
    global_op_history = [multinomial_opinion_to_binomial_ok_opinion(op, W, prior_ok=0.5) for op in buffered_ops]

    # Component and overall binomial opinions
    prior_comp = float(np.sqrt(0.5))
    p_ok_x = []
    p_ok_y = []
    p_ok_comp = []
    p_ok_overall = []
    p_ok_radial = []
    u_x = []
    u_y = []
    u_comp = []
    u_overall = []
    u_radial = []

    for opx, opy, opr in zip(component_x_buffered, component_y_buffered, global_op_history):
        opx_bin = multinomial_opinion_to_binomial_ok_opinion(opx, W, prior_ok=prior_comp)
        opy_bin = multinomial_opinion_to_binomial_ok_opinion(opy, W, prior_ok=prior_comp)
        component_op = opx_bin.multiply(opy_bin)
        overall_op = sl.Fusion.fuse_opinions(sl.FusionType.WEIGHTED, [component_op, opr])

        p_ok_x.append(opx_bin.getProjection()[0])
        p_ok_y.append(opy_bin.getProjection()[0])
        p_ok_comp.append(component_op.getProjection()[0])
        p_ok_overall.append(overall_op.getProjection()[0])
        p_ok_radial.append(opr.getProjection()[0])
        u_x.append(opx_bin.uncertainty())
        u_y.append(opy_bin.uncertainty())
        u_comp.append(component_op.uncertainty())
        u_overall.append(overall_op.uncertainty())
        u_radial.append(opr.uncertainty())

    selfassessor_measures_history = np.asarray(selfassessor_measures_history, dtype=float)

    return {
        "p_ok_griebel_innovation": np.asarray(griebel_p_ok_history, dtype=float),
        "p_ok_radial": np.asarray(p_ok_radial, dtype=float),
        "p_ok_x": np.asarray(p_ok_x, dtype=float),
        "p_ok_y": np.asarray(p_ok_y, dtype=float),
        "p_ok_comp": np.asarray(p_ok_comp, dtype=float),
        "p_ok_overall": np.asarray(p_ok_overall, dtype=float),
        "u_griebel_innovation": np.asarray(griebel_uncertainty_history, dtype=float),
        "u_radial": np.asarray(u_radial, dtype=float),
        "u_x": np.asarray(u_x, dtype=float),
        "u_y": np.asarray(u_y, dtype=float),
        "u_comp": np.asarray(u_comp, dtype=float),
        "u_overall": np.asarray(u_overall, dtype=float),
        "griebel_multinomial_dc": selfassessor_measures_history[:, 0],
        "griebel_multinomial_uncertainty": selfassessor_measures_history[:, 1],
        "griebel_multinomial_threshold": selfassessor_measures_history[:, 2],
        "nis_score": np.asarray(nis_score_history, dtype=float),
        "dist_truncated_gaussian": np.asarray(trunc_memory[: len(p_ok_overall)], dtype=float),
        "dist_turn": np.pad(model_indices, (0, max(0, len(p_ok_overall) - len(model_indices))), constant_values=0)[: len(p_ok_overall)].astype(float),
    }


# -----------------------------------------------------------------------------
# Monte Carlo aggregation / export / plotting
# -----------------------------------------------------------------------------

def aggregate_runs(runs: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    keys = runs[0].keys()
    aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for key in keys:
        min_len = min(len(run[key]) for run in runs)
        arr = np.stack([run[key][:min_len] for run in runs], axis=0)
        aggregated[key] = {
            "all": arr,
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
            "q05": np.quantile(arr, 0.05, axis=0),
            "q50": np.quantile(arr, 0.50, axis=0),
            "q95": np.quantile(arr, 0.95, axis=0),
        }

    return aggregated


def run_monte_carlo(config: ExperimentConfig) -> Dict[str, Dict[str, np.ndarray]]:
    runs = []
    for mc in tqdm(range(config.n_mc), desc="Monte Carlo runs", unit="run"):
        seed = config.seed0 + mc
        runs.append(run_single_simulation(seed=seed, config=config))
    return aggregate_runs(runs)


def save_mc_results(mc_results: Dict[str, Dict[str, np.ndarray]], config: ExperimentConfig) -> Dict[str, Path]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{config.output_prefix}.npz"
    config_path = out_dir / f"{config.output_prefix}_config.json"

    save_dict = {}
    for key, stats in mc_results.items():
        for stat_name, values in stats.items():
            save_dict[f"{key}_{stat_name}"] = values
    np.savez_compressed(npz_path, **save_dict)

    with config_path.open("w") as f:
        json.dump(asdict(config), f, indent=2)

    # CSV with mean and quantile series for the exported core time series
    csv_path = out_dir / f"{config.output_prefix}_time_series.csv"
    core_keys = [
        "p_ok_griebel_innovation",
        "p_ok_radial",
        "p_ok_x",
        "p_ok_y",
        "p_ok_comp",
        "p_ok_overall",
        "u_griebel_innovation",
        "u_radial",
        "u_x",
        "u_y",
        "u_comp",
        "u_overall",
        "griebel_multinomial_dc",
        "griebel_multinomial_uncertainty",
        "griebel_multinomial_threshold",
    ]
    T = len(mc_results[core_keys[0]]["mean"])
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["k"]
        for key in core_keys:
            if key in mc_results:
                header.extend([f"{key}_mean", f"{key}_q05", f"{key}_q50", f"{key}_q95"])
        writer.writerow(header)

        for k in range(T):
            row = [k]
            for key in core_keys:
                if key in mc_results:
                    row.extend([
                        mc_results[key]["mean"][k],
                        mc_results[key]["q05"][k],
                        mc_results[key]["q50"][k],
                        mc_results[key]["q95"][k],
                    ])
            writer.writerow(row)

    return {"npz": npz_path, "config": config_path, "csv": csv_path}


def plot_mc_results(mc_results: Dict[str, Dict[str, np.ndarray]], config: ExperimentConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_series(ax, key: str, label: str, *, linestyle: str = "-", alpha: float = 1.0):
        x = np.arange(len(mc_results[key]["mean"]))
        mean = mc_results[key]["mean"]
        line, = ax.plot(x, mean, label=label, linestyle=linestyle, alpha=alpha)
        if config.show_quantile_band:
            q05 = mc_results[key]["q05"]
            q95 = mc_results[key]["q95"]
            ax.fill_between(x, q05, q95, alpha=0.15, color=line.get_color())

    def add_disturbance_spans(ax):
        disturbance_spans = [
            (config.meas_disturb_k1, config.meas_disturb_k2, "#fee5e5", f"measurement noise x{config.disturbance_factor_meas}"),
            (config.meas_disturb_k3, config.meas_disturb_k4, "#fcb7b7", f"measurement noise x1/{config.disturbance_factor_meas}"),
            # (config.correlated_start, config.correlated_end, "#fcb7b7", "correlated Gaussian"),
            # (config.diagonal_start, config.diagonal_end, "#fcb7b7", "diagonalized Gaussian"),
            (config.truncated_start, config.truncated_end, "#fc8d8d", "truncated Gaussian"),
            (config.turn_start, config.turn_end, "#ef3b2c", "turn / model mismatch"),
            (config.process_disturb_start, config.process_disturb_end, "#b30000", "process noise disturbance"),
        ]
        if config.use_forced_half_innovation:
            disturbance_spans.append(
                (config.forced_half_start, config.forced_half_end, "#67000d", "forced half innovation")
            )

        for start, end, color, label in disturbance_spans:
            if start < end:
                ax.axvspan(start, end, alpha=0.20, color=color, label=label)

    def finalize_axes(ax, ylabel: str, title: str):
        ax.set_xlabel("time step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True)
        ax.legend(loc="best")
        ax.set_title(title)

    def save_figure(fig, stem: str) -> Tuple[Path, Path]:
        png = out_dir / f"{config.output_prefix}_{stem}.png"
        pdf = out_dir / f"{config.output_prefix}_{stem}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        if config.show_plot:
            plt.show()
        plt.close(fig)
        return png, pdf

    main_pairs = [
        ("p_ok_griebel_innovation", "Griebel innovation $P_{OK}$"),
        ("p_ok_radial", "Proposed radial $P_{OK}$"),
        ("p_ok_comp", "Proposed component $P_{OK}$"),
        ("p_ok_overall", "Proposed overall $P_{OK}$"),
    ]
    main_uncertainty_pairs = [
        ("u_griebel_innovation", "Griebel innovation uncertainty"),
        ("griebel_multinomial_uncertainty", "Griebel multinomial uncertainty"),
        ("u_radial", "Proposed radial uncertainty"),
        ("u_comp", "Proposed component uncertainty"),
        ("u_overall", "Proposed overall uncertainty"),
    ]
    component_pairs = [
        ("p_ok_x", "$P_{OK,x}$"),
        ("p_ok_y", "$P_{OK,y}$"),
    ]
    component_uncertainty_pairs = [
        ("u_x", "$u_x$"),
        ("u_y", "$u_y$"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in main_pairs:
        plot_series(ax, key, label)
    if config.overlay_uncertainty_in_pok_plots:
        for key, label in main_uncertainty_pairs:
            plot_series(ax, key, f"{label} (overlay)", linestyle="--", alpha=0.85)
    add_disturbance_spans(ax)
    finalize_axes(ax, r"projected probability $P_{OK}$", f"Monte Carlo mean projected probabilities, N={config.n_mc}")
    png_path, pdf_path = save_figure(fig, "projected_probabilities")

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in component_pairs:
        plot_series(ax, key, label)
    if config.overlay_uncertainty_in_pok_plots:
        for key, label in component_uncertainty_pairs:
            plot_series(ax, key, f"{label} uncertainty (overlay)", linestyle="--", alpha=0.85)
    add_disturbance_spans(ax)
    finalize_axes(ax, r"projected probability $P_{OK}$", f"Component-wise projected probabilities, N={config.n_mc}")
    comp_png, comp_pdf = save_figure(fig, "component_projected_probabilities")

    result_paths = {
        "main_png": png_path,
        "main_pdf": pdf_path,
        "component_png": comp_png,
        "component_pdf": comp_pdf,
    }

    if config.save_uncertainty_plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, label in main_uncertainty_pairs:
            plot_series(ax, key, label)
        add_disturbance_spans(ax)
        finalize_axes(ax, "uncertainty", f"Monte Carlo mean opinion uncertainties, N={config.n_mc}")
        unc_png, unc_pdf = save_figure(fig, "uncertainties")
        result_paths["uncertainty_png"] = unc_png
        result_paths["uncertainty_pdf"] = unc_pdf

        fig, ax = plt.subplots(figsize=(10, 4))
        for key, label in component_uncertainty_pairs:
            plot_series(ax, key, label)
        add_disturbance_spans(ax)
        finalize_axes(ax, "uncertainty", f"Component-wise opinion uncertainties, N={config.n_mc}")
        comp_unc_png, comp_unc_pdf = save_figure(fig, "component_uncertainties")
        result_paths["component_uncertainty_png"] = comp_unc_png
        result_paths["component_uncertainty_pdf"] = comp_unc_pdf

    return result_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte-Carlo KF self-assessment experiment.")
    parser.add_argument("--n-mc", type=int, default=100, help="Number of Monte Carlo runs.")
    parser.add_argument("--seed0", type=int, default=0, help="First random seed.")
    parser.add_argument("--output-dir", type=str, default="mc_results", help="Output directory.")
    parser.add_argument("--output-prefix", type=str, default="mc_kalman_sa", help="Output filename prefix.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate static plots.")
    parser.add_argument("--no-quantile-band", action="store_true", help="Disable q05-q95 fill_between bands in plots.")
    parser.add_argument("--no-uncertainty-plots", action="store_true", help="Do not generate dedicated uncertainty plots.")
    parser.add_argument("--overlay-uncertainty", action="store_true", help="Overlay uncertainty curves in the projected-probability plots.")
    parser.add_argument("--forced-half-innovation", action="store_true", help="Enable signed-half-Gaussian innovation counterexample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n_mc=args.n_mc,
        seed0=args.seed0,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        save_plot=not args.no_plot,
        show_quantile_band=not args.no_quantile_band,
        save_uncertainty_plots=not args.no_uncertainty_plots,
        overlay_uncertainty_in_pok_plots=args.overlay_uncertainty,
        use_forced_half_innovation=args.forced_half_innovation,
    )

    mc_results = run_monte_carlo(config)
    paths = save_mc_results(mc_results, config)
    print("Saved results:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    if config.save_plot:
        plot_paths = plot_mc_results(mc_results, config)
        print("Saved plots:")
        for key, path in plot_paths.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
