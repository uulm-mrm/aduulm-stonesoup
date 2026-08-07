#!/usr/bin/env python3
"""
02 - Multi-sensor Kalman filter with standardized PIT/SL self-assessment.

Direct extension of 01_KalmanFilterWithSelfAssessmentV7(4).py:
- same Stone Soup tracking setup and disturbance timeline,
- same external ``subjective_logic`` library,
- same radial/component PIT channels,
- same LongShortTermMemory7d buffers,
- same conflict-based multinomial-to-binomial mapping.

New:
- two sensor streams,
- event-based processing,
- sensor 1 at 10 Hz and sensor 2 at 12.5 Hz,
- rate-normalised LTST parameters,
- time-reliability/trust discounting before multi-sensor fusion,
- switchable fully synchronous special case.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, chi2, norm

import subjective_logic as sl

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.selfassessor._threshold import calc_threshold_n_diff
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

from aduulm_scripts.utils.add_disturbance import (
    disturbance_measurement_noise,
    disturbance_transition_model,
)


# =============================================================================
# User switches
# =============================================================================

# False: native asynchronous/multi-rate case (10 Hz and 12.5 Hz).
# True: fully synchronous limiting case; sensor 2 is sampled on sensor 1's
#       10 Hz timestamps. In the pure limiting-case check also set dropout=False.
SYNCHRONOUS_SENSOR_SPECIAL_CASE = False

SENSOR_1_RATE_HZ = 10.0
SENSOR_2_RATE_HZ = 12.5
SYNCHRONOUS_REFERENCE_RATE_HZ = 10.0

ENABLE_SENSOR_2_DROPOUT = True
SHOW_DYNAMIC_ANIMATION = True
SHOW_MATPLOTLIB_PLOTS = True

ACTIVATE_DISTURBANCES = True
DISTURB_SENSOR_1 = True
DISTURB_SENSOR_2 = False

SCENARIO_DURATION_S = 140.0
TRUTH_RATE_HZ = 50.0  # exactly represents 10 Hz and 12.5 Hz
TRUTH_DT_S = 1.0 / TRUTH_RATE_HZ

Q_X = 0.25
Q_Y = 0.25
MEASUREMENT_VARIANCE_SENSOR_1 = 1.0
MEASUREMENT_VARIANCE_SENSOR_2 = 1.0

NUM_PIT_BINS = 7
REFERENCE_SHORT_TERM_SAMPLES = 35
REFERENCE_RATE_HZ = 10.0
REFERENCE_SHORT_TERM_HORIZON_S = REFERENCE_SHORT_TERM_SAMPLES / REFERENCE_RATE_HZ
REFERENCE_LONG_TERM_DISCOUNT = 0.999
ALPHA_THRESHOLD_DC = 0.01

LTST_FUSION_TYPE = sl.FusionType.CUMULATIVE
HANDLE_SHORT_TERM_CONFLICT = True
AVERAGE_DC_CONFLICT_HANDLING = True

# rho_s(t)=exp(-age/(K*T_s)); local LTST opinions remain unchanged.
FRESHNESS_TIME_CONSTANT_IN_PERIODS = 2.0

# Credibility regions are plotted only, not classified/evaluated.
CREDIBLE_REGION_ETA = 0.90
CREDIBLE_REGION_TAU_OK = 0.50

ANIMATION_TAIL_LENGTH = .20
SCRIPT_DIR = Path(__file__).resolve().parent

RANDOM_SEED_TRUTH = 0
RANDOM_SEED_SENSOR_1 = 1
RANDOM_SEED_SENSOR_2 = 2


# =============================================================================
# Original first-paper scenario, expressed in seconds
# =============================================================================

OUTLIER_INTERVAL_S = (10.0, 20.0)
INCREASED_MEAS_X_INTERVAL_S = (30.0, 40.0)
DECREASED_MEAS_XY_INTERVAL_S = (50.0, 60.0)
SENSOR_2_DROPOUT_INTERVAL_S = (60.0, 70.0)
TRUNCATED_GAUSSIAN_INTERVAL_S = (70.0, 80.0)
TURN_INTERVAL_S = (90.0, 100.0)
INCREASED_PROCESS_X_INTERVAL_S = (110.0, 130.0)

DISTURBANCE_INTERVALS = [
    (*OUTLIER_INTERVAL_S, "S1 outliers", "tab:red"),
    (*INCREASED_MEAS_X_INTERVAL_S, "S1 increased x-noise", "tab:orange"),
    (*DECREASED_MEAS_XY_INTERVAL_S, "S1 decreased x/y-noise", "tab:blue"),
    (*SENSOR_2_DROPOUT_INTERVAL_S, "S2 unavailable", "tab:gray"),
    (*TRUNCATED_GAUSSIAN_INTERVAL_S, "S1 truncated Gaussian", "tab:purple"),
    (*TURN_INTERVAL_S, "common motion-model mismatch", "tab:green"),
    (*INCREASED_PROCESS_X_INTERVAL_S, "common increased process noise", "tab:brown"),
]


def seconds_to_truth_step(seconds: float) -> int:
    return int(round(seconds * TRUTH_RATE_HZ))


def seconds_to_reference_step(seconds: float) -> int:
    return int(round(seconds * REFERENCE_RATE_HZ))


def robust_cholesky(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        pass

    jitter = 1e-9
    while jitter <= 1e-3:
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(matrix.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return np.linalg.cholesky(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)


def sample_truncated_gaussian_noise_from_cov(
    covariance: np.ndarray,
    rng: np.random.Generator,
    truncation_sigma: float = 1.0,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    sigma = np.sqrt(np.diag(covariance))
    noise = np.zeros(covariance.shape[0])
    for dimension in range(covariance.shape[0]):
        while True:
            candidate = rng.normal(0.0, sigma[dimension])
            if abs(candidate) <= truncation_sigma * sigma[dimension]:
                noise[dimension] = candidate
                break
    return noise.reshape(-1, 1)


# =============================================================================
# Standardized PIT -> LTST -> binomial mapping
# =============================================================================


def scalar_pit_to_opinion(pit_value: float, num_bins: int):
    pit_value = float(np.clip(pit_value, 0.0, 1.0))
    evidence = np.zeros(num_bins, dtype=float)
    index = min(int(np.floor(pit_value * num_bins)), num_bins - 1)
    evidence[index] = 1.0
    distribution = eval(f"sl.DirichletDistribution{num_bins}d").from_evidences(evidence)
    return distribution.as_opinion()


def multinomial_to_binomial_consistency_opinion(
    opinion,
    num_bins: int,
    prior_ok: float = 0.5,
    eps: float = 1e-12,
):
    uncertainty = float(opinion.uncertainty())
    committed_mass = 1.0 - uncertainty

    if committed_mass <= eps:
        result = sl.Opinion2d(0.0, 0.0)
        result.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
        return result

    uniform_reference = np.full(num_bins, 1.0 / num_bins)
    committed_distribution = np.asarray(opinion.belief_masses) / committed_mass
    total_variation = 0.5 * np.sum(np.abs(committed_distribution - uniform_reference))
    maximum_total_variation = 1.0 - 1.0 / num_bins

    disbelief = committed_mass * total_variation / maximum_total_variation
    disbelief = float(np.clip(disbelief, 0.0, committed_mass))
    belief = committed_mass - disbelief

    result = sl.Opinion2d(belief, disbelief)
    result.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
    return result


def calculate_long_term_evidence(num_bins: int, discount: float) -> int:
    return int(
        (-(num_bins - 1) + np.sqrt(
            (num_bins - 1) ** 2 + 4.0 * num_bins / (1.0 - discount + 1e-12)
        )) / 2.0
    )


@lru_cache(maxsize=1)
def load_opinion_thresholds() -> dict[str, float]:
    for path in (
        SCRIPT_DIR / "opinion_threshold_smoothed.json",
        Path.cwd() / "opinion_threshold_smoothed.json",
    ):
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    print(
        "WARNING: opinion_threshold_smoothed.json not found; using "
        "calc_threshold_n_diff fallback."
    )
    return {}


def rate_normalised_ltst_parameters(rate_hz: float) -> tuple[int, float, float]:
    n_st = max(2, int(round(REFERENCE_SHORT_TERM_HORIZON_S * rate_hz)))
    discount = REFERENCE_LONG_TERM_DISCOUNT ** (REFERENCE_RATE_HZ / rate_hz)
    equivalent_evidence = calculate_long_term_evidence(NUM_PIT_BINS, discount)
    key = f"{NUM_PIT_BINS}, {equivalent_evidence}, {ALPHA_THRESHOLD_DC}"
    thresholds = load_opinion_thresholds()
    threshold = float(
        thresholds.get(
            key,
            calc_threshold_n_diff(NUM_PIT_BINS, equivalent_evidence, 0.1),
        )
    )
    return n_st, float(discount), threshold


def create_ltst(rate_hz: float):
    n_st, discount, threshold = rate_normalised_ltst_parameters(rate_hz)
    memory = eval(f"sl.LongShortTermMemory{NUM_PIT_BINS}d")(
        n_st,
        threshold,
        discount,
        LTST_FUSION_TYPE,
        HANDLE_SHORT_TERM_CONFLICT,
        AVERAGE_DC_CONFLICT_HANDLING,
    )
    return memory, n_st, discount, threshold


def vacuous_binomial(prior_ok: float = 0.5):
    result = sl.Opinion2d(0.0, 0.0)
    result.prior_belief_masses = [prior_ok, 1.0 - prior_ok]
    return result


def fuse_average(opinions: Iterable):
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    if len(opinions) == 1:
        return deepcopy(opinions[0])
    return sl.Fusion.fuse_opinions(sl.FusionType.AVERAGE, opinions)


def fuse_weighted(opinions: Iterable):
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    if len(opinions) == 1:
        return deepcopy(opinions[0])
    return sl.Fusion.fuse_opinions(sl.FusionType.WEIGHTED, opinions)


def p_ok(opinion) -> float:
    return float(opinion.getProjection()[0])


@dataclass
class SensorAssessmentState:
    sensor_id: int
    label: str
    nominal_rate_hz: float
    actual_rate_hz: float
    colour: str

    ltst_radial: object = field(init=False)
    ltst_x: object = field(init=False)
    ltst_y: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)

    latest_local_opinion: object = field(init=False)
    latest_timestamp: datetime | None = None

    event_times_s: list[float] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)
    nis_average: list[float] = field(default_factory=list)
    nis_lower: list[float] = field(default_factory=list)
    nis_upper: list[float] = field(default_factory=list)
    nis_window: deque = field(init=False)

    def __post_init__(self) -> None:
        self.ltst_radial, self.n_st, self.discount, self.threshold = create_ltst(
            self.actual_rate_hz
        )
        self.ltst_x, _, _, _ = create_ltst(self.actual_rate_hz)
        self.ltst_y, _, _, _ = create_ltst(self.actual_rate_hz)
        self.latest_local_opinion = vacuous_binomial()
        self.nis_window = deque(maxlen=self.n_st)

    def update(
        self,
        measurement: Detection,
        measurement_prediction,
        timestamp: datetime,
        start_time: datetime,
    ):
        innovation = (
            np.asarray(measurement.state_vector, dtype=float).reshape(-1, 1)
            - np.asarray(measurement_prediction.mean, dtype=float).reshape(-1, 1)
        )
        innovation_covar = np.asarray(measurement_prediction.covar, dtype=float)
        measurement_dimension = innovation.shape[0]

        nis_value = (
            innovation.T @ np.linalg.solve(innovation_covar, innovation)
        ).item()
        radial_pit = float(chi2.cdf(nis_value, df=measurement_dimension))

        cholesky = robust_cholesky(innovation_covar)
        whitened_innovation = np.linalg.solve(cholesky, innovation).flatten()
        component_pit = norm.cdf(whitened_innovation)

        self.ltst_radial.add(scalar_pit_to_opinion(radial_pit, NUM_PIT_BINS))
        self.ltst_x.add(scalar_pit_to_opinion(float(component_pit[0]), NUM_PIT_BINS))
        self.ltst_y.add(scalar_pit_to_opinion(float(component_pit[1]), NUM_PIT_BINS))

        radial_binomial = multinomial_to_binomial_consistency_opinion(
            self.ltst_radial.get_opinion(),
            NUM_PIT_BINS,
            prior_ok=0.5,
        )

        component_prior = float(np.sqrt(0.5))
        x_binomial = multinomial_to_binomial_consistency_opinion(
            self.ltst_x.get_opinion(),
            NUM_PIT_BINS,
            prior_ok=component_prior,
        )
        y_binomial = multinomial_to_binomial_consistency_opinion(
            self.ltst_y.get_opinion(),
            NUM_PIT_BINS,
            prior_ok=component_prior,
        )
        component_binomial = x_binomial.multiply(y_binomial)

        # Same concurrent radial/component fusion as in the first paper.
        self.latest_local_opinion = fuse_weighted(
            [radial_binomial, component_binomial]
        )
        self.latest_timestamp = timestamp

        elapsed_s = (timestamp - start_time).total_seconds()
        self.event_times_s.append(elapsed_s)
        self.p_ok_events.append(p_ok(self.latest_local_opinion))
        self.uncertainty_events.append(float(self.latest_local_opinion.uncertainty()))

        self.nis_window.append(nis_value)
        window_length = len(self.nis_window)
        degrees_of_freedom = measurement_dimension * window_length
        alpha = 0.01
        self.nis_average.append(float(np.mean(self.nis_window)))
        self.nis_lower.append(
            float(chi2.ppf(alpha / 2.0, degrees_of_freedom) / window_length)
        )
        self.nis_upper.append(
            float(chi2.ppf(1.0 - alpha / 2.0, degrees_of_freedom) / window_length)
        )
        return self.latest_local_opinion

    def freshness(self, now: datetime) -> float:
        if self.latest_timestamp is None:
            return 0.0
        age_s = max(0.0, (now - self.latest_timestamp).total_seconds())
        period_s = 1.0 / self.nominal_rate_hz
        time_constant_s = FRESHNESS_TIME_CONSTANT_IN_PERIODS * period_s
        return float(np.exp(-age_s / time_constant_s))

    def time_interpreted_opinion(self, now: datetime):
        # The original local LTST opinion is not changed. Trust discount only
        # changes its interpretation for the current downstream fusion.
        if self.latest_timestamp is None:
            return vacuous_binomial()
        return self.latest_local_opinion.trust_discount(self.freshness(now))


# =============================================================================
# Scenario generation
# =============================================================================


@dataclass(frozen=True)
class SensorDefinition:
    sensor_id: int
    label: str
    nominal_rate_hz: float
    actual_rate_hz: float
    variance: float
    colour: str
    disturb_measurements: bool


@dataclass
class ScenarioData:
    start_time: datetime
    truth: GroundTruthPath
    sensor_definitions: dict[int, SensorDefinition]
    measurements_by_sensor: dict[int, list[Detection]]
    events_by_timestamp: dict[datetime, list[tuple[int, Detection]]]


def build_sensor_definitions() -> dict[int, SensorDefinition]:
    sensor_2_actual_rate = (
        SYNCHRONOUS_REFERENCE_RATE_HZ
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE
        else SENSOR_2_RATE_HZ
    )
    return {
        1: SensorDefinition(
            1, "Sensor 1", SENSOR_1_RATE_HZ, SENSOR_1_RATE_HZ,
            MEASUREMENT_VARIANCE_SENSOR_1, "tab:blue", DISTURB_SENSOR_1,
        ),
        2: SensorDefinition(
            2,
            "Sensor 2",
            sensor_2_actual_rate,
            sensor_2_actual_rate,
            MEASUREMENT_VARIANCE_SENSOR_2,
            "tab:orange",
            DISTURB_SENSOR_2,
        ),
    }


def generate_truth(start_time: datetime) -> GroundTruthPath:
    np.random.seed(RANDOM_SEED_TRUTH)
    truth_dt = timedelta(seconds=TRUTH_DT_S)
    number_of_steps = seconds_to_truth_step(SCENARIO_DURATION_S)

    cv_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(Q_X), ConstantVelocity(Q_Y)]
    )
    right_turn_model = KnownTurnRate(
        [Q_X, Q_Y], np.radians(-20.0)
    )

    disturbance_factor = 32 if ACTIVATE_DISTURBANCES else 1
    transition_configs = {
        "noise_diff_coeff": [[Q_X, Q_Y]],
        "disturb_noise_coeff": [True, False],
        "disturbance_mode": ["jump"],
        "parameters": [[
            [seconds_to_truth_step(INCREASED_PROCESS_X_INTERVAL_S[0]), disturbance_factor],
            [seconds_to_truth_step(INCREASED_PROCESS_X_INTERVAL_S[1]), 1.0 / disturbance_factor],
        ]],
    }

    truth = GroundTruthPath([
        GroundTruthState(
            StateVector([[0.0], [5.0], [0.0], [5.0]]),
            timestamp=start_time,
        )
    ])

    turn_start = seconds_to_truth_step(TURN_INTERVAL_S[0])
    turn_end = seconds_to_truth_step(TURN_INTERVAL_S[1])

    for step in range(1, number_of_steps + 1):
        cv_model = disturbance_transition_model(cv_model, transition_configs, step)
        active_model = right_turn_model if turn_start <= step < turn_end else cv_model
        truth.append(
            GroundTruthState(
                active_model.function(
                    truth[-1], noise=True, time_interval=truth_dt
                ),
                timestamp=start_time + step * truth_dt,
            )
        )
    return truth


def sensor_truth_indices(rate_hz: float) -> np.ndarray:
    stride = int(round(TRUTH_RATE_HZ / rate_hz))
    represented_rate = TRUTH_RATE_HZ / stride
    if not np.isclose(represented_rate, rate_hz):
        raise ValueError(
            f"Truth grid {TRUTH_RATE_HZ:g} Hz cannot exactly represent {rate_hz:g} Hz."
        )
    return np.arange(
        0,
        seconds_to_truth_step(SCENARIO_DURATION_S) + 1,
        stride,
        dtype=int,
    )


def generate_sensor_measurements(
    truth: GroundTruthPath,
    definition: SensorDefinition,
) -> list[Detection]:
    seed = RANDOM_SEED_SENSOR_1 if definition.sensor_id == 1 else RANDOM_SEED_SENSOR_2
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    true_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.eye(2) * definition.variance,
    )
    filter_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.eye(2) * definition.variance,
    )

    disturbance_factor = 2 if ACTIVATE_DISTURBANCES else 1
    measurement_configs = {
        "disturbance_mode": ["jump", "outliers"],
        "parameters": [
            [
                [300, disturbance_factor, [1, 0]],
                [400, 1.0 / disturbance_factor, [1, 0]],
                [500, 1.0 / disturbance_factor, [1, 1]],
                [600, disturbance_factor, [1, 1]],
            ],
            [[100, 200, 10, 8]],
        ],
    }

    detections: list[Detection] = []
    for truth_index in sensor_truth_indices(definition.actual_rate_hz):
        state = truth[truth_index]
        elapsed_s = truth_index / TRUTH_RATE_HZ

        if (
            definition.sensor_id == 2
            and ENABLE_SENSOR_2_DROPOUT
            and SENSOR_2_DROPOUT_INTERVAL_S[0] <= elapsed_s < SENSOR_2_DROPOUT_INTERVAL_S[1]
        ):
            continue

        if definition.disturb_measurements and ACTIVATE_DISTURBANCES:
            true_model = disturbance_measurement_noise(
                true_model,
                measurement_configs,
                seconds_to_reference_step(elapsed_s),
            )

        if (
            definition.disturb_measurements
            and ACTIVATE_DISTURBANCES
            and TRUNCATED_GAUSSIAN_INTERVAL_S[0] <= elapsed_s < TRUNCATED_GAUSSIAN_INTERVAL_S[1]
        ):
            measurement = true_model.function(state, noise=False)
            measurement += sample_truncated_gaussian_noise_from_cov(
                np.asarray(true_model.noise_covar, dtype=float), rng
            )
        else:
            # Same measurement generation path as in 01_...V7.py.
            measurement = true_model.function(state, noise=True)

        detections.append(
            Detection(
                measurement,
                timestamp=state.timestamp,
                measurement_model=filter_model,
            )
        )
    return detections


def build_scenario() -> ScenarioData:
    start_time = datetime.now()
    truth = generate_truth(start_time)
    sensor_definitions = build_sensor_definitions()
    measurements_by_sensor = {
        sensor_id: generate_sensor_measurements(truth, definition)
        for sensor_id, definition in sensor_definitions.items()
    }

    events: dict[datetime, list[tuple[int, Detection]]] = defaultdict(list)
    for sensor_id, detections in measurements_by_sensor.items():
        for detection in detections:
            events[detection.timestamp].append((sensor_id, detection))
    for batch in events.values():
        batch.sort(key=lambda item: item[0])

    return ScenarioData(
        start_time,
        truth,
        sensor_definitions,
        measurements_by_sensor,
        dict(events),
    )


# =============================================================================
# Event-based filtering and subsystem-level SA fusion
# =============================================================================


@dataclass
class ProcessingResult:
    track: Track
    event_timestamps: list[datetime]
    event_times_s: list[float]
    batch_sizes: list[int]
    sensor_states: dict[int, SensorAssessmentState]
    local_p_ok_history: dict[int, list[float]]
    local_uncertainty_history: dict[int, list[float]]
    freshness_history: dict[int, list[float]]
    overall_hold: list
    overall_active_only: list
    overall_time_aware_abf: list
    overall_time_aware_wbf: list
    position_error: list[float]


def process_scenario(scenario: ScenarioData) -> ProcessingResult:
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(Q_X), ConstantVelocity(Q_Y)]
    )
    predictor = KalmanPredictor(transition_model)

    first_detection = scenario.measurements_by_sensor[1][0]
    updater = KalmanUpdater(first_detection.measurement_model)

    prior = GaussianState(
        StateVector([[0.0], [5.0], [0.0], [5.0]]),
        CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0])),
        timestamp=scenario.start_time,
    )

    sensor_states = {
        sensor_id: SensorAssessmentState(
            sensor_id,
            definition.label,
            definition.nominal_rate_hz,
            definition.actual_rate_hz,
            definition.colour,
        )
        for sensor_id, definition in scenario.sensor_definitions.items()
    }

    print("\nRate-normalised LTST settings")
    for state in sensor_states.values():
        print(
            f"  {state.label}: rate={state.actual_rate_hz:g} Hz, "
            f"n_ST={state.n_st}, discount={state.discount:.6f}, "
            f"threshold={state.threshold:.6f}"
        )

    track = Track()
    event_timestamps: list[datetime] = []
    event_times_s: list[float] = []
    batch_sizes: list[int] = []

    local_p_ok_history = {sensor_id: [] for sensor_id in sensor_states}
    local_uncertainty_history = {sensor_id: [] for sensor_id in sensor_states}
    freshness_history = {sensor_id: [] for sensor_id in sensor_states}

    overall_hold = []
    overall_active_only = []
    overall_time_aware_abf = []
    overall_time_aware_wbf = []

    truth_by_timestamp = {state.timestamp: state for state in scenario.truth}
    position_error: list[float] = []

    for timestamp in sorted(scenario.events_by_timestamp):
        batch = scenario.events_by_timestamp[timestamp]
        elapsed_s = (timestamp - scenario.start_time).total_seconds()

        # One causal prediction for the complete timestamp batch.
        prediction: GaussianStatePrediction = predictor.predict(
            prior, timestamp=timestamp
        )

        active_opinions = []
        for sensor_id, measurement in batch:
            measurement_prediction = updater.predict_measurement(
                prediction,
                measurement_model=measurement.measurement_model,
            )
            active_opinions.append(
                sensor_states[sensor_id].update(
                    measurement,
                    measurement_prediction,
                    timestamp,
                    scenario.start_time,
                )
            )

        # Baseline: direct extension of synchronous ABF with held opinions.
        latest_local = [state.latest_local_opinion for state in sensor_states.values()]
        overall_hold.append(fuse_average(latest_local))

        # Baseline: fuse only sources active at this precise event time.
        overall_active_only.append(fuse_average(active_opinions))

        # Proposed: unchanged local LTST opinion + time-reliability interpretation.
        interpreted = [
            state.time_interpreted_opinion(timestamp)
            for state in sensor_states.values()
        ]
        overall_time_aware_abf.append(fuse_average(interpreted))
        overall_time_aware_wbf.append(fuse_weighted(interpreted))

        for sensor_id, state in sensor_states.items():
            local_p_ok_history[sensor_id].append(p_ok(state.latest_local_opinion))
            local_uncertainty_history[sensor_id].append(
                float(state.latest_local_opinion.uncertainty())
            )
            freshness_history[sensor_id].append(state.freshness(timestamp))

        # Only now apply the functional measurement updates. For a simultaneous
        # batch the SA is therefore independent of the sequential update order.
        posterior = prediction
        for _, measurement in batch:
            measurement_prediction = updater.predict_measurement(
                posterior,
                measurement_model=measurement.measurement_model,
            )
            hypothesis = SingleHypothesis(
                posterior,
                measurement,
                measurement_prediction,
            )
            posterior = updater.update(hypothesis)

        prior = posterior
        track.append(posterior)
        event_timestamps.append(timestamp)
        event_times_s.append(elapsed_s)
        batch_sizes.append(len(batch))

        truth_state = truth_by_timestamp.get(timestamp)
        if truth_state is None:
            position_error.append(np.nan)
        else:
            estimate = np.asarray(posterior.state_vector, dtype=float).reshape(-1)
            ground_truth = np.asarray(truth_state.state_vector, dtype=float).reshape(-1)
            position_error.append(
                float(np.linalg.norm(estimate[[0, 2]] - ground_truth[[0, 2]]))
            )

    return ProcessingResult(
        track,
        event_timestamps,
        event_times_s,
        batch_sizes,
        sensor_states,
        local_p_ok_history,
        local_uncertainty_history,
        freshness_history,
        overall_hold,
        overall_active_only,
        overall_time_aware_abf,
        overall_time_aware_wbf,
        position_error,
    )


# =============================================================================
# Credibility-region geometry: visualisation only
# =============================================================================


def beta_side_probabilities(
    belief: float,
    disbelief: float,
    uncertainty: float,
    base_rate: float,
    tau_ok: float,
) -> tuple[float, float]:
    if uncertainty <= 1e-12:
        projection = belief
        if projection > tau_ok:
            return 1.0, 0.0
        if projection < tau_ok:
            return 0.0, 1.0
        return 0.5, 0.5
    alpha = 2.0 * belief / uncertainty + 2.0 * base_rate
    beta_parameter = 2.0 * disbelief / uncertainty + 2.0 * (1.0 - base_rate)
    return (
        float(beta.sf(tau_ok, alpha, beta_parameter)),
        float(beta.cdf(tau_ok, alpha, beta_parameter)),
    )


def projection_interval(uncertainty: float, base_rate: float) -> tuple[float, float]:
    return (
        base_rate * uncertainty,
        1.0 - (1.0 - base_rate) * uncertainty,
    )


def opinion_from_projection(
    uncertainty: float,
    projection: float,
    base_rate: float,
) -> tuple[float, float, float]:
    belief = projection - base_rate * uncertainty
    disbelief = 1.0 - uncertainty - belief
    return float(belief), float(disbelief), float(uncertainty)


def credible_boundary_value(
    uncertainty: float,
    side: str,
    eta: float,
    tau_ok: float,
    base_rate: float,
) -> float | str | None:
    if uncertainty <= 1e-10:
        return tau_ok

    low, high = projection_interval(uncertainty, base_rate)

    def probability(projection: float) -> float:
        belief, disbelief, local_u = opinion_from_projection(
            uncertainty, projection, base_rate
        )
        consistent, inconsistent = beta_side_probabilities(
            belief, disbelief, local_u, base_rate, tau_ok
        )
        return consistent if side == "consistent" else inconsistent

    f_low = probability(low + 1e-10) - eta
    f_high = probability(high - 1e-10) - eta

    if side == "consistent":
        if f_high < 0.0:
            return None
        if f_low >= 0.0:
            return "all"
    else:
        if f_low < 0.0:
            return None
        if f_high >= 0.0:
            return "all"

    return float(
        brentq(
            lambda projection: probability(projection) - eta,
            low + 1e-10,
            high - 1e-10,
        )
    )


def credible_region_geometry(
    eta: float,
    tau_ok: float,
    base_rate: float = 0.5,
    number_of_points: int = 240,
) -> dict[str, list[tuple[float, float, float]]]:
    consistent_boundary = []
    inconsistent_boundary = []
    undecided_lower = []
    undecided_upper = []

    for uncertainty in np.linspace(0.0, 1.0, number_of_points):
        low, high = projection_interval(uncertainty, base_rate)

        consistent = credible_boundary_value(
            uncertainty, "consistent", eta, tau_ok, base_rate
        )
        if consistent is None:
            consistent_limit = high
        elif consistent == "all":
            consistent_limit = low
            consistent_boundary.append(
                opinion_from_projection(uncertainty, low, base_rate)
            )
        else:
            consistent_limit = float(consistent)
            consistent_boundary.append(
                opinion_from_projection(uncertainty, consistent_limit, base_rate)
            )

        inconsistent = credible_boundary_value(
            uncertainty, "inconsistent", eta, tau_ok, base_rate
        )
        if inconsistent is None:
            inconsistent_limit = low
        elif inconsistent == "all":
            inconsistent_limit = high
            inconsistent_boundary.append(
                opinion_from_projection(uncertainty, high, base_rate)
            )
        else:
            inconsistent_limit = float(inconsistent)
            inconsistent_boundary.append(
                opinion_from_projection(uncertainty, inconsistent_limit, base_rate)
            )

        if inconsistent_limit + 1e-9 < consistent_limit:
            undecided_lower.append(
                opinion_from_projection(uncertainty, inconsistent_limit, base_rate)
            )
            undecided_upper.append(
                opinion_from_projection(uncertainty, consistent_limit, base_rate)
            )

    consistent_polygon = list(consistent_boundary) + [
        (1.0 - uncertainty, 0.0, uncertainty)
        for _, _, uncertainty in reversed(consistent_boundary)
    ]
    inconsistent_polygon = list(inconsistent_boundary) + [
        (0.0, 1.0 - uncertainty, uncertainty)
        for _, _, uncertainty in reversed(inconsistent_boundary)
    ]
    undecided_polygon = list(undecided_lower) + list(reversed(undecided_upper))

    return {
        "consistent_polygon": consistent_polygon,
        "inconsistent_polygon": inconsistent_polygon,
        "undecided_polygon": undecided_polygon,
    }


def opinion_xy(belief: float, disbelief: float, uncertainty: float) -> tuple[float, float]:
    return (
        float(belief + 0.5 * uncertainty),
        float(np.sqrt(3.0) / 2.0 * uncertainty),
    )


# =============================================================================
# Plotting and animation
# =============================================================================


def add_disturbance_spans(axis) -> None:
    for start_s, end_s, label, colour in DISTURBANCE_INTERVALS:
        if label == "S2 unavailable" and not ENABLE_SENSOR_2_DROPOUT:
            continue
        axis.axvspan(start_s, end_s, color=colour, alpha=0.08)


def plot_static_results(result: ProcessingResult) -> None:
    event_times = np.asarray(result.event_times_s, dtype=float)

    # Sensor-local SA and temporal availability.
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for sensor_id, state in result.sensor_states.items():
        axes[0].plot(
            event_times,
            result.local_p_ok_history[sensor_id],
            color=state.colour,
            label=rf"$P_{{OK}}^{{({sensor_id})}}$",
        )
        axes[1].plot(
            event_times,
            result.local_uncertainty_history[sensor_id],
            color=state.colour,
            label=rf"$u^{{({sensor_id})}}$",
        )
        axes[2].plot(
            event_times,
            result.freshness_history[sensor_id],
            color=state.colour,
            label=rf"$\rho_{{{sensor_id}}}(t)$",
        )

    axes[3].step(
        event_times,
        result.batch_sizes,
        where="post",
        color="black",
        label="sensors in current event batch",
    )

    axes[0].set_ylabel(r"$P_{OK}$")
    axes[1].set_ylabel("uncertainty")
    axes[2].set_ylabel("freshness")
    axes[3].set_ylabel("batch size")
    axes[3].set_xlabel("time [s]")

    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)

    mode = (
        "fully synchronous 10 Hz special case"
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE
        else "asynchronous multi-rate case: 10 Hz / 12.5 Hz"
    )
    figure.suptitle(f"Sensor-resolved PIT/SL self-assessment ({mode})")
    figure.tight_layout()

    # Overall fusion comparison.
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    series = [
        ("naive last-opinion hold + ABF", result.overall_hold, "tab:gray"),
        ("active sensors only + ABF", result.overall_active_only, "tab:orange"),
        ("proposed time-aware + ABF", result.overall_time_aware_abf, "tab:blue"),
        ("time-aware optional-input + WBF", result.overall_time_aware_wbf, "tab:green"),
    ]
    for label, opinions, colour in series:
        axes[0].plot(
            event_times,
            [p_ok(opinion) for opinion in opinions],
            color=colour,
            label=label,
        )
        axes[1].plot(
            event_times,
            [float(opinion.uncertainty()) for opinion in opinions],
            color=colour,
            label=label,
        )

    axes[0].set_ylabel(r"overall $P_{OK}$")
    axes[1].set_ylabel("overall uncertainty")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    figure.suptitle("Synchronous fusion baseline versus asynchronous time-aware fusion")
    figure.tight_layout()

    # Sensor-specific time-average NIS.
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for axis, (sensor_id, state) in zip(axes, result.sensor_states.items()):
        axis.plot(
            state.event_times_s,
            state.nis_average,
            color=state.colour,
            label=f"{state.label}: time-average NIS",
        )
        axis.plot(
            state.event_times_s,
            state.nis_lower,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="99% interval",
        )
        axis.plot(
            state.event_times_s,
            state.nis_upper,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_ylabel("NIS")
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Sensor-specific innovation-consistency reference")
    figure.tight_layout()

    # Ground-truth based position error, evaluation only.
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.plot(event_times, result.position_error, color="tab:blue")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("position error [m]")
    axis.set_title("Single-run position error (GT is not used by the SA)")
    axis.grid(True)
    add_disturbance_spans(axis)
    figure.tight_layout()

    # Credibility regions remain drawn, but no region decisions are computed.
    geometry = credible_region_geometry(
        CREDIBLE_REGION_ETA,
        CREDIBLE_REGION_TAU_OK,
        base_rate=0.5,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    triangle = np.asarray([
        opinion_xy(0.0, 1.0, 0.0),
        opinion_xy(1.0, 0.0, 0.0),
        opinion_xy(0.0, 0.0, 1.0),
        opinion_xy(0.0, 1.0, 0.0),
    ])
    axis.plot(triangle[:, 0], triangle[:, 1], color="black")

    for key, colour, alpha in (
        ("consistent_polygon", "green", 0.12),
        ("undecided_polygon", "gray", 0.12),
        ("inconsistent_polygon", "red", 0.10),
    ):
        polygon = np.asarray([opinion_xy(*point) for point in geometry[key]])
        if len(polygon) >= 3:
            axis.fill(polygon[:, 0], polygon[:, 1], color=colour, alpha=alpha)

    trajectory = np.asarray([
        opinion_xy(
            float(opinion.belief()),
            float(opinion.disbelief()),
            float(opinion.uncertainty()),
        )
        for opinion in result.overall_time_aware_abf
    ])
    scatter = axis.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=event_times,
        cmap="viridis",
        s=7,
        alpha=0.65,
    )
    figure.colorbar(scatter, ax=axis, label="time [s]")
    axis.text(1.02, -0.02, "belief", ha="left", va="top")
    axis.text(-0.02, -0.02, "disbelief", ha="right", va="top")
    axis.text(0.5, np.sqrt(3.0) / 2.0 + 0.03, "uncertainty", ha="center")
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(
        "Overall opinion trajectory with credible regions\n"
        "(regions are visualised only, not evaluated)"
    )
    figure.tight_layout()


def sanitise_track_covariance(track: Track) -> Track:
    clean = Track()
    for state in track:
        covariance = np.asarray(state.covar, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues.real, 1e-8)
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        clean.append(
            GaussianState(
                state.state_vector,
                covariance,
                timestamp=state.timestamp,
            )
        )
    return clean


def show_dynamic_animation(scenario: ScenarioData, result: ProcessingResult) -> None:
    plotter = AnimatedPlotterly(
        result.event_timestamps,
        tail_length=ANIMATION_TAIL_LENGTH,
    )
    plotter.plot_ground_truths(scenario.truth, [0, 2])

    before = len(plotter.fig.data)
    plotter.plot_measurements(scenario.measurements_by_sensor[1], [0, 2])
    for trace in plotter.fig.data[before:]:
        trace.name = "Sensor 1 measurements"
        if hasattr(trace, "marker"):
            trace.marker.color = "royalblue"

    before = len(plotter.fig.data)
    plotter.plot_measurements(scenario.measurements_by_sensor[2], [0, 2])
    for trace in plotter.fig.data[before:]:
        trace.name = "Sensor 2 measurements"
        if hasattr(trace, "marker"):
            trace.marker.color = "darkorange"

    plotter.plot_tracks(
        sanitise_track_covariance(result.track),
        [0, 2],
        uncertainty=True,
    )

    mode = (
        "fully synchronous 10 Hz special case"
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE
        else "asynchronous: sensor 1 at 10 Hz, sensor 2 at 12.5 Hz"
    )
    plotter.fig.update_layout(
        title=(
            "Dynamic multi-sensor Kalman-filter scenario"
            f"<br><sup>{mode}</sup>"
        ),
        height=850,
    )
    plotter.fig.show(renderer="browser")


def print_summary(result: ProcessingResult) -> None:
    simultaneous = sum(size == 2 for size in result.batch_sizes)
    single = sum(size == 1 for size in result.batch_sizes)
    hold = np.asarray([p_ok(opinion) for opinion in result.overall_hold])
    time_aware = np.asarray([
        p_ok(opinion) for opinion in result.overall_time_aware_abf
    ])

    print("\nEvent statistics")
    print(f"  filter events: {len(result.event_timestamps)}")
    print(f"  simultaneous two-sensor batches: {simultaneous}")
    print(f"  single-sensor events: {single}")
    print(
        "  max |P_OK(hold)-P_OK(time-aware)|: "
        f"{np.max(np.abs(hold - time_aware)):.6f}"
    )
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE and not ENABLE_SENSOR_2_DROPOUT:
        print(
            "  In the pure synchronous special case this difference should "
            "be numerically zero: the new method reduces to ordinary fusion."
        )


def main() -> None:
    print("=" * 80)
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        print("FULLY SYNCHRONOUS SPECIAL CASE: both sensors at 10 Hz")
        if ENABLE_SENSOR_2_DROPOUT:
            print(
                "NOTE: dropout interrupts synchrony in [60 s, 70 s). Set "
                "ENABLE_SENSOR_2_DROPOUT=False for the pure limiting case."
            )
    else:
        print("ASYNCHRONOUS MULTI-RATE CASE: sensor 1 = 10 Hz, sensor 2 = 12.5 Hz")
    print("=" * 80)

    scenario = build_scenario()
    result = process_scenario(scenario)
    print_summary(result)

    if SHOW_DYNAMIC_ANIMATION:
        show_dynamic_animation(scenario, result)
    if SHOW_MATPLOTLIB_PLOTS:
        plot_static_results(result)
        plt.show()


if __name__ == "__main__":
    main()