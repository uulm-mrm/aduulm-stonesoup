#!/usr/bin/env python3
"""
02 - Event-based multi-sensor Kalman filter with standardized PIT/SL
self-assessment.

This script extends the first-paper implementation without replacing its
methodological core:
- Stone Soup is used for truth generation and Kalman filtering,
- the external ``subjective_logic`` library is used unchanged,
- every statistical channel is mapped by PIT to a seven-bin SL opinion,
- the library's TEF-enabled ``LongShortTermMemory7d`` is retained,
- radial and component-wise opinions are mapped and fused as in the first paper.

Multi-sensor additions:
1. a central multi-sensor Kalman filter for the functional track,
2. one isolated assessment Kalman filter per sensor to avoid cross-contamination,
3. a Griebel-style common-prediction assessment as a comparison baseline,
4. a direct batch-NIS/PIT/TEF opinion for the central fusion filter, even when
   the active sensor set and therefore the NIS degrees of freedom change,
5. separate overall opinions for direct central-filter consistency and for
   operational sensor-subsystem requirements (all required / at least one),
6. a separate availability opinion based only on expected measurement events.

The local consistency opinion C_s and the availability opinion A_s are kept
separate. For operational subsystem statements, each sensor path is represented
by H_s = A_s AND C_s. Availability is not used as a continuous freshness score.

Scope assumption for availability assessment:
Every available sensor is expected to provide exactly one already associated
measurement at every scheduled update (P_D = 1). Missed detections, clutter,
association uncertainty, and out-of-sequence measurements are not modelled.
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
from scipy.linalg import block_diag
from scipy.optimize import brentq
from scipy.stats import beta, chi2, kstest, norm

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
SHOW_CREDIBILITY_REGIONS = True
SHOW_POSITION_ERROR = True

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

# Availability is assessed only at scheduled measurement events. Between two
# regular events no continuous age/freshness discount is applied.
ASSUME_DETECTION_PROBABILITY_ONE = True

# Credibility regions are plotted only, not classified/evaluated.
CREDIBLE_REGION_ETA = 0.90
CREDIBLE_REGION_TAU_OK = 0.50

ANIMATION_TAIL_LENGTH = .20
NOMINAL_BURN_IN_S = REFERENCE_SHORT_TERM_HORIZON_S
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


def rate_normalised_temporal_parameters(
    rate_hz: float,
    domain_size: int,
) -> tuple[int, float, float]:
    """Return TEF/LTST parameters with a common physical time horizon."""
    n_st = max(2, int(round(REFERENCE_SHORT_TERM_HORIZON_S * rate_hz)))
    discount = REFERENCE_LONG_TERM_DISCOUNT ** (REFERENCE_RATE_HZ / rate_hz)
    equivalent_evidence = calculate_long_term_evidence(domain_size, discount)
    key = f"{domain_size}, {equivalent_evidence}, {ALPHA_THRESHOLD_DC}"
    thresholds = load_opinion_thresholds()
    threshold = float(
        thresholds.get(
            key,
            calc_threshold_n_diff(
                domain_size, equivalent_evidence, ALPHA_THRESHOLD_DC
            ),
        )
    )
    return n_st, float(discount), threshold


def create_temporal_memory(rate_hz: float, domain_size: int):
    """Create the TEF-enabled memory provided by ``subjective_logic``."""
    n_st, discount, threshold = rate_normalised_temporal_parameters(
        rate_hz, domain_size
    )
    memory = eval(f"sl.LongShortTermMemory{domain_size}d")(
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


def prior_ok(opinion) -> float:
    try:
        return float(opinion.prior_belief_masses[0])
    except (AttributeError, IndexError, TypeError):
        return 0.5


def negate_binomial(opinion):
    """Subjective-logic negation for a binomial opinion."""
    result = sl.Opinion2d(
        float(opinion.disbelief()),
        float(opinion.belief()),
    )
    a = 1.0 - prior_ok(opinion)
    result.prior_belief_masses = [a, 1.0 - a]
    return result


def conjunction_all(opinions: Iterable):
    """Logical AND over distinct sensor-consistency propositions."""
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    result = deepcopy(opinions[0])
    for opinion in opinions[1:]:
        result = result.multiply(opinion)
    return result


def disjunction_all(opinions: Iterable):
    """Logical OR via De Morgan's law."""
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    negated = [negate_binomial(opinion) for opinion in opinions]
    return negate_binomial(conjunction_all(negated))


@dataclass
class SensorConsistencyState:
    """PIT/TEF consistency assessment for one sensor and one filter context."""

    sensor_id: int
    label: str
    actual_rate_hz: float
    colour: str
    context_label: str

    tef_radial: object = field(init=False)
    tef_x: object = field(init=False)
    tef_y: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)

    latest_opinion: object = field(init=False)
    latest_timestamp: datetime | None = None

    event_times_s: list[float] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)
    nis_average: list[float] = field(default_factory=list)
    nis_lower: list[float] = field(default_factory=list)
    nis_upper: list[float] = field(default_factory=list)
    nis_window: deque = field(init=False)

    def __post_init__(self) -> None:
        self.tef_radial, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(self.actual_rate_hz, NUM_PIT_BINS)
        )
        self.tef_x, _, _, _ = create_temporal_memory(
            self.actual_rate_hz, NUM_PIT_BINS
        )
        self.tef_y, _, _, _ = create_temporal_memory(
            self.actual_rate_hz, NUM_PIT_BINS
        )
        self.latest_opinion = vacuous_binomial()
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

        self.tef_radial.add(scalar_pit_to_opinion(radial_pit, NUM_PIT_BINS))
        self.tef_x.add(
            scalar_pit_to_opinion(float(component_pit[0]), NUM_PIT_BINS)
        )
        self.tef_y.add(
            scalar_pit_to_opinion(float(component_pit[1]), NUM_PIT_BINS)
        )

        radial_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_radial.get_opinion(), NUM_PIT_BINS, prior_ok=0.5
        )
        component_prior = float(np.sqrt(0.5))
        x_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_x.get_opinion(), NUM_PIT_BINS, prior_ok=component_prior
        )
        y_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_y.get_opinion(), NUM_PIT_BINS, prior_ok=component_prior
        )
        component_binomial = x_binomial.multiply(y_binomial)

        # Same concurrent radial/component fusion as in the first paper.
        self.latest_opinion = fuse_weighted(
            [radial_binomial, component_binomial]
        )
        self.latest_timestamp = timestamp

        elapsed_s = (timestamp - start_time).total_seconds()
        self.event_times_s.append(elapsed_s)
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.uncertainty_events.append(
            float(self.latest_opinion.uncertainty())
        )

        self.nis_window.append(nis_value)
        window_length = len(self.nis_window)
        degrees_of_freedom = measurement_dimension * window_length
        alpha = 0.01
        self.nis_average.append(float(np.mean(self.nis_window)))
        self.nis_lower.append(
            float(chi2.ppf(alpha / 2.0, degrees_of_freedom) / window_length)
        )
        self.nis_upper.append(
            float(
                chi2.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
                / window_length
            )
        )
        return self.latest_opinion


@dataclass
class AvailabilityAssessmentState:
    """Availability/timeliness opinion from scheduled measurement events."""

    sensor_id: int
    label: str
    actual_rate_hz: float
    colour: str

    tef: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)
    latest_opinion: object = field(init=False)
    event_times_s: list[float] = field(default_factory=list)
    p_available_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tef, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(self.actual_rate_hz, 2)
        )
        self.latest_opinion = vacuous_binomial()

    def update(
        self,
        measurement_arrived: bool,
        timestamp: datetime,
        start_time: datetime,
    ):
        evidence = np.array(
            [1.0, 0.0] if measurement_arrived else [0.0, 1.0],
            dtype=float,
        )
        observation = sl.DirichletDistribution2d.from_evidences(
            evidence
        ).as_opinion()
        self.tef.add(observation)
        self.latest_opinion = self.tef.get_opinion()

        self.event_times_s.append((timestamp - start_time).total_seconds())
        self.p_available_events.append(p_ok(self.latest_opinion))
        self.uncertainty_events.append(
            float(self.latest_opinion.uncertainty())
        )
        return self.latest_opinion

    def operational_health(self, consistency_opinion):
        """Return H_s = A_s AND C_s for the complete sensor path.

        The availability opinion A_s and consistency opinion C_s describe
        distinct propositions. Their conjunction means that the sensor path is
        both available and statistically consistent. This is deliberately not
        a trust discount or a continuous freshness model.
        """
        return self.latest_opinion.multiply(consistency_opinion)


@dataclass
class IsolatedAssessmentFilter:
    """A sensor-only shadow KF used solely to produce isolated innovations."""

    sensor_id: int
    assessment: SensorConsistencyState
    prior: GaussianState
    predictor: KalmanPredictor

    def process_measurement(
        self,
        measurement: Detection,
        start_time: datetime,
    ):
        timestamp = measurement.timestamp
        prediction = self.predictor.predict(self.prior, timestamp=timestamp)
        updater = KalmanUpdater(measurement.measurement_model)
        measurement_prediction = updater.predict_measurement(
            prediction,
            measurement_model=measurement.measurement_model,
        )
        opinion = self.assessment.update(
            measurement,
            measurement_prediction,
            timestamp,
            start_time,
        )
        hypothesis = SingleHypothesis(
            prediction,
            measurement,
            measurement_prediction,
        )
        self.prior = updater.update(hypothesis)
        return opinion


@dataclass
class BatchTrackAssessmentState:
    """Direct consistency assessment of the active central measurement batch."""

    event_rate_hz: float
    tef_radial: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)
    latest_opinion: object = field(init=False)

    event_times_s: list[float] = field(default_factory=list)
    pit_events: list[float] = field(default_factory=list)
    nis_events: list[float] = field(default_factory=list)
    degrees_of_freedom: list[int] = field(default_factory=list)
    active_sensor_counts: list[int] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tef_radial, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(self.event_rate_hz, NUM_PIT_BINS)
        )
        self.latest_opinion = vacuous_binomial()

    def update(
        self,
        nis_value: float,
        degrees_of_freedom: int,
        active_sensor_count: int,
        timestamp: datetime,
        start_time: datetime,
    ):
        pit_value = float(chi2.cdf(nis_value, df=degrees_of_freedom))
        self.tef_radial.add(
            scalar_pit_to_opinion(pit_value, NUM_PIT_BINS)
        )
        self.latest_opinion = multinomial_to_binomial_consistency_opinion(
            self.tef_radial.get_opinion(), NUM_PIT_BINS, prior_ok=0.5
        )

        self.event_times_s.append((timestamp - start_time).total_seconds())
        self.pit_events.append(pit_value)
        self.nis_events.append(float(nis_value))
        self.degrees_of_freedom.append(int(degrees_of_freedom))
        self.active_sensor_counts.append(int(active_sensor_count))
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.uncertainty_events.append(
            float(self.latest_opinion.uncertainty())
        )
        return self.latest_opinion


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


@dataclass(frozen=True)
class ScheduledSensorEvent:
    sensor_id: int
    timestamp: datetime
    measurement: Detection | None

    @property
    def arrived(self) -> bool:
        return self.measurement is not None


@dataclass
class ScenarioData:
    start_time: datetime
    truth: GroundTruthPath
    sensor_definitions: dict[int, SensorDefinition]
    measurements_by_sensor: dict[int, list[Detection]]
    events_by_timestamp: dict[datetime, list[ScheduledSensorEvent]]
    nominal_batch_event_rate_hz: float


def build_sensor_definitions() -> dict[int, SensorDefinition]:
    sensor_2_actual_rate = (
        SYNCHRONOUS_REFERENCE_RATE_HZ
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE
        else SENSOR_2_RATE_HZ
    )
    return {
        1: SensorDefinition(
            1,
            "Sensor 1",
            SENSOR_1_RATE_HZ,
            SENSOR_1_RATE_HZ,
            MEASUREMENT_VARIANCE_SENSOR_1,
            "tab:blue",
            DISTURB_SENSOR_1,
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
            [
                seconds_to_truth_step(INCREASED_PROCESS_X_INTERVAL_S[0]),
                disturbance_factor,
            ],
            [
                seconds_to_truth_step(INCREASED_PROCESS_X_INTERVAL_S[1]),
                1.0 / disturbance_factor,
            ],
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
        cv_model = disturbance_transition_model(
            cv_model, transition_configs, step
        )
        active_model = (
            right_turn_model if turn_start <= step < turn_end else cv_model
        )
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
            f"Truth grid {TRUTH_RATE_HZ:g} Hz cannot exactly represent "
            f"{rate_hz:g} Hz."
        )
    return np.arange(
        0,
        seconds_to_truth_step(SCENARIO_DURATION_S) + 1,
        stride,
        dtype=int,
    )


def generate_sensor_schedule(
    truth: GroundTruthPath,
    definition: SensorDefinition,
) -> tuple[list[ScheduledSensorEvent], list[Detection]]:
    seed = (
        RANDOM_SEED_SENSOR_1
        if definition.sensor_id == 1
        else RANDOM_SEED_SENSOR_2
    )
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

    schedule: list[ScheduledSensorEvent] = []
    detections: list[Detection] = []
    for truth_index in sensor_truth_indices(definition.actual_rate_hz):
        state = truth[truth_index]
        elapsed_s = truth_index / TRUTH_RATE_HZ

        unavailable = (
            definition.sensor_id == 2
            and ENABLE_SENSOR_2_DROPOUT
            and SENSOR_2_DROPOUT_INTERVAL_S[0]
            <= elapsed_s
            < SENSOR_2_DROPOUT_INTERVAL_S[1]
        )
        if unavailable:
            schedule.append(
                ScheduledSensorEvent(
                    definition.sensor_id, state.timestamp, None
                )
            )
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
            and TRUNCATED_GAUSSIAN_INTERVAL_S[0]
            <= elapsed_s
            < TRUNCATED_GAUSSIAN_INTERVAL_S[1]
        ):
            measurement_vector = true_model.function(state, noise=False)
            measurement_vector += sample_truncated_gaussian_noise_from_cov(
                np.asarray(true_model.noise_covar, dtype=float), rng
            )
        else:
            measurement_vector = true_model.function(state, noise=True)

        detection = Detection(
            measurement_vector,
            timestamp=state.timestamp,
            measurement_model=filter_model,
        )
        detections.append(detection)
        schedule.append(
            ScheduledSensorEvent(
                definition.sensor_id, state.timestamp, detection
            )
        )

    return schedule, detections


def build_scenario() -> ScenarioData:
    start_time = datetime.now()
    truth = generate_truth(start_time)
    sensor_definitions = build_sensor_definitions()

    schedules: dict[int, list[ScheduledSensorEvent]] = {}
    measurements_by_sensor: dict[int, list[Detection]] = {}
    for sensor_id, definition in sensor_definitions.items():
        schedules[sensor_id], measurements_by_sensor[sensor_id] = (
            generate_sensor_schedule(truth, definition)
        )

    events: dict[datetime, list[ScheduledSensorEvent]] = defaultdict(list)
    for schedule in schedules.values():
        for event in schedule:
            events[event.timestamp].append(event)
    for batch in events.values():
        batch.sort(key=lambda item: item.sensor_id)

    nominal_batch_event_rate_hz = (
        max(1, len(events) - 1) / SCENARIO_DURATION_S
    )
    return ScenarioData(
        start_time,
        truth,
        sensor_definitions,
        measurements_by_sensor,
        dict(events),
        nominal_batch_event_rate_hz,
    )


# =============================================================================
# Event-based filtering and subsystem-level SA fusion
# =============================================================================


def initial_filter_state(start_time: datetime) -> GaussianState:
    return GaussianState(
        StateVector([[0.0], [5.0], [0.0], [5.0]]),
        CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0])),
        timestamp=start_time,
    )


def stacked_batch_nis(
    prediction: GaussianStatePrediction,
    measurements: list[Detection],
) -> tuple[float, int]:
    """NIS of an independent-noise measurement batch against one prior."""
    z = np.vstack([
        np.asarray(measurement.state_vector, dtype=float).reshape(-1, 1)
        for measurement in measurements
    ])
    h_blocks = [
        np.asarray(measurement.measurement_model.matrix(), dtype=float)
        for measurement in measurements
    ]
    r_blocks = [
        np.asarray(measurement.measurement_model.covar(), dtype=float)
        for measurement in measurements
    ]
    h = np.vstack(h_blocks)
    r = block_diag(*r_blocks)
    x = np.asarray(prediction.state_vector, dtype=float).reshape(-1, 1)
    p = np.asarray(prediction.covar, dtype=float)
    innovation = z - h @ x
    innovation_covar = h @ p @ h.T + r
    nis = (
        innovation.T
        @ np.linalg.solve(innovation_covar, innovation)
    ).item()
    return float(nis), int(z.shape[0])


@dataclass
class ProcessingResult:
    track: Track
    event_timestamps: list[datetime]
    event_times_s: list[float]
    scheduled_batch_sizes: list[int]
    active_batch_sizes: list[int]

    isolated_states: dict[int, SensorConsistencyState]
    common_prediction_states: dict[int, SensorConsistencyState]
    availability_states: dict[int, AvailabilityAssessmentState]
    batch_track_state: BatchTrackAssessmentState

    isolated_p_ok_history: dict[int, list[float]]
    isolated_uncertainty_history: dict[int, list[float]]
    common_p_ok_history: dict[int, list[float]]
    common_uncertainty_history: dict[int, list[float]]
    availability_p_history: dict[int, list[float]]
    availability_uncertainty_history: dict[int, list[float]]
    operational_p_history: dict[int, list[float]]
    operational_uncertainty_history: dict[int, list[float]]

    overall_griebel_abf: list
    overall_all_required: list
    overall_at_least_one: list
    overall_batch_track: list
    position_error: list[float]


def process_scenario(scenario: ScenarioData) -> ProcessingResult:
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(Q_X), ConstantVelocity(Q_Y)]
    )
    central_predictor = KalmanPredictor(transition_model)
    central_prior = initial_filter_state(scenario.start_time)

    isolated_states: dict[int, SensorConsistencyState] = {}
    common_states: dict[int, SensorConsistencyState] = {}
    availability_states: dict[int, AvailabilityAssessmentState] = {}
    isolated_filters: dict[int, IsolatedAssessmentFilter] = {}

    for sensor_id, definition in scenario.sensor_definitions.items():
        isolated_state = SensorConsistencyState(
            sensor_id,
            definition.label,
            definition.actual_rate_hz,
            definition.colour,
            "isolated",
        )
        common_state = SensorConsistencyState(
            sensor_id,
            definition.label,
            definition.actual_rate_hz,
            definition.colour,
            "common prediction",
        )
        availability_state = AvailabilityAssessmentState(
            sensor_id,
            definition.label,
            definition.actual_rate_hz,
            definition.colour,
        )
        isolated_states[sensor_id] = isolated_state
        common_states[sensor_id] = common_state
        availability_states[sensor_id] = availability_state
        isolated_filters[sensor_id] = IsolatedAssessmentFilter(
            sensor_id,
            isolated_state,
            initial_filter_state(scenario.start_time),
            KalmanPredictor(
                CombinedLinearGaussianTransitionModel(
                    [ConstantVelocity(Q_X), ConstantVelocity(Q_Y)]
                )
            ),
        )

    batch_track_state = BatchTrackAssessmentState(
        scenario.nominal_batch_event_rate_hz
    )

    print("\nRate-normalised TEF settings")
    for sensor_id in sorted(isolated_states):
        consistency = isolated_states[sensor_id]
        availability = availability_states[sensor_id]
        print(
            f"  {consistency.label}: rate={consistency.actual_rate_hz:g} Hz, "
            f"consistency n_ST={consistency.n_st}, "
            f"discount={consistency.discount:.6f}, "
            f"availability n_ST={availability.n_st}"
        )
    print(
        "  Central batch channel: "
        f"nominal event rate={scenario.nominal_batch_event_rate_hz:.3f} Hz, "
        f"n_ST={batch_track_state.n_st}, "
        f"discount={batch_track_state.discount:.6f}"
    )

    track = Track()
    event_timestamps: list[datetime] = []
    event_times_s: list[float] = []
    scheduled_batch_sizes: list[int] = []
    active_batch_sizes: list[int] = []

    isolated_p_ok_history = {
        sensor_id: [] for sensor_id in isolated_states
    }
    isolated_uncertainty_history = {
        sensor_id: [] for sensor_id in isolated_states
    }
    common_p_ok_history = {
        sensor_id: [] for sensor_id in common_states
    }
    common_uncertainty_history = {
        sensor_id: [] for sensor_id in common_states
    }
    availability_p_history = {
        sensor_id: [] for sensor_id in availability_states
    }
    availability_uncertainty_history = {
        sensor_id: [] for sensor_id in availability_states
    }
    operational_p_history = {
        sensor_id: [] for sensor_id in isolated_states
    }
    operational_uncertainty_history = {
        sensor_id: [] for sensor_id in isolated_states
    }

    overall_griebel_abf = []
    overall_all_required = []
    overall_at_least_one = []
    overall_batch_track = []

    truth_by_timestamp = {state.timestamp: state for state in scenario.truth}
    position_error: list[float] = []

    for timestamp in sorted(scenario.events_by_timestamp):
        scheduled_batch = scenario.events_by_timestamp[timestamp]
        active_events = [event for event in scheduled_batch if event.arrived]
        active_measurements = [
            event.measurement for event in active_events
            if event.measurement is not None
        ]
        elapsed_s = (timestamp - scenario.start_time).total_seconds()

        # Central causal prediction used by the functional filter and by the
        # Griebel-style common-prediction baseline.
        central_prediction: GaussianStatePrediction = central_predictor.predict(
            central_prior, timestamp=timestamp
        )

        # Availability is updated only at each sensor's scheduled events.
        for event in scheduled_batch:
            availability_states[event.sensor_id].update(
                event.arrived,
                timestamp,
                scenario.start_time,
            )

        # Local consistency opinions from two alternative architectures.
        for event in active_events:
            measurement = event.measurement
            assert measurement is not None

            common_updater = KalmanUpdater(measurement.measurement_model)
            common_measurement_prediction = common_updater.predict_measurement(
                central_prediction,
                measurement_model=measurement.measurement_model,
            )
            common_states[event.sensor_id].update(
                measurement,
                common_measurement_prediction,
                timestamp,
                scenario.start_time,
            )

            isolated_filters[event.sensor_id].process_measurement(
                measurement,
                scenario.start_time,
            )

        # Direct central-filter assessment. The active set changes the NIS
        # degrees of freedom, while the PIT maps every valid null distribution
        # to the same U(0,1) evidence domain before the common TEF.
        if active_measurements:
            batch_nis, batch_df = stacked_batch_nis(
                central_prediction, active_measurements
            )
            batch_track_state.update(
                batch_nis,
                batch_df,
                len(active_measurements),
                timestamp,
                scenario.start_time,
            )

        # H_s = A_s AND C_s: the complete sensor path must be both available
        # and statistically consistent. This is different from trust discount.
        operational_opinions = []
        for sensor_id in sorted(isolated_states):
            operational_opinions.append(
                availability_states[sensor_id].operational_health(
                    isolated_states[sensor_id].latest_opinion
                )
            )

        # Baseline: common-prediction sensor SAs with ABF. In asynchronous
        # mode, holding the last opinion is only a straightforward extension of
        # the published synchronous method, not part of the original method.
        overall_griebel_abf.append(
            fuse_average([
                common_states[sensor_id].latest_opinion
                for sensor_id in sorted(common_states)
            ])
        )

        # Two explicit operational subsystem requirements.
        overall_all_required.append(conjunction_all(operational_opinions))
        overall_at_least_one.append(disjunction_all(operational_opinions))

        # Direct consistency opinion of the actually used central filter.
        overall_batch_track.append(
            deepcopy(batch_track_state.latest_opinion)
        )

        for index, sensor_id in enumerate(sorted(isolated_states)):
            isolated_opinion = isolated_states[sensor_id].latest_opinion
            common_opinion = common_states[sensor_id].latest_opinion
            availability_opinion = availability_states[sensor_id].latest_opinion
            operational_opinion = operational_opinions[index]

            isolated_p_ok_history[sensor_id].append(p_ok(isolated_opinion))
            isolated_uncertainty_history[sensor_id].append(
                float(isolated_opinion.uncertainty())
            )
            common_p_ok_history[sensor_id].append(p_ok(common_opinion))
            common_uncertainty_history[sensor_id].append(
                float(common_opinion.uncertainty())
            )
            availability_p_history[sensor_id].append(
                p_ok(availability_opinion)
            )
            availability_uncertainty_history[sensor_id].append(
                float(availability_opinion.uncertainty())
            )
            operational_p_history[sensor_id].append(
                p_ok(operational_opinion)
            )
            operational_uncertainty_history[sensor_id].append(
                float(operational_opinion.uncertainty())
            )

        # Functional central multi-sensor update, performed only after every
        # assessment for the current timestamp has been calculated.
        central_posterior = central_prediction
        for measurement in active_measurements:
            updater = KalmanUpdater(measurement.measurement_model)
            measurement_prediction = updater.predict_measurement(
                central_posterior,
                measurement_model=measurement.measurement_model,
            )
            hypothesis = SingleHypothesis(
                central_posterior,
                measurement,
                measurement_prediction,
            )
            central_posterior = updater.update(hypothesis)

        central_prior = central_posterior
        track.append(central_posterior)
        event_timestamps.append(timestamp)
        event_times_s.append(elapsed_s)
        scheduled_batch_sizes.append(len(scheduled_batch))
        active_batch_sizes.append(len(active_measurements))

        truth_state = truth_by_timestamp.get(timestamp)
        if truth_state is None:
            position_error.append(np.nan)
        else:
            estimate = np.asarray(
                central_posterior.state_vector, dtype=float
            ).reshape(-1)
            ground_truth = np.asarray(
                truth_state.state_vector, dtype=float
            ).reshape(-1)
            position_error.append(
                float(
                    np.linalg.norm(
                        estimate[[0, 2]] - ground_truth[[0, 2]]
                    )
                )
            )

    return ProcessingResult(
        track,
        event_timestamps,
        event_times_s,
        scheduled_batch_sizes,
        active_batch_sizes,
        isolated_states,
        common_states,
        availability_states,
        batch_track_state,
        isolated_p_ok_history,
        isolated_uncertainty_history,
        common_p_ok_history,
        common_uncertainty_history,
        availability_p_history,
        availability_uncertainty_history,
        operational_p_history,
        operational_uncertainty_history,
        overall_griebel_abf,
        overall_all_required,
        overall_at_least_one,
        overall_batch_track,
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


def is_nominal_time(time_s: float) -> bool:
    if time_s < NOMINAL_BURN_IN_S:
        return False
    return not any(
        start <= time_s < end
        for start, end, _, _ in DISTURBANCE_INTERVALS
        if not (end == SENSOR_2_DROPOUT_INTERVAL_S[1]
                and start == SENSOR_2_DROPOUT_INTERVAL_S[0]
                and not ENABLE_SENSOR_2_DROPOUT)
    )


def add_single_run_nominal_band(
    axis,
    times: np.ndarray,
    values: list[float],
    colour: str,
) -> None:
    """Evaluation-only reference band from undisturbed parts of this run."""
    array = np.asarray(values, dtype=float)
    mask = np.asarray([is_nominal_time(t) for t in times], dtype=bool)
    valid = array[mask & np.isfinite(array)]
    if valid.size < 10:
        return
    lower, median, upper = np.quantile(valid, [0.10, 0.50, 0.90])
    axis.axhspan(lower, upper, color=colour, alpha=0.07)
    axis.axhline(median, color=colour, linestyle=":", linewidth=1.0)


def phase_mean_bdu(opinions: list, mask: np.ndarray):
    selected = [opinion for opinion, keep in zip(opinions, mask) if keep]
    if not selected:
        return None
    return (
        float(np.mean([opinion.belief() for opinion in selected])),
        float(np.mean([opinion.disbelief() for opinion in selected])),
        float(np.mean([opinion.uncertainty() for opinion in selected])),
    )


def plot_static_results(result: ProcessingResult) -> None:
    event_times = np.asarray(result.event_times_s, dtype=float)

    # 1) Local diagnosis: isolated sensor-only assessment versus common prior.
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for row, sensor_id in enumerate(sorted(result.isolated_states)):
        state = result.isolated_states[sensor_id]
        axes[row].plot(
            event_times,
            result.isolated_p_ok_history[sensor_id],
            color=state.colour,
            linewidth=1.7,
            label="isolated sensor-only assessment KF",
        )
        axes[row].plot(
            event_times,
            result.common_p_ok_history[sensor_id],
            color=state.colour,
            linestyle="--",
            linewidth=1.1,
            label="common central prediction baseline",
        )
        add_single_run_nominal_band(
            axes[row],
            event_times,
            result.isolated_p_ok_history[sensor_id],
            state.colour,
        )
        axes[row].set_ylabel(rf"$P_{{OK}}^{{({sensor_id})}}$")
        axes[row].set_title(state.label, loc="left", fontsize=10)

    for sensor_id in sorted(result.availability_states):
        state = result.availability_states[sensor_id]
        axes[2].plot(
            event_times,
            result.availability_p_history[sensor_id],
            color=state.colour,
            label=f"{state.label}: $P_A$",
        )
    axes[2].set_ylabel(r"availability $P_A$")
    axes[2].set_xlabel("time [s]")

    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)

    mode = (
        "fully synchronous 10 Hz special case"
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE
        else "asynchronous multi-rate case: 10 Hz / 12.5 Hz"
    )
    figure.suptitle(
        "Local sensor diagnosis and measurement availability\n"
        f"({mode}; dotted bands are single-run nominal references only)"
    )
    figure.tight_layout()

    # 2) Comparable overall-filter constructions only.
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    comparable = [
        (
            "common-prediction ABF baseline",
            result.overall_griebel_abf,
            "tab:gray",
            "--",
        ),
        (
            "direct central batch-track opinion",
            result.overall_batch_track,
            "tab:purple",
            "-",
        ),
    ]
    for label, opinions, colour, linestyle in comparable:
        axes[0].plot(
            event_times,
            [p_ok(opinion) for opinion in opinions],
            color=colour,
            linestyle=linestyle,
            label=label,
        )
        axes[1].plot(
            event_times,
            [float(opinion.uncertainty()) for opinion in opinions],
            color=colour,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_ylabel(r"overall $P_{OK}$")
    axes[1].set_ylabel("uncertainty")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    figure.suptitle(
        "Overall consistency of the central multi-sensor filter\n"
        "(the asynchronous common-prediction curve holds each sensor's last opinion)"
    )
    figure.tight_layout()

    # 3) Operational subsystem requirements. H_s = A_s AND C_s.
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    operational_series = [
        (
            "all sensor paths required: $H_1 \\wedge H_2$",
            result.overall_all_required,
            "tab:red",
        ),
        (
            "at least one sensor path sufficient: $H_1 \\vee H_2$",
            result.overall_at_least_one,
            "tab:green",
        ),
    ]
    for label, opinions, colour in operational_series:
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
    axes[0].set_ylabel(r"operational $P$")
    axes[1].set_ylabel("uncertainty")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    figure.suptitle(
        "Operational sensor-subsystem health with $H_s=A_s \\wedge C_s$\n"
        "(AND and OR are different system requirements, not competing fusion methods)"
    )
    figure.tight_layout()

    # 4) Sensor-specific NIS references.
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for axis, sensor_id in zip(axes, sorted(result.isolated_states)):
        isolated = result.isolated_states[sensor_id]
        common = result.common_prediction_states[sensor_id]
        axis.plot(
            isolated.event_times_s,
            isolated.nis_average,
            color=isolated.colour,
            label="isolated time-average NIS",
        )
        axis.plot(
            common.event_times_s,
            common.nis_average,
            color=isolated.colour,
            linestyle="--",
            label="common-prior time-average NIS",
        )
        axis.plot(
            isolated.event_times_s,
            isolated.nis_lower,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="99% interval (isolated window)",
        )
        axis.plot(
            isolated.event_times_s,
            isolated.nis_upper,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_ylabel("NIS")
        axis.set_title(isolated.label, loc="left", fontsize=10)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle(
        "Sensor-specific innovation consistency and prior cross-contamination"
    )
    figure.tight_layout()

    # 5) Direct batch channel: one common PIT domain despite varying df.
    batch = result.batch_track_state
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(
        batch.event_times_s,
        batch.p_ok_events,
        color="tab:purple",
        label=r"direct batch-track $P_{OK}$",
    )
    axes[0].plot(
        batch.event_times_s,
        batch.uncertainty_events,
        color="tab:purple",
        linestyle="--",
        alpha=0.75,
        label="batch-track uncertainty",
    )
    for active_count, colour, marker in (
        (1, "tab:blue", "o"),
        (2, "tab:orange", "x"),
    ):
        mask = np.asarray(batch.active_sensor_counts) == active_count
        if np.any(mask):
            axes[1].scatter(
                np.asarray(batch.event_times_s)[mask],
                np.asarray(batch.pit_events)[mask],
                color=colour,
                marker=marker,
                s=10,
                alpha=0.65,
                label=(
                    f"{active_count} active sensor(s), "
                    f"df={2 * active_count}"
                ),
            )
    axes[0].set_ylabel(r"$P_{OK}^{track}$ / $u$")
    axes[1].set_ylabel("batch PIT")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    figure.suptitle(
        "Direct central-filter assessment: PIT standardises changing batch dimension"
    )
    figure.tight_layout()

    # 6) Nominal PIT histograms separated by active batch size.
    figure, axis = plt.subplots(figsize=(10, 5))
    nominal_mask = np.asarray(
        [is_nominal_time(time_s) for time_s in batch.event_times_s],
        dtype=bool,
    )
    batch_sizes = np.asarray(batch.active_sensor_counts)
    pit_values = np.asarray(batch.pit_events)
    bins = np.linspace(0.0, 1.0, NUM_PIT_BINS + 1)
    for size, colour in ((1, "tab:blue"), (2, "tab:orange")):
        values = pit_values[nominal_mask & (batch_sizes == size)]
        if values.size:
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=colour,
                label=f"{size} active sensor(s), n={values.size}",
            )
    axis.axhline(1.0, color="black", linestyle="--", label="U(0,1)")
    axis.set_xlabel("batch PIT")
    axis.set_ylabel("density")
    axis.set_title(
        "Nominal batch PIT by active sensor count\n"
        "(same evidence domain despite different chi-square degrees of freedom)"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # 7) Ground-truth-based position error, evaluation only.
    if SHOW_POSITION_ERROR:
        figure, axis = plt.subplots(figsize=(15, 4))
        axis.plot(event_times, result.position_error, color="tab:blue")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("position error [m]")
        axis.set_title("Central track position error (GT is not used by the SA)")
        axis.grid(True)
        add_disturbance_spans(axis)
        figure.tight_layout()

    # 8) Credibility regions: phase means only, not all time samples.
    if SHOW_CREDIBILITY_REGIONS:
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
                axis.fill(
                    polygon[:, 0], polygon[:, 1], color=colour, alpha=alpha
                )

        opinions = result.overall_batch_track
        nominal_mask_events = np.asarray(
            [is_nominal_time(time_s) for time_s in event_times], dtype=bool
        )
        phases = [("nominal", nominal_mask_events, "black")]
        for start_s, end_s, label, colour in DISTURBANCE_INTERVALS:
            if label == "S2 unavailable" and not ENABLE_SENSOR_2_DROPOUT:
                continue
            mask = (event_times >= start_s) & (event_times < end_s)
            phases.append((label, mask, colour))

        for label, mask, colour in phases:
            mean_bdu = phase_mean_bdu(opinions, mask)
            if mean_bdu is None:
                continue
            x_coord, y_coord = opinion_xy(*mean_bdu)
            axis.scatter(
                [x_coord], [y_coord], color=colour, s=55, edgecolor="white"
            )
            axis.annotate(
                label,
                (x_coord, y_coord),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

        axis.text(1.02, -0.02, "belief", ha="left", va="top")
        axis.text(-0.02, -0.02, "disbelief", ha="right", va="top")
        axis.text(
            0.5,
            np.sqrt(3.0) / 2.0 + 0.03,
            "uncertainty",
            ha="center",
        )
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(
            "Direct batch-track opinion: phase means with credibility regions\n"
            "(visualisation only; no region decisions are evaluated)"
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
            "Dynamic event-based multi-sensor Kalman-filter scenario"
            f"<br><sup>{mode}</sup>"
        ),
        height=850,
    )
    plotter.fig.show(renderer="browser")


def interval_mean(
    event_times_s: list[float],
    values: list[float],
    interval: tuple[float, float],
) -> float:
    times = np.asarray(event_times_s, dtype=float)
    array = np.asarray(values, dtype=float)
    mask = (times >= interval[0]) & (times < interval[1])
    return float(np.nanmean(array[mask])) if np.any(mask) else float("nan")


def nominal_mean(event_times_s: list[float], values: list[float]) -> float:
    times = np.asarray(event_times_s, dtype=float)
    array = np.asarray(values, dtype=float)
    mask = np.asarray([is_nominal_time(t) for t in times], dtype=bool)
    return float(np.nanmean(array[mask])) if np.any(mask) else float("nan")


def print_summary(result: ProcessingResult) -> None:
    simultaneous = sum(size == 2 for size in result.active_batch_sizes)
    single = sum(size == 1 for size in result.active_batch_sizes)
    missing_only = sum(size == 0 for size in result.active_batch_sizes)

    print("\nEvent statistics")
    print(f"  scheduled event timestamps: {len(result.event_timestamps)}")
    print(f"  simultaneous two-sensor measurement batches: {simultaneous}")
    print(f"  single-sensor measurement batches: {single}")
    print(f"  scheduled timestamps without a measurement: {missing_only}")

    nominal_s2_isolated = nominal_mean(
        result.event_times_s, result.isolated_p_ok_history[2]
    )
    nominal_s2_common = nominal_mean(
        result.event_times_s, result.common_p_ok_history[2]
    )

    print("\nSensor-isolation check during Sensor-1-only disturbances")
    print(
        "  Reference means for Sensor 2 in undisturbed intervals: "
        f"isolated={nominal_s2_isolated:.3f}, "
        f"common-prior={nominal_s2_common:.3f}"
    )
    for interval, label in (
        (OUTLIER_INTERVAL_S, "S1 outliers"),
        (INCREASED_MEAS_X_INTERVAL_S, "S1 increased x-noise"),
        (DECREASED_MEAS_XY_INTERVAL_S, "S1 decreased x/y-noise"),
        (TRUNCATED_GAUSSIAN_INTERVAL_S, "S1 truncated Gaussian"),
    ):
        isolated_s2 = interval_mean(
            result.event_times_s,
            result.isolated_p_ok_history[2],
            interval,
        )
        common_s2 = interval_mean(
            result.event_times_s,
            result.common_p_ok_history[2],
            interval,
        )
        print(
            f"  {label}: Sensor 2 mean P_OK "
            f"isolated={isolated_s2:.3f} "
            f"(delta={isolated_s2 - nominal_s2_isolated:+.3f}), "
            f"common-prior={common_s2:.3f} "
            f"(delta={common_s2 - nominal_s2_common:+.3f})"
        )

    if ENABLE_SENSOR_2_DROPOUT:
        all_required = interval_mean(
            result.event_times_s,
            [p_ok(opinion) for opinion in result.overall_all_required],
            SENSOR_2_DROPOUT_INTERVAL_S,
        )
        at_least_one = interval_mean(
            result.event_times_s,
            [p_ok(opinion) for opinion in result.overall_at_least_one],
            SENSOR_2_DROPOUT_INTERVAL_S,
        )
        print("\nOperational subsystem check during Sensor-2 unavailability")
        print(f"  all required mean projection: {all_required:.3f}")
        print(f"  at least one sufficient mean projection: {at_least_one:.3f}")

    batch = result.batch_track_state
    batch_sizes = np.asarray(batch.active_sensor_counts)
    pit_values = np.asarray(batch.pit_events, dtype=float)
    batch_times = np.asarray(batch.event_times_s, dtype=float)
    nominal_mask = np.asarray(
        [is_nominal_time(time_s) for time_s in batch_times], dtype=bool
    )
    print("\nNominal direct-batch PIT calibration")
    for size in sorted(set(batch_sizes.tolist())):
        values = pit_values[nominal_mask & (batch_sizes == size)]
        if values.size < 5:
            continue
        statistic, p_value = kstest(values, "uniform")
        print(
            f"  {size} active sensor(s): n={values.size}, "
            f"KS={statistic:.3f}, p={p_value:.3f}"
        )

    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        print(
            "\nSynchronous special case: both isolated assessment filters "
            "receive measurements at the same timestamps. Sensor-1-only "
            "disturbances should not systematically shift Sensor 2's isolated "
            "nominal distribution; verify this over Monte Carlo runs rather "
            "than from one trajectory."
        )
    else:
        print(
            "\nAsynchronous multi-rate case: local TEFs are updated at each "
            "sensor's native timestamps. The direct batch channel uses the "
            "correct chi-square degrees of freedom for each active sensor set "
            "and maps all events to one PIT evidence domain."
        )
        print(
            "  The common-prediction ABF curve is a last-opinion-held "
            "asynchronous extension of Griebel's synchronous construction."
        )


def main() -> None:
    print("=" * 80)
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        print("FULLY SYNCHRONOUS SPECIAL CASE: both sensors at 10 Hz")
        if ENABLE_SENSOR_2_DROPOUT:
            print(
                "NOTE: scheduled Sensor-2 measurements are missing in [60 s, 70 s). Set "
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