#!/usr/bin/env python3
"""
02 V4 - Event-based multi-sensor Kalman filter with PIT/TEF self-assessment.

Purpose
-------
This script is a test bed for the multi-sensor extension discussed after the
first PIT/Subjective-Logic paper.  It keeps the original Stone Soup simulation
and the external ``subjective_logic`` implementation, but separates the
assessment into clearly defined statements:

1. Sensor-specific isolated consistency C_s
   - one sensor-only shadow KF per sensor,
   - radial NIS + whitened x/y innovation channels,
   - PIT -> TEF -> binomial consistency opinions.

2. Common-prediction PIT/TEF baseline C_s^common
   - exactly the same PIT/TEF mapping as (1),
   - but each active sensor is assessed against the functional central KF prior,
   - used to isolate the architectural cross-contamination effect.

3. Availability A_s
   - one binomial scheduled-output channel per sensor,
   - evidence [1,0] when an expected measurement arrives and [0,1] when it does
     not arrive,
   - processed by a separate TEF,
   - if an expected measurement is missing, the corresponding consistency TEFs
     receive a VACUOUS input: no positive/negative consistency evidence is added,
     but their temporal memory continues to advance.

4. Availability-trust-discounted consistency C~_s = A_s (*) C_s
   - trust discount changes the downstream interpretation of C_s,
   - the original consistency opinion C_s remains unchanged.

5. Direct central-filter batch consistency C_F
   - batch NIS of the actually active measurement set against the central prior,
   - the chi-square degrees of freedom change with the active set,
   - PIT maps every valid null distribution to the same U(0,1) evidence domain,
   - one TEF therefore handles asynchronous single-sensor and simultaneous
     multi-sensor events.

6. Conditional/order consistency for simultaneous two-sensor events
   - C_{2|1}: assess sensor 2 after a virtual central update with sensor 1,
   - C_{1|2}: assess sensor 1 after a virtual central update with sensor 2,
   - matched references C_{s|empty}^{sim} use the same simultaneous events,
   - I_{j->s} = d_norm(C_{s|j}) - d_norm(C_{s|empty}^{sim}) is reported as a
     signed update-order influence score; it is NOT a fault-source score.

7. Direct sensor-to-sensor agreement/disagreement at simultaneous events
   - Delta z = z_1-z_2 removes the unknown common state when both sensors
     measure the same Cartesian quantity,
   - under independent nominal noise, Delta z ~ N(0, R_1+R_2),
   - radial and signed component channels use PIT -> TEF -> SL,
   - d_norm of the resulting pair-agreement opinion is a disagreement measure,
   - signed standardized x/y differences identify the RELATIVE offset direction
     but cannot tell which of two sensors is absolutely correct.

8. Optional Griebel 2023 reference
   - when ``KalmanSelfAssessor`` from the uulm-mrm/aduulm-stonesoup branch is
     available, the script evaluates the published single-sensor SA mechanism
     against the common prediction and forms the paper's projected-probability
     overall construction from the threshold decisions,
   - if the class is unavailable, a placeholder backend returns NaNs and the
     complete script remains executable.

Temporal parametrisation
------------------------
TEF horizons are specified in physical time and converted to sample counts
using the corresponding assessment-event rate:

    local/common sensor consistency horizon = 3.5 s
    direct central-batch horizon            = 2.0 s
    availability horizon                    = 1.0 s
    conditional/order horizon               = 5.0 s
    sensor-pair disagreement horizon         = 5.0 s

Thus n_ST ~= T * f for the event stream feeding the respective TEF. Comparable
channels use matched temporal parametrisation; assessment statements with
different response requirements may use different physical horizons. The
long-term discount is also rate-normalised so that its fading behavior is
expressed consistently in physical time.

The TEF fusion operator is configurable per assessment family. CBF is the
default for temporal PIT observations because nominal KF innovations are
modeled as temporally white, so successive PIT one-hot observations represent
new evidence about the same categorical distribution.

Scope assumptions
-----------------
- P_D = 1 whenever a sensor is expected to measure the tracked object.
- Measurements are already correctly associated to the track.
- No clutter, missed detections, track initiation/deletion, or OOSM handling.
- The conditional-order analysis is primarily a two-sensor diagnostic tool.

Recommended usage
-----------------
1. Run with SYNCHRONOUS_SENSOR_SPECIAL_CASE = True and dropout disabled.
2. Run the nominal asynchronous case at 10 / 12.5 Hz.
3. Enable CROSS_CONTAMINATION_RATE_STRESS_TEST for the dedicated asymmetric-rate
   experiment in which disturbed Sensor 1 is much faster than nominal Sensor 2.
4. Compare isolated vs common-prediction normalized disbelief during S1-only
   faults.
5. Inspect matched conditional opinions C_{s|empty}^{sim} vs C_{s|j}.
6. Inspect direct batch C_F together with the active sensor-set timeline.
7. Enable dropout to inspect A_s, vacuous consistency updates, trust discount,
   and the experimental track-output trust construction.
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
from scipy.stats import chi2, norm
from math import ceil

import subjective_logic as sl

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.models.transition.nonlinear import ConstantTurn
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
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

# Optional original Griebel/Stone-Soup backend.  This import is available in the
# uulm-mrm/aduulm-stonesoup selfassessment_extensions branch.  The fallback is
# deliberate: V3 must remain executable if the reference implementation is not
# installed yet.
try:
    from stonesoup.selfassessor.kalman_selfassessor import KalmanSelfAssessor
    GRIEBEL_NATIVE_IMPORT_AVAILABLE = True
    GRIEBEL_NATIVE_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - optional research dependency
    KalmanSelfAssessor = None
    GRIEBEL_NATIVE_IMPORT_AVAILABLE = False
    GRIEBEL_NATIVE_IMPORT_ERROR = repr(exc)


# =============================================================================
# User switches
# =============================================================================

# False: asynchronous/multi-rate case.
# True: limiting case; both sensors are sampled on the 10 Hz grid.
SYNCHRONOUS_SENSOR_SPECIAL_CASE = True

# Optional dedicated stress test for cross-source prior contamination.
# If enabled (and SYNCHRONOUS_SENSOR_SPECIAL_CASE is False), the disturbed
# Sensor 1 is intentionally much faster than nominal Sensor 2, so several
# faulty S1 updates can affect the central prior before the next S2 update.
CROSS_CONTAMINATION_RATE_STRESS_TEST = False

NOMINAL_SENSOR_1_RATE_HZ = 10.0
NOMINAL_SENSOR_2_RATE_HZ = 12.5
CROSS_CONTAMINATION_SENSOR_1_RATE_HZ = 25.0
CROSS_CONTAMINATION_SENSOR_2_RATE_HZ = 10.0
SYNCHRONOUS_REFERENCE_RATE_HZ = 10.0

if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
    SENSOR_1_RATE_HZ = SYNCHRONOUS_REFERENCE_RATE_HZ
    SENSOR_2_RATE_HZ = SYNCHRONOUS_REFERENCE_RATE_HZ
elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
    SENSOR_1_RATE_HZ = CROSS_CONTAMINATION_SENSOR_1_RATE_HZ
    SENSOR_2_RATE_HZ = CROSS_CONTAMINATION_SENSOR_2_RATE_HZ
else:
    SENSOR_1_RATE_HZ = NOMINAL_SENSOR_1_RATE_HZ
    SENSOR_2_RATE_HZ = NOMINAL_SENSOR_2_RATE_HZ

# For a pure synchronous validation set this to False.
ENABLE_SENSOR_2_DROPOUT = False

SHOW_DYNAMIC_ANIMATION = True
SHOW_MATPLOTLIB_PLOTS = True
SHOW_PROJECTED_PROBABILITY = True  # PP is secondary; d/u are primary plots.
SHOW_POSITION_ERROR = True

# Dynamic Plotly dashboard; the slider uses integer union-event steps.
ANIMATION_FRAME_STRIDE = 5
ANIMATION_MAX_FRAMES = 400
ANIMATION_FRAME_DURATION_MS = 70

# Fixed tail during playback. This keeps each frame approximately constant in
# complexity so the animation does not become slower towards the end.
ANIMATION_TAIL_STEPS = 70

# The final frame is a separate overview: show the complete truth/track route
# once and automatically fit the axes around it.

ANIMATION_FINAL_SHOW_FULL_ROUTE = True
ANIMATION_FINAL_ROUTE_MARGIN = 0.06

ANIMATION_HALF_WIDTH_M = 18.0
ANIMATION_HALF_HEIGHT_M = 18.0

# Match Stone Soup's standard 2D uncertainty ellipse:
# width = 2*sqrt(lambda_max), height = 2*sqrt(lambda_min).
# This is the Mahalanobis-distance-1 covariance contour, not a 95% ellipse.
ANIMATION_COVARIANCE_SCALE = 1.0

ANIMATION_AUTO_OPEN_BROWSER = True
ANIMATION_HTML_FILENAME = "multi_sensor_selfassessment_animation.html"

ACTIVATE_DISTURBANCES = True
DISTURB_SENSOR_1 = True
DISTURB_SENSOR_2 = False

# Filter/assessment motion model.
# False: 4D CV state [x, vx, y, vy]
# True : 5D CT state [x, vx, y, vy, omega] using ConstantTurn and a UKF predictor.
# The ground-truth generator stays 4D, as in the old V7 script.
USE_CT_MODEL = False
CT_TURN_RATE_NOISE = 0.01

# Keep the coordinated turn in the ground truth independently of the other
# fault switches. For CV this creates the intended model mismatch. For CT it
# is a maneuver that the chosen filter model can represent.
ENABLE_GROUND_TRUTH_TURN = True

# Optional original Griebel reference.  "auto" uses KalmanSelfAssessor when
# available and otherwise falls back to a non-crashing placeholder.
ENABLE_GRIEBEL_REFERENCE = True
GRIEBEL_BACKEND = "auto"  # "auto" | "native" | "placeholder"

# In asynchronous mode the published 2023 method is not defined.  This switch
# selects an explicitly labelled straightforward extension for comparison:
# - "hold_last": update the currently active common-prior sensor SA and hold
#   the latest decision of all inactive sensors; update overall PP each union
#   event.  This intentionally exposes stale-opinion and unequal-rate effects.
# - "synchronous_only": update the Griebel reference only when all sensors have
#   a measurement at the same timestamp.
GRIEBEL_ASYNC_EXTENSION_MODE = "hold_last"

# Experimental output-level construction. Sensor support is derived ONLY from
# availability, while C_s remains a local diagnostic. This avoids feeding the same
# consistency information into track trust twice (through C_s and C_F).
#
# ANY: at least one available sensor is sufficient.
# ALL: every configured sensor is required; therefore any dropout reduces support.
TRACK_SENSOR_SUPPORT_RULE = "ANY"  # "ANY" or "ALL"

SCENARIO_DURATION_S = 160.0
TRUTH_RATE_HZ = 100.0  # exactly represents 5, 10, 12.5, 20 and 25 Hz
TRUTH_DT_S = 1.0 / TRUTH_RATE_HZ

Q_X = 0.25
Q_Y = 0.25
MEASUREMENT_VARIANCE_SENSOR_1 = 0.1
MEASUREMENT_VARIANCE_SENSOR_2 = 0.1

NUM_PIT_BINS = 7

# -----------------------------------------------------------------------------
# Time-normalised TEF settings
# -----------------------------------------------------------------------------
# Local sensor-consistency channels assess the PIT distribution over a
# short physical history.  The number of TEF samples is derived from the
# actual sensor rate.
CONSISTENCY_SHORT_TERM_HORIZON_S = 3.5

# The central active-batch filter-consistency statement is intentionally more
# reactive than the local diagnostic channels.
BATCH_SHORT_TERM_HORIZON_S = 2.0

# Availability assesses a different, faster process.
AVAILABILITY_SHORT_TERM_HORIZON_S = 1.0

# Conditional/update-order opinions are only updated at simultaneous sensor
# timestamps. They therefore need a longer physical horizon to accumulate
# enough PIT observations.
CONDITIONAL_SHORT_TERM_HORIZON_S = 5.0
DISAGREEMENT_SHORT_TERM_HORIZON_S = 5.0

REFERENCE_RATE_HZ = 10.0
# Match the first PIT paper's nominal long-term discount at the reference rate.
CONSISTENCY_REFERENCE_LONG_TERM_DISCOUNT = 0.99
AVAILABILITY_REFERENCE_LONG_TERM_DISCOUNT = 0.99
ALPHA_THRESHOLD_DC = 0.01

# The new TEF implementation supports several SL fusion operators.  They are
# configured explicitly per assessment family.  CBF remains the default for
# temporal PIT samples because, under the nominal KF assumptions, successive
# innovations/PIT observations are modeled as temporally independent evidence.
LOCAL_CONSISTENCY_TEF_FUSION = sl.FusionType.CUMULATIVE
BATCH_TEF_FUSION = sl.FusionType.CUMULATIVE
CONDITIONAL_TEF_FUSION = sl.FusionType.CUMULATIVE
DISAGREEMENT_TEF_FUSION = sl.FusionType.CUMULATIVE
AVAILABILITY_TEF_FUSION = sl.FusionType.CUMULATIVE
# These two flags activate the newer TEF conflict/reset behaviour available in
# the subjective_logic implementation used by the project.
HANDLE_SHORT_TERM_CONFLICT = True
AVERAGE_DC_CONFLICT_HANDLING = True

# Griebel 2023 reference settings.  The paper uses W=7 and n_ST=35.  The
# reference deliberately keeps sample-count parametrisation; it is not time
# normalised in the asynchronous extension.
GRIEBEL_NUM_X = 7
GRIEBEL_N_ST = 35
GRIEBEL_N_C = 1
GRIEBEL_ALPHA_THRESHOLD_DC = 0.1
GRIEBEL_TRUST_DISCOUNT = 0.99
GRIEBEL_OVERALL_WINDOW = 35

ASSUME_DETECTION_PROBABILITY_ONE = True
NOMINAL_BURN_IN_S = CONSISTENCY_SHORT_TERM_HORIZON_S
SCRIPT_DIR = Path(__file__).resolve().parent

RANDOM_SEED_TRUTH = 0
RANDOM_SEED_SENSOR_1 = 1
RANDOM_SEED_SENSOR_2 = 2


# =============================================================================
# Disturbance scenario from the first paper, expressed in seconds
# =============================================================================

OUTLIER_INTERVAL_S = (10.0, 20.0)

# Replaces the previous "increased x measurement noise" disturbance.
# The bias is applied only to Sensor 1 and only to the x measurement component.
SENSOR_1_BIAS_INTERVAL_S = (30.0, 40.0)
SENSOR_1_BIAS_VECTOR_M = np.array([2.0, 0.0], dtype=float)

INCREASED_MEAS_XY_INTERVAL_S = (50.0, 60.0)

# Moderate but clearly interpretable sensor degradation:
# multiply the TRUE measurement-noise covariance by 4. This doubles the
# per-component standard deviation while the filter continues assuming the
# nominal covariance.
MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR = 4.0

# Explicit 10 s nominal gap before and after the dropout:
#   previous disturbance ends at 60 s,
#   dropout starts at 70 s and ends at 80 s,
#   next disturbance starts at 90 s.
SENSOR_2_DROPOUT_INTERVAL_S = (70.0, 80.0)

# Shape-only measurement-noise disturbance for Sensor 1:
# mean and covariance remain exactly the nominal model assumptions,
# E[v] = 0 and Cov[v] = R, but the samples are uniform instead of Gaussian.
# This separates a violated distributional assumption from a wrong R matrix.
VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S = (90.0, 100.0)
TURN_INTERVAL_S = (110.0, 120.0)

# Final process-noise mismatch now affects BOTH x and y process components.
INCREASED_PROCESS_XY_INTERVAL_S = (130.0, 150.0)

DISTURBANCE_INTERVALS = [
    (*OUTLIER_INTERVAL_S, "S1 outliers", "tab:red"),
    (*SENSOR_1_BIAS_INTERVAL_S, f"S1 + {int(SENSOR_1_BIAS_VECTOR_M[0])}m x-bias", "tab:orange"),
    (
        *INCREASED_MEAS_XY_INTERVAL_S,
        f"S1 increased x/y-noise (R x{MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR:g})",
        "tab:blue",
    ),
    (*SENSOR_2_DROPOUT_INTERVAL_S, "S2 unavailable", "tab:gray"),
    (*VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S, "S1 variance-matched non-Gaussian noise", "tab:purple"),
    (*TURN_INTERVAL_S, "common motion-model mismatch", "tab:green"),
    (*INCREASED_PROCESS_XY_INTERVAL_S, "common increased x/y process noise", "tab:brown"),
]


def seconds_to_truth_step(seconds: float) -> int:
    return int(round(seconds * TRUTH_RATE_HZ))


def seconds_to_reference_step(seconds: float) -> int:
    return int(round(seconds * REFERENCE_RATE_HZ))


def seconds_to_sensor_step(seconds: float, rate_hz: float) -> int:
    """Convert physical time to the discrete index of a specific sensor stream."""
    return round_half_up(seconds * rate_hz)


def round_half_up(value: float) -> int:
    """Deterministic half-up rounding for sample counts (62.5 -> 63)."""
    return int(np.floor(float(value) + 0.5))


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
            return np.linalg.cholesky(
                matrix + jitter * np.eye(matrix.shape[0])
            )
        except np.linalg.LinAlgError:
            jitter *= 10.0

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return np.linalg.cholesky(
        eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    )


def sample_variance_matched_uniform_noise_from_cov(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample zero-mean non-Gaussian noise with exactly the target covariance.

    Let u_i ~ U(-sqrt(3), sqrt(3)). Then E[u] = 0 and Cov[u] = I.
    With L L^T = covariance, v = L u therefore satisfies
    E[v] = 0 and Cov[v] = covariance while remaining non-Gaussian.
    """
    covariance = np.asarray(covariance, dtype=float)
    chol = robust_cholesky(covariance)
    unit_variance_uniform = rng.uniform(
        -np.sqrt(3.0),
        np.sqrt(3.0),
        size=(covariance.shape[0], 1),
    )
    return chol @ unit_variance_uniform


# =============================================================================
# Subjective-logic helpers and PIT mapping
# =============================================================================


def scalar_pit_to_opinion(pit_value: float, num_bins: int):
    pit_value = float(np.clip(pit_value, 0.0, 1.0))
    evidence = np.zeros(num_bins, dtype=float)
    index = min(int(np.floor(pit_value * num_bins)), num_bins - 1)
    evidence[index] = 1.0
    distribution = eval(
        f"sl.DirichletDistribution{num_bins}d"
    ).from_evidences(evidence)
    return distribution.as_opinion()


def vacuous_multinomial_opinion(domain_size: int):
    """Return a vacuous multinomial opinion with zero evidence.

    This is used as a scheduled TEF tick when an assessment channel cannot
    provide an observation (e.g. because the expected sensor measurement is
    missing). It expresses *absence of consistency evidence*, not inconsistency.
    """
    evidence = np.zeros(domain_size, dtype=float)
    distribution = eval(
        f"sl.DirichletDistribution{domain_size}d"
    ).from_evidences(evidence)
    return distribution.as_opinion()


def multinomial_to_binomial_consistency_opinion(
    opinion,
    num_bins: int,
    prior_ok_value: float = 0.5,
    eps: float = 1e-12,
):
    """Map PIT-bin distribution deviation from uniformity to (b,d,u,a)."""
    uncertainty = float(opinion.uncertainty())
    committed_mass = 1.0 - uncertainty

    if committed_mass <= eps:
        result = sl.Opinion2d(0.0, 0.0)
        result.prior_belief_masses = [prior_ok_value, 1.0 - prior_ok_value]
        return result

    uniform_reference = np.full(num_bins, 1.0 / num_bins)
    committed_distribution = (
        np.asarray(opinion.belief_masses, dtype=float) / committed_mass
    )
    total_variation = 0.5 * np.sum(
        np.abs(committed_distribution - uniform_reference)
    )
    maximum_total_variation = 1.0 - 1.0 / num_bins

    disbelief = committed_mass * total_variation / maximum_total_variation
    disbelief = float(np.clip(disbelief, 0.0, committed_mass))
    belief = committed_mass - disbelief

    result = sl.Opinion2d(belief, disbelief)
    result.prior_belief_masses = [prior_ok_value, 1.0 - prior_ok_value]
    return result


def calculate_long_term_evidence(num_bins: int, discount: float) -> int:
    return int(
        (-(num_bins - 1) + np.sqrt(
            (num_bins - 1) ** 2
            + 4.0 * num_bins / (1.0 - discount + 1e-12)
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
    physical_horizon_s: float,
    reference_discount: float,
) -> tuple[int, float, float]:
    """Derive n_ST, per-sample discount and reset threshold.

    The semantic design parameter is physical time.  TEFs that assess the same
    quantity use the same ``physical_horizon_s`` even when their input rates
    differ.  The per-sample discount is transformed so that one second of
    fading has the same effect at every rate.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    n_st = max(2, round_half_up(physical_horizon_s * rate_hz))
    discount = reference_discount ** (REFERENCE_RATE_HZ / rate_hz)

    equivalent_evidence = calculate_long_term_evidence(domain_size, discount)
    key = f"{domain_size}, {equivalent_evidence}, {ALPHA_THRESHOLD_DC}"
    thresholds = load_opinion_thresholds()
    threshold = float(
        thresholds.get(
            key,
            calc_threshold_n_diff(
                domain_size,
                equivalent_evidence,
                ALPHA_THRESHOLD_DC,
            ),
        )
    )
    return n_st, float(discount), threshold


def create_temporal_memory(
    rate_hz: float,
    domain_size: int,
    physical_horizon_s: float,
    reference_discount: float,
    fusion_type,
):
    n_st, discount, threshold = rate_normalised_temporal_parameters(
        rate_hz,
        domain_size,
        physical_horizon_s,
        reference_discount,
    )
    memory = eval(f"sl.LongShortTermMemory{domain_size}d")(
        n_st,
        threshold,
        discount,
        fusion_type,
        HANDLE_SHORT_TERM_CONFLICT,
        AVERAGE_DC_CONFLICT_HANDLING,
    )
    return memory, n_st, discount, threshold


def vacuous_binomial(prior_ok_value: float = 0.5):
    result = sl.Opinion2d(0.0, 0.0)
    result.prior_belief_masses = [prior_ok_value, 1.0 - prior_ok_value]
    return result


def p_ok(opinion) -> float:
    return float(opinion.getProjection()[0])


def belief(opinion) -> float:
    return float(opinion.belief())


def disbelief(opinion) -> float:
    return float(opinion.disbelief())


def uncertainty(opinion) -> float:
    return float(opinion.uncertainty())


def normalized_disbelief_from_components(
    disbelief_value: float,
    uncertainty_value: float,
    eps: float = 1e-12,
) -> float:
    """Return d_norm = d / (1-u), i.e. inconsistency within committed mass.

    For a vacuous opinion (u ~= 1), d_norm is undefined because no committed
    evidence exists.  NaN is returned deliberately so plots do not suggest
    nominal consistency merely because evidence is absent.
    """
    committed_mass = 1.0 - float(uncertainty_value)
    if committed_mass <= eps:
        return float("nan")
    return float(np.clip(float(disbelief_value) / committed_mass, 0.0, 1.0))


def normalized_disbelief(opinion) -> float:
    return normalized_disbelief_from_components(
        disbelief(opinion), uncertainty(opinion)
    )


def normalized_disbelief_series(
    disbelief_values: Iterable[float],
    uncertainty_values: Iterable[float],
) -> list[float]:
    return [
        normalized_disbelief_from_components(d, u)
        for d, u in zip(disbelief_values, uncertainty_values)
    ]


def prior_ok(opinion) -> float:
    try:
        return float(opinion.prior_belief_masses[0])
    except (AttributeError, IndexError, TypeError):
        return 0.5


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


def negate_binomial(opinion):
    result = sl.Opinion2d(disbelief(opinion), belief(opinion))
    base_rate = 1.0 - prior_ok(opinion)
    result.prior_belief_masses = [base_rate, 1.0 - base_rate]
    return result


def conjunction_all(opinions: Iterable):
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    result = deepcopy(opinions[0])
    for opinion in opinions[1:]:
        result = result.multiply(opinion)
    return result


def disjunction_all(opinions: Iterable):
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    return negate_binomial(
        conjunction_all([negate_binomial(op) for op in opinions])
    )


def trust_discount(trust_opinion, observation_opinion):
    """Reliability trust discount using projected trust probability.

    The observation opinion itself is not modified; this function returns a new
    downstream interpretation.  With p_T = P(trust):

        b' = p_T b
        d' = p_T d
        u' = 1 - p_T (1-u)

    The observation base rate is preserved.
    """
    p_trust = float(np.clip(p_ok(trust_opinion), 0.0, 1.0))
    new_belief = p_trust * belief(observation_opinion)
    new_disbelief = p_trust * disbelief(observation_opinion)
    result = sl.Opinion2d(new_belief, new_disbelief)
    a = prior_ok(observation_opinion)
    result.prior_belief_masses = [a, 1.0 - a]
    return result


# =============================================================================
# Assessment states
# =============================================================================


@dataclass
class SensorConsistencyState:
    """PIT/TEF consistency assessment for one sensor/filter context."""

    sensor_id: int
    label: str
    input_rate_hz: float
    colour: str
    context_label: str
    physical_horizon_s: float = CONSISTENCY_SHORT_TERM_HORIZON_S
    fusion_type: object = LOCAL_CONSISTENCY_TEF_FUSION

    tef_radial: object = field(init=False)
    tef_x: object = field(init=False)
    tef_y: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)
    latest_opinion: object = field(init=False)
    latest_timestamp: datetime | None = None

    event_times_s: list[float] = field(default_factory=list)
    radial_pit_events: list[float] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    belief_events: list[float] = field(default_factory=list)
    disbelief_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)
    nis_events: list[float] = field(default_factory=list)
    nis_average: list[float] = field(default_factory=list)
    nis_lower: list[float] = field(default_factory=list)
    nis_upper: list[float] = field(default_factory=list)
    nis_window: deque = field(init=False)

    def __post_init__(self) -> None:
        kwargs = dict(
            rate_hz=self.input_rate_hz,
            domain_size=NUM_PIT_BINS,
            physical_horizon_s=self.physical_horizon_s,
            reference_discount=CONSISTENCY_REFERENCE_LONG_TERM_DISCOUNT,
            fusion_type=self.fusion_type,
        )
        self.tef_radial, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(**kwargs)
        )
        self.tef_x, _, _, _ = create_temporal_memory(**kwargs)
        self.tef_y, _, _, _ = create_temporal_memory(**kwargs)
        self.latest_opinion = vacuous_binomial()
        self.nis_window = deque(maxlen=self.n_st)

    def _refresh_latest_opinion(self) -> None:
        radial_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_radial.get_opinion(), NUM_PIT_BINS, prior_ok_value=0.5
        )
        # sqrt(0.5)^2 = 0.5 after logical multiplication of x and y.
        component_prior = float(np.sqrt(0.5))
        x_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_x.get_opinion(),
            NUM_PIT_BINS,
            prior_ok_value=component_prior,
        )
        y_binomial = multinomial_to_binomial_consistency_opinion(
            self.tef_y.get_opinion(),
            NUM_PIT_BINS,
            prior_ok_value=component_prior,
        )
        component_binomial = x_binomial.multiply(y_binomial)

        # Same concurrent radial/component construction as the first paper.
        self.latest_opinion = fuse_weighted(
            [radial_binomial, component_binomial]
        )

    def advance_without_measurement(
        self,
        timestamp: datetime,
        start_time: datetime,
    ):
        """Advance the TEF clock without adding consistency evidence.

        A missing expected measurement provides no innovation/PIT sample.
        Therefore a vacuous multinomial opinion is added to each temporal
        consistency channel. This does NOT count as evidence for consistency or
        inconsistency; it merely lets the temporal memory age on the known
        sensor schedule.
        """
        self.tef_radial.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self.tef_x.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self.tef_y.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self._refresh_latest_opinion()
        self.latest_timestamp = timestamp

        elapsed_s = (timestamp - start_time).total_seconds()
        self.event_times_s.append(elapsed_s)
        self.radial_pit_events.append(float("nan"))
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.belief_events.append(belief(self.latest_opinion))
        self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion))

        # There is no NIS sample at this scheduled instant. NaNs make the
        # diagnostic plot explicitly show that the statistic is unavailable.
        self.nis_events.append(float("nan"))
        self.nis_average.append(float("nan"))
        self.nis_lower.append(float("nan"))
        self.nis_upper.append(float("nan"))
        return self.latest_opinion

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

        self._refresh_latest_opinion()
        self.latest_timestamp = timestamp

        elapsed_s = (timestamp - start_time).total_seconds()
        self.event_times_s.append(elapsed_s)
        self.radial_pit_events.append(radial_pit)
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.belief_events.append(belief(self.latest_opinion))
        self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion))
        self.nis_events.append(float(nis_value))

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
class SensorDisagreementState:
    """Pairwise agreement opinion for two simultaneously measuring sensors.

    Belief means mutual statistical agreement; normalized disbelief therefore
    acts as a sensor-to-sensor disagreement score. The central filter prior is
    not used.
    """
    sensor_ids: tuple[int, int]
    input_rate_hz: float
    physical_horizon_s: float = DISAGREEMENT_SHORT_TERM_HORIZON_S
    fusion_type: object = DISAGREEMENT_TEF_FUSION
    tef_radial: object = field(init=False)
    tef_x: object = field(init=False)
    tef_y: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)
    latest_opinion: object = field(init=False)
    event_times_s: list[float] = field(default_factory=list)
    belief_events: list[float] = field(default_factory=list)
    disbelief_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    radial_nis_events: list[float] = field(default_factory=list)
    standardized_x_events: list[float] = field(default_factory=list)
    standardized_y_events: list[float] = field(default_factory=list)
    mean_standardized_x_events: list[float] = field(default_factory=list)
    mean_standardized_y_events: list[float] = field(default_factory=list)
    standardized_x_window: deque = field(init=False)
    standardized_y_window: deque = field(init=False)

    def __post_init__(self) -> None:
        kwargs = dict(rate_hz=self.input_rate_hz, domain_size=NUM_PIT_BINS,
                      physical_horizon_s=self.physical_horizon_s,
                      reference_discount=CONSISTENCY_REFERENCE_LONG_TERM_DISCOUNT,
                      fusion_type=self.fusion_type)
        self.tef_radial, self.n_st, self.discount, self.threshold = create_temporal_memory(**kwargs)
        self.tef_x, _, _, _ = create_temporal_memory(**kwargs)
        self.tef_y, _, _, _ = create_temporal_memory(**kwargs)
        self.latest_opinion = vacuous_binomial()
        self.standardized_x_window = deque(maxlen=self.n_st)
        self.standardized_y_window = deque(maxlen=self.n_st)

    def _refresh_latest_opinion(self) -> None:
        radial = multinomial_to_binomial_consistency_opinion(self.tef_radial.get_opinion(), NUM_PIT_BINS, 0.5)
        component_prior = float(np.sqrt(0.5))
        x_op = multinomial_to_binomial_consistency_opinion(self.tef_x.get_opinion(), NUM_PIT_BINS, component_prior)
        y_op = multinomial_to_binomial_consistency_opinion(self.tef_y.get_opinion(), NUM_PIT_BINS, component_prior)
        self.latest_opinion = fuse_weighted([radial, x_op.multiply(y_op)])

    def advance_without_pair(self, timestamp: datetime, start_time: datetime):
        self.tef_radial.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self.tef_x.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self.tef_y.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self._refresh_latest_opinion()
        self.event_times_s.append((timestamp-start_time).total_seconds())
        self.belief_events.append(belief(self.latest_opinion)); self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion)); self.p_ok_events.append(p_ok(self.latest_opinion))
        self.radial_nis_events.append(float('nan')); self.standardized_x_events.append(float('nan')); self.standardized_y_events.append(float('nan'))
        self.mean_standardized_x_events.append(float(np.mean(self.standardized_x_window)) if self.standardized_x_window else float('nan'))
        self.mean_standardized_y_events.append(float(np.mean(self.standardized_y_window)) if self.standardized_y_window else float('nan'))
        return self.latest_opinion

    def update(self, measurement_1: Detection, measurement_2: Detection, timestamp: datetime, start_time: datetime):
        z1=np.asarray(measurement_1.state_vector,dtype=float).reshape(-1,1); z2=np.asarray(measurement_2.state_vector,dtype=float).reshape(-1,1)
        h1=np.asarray(measurement_1.measurement_model.matrix(),dtype=float); h2=np.asarray(measurement_2.measurement_model.matrix(),dtype=float)
        if z1.shape != z2.shape or h1.shape != h2.shape or not np.allclose(h1,h2):
            raise ValueError('SensorDisagreementState requires equal measurement dimensions and the same H matrix.')
        r1=np.asarray(measurement_1.measurement_model.covar(),dtype=float); r2=np.asarray(measurement_2.measurement_model.covar(),dtype=float)
        delta_z=z1-z2; s_delta=r1+r2; dim=delta_z.shape[0]
        nis=(delta_z.T @ np.linalg.solve(s_delta,delta_z)).item(); pit_radial=float(chi2.cdf(nis,df=dim))
        standardized=np.linalg.solve(robust_cholesky(s_delta),delta_z).flatten(); pit_components=norm.cdf(standardized)
        self.tef_radial.add(scalar_pit_to_opinion(pit_radial,NUM_PIT_BINS)); self.tef_x.add(scalar_pit_to_opinion(float(pit_components[0]),NUM_PIT_BINS)); self.tef_y.add(scalar_pit_to_opinion(float(pit_components[1]),NUM_PIT_BINS))
        self._refresh_latest_opinion(); self.standardized_x_window.append(float(standardized[0])); self.standardized_y_window.append(float(standardized[1]))
        self.event_times_s.append((timestamp-start_time).total_seconds()); self.belief_events.append(belief(self.latest_opinion)); self.disbelief_events.append(disbelief(self.latest_opinion)); self.uncertainty_events.append(uncertainty(self.latest_opinion)); self.p_ok_events.append(p_ok(self.latest_opinion)); self.radial_nis_events.append(float(nis)); self.standardized_x_events.append(float(standardized[0])); self.standardized_y_events.append(float(standardized[1])); self.mean_standardized_x_events.append(float(np.mean(self.standardized_x_window))); self.mean_standardized_y_events.append(float(np.mean(self.standardized_y_window)))
        return self.latest_opinion


@dataclass
class AvailabilityAssessmentState:
    """Scheduled-output availability opinion A_s."""

    sensor_id: int
    label: str
    input_rate_hz: float
    colour: str

    tef: object = field(init=False)
    n_st: int = field(init=False)
    discount: float = field(init=False)
    threshold: float = field(init=False)
    latest_opinion: object = field(init=False)

    event_times_s: list[float] = field(default_factory=list)
    p_available_events: list[float] = field(default_factory=list)
    belief_events: list[float] = field(default_factory=list)
    disbelief_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tef, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(
                self.input_rate_hz,
                2,
                AVAILABILITY_SHORT_TERM_HORIZON_S,
                AVAILABILITY_REFERENCE_LONG_TERM_DISCOUNT,
                AVAILABILITY_TEF_FUSION,
            )
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
        self.belief_events.append(belief(self.latest_opinion))
        self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion))
        return self.latest_opinion


@dataclass
class IsolatedAssessmentFilter:
    """Sensor-only shadow KF used solely to obtain isolated innovations."""

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
    """Direct central-filter consistency C_F from the active measurement batch."""

    input_rate_hz: float
    physical_horizon_s: float = BATCH_SHORT_TERM_HORIZON_S
    fusion_type: object = BATCH_TEF_FUSION
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
    active_sensor_sets: list[str] = field(default_factory=list)
    p_ok_events: list[float] = field(default_factory=list)
    belief_events: list[float] = field(default_factory=list)
    disbelief_events: list[float] = field(default_factory=list)
    uncertainty_events: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tef_radial, self.n_st, self.discount, self.threshold = (
            create_temporal_memory(
                self.input_rate_hz,
                NUM_PIT_BINS,
                self.physical_horizon_s,
                CONSISTENCY_REFERENCE_LONG_TERM_DISCOUNT,
                self.fusion_type,
            )
        )
        self.latest_opinion = vacuous_binomial()

    def advance_without_measurement(
        self,
        timestamp: datetime,
        start_time: datetime,
    ):
        """Advance direct-batch TEF when a scheduled union event has no data."""
        self.tef_radial.add(vacuous_multinomial_opinion(NUM_PIT_BINS))
        self.latest_opinion = multinomial_to_binomial_consistency_opinion(
            self.tef_radial.get_opinion(),
            NUM_PIT_BINS,
            prior_ok_value=0.5,
        )

        self.event_times_s.append((timestamp - start_time).total_seconds())
        self.pit_events.append(float("nan"))
        self.nis_events.append(float("nan"))
        self.degrees_of_freedom.append(0)
        self.active_sensor_counts.append(0)
        self.active_sensor_sets.append("none")
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.belief_events.append(belief(self.latest_opinion))
        self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion))
        return self.latest_opinion

    def update(
        self,
        nis_value: float,
        degrees_of_freedom: int,
        active_sensor_ids: list[int],
        timestamp: datetime,
        start_time: datetime,
    ):
        pit_value = float(chi2.cdf(nis_value, df=degrees_of_freedom))
        self.tef_radial.add(
            scalar_pit_to_opinion(pit_value, NUM_PIT_BINS)
        )
        self.latest_opinion = multinomial_to_binomial_consistency_opinion(
            self.tef_radial.get_opinion(),
            NUM_PIT_BINS,
            prior_ok_value=0.5,
        )

        self.event_times_s.append((timestamp - start_time).total_seconds())
        self.pit_events.append(pit_value)
        self.nis_events.append(float(nis_value))
        self.degrees_of_freedom.append(int(degrees_of_freedom))
        active_sensor_ids = sorted(int(sensor_id) for sensor_id in active_sensor_ids)
        self.active_sensor_counts.append(len(active_sensor_ids))
        self.active_sensor_sets.append(
            "+".join(f"S{sensor_id}" for sensor_id in active_sensor_ids)
        )
        self.p_ok_events.append(p_ok(self.latest_opinion))
        self.belief_events.append(belief(self.latest_opinion))
        self.disbelief_events.append(disbelief(self.latest_opinion))
        self.uncertainty_events.append(uncertainty(self.latest_opinion))
        return self.latest_opinion


@dataclass
class GriebelMeasure:
    delta: float = np.nan
    uncertainty: float = np.nan
    eta: float = np.nan

    @property
    def valid(self) -> bool:
        return bool(
            np.isfinite(self.delta)
            and np.isfinite(self.eta)
        )

    @property
    def accepted(self) -> bool | None:
        if not self.valid:
            return None
        return bool(self.delta < self.eta)


class GriebelReferenceBackend:
    """Adapter around the optional aduulm-stonesoup KalmanSelfAssessor.

    This class deliberately uses only the public-ish interface already used in
    the user's first-paper script: ``assess`` and ``get_sas_measures``.  The
    overall PP reference follows the 2023 paper's threshold-decision / sliding-
    window construction.  No attempt is made to translate Griebel's internal
    multinomial opinions into the external ``subjective_logic`` package.
    """

    def __init__(self, sensor_ids: Iterable[int], dim_meas: int = 2):
        self.sensor_ids = sorted(sensor_ids)
        self.enabled = ENABLE_GRIEBEL_REFERENCE
        self.native = False
        self.status = "disabled"

        self.assessors: dict[int, object] = {}
        self.latest_measure = {
            sensor_id: GriebelMeasure() for sensor_id in self.sensor_ids
        }
        self.sensor_times = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_delta = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_eta = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_uncertainty = {
            sensor_id: [] for sensor_id in self.sensor_ids
        }

        self.overall_window: deque = deque(maxlen=GRIEBEL_OVERALL_WINDOW)
        self.overall_times_s: list[float] = []
        self.overall_p_ok: list[float] = []
        self.overall_uncertainty: list[float] = []
        self.overall_true_fraction: list[float] = []

        if not self.enabled:
            return

        requested = GRIEBEL_BACKEND.lower()
        allow_native = requested in {"auto", "native"}
        if allow_native and GRIEBEL_NATIVE_IMPORT_AVAILABLE:
            try:
                for sensor_id in self.sensor_ids:
                    self.assessors[sensor_id] = KalmanSelfAssessor(
                        num_X=GRIEBEL_NUM_X,
                        n_st=GRIEBEL_N_ST,
                        n_c=GRIEBEL_N_C,
                        dim_meas=dim_meas,
                        alpha_threshold_dc=GRIEBEL_ALPHA_THRESHOLD_DC,
                        trust_discount=GRIEBEL_TRUST_DISCOUNT,
                    )
                self.native = True
                self.status = "native KalmanSelfAssessor"
                return
            except Exception as exc:  # noqa: BLE001
                self.status = f"native init failed -> placeholder: {exc!r}"
                if requested == "native":
                    print("WARNING:", self.status)

        self.native = False
        if requested == "placeholder":
            self.status = "placeholder requested"
        elif not GRIEBEL_NATIVE_IMPORT_AVAILABLE:
            self.status = (
                "placeholder; KalmanSelfAssessor import unavailable: "
                f"{GRIEBEL_NATIVE_IMPORT_ERROR}"
            )
        elif self.status == "disabled":
            self.status = "placeholder"

    def should_update_event(
        self,
        active_sensor_ids: list[int],
    ) -> bool:
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
            return True
        if GRIEBEL_ASYNC_EXTENSION_MODE == "hold_last":
            return True
        if GRIEBEL_ASYNC_EXTENSION_MODE == "synchronous_only":
            return set(active_sensor_ids) == set(self.sensor_ids)
        raise ValueError(
            "GRIEBEL_ASYNC_EXTENSION_MODE must be 'hold_last' or "
            "'synchronous_only'"
        )

    def assess_sensor(
        self,
        sensor_id: int,
        measurement: Detection,
        measurement_prediction,
        timestamp: datetime,
        start_time: datetime,
    ) -> GriebelMeasure:
        if not self.native:
            return self.latest_measure[sensor_id]

        assessor = self.assessors[sensor_id]
        assessor.assess(
            measurement_prediction.mean,
            measurement_prediction.covar,
            measurement.state_vector,
        )
        values = np.asarray(assessor.get_sas_measures(), dtype=float).reshape(-1)
        if values.size < 3:
            measure = GriebelMeasure()
        else:
            measure = GriebelMeasure(
                delta=float(values[0]),
                uncertainty=float(values[1]),
                eta=float(values[2]),
            )
        self.latest_measure[sensor_id] = measure
        self.sensor_times[sensor_id].append(
            (timestamp - start_time).total_seconds()
        )
        self.sensor_delta[sensor_id].append(measure.delta)
        self.sensor_eta[sensor_id].append(measure.eta)
        self.sensor_uncertainty[sensor_id].append(measure.uncertainty)
        return measure

    def finish_event(
        self,
        timestamp: datetime,
        start_time: datetime,
        active_sensor_ids: list[int],
    ) -> None:
        if not self.native or not self.should_update_event(active_sensor_ids):
            return

        # The synchronous publication always has all current decisions.  The
        # explicit async hold-last extension uses the most recent decision for
        # inactive sensors once all assessors have been initialised.
        decisions = []
        for sensor_id in self.sensor_ids:
            accepted = self.latest_measure[sensor_id].accepted
            if accepted is None:
                return
            decisions.append(float(accepted))

        true_fraction = float(np.mean(decisions))
        evidence = np.array([true_fraction, 1.0 - true_fraction], dtype=float)
        step_opinion = sl.DirichletDistribution2d.from_evidences(
            evidence
        ).as_opinion()
        self.overall_window.append(step_opinion)
        if len(self.overall_window) == 1:
            overall = deepcopy(self.overall_window[0])
        else:
            overall = sl.Fusion.fuse_opinions(
                sl.FusionType.CUMULATIVE,
                list(self.overall_window),
            )

        self.overall_times_s.append((timestamp - start_time).total_seconds())
        self.overall_p_ok.append(p_ok(overall))
        self.overall_uncertainty.append(uncertainty(overall))
        self.overall_true_fraction.append(true_fraction)


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
    nominal_simultaneous_event_rate_hz: float


def build_sensor_definitions() -> dict[int, SensorDefinition]:
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
            SENSOR_2_RATE_HZ,
            SENSOR_2_RATE_HZ,
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
        "disturb_noise_coeff": [True, True],
        "disturbance_mode": ["jump"],
        "parameters": [[
            [
                seconds_to_truth_step(INCREASED_PROCESS_XY_INTERVAL_S[0]),
                disturbance_factor,
            ],
            [
                seconds_to_truth_step(INCREASED_PROCESS_XY_INTERVAL_S[1]),
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
            right_turn_model
            if ENABLE_GROUND_TRUTH_TURN and turn_start <= step < turn_end
            else cv_model
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
        ndim_state=5 if USE_CT_MODEL else 4,
        mapping=(0, 2),
        noise_covar=np.eye(2) * definition.variance,
    )

    # The third disturbance uses its own physically interpretable covariance
    # factor. The helper is stateful, so the inverse factor restores the
    # nominal covariance at the end of the interval.
    measurement_noise_factor = (
        MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR
        if ACTIVATE_DISTURBANCES
        else 1.0
    )

    # disturbance_measurement_noise() was originally called once per sample
    # with a strictly increasing sample index k.  Therefore all disturbance
    # boundaries are converted to the ACTUAL sensor time base.  Passing a
    # rounded 10-Hz reference index to a 20-Hz sensor repeats the same k two or
    # three times and can apply stateful jump/outlier logic repeatedly.
    sensor_step = lambda seconds: seconds_to_sensor_step(  # noqa: E731
        seconds, definition.actual_rate_hz
    )
    measurement_configs = {
        "disturbance_mode": ["jump", "outliers"],
        "parameters": [
            [
                [
                    sensor_step(INCREASED_MEAS_XY_INTERVAL_S[0]),
                    measurement_noise_factor,
                    [1, 1],
                ],
                [
                    sensor_step(INCREASED_MEAS_XY_INTERVAL_S[1]),
                    1.0 / measurement_noise_factor,
                    [1, 1],
                ],
            ],
            [[
                sensor_step(OUTLIER_INTERVAL_S[0]),
                sensor_step(OUTLIER_INTERVAL_S[1]),
                10,
                8,
            ]],
        ],
    }

    schedule: list[ScheduledSensorEvent] = []
    detections: list[Detection] = []
    sensor_indices = sensor_truth_indices(definition.actual_rate_hz)
    for measurement_index, truth_index in enumerate(sensor_indices):
        state = truth[truth_index]
        elapsed_s = truth_index / TRUTH_RATE_HZ

        # Advance the disturbance model on every SCHEDULED sensor tick.  This is
        # done before dropout handling so that a temporary missing measurement
        # does not freeze the disturbance model's internal discrete-time state.
        if definition.disturb_measurements and ACTIVATE_DISTURBANCES:
            true_model = disturbance_measurement_noise(
                true_model,
                measurement_configs,
                measurement_index,
            )

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
                    definition.sensor_id,
                    state.timestamp,
                    None,
                )
            )
            continue

        if (
            definition.disturb_measurements
            and ACTIVATE_DISTURBANCES
            and VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[0]
            <= elapsed_s
            < VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[1]
        ):
            # Deliberately violate ONLY the Gaussian shape assumption:
            # the filter's assumed R remains correct in mean/covariance.
            measurement_vector = true_model.function(state, noise=False)
            measurement_vector += sample_variance_matched_uniform_noise_from_cov(
                np.asarray(true_model.noise_covar, dtype=float), rng
            )
        else:
            measurement_vector = true_model.function(state, noise=True)

        if (
            definition.sensor_id == 1
            and definition.disturb_measurements
            and ACTIVATE_DISTURBANCES
            and SENSOR_1_BIAS_INTERVAL_S[0]
            <= elapsed_s
            < SENSOR_1_BIAS_INTERVAL_S[1]
        ):
            measurement_vector = StateVector(
                np.asarray(measurement_vector, dtype=float).reshape(-1, 1)
                + SENSOR_1_BIAS_VECTOR_M.reshape(-1, 1)
            )

        detection = Detection(
            measurement_vector,
            timestamp=state.timestamp,
            measurement_model=filter_model,
        )
        detections.append(detection)
        schedule.append(
            ScheduledSensorEvent(
                definition.sensor_id,
                state.timestamp,
                detection,
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

    # Unique union-event rate; this drives the direct batch TEF.
    nominal_batch_event_rate_hz = max(1, len(events) - 1) / SCENARIO_DURATION_S

    # Scheduled simultaneous timestamps, independent of dropout.  This rate
    # drives the conditional/order TEFs and therefore reflects how frequently
    # that diagnostic information is physically available.
    simultaneous_count = sum(
        len(batch) == len(sensor_definitions)
        for batch in events.values()
    )
    nominal_simultaneous_event_rate_hz = max(
        1.0 / SCENARIO_DURATION_S,
        max(1, simultaneous_count - 1) / SCENARIO_DURATION_S,
    )

    return ScenarioData(
        start_time=start_time,
        truth=truth,
        sensor_definitions=sensor_definitions,
        measurements_by_sensor=measurements_by_sensor,
        events_by_timestamp=dict(events),
        nominal_batch_event_rate_hz=nominal_batch_event_rate_hz,
        nominal_simultaneous_event_rate_hz=nominal_simultaneous_event_rate_hz,
    )


# =============================================================================
# Event-based filtering and assessment
# =============================================================================


def make_transition_model():
    if USE_CT_MODEL:
        return ConstantTurn([Q_X, Q_Y], CT_TURN_RATE_NOISE)
    return CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(Q_X), ConstantVelocity(Q_Y)]
    )


def make_predictor():
    if USE_CT_MODEL:
        return UnscentedKalmanPredictor(make_transition_model())
    return KalmanPredictor(make_transition_model())


def initial_filter_state(start_time: datetime) -> GaussianState:
    if USE_CT_MODEL:
        return GaussianState(
            StateVector([[0.0], [5.0], [0.0], [5.0], [0.0]]),
            CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0, 0.1])),
            timestamp=start_time,
        )
    return GaussianState(
        StateVector([[0.0], [5.0], [0.0], [5.0]]),
        CovarianceMatrix(np.diag([0.5, 1.0, 0.5, 1.0])),
        timestamp=start_time,
    )


def predict_measurement_from_state(
    state,
    measurement: Detection,
):
    updater = KalmanUpdater(measurement.measurement_model)
    prediction = updater.predict_measurement(
        state,
        measurement_model=measurement.measurement_model,
    )
    return updater, prediction


def apply_measurement_update(state, measurement: Detection):
    updater, measurement_prediction = predict_measurement_from_state(
        state, measurement
    )
    hypothesis = SingleHypothesis(
        state,
        measurement,
        measurement_prediction,
    )
    return updater.update(hypothesis), measurement_prediction


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
    active_batch_sizes: list[int]

    isolated_states: dict[int, SensorConsistencyState]
    common_prediction_states: dict[int, SensorConsistencyState]
    availability_states: dict[int, AvailabilityAssessmentState]
    conditional_states: dict[tuple[int, int], SensorConsistencyState]
    conditional_reference_states: dict[int, SensorConsistencyState]
    disagreement_state: SensorDisagreementState | None
    conditional_influence_times_s: list[float]
    conditional_influence_scores: dict[tuple[int, int], list[float]]
    conditional_order_asymmetry: list[float]
    batch_track_state: BatchTrackAssessmentState
    griebel: GriebelReferenceBackend

    trusted_sensor_history: dict[int, list]
    operational_sensor_history: dict[int, list]
    sensor_support_history: list
    track_output_trust_history: list
    common_abf_history: list
    batch_history: list
    position_error: list[float]
    order_decomposition_error_12: list[float]
    order_decomposition_error_21: list[float]


def process_scenario(scenario: ScenarioData) -> ProcessingResult:
    sensor_ids = sorted(scenario.sensor_definitions)

    central_predictor = make_predictor()
    central_prior = initial_filter_state(scenario.start_time)

    isolated_states: dict[int, SensorConsistencyState] = {}
    common_states: dict[int, SensorConsistencyState] = {}
    availability_states: dict[int, AvailabilityAssessmentState] = {}
    isolated_filters: dict[int, IsolatedAssessmentFilter] = {}

    for sensor_id, definition in scenario.sensor_definitions.items():
        isolated_states[sensor_id] = SensorConsistencyState(
            sensor_id=sensor_id,
            label=definition.label,
            input_rate_hz=definition.actual_rate_hz,
            colour=definition.colour,
            context_label="isolated sensor-only KF",
            fusion_type=LOCAL_CONSISTENCY_TEF_FUSION,
        )
        common_states[sensor_id] = SensorConsistencyState(
            sensor_id=sensor_id,
            label=definition.label,
            input_rate_hz=definition.actual_rate_hz,
            colour=definition.colour,
            context_label="common central prediction",
            fusion_type=LOCAL_CONSISTENCY_TEF_FUSION,
        )
        availability_states[sensor_id] = AvailabilityAssessmentState(
            sensor_id=sensor_id,
            label=definition.label,
            input_rate_hz=definition.actual_rate_hz,
            colour=definition.colour,
        )
        isolated_filters[sensor_id] = IsolatedAssessmentFilter(
            sensor_id=sensor_id,
            assessment=isolated_states[sensor_id],
            prior=initial_filter_state(scenario.start_time),
            predictor=make_predictor(),
        )

    batch_track_state = BatchTrackAssessmentState(
        input_rate_hz=scenario.nominal_batch_event_rate_hz
    )

    # Ordered pair (target, conditioning sensor): C_{target | conditioning}.
    # A matched reference C_{target | emptyset}^{sim} runs on EXACTLY the same
    # simultaneous timestamps and with exactly the same TEF parametrisation.
    # Therefore the only difference between both channels is the virtual update
    # with the conditioning sensor.
    conditional_states: dict[tuple[int, int], SensorConsistencyState] = {}
    conditional_reference_states: dict[int, SensorConsistencyState] = {}
    if len(sensor_ids) == 2:
        for target in sensor_ids:
            conditioning = sensor_ids[1] if target == sensor_ids[0] else sensor_ids[0]
            definition = scenario.sensor_definitions[target]

            conditional_reference_states[target] = SensorConsistencyState(
                sensor_id=target,
                label=f"{definition.label} | common prior @ simultaneous events",
                input_rate_hz=scenario.nominal_simultaneous_event_rate_hz,
                colour=definition.colour,
                context_label="matched simultaneous common-prior reference",
                physical_horizon_s=CONDITIONAL_SHORT_TERM_HORIZON_S,
                fusion_type=CONDITIONAL_TEF_FUSION,
            )
            conditional_states[(target, conditioning)] = SensorConsistencyState(
                sensor_id=target,
                label=f"{definition.label} | Sensor {conditioning}",
                input_rate_hz=scenario.nominal_simultaneous_event_rate_hz,
                colour=definition.colour,
                context_label=f"conditional after Sensor {conditioning}",
                physical_horizon_s=CONDITIONAL_SHORT_TERM_HORIZON_S,
                fusion_type=CONDITIONAL_TEF_FUSION,
            )

    disagreement_state = SensorDisagreementState((sensor_ids[0], sensor_ids[1]), scenario.nominal_simultaneous_event_rate_hz) if len(sensor_ids)==2 else None

    griebel = GriebelReferenceBackend(sensor_ids, dim_meas=2)

    print("\nTime-normalised TEF settings")
    print(
        f"  consistency horizon: {CONSISTENCY_SHORT_TERM_HORIZON_S:g} s"
    )
    print(
        f"  batch horizon: {BATCH_SHORT_TERM_HORIZON_S:g} s"
    )
    print(
        f"  availability horizon: {AVAILABILITY_SHORT_TERM_HORIZON_S:g} s"
    )
    print(
        f"  conditional/order horizon: {CONDITIONAL_SHORT_TERM_HORIZON_S:g} s"
    )
    print(f"  sensor-pair disagreement horizon: {DISAGREEMENT_SHORT_TERM_HORIZON_S:g} s")
    print(
        "  TEF fusion types: "
        f"local={LOCAL_CONSISTENCY_TEF_FUSION}, "
        f"batch={BATCH_TEF_FUSION}, "
        f"conditional={CONDITIONAL_TEF_FUSION}, "
        f"disagreement={DISAGREEMENT_TEF_FUSION}, "
        f"availability={AVAILABILITY_TEF_FUSION}"
    )
    for sensor_id in sensor_ids:
        consistency = isolated_states[sensor_id]
        availability = availability_states[sensor_id]
        print(
            f"  {consistency.label}: rate={consistency.input_rate_hz:g} Hz, "
            f"C n_ST={consistency.n_st}, gamma={consistency.discount:.6f}; "
            f"A n_ST={availability.n_st}, gamma={availability.discount:.6f}"
        )
    print(
        "  central batch: "
        f"rate~{scenario.nominal_batch_event_rate_hz:.3f} Hz, "
        f"T_ST={BATCH_SHORT_TERM_HORIZON_S:g} s, "
        f"n_ST={batch_track_state.n_st}, "
        f"gamma={batch_track_state.discount:.6f}"
    )
    if conditional_states:
        state = next(iter(conditional_states.values()))
        print(
            "  simultaneous/conditional: "
            f"rate~{scenario.nominal_simultaneous_event_rate_hz:.3f} Hz, "
            f"T_ST={CONDITIONAL_SHORT_TERM_HORIZON_S:g} s, "
            f"n_ST={state.n_st}, gamma={state.discount:.6f}"
        )
    print(f"  Griebel reference: {griebel.status}")

    track = Track()
    event_timestamps: list[datetime] = []
    event_times_s: list[float] = []
    active_batch_sizes: list[int] = []

    trusted_sensor_history = {sensor_id: [] for sensor_id in sensor_ids}
    operational_sensor_history = {sensor_id: [] for sensor_id in sensor_ids}
    sensor_support_history: list = []
    track_output_trust_history: list = []
    common_abf_history: list = []
    batch_history: list = []

    truth_by_timestamp = {state.timestamp: state for state in scenario.truth}
    position_error: list[float] = []
    order_decomposition_error_12: list[float] = []
    order_decomposition_error_21: list[float] = []
    conditional_influence_times_s: list[float] = []
    conditional_influence_scores = {(1, 2): [], (2, 1): []}
    conditional_order_asymmetry: list[float] = []

    for timestamp in sorted(scenario.events_by_timestamp):
        scheduled_batch = scenario.events_by_timestamp[timestamp]
        active_events = [event for event in scheduled_batch if event.arrived]
        active_measurements = [
            event.measurement
            for event in active_events
            if event.measurement is not None
        ]
        active_by_id = {
            event.sensor_id: event.measurement
            for event in active_events
            if event.measurement is not None
        }
        active_sensor_ids = sorted(active_by_id)
        elapsed_s = (timestamp - scenario.start_time).total_seconds()

        # Functional central prediction.  All assessments for this timestamp are
        # evaluated before the actual functional measurement updates.
        central_prediction: GaussianStatePrediction = central_predictor.predict(
            central_prior,
            timestamp=timestamp,
        )

        # ------------------------------------------------------------------
        # A_s: update only when this sensor had a scheduled event.
        # ------------------------------------------------------------------
        for event in scheduled_batch:
            availability_states[event.sensor_id].update(
                event.arrived,
                timestamp,
                scenario.start_time,
            )
            if not event.arrived:
                # No innovation exists, so consistency must neither be rewarded
                # nor penalised. A vacuous TEF input advances temporal memory
                # while explicitly representing missing consistency evidence.
                common_states[event.sensor_id].advance_without_measurement(
                    timestamp,
                    scenario.start_time,
                )
                isolated_states[event.sensor_id].advance_without_measurement(
                    timestamp,
                    scenario.start_time,
                )

        # ------------------------------------------------------------------
        # C_s^common and C_s^iso, plus optional Griebel single-sensor SA.
        # ------------------------------------------------------------------
        griebel_update_this_event = griebel.should_update_event(active_sensor_ids)
        for event in active_events:
            measurement = event.measurement
            assert measurement is not None

            common_updater, common_measurement_prediction = (
                predict_measurement_from_state(
                    central_prediction,
                    measurement,
                )
            )
            del common_updater

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

            if griebel_update_this_event:
                griebel.assess_sensor(
                    event.sensor_id,
                    measurement,
                    common_measurement_prediction,
                    timestamp,
                    scenario.start_time,
                )

        griebel.finish_event(
            timestamp,
            scenario.start_time,
            active_sensor_ids,
        )

        # ------------------------------------------------------------------
        # Conditional/order opinions only when BOTH sensors currently provide
        # a measurement.  These virtual updates do not alter the functional KF.
        # ------------------------------------------------------------------
        scheduled_sensor_ids = {event.sensor_id for event in scheduled_batch}
        if len(sensor_ids) == 2 and scheduled_sensor_ids == set(sensor_ids):
            first_id, second_id = sensor_ids

            if set(active_sensor_ids) == set(sensor_ids):
                if disagreement_state is not None:
                    disagreement_state.update(active_by_id[first_id], active_by_id[second_id], timestamp, scenario.start_time)

                # Matched C_{s|emptyset}^{sim} references: evaluate the target
                # measurement directly against the untouched central prior.
                for target in sensor_ids:
                    _, common_prediction_sim = predict_measurement_from_state(
                        central_prediction,
                        active_by_id[target],
                    )
                    conditional_reference_states[target].update(
                        active_by_id[target],
                        common_prediction_sim,
                        timestamp,
                        scenario.start_time,
                    )

                # C_{second | first}
                posterior_after_first, _ = apply_measurement_update(
                    central_prediction,
                    active_by_id[first_id],
                )
                _, prediction_second_after_first = predict_measurement_from_state(
                    posterior_after_first,
                    active_by_id[second_id],
                )
                conditional_states[(second_id, first_id)].update(
                    active_by_id[second_id],
                    prediction_second_after_first,
                    timestamp,
                    scenario.start_time,
                )

                # C_{first | second}
                posterior_after_second, _ = apply_measurement_update(
                    central_prediction,
                    active_by_id[second_id],
                )
                _, prediction_first_after_second = predict_measurement_from_state(
                    posterior_after_second,
                    active_by_id[first_id],
                )
                conditional_states[(first_id, second_id)].update(
                    active_by_id[first_id],
                    prediction_first_after_second,
                    timestamp,
                    scenario.start_time,
                )
                i_2_to_1 = normalized_disbelief(conditional_states[(first_id, second_id)].latest_opinion) - normalized_disbelief(conditional_reference_states[first_id].latest_opinion)
                i_1_to_2 = normalized_disbelief(conditional_states[(second_id, first_id)].latest_opinion) - normalized_disbelief(conditional_reference_states[second_id].latest_opinion)
                conditional_influence_times_s.append(elapsed_s)
                conditional_influence_scores[(first_id, second_id)].append(i_2_to_1)
                conditional_influence_scores[(second_id, first_id)].append(i_1_to_2)
                conditional_order_asymmetry.append(i_1_to_2 - i_2_to_1)
            else:
                # A simultaneous comparison was scheduled but cannot be formed.
                # Advance matched reference and conditional TEFs with vacuous
                # information so both retain the same physical event clock.
                if disagreement_state is not None:
                    disagreement_state.advance_without_pair(timestamp, scenario.start_time)
                for reference_state in conditional_reference_states.values():
                    reference_state.advance_without_measurement(
                        timestamp,
                        scenario.start_time,
                    )
                for conditional_state in conditional_states.values():
                    conditional_state.advance_without_measurement(
                        timestamp,
                        scenario.start_time,
                    )

        # ------------------------------------------------------------------
        # Direct central batch C_F: variable df -> PIT -> common U(0,1) domain.
        # ------------------------------------------------------------------
        if active_measurements:
            batch_nis, batch_df = stacked_batch_nis(
                central_prediction,
                active_measurements,
            )
            batch_track_state.update(
                batch_nis,
                batch_df,
                active_sensor_ids,
                timestamp,
                scenario.start_time,
            )

            # For the linear independent-noise two-sensor case, the batch NIS
            # must equal either sequential decomposition.  Record the numerical
            # residual as a sanity check of the order analysis.
            if len(sensor_ids) == 2 and set(active_sensor_ids) == set(sensor_ids):
                first_id, second_id = sensor_ids
                eps_first = conditional_reference_states[first_id].nis_events[-1]
                eps_second = conditional_reference_states[second_id].nis_events[-1]
                eps_second_after_first = conditional_states[(second_id, first_id)].nis_events[-1]
                eps_first_after_second = conditional_states[(first_id, second_id)].nis_events[-1]
                order_decomposition_error_12.append(
                    abs(batch_nis - (eps_first + eps_second_after_first))
                )
                order_decomposition_error_21.append(
                    abs(batch_nis - (eps_second + eps_first_after_second))
                )
        else:
            # A scheduled union event exists but no measurement is available.
            # Keep the batch TEF on its physical event clock using no-evidence.
            batch_track_state.advance_without_measurement(
                timestamp,
                scenario.start_time,
            )

        # ------------------------------------------------------------------
        # Derived trust/health statements.  These do not feed back into C_s.
        # ------------------------------------------------------------------
        availability_support_inputs = []
        for sensor_id in sensor_ids:
            c_s = isolated_states[sensor_id].latest_opinion
            a_s = availability_states[sensor_id].latest_opinion

            trusted = trust_discount(a_s, c_s)
            trusted_sensor_history[sensor_id].append(trusted)

            # Distinct diagnostic proposition: available AND consistent.
            # It is intentionally NOT used to discount C_F, otherwise
            # consistency would enter track trust once through C_s and again
            # through the direct central-filter opinion C_F.
            h_s = a_s.multiply(c_s)
            operational_sensor_history[sensor_id].append(h_s)

            availability_support_inputs.append(a_s)

        if TRACK_SENSOR_SUPPORT_RULE.upper() == "ANY":
            sensor_support = disjunction_all(availability_support_inputs)
        elif TRACK_SENSOR_SUPPORT_RULE.upper() == "ALL":
            sensor_support = conjunction_all(availability_support_inputs)
        else:
            raise ValueError("TRACK_SENSOR_SUPPORT_RULE must be ANY or ALL")
        sensor_support_history.append(sensor_support)

        # Experimental output-level trust construction:
        # sensor support acts as reliability trust in the current C_F statement.
        # Keeping this separate in the result allows direct comparison with C_F.
        track_output_trust = trust_discount(
            sensor_support,
            batch_track_state.latest_opinion,
        )
        track_output_trust_history.append(track_output_trust)

        # Common-prediction PIT/TEF architecture baseline.  This holds the last
        # inactive sensor opinion in asynchronous mode and is NOT called Griebel.
        common_abf_history.append(
            fuse_average([
                common_states[sensor_id].latest_opinion
                for sensor_id in sensor_ids
            ])
        )
        batch_history.append(deepcopy(batch_track_state.latest_opinion))

        # ------------------------------------------------------------------
        # Actual functional multi-sensor KF update after all current SA paths.
        # ------------------------------------------------------------------
        central_posterior = central_prediction
        for measurement in active_measurements:
            central_posterior, _ = apply_measurement_update(
                central_posterior,
                measurement,
            )

        central_prior = central_posterior
        track.append(central_posterior)
        event_timestamps.append(timestamp)
        event_times_s.append(elapsed_s)
        active_batch_sizes.append(len(active_measurements))

        truth_state = truth_by_timestamp.get(timestamp)
        if truth_state is None:
            position_error.append(np.nan)
        else:
            estimate = np.asarray(
                central_posterior.state_vector,
                dtype=float,
            ).reshape(-1)
            ground_truth = np.asarray(
                truth_state.state_vector,
                dtype=float,
            ).reshape(-1)
            position_error.append(
                float(
                    np.linalg.norm(
                        estimate[[0, 2]] - ground_truth[[0, 2]]
                    )
                )
            )

    return ProcessingResult(
        track=track,
        event_timestamps=event_timestamps,
        event_times_s=event_times_s,
        active_batch_sizes=active_batch_sizes,
        isolated_states=isolated_states,
        common_prediction_states=common_states,
        availability_states=availability_states,
        conditional_states=conditional_states,
        conditional_reference_states=conditional_reference_states,
        disagreement_state=disagreement_state,
        conditional_influence_times_s=conditional_influence_times_s,
        conditional_influence_scores=conditional_influence_scores,
        conditional_order_asymmetry=conditional_order_asymmetry,
        batch_track_state=batch_track_state,
        griebel=griebel,
        trusted_sensor_history=trusted_sensor_history,
        operational_sensor_history=operational_sensor_history,
        sensor_support_history=sensor_support_history,
        track_output_trust_history=track_output_trust_history,
        common_abf_history=common_abf_history,
        batch_history=batch_history,
        position_error=position_error,
        order_decomposition_error_12=order_decomposition_error_12,
        order_decomposition_error_21=order_decomposition_error_21,
    )


# =============================================================================
# Plotting helpers
# =============================================================================


def add_disturbance_spans(axis) -> None:
    for start_s, end_s, label, colour in DISTURBANCE_INTERVALS:
        if label == "S2 unavailable" and not ENABLE_SENSOR_2_DROPOUT:
            continue
        if label == "common motion-model mismatch" and not ENABLE_GROUND_TRUTH_TURN:
            continue
        axis.axvspan(start_s, end_s, color=colour, alpha=0.07)


def is_nominal_time(time_s: float) -> bool:
    if time_s < NOMINAL_BURN_IN_S:
        return False
    for start, end, label, _ in DISTURBANCE_INTERVALS:
        if label == "S2 unavailable" and not ENABLE_SENSOR_2_DROPOUT:
            continue
        if label == "common motion-model mismatch":
            if not ENABLE_GROUND_TRUTH_TURN:
                continue
            if USE_CT_MODEL:
                # The same ground-truth turn is nominal for the CT filter.
                continue
        if start <= time_s < end:
            return False
    return True


def plot_opinion_pair(
    axis_d,
    axis_u,
    times,
    opinions,
    *,
    label: str,
    colour: str,
    linestyle: str = "-",
    linewidth: float = 1.5,
):
    axis_d.plot(
        times,
        [normalized_disbelief(op) for op in opinions],
        label=label,
        color=colour,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    axis_u.plot(
        times,
        [uncertainty(op) for op in opinions],
        label=label,
        color=colour,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def plot_static_results(result: ProcessingResult) -> None:
    event_times = np.asarray(result.event_times_s, dtype=float)
    sensor_ids = sorted(result.isolated_states)

    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        mode = "fully synchronous 10 Hz"
    elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
        mode = (
            "asynchronous cross-contamination stress: "
            f"{SENSOR_1_RATE_HZ:g} Hz / {SENSOR_2_RATE_HZ:g} Hz"
        )
    else:
        mode = "asynchronous 10 Hz / 12.5 Hz"

    # ------------------------------------------------------------------
    # Figure 1: local diagnosis - primary evaluation uses d_norm + u.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3,
        len(sensor_ids),
        figsize=(16, 10),
        sharex="col",
        squeeze=False,
    )
    for col, sensor_id in enumerate(sensor_ids):
        iso = result.isolated_states[sensor_id]
        common = result.common_prediction_states[sensor_id]

        axes[0, col].plot(
            iso.event_times_s,
            normalized_disbelief_series(
                iso.disbelief_events, iso.uncertainty_events
            ),
            color=iso.colour,
            linewidth=1.7,
            label=r"isolated $d_{C,\mathrm{norm}}$",
        )
        axes[0, col].plot(
            common.event_times_s,
            normalized_disbelief_series(
                common.disbelief_events, common.uncertainty_events
            ),
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label=r"common-prior $d_{C,\mathrm{norm}}$",
        )
        axes[0, col].set_ylabel(r"normalized disbelief $d_{\mathrm{norm}}$")

        axes[1, col].plot(
            iso.event_times_s,
            iso.uncertainty_events,
            color=iso.colour,
            linewidth=1.7,
            label="isolated $u_C$",
        )
        axes[1, col].plot(
            common.event_times_s,
            common.uncertainty_events,
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label="common-prior $u_C$",
        )
        axes[1, col].set_ylabel("uncertainty")

        axes[2, col].plot(
            iso.event_times_s,
            iso.nis_average,
            color=iso.colour,
            linewidth=1.5,
            label="isolated avg. NIS",
        )
        axes[2, col].plot(
            common.event_times_s,
            common.nis_average,
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label="common-prior avg. NIS",
        )
        axes[2, col].plot(
            iso.event_times_s,
            iso.nis_lower,
            color="black",
            linestyle=":",
            linewidth=0.9,
            label="99% interval",
        )
        axes[2, col].plot(
            iso.event_times_s,
            iso.nis_upper,
            color="black",
            linestyle=":",
            linewidth=0.9,
        )
        axes[2, col].set_ylabel("NIS")
        axes[2, col].set_xlabel("time [s]")

        for row in range(3):
            axes[row, col].grid(True)
            axes[row, col].legend(loc="upper right", fontsize=8)
            add_disturbance_spans(axes[row, col])
            if row < 2:
                axes[row, col].set_ylim(-0.02, 1.02)
        axes[0, col].set_title(iso.label)

    fig.suptitle(
        "Sensor-specific consistency: isolated shadow KF vs common central prior\n"
        rf"({mode}; $d_{{\mathrm{{norm}}}}=d/(1-u)$ = normalized inconsistency, $u$ = lack of evidence)"
    )
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: availability and trust discount.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for sensor_id in sensor_ids:
        availability = result.availability_states[sensor_id]
        colour = result.isolated_states[sensor_id].colour
        axes[0].plot(
            availability.event_times_s,
            availability.p_available_events,
            color=colour,
            label=f"Sensor {sensor_id}: $P(A_s)$",
        )
        axes[1].plot(
            availability.event_times_s,
            availability.uncertainty_events,
            color=colour,
            label=f"Sensor {sensor_id}: $u(A_s)$",
        )
        axes[2].plot(
            event_times,
            [uncertainty(op) for op in result.trusted_sensor_history[sensor_id]],
            color=colour,
            label=rf"Sensor {sensor_id}: $u(A_s \otimes C_s)$",
        )
    axes[0].set_ylabel(r"availability $P(A_s)$")
    axes[1].set_ylabel(r"availability $u(A_s)$")
    axes[2].set_ylabel(r"trusted consistency $u$")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    fig.suptitle(
        "Availability is a separate opinion and acts as reliability trust\n"
        rf"($T_A={AVAILABILITY_SHORT_TERM_HORIZON_S:g}$ s; no continuous freshness decay)"
    )
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3: direct central-filter batch opinion and active sensor set.
    # ------------------------------------------------------------------
    batch = result.batch_track_state
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(
        batch.event_times_s,
        normalized_disbelief_series(
            batch.disbelief_events, batch.uncertainty_events
        ),
        color="tab:purple",
        label=r"direct batch $d_{C_F,\mathrm{norm}}$",
    )
    axes[1].plot(
        batch.event_times_s,
        batch.uncertainty_events,
        color="tab:purple",
        label=r"direct batch $u_{C_F}$",
    )

    # Categorical active-set timeline. This shows which measurement batch
    # actually generated each central-filter PIT observation.
    labels_in_order = []
    for label in batch.active_sensor_sets:
        if label not in labels_in_order:
            labels_in_order.append(label)
    active_set_to_y = {label: idx for idx, label in enumerate(labels_in_order)}
    active_set_y = [active_set_to_y[label] for label in batch.active_sensor_sets]

    axes[2].scatter(
        batch.event_times_s,
        active_set_y,
        c="tab:blue",
        s=10,
        alpha=0.7,
    )
    axes[2].set_yticks(list(active_set_to_y.values()))
    axes[2].set_yticklabels(list(active_set_to_y.keys()))
    axes[2].set_ylabel("active set")
    axes[2].set_xlabel("time [s]")

    axes[0].set_ylabel(r"normalized disbelief $d_{\mathrm{norm}}$")
    axes[1].set_ylabel("uncertainty")
    for axis in axes:
        axis.grid(True)
        add_disturbance_spans(axis)
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle(
        "Direct central-filter consistency $C_F$: "
        rf"$T_{{ST}}={BATCH_SHORT_TERM_HORIZON_S:g}$ s, "
        "variable active measurement batch"
    )
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 4: conditional/order opinions at simultaneous timestamps.
    # ------------------------------------------------------------------
    if result.conditional_states:
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

        for (target, conditioning), state in sorted(result.conditional_states.items()):
            reference = result.conditional_reference_states[target]

            conditional_d_norm = normalized_disbelief_series(
                state.disbelief_events,
                state.uncertainty_events,
            )
            reference_d_norm = normalized_disbelief_series(
                reference.disbelief_events,
                reference.uncertainty_events,
            )

            conditional_label = rf"$C_{{{target}|{conditioning}}}$"
            reference_label = rf"$C_{{{target}|\emptyset}}^{{\mathrm{{sim}}}}$"

            # Same timestamps, same rate, same TEF horizon/operator. The only
            # difference is whether the conditioning sensor was virtually
            # processed before assessing the target sensor.
            axes[0].plot(
                state.event_times_s,
                conditional_d_norm,
                color=state.colour,
                linewidth=1.6,
                label=conditional_label,
            )
            axes[0].plot(
                reference.event_times_s,
                reference_d_norm,
                color=state.colour,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=reference_label,
            )

            delta = np.asarray(conditional_d_norm) - np.asarray(reference_d_norm)
            axes[1].plot(
                state.event_times_s,
                delta,
                color=state.colour,
                linewidth=1.5,
                label=rf"$I_{{{conditioning}\rightarrow{target}}}$",
            )

            axes[2].plot(
                state.event_times_s,
                state.uncertainty_events,
                color=state.colour,
                linewidth=1.6,
                label=conditional_label + r" $u$",
            )
            axes[2].plot(
                reference.event_times_s,
                reference.uncertainty_events,
                color=state.colour,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=reference_label + r" $u$",
            )

        axes[0].set_ylabel(r"normalized disbelief $d_{\mathrm{norm}}$")
        axes[1].set_ylabel(
            r"$I_{j\rightarrow s} = "
            r"d_{\mathrm{norm}}(C_{s|j})"
            r"-d_{\mathrm{norm}}(C_{s|\emptyset}^{\mathrm{sim}})$"
        )
        axes[2].set_ylabel("uncertainty")
        axes[2].set_xlabel("time [s]")
        axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        if result.conditional_order_asymmetry:
            axes[1].plot(result.conditional_influence_times_s, result.conditional_order_asymmetry,
                         color="black", linestyle=":", linewidth=1.2,
                         label=r"$A_I=I_{1\rightarrow2}-I_{2\rightarrow1}$ (diagnostic only)")

        for axis in axes:
            axis.grid(True)
            axis.legend(loc="upper right")
            add_disturbance_spans(axis)

        axes[0].set_ylim(-0.02, 1.02)
        axes[2].set_ylim(-0.02, 1.02)
        fig.suptitle(
            "Matched conditional influence diagnostics\n"
            rf"same simultaneous events, same $T_{{ST}}={CONDITIONAL_SHORT_TERM_HORIZON_S:g}$ s "
            r"TEF; $I_{j\rightarrow s}$ quantifies update-order influence, not fault attribution"
        )
        fig.tight_layout()

    # ------------------------------------------------------------------
    # V4: direct sensor-to-sensor disagreement at simultaneous timestamps.
    # ------------------------------------------------------------------
    if result.disagreement_state is not None:
        pair = result.disagreement_state
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        pair_d_norm = normalized_disbelief_series(pair.disbelief_events, pair.uncertainty_events)
        axes[0].plot(pair.event_times_s, pair_d_norm, color="tab:cyan", linewidth=1.7,
                     label=r"pair disagreement $d_{\mathrm{norm},D_{12}}$")
        axes[1].plot(pair.event_times_s, pair.uncertainty_events, color="tab:cyan", linewidth=1.5,
                     label=r"pair agreement uncertainty $u_{D_{12}}$")
        axes[2].plot(pair.event_times_s, pair.mean_standardized_x_events, color="tab:blue", linewidth=1.5,
                     label=r"mean $(z_{1,x}-z_{2,x})/\sigma_{\Delta x}$")
        axes[2].plot(pair.event_times_s, pair.mean_standardized_y_events, color="tab:orange", linewidth=1.5,
                     label=r"mean $(z_{1,y}-z_{2,y})/\sigma_{\Delta y}$")
        axes[2].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axes[2].axhline(2.0, color="gray", linestyle="--", linewidth=0.9)
        axes[2].axhline(-2.0, color="gray", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel(r"normalized disagreement $d_{\mathrm{norm}}$")
        axes[1].set_ylabel("uncertainty")
        axes[2].set_ylabel("standardized relative offset")
        axes[2].set_xlabel("time [s]")
        axes[0].set_ylim(-0.02,1.02); axes[1].set_ylim(-0.02,1.02)
        for axis in axes:
            axis.grid(True); axis.legend(loc="upper right"); add_disturbance_spans(axis)
        fig.suptitle("Direct sensor-to-sensor disagreement diagnostic\n" +
                     r"$\Delta z=z_1-z_2,\quad S_\Delta=R_1+R_2$; independent of the central KF prior")
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 5: architecture baseline + optional original Griebel reference.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(
        event_times,
        [normalized_disbelief(op) for op in result.common_abf_history],
        color="tab:gray",
        linestyle="--",
        label=r"common-prior PIT/TEF ABF baseline: $d_{\mathrm{norm}}$",
    )
    axes[0].plot(
        event_times,
        [normalized_disbelief(op) for op in result.batch_history],
        color="tab:purple",
        label=r"proposed direct batch $C_F$: $d_{\mathrm{norm}}$",
    )
    axes[1].plot(
        event_times,
        [uncertainty(op) for op in result.common_abf_history],
        color="tab:gray",
        linestyle="--",
        label="common-prior PIT/TEF ABF baseline: uncertainty",
    )
    axes[1].plot(
        event_times,
        [uncertainty(op) for op in result.batch_history],
        color="tab:purple",
        label="proposed direct batch $C_F$: uncertainty",
    )

    if result.griebel.native and result.griebel.overall_times_s:
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
            griebel_label = "Griebel 2023 PP overall reference"
        elif GRIEBEL_ASYNC_EXTENSION_MODE == "hold_last":
            griebel_label = "Griebel-style async hold-last PP (not published method)"
        else:
            griebel_label = "Griebel PP on simultaneous full batches only"
        axes[2].plot(
            result.griebel.overall_times_s,
            result.griebel.overall_p_ok,
            color="black",
            linewidth=1.5,
            label=griebel_label,
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Griebel native backend unavailable / placeholder active",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    if SHOW_PROJECTED_PROBABILITY:
        axes[2].plot(
            event_times,
            [p_ok(op) for op in result.batch_history],
            color="tab:purple",
            alpha=0.75,
            label="proposed direct batch PP (secondary view)",
        )
    axes[0].set_ylabel(r"normalized disbelief $d_{\mathrm{norm}}$")
    axes[1].set_ylabel("uncertainty")
    axes[2].set_ylabel("projected probability")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    fig.suptitle(
        "Reference comparison\n"
        "Griebel is native in the synchronous case; async hold-last is explicitly only a straightforward extension"
    )
    fig.tight_layout()

    # Local Griebel threshold outputs are useful for checking reproduction.
    if result.griebel.native:
        fig, axes = plt.subplots(
            len(sensor_ids), 1, figsize=(15, 6), sharex=True, squeeze=False
        )
        for row, sensor_id in enumerate(sensor_ids):
            axis = axes[row, 0]
            axis.plot(
                result.griebel.sensor_times[sensor_id],
                result.griebel.sensor_delta[sensor_id],
                color=result.isolated_states[sensor_id].colour,
                label=rf"Griebel $\delta^{{({sensor_id})}}$",
            )
            axis.plot(
                result.griebel.sensor_times[sensor_id],
                result.griebel.sensor_eta[sensor_id],
                color="black",
                linestyle="--",
                label=rf"Griebel $\eta^{{({sensor_id})}}$",
            )
            axis.grid(True)
            axis.legend(loc="upper right")
            axis.set_ylabel("DC / threshold")
            add_disturbance_spans(axis)
        axes[-1, 0].set_xlabel("time [s]")
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
            griebel_local_title = (
                "Native Griebel single-sensor SA reproduction: threshold decisions"
            )
        else:
            griebel_local_title = (
                "Native Griebel single-sensor assessor in the asynchronous "
                "hold-last extension: threshold decisions"
            )
        fig.suptitle(griebel_local_title)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 6: experimental output-level trust view.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(
        event_times,
        [p_ok(op) for op in result.sensor_support_history],
        color="tab:green",
        label=rf"availability support ({TRACK_SENSOR_SUPPORT_RULE}) $P$",
    )
    axes[1].plot(
        event_times,
        [normalized_disbelief(op) for op in result.batch_history],
        color="tab:purple",
        label=r"filter consistency $d_{C_F,\mathrm{norm}}$",
    )
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.batch_history],
        color="tab:purple",
        label=r"filter consistency $u_{C_F}$",
    )
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.track_output_trust_history],
        color="tab:red",
        linestyle="--",
        label=r"availability-discounted track-output $u$",
    )
    axes[0].set_ylabel("availability support")
    axes[1].set_ylabel(r"normalized disbelief $d_{\mathrm{norm}}$")
    axes[2].set_ylabel("uncertainty")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    fig.suptitle(
        "Experimental track-output trust construction\n"
        r"availability support acts as reliability trust for $C_F$; "
        r"local $C_s$ remains diagnostic and is not counted twice"
    )
    fig.tight_layout()

    if SHOW_POSITION_ERROR:
        fig, axis = plt.subplots(figsize=(15, 4))
        axis.plot(event_times, result.position_error, color="tab:blue")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("position error [m]")
        axis.set_title("Central track position error (GT used for offline evaluation only)")
        axis.grid(True)
        add_disturbance_spans(axis)
        fig.tight_layout()


# =============================================================================
# Dynamic animation
# =============================================================================


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


def _op_triplet(opinion) -> tuple[float, float, float]:
    return (
        float(belief(opinion)),
        float(disbelief(opinion)),
        float(uncertainty(opinion)),
    )


def _vacuous_triplet() -> tuple[float, float, float]:
    return (0.0, 0.0, 1.0)


def _state_triplet_at_time(state, elapsed_s: float) -> tuple[float, float, float]:
    """Last available opinion at/before elapsed_s."""
    if not state.event_times_s:
        return _vacuous_triplet()
    times = np.asarray(state.event_times_s, dtype=float)
    index = int(np.searchsorted(times, elapsed_s, side="right") - 1)
    if index < 0:
        return _vacuous_triplet()
    return (
        float(state.belief_events[index]),
        float(state.disbelief_events[index]),
        float(state.uncertainty_events[index]),
    )


def _ternary_marker_trace(go, entries, marker_size: int = 12):
    labels, b_values, d_values, u_values, colours = [], [], [], [], []
    for label, value, colour in entries:
        b_value, d_value, u_value = value if isinstance(value, tuple) else _op_triplet(value)
        labels.append(label)
        b_values.append(b_value)
        d_values.append(d_value)
        u_values.append(u_value)
        colours.append(colour)

    return go.Scatterternary(
        a=u_values,
        b=d_values,
        c=b_values,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=colours,
            line=dict(width=1, color="black"),
        ),
        text=labels,
        hovertemplate=(
            "%{text}<br>"
            "belief=%{c:.3f}<br>"
            "disbelief=%{b:.3f}<br>"
            "uncertainty=%{a:.3f}<extra></extra>"
        ),
        showlegend=False,
    )


def _configure_ternary_axes(fig) -> None:
    """Same ternary appearance as 01_KalmanFilterWithSelfAssessmentV7."""
    for key in [key for key in fig.layout if str(key).startswith("ternary")]:
        fig.layout[key].update(
            sum=1,
            # V7 relies on Plotly's standard light-blue plotting background.
            bgcolor="#E5ECF6",
            aaxis=dict(
                title="uncertainty",
                showgrid=True,
                gridcolor="black",
                ticks="",
                linecolor="rgba(0,0,0,0)",
                showticklabels=False,
            ),
            baxis=dict(
                title="disbelief",
                showgrid=True,
                gridcolor="black",
                ticks="",
                linecolor="rgba(0,0,0,0)",
                showticklabels=False,
            ),
            caxis=dict(
                title="belief",
                showgrid=True,
                gridcolor="black",
                ticks="",
                linecolor="rgba(0,0,0,0)",
                showticklabels=False,
            ),
        )


def _active_disturbance_labels(elapsed_s: float) -> list[str]:
    labels: list[str] = []

    if ACTIVATE_DISTURBANCES and DISTURB_SENSOR_1:
        for start_s, end_s, label, _ in DISTURBANCE_INTERVALS:
            if label.startswith("S1 ") and start_s <= elapsed_s < end_s:
                labels.append(label)

    if ACTIVATE_DISTURBANCES and DISTURB_SENSOR_2:
        for start_s, end_s, label, _ in DISTURBANCE_INTERVALS:
            if (
                label.startswith("S2 ")
                and "unavailable" not in label
                and start_s <= elapsed_s < end_s
            ):
                labels.append(label)

    if (
        ENABLE_SENSOR_2_DROPOUT
        and SENSOR_2_DROPOUT_INTERVAL_S[0]
        <= elapsed_s
        < SENSOR_2_DROPOUT_INTERVAL_S[1]
    ):
        labels.append("S2 unavailable")

    if (
        ENABLE_GROUND_TRUTH_TURN
        and not USE_CT_MODEL
        and TURN_INTERVAL_S[0] <= elapsed_s < TURN_INTERVAL_S[1]
    ):
        labels.append("common motion-model mismatch (coordinated turn)")

    if ACTIVATE_DISTURBANCES and (
        INCREASED_PROCESS_XY_INTERVAL_S[0]
        <= elapsed_s
        < INCREASED_PROCESS_XY_INTERVAL_S[1]
    ):
        labels.append("common increased x/y process noise")

    return labels


def _disturbance_annotation(step: int, elapsed_s: float) -> dict:
    active = _active_disturbance_labels(elapsed_s)

    if active:
        title = "ACTIVE DISTURBANCE"
        body = "<br>".join(active)
        bg = "rgba(190,35,35,0.96)"
        border = "rgb(125,15,15)"
    elif ACTIVATE_DISTURBANCES or ENABLE_SENSOR_2_DROPOUT:
        title = "DISTURBANCE STATUS"
        body = "Nominal operation"
        bg = "rgba(25,145,70,0.96)"
        border = "rgb(10,95,40)"
    else:
        title = "DISTURBANCE STATUS"
        body = "Disturbances disabled"
        bg = "rgba(70,130,180,0.95)"
        border = "rgb(45,90,125)"

    return dict(
        x=0.29,
        y=0.985,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        text=(
            f"<b>{title}</b><br>{body}<br>"
            f"<span style='font-size:11px'>Step {int(step)}</span>"
        ),
        showarrow=False,
        align="center",
        bgcolor=bg,
        bordercolor=border,
        borderwidth=2,
        borderpad=8,
        font=dict(color="white", size=13),
    )


def _position_xy(state) -> np.ndarray:
    vector = np.asarray(state.state_vector, dtype=float).reshape(-1)
    return np.array([float(vector[0]), float(vector[2])], dtype=float)


def _covariance_ellipse_relative(
    state,
    sigma_scale: float = ANIMATION_COVARIANCE_SCALE,
    n_points: int = 60,
):
    covariance = np.asarray(state.covar, dtype=float)
    p_xy = covariance[np.ix_([0, 2], [0, 2])]
    p_xy = 0.5 * (p_xy + p_xy.T)
    eigvals, eigvecs = np.linalg.eigh(p_xy)
    eigvals = np.maximum(eigvals, 1e-10)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = eigvecs @ np.diag(sigma_scale * np.sqrt(eigvals)) @ circle
    return ellipse[0, :], ellipse[1, :]


def show_dynamic_animation(
    scenario: ScenarioData,
    result: ProcessingResult,
) -> None:
    """V7-style Plotly dashboard with follow-view and SL opinion triangles."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(result.track) == 0 or len(result.event_timestamps) == 0:
        print("Dynamic animation skipped: no track/event data.")
        return

    all_steps = list(range(len(result.event_timestamps)))
    requested_stride = max(1, int(ANIMATION_FRAME_STRIDE))
    stride_for_frame_cap = max(
        1,
        int(ceil(len(all_steps) / max(1, int(ANIMATION_MAX_FRAMES)))),
    )
    stride = max(requested_stride, stride_for_frame_cap)
    frame_steps = all_steps[::stride]
    if frame_steps[-1] != all_steps[-1]:
        frame_steps.append(all_steps[-1])

    print(
        "Animation frames: "
        f"{len(frame_steps)} / {len(all_steps)} union-event steps "
        f"(effective stride={stride})"
    )

    truth_by_timestamp = {state.timestamp: state for state in scenario.truth}
    measurement_records = {}
    for sensor_id, measurements in scenario.measurements_by_sensor.items():
        records = []
        for measurement in measurements:
            z = np.asarray(measurement.state_vector, dtype=float).reshape(-1)
            records.append((measurement.timestamp, float(z[0]), float(z[1])))
        measurement_records[sensor_id] = records

    def history_triplet(history, step):
        if not history:
            return _vacuous_triplet()
        return _op_triplet(history[min(step, len(history) - 1)])

    def opinion_panels(step: int):
        elapsed_s = float(result.event_times_s[step])

        c1_iso = _state_triplet_at_time(result.isolated_states[1], elapsed_s)
        c1_common = _state_triplet_at_time(result.common_prediction_states[1], elapsed_s)
        c2_iso = _state_triplet_at_time(result.isolated_states[2], elapsed_s)
        c2_common = _state_triplet_at_time(result.common_prediction_states[2], elapsed_s)
        a1 = _state_triplet_at_time(result.availability_states[1], elapsed_s)
        a2 = _state_triplet_at_time(result.availability_states[2], elapsed_s)

        trusted1 = history_triplet(result.trusted_sensor_history[1], step)
        trusted2 = history_triplet(result.trusted_sensor_history[2], step)
        h1 = history_triplet(result.operational_sensor_history[1], step)
        h2 = history_triplet(result.operational_sensor_history[2], step)
        c_f = history_triplet(result.batch_history, step)
        common_abf = history_triplet(result.common_abf_history, step)
        support = history_triplet(result.sensor_support_history, step)
        track_trust = history_triplet(result.track_output_trust_history, step)
        pair_agreement = _state_triplet_at_time(result.disagreement_state, elapsed_s) if result.disagreement_state is not None else _vacuous_triplet()

        c12 = _state_triplet_at_time(result.conditional_states[(1, 2)], elapsed_s)
        c1ref = _state_triplet_at_time(result.conditional_reference_states[1], elapsed_s)
        c21 = _state_triplet_at_time(result.conditional_states[(2, 1)], elapsed_s)
        c2ref = _state_triplet_at_time(result.conditional_reference_states[2], elapsed_s)

        return [
            [("C1 isolated", c1_iso, "royalblue"), ("C1 common", c1_common, "gray")],
            [("C2 isolated", c2_iso, "darkorange"), ("C2 common", c2_common, "gray")],
            [("A1", a1, "royalblue"), ("A2", a2, "darkorange")],
            [("A1 ⊗ C1", trusted1, "royalblue"), ("A2 ⊗ C2", trusted2, "darkorange")],
            [
                ("C_F", c_f, "purple"),
                ("common ABF", common_abf, "gray"),
                ("availability support", support, "green"),
                ("track trust", track_trust, "red"),
                ("pair agreement D12", pair_agreement, "cyan"),
            ],
            [("H1 = A1 ∧ C1", h1, "royalblue"), ("H2 = A2 ∧ C2", h2, "darkorange")],
            [("C1|2", c12, "royalblue"), ("C1|∅ sim", c1ref, "deepskyblue")],
            [("C2|1", c21, "darkorange"), ("C2|∅ sim", c2ref, "gold")],
        ]

    fig = make_subplots(
        rows=4,
        cols=4,
        specs=[
            [{"type": "xy", "rowspan": 4, "colspan": 2}, None, {"type": "ternary"}, {"type": "ternary"}],
            [None, None, {"type": "ternary"}, {"type": "ternary"}],
            [None, None, {"type": "ternary"}, {"type": "ternary"}],
            [None, None, {"type": "ternary"}, {"type": "ternary"}],
        ],
        subplot_titles=[
            "Track follow view",
            "Sensor 1 consistency",
            "Sensor 2 consistency",
            "Availability",
            "Availability-discounted",
            "Central / system / track",
            "Operational sensor path",
            "Conditional Sensor 1",
            "Conditional Sensor 2",
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.05,
    )

    # V7 shifts opinion titles in paper coordinates to the left and down.
    for annotation in fig.layout.annotations:
        if annotation.text == "Track follow view":
            annotation.update(font=dict(size=15))
        else:
            annotation.update(
                x=annotation.x - 0.10,
                y=annotation.y - 0.05,
                xanchor="left",
                align="left",
                font=dict(size=14),
            )

    final_step = len(result.event_timestamps) - 1

    def _route_first_step(step: int) -> int:
        if ANIMATION_FINAL_SHOW_FULL_ROUTE and step == final_step:
            return 0
        return max(0, step - int(ANIMATION_TAIL_STEPS) + 1)

    def _tracking_axis_ranges(step: int):
        if not (ANIMATION_FINAL_SHOW_FULL_ROUTE and step == final_step):
            return (
                [-ANIMATION_HALF_WIDTH_M, ANIMATION_HALF_WIDTH_M],
                [-ANIMATION_HALF_HEIGHT_M, ANIMATION_HALF_HEIGHT_M],
            )

        center = _position_xy(result.track[step])
        point_sets = [
            np.asarray(
                [_position_xy(state) for state in result.track[: step + 1]],
                dtype=float,
            ) - center
        ]

        truth_points_full = []
        for timestamp in result.event_timestamps[: step + 1]:
            truth_state = truth_by_timestamp.get(timestamp)
            if truth_state is not None:
                truth_points_full.append(_position_xy(truth_state))
        if truth_points_full:
            point_sets.append(
                np.asarray(truth_points_full, dtype=float) - center
            )

        all_points = np.vstack(point_sets)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        full_span = max(float(x_max - x_min), float(y_max - y_min), 1.0)

        half_span = (
            0.5
            * full_span
            * (1.0 + 2.0 * float(ANIMATION_FINAL_ROUTE_MARGIN))
        )
        x_mid = 0.5 * float(x_min + x_max)
        y_mid = 0.5 * float(y_min + y_max)

        return (
            [x_mid - half_span, x_mid + half_span],
            [y_mid - half_span, y_mid + half_span],
        )

    def tracking_traces(step: int):
        current_state = result.track[step]
        center = _position_xy(current_state)

        route_first_step = _route_first_step(step)
        measurement_first_step = max(
            0, step - int(ANIMATION_TAIL_STEPS) + 1
        )

        track_xy = np.asarray(
            [
                _position_xy(state)
                for state in result.track[route_first_step : step + 1]
            ],
            dtype=float,
        )
        track_rel = track_xy - center

        truth_points = []
        for timestamp in result.event_timestamps[route_first_step : step + 1]:
            truth_state = truth_by_timestamp.get(timestamp)
            if truth_state is not None:
                truth_points.append(_position_xy(truth_state))
        truth_rel = (
            np.asarray(truth_points, dtype=float) - center
            if truth_points
            else np.empty((0, 2), dtype=float)
        )

        current_timestamp = result.event_timestamps[step]
        first_timestamp = result.event_timestamps[measurement_first_step]
        measurement_xy = {}
        for sensor_id, records in measurement_records.items():
            points = [
                [x, y]
                for timestamp, x, y in records
                if first_timestamp <= timestamp <= current_timestamp
            ]
            measurement_xy[sensor_id] = (
                np.asarray(points, dtype=float) - center
                if points
                else np.empty((0, 2), dtype=float)
            )

        ellipse_x, ellipse_y = _covariance_ellipse_relative(current_state)

        return [
            go.Scattergl(
                x=truth_rel[:, 0] if len(truth_rel) else [],
                y=truth_rel[:, 1] if len(truth_rel) else [],
                mode="lines",
                line=dict(color="black", width=2),
                name="Ground truth",
            ),
            go.Scattergl(
                x=track_rel[:, 0],
                y=track_rel[:, 1],
                mode="lines",
                line=dict(color="crimson", width=2.5),
                name="Central track",
            ),
            go.Scattergl(
                x=measurement_xy[1][:, 0] if len(measurement_xy[1]) else [],
                y=measurement_xy[1][:, 1] if len(measurement_xy[1]) else [],
                mode="markers",
                marker=dict(size=5, color="royalblue", opacity=0.6),
                name="Sensor 1 measurements",
            ),
            go.Scattergl(
                x=measurement_xy[2][:, 0] if len(measurement_xy[2]) else [],
                y=measurement_xy[2][:, 1] if len(measurement_xy[2]) else [],
                mode="markers",
                marker=dict(size=5, color="darkorange", opacity=0.6),
                name="Sensor 2 measurements",
            ),
            go.Scatter(
                x=ellipse_x,
                y=ellipse_y,
                mode="lines",
                line=dict(color="crimson", width=1.5, dash="dot"),
                name="Track uncertainty ellipse",
            ),
            go.Scattergl(
                x=[0.0],
                y=[0.0],
                mode="markers",
                marker=dict(size=8, color="crimson", symbol="x"),
                name="Current track",
            ),
        ]

    initial_step = frame_steps[0]
    dynamic_indices = []

    for trace in tracking_traces(initial_step):
        fig.add_trace(trace, row=1, col=1)
        dynamic_indices.append(len(fig.data) - 1)

    ternary_positions = [
        (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 3), (3, 4),
        (4, 3), (4, 4),
    ]
    for entries, (row, col) in zip(opinion_panels(initial_step), ternary_positions):
        fig.add_trace(_ternary_marker_trace(go, entries), row=row, col=col)
        dynamic_indices.append(len(fig.data) - 1)

    initial_x_range, initial_y_range = _tracking_axis_ranges(initial_step)
    fig.update_xaxes(
        range=initial_x_range,
        autorange=False,
        title_text="relative x [m]",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=initial_y_range,
        autorange=False,
        title_text="relative y [m]",
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    _configure_ternary_axes(fig)

    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        rate_mode = "synchronous"
    elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
        rate_mode = "cross-contamination stress"
    else:
        rate_mode = "asynchronous multi-rate"

    model_label = "CT/UKF" if USE_CT_MODEL else "CV/KF"
    static_annotations = [
        annotation.to_plotly_json()
        for annotation in (fig.layout.annotations or [])
    ]

    fig.update_layout(
        title=(
            "Dynamic event-based multi-sensor self-assessment"
            f"<br><sup>{rate_mode}; S1={SENSOR_1_RATE_HZ:g} Hz, "
            f"S2={SENSOR_2_RATE_HZ:g} Hz; filter={model_label}</sup>"
        ),
        width=1750,
        height=1120,
        margin=dict(t=120, b=95, l=60, r=30),
        annotations=static_annotations + [
            _disturbance_annotation(
                initial_step,
                float(result.event_times_s[initial_step]),
            )
        ],
        # The only global legend belongs to the tracking panel. Place it inside
        # that panel so it cannot collide with the x-axis label or slider.
        legend=dict(
            orientation="v",
            x=0.012,
            y=0.80,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(80,80,80,0.35)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )

    frames = []
    for step in frame_steps:
        frame_data = tracking_traces(step)
        for entries in opinion_panels(step):
            frame_data.append(_ternary_marker_trace(go, entries))

        frame_x_range, frame_y_range = _tracking_axis_ranges(step)
        frames.append(
            go.Frame(
                name=str(int(step)),
                data=frame_data,
                traces=dynamic_indices,
                layout=go.Layout(
                    xaxis=dict(range=frame_x_range, autorange=False),
                    yaxis=dict(
                        range=frame_y_range,
                        autorange=False,
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                    annotations=static_annotations + [
                        _disturbance_annotation(
                            step,
                            float(result.event_times_s[step]),
                        )
                    ],
                ),
            )
        )
    fig.frames = tuple(frames)

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(int(step))],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(int(step)),
        )
        for step in frame_steps
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=-0.075,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=ANIMATION_FRAME_DURATION_MS,
                                    redraw=True,
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
                                frame=dict(duration=0, redraw=True),
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
                steps=slider_steps,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=48),
            )
        ],
    )

    output_path = SCRIPT_DIR / ANIMATION_HTML_FILENAME
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        auto_open=ANIMATION_AUTO_OPEN_BROWSER,
        auto_play=False,
    )
    print(f"Dynamic animation written to: {output_path}")


# =============================================================================
# Summary / diagnostics
# =============================================================================


def interval_mean(
    times: list[float],
    values: list[float],
    interval: tuple[float, float],
) -> float:
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = (t >= interval[0]) & (t < interval[1])
    return float(np.nanmean(v[mask])) if np.any(mask) else float("nan")


def nominal_mean(times: list[float], values: list[float]) -> float:
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.asarray([is_nominal_time(x) for x in t], dtype=bool)
    return float(np.nanmean(v[mask])) if np.any(mask) else float("nan")


def print_summary(result: ProcessingResult) -> None:
    simultaneous = sum(size == 2 for size in result.active_batch_sizes)
    single = sum(size == 1 for size in result.active_batch_sizes)
    zero = sum(size == 0 for size in result.active_batch_sizes)

    print("\nEvent statistics")
    print(f"  union event timestamps: {len(result.event_times_s)}")
    print(f"  simultaneous active two-sensor batches: {simultaneous}")
    print(f"  single-sensor active batches: {single}")
    print(f"  scheduled timestamps without active measurement: {zero}")

    print("\nGriebel reference")
    print(f"  backend: {result.griebel.status}")
    print(f"  async extension mode: {GRIEBEL_ASYNC_EXTENSION_MODE}")
    if result.griebel.native:
        print(
            f"  overall PP samples: {len(result.griebel.overall_p_ok)}"
        )

    # The key cross-contamination diagnostic is nominal Sensor-2 d_norm
    # during Sensor-1-only disturbance intervals.
    s2_iso = result.isolated_states[2]
    s2_common = result.common_prediction_states[2]
    s2_iso_d_norm = normalized_disbelief_series(
        s2_iso.disbelief_events, s2_iso.uncertainty_events
    )
    s2_common_d_norm = normalized_disbelief_series(
        s2_common.disbelief_events, s2_common.uncertainty_events
    )
    nominal_iso = nominal_mean(s2_iso.event_times_s, s2_iso_d_norm)
    nominal_common = nominal_mean(
        s2_common.event_times_s,
        s2_common_d_norm,
    )

    print("\nSensor-2 cross-contamination check (Sensor 1 disturbed only)")
    print(
        f"  nominal mean d_norm: isolated={nominal_iso:.3f}, "
        f"common-prior={nominal_common:.3f}"
    )
    for interval, label in (
        (OUTLIER_INTERVAL_S, "S1 outliers"),
        (SENSOR_1_BIAS_INTERVAL_S, f"S1 +{SENSOR_1_BIAS_VECTOR_M[0]:g} m x-bias"),
        (INCREASED_MEAS_XY_INTERVAL_S, f"S1 increased x/y-noise (R x{MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR:g})"),
        (VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S, "S1 variance-matched non-Gaussian noise"),
    ):
        iso = interval_mean(s2_iso.event_times_s, s2_iso_d_norm, interval)
        common = interval_mean(
            s2_common.event_times_s,
            s2_common_d_norm,
            interval,
        )
        print(
            f"  {label}: isolated d_norm={iso:.3f} "
            f"(delta={iso - nominal_iso:+.3f}), "
            f"common-prior d_norm={common:.3f} "
            f"(delta={common - nominal_common:+.3f})"
        )

    if result.conditional_states:
        print("\nConditional/order channels")
        for (target, conditioning), state in sorted(result.conditional_states.items()):
            print(
                f"  C_{{{target}|{conditioning}}}: "
                f"events={len(state.event_times_s)}, "
                f"rate~{state.input_rate_hz:.3f} Hz, n_ST={state.n_st}, "
                f"nominal mean d_norm={nominal_mean(state.event_times_s, normalized_disbelief_series(state.disbelief_events, state.uncertainty_events)):.3f}, "
                f"nominal mean u={nominal_mean(state.event_times_s, state.uncertainty_events):.3f}"
            )
        if result.order_decomposition_error_12:
            print(
                "  batch/sequential NIS identity max errors: "
                f"1->2={max(result.order_decomposition_error_12):.3e}, "
                f"2->1={max(result.order_decomposition_error_21):.3e}"
            )

    if ENABLE_SENSOR_2_DROPOUT:
        a2 = result.availability_states[2]
        print("\nSensor-2 dropout")
        print(
            "  mean P(A2) during dropout: "
            f"{interval_mean(a2.event_times_s, a2.p_available_events, SENSOR_2_DROPOUT_INTERVAL_S):.3f}"
        )
        print(
            "  mean local C2 consistency uncertainty during dropout: "
            f"{interval_mean(result.isolated_states[2].event_times_s, result.isolated_states[2].uncertainty_events, SENSOR_2_DROPOUT_INTERVAL_S):.3f}"
        )
        print(
            "  mean track-output uncertainty during dropout: "
            f"{interval_mean(result.event_times_s, [uncertainty(op) for op in result.track_output_trust_history], SENSOR_2_DROPOUT_INTERVAL_S):.3f}"
        )

    print("\nInterpretation reminder")
    print("  d_norm=d/(1-u): normalized inconsistency within committed evidence")
    print("  C_s^iso      : sensor/path consistency using only that sensor history")
    print("  C_s^common   : same PIT/TEF mapping but central common prior")
    print("  A_s           : expected output availability")
    print("  A_s (*) C_s   : availability-trust-discounted interpretation")
    print("  C_{s|empty}^{sim}: matched simultaneous common-prior reference")
    print("  C_{s|j}       : same simultaneous channel after virtual update with sensor j")
    print("  I_{j->s}      : conditional update-order influence; not fault attribution")
    print("  D_12          : direct pair agreement; d_norm quantifies disagreement")
    print("  C_F           : direct consistency of the actually used central measurement batch")
    print("  H_s=A_s AND C_s: optional operational sensor-path diagnostic")
    print("  sensor support : availability-only system requirement (ANY/ALL)")
    print("  track trust    : trust discount of C_F by availability support")


def main() -> None:
    print("=" * 88)
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        print("V3 SYNCHRONOUS SPECIAL CASE: Sensor 1 = Sensor 2 = 10 Hz")
        if ENABLE_SENSOR_2_DROPOUT:
            print(
                "NOTE: Sensor-2 dropout is enabled. Disable it for the pure "
                "synchronous limiting-case reproduction."
            )
    elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
        print(
            "V3 ASYNCHRONOUS CROSS-CONTAMINATION STRESS TEST: "
            f"disturbed Sensor 1 = {SENSOR_1_RATE_HZ:g} Hz, "
            f"nominal Sensor 2 = {SENSOR_2_RATE_HZ:g} Hz"
        )
    else:
        print(f"V3 ASYNCHRONOUS MULTI-RATE CASE: Sensor 1 = {SENSOR_1_RATE_HZ:g} Hz, Sensor 2 = {SENSOR_2_RATE_HZ:g} Hz")
    print(f"Filter motion model: {'CT' if USE_CT_MODEL else 'CV'}")
    print("=" * 88)

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