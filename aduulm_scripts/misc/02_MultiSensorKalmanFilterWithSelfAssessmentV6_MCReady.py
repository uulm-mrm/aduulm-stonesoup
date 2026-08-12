#!/usr/bin/env python3
"""
02 V8 - Dynamic N-sensor Kalman filter with PIT/TEF self-assessment.

Purpose
-------
This script is a test bed for the asynchronous multi-sensor extension of the
PIT/Subjective-Logic self-assessment approach. It keeps the Stone Soup
simulation and the external ``subjective_logic`` implementation, while
separating the assessment into semantically distinct opinions:

1. Sensor-specific isolated consistency C_s
   - one sensor-only shadow KF per sensor,
   - radial NIS + whitened x/y innovation channels,
   - PIT -> TEF -> binomial consistency opinions.

2. Common-prediction PIT/TEF baseline C_s^common
   - the same PIT/TEF mapping as C_s,
   - but each active sensor is assessed against the functional central KF prior,
   - retained as an architectural cross-contamination / Griebel comparison.

3. Availability A_s
   - one scheduled-output opinion per sensor,
   - evidence [1,0] when an expected measurement arrives and [0,1] when it does
     not arrive,
   - if an expected measurement is missing, consistency TEFs are FROZEN:
     missingness is represented only by A_s.

4. Availability-trust-discounted isolated consistency (diagnostic only)
       C~_s = A_s (*) C_s^iso
   - retained to inspect each local path under its own availability,
   - it is deliberately NOT fused into the base track-consistency opinion.

5. Pairwise sensor-agreement matrix G=[G_ij]
   - one direct sensor-to-sensor opinion is generated for every unique pair i<j,
   - Delta z = z_i-z_j removes the common state for equal measurement models,
   - under independent nominal measurement noise, Delta z ~ N(0,R_i+R_j),
   - the lower triangle is redundant because G_ij = G_ji.

6. Base, strict, and pair-conditioned track-consistency branches
       omega_B      = WBF(C_1^iso, ..., C_N^iso, C_F)
       omega_strict = AND(C_1^iso, ..., C_N^iso, C_F)
       omega_C,ij   = Deduction(G_ij; omega_B, omega_strict)
       omega_C      = WBF({omega_C,ij}_{i<j})
   - every unique matrix entry performs its own deduction,
   - the pair-conditioned opinions remain available for fault-source isolation.

7. Direct central-filter batch consistency C_F
   - batch NIS of the actually active measurement set against the central prior,
   - the chi-square degrees of freedom change with the active set,
   - PIT maps every valid null distribution to the same U(0,1) evidence domain.

8. Availability, track trust, and system health
       omega_A = ABF(A_1, ..., A_N)
       omega_T = trust_discount(omega_A, omega_C)
       omega_H = omega_C * omega_A
   - availability remains separate from statistical consistency,
   - a dropout therefore raises missing-information uncertainty rather than
     manufacturing statistical inconsistency.

9. Evaluation structure
   - Evaluation A retains native Griebel common-prior single-sensor SA only as
     the source-isolation / cross-contamination reference.
   - Evaluation B shows the proposed overall opinion path itself.
   - Evaluation C uses availability, central-batch information and the G_ij /
     omega_C,ij matrices for system-level reasoning and fault-source isolation.

Temporal parametrisation
------------------------
TEF horizons are specified in physical time and converted to sample counts
using the corresponding assessment-event rate:

    local/common sensor consistency horizon = 5.0 s
    direct central-batch horizon            = 5.0 s
    availability horizon                    = 1.0 s
    sensor-pair disagreement horizon        = 5.0 s

Thus n_ST ~= T * f for the event stream feeding the respective TEF. The
long-term discount is rate-normalised as well.

Scope assumptions
-----------------
- P_D = 1 whenever a sensor is expected to measure the tracked object.
- Measurements are already correctly associated to the track.
- No clutter, uncertain detection process, track initiation/deletion, or OOSM.
- The sensor-to-sensor channel requires a simultaneous pair and freezes when no
  current pair exists.
"""


from __future__ import annotations

import csv
import json
import os
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
# Monte-Carlo drivers can request a non-interactive backend before importing this module.
matplotlib.use("Agg" if os.environ.get("SELFASSESSMENT_MC_HEADLESS") == "1" else "TkAgg")
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

# Number of configured measurement sensors.  The complete SA architecture below
# is generated dynamically for this number of sensors.  N=2 reproduces the old
# pairwise special case; N>=3 additionally yields an agreement/deduction matrix.
NUM_SENSORS = 4
if NUM_SENSORS < 1:
    raise ValueError("NUM_SENSORS must be >= 1")

# False: asynchronous/multi-rate case.
# True: all configured sensors are sampled synchronously on the reference grid.
SYNCHRONOUS_SENSOR_SPECIAL_CASE = True
SYNCHRONOUS_REFERENCE_RATE_HZ = 10.0

# In the generic asynchronous case the rates are assigned cyclically from this
# pattern.  Every rate must be exactly representable on TRUTH_RATE_HZ below.
ASYNC_SENSOR_RATE_PATTERN_HZ = (5.0, 10.0, 12.5, 20.0)
# Optional explicit rates in asynchronous mode, e.g. {1: 10.0, 2: 20.0, 3: 25.0}.
SENSOR_RATE_OVERRIDES_HZ: dict[int, float] = {}

# Optional dedicated stress test for cross-source prior contamination.  Sensor 1
# is faster than the remaining sensors, so its faulty updates can alter the
# common central prior before the next nominal-sensor update.
CROSS_CONTAMINATION_RATE_STRESS_TEST = False
CROSS_CONTAMINATION_DISTURBED_SENSOR_ID = 1
CROSS_CONTAMINATION_DISTURBED_RATE_HZ = 25.0
CROSS_CONTAMINATION_OTHER_RATE_HZ = 10.0

# Disturbance assignment.  The existing measurement disturbances (outliers,
# bias, R mismatch, non-Gaussian noise) are applied to every sensor ID contained
# here.  The default keeps the previous Sensor-1-only experiment.
DISTURBED_SENSOR_IDS = {2}

# Availability/dropout is a separate proposition.  The selected sensor is
# removed only during SENSOR_DROPOUT_INTERVAL_S; all consistency channels freeze
# when its measurement is missing.
ENABLE_SENSOR_DROPOUT = True
DROPOUT_SENSOR_ID = 2

# Static plots are fully N-sensor aware. The dynamic Plotly dashboard is generated
# for up to four sensors; for N>4 it is skipped to keep the dashboard readable.
SHOW_DYNAMIC_ANIMATION = True
SHOW_MATPLOTLIB_PLOTS = True
SHOW_POSITION_ERROR = True
SHOW_PAIRWISE_TIME_SERIES = NUM_SENSORS <= 4
SHOW_PAIRWISE_MATRIX_FIGURE = True
# Dedicated per-pair trajectory figure: one subplot for every unique i<j pair,
# showing the selected normalized score of G_ij and omega_C,ij.
SHOW_PAIRWISE_DEDUCTION_TRAJECTORIES = True
PAIR_TRAJECTORY_MAX_COLUMNS = 3

# Normalized committed-mass score used in all plots/diagnostic summaries.
# False -> d_norm = d/(1-u): high values mean inconsistency/disagreement.
# True  -> b_norm = b/(1-u): high values mean consistency/agreement.
# For every non-vacuous binomial opinion, b_norm = 1 - d_norm exactly.
USE_NORMALIZED_BELIEF = True

# None -> automatically use the midpoint of the Sensor-1 bias interval.
PAIR_MATRIX_SNAPSHOT_TIME_S = None

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

# Fine-grained disturbance switches used by the Monte-Carlo driver.  Defaults
# preserve the interactive all-disturbances scenario of V8.3.  The optional
# per-fault sensor sets fall back to DISTURBED_SENSOR_IDS when left as None.
ENABLE_OUTLIER_DISTURBANCE = True
ENABLE_BIAS_DISTURBANCE = True
ENABLE_MEASUREMENT_NOISE_DISTURBANCE = True
ENABLE_NON_GAUSSIAN_DISTURBANCE = True
ENABLE_PROCESS_NOISE_DISTURBANCE = True

OUTLIER_SENSOR_IDS: set[int] | None = None
BIAS_SENSOR_IDS: set[int] | None = None
MEASUREMENT_NOISE_SENSOR_IDS: set[int] | None = None
NON_GAUSSIAN_SENSOR_IDS: set[int] | None = None

# Multiplicative process-noise disturbance used in generate_truth().
PROCESS_NOISE_INCREASE_FACTOR = 32.0

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

# In asynchronous mode the published 2023 setup does not define a unique
# event-varying active-set semantics.  We therefore compare explicitly labelled
# straightforward extensions rather than calling any of them the published
# method:
#
# - "active_only" (recommended/default):
#     assess every currently active sensor and ABF-fuse only the current active
#     source set.  This avoids stale information and is the fairest best-effort
#     asynchronous extension.
#
# - "hold_last":
#     assess every active sensor but ABF-fuse the most recent assessment of all
#     configured sensors.  This deliberately exposes stale-source semantics and
#     is useful for rate-stress/dropout experiments.
#
# - "synchronous_only":
#     update the reference only on timestamps at which all configured sensors
#     provide a current measurement.
GRIEBEL_ASYNC_EXTENSION_MODE = "active_only"

# Evaluation B uses Griebel's *pre-threshold* multinomial source opinions.
# get_sas_measures() itself exposes only (delta, uncertainty, eta), therefore
# the adapter below reads KalmanSelfAssessor._op_st directly and validates its
# numerical components by reproducing the returned delta/u before ABF.  No
# opinion is reconstructed from a hard decision and
# no projected-probability comparison is used.

# -----------------------------------------------------------------------------
# Evaluation layout
# -----------------------------------------------------------------------------
# Evaluation A: common-prior/Griebel source assessment vs isolated sensor paths.
# Evaluation B: the proposed overall opinion path itself (omega_B -> omega_C).
# Evaluation C: availability, system-level reasoning and pairwise fault isolation.
# No AUROC/TPR/decision-threshold evaluation is generated in this version.

# Output-level construction is hierarchical and separates inconsistency
# evidence from missing-information uncertainty:
#
#   omega_B      = WBF(C_1^iso, C_2^iso, C_F)
#   omega_strict = AND(C_1^iso, ..., C_N^iso, C_F)
#   omega_C      = Deduction(
#                      G_12;
#                      omega_{C|G_12}     = omega_B,
#                      omega_{C|not G_12} = omega_strict
#                  )
#   omega_A      = ABF(A_1, A_2)
#   omega_T      = trust_discount(omega_A, omega_C)
#
# Rationale:
# - Under normal pair agreement, WBF avoids the pessimism of always applying
#   conjunction to several merely supportive consistency opinions.
# - Under pair disagreement, deduction switches towards a stricter requirement:
#   all local/central consistency propositions must support the track.
# - Availability is applied only afterwards as reliability trust, so dropout
#   moves committed mass to uncertainty rather than manufacturing disbelief.
# - All temporal assessment families use the same physical T_ST=5 s; rate
#   differences are handled by time-normalised sample counts/discounts.

SCENARIO_DURATION_S = 160.0
TRUTH_RATE_HZ = 100.0  # exactly represents 5, 10, 12.5, 20 and 25 Hz
TRUTH_DT_S = 1.0 / TRUTH_RATE_HZ

Q_X = 0.25
Q_Y = 0.25
MEASUREMENT_VARIANCE = 0.1

NUM_PIT_BINS = 7

# -----------------------------------------------------------------------------
# Time-normalised TEF settings
# -----------------------------------------------------------------------------
UNIFORM_TIME_HORIZON = 5.0
# Local sensor-consistency channels assess the PIT distribution over a
# short physical history.  The number of TEF samples is derived from the
# actual sensor rate.
CONSISTENCY_SHORT_TERM_HORIZON_S = UNIFORM_TIME_HORIZON

# The same physical short-term horizon is used across consistency families.
# Different event rates are handled by rate-normalised n_ST values.
BATCH_SHORT_TERM_HORIZON_S = UNIFORM_TIME_HORIZON

# Availability is a distinct proposition but uses the same physical horizon so
# that output-level opinions refer to comparable recent time spans.
AVAILABILITY_SHORT_TERM_HORIZON_S = 1.0

# Pair agreement/disagreement is updated only at simultaneous sensor
# timestamps. The same physical horizon is used for comparability; its sample
# count is derived from the actual simultaneous-event rate.
DISAGREEMENT_SHORT_TERM_HORIZON_S = UNIFORM_TIME_HORIZON

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

ASSUME_DETECTION_PROBABILITY_ONE = True
NOMINAL_BURN_IN_S = CONSISTENCY_SHORT_TERM_HORIZON_S
SCRIPT_DIR = Path(__file__).resolve().parent

RANDOM_SEED_TRUTH = 0
RANDOM_SEED_SENSOR_BASE = 100


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
MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR = 2.0

# Explicit 10 s nominal gap before and after the dropout:
#   previous disturbance ends at 60 s,
#   dropout starts at 70 s and ends at 80 s,
#   next disturbance starts at 90 s.
SENSOR_DROPOUT_INTERVAL_S = (70.0, 80.0)

# Shape-only measurement-noise disturbance for Sensor 1.
# Both selectable variants preserve the nominal first two moments
#
#     E[v] = 0,    Cov[v] = R,
#
# while violating the assumed single-Gaussian distribution.
#
# False -> symmetric bimodal Gaussian mixture
# True  -> contaminated/heavy-tailed Gaussian mixture
USE_HEAVY_TAILED_NON_GAUSSIAN = True

VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S = (90.0, 100.0)

# --- Variant A: symmetric bimodal Gaussian mixture -------------------------
# Standardized component:
#   u ~ 0.5*N(-mu_b, sigma_b^2) + 0.5*N(+mu_b, sigma_b^2)
# with mu_b^2 + sigma_b^2 = 1.
BIMODAL_MODE_OFFSET_STD = 0.90
BIMODAL_WITHIN_MODE_STD = float(
    np.sqrt(1.0 - BIMODAL_MODE_OFFSET_STD ** 2)
)

# --- Variant B: contaminated/heavy-tailed Gaussian mixture -----------------
# Standardized component:
#   u ~ p_core*N(0, sigma_core^2)
#      + (1-p_core)*N(0, sigma_tail^2)
#
# sigma_tail is chosen so that Var[u] = 1 exactly in the population.
# With the values below:
#   90% N(0, 0.5^2) + 10% N(0, 2.7839^2).
HEAVY_TAIL_CORE_PROBABILITY = 0.90
HEAVY_TAIL_CORE_STD = 0.20
HEAVY_TAIL_TAIL_STD = float(
    np.sqrt(
        (
            1.0
            - HEAVY_TAIL_CORE_PROBABILITY * HEAVY_TAIL_CORE_STD ** 2
        )
        / (1.0 - HEAVY_TAIL_CORE_PROBABILITY)
    )
)

NON_GAUSSIAN_DISTURBANCE_LABEL = (
    "S1 variance-matched heavy-tailed mixture"
    if USE_HEAVY_TAILED_NON_GAUSSIAN
    else "S1 variance-matched bimodal mixture"
)
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
    (*SENSOR_DROPOUT_INTERVAL_S, f"S{DROPOUT_SENSOR_ID} unavailable", "tab:gray"),
    (
        *VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S,
        NON_GAUSSIAN_DISTURBANCE_LABEL,
        "tab:purple",
    ),
    (*TURN_INTERVAL_S, "common motion-model mismatch", "tab:green"),
    (*INCREASED_PROCESS_XY_INTERVAL_S, "common increased x/y process noise", "tab:brown"),
]


def configured_sensor_rate_hz(sensor_id: int) -> float:
    """Return the configured nominal/actual rate for one sensor ID."""
    if not 1 <= int(sensor_id) <= NUM_SENSORS:
        raise ValueError(f"invalid sensor_id={sensor_id} for N={NUM_SENSORS}")
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        return float(SYNCHRONOUS_REFERENCE_RATE_HZ)
    if CROSS_CONTAMINATION_RATE_STRESS_TEST:
        return float(
            CROSS_CONTAMINATION_DISTURBED_RATE_HZ
            if sensor_id == CROSS_CONTAMINATION_DISTURBED_SENSOR_ID
            else CROSS_CONTAMINATION_OTHER_RATE_HZ
        )
    if sensor_id in SENSOR_RATE_OVERRIDES_HZ:
        return float(SENSOR_RATE_OVERRIDES_HZ[sensor_id])
    return float(ASYNC_SENSOR_RATE_PATTERN_HZ[(sensor_id - 1) % len(ASYNC_SENSOR_RATE_PATTERN_HZ)])


def configured_sensor_colour(sensor_id: int) -> str:
    """Stable Matplotlib colour for arbitrary sensor counts."""
    palette = (
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    )
    return palette[(sensor_id - 1) % len(palette)]


def configured_sensor_rates() -> dict[int, float]:
    return {sensor_id: configured_sensor_rate_hz(sensor_id) for sensor_id in range(1, NUM_SENSORS + 1)}


# Backwards-compatible aliases used only by the legacy two-sensor animation.
SENSOR_1_RATE_HZ = configured_sensor_rate_hz(1) if NUM_SENSORS >= 1 else SYNCHRONOUS_REFERENCE_RATE_HZ
SENSOR_2_RATE_HZ = configured_sensor_rate_hz(2) if NUM_SENSORS >= 2 else SENSOR_1_RATE_HZ
DISTURB_SENSOR_1 = 1 in DISTURBED_SENSOR_IDS
DISTURB_SENSOR_2 = 2 in DISTURBED_SENSOR_IDS
ENABLE_SENSOR_2_DROPOUT = ENABLE_SENSOR_DROPOUT and DROPOUT_SENSOR_ID == 2 and NUM_SENSORS >= 2
SENSOR_2_DROPOUT_INTERVAL_S = SENSOR_DROPOUT_INTERVAL_S
RANDOM_SEED_SENSOR_1 = RANDOM_SEED_SENSOR_BASE + 1
RANDOM_SEED_SENSOR_2 = RANDOM_SEED_SENSOR_BASE + 2


def effective_fault_sensor_ids(specific_ids: set[int] | None) -> set[int]:
    """Return validated sensor IDs for one measurement-fault family."""
    source = DISTURBED_SENSOR_IDS if specific_ids is None else specific_ids
    return {
        int(sensor_id)
        for sensor_id in source
        if 1 <= int(sensor_id) <= int(NUM_SENSORS)
    }


def sensor_has_any_measurement_disturbance(sensor_id: int) -> bool:
    """Whether any enabled measurement disturbance targets ``sensor_id``."""
    if not ACTIVATE_DISTURBANCES:
        return False
    checks = (
        (ENABLE_OUTLIER_DISTURBANCE, OUTLIER_SENSOR_IDS),
        (ENABLE_BIAS_DISTURBANCE, BIAS_SENSOR_IDS),
        (ENABLE_MEASUREMENT_NOISE_DISTURBANCE, MEASUREMENT_NOISE_SENSOR_IDS),
        (ENABLE_NON_GAUSSIAN_DISTURBANCE, NON_GAUSSIAN_SENSOR_IDS),
    )
    return any(
        enabled and int(sensor_id) in effective_fault_sensor_ids(sensor_ids)
        for enabled, sensor_ids in checks
    )


def _sensor_id_label(sensor_ids: set[int]) -> str:
    if not sensor_ids:
        return "none"
    return "/".join(f"S{sensor_id}" for sensor_id in sorted(sensor_ids))


def rebuild_disturbance_intervals() -> list[tuple[float, float, str, str]]:
    """Rebuild plot/evaluation intervals from the currently active scenario."""
    intervals: list[tuple[float, float, str, str]] = []
    if ACTIVATE_DISTURBANCES and ENABLE_OUTLIER_DISTURBANCE:
        ids = effective_fault_sensor_ids(OUTLIER_SENSOR_IDS)
        if ids:
            intervals.append((*OUTLIER_INTERVAL_S, f"{_sensor_id_label(ids)} outliers", "tab:red"))
    if ACTIVATE_DISTURBANCES and ENABLE_BIAS_DISTURBANCE:
        ids = effective_fault_sensor_ids(BIAS_SENSOR_IDS)
        if ids:
            intervals.append((
                *SENSOR_1_BIAS_INTERVAL_S,
                f"{_sensor_id_label(ids)} + {SENSOR_1_BIAS_VECTOR_M[0]:g}m x-bias",
                "tab:orange",
            ))
    if ACTIVATE_DISTURBANCES and ENABLE_MEASUREMENT_NOISE_DISTURBANCE:
        ids = effective_fault_sensor_ids(MEASUREMENT_NOISE_SENSOR_IDS)
        if ids:
            intervals.append((
                *INCREASED_MEAS_XY_INTERVAL_S,
                f"{_sensor_id_label(ids)} increased x/y-noise (R x{MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR:g})",
                "tab:blue",
            ))
    if ENABLE_SENSOR_DROPOUT and 1 <= int(DROPOUT_SENSOR_ID) <= int(NUM_SENSORS):
        intervals.append((
            *SENSOR_DROPOUT_INTERVAL_S,
            f"S{DROPOUT_SENSOR_ID} unavailable",
            "tab:gray",
        ))
    if ACTIVATE_DISTURBANCES and ENABLE_NON_GAUSSIAN_DISTURBANCE:
        ids = effective_fault_sensor_ids(NON_GAUSSIAN_SENSOR_IDS)
        if ids:
            kind = "variance-matched heavy-tailed mixture" if USE_HEAVY_TAILED_NON_GAUSSIAN else "variance-matched bimodal mixture"
            intervals.append((
                *VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S,
                f"{_sensor_id_label(ids)} {kind}",
                "tab:purple",
            ))
    if ENABLE_GROUND_TRUTH_TURN:
        intervals.append((*TURN_INTERVAL_S, "common motion-model mismatch", "tab:green"))
    if ACTIVATE_DISTURBANCES and ENABLE_PROCESS_NOISE_DISTURBANCE:
        intervals.append((
            *INCREASED_PROCESS_XY_INTERVAL_S,
            f"common increased x/y process noise (Q x{PROCESS_NOISE_INCREASE_FACTOR:g})",
            "tab:brown",
        ))
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    return intervals


def refresh_derived_configuration() -> None:
    """Recompute values derived from user/Monte-Carlo configuration globals."""
    global TRUTH_DT_S
    global BIMODAL_WITHIN_MODE_STD, HEAVY_TAIL_TAIL_STD
    global NON_GAUSSIAN_DISTURBANCE_LABEL, DISTURBANCE_INTERVALS
    global SENSOR_1_RATE_HZ, SENSOR_2_RATE_HZ
    global DISTURB_SENSOR_1, DISTURB_SENSOR_2
    global ENABLE_SENSOR_2_DROPOUT, SENSOR_2_DROPOUT_INTERVAL_S
    global RANDOM_SEED_SENSOR_1, RANDOM_SEED_SENSOR_2
    global NOMINAL_BURN_IN_S

    if NUM_SENSORS < 1:
        raise ValueError("NUM_SENSORS must be >= 1")
    TRUTH_DT_S = 1.0 / float(TRUTH_RATE_HZ)
    BIMODAL_WITHIN_MODE_STD = float(np.sqrt(max(0.0, 1.0 - BIMODAL_MODE_OFFSET_STD ** 2)))
    if not 0.0 < HEAVY_TAIL_CORE_PROBABILITY < 1.0:
        raise ValueError("HEAVY_TAIL_CORE_PROBABILITY must lie in (0, 1)")
    tail_variance = (
        1.0 - HEAVY_TAIL_CORE_PROBABILITY * HEAVY_TAIL_CORE_STD ** 2
    ) / (1.0 - HEAVY_TAIL_CORE_PROBABILITY)
    if tail_variance <= 0.0:
        raise ValueError("heavy-tail parameters imply a non-positive tail variance")
    HEAVY_TAIL_TAIL_STD = float(np.sqrt(tail_variance))
    ids = effective_fault_sensor_ids(NON_GAUSSIAN_SENSOR_IDS)
    kind = "variance-matched heavy-tailed mixture" if USE_HEAVY_TAILED_NON_GAUSSIAN else "variance-matched bimodal mixture"
    NON_GAUSSIAN_DISTURBANCE_LABEL = f"{_sensor_id_label(ids)} {kind}"
    DISTURBANCE_INTERVALS = rebuild_disturbance_intervals()

    SENSOR_1_RATE_HZ = configured_sensor_rate_hz(1) if NUM_SENSORS >= 1 else SYNCHRONOUS_REFERENCE_RATE_HZ
    SENSOR_2_RATE_HZ = configured_sensor_rate_hz(2) if NUM_SENSORS >= 2 else SENSOR_1_RATE_HZ
    DISTURB_SENSOR_1 = sensor_has_any_measurement_disturbance(1) if NUM_SENSORS >= 1 else False
    DISTURB_SENSOR_2 = sensor_has_any_measurement_disturbance(2) if NUM_SENSORS >= 2 else False
    ENABLE_SENSOR_2_DROPOUT = ENABLE_SENSOR_DROPOUT and DROPOUT_SENSOR_ID == 2 and NUM_SENSORS >= 2
    SENSOR_2_DROPOUT_INTERVAL_S = SENSOR_DROPOUT_INTERVAL_S
    RANDOM_SEED_SENSOR_1 = RANDOM_SEED_SENSOR_BASE + 1
    RANDOM_SEED_SENSOR_2 = RANDOM_SEED_SENSOR_BASE + 2
    NOMINAL_BURN_IN_S = CONSISTENCY_SHORT_TERM_HORIZON_S


MONTE_CARLO_CONFIG_KEYS = (
    "NUM_SENSORS",
    "SYNCHRONOUS_SENSOR_SPECIAL_CASE",
    "SYNCHRONOUS_REFERENCE_RATE_HZ",
    "ASYNC_SENSOR_RATE_PATTERN_HZ",
    "SENSOR_RATE_OVERRIDES_HZ",
    "CROSS_CONTAMINATION_RATE_STRESS_TEST",
    "CROSS_CONTAMINATION_DISTURBED_SENSOR_ID",
    "CROSS_CONTAMINATION_DISTURBED_RATE_HZ",
    "CROSS_CONTAMINATION_OTHER_RATE_HZ",
    "DISTURBED_SENSOR_IDS",
    "ENABLE_SENSOR_DROPOUT",
    "DROPOUT_SENSOR_ID",
    "USE_NORMALIZED_BELIEF",
    "ACTIVATE_DISTURBANCES",
    "ENABLE_OUTLIER_DISTURBANCE",
    "ENABLE_BIAS_DISTURBANCE",
    "ENABLE_MEASUREMENT_NOISE_DISTURBANCE",
    "ENABLE_NON_GAUSSIAN_DISTURBANCE",
    "ENABLE_PROCESS_NOISE_DISTURBANCE",
    "OUTLIER_SENSOR_IDS",
    "BIAS_SENSOR_IDS",
    "MEASUREMENT_NOISE_SENSOR_IDS",
    "NON_GAUSSIAN_SENSOR_IDS",
    "PROCESS_NOISE_INCREASE_FACTOR",
    "USE_CT_MODEL",
    "CT_TURN_RATE_NOISE",
    "ENABLE_GROUND_TRUTH_TURN",
    "ENABLE_GRIEBEL_REFERENCE",
    "GRIEBEL_BACKEND",
    "GRIEBEL_ASYNC_EXTENSION_MODE",
    "SCENARIO_DURATION_S",
    "TRUTH_RATE_HZ",
    "Q_X",
    "Q_Y",
    "MEASUREMENT_VARIANCE",
    "NUM_PIT_BINS",
    "CONSISTENCY_SHORT_TERM_HORIZON_S",
    "BATCH_SHORT_TERM_HORIZON_S",
    "AVAILABILITY_SHORT_TERM_HORIZON_S",
    "DISAGREEMENT_SHORT_TERM_HORIZON_S",
    "REFERENCE_RATE_HZ",
    "CONSISTENCY_REFERENCE_LONG_TERM_DISCOUNT",
    "AVAILABILITY_REFERENCE_LONG_TERM_DISCOUNT",
    "ALPHA_THRESHOLD_DC",
    "HANDLE_SHORT_TERM_CONFLICT",
    "AVERAGE_DC_CONFLICT_HANDLING",
    "GRIEBEL_NUM_X",
    "GRIEBEL_N_ST",
    "GRIEBEL_N_C",
    "GRIEBEL_ALPHA_THRESHOLD_DC",
    "GRIEBEL_TRUST_DISCOUNT",
    "OUTLIER_INTERVAL_S",
    "SENSOR_1_BIAS_INTERVAL_S",
    "SENSOR_1_BIAS_VECTOR_M",
    "INCREASED_MEAS_XY_INTERVAL_S",
    "MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR",
    "SENSOR_DROPOUT_INTERVAL_S",
    "USE_HEAVY_TAILED_NON_GAUSSIAN",
    "VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S",
    "BIMODAL_MODE_OFFSET_STD",
    "HEAVY_TAIL_CORE_PROBABILITY",
    "HEAVY_TAIL_CORE_STD",
    "TURN_INTERVAL_S",
    "INCREASED_PROCESS_XY_INTERVAL_S",
    "RANDOM_SEED_TRUTH",
    "RANDOM_SEED_SENSOR_BASE",
)


def configuration_snapshot() -> dict[str, object]:
    """Return a copyable snapshot of all supported Monte-Carlo parameters."""
    snapshot: dict[str, object] = {}
    namespace = globals()
    for key in MONTE_CARLO_CONFIG_KEYS:
        value = namespace[key]
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, set):
            value = set(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, tuple):
            value = tuple(value)
        snapshot[key] = value
    return snapshot


def apply_configuration(overrides: dict[str, object]) -> None:
    """Apply a configuration dictionary and refresh all derived globals."""
    namespace = globals()
    unknown = sorted(set(overrides) - set(MONTE_CARLO_CONFIG_KEYS))
    if unknown:
        raise KeyError(f"unsupported Monte-Carlo configuration keys: {unknown}")
    set_valued = {
        "DISTURBED_SENSOR_IDS",
        "OUTLIER_SENSOR_IDS",
        "BIAS_SENSOR_IDS",
        "MEASUREMENT_NOISE_SENSOR_IDS",
        "NON_GAUSSIAN_SENSOR_IDS",
    }
    tuple_valued = {
        "ASYNC_SENSOR_RATE_PATTERN_HZ",
        "OUTLIER_INTERVAL_S",
        "SENSOR_1_BIAS_INTERVAL_S",
        "INCREASED_MEAS_XY_INTERVAL_S",
        "SENSOR_DROPOUT_INTERVAL_S",
        "VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S",
        "TURN_INTERVAL_S",
        "INCREASED_PROCESS_XY_INTERVAL_S",
    }
    for key, value in overrides.items():
        if key in set_valued and value is not None:
            value = set(value)
        elif key in tuple_valued:
            value = tuple(value)
        elif key == "SENSOR_RATE_OVERRIDES_HZ":
            value = {int(k): float(v) for k, v in dict(value).items()}
        elif key == "SENSOR_1_BIAS_VECTOR_M":
            value = np.asarray(value, dtype=float).reshape(2)
        namespace[key] = value
    refresh_derived_configuration()


def set_run_seeds(truth_seed: int, sensor_seed_base: int) -> None:
    """Set independent deterministic seeds for one Monte-Carlo run."""
    global RANDOM_SEED_TRUTH, RANDOM_SEED_SENSOR_BASE
    RANDOM_SEED_TRUTH = int(truth_seed)
    RANDOM_SEED_SENSOR_BASE = int(sensor_seed_base)
    refresh_derived_configuration()


# Ensure the import-time defaults and aliases are internally consistent.
refresh_derived_configuration()


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


def sample_variance_matched_bimodal_noise_from_cov(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample variance-matched symmetric bimodal Gaussian-mixture noise.

    Each standardized component follows

        0.5*N(-mu_b, sigma_b^2) + 0.5*N(+mu_b, sigma_b^2)

    with mu_b^2 + sigma_b^2 = 1. Therefore every component has zero mean
    and unit variance while being distinctly non-Gaussian/bimodal.

    Independent mode signs are used per component, so the standardized
    covariance remains I. With L L^T = covariance and v = L u:

        E[v] = 0,   Cov[v] = covariance.

    Thus only the distributional shape assumption is violated; the filter's
    nominal mean and R matrix remain correct.
    """
    covariance = np.asarray(covariance, dtype=float)
    chol = robust_cholesky(covariance)

    dimension = covariance.shape[0]
    mode_sign = rng.choice(
        np.array([-1.0, 1.0]),
        size=(dimension, 1),
    )
    within_mode_noise = rng.normal(
        loc=0.0,
        scale=BIMODAL_WITHIN_MODE_STD,
        size=(dimension, 1),
    )
    standardized_noise = (
        mode_sign * BIMODAL_MODE_OFFSET_STD
        + within_mode_noise
    )

    return chol @ standardized_noise


def sample_variance_matched_heavy_tailed_noise_from_cov(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a variance-matched contaminated Gaussian mixture.

    Each standardized component independently follows

        p_core*N(0, sigma_core^2)
        + (1-p_core)*N(0, sigma_tail^2),

    with sigma_tail selected so that the population variance is exactly one.

    Hence v = chol(R) @ u has zero mean and covariance R, but its distribution
    is leptokurtic/heavy-tailed. This is a stationary alternative noise model
    over the complete disturbance interval, unlike the separate sporadic
    outlier disturbance.
    """
    covariance = np.asarray(covariance, dtype=float)
    chol = robust_cholesky(covariance)

    dimension = covariance.shape[0]
    use_core = (
        rng.random(size=(dimension, 1))
        < HEAVY_TAIL_CORE_PROBABILITY
    )
    component_std = np.where(
        use_core,
        HEAVY_TAIL_CORE_STD,
        HEAVY_TAIL_TAIL_STD,
    )
    standardized_noise = rng.normal(
        loc=0.0,
        scale=component_std,
        size=(dimension, 1),
    )
    return chol @ standardized_noise


def sample_variance_matched_non_gaussian_noise_from_cov(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the selected variance-matched non-Gaussian variant."""
    if USE_HEAVY_TAILED_NON_GAUSSIAN:
        return sample_variance_matched_heavy_tailed_noise_from_cov(
            covariance,
            rng,
        )

    return sample_variance_matched_bimodal_noise_from_cov(
        covariance,
        rng,
    )


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
    evidence exists. NaN is returned deliberately.
    """
    committed_mass = 1.0 - float(uncertainty_value)
    if committed_mass <= eps:
        return float("nan")
    return float(np.clip(float(disbelief_value) / committed_mass, 0.0, 1.0))


def normalized_belief_from_components(
    belief_value: float,
    uncertainty_value: float,
    eps: float = 1e-12,
) -> float:
    """Return b_norm = b / (1-u), i.e. consistency within committed mass.

    For a vacuous opinion (u ~= 1), b_norm is undefined because no committed
    evidence exists. For a valid non-vacuous binomial opinion,

        b_norm + d_norm = 1.
    """
    committed_mass = 1.0 - float(uncertainty_value)
    if committed_mass <= eps:
        return float("nan")
    return float(np.clip(float(belief_value) / committed_mass, 0.0, 1.0))


def normalized_disbelief(opinion) -> float:
    return normalized_disbelief_from_components(
        disbelief(opinion), uncertainty(opinion)
    )


def normalized_belief(opinion) -> float:
    return normalized_belief_from_components(
        belief(opinion), uncertainty(opinion)
    )


def normalized_disbelief_series(
    disbelief_values: Iterable[float],
    uncertainty_values: Iterable[float],
) -> list[float]:
    return [
        normalized_disbelief_from_components(d, u)
        for d, u in zip(disbelief_values, uncertainty_values)
    ]


def normalized_belief_series(
    belief_values: Iterable[float],
    uncertainty_values: Iterable[float],
) -> list[float]:
    return [
        normalized_belief_from_components(b, u)
        for b, u in zip(belief_values, uncertainty_values)
    ]


def normalized_score(opinion) -> float:
    """Return the normalized score selected by USE_NORMALIZED_BELIEF."""
    return normalized_belief(opinion) if USE_NORMALIZED_BELIEF else normalized_disbelief(opinion)


def normalized_score_series(
    belief_values: Iterable[float],
    disbelief_values: Iterable[float],
    uncertainty_values: Iterable[float],
) -> list[float]:
    """Return b_norm or d_norm for a stored binomial-opinion time series."""
    if USE_NORMALIZED_BELIEF:
        return normalized_belief_series(belief_values, uncertainty_values)
    return normalized_disbelief_series(disbelief_values, uncertainty_values)


def normalized_score_symbol() -> str:
    return "b" if USE_NORMALIZED_BELIEF else "d"


def normalized_score_name() -> str:
    return "normalized belief" if USE_NORMALIZED_BELIEF else "normalized disbelief"


def normalized_score_semantics() -> str:
    if USE_NORMALIZED_BELIEF:
        return "0 = inconsistency / disagreement, 1 = nominal / agreement"
    return "0 = nominal / agreement, 1 = inconsistency / disagreement"


def normalized_score_cmap() -> str:
    # b_norm grows from bad->good, d_norm from good->bad.
    return "RdYlGn" if USE_NORMALIZED_BELIEF else "RdYlGn_r"


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


def fuse_cumulative(opinions: Iterable):
    """Cumulative belief fusion for independent evidence sources."""
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    if len(opinions) == 1:
        return deepcopy(opinions[0])
    return sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, opinions)


def logical_and(opinions: Iterable):
    """Conjoin distinct binomial propositions using SL multiplication.

    This is intentionally different from WBF: the inputs are logical
    requirements for a higher-level proposition rather than multiple sources
    assessing the same proposition.
    """
    opinions = list(opinions)
    if not opinions:
        return vacuous_binomial()
    result = deepcopy(opinions[0])
    for opinion in opinions[1:]:
        result = result.multiply(opinion)
    return result


def dogmatic_false_binomial(prior_ok_value: float = 0.5):
    """Return a dogmatic opinion that the proposition is false."""
    result = sl.Opinion2d(0.0, 1.0)
    result.prior_belief_masses = [prior_ok_value, 1.0 - prior_ok_value]
    return result


def subjective_logic_deduction(
    antecedent_opinion,
    consequent_if_true,
    consequent_if_false,
):
    """Use the external Subjective-Logic deduction operator.

    The Python binding exposes the bound method as
    antecedent.deduction(consequent_if_true, consequent_if_false).
    """
    try:
        return antecedent_opinion.deduction(
            consequent_if_true,
            consequent_if_false,
        )
    except TypeError as exc:
        raise RuntimeError(
            "subjective_logic deduction API mismatch. Expected "
            "Opinion2d.deduction(opinion_if_true, opinion_if_false)."
        ) from exc


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
        """Record a scheduled instant with no measurement while freezing C_s.

        A missing expected measurement provides no innovation and therefore no
        PIT sample.  The consistency TEFs are deliberately NOT advanced with a
        vacuous opinion: ``latest_opinion`` remains exactly at its last
        evidence-supported value.  Missingness is handled exclusively by the
        separate availability opinion A_s and, downstream, by trust discount.

        The current scheduled time is still recorded for plotting.  NIS/PIT are
        NaN because the corresponding statistics are undefined without a
        measurement.  ``latest_timestamp`` is intentionally left unchanged so
        it continues to denote the time of the most recent actual consistency
        evidence.
        """
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
        """Record a scheduled pair instant without changing disagreement evidence."""
        # No simultaneous measurement pair exists, so the pairwise diagnostic
        # has no new statistical evidence.  Freeze its TEFs/opinion and let
        # sensor availability carry the missing-information semantics.
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
        """Record an empty union event while freezing central consistency C_F."""
        # With no active measurement there is no batch innovation/NIS/PIT.
        # Therefore C_F receives no vacuous pseudo-observation and its TEF is
        # frozen.  Any loss of trust caused by absent sensors is represented by
        # availability support and the downstream trust discount.
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
            and np.isfinite(self.uncertainty)
        )

    @property
    def accepted(self) -> bool | None:
        if not self.valid:
            return None
        return bool(self.delta < self.eta)


class GriebelReferenceBackend:
    """Minimal adapter for Evaluation A: native common-prior single-sensor SA.

    The V8 N-sensor architecture deliberately does not construct a Griebel
    multi-source fusion.  Each native ``KalmanSelfAssessor`` simply receives the
    measurement prediction generated from the functional common central prior
    and exposes its public (delta, uncertainty, eta) output.  This is retained
    only as the common-prior reference for the source-isolation experiment.
    """

    def __init__(self, sensor_ids: Iterable[int], dim_meas: int = 2):
        self.sensor_ids = sorted(int(sensor_id) for sensor_id in sensor_ids)
        self.enabled = bool(ENABLE_GRIEBEL_REFERENCE)
        self.native = False
        self.status = "disabled"
        self.assessors: dict[int, object] = {}
        self.latest_measure = {sensor_id: GriebelMeasure() for sensor_id in self.sensor_ids}
        self.latest_timestamp: dict[int, datetime | None] = {sensor_id: None for sensor_id in self.sensor_ids}
        self.sensor_times = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_delta = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_eta = {sensor_id: [] for sensor_id in self.sensor_ids}
        self.sensor_uncertainty = {sensor_id: [] for sensor_id in self.sensor_ids}

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
                self.status = "native KalmanSelfAssessor (common-prior single-sensor reference)"
                return
            except Exception as exc:  # noqa: BLE001
                self.status = f"native init failed -> placeholder: {exc!r}"
                if requested == "native":
                    print("WARNING:", self.status)

        if requested == "placeholder":
            self.status = "placeholder requested"
        elif not GRIEBEL_NATIVE_IMPORT_AVAILABLE:
            self.status = (
                "placeholder; KalmanSelfAssessor import unavailable: "
                f"{GRIEBEL_NATIVE_IMPORT_ERROR}"
            )
        elif self.status == "disabled":
            self.status = "placeholder"

    @staticmethod
    def _validate_async_mode() -> str:
        mode = GRIEBEL_ASYNC_EXTENSION_MODE.lower()
        valid = {"active_only", "hold_last", "synchronous_only"}
        if mode not in valid:
            raise ValueError(
                "GRIEBEL_ASYNC_EXTENSION_MODE must be one of "
                f"{sorted(valid)}, got {GRIEBEL_ASYNC_EXTENSION_MODE!r}"
            )
        return mode

    def should_update_event(self, active_sensor_ids: list[int]) -> bool:
        if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
            return True
        mode = self._validate_async_mode()
        if mode in {"active_only", "hold_last"}:
            return True
        return set(active_sensor_ids) == set(self.sensor_ids)

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
        measure = (
            GriebelMeasure(
                delta=float(values[0]),
                uncertainty=float(values[1]),
                eta=float(values[2]),
            )
            if values.size >= 3
            else GriebelMeasure()
        )
        self.latest_measure[sensor_id] = measure
        self.latest_timestamp[sensor_id] = timestamp
        self.sensor_times[sensor_id].append((timestamp - start_time).total_seconds())
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
        return


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
    pair_event_rates_hz: dict[tuple[int, int], float]


def build_sensor_definitions() -> dict[int, SensorDefinition]:
    definitions: dict[int, SensorDefinition] = {}
    for sensor_id in range(1, NUM_SENSORS + 1):
        rate_hz = configured_sensor_rate_hz(sensor_id)
        definitions[sensor_id] = SensorDefinition(
            sensor_id=sensor_id,
            label=f"Sensor {sensor_id}",
            nominal_rate_hz=rate_hz,
            actual_rate_hz=rate_hz,
            variance=MEASUREMENT_VARIANCE,
            colour=configured_sensor_colour(sensor_id),
            disturb_measurements=sensor_has_any_measurement_disturbance(sensor_id),
        )
    return definitions


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

    disturbance_factor = (
        PROCESS_NOISE_INCREASE_FACTOR
        if ACTIVATE_DISTURBANCES and ENABLE_PROCESS_NOISE_DISTURBANCE
        else 1.0
    )
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
    seed = RANDOM_SEED_SENSOR_BASE + int(definition.sensor_id)
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
        if (
            ACTIVATE_DISTURBANCES
            and ENABLE_MEASUREMENT_NOISE_DISTURBANCE
            and definition.sensor_id in effective_fault_sensor_ids(
                MEASUREMENT_NOISE_SENSOR_IDS
            )
        )
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
    outlier_enabled_for_sensor = (
        ACTIVATE_DISTURBANCES
        and ENABLE_OUTLIER_DISTURBANCE
        and definition.sensor_id in effective_fault_sensor_ids(OUTLIER_SENSOR_IDS)
    )
    outlier_interval = (
        OUTLIER_INTERVAL_S
        if outlier_enabled_for_sensor
        else (SCENARIO_DURATION_S + 10.0, SCENARIO_DURATION_S + 20.0)
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
                sensor_step(outlier_interval[0]),
                sensor_step(outlier_interval[1]),
                5,
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
            definition.sensor_id == DROPOUT_SENSOR_ID
            and ENABLE_SENSOR_DROPOUT
            and SENSOR_DROPOUT_INTERVAL_S[0] <= elapsed_s < SENSOR_DROPOUT_INTERVAL_S[1]
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
            ACTIVATE_DISTURBANCES
            and ENABLE_NON_GAUSSIAN_DISTURBANCE
            and definition.sensor_id in effective_fault_sensor_ids(
                NON_GAUSSIAN_SENSOR_IDS
            )
            and VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[0]
            <= elapsed_s
            < VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[1]
        ):
            # Deliberately violate ONLY the assumed single-Gaussian shape.
            # Both variants retain zero mean and the nominal covariance R in
            # the population.
            measurement_vector = true_model.function(state, noise=False)
            measurement_vector += sample_variance_matched_non_gaussian_noise_from_cov(
                np.asarray(true_model.noise_covar, dtype=float),
                rng,
            )
        else:
            measurement_vector = true_model.function(state, noise=True)

        if (
            ACTIVATE_DISTURBANCES
            and ENABLE_BIAS_DISTURBANCE
            and definition.sensor_id in effective_fault_sensor_ids(BIAS_SENSOR_IDS)
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

    # Pair-specific scheduled simultaneous rates, independent of dropout.  Each
    # G_ij TEF is parametrised by the physical rate at which that specific pair
    # can provide direct sensor-to-sensor evidence.
    scheduled_ids_by_timestamp = {
        timestamp: {event.sensor_id for event in batch}
        for timestamp, batch in events.items()
    }
    pair_event_rates_hz: dict[tuple[int, int], float] = {}
    for pair in combinations(sorted(sensor_definitions), 2):
        pair_set = set(pair)
        simultaneous_count = sum(
            pair_set.issubset(scheduled_ids)
            for scheduled_ids in scheduled_ids_by_timestamp.values()
        )
        pair_event_rates_hz[pair] = max(
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
        pair_event_rates_hz=pair_event_rates_hz,
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
    disagreement_states: dict[tuple[int, int], SensorDisagreementState]
    batch_track_state: BatchTrackAssessmentState
    griebel: GriebelReferenceBackend

    # C~_s = A_s (*) C_s^iso, retained as a local diagnostic only.
    trusted_sensor_history: dict[int, list]

    # Base current-track consistency under the normal agreement regime:
    # omega_B = WBF(C_1^iso, ..., C_N^iso, C_F)
    track_base_history: list

    # Stricter conditional branch used only when pair agreement is doubtful:
    # omega_strict = AND(C_1^iso, ..., C_N^iso, C_F)
    track_strict_history: list

    # Pair-conditioned opinions, one per unique upper-triangular agreement entry:
    # omega_C,ij = Deduction(G_ij; omega_B, omega_strict)
    pair_conditioned_history: dict[tuple[int, int], list]

    # Overall current-track consistency:
    # omega_C = WBF({omega_C,ij | i<j and G_ij has evidence}); for N=1 -> omega_B
    track_consistency_history: list

    # Explicit symmetric opinion matrices generated at every union event.
    agreement_matrix_history: list[np.ndarray]
    deduction_matrix_history: list[np.ndarray]

    # Availability support from independent expected-output evidence:
    # omega_A = ABF(A_1, ..., A_N)
    combined_availability_history: list

    # Final current-track trust (unchanged):
    # omega_T = trust_discount(omega_A, omega_C)
    track_output_trust_history: list

    # System health as logical conjunction of track consistency and availability:
    # omega_H = omega_C * omega_A
    system_health_history: list

    # Architecture/reference histories retained for comparison.
    common_abf_history: list
    batch_history: list
    position_error: list[float]


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

    # One direct agreement/disagreement TEF for every unique sensor pair i<j.
    # The collection is the upper triangle of the symmetric agreement matrix.
    disagreement_states: dict[tuple[int, int], SensorDisagreementState] = {
        pair: SensorDisagreementState(pair, scenario.pair_event_rates_hz[pair])
        for pair in combinations(sensor_ids, 2)
    }

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
    print(f"  sensor-pair disagreement horizon: {DISAGREEMENT_SHORT_TERM_HORIZON_S:g} s")
    if USE_HEAVY_TAILED_NON_GAUSSIAN:
        print(
            "  non-Gaussian disturbance: heavy-tailed mixture, "
            f"{100.0 * HEAVY_TAIL_CORE_PROBABILITY:.0f}% "
            f"N(0,{HEAVY_TAIL_CORE_STD:.3f}^2) + "
            f"{100.0 * (1.0 - HEAVY_TAIL_CORE_PROBABILITY):.0f}% "
            f"N(0,{HEAVY_TAIL_TAIL_STD:.3f}^2)"
        )
    else:
        print(
            "  non-Gaussian disturbance: symmetric bimodal mixture, "
            f"mode offset={BIMODAL_MODE_OFFSET_STD:.3f}, "
            f"within-mode sigma={BIMODAL_WITHIN_MODE_STD:.3f}"
        )
    print(
        "  TEF fusion types: "
        f"local={LOCAL_CONSISTENCY_TEF_FUSION}, "
        f"batch={BATCH_TEF_FUSION}, "
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
    for pair, pair_state in disagreement_states.items():
        i, j = pair
        print(
            f"  sensor-pair agreement/disagreement G_{i}{j}: "
            f"rate~{pair_state.input_rate_hz:.3f} Hz, "
            f"T_ST={DISAGREEMENT_SHORT_TERM_HORIZON_S:g} s, "
            f"n_ST={pair_state.n_st}, "
            f"gamma={pair_state.discount:.6f}"
        )
    print(f"  Griebel reference (Evaluation A only): {griebel.status}")
    print(f"  Griebel async extension: {GRIEBEL_ASYNC_EXTENSION_MODE}")

    track = Track()
    event_timestamps: list[datetime] = []
    event_times_s: list[float] = []
    active_batch_sizes: list[int] = []

    trusted_sensor_history = {sensor_id: [] for sensor_id in sensor_ids}
    track_base_history: list = []
    track_strict_history: list = []
    pair_conditioned_history = {pair: [] for pair in disagreement_states}
    track_consistency_history: list = []
    agreement_matrix_history: list[np.ndarray] = []
    deduction_matrix_history: list[np.ndarray] = []
    combined_availability_history: list = []
    track_output_trust_history: list = []
    system_health_history: list = []
    common_abf_history: list = []
    batch_history: list = []

    truth_by_timestamp = {state.timestamp: state for state in scenario.truth}
    position_error: list[float] = []

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
                # nor penalised. Freeze the consistency TEFs/opinions; the
                # missing-information semantics are carried only by A_s.
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
        # Direct pairwise agreement matrix G = [G_ij].
        #
        # Only unique pairs i<j own a TEF.  If both sensors of a pair are
        # scheduled at the current timestamp, the pair is either updated from
        # the two arrived measurements or frozen if one of them is unavailable.
        # The full symmetric matrix is generated later from these upper-triangle
        # states.
        # ------------------------------------------------------------------
        scheduled_sensor_ids = {event.sensor_id for event in scheduled_batch}
        for pair, pair_state in disagreement_states.items():
            i, j = pair
            if i in scheduled_sensor_ids and j in scheduled_sensor_ids:
                if i in active_by_id and j in active_by_id:
                    pair_state.update(
                        active_by_id[i],
                        active_by_id[j],
                        timestamp,
                        scenario.start_time,
                    )
                else:
                    pair_state.advance_without_pair(timestamp, scenario.start_time)

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

        else:
            # A scheduled union event exists but no measurement is available.
            # Freeze C_F; availability support carries the loss of information.
            batch_track_state.advance_without_measurement(
                timestamp,
                scenario.start_time,
            )

        # ------------------------------------------------------------------
        # Hierarchical Subjective-Logic track-trust construction.
        #
        # Statistical consistency and information availability are deliberately
        # kept separate until the final trust-discount step.
        #
        # 1) Keep availability-discounted local opinions only as diagnostics:
        #       C~_s = A_s (*) C_s^iso
        #
        # 2) Nominal/base track-consistency opinion:
        #       omega_B = WBF(C_1^iso, ..., C_N^iso, C_F)
        #
        # 3) Stricter branch for the case that the sensors disagree:
        #       omega_strict = AND(C_1^iso, ..., C_N^iso, C_F)
        #
        # 4) Every upper-triangular agreement opinion G_ij is a contextual
        #    antecedent for one pair-conditioned opinion:
        #
        #       omega_C,ij = Deduction(G_ij; omega_B, omega_strict),  i<j
        #
        #    The resulting opinions all address the same track-consistency
        #    proposition and are combined with WBF to obtain omega_C.  For N=2
        #    this collapses exactly to the previous single G_12 deduction.
        #
        # 5) Combine availability opinions using ABF and apply it as reliability
        #    trust to the already constructed consistency opinion:
        #
        #       omega_A = ABF(A_1, ..., A_N)
        #       omega_T = omega_A (*) omega_C
        #
        #    Trust discounting scales belief and disbelief equally and transfers
        #    the removed committed mass into uncertainty.  A sensor dropout can
        #    therefore increase u_T without inventing statistical inconsistency.
        # ------------------------------------------------------------------
        local_consistency_inputs = []
        availability_inputs = []

        for sensor_id in sensor_ids:
            c_s_iso = deepcopy(isolated_states[sensor_id].latest_opinion)
            a_s = deepcopy(availability_states[sensor_id].latest_opinion)

            # Diagnostic: how the local consistency opinion looks when interpreted
            # through the current availability of that same sensor.
            trusted = trust_discount(a_s, c_s_iso)
            trusted_sensor_history[sensor_id].append(trusted)

            local_consistency_inputs.append(c_s_iso)
            availability_inputs.append(a_s)

        central_consistency = deepcopy(batch_track_state.latest_opinion)

        # Nominal branch: several related consistency sources support the same
        # higher-level statement that the current track generation is consistent.
        track_base = fuse_weighted(
            [*local_consistency_inputs, central_consistency]
        )
        track_base_history.append(track_base)

        # Disagreement branch: if the sensors do not agree, retaining trust in
        # the track requires simultaneous support from all local and central
        # consistency statements.  This branch is NOT applied in the nominal
        # agreement regime.
        track_strict = logical_and(
            [*local_consistency_inputs, central_consistency]
        )
        track_strict_history.append(track_strict)

        # Every unique upper-triangular agreement opinion G_ij conditions its own
        # track-consistency opinion.  This preserves the diagnostic structure:
        #
        #   omega_C,ij = Deduction(G_ij; omega_B, omega_strict)
        #
        # All omega_C,ij assess the same target proposition and are therefore
        # combined with WBF, which avoids artificial evidence accumulation.
        # A pair with a still-vacuous G_ij contributes no conditioning yet.
        pair_conditioned_now: dict[tuple[int, int], object] = {}
        valid_pair_conditioned: list = []
        for pair, pair_state in disagreement_states.items():
            pair_agreement = deepcopy(pair_state.latest_opinion)
            if uncertainty(pair_agreement) < 1.0 - 1e-12:
                conditioned = subjective_logic_deduction(
                    pair_agreement,
                    track_base,
                    track_strict,
                )
                valid_pair_conditioned.append(conditioned)
            else:
                conditioned = deepcopy(track_base)
            pair_conditioned_now[pair] = conditioned
            pair_conditioned_history[pair].append(deepcopy(conditioned))

        track_consistency = (
            fuse_weighted(valid_pair_conditioned)
            if valid_pair_conditioned
            else deepcopy(track_base)
        )
        track_consistency_history.append(track_consistency)

        # Generate the explicit symmetric matrices requested for diagnostics.
        # The diagonal is None because self-agreement is not an assessed channel.
        sensor_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
        agreement_matrix = np.empty((len(sensor_ids), len(sensor_ids)), dtype=object)
        deduction_matrix = np.empty((len(sensor_ids), len(sensor_ids)), dtype=object)
        agreement_matrix[:] = None
        deduction_matrix[:] = None
        for pair, pair_state in disagreement_states.items():
            i, j = pair
            row = sensor_index[i]
            col = sensor_index[j]
            agreement_opinion = deepcopy(pair_state.latest_opinion)
            deduction_opinion = deepcopy(pair_conditioned_now[pair])
            agreement_matrix[row, col] = agreement_opinion
            agreement_matrix[col, row] = deepcopy(agreement_opinion)
            deduction_matrix[row, col] = deduction_opinion
            deduction_matrix[col, row] = deepcopy(deduction_opinion)
        agreement_matrix_history.append(agreement_matrix)
        deduction_matrix_history.append(deduction_matrix)

        # Availability is intentionally the final modifier.  ABF aggregates the
        # expected-output availability opinions of the sensor paths; the resulting
        # opinion acts as reliability trust for the consistency-derived track
        # opinion, shifting missing-information effects to uncertainty.
        combined_availability = fuse_average(availability_inputs)
        combined_availability_history.append(combined_availability)

        track_output_trust = trust_discount(
            combined_availability,
            track_consistency,
        )
        track_output_trust_history.append(track_output_trust)

        # System health is a separate higher-level proposition.  Unlike the
        # reliability discount used for omega_T, this is the SL conjunction
        # requested for H: omega_H = omega_C * omega_A.
        system_health = logical_and([
            track_consistency,
            combined_availability,
        ])
        system_health_history.append(system_health)

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
        disagreement_states=disagreement_states,
        batch_track_state=batch_track_state,
        griebel=griebel,
        trusted_sensor_history=trusted_sensor_history,
        track_base_history=track_base_history,
        track_strict_history=track_strict_history,
        pair_conditioned_history=pair_conditioned_history,
        track_consistency_history=track_consistency_history,
        agreement_matrix_history=agreement_matrix_history,
        deduction_matrix_history=deduction_matrix_history,
        combined_availability_history=combined_availability_history,
        track_output_trust_history=track_output_trust_history,
        system_health_history=system_health_history,
        common_abf_history=common_abf_history,
        batch_history=batch_history,
        position_error=position_error,
    )


# =============================================================================
# Plotting helpers
# =============================================================================


def add_disturbance_spans(axis) -> None:
    for start_s, end_s, label, colour in DISTURBANCE_INTERVALS:
        if label == f"S{DROPOUT_SENSOR_ID} unavailable" and (
            not ENABLE_SENSOR_DROPOUT or DROPOUT_SENSOR_ID > NUM_SENSORS
        ):
            continue
        if label == "common motion-model mismatch" and not ENABLE_GROUND_TRUTH_TURN:
            continue
        axis.axvspan(start_s, end_s, color=colour, alpha=0.07)


def is_nominal_time(time_s: float) -> bool:
    if time_s < NOMINAL_BURN_IN_S:
        return False
    for start, end, label, _ in DISTURBANCE_INTERVALS:
        if label == f"S{DROPOUT_SENSOR_ID} unavailable" and (
            not ENABLE_SENSOR_DROPOUT or DROPOUT_SENSOR_ID > NUM_SENSORS
        ):
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
        [normalized_score(op) for op in opinions],
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


# -----------------------------------------------------------------------------
# BEGIN EXPERIMENTAL EVALUATION B METRICS
# Everything in this block is offline evaluation only.  It can be removed
# without changing filtering, TEFs, SL opinions, fusion, track trust or health.
# -----------------------------------------------------------------------------


def plot_static_results(result: ProcessingResult) -> None:
    event_times = np.asarray(result.event_times_s, dtype=float)
    sensor_ids = sorted(result.isolated_states)

    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        mode = "fully synchronous 10 Hz"
    elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
        mode = "asynchronous cross-contamination rate stress"
    else:
        rate_summary = ", ".join(
            f"S{sensor_id}={result.isolated_states[sensor_id].input_rate_hz:g} Hz"
            for sensor_id in sensor_ids
        )
        mode = f"asynchronous multi-rate: {rate_summary}"

    # ------------------------------------------------------------------
    # Figure 1: local diagnosis - selected normalized committed score + u.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3,
        len(sensor_ids),
        figsize=(max(14, 4.5 * len(sensor_ids)), 10),
        sharex="col",
        squeeze=False,
    )
    for col, sensor_id in enumerate(sensor_ids):
        iso = result.isolated_states[sensor_id]
        common = result.common_prediction_states[sensor_id]

        axes[0, col].plot(
            iso.event_times_s,
            normalized_score_series(
                iso.belief_events, iso.disbelief_events, iso.uncertainty_events
            ),
            color=iso.colour,
            linewidth=1.7,
            label=rf"isolated ${normalized_score_symbol()}_{{\mathrm{{norm}}}}(C_s)$",
        )
        axes[0, col].plot(
            common.event_times_s,
            normalized_score_series(
                common.belief_events, common.disbelief_events, common.uncertainty_events
            ),
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label=rf"common-prior ${normalized_score_symbol()}_{{\mathrm{{norm}}}}(C_s)$",
        )
        axes[0, col].set_ylabel(normalized_score_name() + rf" ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")

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
        + (
            rf"({mode}; ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$ = "
            + (r"$b/(1-u)$ = normalized consistency" if USE_NORMALIZED_BELIEF else r"$d/(1-u)$ = normalized inconsistency")
            + r", $u$ = lack of evidence)"
        )
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
        normalized_score_series(
            batch.belief_events, batch.disbelief_events, batch.uncertainty_events
        ),
        color="tab:purple",
        label=rf"direct batch ${normalized_score_symbol()}_{{\mathrm{{norm}}}}(C_F)$",
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

    axes[0].set_ylabel(normalized_score_name() + rf" ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
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
    # Figure 4: all unique pairwise agreement opinions and their deductions.
    # ------------------------------------------------------------------
    if result.disagreement_states and SHOW_PAIRWISE_TIME_SERIES:
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        for pair, pair_state in sorted(result.disagreement_states.items()):
            i, j = pair
            colour = configured_sensor_colour(i + j - 1)
            pair_score = normalized_score_series(
                pair_state.belief_events,
                pair_state.disbelief_events,
                pair_state.uncertainty_events,
            )
            axes[0].plot(
                pair_state.event_times_s, pair_score, linewidth=1.45,
                label=rf"$G_{{{i}{j}}}$ " + ("agreement" if USE_NORMALIZED_BELIEF else "disagreement"),
            )
            axes[1].plot(
                pair_state.event_times_s, pair_state.uncertainty_events, linewidth=1.35,
                label=rf"$u(G_{{{i}{j}}})$",
            )
            axes[2].plot(
                event_times,
                [normalized_score(op) for op in result.pair_conditioned_history[pair]],
                linewidth=1.45,
                label=rf"${normalized_score_symbol()}_{{\mathrm{{norm}}}}(\omega_{{C,{i}{j}}})$",
            )

        axes[0].set_ylabel(rf"pair ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
        axes[1].set_ylabel("pair uncertainty")
        axes[2].set_ylabel(rf"deduced ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True)
            axis.legend(loc="upper right", fontsize=8, ncol=max(1, min(4, len(result.disagreement_states))))
            add_disturbance_spans(axis)
        fig.suptitle(
            "Pairwise agreement matrix channels and pair-conditioned track consistency\n"
            r"$\omega_{C,ij}=\mathrm{Deduction}(G_{ij};\omega_B,\omega_{\mathrm{strict}})$, $i<j$"
        )
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 5: upper-triangular agreement/deduction matrices at one selected
    # diagnostic time.  This replaces the previous AUROC/TPR Figure 5 and is
    # directly useful for N-sensor fault isolation.
    # ------------------------------------------------------------------
    if result.disagreement_states and SHOW_PAIRWISE_MATRIX_FIGURE:
        snapshot_time_s = (
            float(PAIR_MATRIX_SNAPSHOT_TIME_S)
            if PAIR_MATRIX_SNAPSHOT_TIME_S is not None
            else 0.5 * (SENSOR_1_BIAS_INTERVAL_S[0] + SENSOR_1_BIAS_INTERVAL_S[1])
        )
        event_array = np.asarray(result.event_times_s, dtype=float)
        snapshot_index = int(np.searchsorted(event_array, snapshot_time_s, side="right") - 1)
        snapshot_index = int(np.clip(snapshot_index, 0, len(event_array) - 1))
        actual_snapshot_time = float(event_array[snapshot_index])
        agreement_objects = result.agreement_matrix_history[snapshot_index]
        deduction_objects = result.deduction_matrix_history[snapshot_index]
        n = len(sensor_ids)
        agreement_values = np.full((n, n), np.nan, dtype=float)
        deduction_values = np.full((n, n), np.nan, dtype=float)
        for row in range(n):
            for col in range(row + 1, n):
                agreement_op = agreement_objects[row, col]
                deduction_op = deduction_objects[row, col]
                if agreement_op is not None:
                    agreement_values[row, col] = normalized_score(agreement_op)
                if deduction_op is not None:
                    deduction_values[row, col] = normalized_score(deduction_op)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        score_symbol = normalized_score_symbol()
        for axis, values, title in (
            (axes[0], agreement_values, rf"Agreement matrix: ${score_symbol}_{{\mathrm{{norm}}}}(G_{{ij}})$"),
            (axes[1], deduction_values, rf"Pair-conditioned matrix: ${score_symbol}_{{\mathrm{{norm}}}}(\omega_{{C,ij}})$"),
        ):
            # Semantic traffic-light scale follows the selected score:
            # d_norm: 0 green -> 1 red; b_norm: 0 red -> 1 green.
            image = axis.imshow(
                np.ma.masked_invalid(values),
                vmin=0.0,
                vmax=1.0,
                cmap=normalized_score_cmap(),
            )
            axis.set_xticks(np.arange(n), [f"S{s}" for s in sensor_ids])
            axis.set_yticks(np.arange(n), [f"S{s}" for s in sensor_ids])
            axis.set_title(title)
            for row in range(n):
                for col in range(row + 1, n):
                    value = values[row, col]
                    if np.isfinite(value):
                        # Select black/white annotation text from the actual cell
                        # luminance so the value remains readable on green/yellow/red.
                        rgba = image.cmap(image.norm(float(value)))
                        luminance = (
                            0.2126 * rgba[0]
                            + 0.7152 * rgba[1]
                            + 0.0722 * rgba[2]
                        )
                        text_colour = "black" if luminance > 0.55 else "white"
                        annotation = f"{value:.2f}"
                    else:
                        text_colour = "0.35"
                        annotation = "--"
                    axis.text(
                        col,
                        row,
                        annotation,
                        ha="center",
                        va="center",
                        color=text_colour,
                        fontsize=13,
                        fontweight="bold",
                    )
            # Explicitly mark diagonal/lower triangle as redundant/not evaluated.
            for row in range(n):
                for col in range(0, row + 1):
                    axis.text(
                        col, row, "·",
                        ha="center", va="center",
                        color="0.60", fontsize=12,
                    )
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(
                rf"${score_symbol}_{{\mathrm{{norm}}}}$: " + normalized_score_semantics()
            )
        fig.suptitle(
            f"N-sensor pairwise diagnostic matrices at t={actual_snapshot_time:.1f} s\n"
            "Only the upper triangle contains unique sensor pairs"
        )
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Dedicated per-pair trajectories: each unique upper-triangular matrix
    # entry gets its own subplot. This directly shows how G_ij drives the
    # corresponding pair-conditioned opinion omega_C,ij over time.
    # ------------------------------------------------------------------
    if result.disagreement_states and SHOW_PAIRWISE_DEDUCTION_TRAJECTORIES:
        pairs = sorted(result.disagreement_states)
        n_pairs = len(pairs)
        n_cols = max(1, min(PAIR_TRAJECTORY_MAX_COLUMNS, n_pairs))
        n_rows = int(np.ceil(n_pairs / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.5 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for pair_index, pair in enumerate(pairs):
            row = pair_index // n_cols
            col = pair_index % n_cols
            axis = axes[row, col]
            i, j = pair
            pair_state = result.disagreement_states[pair]

            pair_score = normalized_score_series(
                pair_state.belief_events,
                pair_state.disbelief_events,
                pair_state.uncertainty_events,
            )
            deduction_score = [
                normalized_score(opinion)
                for opinion in result.pair_conditioned_history[pair]
            ]

            axis.plot(
                pair_state.event_times_s,
                pair_score,
                color="tab:orange",
                linewidth=1.45,
                label=rf"$G_{{{i}{j}}}$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
            )
            axis.plot(
                event_times,
                deduction_score,
                color="tab:blue",
                linewidth=1.65,
                label=rf"$\omega_{{C,{i}{j}}}$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
            )
            axis.set_title(rf"Sensor pair $S_{i}$--$S_{j}$")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True)
            add_disturbance_spans(axis)
            axis.legend(loc="upper right", fontsize=8)

            if col == 0:
                axis.set_ylabel(rf"${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
            if row == n_rows - 1:
                axis.set_xlabel("time [s]")

        # Hide unused grid cells when the pair count is not a multiple of n_cols.
        for pair_index in range(n_pairs, n_rows * n_cols):
            row = pair_index // n_cols
            col = pair_index % n_cols
            axes[row, col].axis("off")

        fig.suptitle(
            "Individual pairwise agreement and pair-conditioned consistency trajectories\n"
            r"$\omega_{C,ij}=\mathrm{Deduction}(G_{ij};\omega_B,\omega_{\mathrm{strict}})$, $i<j$"
        )
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Native Griebel local outputs retained for the source-isolation /
    # cross-contamination evaluation (Evaluation A).  No Griebel overall-fusion
    # comparison is constructed in V8.
    # ------------------------------------------------------------------
    if result.griebel.native:
        fig, axes = plt.subplots(
            2,
            len(sensor_ids),
            figsize=(max(14, 4.5 * len(sensor_ids)), 7),
            sharex="col",
            squeeze=False,
        )

        for col, sensor_id in enumerate(sensor_ids):
            colour = result.isolated_states[sensor_id].colour

            axis = axes[0, col]
            axis.plot(
                result.griebel.sensor_times[sensor_id],
                result.griebel.sensor_delta[sensor_id],
                color=colour,
                label=rf"native Griebel $\delta^{{({sensor_id})}}$",
            )
            axis.plot(
                result.griebel.sensor_times[sensor_id],
                result.griebel.sensor_eta[sensor_id],
                color="black",
                linestyle="--",
                label=rf"native Griebel $\eta^{{({sensor_id})}}$",
            )
            axis.set_title(f"Sensor {sensor_id}")
            axis.set_ylabel("DC / threshold")
            axis.grid(True)
            axis.legend(loc="upper right")
            add_disturbance_spans(axis)

            axis = axes[1, col]
            axis.plot(
                result.griebel.sensor_times[sensor_id],
                result.griebel.sensor_uncertainty[sensor_id],
                color=colour,
                label=rf"native Griebel $u^{{({sensor_id})}}$",
            )
            axis.set_ylabel("uncertainty")
            axis.set_xlabel("time [s]")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(True)
            axis.legend(loc="upper right")
            add_disturbance_spans(axis)

        if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
            griebel_local_title = (
                "Native Griebel single-sensor SA outputs"
            )
        else:
            griebel_local_title = (
                "Native Griebel single-sensor SA outputs under the explicitly "
                f"labelled {GRIEBEL_ASYNC_EXTENSION_MODE!r} event semantics"
            )

        fig.suptitle(griebel_local_title)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Hierarchical consistency -> conditional reasoning -> availability trust.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    # Raw online consistency inputs.  Availability is intentionally not folded
    # into these curves so statistical inconsistency and missing information stay
    # visually distinguishable.
    for sensor_id in sensor_ids:
        state = result.isolated_states[sensor_id]
        axes[0].plot(
            state.event_times_s,
            normalized_score_series(
                state.belief_events,
                state.disbelief_events,
                state.uncertainty_events,
            ),
            color=state.colour,
            linewidth=1.35,
            label=rf"$C_{{{sensor_id}}}^{{iso}}$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
        )

    axes[0].plot(
        event_times,
        [normalized_score(op) for op in result.batch_history],
        color="tab:purple",
        linewidth=1.15,
        alpha=0.85,
        label=rf"$C_F$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
    )

    # Higher-level consistency path and final track trust.
    axes[1].plot(
        event_times,
        [normalized_score(op) for op in result.track_base_history],
        color="tab:green",
        linewidth=1.35,
        label=r"$\omega_B=\mathrm{WBF}(C_1^{iso},\ldots,C_N^{iso},C_F)$",
    )
    axes[1].plot(
        event_times,
        [normalized_score(op) for op in result.track_strict_history],
        color="tab:gray",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label=r"$\omega_{\mathrm{strict}}=(\bigwedge_s C_s^{iso})\wedge C_F$",
    )
    axes[1].plot(
        event_times,
        [normalized_score(op) for op in result.track_consistency_history],
        color="tab:blue",
        linewidth=1.6,
        label=r"$\omega_C=\mathrm{WBF}_{i<j}(\omega_{C,ij})$",
    )
    axes[1].plot(
        event_times,
        [normalized_score(op) for op in result.track_output_trust_history],
        color="tab:red",
        linewidth=1.9,
        label=rf"final track trust $\omega_T$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
    )
    axes[1].plot(
        event_times,
        [normalized_score(op) for op in result.system_health_history],
        color="tab:pink",
        linewidth=1.7,
        linestyle="--",
        label=rf"system health $\omega_H=\omega_C\cdot\omega_A$: ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$",
    )

    # Availability should manifest primarily as uncertainty in the final output.
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.combined_availability_history],
        color="tab:orange",
        linewidth=1.35,
        label=r"aggregated availability $u_A$",
    )
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.track_consistency_history],
        color="tab:blue",
        linewidth=1.35,
        label=r"pair-conditioned overall consistency $u_C$",
    )
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.track_output_trust_history],
        color="tab:red",
        linewidth=1.9,
        label=r"final track trust $u_T$",
    )
    axes[2].plot(
        event_times,
        [uncertainty(op) for op in result.system_health_history],
        color="tab:pink",
        linewidth=1.7,
        linestyle="--",
        label=r"system health $u_H$",
    )

    axes[0].set_title("Consistency assessment inputs")
    axes[1].set_title("Track consistency, final track trust, and system health")
    axes[2].set_title("Availability and uncertainty propagation")
    axes[0].set_ylabel(rf"input ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
    axes[1].set_ylabel(rf"track / health ${normalized_score_symbol()}_{{\mathrm{{norm}}}}$")
    axes[2].set_ylabel("uncertainty")
    axes[2].set_xlabel("time [s]")

    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")
        add_disturbance_spans(axis)
        axis.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Hierarchical Subjective-Logic self-assessment — N-sensor track trust and system health\n"
        r"$\omega_B=\mathrm{WBF}(C_1^{iso},\ldots,C_N^{iso},C_F)$; "
        r"$\omega_{C,ij}=\mathrm{Deduction}(G_{ij};\omega_B,\omega_{\mathrm{strict}})$; "
        r"$\omega_C=\mathrm{WBF}_{i<j}(\omega_{C,ij})$; "
        r"$\omega_A=\mathrm{ABF}(A_1,\ldots,A_N)$; "
        r"$\omega_T=\omega_A\otimes\omega_C$; $\omega_H=\omega_C\cdot\omega_A$"
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
    """Return active disturbance labels for the dynamically configured sensors."""
    labels: list[str] = []

    if ACTIVATE_DISTURBANCES:
        for sensor_id in sorted(DISTURBED_SENSOR_IDS):
            if not 1 <= int(sensor_id) <= NUM_SENSORS:
                continue
            prefix = f"S{sensor_id}"
            if OUTLIER_INTERVAL_S[0] <= elapsed_s < OUTLIER_INTERVAL_S[1]:
                labels.append(f"{prefix} outliers")
            if SENSOR_1_BIAS_INTERVAL_S[0] <= elapsed_s < SENSOR_1_BIAS_INTERVAL_S[1]:
                labels.append(
                    f"{prefix} + {int(SENSOR_1_BIAS_VECTOR_M[0])}m x-bias"
                )
            if (
                INCREASED_MEAS_XY_INTERVAL_S[0]
                <= elapsed_s
                < INCREASED_MEAS_XY_INTERVAL_S[1]
            ):
                labels.append(
                    f"{prefix} increased x/y-noise "
                    f"(R x{MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR:g})"
                )
            if (
                VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[0]
                <= elapsed_s
                < VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S[1]
            ):
                distribution_label = (
                    "variance-matched heavy-tailed mixture"
                    if USE_HEAVY_TAILED_NON_GAUSSIAN
                    else "variance-matched bimodal mixture"
                )
                labels.append(f"{prefix} {distribution_label}")

    if (
        ENABLE_SENSOR_DROPOUT
        and 1 <= DROPOUT_SENSOR_ID <= NUM_SENSORS
        and SENSOR_DROPOUT_INTERVAL_S[0]
        <= elapsed_s
        < SENSOR_DROPOUT_INTERVAL_S[1]
    ):
        labels.append(f"S{DROPOUT_SENSOR_ID} unavailable")

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
    """Dynamic Plotly dashboard for one to four configured sensors.

    The left half keeps the V7 follow-view.  The right half is generated
    dynamically: one consistency triangle per sensor plus shared triangles for
    availability, aggregated availability, all unique G_ij pair opinions,
    central-batch consistency, overall omega_C, and the final omega_T/omega_H
    opinions.  For N>4 the caller skips this dashboard to avoid excessive
    visual density.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if NUM_SENSORS > 4:
        print(
            f"Dynamic animation skipped: N={NUM_SENSORS} > 4. "
            "Static N-sensor figures remain available."
        )
        return

    if len(result.track) == 0 or len(result.event_timestamps) == 0:
        print("Dynamic animation skipped: no track/event data.")
        return

    sensor_ids = sorted(result.isolated_states)
    sensor_colours = {
        sensor_id: colour
        for sensor_id, colour in zip(
            sensor_ids,
            ("royalblue", "darkorange", "seagreen", "mediumpurple"),
        )
    }
    pair_palette = (
        "deepskyblue",
        "goldenrod",
        "limegreen",
        "hotpink",
        "sienna",
        "teal",
    )

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
        f"(effective stride={stride}; N={NUM_SENSORS})"
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
        panels = []
        titles = []

        # One sensor-specific consistency panel per configured sensor.
        for sensor_id in sensor_ids:
            c_iso = _state_triplet_at_time(
                result.isolated_states[sensor_id], elapsed_s
            )
            c_common = _state_triplet_at_time(
                result.common_prediction_states[sensor_id], elapsed_s
            )
            panels.append(
                [
                    (
                        f"C{sensor_id} isolated",
                        c_iso,
                        sensor_colours[sensor_id],
                    ),
                    (f"C{sensor_id} common", c_common, "gray"),
                ]
            )
            titles.append(f"Sensor {sensor_id} consistency")

        availability_entries = []
        for sensor_id in sensor_ids:
            availability_entries.append(
                (
                    f"A{sensor_id}",
                    _state_triplet_at_time(
                        result.availability_states[sensor_id], elapsed_s
                    ),
                    sensor_colours[sensor_id],
                )
            )
        panels.append(availability_entries)
        titles.append("Sensor availability")

        combined_availability = history_triplet(
            result.combined_availability_history, step
        )
        panels.append(
            [("omega_A ABF", combined_availability, "darkorange")]
        )
        titles.append("Availability (ABF)")

        pair_entries = []
        for pair_index, (pair, state) in enumerate(
            sorted(result.disagreement_states.items())
        ):
            i, j = pair
            pair_entries.append(
                (
                    f"G{i}{j}",
                    _state_triplet_at_time(state, elapsed_s),
                    pair_palette[pair_index % len(pair_palette)],
                )
            )
        if not pair_entries:
            pair_entries = [
                ("No sensor pair", _vacuous_triplet(), "lightgray")
            ]
        panels.append(pair_entries)
        titles.append("Pairwise sensor agreement")

        c_f = history_triplet(result.batch_history, step)
        panels.append([("C_F", c_f, "purple")])
        titles.append("Central track-filter")

        track_consistency = history_triplet(
            result.track_consistency_history, step
        )
        panels.append(
            [("omega_C overall consistency", track_consistency, "green")]
        )
        titles.append("Pair-conditioned track")

        track_trust = history_triplet(
            result.track_output_trust_history, step
        )
        system_health = history_triplet(
            result.system_health_history, step
        )
        panels.append(
            [
                ("omega_T final track trust", track_trust, "red"),
                ("omega_H system health", system_health, "magenta"),
            ]
        )
        titles.append("Final opinions")

        return panels, titles

    initial_step = frame_steps[0]
    initial_panels, panel_titles = opinion_panels(initial_step)
    n_panel_rows = int(ceil(len(initial_panels) / 2.0))

    # Track view occupies the complete left half. The right half contains up to
    # two dynamically generated ternary plots per row.
    specs = []
    for row_index in range(n_panel_rows):
        if row_index == 0:
            left_specs = [
                {"type": "xy", "rowspan": n_panel_rows, "colspan": 2},
                None,
            ]
        else:
            left_specs = [None, None]

        panel_index_left = 2 * row_index
        panel_index_right = panel_index_left + 1
        right_specs = [
            {"type": "ternary"}
            if panel_index_left < len(initial_panels)
            else None,
            {"type": "ternary"}
            if panel_index_right < len(initial_panels)
            else None,
        ]
        specs.append(left_specs + right_specs)

    fig = make_subplots(
        rows=n_panel_rows,
        cols=4,
        specs=specs,
        subplot_titles=["Track follow view"] + panel_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.05,
    )

    # Keep the established V7 title positioning, but make the offset slightly
    # smaller for five-row layouts used by N=3/4.
    opinion_title_x_shift = 0.10 if n_panel_rows <= 4 else 0.085
    opinion_title_y_shift = 0.045 if n_panel_rows <= 4 else 0.035
    for annotation in fig.layout.annotations:
        if annotation.text == "Track follow view":
            annotation.update(font=dict(size=15))
        else:
            annotation.update(
                x=annotation.x - opinion_title_x_shift,
                y=annotation.y - opinion_title_y_shift,
                xanchor="left",
                align="left",
                font=dict(size=13 if n_panel_rows >= 5 else 14),
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

        traces = [
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
        ]

        for sensor_id in sensor_ids:
            points = measurement_xy.get(sensor_id, np.empty((0, 2), dtype=float))
            traces.append(
                go.Scattergl(
                    x=points[:, 0] if len(points) else [],
                    y=points[:, 1] if len(points) else [],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=sensor_colours[sensor_id],
                        opacity=0.6,
                    ),
                    name=f"Sensor {sensor_id} measurements",
                )
            )

        traces.extend(
            [
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
        )
        return traces

    dynamic_indices = []

    for trace in tracking_traces(initial_step):
        fig.add_trace(trace, row=1, col=1)
        dynamic_indices.append(len(fig.data) - 1)

    ternary_positions = []
    for panel_index in range(len(initial_panels)):
        row = panel_index // 2 + 1
        col = 3 + panel_index % 2
        ternary_positions.append((row, col))

    for entries, (row, col) in zip(initial_panels, ternary_positions):
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
    rate_text = ", ".join(
        f"S{sensor_id}={configured_sensor_rate_hz(sensor_id):g} Hz"
        for sensor_id in sensor_ids
    )
    static_annotations = [
        annotation.to_plotly_json()
        for annotation in (fig.layout.annotations or [])
    ]

    # Increase the dashboard height for the fifth opinion row used at N=3/4.
    dashboard_height = 1120 if n_panel_rows <= 4 else 1320
    fig.update_layout(
        title=(
            "Dynamic event-based multi-sensor self-assessment — track trust and system health"
            f"<br><sup>{rate_mode}; N={NUM_SENSORS}; {rate_text}; "
            f"filter={model_label}</sup>"
        ),
        width=1750,
        height=dashboard_height,
        margin=dict(t=120, b=95, l=60, r=30),
        annotations=static_annotations + [
            _disturbance_annotation(
                initial_step,
                float(result.event_times_s[initial_step]),
            )
        ],
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
        panels, _ = opinion_panels(step)
        for entries in panels:
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
    batch_size_counts = {
        size: result.active_batch_sizes.count(size)
        for size in sorted(set(result.active_batch_sizes))
    }

    print("\nEvent statistics")
    print(f"  configured sensors: {len(result.isolated_states)}")
    print(f"  unique agreement pairs: {len(result.disagreement_states)}")
    print(f"  union event timestamps: {len(result.event_times_s)}")
    print(
        "  active measurement batch sizes: "
        + ", ".join(f"{size} -> {count}" for size, count in batch_size_counts.items())
    )

    print("\nGriebel reference (Evaluation A only)")
    print(f"  backend: {result.griebel.status}")
    print(f"  async extension mode: {GRIEBEL_ASYNC_EXTENSION_MODE}")
    print("  no Griebel multi-source ABF / AUROC / TPR comparison is generated in V8")

    # Cross-contamination diagnostic: compare a nominal reference sensor against
    # Sensor 1 disturbances when at least two sensors are configured.
    if len(result.isolated_states) >= 2:
        reference_sensor_id = 2
        reference_iso = result.isolated_states[reference_sensor_id]
        reference_common = result.common_prediction_states[reference_sensor_id]
        reference_iso_score = normalized_score_series(
            reference_iso.belief_events,
            reference_iso.disbelief_events,
            reference_iso.uncertainty_events,
        )
        reference_common_score = normalized_score_series(
            reference_common.belief_events,
            reference_common.disbelief_events,
            reference_common.uncertainty_events,
        )
        nominal_iso = nominal_mean(reference_iso.event_times_s, reference_iso_score)
        nominal_common = nominal_mean(reference_common.event_times_s, reference_common_score)
        print(f"\nSensor-{reference_sensor_id} cross-contamination check (Sensor 1 disturbed)")
        print(
            f"  nominal mean {normalized_score_symbol()}_norm: isolated={nominal_iso:.3f}, "
            f"common-prior={nominal_common:.3f}"
        )
        for interval, label in (
            (OUTLIER_INTERVAL_S, "S1 outliers"),
            (SENSOR_1_BIAS_INTERVAL_S, f"S1 +{SENSOR_1_BIAS_VECTOR_M[0]:g} m x-bias"),
            (INCREASED_MEAS_XY_INTERVAL_S, f"S1 increased x/y-noise (R x{MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR:g})"),
            (VARIANCE_MATCHED_NON_GAUSSIAN_INTERVAL_S, NON_GAUSSIAN_DISTURBANCE_LABEL),
        ):
            iso_value = interval_mean(reference_iso.event_times_s, reference_iso_score, interval)
            common_value = interval_mean(reference_common.event_times_s, reference_common_score, interval)
            print(
                f"  {label}: isolated {normalized_score_symbol()}_norm={iso_value:.3f} "
                f"(delta={iso_value - nominal_iso:+.3f}), "
                f"common-prior {normalized_score_symbol()}_norm={common_value:.3f} "
                f"(delta={common_value - nominal_common:+.3f})"
            )

    if ENABLE_SENSOR_DROPOUT and DROPOUT_SENSOR_ID in result.availability_states:
        availability_state = result.availability_states[DROPOUT_SENSOR_ID]
        print(f"\nSensor-{DROPOUT_SENSOR_ID} dropout")
        print(
            f"  mean P(A{DROPOUT_SENSOR_ID}) during dropout: "
            f"{interval_mean(availability_state.event_times_s, availability_state.p_available_events, SENSOR_DROPOUT_INTERVAL_S):.3f}"
        )
        print(
            f"  mean frozen local C{DROPOUT_SENSOR_ID} consistency uncertainty during dropout: "
            f"{interval_mean(result.isolated_states[DROPOUT_SENSOR_ID].event_times_s, result.isolated_states[DROPOUT_SENSOR_ID].uncertainty_events, SENSOR_DROPOUT_INTERVAL_S):.3f}"
        )
        print(
            "  mean track-output uncertainty during dropout: "
            f"{interval_mean(result.event_times_s, [uncertainty(op) for op in result.track_output_trust_history], SENSOR_DROPOUT_INTERVAL_S):.3f}"
        )

    print("\nInterpretation reminder")
    if USE_NORMALIZED_BELIEF:
        print("  b_norm=b/(1-u): normalized consistency within committed evidence (=1-d_norm)")
    else:
        print("  d_norm=d/(1-u): normalized inconsistency within committed evidence (=1-b_norm)")
    print("  C_s^iso       : sensor/path consistency using only that sensor history")
    print("  C_s^common    : same PIT/TEF mapping but central common prior")
    print("  A_s           : expected output availability")
    print("  dropout       : freezes statistical consistency channels; A_s carries missingness")
    print("  C~_s=A_s(*)C_s^iso: local availability-discounted diagnostic only")
    print(
        "  G_ij          : upper-triangular direct pair-agreement opinions; "
        + ("b_norm quantifies agreement" if USE_NORMALIZED_BELIEF else "d_norm quantifies disagreement")
    )
    print("  C_F           : direct consistency of the actually used central measurement batch")
    print("  omega_B       : WBF(C_1^iso,...,C_N^iso,C_F), nominal/base track consistency")
    print("  omega_strict  : AND(C_1^iso,...,C_N^iso,C_F), conditional disagreement branch")
    print("  omega_C,ij    : Deduction(G_ij; omega_B, omega_strict) for every unique pair i<j")
    print("  omega_C       : WBF of all evidence-supported omega_C,ij (N=1 -> omega_B)")
    print("  omega_A       : ABF(A_1,...,A_N), aggregated expected-information availability")
    print("  omega_T       : trust_discount(omega_A, omega_C), current-track trustworthiness")
    print("  omega_H       : omega_C * omega_A, tracking-system health")


# =============================================================================
# Monte-Carlo / CSV export helpers
# =============================================================================


def _safe_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _base_rate_from_components(
    belief_value: float,
    uncertainty_value: float,
    projected_probability: float,
) -> float:
    if np.isfinite(uncertainty_value) and uncertainty_value > 1e-12:
        estimate = (projected_probability - belief_value) / uncertainty_value
        return float(np.clip(estimate, 0.0, 1.0))
    return 0.5


def opinion_record_from_components(
    *,
    scenario_name: str,
    run_index: int,
    time_s: float,
    channel_type: str,
    channel_id: str,
    belief_value: float,
    disbelief_value: float,
    uncertainty_value: float,
    projected_probability: float,
    base_rate: float | None = None,
) -> dict[str, object]:
    b = _safe_float(belief_value)
    d = _safe_float(disbelief_value)
    u = _safe_float(uncertainty_value)
    p = _safe_float(projected_probability)
    a = (
        _base_rate_from_components(b, u, p)
        if base_rate is None
        else _safe_float(base_rate)
    )
    b_norm = normalized_belief_from_components(b, u)
    d_norm = normalized_disbelief_from_components(d, u)
    return {
        "scenario": scenario_name,
        "run": int(run_index),
        "time_s": float(time_s),
        "channel_type": channel_type,
        "channel_id": channel_id,
        "b": b,
        "d": d,
        "u": u,
        "a": a,
        "p_ok": p,
        "b_norm": b_norm,
        "d_norm": d_norm,
        "selected_score": b_norm if USE_NORMALIZED_BELIEF else d_norm,
    }


def opinion_record_from_opinion(
    *,
    scenario_name: str,
    run_index: int,
    time_s: float,
    channel_type: str,
    channel_id: str,
    opinion,
) -> dict[str, object]:
    return opinion_record_from_components(
        scenario_name=scenario_name,
        run_index=run_index,
        time_s=time_s,
        channel_type=channel_type,
        channel_id=channel_id,
        belief_value=belief(opinion),
        disbelief_value=disbelief(opinion),
        uncertainty_value=uncertainty(opinion),
        projected_probability=p_ok(opinion),
        base_rate=prior_ok(opinion),
    )


def iter_opinion_records(
    result: ProcessingResult,
    *,
    scenario_name: str,
    run_index: int,
):
    """Yield every SL opinion time series in a single LaTeX-friendly schema."""
    for sensor_id, state in sorted(result.isolated_states.items()):
        for time_s, b, d, u, p in zip(
            state.event_times_s,
            state.belief_events,
            state.disbelief_events,
            state.uncertainty_events,
            state.p_ok_events,
        ):
            yield opinion_record_from_components(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="C_iso",
                channel_id=f"S{sensor_id}",
                belief_value=b,
                disbelief_value=d,
                uncertainty_value=u,
                projected_probability=p,
            )

    for sensor_id, state in sorted(result.common_prediction_states.items()):
        for time_s, b, d, u, p in zip(
            state.event_times_s,
            state.belief_events,
            state.disbelief_events,
            state.uncertainty_events,
            state.p_ok_events,
        ):
            yield opinion_record_from_components(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="C_common",
                channel_id=f"S{sensor_id}",
                belief_value=b,
                disbelief_value=d,
                uncertainty_value=u,
                projected_probability=p,
            )

    for sensor_id, state in sorted(result.availability_states.items()):
        for time_s, b, d, u, p in zip(
            state.event_times_s,
            state.belief_events,
            state.disbelief_events,
            state.uncertainty_events,
            state.p_available_events,
        ):
            yield opinion_record_from_components(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="A",
                channel_id=f"S{sensor_id}",
                belief_value=b,
                disbelief_value=d,
                uncertainty_value=u,
                projected_probability=p,
            )

    for pair, state in sorted(result.disagreement_states.items()):
        pair_id = f"S{pair[0]}_S{pair[1]}"
        for time_s, b, d, u, p in zip(
            state.event_times_s,
            state.belief_events,
            state.disbelief_events,
            state.uncertainty_events,
            state.p_ok_events,
        ):
            yield opinion_record_from_components(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="G",
                channel_id=pair_id,
                belief_value=b,
                disbelief_value=d,
                uncertainty_value=u,
                projected_probability=p,
            )

    union_histories = {
        "omega_B": result.track_base_history,
        "omega_strict": result.track_strict_history,
        "omega_C": result.track_consistency_history,
        "omega_A": result.combined_availability_history,
        "omega_T": result.track_output_trust_history,
        "omega_H": result.system_health_history,
        "C_F": result.batch_history,
        "C_common_ABF": result.common_abf_history,
    }
    for channel_type, opinions in union_histories.items():
        for time_s, op in zip(result.event_times_s, opinions):
            yield opinion_record_from_opinion(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type=channel_type,
                channel_id="overall",
                opinion=op,
            )

    for sensor_id, opinions in sorted(result.trusted_sensor_history.items()):
        for time_s, op in zip(result.event_times_s, opinions):
            yield opinion_record_from_opinion(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="C_iso_availability_discounted",
                channel_id=f"S{sensor_id}",
                opinion=op,
            )

    for pair, opinions in sorted(result.pair_conditioned_history.items()):
        pair_id = f"S{pair[0]}_S{pair[1]}"
        for time_s, op in zip(result.event_times_s, opinions):
            yield opinion_record_from_opinion(
                scenario_name=scenario_name,
                run_index=run_index,
                time_s=time_s,
                channel_type="omega_C_pair",
                channel_id=pair_id,
                opinion=op,
            )


def iter_diagnostic_records(
    result: ProcessingResult,
    *,
    scenario_name: str,
    run_index: int,
):
    """Yield non-opinion diagnostics (NIS/PIT/residuals/position error/Griebel)."""
    for sensor_id, state in sorted(result.isolated_states.items()):
        for values in zip(
            state.event_times_s,
            state.radial_pit_events,
            state.nis_events,
            state.nis_average,
            state.nis_lower,
            state.nis_upper,
        ):
            time_s, pit, nis, nis_avg, nis_low, nis_high = values
            yield {
                "scenario": scenario_name,
                "run": int(run_index),
                "time_s": float(time_s),
                "diagnostic_type": "sensor_iso",
                "diagnostic_id": f"S{sensor_id}",
                "pit": pit,
                "nis": nis,
                "nis_average": nis_avg,
                "nis_lower": nis_low,
                "nis_upper": nis_high,
                "standardized_x": np.nan,
                "standardized_y": np.nan,
                "mean_standardized_x": np.nan,
                "mean_standardized_y": np.nan,
                "degrees_of_freedom": np.nan,
                "active_sensor_count": np.nan,
                "active_sensor_set": "",
                "position_error_m": np.nan,
                "griebel_delta": np.nan,
                "griebel_uncertainty": np.nan,
                "griebel_eta": np.nan,
            }

    for pair, state in sorted(result.disagreement_states.items()):
        pair_id = f"S{pair[0]}_S{pair[1]}"
        for values in zip(
            state.event_times_s,
            state.radial_nis_events,
            state.standardized_x_events,
            state.standardized_y_events,
            state.mean_standardized_x_events,
            state.mean_standardized_y_events,
        ):
            time_s, nis, sx, sy, msx, msy = values
            yield {
                "scenario": scenario_name,
                "run": int(run_index),
                "time_s": float(time_s),
                "diagnostic_type": "pair",
                "diagnostic_id": pair_id,
                "pit": np.nan,
                "nis": nis,
                "nis_average": np.nan,
                "nis_lower": np.nan,
                "nis_upper": np.nan,
                "standardized_x": sx,
                "standardized_y": sy,
                "mean_standardized_x": msx,
                "mean_standardized_y": msy,
                "degrees_of_freedom": np.nan,
                "active_sensor_count": np.nan,
                "active_sensor_set": "",
                "position_error_m": np.nan,
                "griebel_delta": np.nan,
                "griebel_uncertainty": np.nan,
                "griebel_eta": np.nan,
            }

    batch = result.batch_track_state
    for values in zip(
        batch.event_times_s,
        batch.pit_events,
        batch.nis_events,
        batch.degrees_of_freedom,
        batch.active_sensor_counts,
        batch.active_sensor_sets,
    ):
        time_s, pit, nis, dof, count, sensor_set = values
        yield {
            "scenario": scenario_name,
            "run": int(run_index),
            "time_s": float(time_s),
            "diagnostic_type": "central_batch",
            "diagnostic_id": "C_F",
            "pit": pit,
            "nis": nis,
            "nis_average": np.nan,
            "nis_lower": np.nan,
            "nis_upper": np.nan,
            "standardized_x": np.nan,
            "standardized_y": np.nan,
            "mean_standardized_x": np.nan,
            "mean_standardized_y": np.nan,
            "degrees_of_freedom": dof,
            "active_sensor_count": count,
            "active_sensor_set": sensor_set,
            "position_error_m": np.nan,
            "griebel_delta": np.nan,
            "griebel_uncertainty": np.nan,
            "griebel_eta": np.nan,
        }

    for time_s, active_count, error in zip(
        result.event_times_s,
        result.active_batch_sizes,
        result.position_error,
    ):
        yield {
            "scenario": scenario_name,
            "run": int(run_index),
            "time_s": float(time_s),
            "diagnostic_type": "track",
            "diagnostic_id": "central_track",
            "pit": np.nan,
            "nis": np.nan,
            "nis_average": np.nan,
            "nis_lower": np.nan,
            "nis_upper": np.nan,
            "standardized_x": np.nan,
            "standardized_y": np.nan,
            "mean_standardized_x": np.nan,
            "mean_standardized_y": np.nan,
            "degrees_of_freedom": np.nan,
            "active_sensor_count": active_count,
            "active_sensor_set": "",
            "position_error_m": error,
            "griebel_delta": np.nan,
            "griebel_uncertainty": np.nan,
            "griebel_eta": np.nan,
        }

    for sensor_id in sorted(result.griebel.sensor_ids):
        for time_s, delta, eta, u in zip(
            result.griebel.sensor_times[sensor_id],
            result.griebel.sensor_delta[sensor_id],
            result.griebel.sensor_eta[sensor_id],
            result.griebel.sensor_uncertainty[sensor_id],
        ):
            yield {
                "scenario": scenario_name,
                "run": int(run_index),
                "time_s": float(time_s),
                "diagnostic_type": "griebel_common_prior",
                "diagnostic_id": f"S{sensor_id}",
                "pit": np.nan,
                "nis": np.nan,
                "nis_average": np.nan,
                "nis_lower": np.nan,
                "nis_upper": np.nan,
                "standardized_x": np.nan,
                "standardized_y": np.nan,
                "mean_standardized_x": np.nan,
                "mean_standardized_y": np.nan,
                "degrees_of_freedom": np.nan,
                "active_sensor_count": np.nan,
                "active_sensor_set": "",
                "position_error_m": np.nan,
                "griebel_delta": delta,
                "griebel_uncertainty": u,
                "griebel_eta": eta,
            }


def _write_dict_rows(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_result_csv_bundle(
    result: ProcessingResult,
    output_dir: Path | str,
    *,
    scenario_name: str,
    run_index: int,
) -> tuple[Path, Path]:
    """Write exhaustive per-run raw opinion and diagnostic CSV files."""
    output_dir = Path(output_dir)
    prefix = f"run_{int(run_index):04d}"
    opinion_path = output_dir / f"{prefix}_opinions.csv"
    diagnostic_path = output_dir / f"{prefix}_diagnostics.csv"
    _write_dict_rows(
        opinion_path,
        [
            "scenario", "run", "time_s", "channel_type", "channel_id",
            "b", "d", "u", "a", "p_ok", "b_norm", "d_norm", "selected_score",
        ],
        iter_opinion_records(result, scenario_name=scenario_name, run_index=run_index),
    )
    _write_dict_rows(
        diagnostic_path,
        [
            "scenario", "run", "time_s", "diagnostic_type", "diagnostic_id",
            "pit", "nis", "nis_average", "nis_lower", "nis_upper",
            "standardized_x", "standardized_y", "mean_standardized_x",
            "mean_standardized_y", "degrees_of_freedom", "active_sensor_count",
            "active_sensor_set", "position_error_m", "griebel_delta",
            "griebel_uncertainty", "griebel_eta",
        ],
        iter_diagnostic_records(result, scenario_name=scenario_name, run_index=run_index),
    )
    return opinion_path, diagnostic_path


def _interval_masks_for_times(times: np.ndarray):
    """Yield (name, mask) for nominal operation and every active disturbance."""
    nominal_mask = np.asarray([is_nominal_time(float(t)) for t in times], dtype=bool)
    yield "nominal", nominal_mask
    for start_s, end_s, label, _ in DISTURBANCE_INTERVALS:
        yield label, (times >= float(start_s)) & (times < float(end_s))


def collect_interval_metric_records(
    result: ProcessingResult,
    *,
    scenario_name: str,
    run_index: int,
) -> list[dict[str, object]]:
    """Return per-run interval metrics in a generic long format.

    This is the primary Monte-Carlo table.  The driver aggregates ``value`` over
    runs and writes mean/std/SEM/95%-CI columns that can be consumed directly by
    pgfplotstable/LaTeX.
    """
    records = list(iter_opinion_records(
        result,
        scenario_name=scenario_name,
        run_index=run_index,
    ))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["channel_type"]), str(row["channel_id"]))].append(row)

    output: list[dict[str, object]] = []
    opinion_metrics = ("b", "d", "u", "p_ok", "b_norm", "d_norm", "selected_score")
    for (channel_type, channel_id), rows in grouped.items():
        times = np.asarray([row["time_s"] for row in rows], dtype=float)
        for interval_name, mask in _interval_masks_for_times(times):
            if not np.any(mask):
                continue
            for metric in opinion_metrics:
                values = np.asarray([row[metric] for row in rows], dtype=float)[mask]
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                output.append({
                    "scenario": scenario_name,
                    "run": int(run_index),
                    "interval": interval_name,
                    "channel_type": channel_type,
                    "channel_id": channel_id,
                    "metric": metric,
                    "value": float(np.mean(finite)),
                    "within_run_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "n_samples": int(finite.size),
                })

    # Track-position metrics are useful primary-performance context and are kept
    # in the same long-format table.
    track_times = np.asarray(result.event_times_s, dtype=float)
    errors = np.asarray(result.position_error, dtype=float)
    for interval_name, mask in _interval_masks_for_times(track_times):
        finite = errors[mask & np.isfinite(errors)]
        if finite.size == 0:
            continue
        for metric, value in (
            ("position_error_mean_m", float(np.mean(finite))),
            ("position_error_rmse_m", float(np.sqrt(np.mean(finite ** 2)))),
            ("position_error_max_m", float(np.max(finite))),
        ):
            output.append({
                "scenario": scenario_name,
                "run": int(run_index),
                "interval": interval_name,
                "channel_type": "track",
                "channel_id": "central_track",
                "metric": metric,
                "value": value,
                "within_run_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "n_samples": int(finite.size),
            })

    # Native Griebel reference metrics, if available in this run.
    for sensor_id in sorted(result.griebel.sensor_ids):
        times = np.asarray(result.griebel.sensor_times[sensor_id], dtype=float)
        if times.size == 0:
            continue
        series = {
            "griebel_delta": np.asarray(result.griebel.sensor_delta[sensor_id], dtype=float),
            "griebel_uncertainty": np.asarray(result.griebel.sensor_uncertainty[sensor_id], dtype=float),
            "griebel_eta": np.asarray(result.griebel.sensor_eta[sensor_id], dtype=float),
        }
        for interval_name, mask in _interval_masks_for_times(times):
            if not np.any(mask):
                continue
            for metric, values in series.items():
                finite = values[mask & np.isfinite(values)]
                if finite.size == 0:
                    continue
                output.append({
                    "scenario": scenario_name,
                    "run": int(run_index),
                    "interval": interval_name,
                    "channel_type": "griebel_common_prior",
                    "channel_id": f"S{sensor_id}",
                    "metric": metric,
                    "value": float(np.mean(finite)),
                    "within_run_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "n_samples": int(finite.size),
                })
    return output


def run_simulation(
    *,
    overrides: dict[str, object] | None = None,
    truth_seed: int | None = None,
    sensor_seed_base: int | None = None,
) -> tuple[ScenarioData, ProcessingResult]:
    """Programmatic entry point used by the separate Monte-Carlo driver."""
    if overrides:
        apply_configuration(overrides)
    else:
        refresh_derived_configuration()
    if truth_seed is not None or sensor_seed_base is not None:
        set_run_seeds(
            RANDOM_SEED_TRUTH if truth_seed is None else int(truth_seed),
            RANDOM_SEED_SENSOR_BASE if sensor_seed_base is None else int(sensor_seed_base),
        )
    scenario = build_scenario()
    result = process_scenario(scenario)
    return scenario, result


def main() -> None:
    refresh_derived_configuration()
    rates = configured_sensor_rates()
    print("=" * 88)
    print("SCRIPT BUILD: V9_MC_READY_N_SENSOR_2026-08-12")
    print("SELF-ASSESSMENT PIPELINE: DYNAMIC N-SENSOR TRACK TRUST + SYSTEM HEALTH")
    print(
        "Normalized plot score: "
        + ("b_norm=b/(1-u) [consistency/agreement]" if USE_NORMALIZED_BELIEF
           else "d_norm=d/(1-u) [inconsistency/disagreement]")
    )
    print(
        "omega_B=WBF(C1_iso,...,CN_iso,C_F); "
        "omega_C,ij=Deduction(Gij; omega_B, omega_strict); "
        "omega_C=WBF(pair deductions); omega_A=ABF(A1,...,AN); "
        "omega_T=omega_A(*)omega_C; omega_H=omega_C*omega_A"
    )
    rate_text = ", ".join(f"S{i}={rate:g} Hz" for i, rate in rates.items())
    print(f"Configured sensors: N={NUM_SENSORS}; {rate_text}")
    if SYNCHRONOUS_SENSOR_SPECIAL_CASE:
        print("Mode: synchronous")
    elif CROSS_CONTAMINATION_RATE_STRESS_TEST:
        print("Mode: asynchronous cross-contamination rate stress")
    else:
        print("Mode: asynchronous multi-rate")
    if ENABLE_SENSOR_DROPOUT:
        print(f"Availability disturbance: S{DROPOUT_SENSOR_ID} dropout enabled")
    print(f"Filter motion model: {'CT' if USE_CT_MODEL else 'CV'}")
    print("=" * 88)

    scenario = build_scenario()
    result = process_scenario(scenario)
    print_summary(result)

    if SHOW_DYNAMIC_ANIMATION:
        if NUM_SENSORS <= 4:
            show_dynamic_animation(scenario, result)
        else:
            print(
                "Dynamic Plotly animation skipped for N>4; the static N-sensor "
                "agreement/deduction matrix figures remain available."
            )
    if SHOW_MATPLOTLIB_PLOTS:
        plot_static_results(result)
        plt.show()


if __name__ == "__main__":
    main()