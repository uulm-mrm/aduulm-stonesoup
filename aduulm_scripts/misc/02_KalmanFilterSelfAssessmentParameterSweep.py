#!/usr/bin/env python3
"""
Parameter sweep for the PIT/Subjective-Logic Kalman-filter self-assessment.

The workflow is intentionally split into two stages:

1. Generate raw Kalman-filter/PIT caches for multiple Monte-Carlo seeds.
   These caches contain quantities that do not depend on the SL mapping or
   LTST parameters.

2. Rebuild the SL mapping for many combinations of
       W, n_ST, discount, base rate a
   and evaluate many decision policies
       eta, tau_OK
   without rerunning the Kalman filter.

Examples
--------
Generate 20 Monte-Carlo caches:

    python 02_KalmanFilterSelfAssessmentParameterSweep.py generate \
        --simulation-script 01_KalmanFilterWithSelfAssessmentV7_SweepReady.py \
        --output-dir sweep_caches \
        --seeds 0:20

Run the sweep:

    python 02_KalmanFilterSelfAssessmentParameterSweep.py sweep \
        --cache-glob "sweep_caches/*.npz" \
        --output-dir sweep_results

The script writes run-level, event-level, transition-level, aggregated,
ranked, and Pareto-front CSV files plus transition-focused trade-off plots.

The primary ranking objective is to obtain confidently correct decisions in
steady nominal and disturbed intervals. The undecided state is treated as a
transition state and should therefore be rare outside known event-onset and
event-recovery windows. If no parameter combination meets the explicit design
requirements, the script reports this clearly and ranks candidates by the
severity of their normalized deficits instead of presenting an infeasible
candidate as satisfactory.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import beta, rankdata

import subjective_logic as sl


REGION_INCONSISTENT = -1
REGION_UNDECIDED = 0
REGION_CONSISTENT = 1


# ============================================================================
# Generic helpers
# ============================================================================


def parse_number_list(text: str, cast=float) -> list:
    """Parses comma-separated numeric values."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_seed_spec(text: str) -> list[int]:
    """Parses either '0,1,2' or a half-open range such as '0:20'."""
    text = text.strip()
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "Seed ranges must be START:STOP or START:STOP:STEP."
            )
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        return list(range(start, stop, step))
    return parse_number_list(text, cast=int)


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def mean_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def median_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmedian(array))


def quantile_or_nan(values: Sequence[float], q: float) -> float:
    """Returns a finite-sample quantile or NaN for an empty input."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def harmonic_mean_or_nan(x: float, y: float) -> float:
    """Harmonic mean used for transition-localization precision and recall."""
    if not np.isfinite(x) or not np.isfinite(y) or x + y <= 0.0:
        return float("nan")
    return float(2.0 * x * y / (x + y))


# ============================================================================
# Cache generation
# ============================================================================


def find_project_root(simulation_script: Path) -> Path:
    """Return the repository root that contains the aduulm_scripts package."""
    for candidate in (simulation_script.parent, *simulation_script.parents):
        if (candidate / "aduulm_scripts").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate the repository root containing 'aduulm_scripts'. "
        f"Simulation script: {simulation_script}"
    )


def generate_caches(args: argparse.Namespace) -> None:
    simulation_script = Path(args.simulation_script).resolve()
    if not simulation_script.exists():
        raise FileNotFoundError(simulation_script)

    project_root = find_project_root(simulation_script)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_spec(args.seeds)
    if not seeds:
        raise ValueError("No seeds selected.")

    print(f"Generating {len(seeds)} cache file(s) in {output_dir}")

    for run_index, seed in enumerate(seeds, start=1):
        cache_path = output_dir / f"kf_sa_seed_{seed:04d}.npz"
        if cache_path.exists() and not args.overwrite:
            print(f"[{run_index}/{len(seeds)}] Skip existing {cache_path.name}")
            continue

        env = os.environ.copy()
        env["SA_SWEEP_EXPORT_ONLY"] = "1"
        env["SA_SWEEP_CACHE_PATH"] = str(cache_path)
        env["SA_SIMULATION_SEED"] = str(seed)
        env["SA_MEASUREMENT_SEED"] = str(seed + args.measurement_seed_offset)

        # The simulation is launched as a file from aduulm_scripts/misc.
        # Therefore, Python would otherwise only add that directory to
        # sys.path and imports such as
        #     from aduulm_scripts.utils...
        # would fail. Prepend the repository root explicitly while preserving
        # any existing PYTHONPATH entries from the active virtual environment.
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root)
        if existing_pythonpath:
            env["PYTHONPATH"] += os.pathsep + existing_pythonpath

        print(f"[{run_index}/{len(seeds)}] Seed {seed}")
        subprocess.run(
            [sys.executable, str(simulation_script)],
            cwd=str(simulation_script.parent),
            env=env,
            check=True,
        )

    print("Cache generation finished.")


# ============================================================================
# Cache representation
# ============================================================================


@dataclass(frozen=True)
class SweepCache:
    path: Path
    radial_pit: np.ndarray
    component_x_pit: np.ndarray
    component_y_pit: np.ndarray
    fault_mask: np.ndarray
    event_id: np.ndarray
    position_error: np.ndarray
    metadata: dict

    @property
    def simulation_seed(self) -> int:
        return int(self.metadata.get("simulation_seed", -1))

    @property
    def dt_seconds(self) -> float:
        return float(self.metadata.get("dt_seconds", 1.0))

    @property
    def events(self) -> list[dict]:
        return list(self.metadata.get("events", []))



def load_cache(path: str | Path) -> SweepCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        cache = SweepCache(
            path=path,
            radial_pit=np.asarray(data["radial_pit"], dtype=float),
            component_x_pit=np.asarray(data["component_x_pit"], dtype=float),
            component_y_pit=np.asarray(data["component_y_pit"], dtype=float),
            fault_mask=np.asarray(data["fault_mask"], dtype=bool),
            event_id=np.asarray(data["event_id"], dtype=int),
            position_error=np.asarray(data["position_error"], dtype=float),
            metadata=metadata,
        )

    lengths = {
        len(cache.radial_pit),
        len(cache.component_x_pit),
        len(cache.component_y_pit),
        len(cache.fault_mask),
        len(cache.event_id),
    }
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths in cache {path}")

    return cache


# ============================================================================
# Subjective-Logic mapping
# ============================================================================


def calculate_lt_evidence(num_bins: int, discount: float) -> int:
    return int(
        (
            -(num_bins - 1)
            + np.sqrt(
                (num_bins - 1) ** 2
                + (4.0 * num_bins) / (1.0 - discount + 1e-12)
            )
        )
        / 2.0
    )



def load_threshold_table(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        warnings.warn(
            f"Threshold table {path} does not exist. Falling back to 1.0. "
            "This is normally harmless while LTST conflict handling is disabled."
        )
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): float(value) for key, value in raw.items()}



def resolve_ltst_threshold(
    num_bins: int,
    discount: float,
    threshold_table: dict[str, float],
    threshold_alpha: float,
) -> float:
    effective_evidence = calculate_lt_evidence(num_bins, discount)
    candidates = [
        f"{num_bins}, {effective_evidence}, {threshold_alpha}",
        f"{num_bins}, {effective_evidence}, {threshold_alpha:g}",
    ]
    for key in candidates:
        if key in threshold_table:
            return float(threshold_table[key])

    # In the supplied baseline HANDLE_ST_CONFLICT=False. The threshold is then
    # not intended as a tuning variable for the mapping output. A permissive
    # fallback keeps the sweep operational if a table entry is unavailable.
    return 1.0



def opinion_class(num_bins: int):
    return getattr(sl, f"Opinion{num_bins}d")



def dirichlet_class(num_bins: int):
    return getattr(sl, f"DirichletDistribution{num_bins}d")



def ltst_class(num_bins: int):
    return getattr(sl, f"LongShortTermMemory{num_bins}d")



def scalar_pit_to_opinion(value: float, num_bins: int):
    value = float(np.clip(value, 0.0, 1.0))
    evidence = np.zeros(num_bins, dtype=float)
    index = min(int(np.floor(value * num_bins)), num_bins - 1)
    evidence[index] = 1.0
    return dirichlet_class(num_bins).from_evidences(evidence).as_opinion()



def multinomial_to_binomial_consistency(
    opinion,
    num_bins: int,
    positive_base_rate: float,
    eps: float = 1e-12,
):
    """The normalized-TV mapping used in the supplied V7 implementation."""
    uncertainty = float(opinion.uncertainty())
    committed_mass = 1.0 - uncertainty

    if committed_mass <= eps:
        result = sl.Opinion2d(0.0, 0.0)
        result.prior_belief_masses = [
            positive_base_rate,
            1.0 - positive_base_rate,
        ]
        return result

    uniform_reference = np.full(num_bins, 1.0 / num_bins, dtype=float)
    belief = np.asarray(opinion.belief_masses, dtype=float)
    committed_distribution = belief / committed_mass

    total_variation = 0.5 * np.sum(
        np.abs(committed_distribution - uniform_reference)
    )
    maximum_total_variation = 1.0 - 1.0 / num_bins

    disbelief = committed_mass * total_variation / maximum_total_variation
    disbelief = float(np.clip(disbelief, 0.0, committed_mass))
    belief_consistent = committed_mass - disbelief

    result = sl.Opinion2d(belief_consistent, disbelief)
    result.prior_belief_masses = [
        positive_base_rate,
        1.0 - positive_base_rate,
    ]
    return result


@dataclass(frozen=True)
class MappingOutput:
    belief: np.ndarray
    disbelief: np.ndarray
    uncertainty: np.ndarray
    base_rate: np.ndarray
    projected_probability: np.ndarray



def run_mapping(
    cache: SweepCache,
    num_bins: int,
    short_window_size: int,
    discount: float,
    base_rate: float,
    threshold: float,
) -> MappingOutput:
    """
    Reconstructs the radial, component-wise, and fused overall opinions.

    This mirrors the active V7 path:
      - one radial PIT channel,
      - two component PIT channels,
      - one LTST buffer per channel,
      - normalized-TV multinomial-to-binomial mapping,
      - logical multiplication of x and y component opinions,
      - weighted fusion of component and radial opinions.
    """
    if not (0.0 < base_rate < 1.0):
        raise ValueError("base_rate must be in (0, 1).")

    fusion_type = sl.FusionType.CUMULATIVE
    handle_short_term_conflict = False
    average_conflict_handling = False

    memory_type = ltst_class(num_bins)
    memories = [
        memory_type(
            short_window_size,
            threshold,
            discount,
            fusion_type,
            handle_short_term_conflict,
            average_conflict_handling,
        )
        for _ in range(3)
    ]

    component_channel_prior = float(np.sqrt(base_rate))

    n_samples = len(cache.radial_pit)
    belief = np.empty(n_samples, dtype=float)
    disbelief = np.empty(n_samples, dtype=float)
    uncertainty = np.empty(n_samples, dtype=float)
    actual_base_rate = np.empty(n_samples, dtype=float)
    projected_probability = np.empty(n_samples, dtype=float)

    for index, (u_radial, u_x, u_y) in enumerate(
        zip(cache.radial_pit, cache.component_x_pit, cache.component_y_pit)
    ):
        observed = [
            scalar_pit_to_opinion(u_radial, num_bins),
            scalar_pit_to_opinion(u_x, num_bins),
            scalar_pit_to_opinion(u_y, num_bins),
        ]

        for memory, observation in zip(memories, observed):
            memory.add(observation)

        radial_multinomial = memories[0].get_opinion()
        x_multinomial = memories[1].get_opinion()
        y_multinomial = memories[2].get_opinion()

        radial_binomial = multinomial_to_binomial_consistency(
            radial_multinomial,
            num_bins,
            positive_base_rate=base_rate,
        )
        x_binomial = multinomial_to_binomial_consistency(
            x_multinomial,
            num_bins,
            positive_base_rate=component_channel_prior,
        )
        y_binomial = multinomial_to_binomial_consistency(
            y_multinomial,
            num_bins,
            positive_base_rate=component_channel_prior,
        )

        component_binomial = x_binomial.multiply(y_binomial)
        overall = sl.Fusion.fuse_opinions(
            sl.FusionType.WEIGHTED,
            [component_binomial, radial_binomial],
        )

        base_rates = np.asarray(overall.prior_belief_masses, dtype=float)
        a = float(base_rates[0])

        belief[index] = float(overall.belief())
        disbelief[index] = float(overall.disbelief())
        uncertainty[index] = float(overall.uncertainty())
        actual_base_rate[index] = a
        projected_probability[index] = belief[index] + a * uncertainty[index]

    return MappingOutput(
        belief=belief,
        disbelief=disbelief,
        uncertainty=uncertainty,
        base_rate=actual_base_rate,
        projected_probability=projected_probability,
    )


# ============================================================================
# Beta-credible decision layer
# ============================================================================


@dataclass(frozen=True)
class DecisionOutput:
    q_consistent: np.ndarray
    q_inconsistent: np.ndarray
    region_code: np.ndarray



def classify_mapping_output(
    mapping: MappingOutput,
    eta: float,
    tau_ok: float,
) -> DecisionOutput:
    if not (0.5 < eta < 1.0):
        raise ValueError("eta must lie in (0.5, 1).")
    if not (0.0 < tau_ok < 1.0):
        raise ValueError("tau_ok must lie in (0, 1).")

    b = mapping.belief
    d = mapping.disbelief
    u = mapping.uncertainty
    a = mapping.base_rate

    q_consistent = np.empty_like(b)
    q_inconsistent = np.empty_like(b)

    dogmatic = u <= 1e-12
    regular = ~dogmatic

    alpha = 2.0 * b[regular] / u[regular] + 2.0 * a[regular]
    beta_param = (
        2.0 * d[regular] / u[regular]
        + 2.0 * (1.0 - a[regular])
    )

    q_consistent[regular] = beta.sf(tau_ok, alpha, beta_param)
    q_inconsistent[regular] = beta.cdf(tau_ok, alpha, beta_param)

    p = mapping.projected_probability[dogmatic]
    q_consistent[dogmatic] = np.where(
        p > tau_ok,
        1.0,
        np.where(p < tau_ok, 0.0, 0.5),
    )
    q_inconsistent[dogmatic] = 1.0 - q_consistent[dogmatic]

    region_code = np.full(len(b), REGION_UNDECIDED, dtype=int)
    region_code[q_consistent >= eta] = REGION_CONSISTENT
    region_code[q_inconsistent >= eta] = REGION_INCONSISTENT

    return DecisionOutput(
        q_consistent=np.clip(q_consistent, 0.0, 1.0),
        q_inconsistent=np.clip(q_inconsistent, 0.0, 1.0),
        region_code=region_code,
    )


# ============================================================================
# Evaluation metrics
# ============================================================================


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    n_positive = int(np.sum(labels))
    n_negative = int(np.sum(~labels))
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    rank_sum_positive = float(np.sum(ranks[labels]))
    auc = (
        rank_sum_positive - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    return float(auc)



def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    n_positive = int(np.sum(labels))
    if n_positive == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(int)
    cumulative_true_positives = np.cumsum(sorted_labels)
    precision = cumulative_true_positives / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / n_positive)



def first_persistent_index(
    values: np.ndarray,
    target: int,
    start: int,
    end: int,
    persistence: int,
) -> int | None:
    """First index of a complete persistent run in [start, end)."""
    start = max(0, int(start))
    end = min(len(values), int(end))
    persistence = max(1, int(persistence))

    run_length = 0
    run_start = start
    for index in range(start, end):
        if int(values[index]) == target:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= persistence:
                return run_start
        else:
            run_length = 0
    return None



def count_region_transitions(region_code: np.ndarray, mask: np.ndarray) -> int:
    selected = np.asarray(region_code)[np.asarray(mask, dtype=bool)]
    if len(selected) < 2:
        return 0
    return int(np.sum(selected[1:] != selected[:-1]))



def mean_dwell_length(region_code: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(region_code)[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        return float("nan")

    changes = np.flatnonzero(selected[1:] != selected[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(selected)]))
    dwell_lengths = np.diff(boundaries)
    return float(np.mean(dwell_lengths))


def build_transition_mask(
    n_samples: int,
    events: list[dict],
    warmup: int,
    before_steps: int,
    after_steps: int,
) -> tuple[np.ndarray, list[dict]]:
    """Build fixed windows around every known disturbance onset and end."""
    mask = np.zeros(n_samples, dtype=bool)
    boundaries: list[dict] = []

    for event_index, event in enumerate(events):
        specifications = (
            ("onset", int(event["start"]), REGION_CONSISTENT, REGION_INCONSISTENT),
            ("recovery", int(event["end"]), REGION_INCONSISTENT, REGION_CONSISTENT),
        )
        for transition_type, boundary, source_region, target_region in specifications:
            start = max(int(warmup), boundary - int(before_steps))
            end = min(n_samples, boundary + int(after_steps))
            if start >= end:
                continue
            mask[start:end] = True
            boundaries.append({
                "event_index": event_index,
                "event_name": str(event["name"]),
                "transition_type": transition_type,
                "boundary_step": boundary,
                "window_start": start,
                "window_end": end,
                "source_region": source_region,
                "target_region": target_region,
            })

    return mask, boundaries


def transition_evaluation(
    cache: SweepCache,
    decision: DecisionOutput,
    persistence: int,
    warmup: int,
    before_steps: int,
    after_steps: int,
    parameter_values: dict,
) -> tuple[list[dict], dict, np.ndarray]:
    """Evaluate completion and whether region changes pass through undecided."""
    region = np.asarray(decision.region_code, dtype=int)
    transition_mask, boundaries = build_transition_mask(
        n_samples=len(region),
        events=cache.events,
        warmup=warmup,
        before_steps=before_steps,
        after_steps=after_steps,
    )

    rows: list[dict] = []
    completion_values: list[int] = []
    mediated_values: list[int] = []
    direct_values: list[int] = []
    undecided_presence_values: list[int] = []
    settling_delays: list[float] = []
    undecided_steps_values: list[float] = []
    path_lengths: list[float] = []
    onset_mediated: list[int] = []
    recovery_mediated: list[int] = []

    for info in boundaries:
        boundary = int(info["boundary_step"])
        window_start = int(info["window_start"])
        window_end = int(info["window_end"])
        source_region = int(info["source_region"])
        target_region = int(info["target_region"])

        target_index = first_persistent_index(
            region,
            target=target_region,
            start=max(boundary, window_start),
            end=window_end,
            persistence=persistence,
        )
        completed = target_index is not None
        path_start = max(boundary, window_start)
        path_end = int(target_index) if completed else window_end
        path = region[path_start:path_end]
        undecided_steps = int(np.sum(path == REGION_UNDECIDED))
        has_undecided = undecided_steps > 0
        mediated = bool(completed and has_undecided)
        direct = bool(completed and not has_undecided)
        source_present = bool(np.any(
            region[window_start:max(boundary, window_start)] == source_region
        ))

        settling_delay = (
            int(target_index - boundary) if completed else float("nan")
        )
        if completed:
            settling_delays.append(float(settling_delay))

        path_length = max(0, path_end - path_start)
        completion_values.append(int(completed))
        mediated_values.append(int(mediated))
        direct_values.append(int(direct))
        undecided_presence_values.append(int(has_undecided))
        undecided_steps_values.append(float(undecided_steps))
        path_lengths.append(float(path_length))

        if info["transition_type"] == "onset":
            onset_mediated.append(int(mediated))
        else:
            recovery_mediated.append(int(mediated))

        rows.append({
            **parameter_values,
            "simulation_seed": cache.simulation_seed,
            **info,
            "source_present_before_boundary": int(source_present),
            "completed": int(completed),
            "mediated_by_undecided": int(mediated),
            "direct_switch": int(direct),
            "has_undecided_in_transition_path": int(has_undecided),
            "settling_delay_steps": settling_delay,
            "settling_delay_seconds": (
                settling_delay * cache.dt_seconds
                if np.isfinite(settling_delay) else float("nan")
            ),
            "undecided_steps_before_target": undecided_steps,
            "transition_path_length_steps": path_length,
            "undecided_fraction_on_transition_path": safe_rate(
                undecided_steps, path_length
            ),
        })

    summary = {
        "transition_count": len(rows),
        "transition_completion_rate": mean_or_nan(completion_values),
        "mediated_transition_rate": mean_or_nan(mediated_values),
        "direct_transition_rate": mean_or_nan(direct_values),
        "transition_undecided_presence_rate": mean_or_nan(undecided_presence_values),
        "onset_mediated_transition_rate": mean_or_nan(onset_mediated),
        "recovery_mediated_transition_rate": mean_or_nan(recovery_mediated),
        "mean_transition_settling_delay_steps": mean_or_nan(settling_delays),
        "median_transition_settling_delay_steps": median_or_nan(settling_delays),
        "mean_undecided_steps_per_transition": mean_or_nan(undecided_steps_values),
        "mean_transition_path_length_steps": mean_or_nan(path_lengths),
    }
    return rows, summary, transition_mask



def event_evaluation(
    cache: SweepCache,
    decision: DecisionOutput,
    persistence: int,
    warmup: int,
    parameter_values: dict,
) -> tuple[list[dict], dict]:
    event_rows: list[dict] = []
    detection_delays = []
    recovery_delays = []
    detected_count = 0

    events = cache.events
    n_samples = len(decision.region_code)

    for event_index, event in enumerate(events):
        start = max(warmup, int(event["start"]))
        end = min(n_samples, int(event["end"]))
        if start >= end:
            continue

        event_slice = slice(start, end)
        event_regions = decision.region_code[event_slice]

        detection_index = first_persistent_index(
            decision.region_code,
            target=REGION_INCONSISTENT,
            start=start,
            end=end,
            persistence=persistence,
        )
        detected = detection_index is not None
        if detected:
            detected_count += 1
            delay_steps = int(detection_index - int(event["start"]))
            detection_delays.append(delay_steps)
        else:
            delay_steps = float("nan")

        next_start = (
            int(events[event_index + 1]["start"])
            if event_index + 1 < len(events)
            else n_samples
        )
        recovery_index = first_persistent_index(
            decision.region_code,
            target=REGION_CONSISTENT,
            start=int(event["end"]),
            end=next_start,
            persistence=persistence,
        )
        if recovery_index is not None:
            recovery_delay = int(recovery_index - int(event["end"]))
            recovery_delays.append(recovery_delay)
        else:
            recovery_delay = float("nan")

        position_error = cache.position_error[event_slice]

        row = {
            **parameter_values,
            "simulation_seed": cache.simulation_seed,
            "event_index": event_index,
            "event_name": str(event["name"]),
            "event_start": int(event["start"]),
            "event_end": int(event["end"]),
            "detected": int(detected),
            "detection_delay_steps": delay_steps,
            "detection_delay_seconds": (
                delay_steps * cache.dt_seconds if detected else float("nan")
            ),
            "recovery_delay_steps": recovery_delay,
            "recovery_delay_seconds": (
                recovery_delay * cache.dt_seconds
                if np.isfinite(recovery_delay)
                else float("nan")
            ),
            "event_consistent_rate": float(
                np.mean(event_regions == REGION_CONSISTENT)
            ),
            "event_undecided_rate": float(
                np.mean(event_regions == REGION_UNDECIDED)
            ),
            "event_inconsistent_rate": float(
                np.mean(event_regions == REGION_INCONSISTENT)
            ),
            "event_mean_q_inconsistent": float(
                np.mean(decision.q_inconsistent[event_slice])
            ),
            "event_max_q_inconsistent": float(
                np.max(decision.q_inconsistent[event_slice])
            ),
            "event_mean_position_error": mean_or_nan(position_error),
            "event_max_position_error": (
                float(np.nanmax(position_error))
                if np.any(np.isfinite(position_error))
                else float("nan")
            ),
        }
        event_rows.append(row)

    summary = {
        "event_detection_rate": safe_rate(detected_count, len(event_rows)),
        "mean_detection_delay_steps": mean_or_nan(detection_delays),
        "median_detection_delay_steps": median_or_nan(detection_delays),
        "worst_detection_delay_steps": (
            float(np.max(detection_delays)) if detection_delays else float("nan")
        ),
        "mean_detection_delay_seconds": (
            mean_or_nan(detection_delays) * cache.dt_seconds
            if detection_delays
            else float("nan")
        ),
        "mean_recovery_delay_steps": mean_or_nan(recovery_delays),
        "mean_recovery_delay_seconds": (
            mean_or_nan(recovery_delays) * cache.dt_seconds
            if recovery_delays
            else float("nan")
        ),
        "undetected_event_count": int(len(event_rows) - detected_count),
    }

    return event_rows, summary



def evaluate_run(
    cache: SweepCache,
    mapping: MappingOutput,
    decision: DecisionOutput,
    num_bins: int,
    short_window_size: int,
    discount: float,
    base_rate: float,
    eta: float,
    tau_ok: float,
    persistence: int,
    minimum_warmup: int,
    transition_before_steps: int,
    transition_after_steps: int,
) -> tuple[dict, list[dict], list[dict]]:
    n_samples = len(decision.region_code)
    warmup = min(n_samples, max(int(minimum_warmup), int(short_window_size)))

    evaluation_mask = np.arange(n_samples) >= warmup
    nominal_mask = evaluation_mask & ~cache.fault_mask
    fault_mask = evaluation_mask & cache.fault_mask
    region = np.asarray(decision.region_code, dtype=int)
    decided_mask = evaluation_mask & (region != REGION_UNDECIDED)

    correct_decision = (
        (nominal_mask & (region == REGION_CONSISTENT))
        | (fault_mask & (region == REGION_INCONSISTENT))
    )

    parameter_values = {
        "num_bins": int(num_bins),
        "short_window_size": int(short_window_size),
        "discount": float(discount),
        "base_rate": float(base_rate),
        "eta": float(eta),
        "tau_ok": float(tau_ok),
    }

    event_rows, event_summary = event_evaluation(
        cache=cache,
        decision=decision,
        persistence=persistence,
        warmup=warmup,
        parameter_values=parameter_values,
    )
    transition_rows, transition_summary, transition_mask = transition_evaluation(
        cache=cache,
        decision=decision,
        persistence=persistence,
        warmup=warmup,
        before_steps=transition_before_steps,
        after_steps=transition_after_steps,
        parameter_values=parameter_values,
    )

    steady_mask = evaluation_mask & ~transition_mask
    steady_nominal_mask = steady_mask & ~cache.fault_mask
    steady_fault_mask = steady_mask & cache.fault_mask
    steady_decided_mask = steady_mask & (region != REGION_UNDECIDED)

    nominal_count = int(np.sum(nominal_mask))
    fault_count = int(np.sum(fault_mask))
    decided_count = int(np.sum(decided_mask))
    evaluation_length = int(np.sum(evaluation_mask))
    steady_count = int(np.sum(steady_mask))
    steady_nominal_count = int(np.sum(steady_nominal_mask))
    steady_fault_count = int(np.sum(steady_fault_mask))
    steady_decided_count = int(np.sum(steady_decided_mask))
    transition_sample_count = int(np.sum(evaluation_mask & transition_mask))

    transitions = count_region_transitions(region, evaluation_mask)
    first_nominal_consistent = first_persistent_index(
        region,
        target=REGION_CONSISTENT,
        start=warmup,
        end=(cache.events[0]["start"] if cache.events else n_samples),
        persistence=persistence,
    )

    undecided_mask = evaluation_mask & (region == REGION_UNDECIDED)
    undecided_count = int(np.sum(undecided_mask))
    undecided_in_transition_count = int(np.sum(undecided_mask & transition_mask))
    undecided_outside_transition_count = int(np.sum(undecided_mask & ~transition_mask))

    steady_correct_mask = (
        (steady_nominal_mask & (region == REGION_CONSISTENT))
        | (steady_fault_mask & (region == REGION_INCONSISTENT))
    )

    steady_nominal_consistent = safe_rate(
        np.sum(steady_nominal_mask & (region == REGION_CONSISTENT)),
        steady_nominal_count,
    )
    steady_fault_inconsistent = safe_rate(
        np.sum(steady_fault_mask & (region == REGION_INCONSISTENT)),
        steady_fault_count,
    )

    steady_nominal_undecided = safe_rate(
        np.sum(steady_nominal_mask & (region == REGION_UNDECIDED)),
        steady_nominal_count,
    )
    steady_fault_undecided = safe_rate(
        np.sum(steady_fault_mask & (region == REGION_UNDECIDED)),
        steady_fault_count,
    )
    steady_nominal_false_inconsistent = safe_rate(
        np.sum(steady_nominal_mask & (region == REGION_INCONSISTENT)),
        steady_nominal_count,
    )
    steady_fault_false_consistent = safe_rate(
        np.sum(steady_fault_mask & (region == REGION_CONSISTENT)),
        steady_fault_count,
    )

    steady_worst_class_correct = min(
        steady_nominal_consistent,
        steady_fault_inconsistent,
    )
    steady_balanced_undecided = 0.5 * (
        steady_nominal_undecided + steady_fault_undecided
    )
    steady_confident_error = safe_rate(
        np.sum(steady_nominal_mask & (region == REGION_INCONSISTENT))
        + np.sum(steady_fault_mask & (region == REGION_CONSISTENT)),
        steady_count,
    )

    undecided_localization_precision = safe_rate(
        undecided_in_transition_count,
        undecided_count,
    )
    transition_undecided_recall = safe_rate(
        undecided_in_transition_count,
        transition_sample_count,
    )
    undecided_transition_f1 = harmonic_mean_or_nan(
        undecided_localization_precision,
        transition_undecided_recall,
    )

    q_consistent_steady_nominal = decision.q_consistent[steady_nominal_mask]
    q_inconsistent_steady_fault = decision.q_inconsistent[steady_fault_mask]

    # Diagnostic eta ceilings: to classify at least p of a class confidently,
    # eta must not exceed the (1-p)-quantile of the corresponding posterior
    # credibility values. These metrics reveal whether a fixed eta is
    # achievable at all for the generated opinions.
    eta_ceiling_90pct_nominal_consistent = quantile_or_nan(
        q_consistent_steady_nominal, 0.10
    )
    eta_ceiling_80pct_fault_inconsistent = quantile_or_nan(
        q_inconsistent_steady_fault, 0.20
    )

    row = {
        **parameter_values,
        "simulation_seed": cache.simulation_seed,
        "cache_file": cache.path.name,
        "warmup_steps": warmup,
        "transition_window_before_steps": int(transition_before_steps),
        "transition_window_after_steps": int(transition_after_steps),
        "nominal_sample_count": nominal_count,
        "fault_sample_count": fault_count,
        "steady_sample_count": steady_count,
        "transition_sample_count": transition_sample_count,

        "nominal_consistent_rate": safe_rate(np.sum(nominal_mask & (region == REGION_CONSISTENT)), nominal_count),
        "nominal_undecided_rate": safe_rate(np.sum(nominal_mask & (region == REGION_UNDECIDED)), nominal_count),
        "nominal_false_inconsistent_rate": safe_rate(np.sum(nominal_mask & (region == REGION_INCONSISTENT)), nominal_count),
        "fault_false_consistent_rate": safe_rate(np.sum(fault_mask & (region == REGION_CONSISTENT)), fault_count),
        "fault_undecided_rate": safe_rate(np.sum(fault_mask & (region == REGION_UNDECIDED)), fault_count),
        "fault_inconsistent_rate": safe_rate(np.sum(fault_mask & (region == REGION_INCONSISTENT)), fault_count),

        "steady_nominal_consistent_rate": steady_nominal_consistent,
        "steady_nominal_undecided_rate": steady_nominal_undecided,
        "steady_nominal_false_inconsistent_rate": steady_nominal_false_inconsistent,
        "steady_fault_false_consistent_rate": steady_fault_false_consistent,
        "steady_fault_undecided_rate": steady_fault_undecided,
        "steady_fault_inconsistent_rate": steady_fault_inconsistent,
        "steady_undecided_rate": safe_rate(undecided_outside_transition_count, steady_count),
        "steady_balanced_undecided_rate": steady_balanced_undecided,
        "steady_balanced_correct_rate": 0.5 * (steady_nominal_consistent + steady_fault_inconsistent),
        "steady_worst_class_correct_rate": steady_worst_class_correct,
        "steady_confident_error_rate": steady_confident_error,
        "steady_decision_coverage": safe_rate(steady_decided_count, steady_count),
        "steady_selective_accuracy": safe_rate(np.sum(steady_correct_mask & steady_decided_mask), steady_decided_count),

        "undecided_total_count": undecided_count,
        "undecided_in_transition_count": undecided_in_transition_count,
        "undecided_outside_transition_count": undecided_outside_transition_count,
        "undecided_localization_precision": undecided_localization_precision,
        "transition_undecided_sample_rate": transition_undecided_recall,
        "undecided_transition_f1": undecided_transition_f1,

        "mean_q_consistent_steady_nominal": mean_or_nan(q_consistent_steady_nominal),
        "median_q_consistent_steady_nominal": median_or_nan(q_consistent_steady_nominal),
        "q05_consistent_steady_nominal": quantile_or_nan(q_consistent_steady_nominal, 0.05),
        "q10_consistent_steady_nominal": quantile_or_nan(q_consistent_steady_nominal, 0.10),
        "mean_q_inconsistent_steady_fault": mean_or_nan(q_inconsistent_steady_fault),
        "median_q_inconsistent_steady_fault": median_or_nan(q_inconsistent_steady_fault),
        "q10_inconsistent_steady_fault": quantile_or_nan(q_inconsistent_steady_fault, 0.10),
        "q20_inconsistent_steady_fault": quantile_or_nan(q_inconsistent_steady_fault, 0.20),
        "eta_ceiling_90pct_nominal_consistent": eta_ceiling_90pct_nominal_consistent,
        "eta_ceiling_80pct_fault_inconsistent": eta_ceiling_80pct_fault_inconsistent,
        "nominal_consistency_credibility_margin_q10": eta_ceiling_90pct_nominal_consistent - eta,
        "fault_inconsistency_credibility_margin_q20": eta_ceiling_80pct_fault_inconsistent - eta,

        "decision_coverage": safe_rate(decided_count, evaluation_length),
        "selective_accuracy": safe_rate(np.sum(correct_decision & decided_mask), decided_count),
        "selective_risk": (1.0 - safe_rate(np.sum(correct_decision & decided_mask), decided_count) if decided_count > 0 else float("nan")),
        "balanced_three_state_accuracy": 0.5 * (
            safe_rate(np.sum(nominal_mask & (region == REGION_CONSISTENT)), nominal_count)
            + safe_rate(np.sum(fault_mask & (region == REGION_INCONSISTENT)), fault_count)
        ),

        "q_inconsistent_roc_auc": roc_auc_binary(cache.fault_mask[evaluation_mask], decision.q_inconsistent[evaluation_mask]),
        "q_inconsistent_average_precision": average_precision_binary(cache.fault_mask[evaluation_mask], decision.q_inconsistent[evaluation_mask]),
        "mean_q_inconsistent_nominal": mean_or_nan(decision.q_inconsistent[nominal_mask]),
        "mean_q_inconsistent_fault": mean_or_nan(decision.q_inconsistent[fault_mask]),
        "q_inconsistent_mean_separation": mean_or_nan(decision.q_inconsistent[fault_mask]) - mean_or_nan(decision.q_inconsistent[nominal_mask]),

        "region_transition_count": transitions,
        "region_transitions_per_1000_steps": safe_rate(1000.0 * transitions, evaluation_length),
        "mean_region_dwell_steps": mean_dwell_length(region, evaluation_mask),
        "initial_consistent_delay_steps": (int(first_nominal_consistent) if first_nominal_consistent is not None else float("nan")),
        "mean_position_error_nominal": mean_or_nan(cache.position_error[nominal_mask]),
        "mean_position_error_fault": mean_or_nan(cache.position_error[fault_mask]),
        **event_summary,
        **transition_summary,
    }
    return row, event_rows, transition_rows


# ============================================================================
# Aggregation, ranking, and Pareto front
# ============================================================================


PARAMETER_COLUMNS = [
    "num_bins",
    "short_window_size",
    "discount",
    "base_rate",
    "eta",
    "tau_ok",
]


IMPORTANT_METRICS = [
    "nominal_consistent_rate", "nominal_undecided_rate", "nominal_false_inconsistent_rate",
    "fault_false_consistent_rate", "fault_undecided_rate", "fault_inconsistent_rate",
    "steady_nominal_consistent_rate", "steady_nominal_undecided_rate",
    "steady_nominal_false_inconsistent_rate", "steady_fault_false_consistent_rate",
    "steady_fault_undecided_rate", "steady_fault_inconsistent_rate",
    "steady_undecided_rate", "steady_balanced_undecided_rate",
    "steady_balanced_correct_rate", "steady_worst_class_correct_rate",
    "steady_confident_error_rate", "steady_decision_coverage",
    "steady_selective_accuracy", "undecided_localization_precision",
    "transition_undecided_sample_rate", "undecided_transition_f1",
    "mean_q_consistent_steady_nominal", "median_q_consistent_steady_nominal",
    "q05_consistent_steady_nominal", "q10_consistent_steady_nominal",
    "mean_q_inconsistent_steady_fault", "median_q_inconsistent_steady_fault",
    "q10_inconsistent_steady_fault", "q20_inconsistent_steady_fault",
    "eta_ceiling_90pct_nominal_consistent",
    "eta_ceiling_80pct_fault_inconsistent",
    "nominal_consistency_credibility_margin_q10",
    "fault_inconsistency_credibility_margin_q20",
    "transition_completion_rate", "mediated_transition_rate", "direct_transition_rate",
    "transition_undecided_presence_rate", "onset_mediated_transition_rate",
    "recovery_mediated_transition_rate", "mean_transition_settling_delay_steps",
    "median_transition_settling_delay_steps", "mean_undecided_steps_per_transition",
    "mean_transition_path_length_steps", "decision_coverage", "selective_accuracy",
    "selective_risk", "balanced_three_state_accuracy", "q_inconsistent_roc_auc",
    "q_inconsistent_average_precision", "q_inconsistent_mean_separation",
    "region_transitions_per_1000_steps", "mean_region_dwell_steps",
    "event_detection_rate", "mean_detection_delay_steps", "median_detection_delay_steps",
    "worst_detection_delay_steps", "mean_recovery_delay_steps", "undetected_event_count",
]



def aggregate_run_results(run_results: pd.DataFrame) -> pd.DataFrame:
    grouped = run_results.groupby(PARAMETER_COLUMNS, dropna=False)

    named_aggregations = {
        "num_runs": ("simulation_seed", "count"),
    }
    for metric in IMPORTANT_METRICS:
        named_aggregations[f"{metric}_mean"] = (metric, "mean")
        named_aggregations[f"{metric}_std"] = (metric, "std")
        named_aggregations[f"{metric}_median"] = (metric, "median")

    summary = grouped.agg(**named_aggregations).reset_index()
    return summary



def add_feasibility_and_rank(
    summary: pd.DataFrame,
    max_false_alarm: float,
    max_false_consistent: float,
    min_event_detection_rate: float,
    max_steady_undecided: float,
    min_steady_nominal_consistent: float,
    min_steady_fault_inconsistent: float,
    min_steady_worst_class_correct: float,
    min_undecided_localization_precision: float,
    min_mediated_transition_rate: float,
) -> pd.DataFrame:
    """Applies explicit requirements and a decisiveness-first ranking.

    The correct confident rates for the two steady-state classes are treated
    separately. This avoids a balanced average hiding the fact that one class
    is almost never classified confidently. If all configurations are
    infeasible, weighted normalized deficits determine which candidate is
    closest to the stated design requirements.
    """
    ranked = summary.copy()

    constraints = {
        "steady_nominal_consistent_rate_mean": (
            "min", min_steady_nominal_consistent, 3.0
        ),
        "steady_fault_inconsistent_rate_mean": (
            "min", min_steady_fault_inconsistent, 3.0
        ),
        "steady_worst_class_correct_rate_mean": (
            "min", min_steady_worst_class_correct, 4.0
        ),
        "steady_undecided_rate_mean": (
            "max", max_steady_undecided, 3.0
        ),
        "steady_nominal_false_inconsistent_rate_mean": (
            "max", max_false_alarm, 2.0
        ),
        "steady_fault_false_consistent_rate_mean": (
            "max", max_false_consistent, 3.0
        ),
        "undecided_localization_precision_mean": (
            "min", min_undecided_localization_precision, 1.0
        ),
        "mediated_transition_rate_mean": (
            "min", min_mediated_transition_rate, 1.0
        ),
        "event_detection_rate_mean": (
            "min", min_event_detection_rate, 2.0
        ),
    }

    flags = []
    weighted_deficits = []
    for column, (direction, threshold, weight) in constraints.items():
        values = ranked[column].astype(float)
        if direction == "max":
            violation = np.maximum(values - threshold, 0.0)
            scale = max(1.0 - threshold, 1e-12)
        else:
            violation = np.maximum(threshold - values, 0.0)
            scale = max(threshold, 1e-12)

        flags.append((violation > 0.0).astype(int))
        weighted_deficits.append(weight * violation / scale)

    ranked["constraint_violation_count"] = np.sum(
        np.column_stack(flags), axis=1
    )
    ranked["weighted_normalized_constraint_deficit"] = np.sum(
        np.column_stack(weighted_deficits), axis=1
    )
    # Backwards-compatible alias.
    ranked["normalized_constraint_violation"] = ranked[
        "weighted_normalized_constraint_deficit"
    ]
    ranked["design_feasible"] = ranked["constraint_violation_count"] == 0
    ranked["safety_feasible"] = ranked["design_feasible"]

    ranked = ranked.sort_values(
        by=[
            "design_feasible",
            "weighted_normalized_constraint_deficit",
            "steady_worst_class_correct_rate_mean",
            "steady_nominal_consistent_rate_mean",
            "steady_fault_inconsistent_rate_mean",
            "steady_undecided_rate_mean",
            "steady_confident_error_rate_mean",
            "undecided_transition_f1_mean",
            "undecided_localization_precision_mean",
            "mediated_transition_rate_mean",
            "median_transition_settling_delay_steps_mean",
            "region_transitions_per_1000_steps_mean",
        ],
        ascending=[
            False, True, False, False, False, True, True,
            False, False, False, True, True,
        ],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked



def pareto_front_mask(
    values: np.ndarray,
) -> np.ndarray:
    """Returns non-dominated rows for minimization objectives."""
    values = np.asarray(values, dtype=float)
    finite_rows = np.all(np.isfinite(values), axis=1)
    is_pareto = np.zeros(len(values), dtype=bool)

    finite_indices = np.flatnonzero(finite_rows)
    for index in finite_indices:
        candidate = values[index]
        dominated = np.any(
            np.all(values[finite_rows] <= candidate, axis=1)
            & np.any(values[finite_rows] < candidate, axis=1)
        )
        is_pareto[index] = not dominated

    return is_pareto



def write_tradeoff_plot(summary: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Could not create trade-off plot: {exc}")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(
        summary["steady_nominal_consistent_rate_mean"],
        summary["steady_fault_inconsistent_rate_mean"],
        c=summary["steady_undecided_rate_mean"],
        s=45,
    )
    ax.set_xlabel("Mean confidently-consistent rate in steady nominal intervals")
    ax.set_ylabel("Mean confidently-inconsistent rate in steady disturbed intervals")
    ax.set_title("Class-specific steady-state decision performance")
    ax.grid(True, alpha=0.3)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Mean steady-state undecided rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================================
# Sweep execution
# ============================================================================


def run_sweep(args: argparse.Namespace) -> None:
    cache_paths = sorted(Path(path) for path in glob.glob(args.cache_glob))
    if not cache_paths:
        raise FileNotFoundError(
            f"No cache files match --cache-glob {args.cache_glob!r}"
        )

    caches = [load_cache(path) for path in cache_paths]
    if len(caches) < 10:
        warnings.warn(
            f"Only {len(caches)} Monte-Carlo cache(s) found. This is useful "
            "for debugging, but a paper evaluation should use substantially "
            "more independent runs and held-out test seeds."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_bins_values = parse_number_list(args.num_bins, int)
    short_window_values = parse_number_list(args.short_window_sizes, int)
    discount_values = parse_number_list(args.discounts, float)
    base_rate_values = parse_number_list(args.base_rates, float)
    eta_values = parse_number_list(args.etas, float)
    tau_values = parse_number_list(args.tau_ok_values, float)

    threshold_table = load_threshold_table(
        Path(args.threshold_json).resolve() if args.threshold_json else None
    )

    mapping_configs = []
    for num_bins, short_window, discount, base_rate in itertools.product(
        num_bins_values,
        short_window_values,
        discount_values,
        base_rate_values,
    ):
        expected_samples_per_bin = short_window / num_bins
        if expected_samples_per_bin < args.min_expected_bin_count:
            continue
        mapping_configs.append(
            (num_bins, short_window, discount, base_rate)
        )

    decision_configs = list(itertools.product(eta_values, tau_values))

    print(
        f"Caches: {len(caches)} | Mapping configurations: "
        f"{len(mapping_configs)} | Decision configurations: "
        f"{len(decision_configs)}"
    )

    run_rows: list[dict] = []
    event_rows: list[dict] = []
    transition_rows: list[dict] = []

    total_mapping_runs = len(caches) * len(mapping_configs)
    mapping_counter = 0

    for cache in caches:
        for num_bins, short_window, discount, base_rate in mapping_configs:
            mapping_counter += 1
            print(
                f"[{mapping_counter}/{total_mapping_runs}] "
                f"seed={cache.simulation_seed}, W={num_bins}, "
                f"n_ST={short_window}, lambda={discount:.4f}, "
                f"a={base_rate:.3f}"
            )

            threshold = resolve_ltst_threshold(
                num_bins=num_bins,
                discount=discount,
                threshold_table=threshold_table,
                threshold_alpha=args.threshold_alpha,
            )

            mapping = run_mapping(
                cache=cache,
                num_bins=num_bins,
                short_window_size=short_window,
                discount=discount,
                base_rate=base_rate,
                threshold=threshold,
            )

            for eta, tau_ok in decision_configs:
                decision = classify_mapping_output(
                    mapping=mapping,
                    eta=eta,
                    tau_ok=tau_ok,
                )

                run_row, run_event_rows, run_transition_rows = evaluate_run(
                    cache=cache,
                    mapping=mapping,
                    decision=decision,
                    num_bins=num_bins,
                    short_window_size=short_window,
                    discount=discount,
                    base_rate=base_rate,
                    eta=eta,
                    tau_ok=tau_ok,
                    persistence=args.persistence,
                    minimum_warmup=args.minimum_warmup,
                    transition_before_steps=args.transition_window_before,
                    transition_after_steps=args.transition_window_after,
                )
                run_rows.append(run_row)
                event_rows.extend(run_event_rows)
                transition_rows.extend(run_transition_rows)

    run_results = pd.DataFrame(run_rows)
    event_results = pd.DataFrame(event_rows)
    transition_results = pd.DataFrame(transition_rows)

    run_results.to_csv(output_dir / "parameter_sweep_run_metrics.csv", index=False)
    event_results.to_csv(output_dir / "parameter_sweep_event_metrics.csv", index=False)
    transition_results.to_csv(
        output_dir / "parameter_sweep_transition_metrics.csv", index=False
    )

    summary = aggregate_run_results(run_results)
    ranked = add_feasibility_and_rank(
        summary=summary,
        max_false_alarm=args.max_false_alarm,
        max_false_consistent=args.max_false_consistent,
        min_event_detection_rate=args.min_event_detection_rate,
        max_steady_undecided=args.max_steady_undecided,
        min_steady_nominal_consistent=args.min_steady_nominal_consistent,
        min_steady_fault_inconsistent=args.min_steady_fault_inconsistent,
        min_steady_worst_class_correct=args.min_steady_worst_class_correct,
        min_undecided_localization_precision=args.min_undecided_localization_precision,
        min_mediated_transition_rate=args.min_mediated_transition_rate,
    )

    pareto_objectives = np.column_stack(
        [
            1.0 - ranked["steady_nominal_consistent_rate_mean"],
            1.0 - ranked["steady_fault_inconsistent_rate_mean"],
            ranked["steady_undecided_rate_mean"],
            ranked["steady_confident_error_rate_mean"],
            1.0 - ranked["undecided_transition_f1_mean"].fillna(0.0),
            ranked["median_transition_settling_delay_steps_mean"],
        ]
    )
    ranked["pareto_optimal"] = pareto_front_mask(pareto_objectives)

    summary.to_csv(output_dir / "parameter_sweep_summary.csv", index=False)
    ranked.to_csv(output_dir / "parameter_sweep_ranked.csv", index=False)
    ranked.loc[ranked["pareto_optimal"]].to_csv(
        output_dir / "parameter_sweep_pareto_front.csv",
        index=False,
    )
    ranked.head(args.top_n).to_csv(
        output_dir / "parameter_sweep_top_candidates.csv",
        index=False,
    )

    write_tradeoff_plot(
        ranked,
        output_dir / "parameter_sweep_undecided_tradeoff.png",
    )

    print("\nTop candidates:")
    display_columns = [
        "rank", *PARAMETER_COLUMNS, "design_feasible",
        "constraint_violation_count",
        "weighted_normalized_constraint_deficit",
        "steady_nominal_consistent_rate_mean",
        "steady_fault_inconsistent_rate_mean",
        "steady_worst_class_correct_rate_mean",
        "steady_undecided_rate_mean",
        "steady_nominal_false_inconsistent_rate_mean",
        "steady_fault_false_consistent_rate_mean",
        "undecided_localization_precision_mean",
        "undecided_transition_f1_mean",
        "mediated_transition_rate_mean",
        "eta_ceiling_90pct_nominal_consistent_mean",
        "eta_ceiling_80pct_fault_inconsistent_mean",
        "event_detection_rate_mean", "pareto_optimal",
    ]
    print(ranked[display_columns].head(args.top_n).to_string(index=False))
    feasible_count = int(ranked["design_feasible"].sum())
    if feasible_count == 0:
        best = ranked.iloc[0]
        print(
            "\nWARNING: No parameter combination satisfies all design "
            "requirements. The first row is only the least-infeasible "
            "candidate and must not be interpreted as an acceptable result."
        )
        print(
            "Diagnostic eta ceilings of the least-infeasible candidate: "
            f"eta <= {best['eta_ceiling_90pct_nominal_consistent_mean']:.3f} "
            "for approximately 90% confidently-consistent nominal samples, "
            f"and eta <= {best['eta_ceiling_80pct_fault_inconsistent_mean']:.3f} "
            "for approximately 80% confidently-inconsistent fault samples."
        )
    else:
        print(f"\nFeasible parameter combinations: {feasible_count}")
    print(f"\nResults written to: {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte-Carlo parameter sweep for the SL Kalman-filter SA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate reusable raw KF/PIT caches for multiple seeds.",
    )
    generate_parser.add_argument("--simulation-script", required=True)
    generate_parser.add_argument("--output-dir", default="sweep_caches")
    generate_parser.add_argument(
        "--seeds",
        default="0:20",
        help="Comma list or half-open range, e.g. 0,1,2 or 0:20.",
    )
    generate_parser.add_argument(
        "--measurement-seed-offset",
        type=int,
        default=10000,
        help="Offset applied to each simulation seed for measurement sampling.",
    )
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.set_defaults(func=generate_caches)

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run the SL/LTST and decision-policy parameter sweep.",
    )
    sweep_parser.add_argument("--cache-glob", required=True)
    sweep_parser.add_argument("--output-dir", default="sweep_results")

    # Mapping and memory parameters
    sweep_parser.add_argument("--num-bins", default="5,7,10")
    sweep_parser.add_argument(
        "--short-window-sizes",
        default="25,35,50,70",
    )
    sweep_parser.add_argument(
        "--discounts",
        default="0.95,0.98,0.99,0.995",
    )
    sweep_parser.add_argument(
        "--base-rates",
        default="0.5",
        help=(
            "Keep 0.5 for the primary analysis. Other values should be used "
            "only for an explicitly reported prior-sensitivity analysis."
        ),
    )
    sweep_parser.add_argument(
        "--min-expected-bin-count",
        type=float,
        default=4.0,
        help="Discard combinations with n_ST/W below this value.",
    )

    # Decision-policy parameters
    sweep_parser.add_argument("--etas", default="0.90,0.95,0.99")
    sweep_parser.add_argument(
        "--tau-ok-values",
        default="0.50,0.60,0.70,0.80",
    )

    # LTST threshold configuration
    sweep_parser.add_argument(
        "--threshold-json",
        default="opinion_threshold_smoothed.json",
    )
    sweep_parser.add_argument(
        "--threshold-alpha",
        type=float,
        default=0.01,
    )

    # Event and time-series evaluation
    sweep_parser.add_argument(
        "--persistence",
        type=int,
        default=3,
        help="Required consecutive samples for event detection/recovery.",
    )
    sweep_parser.add_argument(
        "--minimum-warmup",
        type=int,
        default=0,
        help="Warm-up is max(minimum_warmup, n_ST).",
    )
    sweep_parser.add_argument("--transition-window-before", type=int, default=5)
    sweep_parser.add_argument(
        "--transition-window-after",
        type=int,
        default=35,
        help="Fixed samples after each onset/end included in transition evaluation.",
    )

    # Explicit selection constraints. These are design requirements, not
    # universal statistical constants and must be justified by the use case.
    sweep_parser.add_argument("--max-false-alarm", type=float, default=0.01)
    sweep_parser.add_argument(
        "--max-false-consistent",
        type=float,
        default=0.05,
    )
    sweep_parser.add_argument(
        "--min-event-detection-rate",
        type=float,
        default=0.80,
    )
    sweep_parser.add_argument("--max-steady-undecided", type=float, default=0.10)
    sweep_parser.add_argument(
        "--min-steady-nominal-consistent", type=float, default=0.90,
        help="Required confidently-consistent rate in steady nominal intervals.",
    )
    sweep_parser.add_argument(
        "--min-steady-fault-inconsistent", type=float, default=0.80,
        help="Required confidently-inconsistent rate in steady disturbed intervals.",
    )
    sweep_parser.add_argument(
        "--min-steady-worst-class-correct", type=float, default=0.80,
        help="Required minimum of the two class-specific steady correct rates.",
    )
    sweep_parser.add_argument(
        "--min-undecided-localization-precision", type=float, default=0.70,
        help="Required fraction of all undecided samples that lie in transition windows.",
    )
    sweep_parser.add_argument("--min-mediated-transition-rate", type=float, default=0.70)
    sweep_parser.add_argument("--top-n", type=int, default=20)
    sweep_parser.set_defaults(func=run_sweep)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()