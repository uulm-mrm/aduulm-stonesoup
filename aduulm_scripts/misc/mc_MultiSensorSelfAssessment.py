#!/usr/bin/env python3
"""Monte-Carlo driver for the V9 N-sensor PIT/Subjective-Logic test bed.

The simulation module remains the single source of truth for the KF/SA method.
This file contains only experiment/scenario configuration, repeated seeded runs,
and CSV aggregation for paper figures/tables.

Output layout
-------------
<RESULT_ROOT>/<EXPERIMENT_NAME>/
    scenario_configurations.csv
    runs.csv
    interval_metrics_per_run.csv
    latex_interval_summary.csv
    latex_timeseries_summary.csv
    raw/<scenario>/run_XXXX_opinions.csv       (optional)
    raw/<scenario>/run_XXXX_diagnostics.csv    (optional)

The two ``latex_*.csv`` files are intentionally plain, long-format CSV files with
stable ASCII column names and decimal points. They can be consumed directly by
pgfplotstable/pgfplots or preprocessed into a smaller table for a specific figure.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os

# Avoid CPU oversubscription: Monte-Carlo parallelism is handled with processes.
# Each worker therefore keeps BLAS/OpenMP-style numerical kernels single-threaded.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# User configuration
# =============================================================================

EXPERIMENT_NAME = "n_sensor_selfassessment_mc"
N_RUNS = 100
MASTER_SEED = 20260812
CONTINUE_ON_ERROR = True

# -----------------------------------------------------------------------------
# Parallel execution
# -----------------------------------------------------------------------------
# CPU-bound simulation runs are independent and are therefore executed in
# separate processes.  Leave one logical CPU free by default so the machine
# remains responsive. Set MAX_WORKERS explicitly if desired.
def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)

MAX_WORKERS = max(1, _available_cpu_count() - 1)
USE_PARALLEL_EXECUTION = True
QUIET_WORKERS = True

# Scenario selection is independent of the disturbance flags in BASE_OVERRIDES.
# None -> execute every entry in SCENARIOS.
# Example: {"nominal", "s1_bias", "common_process_noise_mismatch"}
SCENARIOS_TO_RUN: set[str] | None = {
    "nominal"
}

# Raw per-run files are exhaustive but can become large for many scenarios/runs.
# The two aggregated latex_*.csv files are written regardless of this switch.
SAVE_RAW_RUN_CSV = False
SAVE_RAW_DIAGNOSTICS = False
SAVE_TIMESERIES_MC_SUMMARY = True

RESULT_ROOT = Path(__file__).resolve().parent / "mc_results"
SIMULATION_SCRIPT = (
    Path(__file__).resolve().parent
    / "02_MultiSensorKalmanFilterWithSelfAssessmentV6_MCReady.py"
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    overrides: dict[str, Any]
    description: str = ""


# Parameters applied to every scenario before scenario-specific overrides.
# Change N, sensor rates, filter model, TEF horizons, etc. here.
BASE_OVERRIDES: dict[str, Any] = {
    "NUM_SENSORS": 2,
    "SYNCHRONOUS_SENSOR_SPECIAL_CASE": True,
    "SYNCHRONOUS_REFERENCE_RATE_HZ": 10.0,
    "USE_NORMALIZED_BELIEF": True,
    "USE_CT_MODEL": False,
    "ENABLE_GRIEBEL_REFERENCE": False,
    "ACTIVATE_DISTURBANCES": True,
    # Start every scenario from a clean no-fault baseline. Individual scenarios
    # below then enable exactly the disturbance family they need.
    "ENABLE_OUTLIER_DISTURBANCE": False,
    "ENABLE_BIAS_DISTURBANCE": False,
    "ENABLE_MEASUREMENT_NOISE_DISTURBANCE": False,
    "ENABLE_NON_GAUSSIAN_DISTURBANCE": False,
    "ENABLE_PROCESS_NOISE_DISTURBANCE": False,
    "ENABLE_SENSOR_DROPOUT": False,
    "ENABLE_GROUND_TRUTH_TURN": False,
}


# Suggested clean Monte-Carlo scenarios. Remove/reorder/add entries freely.
# All interval locations and amplitudes can also be overridden here.
SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        "nominal",
        {},
        "No injected disturbance after the common burn-in.",
    ),
    ScenarioSpec(
        "s1_outliers",
        {
            "ENABLE_OUTLIER_DISTURBANCE": True,
            "OUTLIER_SENSOR_IDS": {1},
        },
        "Sporadic outliers on Sensor 1.",
    ),
    ScenarioSpec(
        "s1_bias",
        {
            "ENABLE_BIAS_DISTURBANCE": True,
            "BIAS_SENSOR_IDS": {1},
            "SENSOR_1_BIAS_VECTOR_M": np.array([2.0, 0.0]),
        },
        "Constant +2 m x-bias on Sensor 1 during the bias interval.",
    ),
    ScenarioSpec(
        "s1_measurement_noise",
        {
            "ENABLE_MEASUREMENT_NOISE_DISTURBANCE": True,
            "MEASUREMENT_NOISE_SENSOR_IDS": {1},
            "MEASUREMENT_NOISE_VARIANCE_INCREASE_FACTOR": 2.0,
        },
        "Measurement-noise covariance mismatch on Sensor 1.",
    ),
    ScenarioSpec(
        "s1_non_gaussian",
        {
            "ENABLE_NON_GAUSSIAN_DISTURBANCE": True,
            "NON_GAUSSIAN_SENSOR_IDS": {1},
            "USE_HEAVY_TAILED_NON_GAUSSIAN": True,
        },
        "Variance-matched non-Gaussian noise on Sensor 1.",
    ),
    ScenarioSpec(
        "s2_dropout",
        {
            "ENABLE_SENSOR_DROPOUT": True,
            "DROPOUT_SENSOR_ID": 2,
        },
        "Sensor 2 unavailable during the dropout interval.",
    ),
    ScenarioSpec(
        "common_motion_model_mismatch",
        {
            "ENABLE_GROUND_TRUTH_TURN": True,
            "USE_CT_MODEL": False,
        },
        "Coordinated ground-truth turn while the assessment/filter uses CV.",
    ),
    ScenarioSpec(
        "common_process_noise_mismatch",
        {
            "ENABLE_PROCESS_NOISE_DISTURBANCE": True,
            "PROCESS_NOISE_INCREASE_FACTOR": 32.0,
        },
        "Common process-noise mismatch in x/y.",
    ),
    # Evaluation-A example. Enable it when the native Griebel backend is available.
    # ScenarioSpec(
    #     "evalA_s1_bias_cross_contamination",
    #     {
    #         "NUM_SENSORS": 2,
    #         "SYNCHRONOUS_SENSOR_SPECIAL_CASE": False,
    #         "CROSS_CONTAMINATION_RATE_STRESS_TEST": True,
    #         "CROSS_CONTAMINATION_DISTURBED_SENSOR_ID": 1,
    #         "CROSS_CONTAMINATION_DISTURBED_RATE_HZ": 25.0,
    #         "CROSS_CONTAMINATION_OTHER_RATE_HZ": 10.0,
    #         "ENABLE_BIAS_DISTURBANCE": True,
    #         "BIAS_SENSOR_IDS": {1},
    #         "ENABLE_GRIEBEL_REFERENCE": True,
    #     },
    #     "Common-prior cross-contamination reference against isolated channels.",
    # ),
]


def _selected_scenarios() -> list[tuple[int, ScenarioSpec]]:
    """Return selected scenarios together with their stable original indices."""
    if SCENARIOS_TO_RUN is None:
        return list(enumerate(SCENARIOS))
    requested = set(SCENARIOS_TO_RUN)
    known = {scenario.name for scenario in SCENARIOS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown scenario names in SCENARIOS_TO_RUN: {unknown}")
    return [
        (index, scenario)
        for index, scenario in enumerate(SCENARIOS)
        if scenario.name in requested
    ]


# =============================================================================
# Infrastructure
# =============================================================================


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.count) if self.count > 0 else math.nan

    @property
    def ci95_half_width(self) -> float:
        return 1.96 * self.sem if self.count > 0 else math.nan


OPINION_METRICS = ("b", "d", "u", "p_ok", "b_norm", "d_norm", "selected_score")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    # Enum-like values used by Subjective Logic are serialized by name/string.
    if not isinstance(value, (str, int, float, bool, type(None), list)):
        return str(value)
    return value


def _load_simulation_module():
    if not SIMULATION_SCRIPT.exists():
        raise FileNotFoundError(
            f"Simulation module not found: {SIMULATION_SCRIPT}. "
            "Place the V9_MCReady script next to this Monte-Carlo driver or "
            "change SIMULATION_SCRIPT above."
        )
    # Must be set before the imported module selects a Matplotlib backend.
    os.environ["SELFASSESSMENT_MC_HEADLESS"] = "1"
    spec = importlib.util.spec_from_file_location("selfassessment_mc_sim", SIMULATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load simulation module from {SIMULATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_scenario_configuration_csv(path: Path, sim, baseline: dict[str, Any], scenarios: list[ScenarioSpec]) -> None:
    fieldnames = ["scenario", "description", "parameter", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenarios:
            sim.apply_configuration(baseline)
            sim.apply_configuration(BASE_OVERRIDES)
            sim.apply_configuration(scenario.overrides)
            resolved = sim.configuration_snapshot()
            for key in sorted(resolved):
                writer.writerow({
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "parameter": key,
                    "value": json.dumps(_jsonable(resolved[key]), separators=(",", ":")),
                })


def _scenario_seeds(scenario_index: int, run_index: int) -> tuple[int, int]:
    # Wide deterministic spacing avoids overlap between truth and sensor streams.
    base = int(MASTER_SEED) + int(scenario_index) * 10_000_000
    truth_seed = base + int(run_index)
    sensor_seed_base = base + 1_000_000 + int(run_index) * 1000
    return truth_seed, sensor_seed_base


def _append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _update_timeseries_stats(
    accumulator: dict[tuple[str, str, str, float, str], OnlineStats],
    sim,
    result,
    scenario_name: str,
    run_index: int,
) -> None:
    for row in sim.iter_opinion_records(
        result,
        scenario_name=scenario_name,
        run_index=run_index,
    ):
        time_s = round(float(row["time_s"]), 9)
        for metric in OPINION_METRICS:
            key = (
                scenario_name,
                str(row["channel_type"]),
                str(row["channel_id"]),
                time_s,
                metric,
            )
            accumulator.setdefault(key, OnlineStats()).add(float(row[metric]))

    # Track position error is included in the same publication-ready time-series file.
    for time_s, value in zip(result.event_times_s, result.position_error):
        key = (
            scenario_name,
            "track",
            "central_track",
            round(float(time_s), 9),
            "position_error_m",
        )
        accumulator.setdefault(key, OnlineStats()).add(float(value))


def _write_interval_summary(
    path: Path,
    interval_values: dict[tuple[str, str, str, str, str], list[float]],
) -> None:
    fieldnames = [
        "scenario", "interval", "channel_type", "channel_id", "metric",
        "n_runs", "mean", "std", "sem", "ci95_low", "ci95_high", "min", "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(interval_values):
            scenario, interval, channel_type, channel_id, metric = key
            values = np.asarray(interval_values[key], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            sem = std / math.sqrt(values.size)
            half = 1.96 * sem
            writer.writerow({
                "scenario": scenario,
                "interval": interval,
                "channel_type": channel_type,
                "channel_id": channel_id,
                "metric": metric,
                "n_runs": int(values.size),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })


def _write_timeseries_summary(
    path: Path,
    accumulator: dict[tuple[str, str, str, float, str], OnlineStats],
) -> None:
    fieldnames = [
        "scenario", "time_s", "channel_type", "channel_id", "metric",
        "n_runs", "mean", "std", "sem", "ci95_low", "ci95_high", "min", "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(accumulator):
            scenario, channel_type, channel_id, time_s, metric = key
            stats = accumulator[key]
            if stats.count == 0:
                continue
            half = stats.ci95_half_width
            writer.writerow({
                "scenario": scenario,
                "time_s": time_s,
                "channel_type": channel_type,
                "channel_id": channel_id,
                "metric": metric,
                "n_runs": stats.count,
                "mean": stats.mean,
                "std": stats.std,
                "sem": stats.sem,
                "ci95_low": stats.mean - half,
                "ci95_high": stats.mean + half,
                "min": stats.minimum,
                "max": stats.maximum,
            })


# =============================================================================
# Parallel worker infrastructure
# =============================================================================

_WORKER_SIM = None
_WORKER_BASELINE: dict[str, Any] | None = None


def _configure_sim_for_batch(sim) -> None:
    """Disable every visualization in a repeated Monte-Carlo worker."""
    sim.SHOW_DYNAMIC_ANIMATION = False
    sim.SHOW_MATPLOTLIB_PLOTS = False
    sim.SHOW_POSITION_ERROR = False
    sim.SHOW_PAIRWISE_TIME_SERIES = False
    sim.SHOW_PAIRWISE_MATRIX_FIGURE = False
    sim.SHOW_PAIRWISE_DEDUCTION_TRAJECTORIES = False


def _worker_initializer() -> None:
    """Load the simulation module once per worker process."""
    global _WORKER_SIM, _WORKER_BASELINE
    _WORKER_SIM = _load_simulation_module()
    _configure_sim_for_batch(_WORKER_SIM)
    _WORKER_BASELINE = _WORKER_SIM.configuration_snapshot()


def _run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one independent (scenario, run) realization in a worker."""
    sim = _WORKER_SIM
    baseline = _WORKER_BASELINE
    if sim is None or baseline is None:
        # Sequential/debug fallback.
        sim = _load_simulation_module()
        _configure_sim_for_batch(sim)
        baseline = sim.configuration_snapshot()

    scenario_index = int(task["scenario_index"])
    scenario_name = str(task["scenario_name"])
    scenario_overrides = dict(task["scenario_overrides"])
    run_index = int(task["run_index"])
    raw_dir = Path(task["raw_dir"])
    need_export = bool(task["need_export"])

    started = time.perf_counter()
    truth_seed, sensor_seed_base = _scenario_seeds(scenario_index, run_index)
    status = "ok"
    error = ""
    metric_rows: list[dict[str, Any]] = []
    opinion_path = ""
    diagnostic_path = ""

    try:
        sim.apply_configuration(baseline)
        sim.apply_configuration(BASE_OVERRIDES)
        sim.apply_configuration(scenario_overrides)
        sim.set_run_seeds(truth_seed, sensor_seed_base)

        if QUIET_WORKERS:
            raw_dir.mkdir(parents=True, exist_ok=True)
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with redirect_stdout(sink), redirect_stderr(sink):
                    _, result = sim.run_simulation()
        else:
            _, result = sim.run_simulation()

        if need_export:
            raw_dir.mkdir(parents=True, exist_ok=True)
            op_path, diag_path = sim.export_result_csv_bundle(
                result,
                raw_dir,
                scenario_name=scenario_name,
                run_index=run_index,
            )
            opinion_path = str(op_path)
            diagnostic_path = str(diag_path)

        metric_rows = sim.collect_interval_metric_records(
            result,
            scenario_name=scenario_name,
            run_index=run_index,
        )
        num_sensors = int(sim.NUM_SENSORS)
        score_name = "b_norm" if sim.USE_NORMALIZED_BELIEF else "d_norm"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        num_sensors = int(getattr(sim, "NUM_SENSORS", -1))
        score_name = "b_norm" if getattr(sim, "USE_NORMALIZED_BELIEF", False) else "d_norm"

    return {
        "scenario": scenario_name,
        "run": run_index,
        "truth_seed": truth_seed,
        "sensor_seed_base": sensor_seed_base,
        "status": status,
        "runtime_s": time.perf_counter() - started,
        "num_sensors": num_sensors,
        "normalized_score": score_name,
        "error": error,
        "metric_rows": metric_rows,
        "opinion_path": opinion_path,
        "diagnostic_path": diagnostic_path,
    }


def _update_timeseries_stats_from_raw(
    accumulator: dict[tuple[str, str, str, float, str], OnlineStats],
    opinion_path: Path,
    diagnostic_path: Path,
) -> None:
    """Reconstruct the publication time-series statistics from raw per-run CSVs."""
    with opinion_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = str(row["scenario"])
            time_s = round(float(row["time_s"]), 9)
            channel_type = str(row["channel_type"])
            channel_id = str(row["channel_id"])
            for metric in OPINION_METRICS:
                key = (scenario, channel_type, channel_id, time_s, metric)
                accumulator.setdefault(key, OnlineStats()).add(float(row[metric]))

    if diagnostic_path.exists():
        with diagnostic_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("diagnostic_type") != "track":
                    continue
                value = float(row["position_error_m"])
                key = (
                    str(row["scenario"]),
                    "track",
                    "central_track",
                    round(float(row["time_s"]), 9),
                    "position_error_m",
                )
                accumulator.setdefault(key, OnlineStats()).add(value)


def _scenario_timeseries_postprocess(task: dict[str, Any]) -> str:
    """Aggregate one scenario's raw run CSVs and write a partial summary."""
    scenario_name = str(task["scenario_name"])
    opinion_paths = [Path(p) for p in task["opinion_paths"]]
    diagnostic_paths = [Path(p) for p in task["diagnostic_paths"]]
    partial_path = Path(task["partial_path"])
    accumulator: dict[tuple[str, str, str, float, str], OnlineStats] = {}
    for op_path, diag_path in zip(opinion_paths, diagnostic_paths):
        _update_timeseries_stats_from_raw(accumulator, op_path, diag_path)
    _write_timeseries_summary(partial_path, accumulator)
    return str(partial_path)


def _concatenate_csv_files(paths: list[Path], output_path: Path) -> None:
    """Concatenate same-schema CSVs while keeping exactly one header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", newline="", encoding="utf-8") as in_handle:
                for line_index, line in enumerate(in_handle):
                    if line_index == 0:
                        if wrote_header:
                            continue
                        wrote_header = True
                    out_handle.write(line)


# =============================================================================
# Monte-Carlo execution
# =============================================================================


def main() -> None:
    selected = _selected_scenarios()
    if not selected:
        raise RuntimeError("No scenarios selected.")
    selected_specs = [spec for _, spec in selected]

    # Load once in the parent only for configuration snapshots/metadata.
    sim = _load_simulation_module()
    _configure_sim_for_batch(sim)
    baseline = sim.configuration_snapshot()

    output_dir = RESULT_ROOT / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw"
    temp_root = output_dir / "_parallel_tmp"
    temp_raw_root = temp_root / "raw"
    temp_summary_root = temp_root / "timeseries_summary"
    temp_root.mkdir(parents=True, exist_ok=True)

    # Time-series aggregation needs the per-run opinion/diagnostic streams.  When
    # raw CSV retention is disabled, they are written temporarily and deleted at
    # the end of the experiment.
    need_per_run_export = SAVE_RAW_RUN_CSV or SAVE_TIMESERIES_MC_SUMMARY
    export_root = raw_root if SAVE_RAW_RUN_CSV else temp_raw_root
    if need_per_run_export:
        export_root.mkdir(parents=True, exist_ok=True)

    run_csv = output_dir / "runs.csv"
    per_run_metric_csv = output_dir / "interval_metrics_per_run.csv"
    for path in (run_csv, per_run_metric_csv):
        if path.exists():
            path.unlink()

    _write_scenario_configuration_csv(
        output_dir / "scenario_configurations.csv",
        sim,
        baseline,
        selected_specs,
    )

    interval_values: dict[tuple[str, str, str, str, str], list[float]] = {}
    raw_files_by_scenario: dict[str, dict[str, list[str]]] = {
        spec.name: {"opinions": [], "diagnostics": []}
        for spec in selected_specs
    }

    run_fields = [
        "scenario", "run", "truth_seed", "sensor_seed_base", "status",
        "runtime_s", "num_sensors", "normalized_score", "error",
    ]
    metric_fields = [
        "scenario", "run", "interval", "channel_type", "channel_id",
        "metric", "value", "within_run_std", "n_samples",
    ]

    tasks: list[dict[str, Any]] = []
    for scenario_index, spec in selected:
        scenario_raw_dir = export_root / spec.name
        if need_per_run_export:
            scenario_raw_dir.mkdir(parents=True, exist_ok=True)
        for run_index in range(int(N_RUNS)):
            tasks.append({
                "scenario_index": scenario_index,
                "scenario_name": spec.name,
                "scenario_overrides": spec.overrides,
                "run_index": run_index,
                "raw_dir": str(scenario_raw_dir),
                "need_export": need_per_run_export,
            })

    total_requested = len(tasks)
    workers = min(MAX_WORKERS, total_requested) if USE_PARALLEL_EXECUTION else 1
    print(f"Monte-Carlo experiment: {EXPERIMENT_NAME}")
    print(
        f"Scenarios: {len(selected_specs)}/{len(SCENARIOS)}, "
        f"runs/scenario: {N_RUNS}, total runs: {total_requested}"
    )
    print("Selected: " + ", ".join(spec.name for spec in selected_specs))
    print(f"Parallel workers: {workers}")
    print(f"Output: {output_dir}")

    completed = 0
    wall_started = time.perf_counter()

    def consume(worker_result: dict[str, Any]) -> None:
        nonlocal completed
        run_row = {key: worker_result[key] for key in run_fields}
        _append_csv_rows(run_csv, run_fields, [run_row])

        metric_rows = worker_result["metric_rows"]
        if metric_rows:
            _append_csv_rows(per_run_metric_csv, metric_fields, metric_rows)
            for row in metric_rows:
                key = (
                    str(row["scenario"]),
                    str(row["interval"]),
                    str(row["channel_type"]),
                    str(row["channel_id"]),
                    str(row["metric"]),
                )
                interval_values.setdefault(key, []).append(float(row["value"]))

        scenario_name = str(worker_result["scenario"])
        if worker_result["opinion_path"]:
            raw_files_by_scenario[scenario_name]["opinions"].append(worker_result["opinion_path"])
        if worker_result["diagnostic_path"]:
            raw_files_by_scenario[scenario_name]["diagnostics"].append(worker_result["diagnostic_path"])

        completed += 1
        status = worker_result["status"]
        if status != "ok":
            print(
                f"[{completed:4d}/{total_requested}] {scenario_name} "
                f"run {worker_result['run']:04d}: FAILED: {worker_result['error']}"
            )
            if not CONTINUE_ON_ERROR:
                raise RuntimeError(worker_result["error"])
        else:
            elapsed = time.perf_counter() - wall_started
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total_requested - completed) / rate if rate > 0 else math.nan
            print(
                f"[{completed:4d}/{total_requested}] {scenario_name:<32s} "
                f"run {worker_result['run']:04d} | "
                f"worker {worker_result['runtime_s']:6.2f}s | "
                f"ETA {remaining/60.0:6.1f} min"
            )

    if USE_PARALLEL_EXECUTION and workers > 1:
        # spawn avoids inheriting mutable simulation/module state across workers.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_initializer,
        ) as executor:
            futures = [executor.submit(_run_one_task, task) for task in tasks]
            for future in as_completed(futures):
                consume(future.result())
    else:
        _worker_initializer()
        for task in tasks:
            consume(_run_one_task(task))

    _write_interval_summary(
        output_dir / "latex_interval_summary.csv",
        interval_values,
    )

    # The expensive simulations above are fully parallel.  Time-series CSV
    # reduction is also parallelized across scenarios to keep post-processing
    # from becoming the next bottleneck.
    if SAVE_TIMESERIES_MC_SUMMARY:
        temp_summary_root.mkdir(parents=True, exist_ok=True)
        post_tasks = []
        for spec in selected_specs:
            files = raw_files_by_scenario[spec.name]
            op_paths = sorted(files["opinions"])
            diag_paths = sorted(files["diagnostics"])
            if not op_paths:
                continue
            post_tasks.append({
                "scenario_name": spec.name,
                "opinion_paths": op_paths,
                "diagnostic_paths": diag_paths,
                "partial_path": str(temp_summary_root / f"{spec.name}.csv"),
            })

        partial_paths: list[Path] = []
        post_workers = min(workers, len(post_tasks)) if post_tasks else 1
        if post_tasks and post_workers > 1:
            with ProcessPoolExecutor(max_workers=post_workers, mp_context=mp.get_context("spawn")) as executor:
                futures = [executor.submit(_scenario_timeseries_postprocess, task) for task in post_tasks]
                for future in as_completed(futures):
                    partial_paths.append(Path(future.result()))
        else:
            for task in post_tasks:
                partial_paths.append(Path(_scenario_timeseries_postprocess(task)))

        # Deterministic scenario order in the final file.
        order = {spec.name: index for index, spec in enumerate(selected_specs)}
        partial_paths.sort(key=lambda p: order.get(p.stem, 10**9))
        _concatenate_csv_files(
            partial_paths,
            output_dir / "latex_timeseries_summary.csv",
        )

    # Diagnostics may have been needed temporarily for position-error aggregation.
    if SAVE_RAW_RUN_CSV and not SAVE_RAW_DIAGNOSTICS:
        for files in raw_files_by_scenario.values():
            for path_string in files["diagnostics"]:
                path = Path(path_string)
                if path.exists():
                    path.unlink()

    if not SAVE_RAW_RUN_CSV and temp_raw_root.exists():
        import shutil
        shutil.rmtree(temp_raw_root, ignore_errors=True)
    if temp_summary_root.exists():
        import shutil
        shutil.rmtree(temp_summary_root, ignore_errors=True)
    try:
        temp_root.rmdir()
    except OSError:
        pass

    wall_runtime = time.perf_counter() - wall_started
    print("\nFinished.")
    print(f"  Wall-clock runtime        : {wall_runtime/60.0:.2f} min")
    print(f"  Parallel workers          : {workers}")
    print(f"  Per-run interval metrics  : {per_run_metric_csv}")
    print(f"  LaTeX interval summary    : {output_dir / 'latex_interval_summary.csv'}")
    if SAVE_TIMESERIES_MC_SUMMARY:
        print(f"  LaTeX time-series summary : {output_dir / 'latex_timeseries_summary.csv'}")
    if SAVE_RAW_RUN_CSV:
        print(f"  Raw run CSVs              : {raw_root}")


if __name__ == "__main__":
    main()