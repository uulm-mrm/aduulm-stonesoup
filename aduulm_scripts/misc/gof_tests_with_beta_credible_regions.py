import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import subjective_logic as sl
from tqdm.auto import tqdm

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

def samples_to_multinomial_opinion(samples, W=7):
    """
    Converts u-samples in [0,1] to a W-dimensional multinomial opinion
    by cumulative fusion of one-hot evidence opinions.
    """
    ops = [scalar_u_to_opinion(u, W) for u in samples]
    op = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, ops)
    return op


def samples_to_multinomial_opinion_fast(samples, W=7):
    counts, _ = np.histogram(samples, bins=np.linspace(0.0, 1.0, W + 1))
    dist = eval(f"sl.DirichletDistribution{W}d").from_evidences(counts.astype(float))
    return dist.as_opinion()

def evaluate_density_with_sl_mapping(
    density_fn,
    n=35,
    W=7,
    prior_ok=0.5,
    seed=None,
):
    """
    Samples from a density, builds multinomial opinion, maps it to binomial OK opinion.
    """
    samples = sample_from_density(density_fn, n=n, seed=seed)

    op_multi = samples_to_multinomial_opinion_fast(samples, W=W)
    op_bin = multinomial_opinion_to_binomial_ok_opinion(
        op_multi,
        W=W,
        prior_ok=prior_ok,
    )

    p_ok = op_bin.getProjection()[0]

    return {
        "samples": samples,
        "op_multi": op_multi,
        "op_bin": op_bin,
        "p_ok": p_ok,
        "belief": op_bin.belief(),
        "disbelief": op_bin.disbelief(),
        "uncertainty": op_bin.uncertainty(),
    }

def p_ok_from_samples_fast(samples, W=7, prior_ok=0.5, eps=1e-12):
    counts, _ = np.histogram(samples, bins=np.linspace(0.0, 1.0, W + 1))
    counts = counts.astype(float)

    R = np.sum(counts)
    if R <= eps:
        return prior_ok

    u = W / (W + R)
    c = 1.0 - u

    b_tilde = counts / R
    a = np.ones(W) / W

    tv = 0.5 * np.sum(np.abs(b_tilde - a))
    tv_max = 1.0 - 1.0 / W

    d = c * tv / tv_max
    d = np.clip(d, 0.0, c)

    b = c - d
    p_ok = b + prior_ok * u

    return float(p_ok)

# ============================================================
# Piecewise density helpers on [0, 1]
# ============================================================

def _as_array(x):
    return np.asarray(x, dtype=float)


def bump_density(x, intervals, pi=0.20):
    """
    Mixture:
        f(x) = (1-pi) * Uniform(0,1)
             + pi     * Uniform(union of intervals)

    intervals: list of (lo, hi)
    pi: probability mass assigned to bump intervals
    """
    x = _as_array(x)
    f = np.ones_like(x) * (1.0 - pi)

    total_width = sum(hi - lo for lo, hi in intervals)
    for lo, hi in intervals:
        mask = (x >= lo) & (x <= hi)
        f[mask] += pi / total_width

    return f


def gap_density(x, intervals, rho=0.0):
    """
    Piecewise distribution with reduced density in intervals.

    rho = 0.0 means complete gap.
    rho = 0.2 means density inside gap is 20% of nominal.

    The density is normalized automatically.
    """
    x = _as_array(x)

    gap_width = sum(hi - lo for lo, hi in intervals)
    outside_width = 1.0 - gap_width

    # normalization:
    # c_out * outside_width + c_in * gap_width = 1
    # c_in = rho * c_out
    c_out = 1.0 / (outside_width + rho * gap_width)
    c_in = rho * c_out

    f = np.ones_like(x) * c_out

    for lo, hi in intervals:
        mask = (x >= lo) & (x <= hi)
        f[mask] = c_in

    return f


# ============================================================
# Fig. 1-like densities
# ============================================================

def density_bump_edges(x):
    return bump_density(
        x,
        intervals=[(0.00, 0.025), (0.975, 1.00)],
        pi=0.25,
    )


def density_gap_edges(x):
    return gap_density(
        x,
        intervals=[(0.00, 0.075), (0.925, 1.00)],
        rho=0.0,
    )


def density_bump_middle(x):
    return bump_density(
        x,
        intervals=[(0.45, 0.55)],
        pi=0.12,
    )


def density_gap_middle(x):
    return gap_density(
        x,
        intervals=[(0.45, 0.55)],
        rho=0.0,
    )


def density_bump_sides(x):
    return bump_density(
        x,
        intervals=[(0.2, 0.3), (0.7, 0.8)],
        pi=0.25,
    )


def density_gap_sides(x):
    return gap_density(
        x,
        intervals=[(0.25, 0.35), (0.65, 0.75)],
        rho=0.0,
    )


DENSITIES = {
    "Bump (Edges)": density_bump_edges,
    "Gap (Edges)": density_gap_edges,
    "Bump (Middle)": density_bump_middle,
    "Gap (Middle)": density_gap_middle,
    "Bump (Sides)": density_bump_sides,
    "Gap (Sides)": density_gap_sides,
}

from scipy.stats import beta as beta_dist
from scipy.optimize import brentq
import matplotlib.tri as mtri
from scipy.stats import norm, laplace
import numpy as np


# ============================================================
# Figure 3-like distributions from Covington & Miller
# ============================================================

def density_uniform(x):
    x = np.asarray(x, dtype=float)
    return np.ones_like(x)


def sample_uniform(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n)


def make_beta_density(alpha, beta):
    def density(x):
        x = np.asarray(x, dtype=float)
        return beta_dist.pdf(x, alpha, beta)
    return density


def make_beta_sampler(alpha, beta):
    def sampler(n, seed=None):
        rng = np.random.default_rng(seed)
        return rng.beta(alpha, beta, size=n)
    return sampler


def density_phi_laplace(x, eps=1e-12):
    """
    Density of U = Phi(Y), where Y ~ Laplace(0,1)
    and Phi is the standard normal CDF.

    Transformation:
        u = Phi(y)
        y = Phi^{-1}(u)

    Density:
        f_U(u) = f_Y(Phi^{-1}(u)) / phi(Phi^{-1}(u))
    """
    x = np.asarray(x, dtype=float)
    u = np.clip(x, eps, 1.0 - eps)

    y = norm.ppf(u)
    f_y = laplace.pdf(y, loc=0.0, scale=1.0)
    phi_y = norm.pdf(y)

    return f_y / phi_y


def sample_phi_laplace(n, seed=None):
    """
    Sample U = Phi(Y), Y ~ Laplace(0,1).
    """
    rng = np.random.default_rng(seed)
    y = rng.laplace(loc=0.0, scale=1.0, size=n)
    return norm.cdf(y)


def density_discrete_uniform_001_099(x):
    """
    This is not a true continuous density.
    For plotting only: returns zeros.

    The distribution is discrete on:
        {0.01, 0.02, ..., 0.99}
    """
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def sample_discrete_uniform_001_099(n, seed=None):
    rng = np.random.default_rng(seed)
    support = np.arange(1, 100) / 100.0
    return rng.choice(support, size=n, replace=True)


FIG3_DENSITIES = {
    "Uniform": density_uniform,
    "Beta(1.2, 0.8)": make_beta_density(1.2, 0.8),
    "Beta(0.6, 0.6)": make_beta_density(0.6, 0.6),
    "Beta(1.6, 1.6)": make_beta_density(1.6, 1.6),
    "Phi(Laplace(0,1))": density_phi_laplace,
    "Discrete Uniform": density_discrete_uniform_001_099,
}


FIG3_SAMPLERS = {
    "Uniform": sample_uniform,
    "Beta(1.2, 0.8)": make_beta_sampler(1.2, 0.8),
    "Beta(0.6, 0.6)": make_beta_sampler(0.6, 0.6),
    "Beta(1.6, 1.6)": make_beta_sampler(1.6, 1.6),
    "Phi(Laplace(0,1))": sample_phi_laplace,
    "Discrete Uniform": sample_discrete_uniform_001_099,
}

def sample_from_density(density_fn, n, grid_size=5000, seed=None):
    """
    Generic rejection sampler for densities on [0,1].
    Assumes density_fn is normalized on [0,1].
    """
    rng = np.random.default_rng(seed)

    grid = np.linspace(0.0, 1.0, grid_size)
    f_max = float(np.max(density_fn(grid)))

    samples = []

    while len(samples) < n:
        batch_size = max(1000, 2 * (n - len(samples)))

        x = rng.uniform(0.0, 1.0, size=batch_size)
        y = rng.uniform(0.0, f_max, size=batch_size)

        accept = y <= density_fn(x)
        samples.extend(x[accept].tolist())

    return np.array(samples[:n])

def plot_fig1_like_densities(n=200, bins=7, seed=42):
    x_grid = np.linspace(0.0, 1.0, 2000)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=False)
    axes = axes.ravel()

    for ax, (name, density_fn) in zip(axes, DENSITIES.items()):
        samples = sample_from_density(density_fn, n=n, seed=seed)

        ax.hist(samples, bins=bins, density=True, alpha=0.35, label=f"histogram (n={n})")
        ax.plot(x_grid, density_fn(x_grid), linewidth=2, label="true density")

        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.grid(True)

    axes[0].legend()
    fig.tight_layout()
    return fig

def plot_fig3_like_densities(n=200, bins=50, seed=42):
    x_grid = np.linspace(0.001, 0.999, 2000)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()

    for ax, (name, sampler_fn) in zip(axes, FIG3_SAMPLERS.items()):
        samples = sampler_fn(n=n, seed=seed)

        ax.hist(samples, bins=bins, density=True, alpha=0.35, label=f"histogram (n={n})")

        density_fn = FIG3_DENSITIES[name]
        if name != "Discrete Uniform":
            ax.plot(x_grid, density_fn(x_grid), linewidth=2, label="true density")
        else:
            support = np.arange(1, 100) / 100.0
            ax.vlines(
                support,
                ymin=0,
                ymax=1.0,
                linewidth=0.5,
                alpha=0.6,
                label="support points"
            )

        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.grid(True)

    axes[0].legend()
    fig.tight_layout()
    return fig

def estimate_detection_rate_from_sampler_fast(
    sampler_fn,
    tau,
    W=7,
    n=35,
    N=10000,
    prior_ok=0.5,
    seed=1,
    pbar=None,
):
    rng = np.random.default_rng(seed)
    p_ok_vals = np.empty(N, dtype=float)

    for i in range(N):
        samples = sampler_fn(
            n=n,
            seed=rng.integers(0, 2**32 - 1),
        )

        p_ok_vals[i] = p_ok_from_samples_fast(
            samples,
            W=W,
            prior_ok=prior_ok,
        )

        if pbar is not None:
            pbar.update(1)

    alarms = p_ok_vals < tau

    return {
        "DR": float(np.mean(alarms)),
        "mean_p_ok": float(np.mean(p_ok_vals)),
        "std_p_ok": float(np.std(p_ok_vals)),
        "q01_p_ok": float(np.quantile(p_ok_vals, 0.01)),
        "q05_p_ok": float(np.quantile(p_ok_vals, 0.05)),
        "q50_p_ok": float(np.quantile(p_ok_vals, 0.50)),
        "q95_p_ok": float(np.quantile(p_ok_vals, 0.95)),
    }

from tqdm.auto import tqdm

def calibrate_p_ok_threshold_uniform_singlebar(
    W=7,
    n=35,
    alpha=0.01,
    N=10000,
    prior_ok=0.5,
    seed=0,
    pbar=None,
):
    """
    Same as calibrate_p_ok_threshold_uniform, but without its own tqdm bar.
    It updates the shared global pbar instead.
    """
    rng = np.random.default_rng(seed)
    vals = np.empty(N, dtype=float)

    for i in range(N):
        samples = rng.uniform(0.0, 1.0, size=n)
        vals[i] = p_ok_from_samples_fast(
            samples,
            W=W,
            prior_ok=prior_ok,
        )

        if pbar is not None:
            pbar.update(1)

    return float(np.quantile(vals, alpha))

def run_fig3_like_evaluation(
    samplers,
    W=50,
    n=200,
    alpha=0.05,
    N_calib=100000,
    N_power=100000,
    prior_ok=0.5,
    seed=0,
    show_progress=True,
):
    """
    Figure-3-like evaluation:
        H0: Uniform(0,1)
        alternatives: beta, transformed Laplace, discrete uniform
        power = P(P_OK < tau_alpha | alternative)
    """
    total_steps = N_calib + len(samplers) * N_power

    pbar = tqdm(
        total=total_steps,
        desc="Calibrating H0",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    ) if show_progress else None

    tau = calibrate_p_ok_threshold_uniform_singlebar(
        W=W,
        n=n,
        alpha=alpha,
        N=N_calib,
        prior_ok=prior_ok,
        seed=seed,
        pbar=pbar,
    )

    results = {}

    for j, (name, sampler_fn) in enumerate(samplers.items()):
        if pbar is not None:
            pbar.set_description(f"Power: {name}")

        res = estimate_detection_rate_from_sampler_fast(
            sampler_fn=sampler_fn,
            tau=tau,
            W=W,
            n=n,
            N=N_power,
            prior_ok=prior_ok,
            seed=seed + 1000 + j,
            pbar=pbar,
        )

        results[name] = {
            "power": res["DR"],
            "mean_p_ok": res["mean_p_ok"],
            "std_p_ok": res["std_p_ok"],
            "q01_p_ok": res["q01_p_ok"],
            "q05_p_ok": res["q05_p_ok"],
            "q50_p_ok": res["q50_p_ok"],
            "q95_p_ok": res["q95_p_ok"],
            "tau": tau,
            "W": W,
            "n": n,
            "alpha": alpha,
        }

    if pbar is not None:
        pbar.set_description("Done")
        pbar.close()

    return tau, results

def calibrate_p_ok_threshold_uniform(
    W=7,
    n=35,
    alpha=0.01,
    N=10000,
    prior_ok=0.5,
    seed=0,
    show_progress=True,
):
    """
    Calibrates lower threshold for P_OK under H0: U(0,1).

    Alarm if:
        P_OK < tau
    """
    rng = np.random.default_rng(seed)
    vals = []

    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Calibrating H0 W={W}, n={n}, alpha={alpha}", unit="run", leave=True)

    for _ in iterator:
        samples = rng.uniform(0.0, 1.0, size=n)

        op_multi = samples_to_multinomial_opinion_fast(samples, W=W)
        op_bin = multinomial_opinion_to_binomial_ok_opinion(
            op_multi,
            W=W,
            prior_ok=prior_ok,
        )

        vals.append(op_bin.getProjection()[0])

    return np.quantile(vals, alpha)


def estimate_detection_rate_fast(
    density_fn,
    tau,
    W=7,
    n=35,
    N=10000,
    prior_ok=0.5,
    seed=1,
    name="Alternative",
    show_progress=True,
    tqdm_position=0,
):
    rng = np.random.default_rng(seed)
    p_ok_vals = np.empty(N, dtype=float)

    iterator = range(N)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=f"Power: {name}",
            unit="run",
            leave=True,
            position=tqdm_position,
            dynamic_ncols=True,
        )

    for i in iterator:
        samples = sample_from_density(
            density_fn,
            n=n,
            seed=rng.integers(0, 2**32 - 1),
        )
        p_ok_vals[i] = p_ok_from_samples_fast(samples, W=W, prior_ok=prior_ok)

    alarms = p_ok_vals < tau

    return {
        "DR": float(np.mean(alarms)),
        "mean_p_ok": float(np.mean(p_ok_vals)),
        "std_p_ok": float(np.std(p_ok_vals)),
        "q01_p_ok": float(np.quantile(p_ok_vals, 0.01)),
        "q05_p_ok": float(np.quantile(p_ok_vals, 0.05)),
        "q50_p_ok": float(np.quantile(p_ok_vals, 0.50)),
        "q95_p_ok": float(np.quantile(p_ok_vals, 0.95)),
    }

def estimate_detection_rate_from_density_fast(
    density_fn,
    tau,
    W=7,
    n=35,
    N=10000,
    prior_ok=0.5,
    seed=1,
    pbar=None,
):
    rng = np.random.default_rng(seed)
    p_ok_vals = np.empty(N, dtype=float)

    for i in range(N):
        samples = sample_from_density(
            density_fn,
            n=n,
            seed=rng.integers(0, 2**32 - 1),
        )

        p_ok_vals[i] = p_ok_from_samples_fast(
            samples,
            W=W,
            prior_ok=prior_ok,
        )

        if pbar is not None:
            pbar.update(1)

    alarms = p_ok_vals < tau

    return {
        "DR": float(np.mean(alarms)),
        "mean_p_ok": float(np.mean(p_ok_vals)),
        "std_p_ok": float(np.std(p_ok_vals)),
        "q01_p_ok": float(np.quantile(p_ok_vals, 0.01)),
        "q05_p_ok": float(np.quantile(p_ok_vals, 0.05)),
        "q50_p_ok": float(np.quantile(p_ok_vals, 0.50)),
        "q95_p_ok": float(np.quantile(p_ok_vals, 0.95)),
    }

def run_table1_like_evaluation(
    densities,
    W=20,
    n=200,
    alpha=0.05,
    N_calib=100000,
    N_power=100000,
    prior_ok=0.5,
    seed=0,
    show_progress=True,
):
    """
    Semantically matches Table 1 setup:
        H0: Uniform(0,1)
        sample size n=200
        level alpha=0.05
        power = rejection rate under each alternative
    """

    total_steps = N_calib + len(densities) * N_power

    pbar = tqdm(
        total=total_steps,
        desc="Calibrating H0",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    ) if show_progress else None

    tau = calibrate_p_ok_threshold_uniform_singlebar(
        W=W,
        n=n,
        alpha=alpha,
        N=N_calib,
        prior_ok=prior_ok,
        seed=seed,
        pbar=pbar,
    )

    results = {}

    for j, (name, density_fn) in enumerate(densities.items()):
        if pbar is not None:
            pbar.set_description(f"Power: {name}")

        res = estimate_detection_rate_from_density_fast(
            density_fn=density_fn,
            tau=tau,
            W=W,
            n=n,
            N=N_power,
            prior_ok=prior_ok,
            seed=seed + 1000 + j,
            pbar=pbar,
        )

        results[name] = {
            "power": res["DR"],
            "mean_p_ok": res["mean_p_ok"],
            "std_p_ok": res["std_p_ok"],
            "q01_p_ok": res["q01_p_ok"],
            "q05_p_ok": res["q05_p_ok"],
            "q50_p_ok": res["q50_p_ok"],
            "q95_p_ok": res["q95_p_ok"],
            "tau": tau,
            "W": W,
            "n": n,
            "alpha": alpha,
        }

    if pbar is not None:
        pbar.set_description("Done")
        pbar.close()

    return tau, results


import json
import csv
from pathlib import Path
def save_results_table(results, output_dir="results", filename_prefix="gof_results"):
    """
    Saves the printed result table in reusable formats:
        - JSON: full structured results
        - CSV: table format
        - TEX: LaTeX table rows for paper integration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"
    tex_path = output_dir / f"{filename_prefix}.tex"

    # -----------------------------
    # JSON
    # -----------------------------
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------
    # CSV
    # -----------------------------
    fieldnames = [
        "name",
        "power",
        "mean_p_ok",
        "std_p_ok",
        "q01_p_ok",
        "q05_p_ok",
        "q50_p_ok",
        "q95_p_ok",
        "tau",
        "W",
        "n",
        "alpha",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, res in results.items():
            row = {"name": name}
            row.update(res)
            writer.writerow(row)

    # -----------------------------
    # LaTeX table rows
    # -----------------------------
    with open(tex_path, "w") as f:
        f.write("% Automatically generated result table rows\n")
        f.write("% name & power & mean P_OK & std P_OK & q01 & q50 & q95 \\\\\n")

        for name, res in results.items():
            f.write(
                f"{name} & "
                f"{res['power']:.3f} & "
                f"{res['mean_p_ok']:.3f} & "
                f"{res['std_p_ok']:.3f} & "
                f"{res['q01_p_ok']:.3f} & "
                f"{res['q50_p_ok']:.3f} & "
                f"{res['q95_p_ok']:.3f} \\\\\n"
            )

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "tex": str(tex_path),
    }


# ============================================================
# Beta-credible end-of-run classification
# ============================================================

REGION_CONFIDENTLY_CONSISTENT = 1
REGION_UNDECIDED = 0
REGION_CONFIDENTLY_INCONSISTENT = -1


def binomial_opinion_components_from_samples_fast(
    samples,
    W=7,
    prior_ok=0.5,
    eps=1e-12,
):
    """
    Computes the binomial OK opinion induced by a complete sample run.

    Returns
    -------
    dict
        belief, disbelief, uncertainty, projected_probability
    """
    counts, _ = np.histogram(
        samples,
        bins=np.linspace(0.0, 1.0, W + 1),
    )
    counts = counts.astype(float)

    total_evidence = float(np.sum(counts))
    if total_evidence <= eps:
        return {
            "belief": 0.0,
            "disbelief": 0.0,
            "uncertainty": 1.0,
            "projected_probability": float(prior_ok),
        }

    uncertainty = W / (W + total_evidence)
    committed_mass = 1.0 - uncertainty

    empirical_bin_distribution = counts / total_evidence
    uniform_reference = np.ones(W, dtype=float) / W

    tv_distance = 0.5 * np.sum(
        np.abs(empirical_bin_distribution - uniform_reference)
    )
    tv_max = 1.0 - 1.0 / W

    disbelief = committed_mass * tv_distance / tv_max
    disbelief = float(np.clip(disbelief, 0.0, committed_mass))
    belief = float(committed_mass - disbelief)

    projected_probability = belief + prior_ok * uncertainty

    return {
        "belief": belief,
        "disbelief": disbelief,
        "uncertainty": float(uncertainty),
        "projected_probability": float(projected_probability),
    }


def beta_credible_probabilities(
    belief,
    disbelief,
    uncertainty,
    prior_ok=0.5,
    tau_ok=0.5,
    eps=1e-12,
):
    """
    Computes the posterior probabilities

        q_consistent   = Pr(theta > tau_ok | omega)
        q_inconsistent = Pr(theta < tau_ok | omega)

    for the beta distribution induced by a binomial subjective-logic
    opinion omega = (belief, disbelief, uncertainty, prior_ok).

    The implementation accepts scalars or NumPy arrays.
    """
    belief = np.asarray(belief, dtype=float)
    disbelief = np.asarray(disbelief, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    q_consistent = np.empty_like(uncertainty, dtype=float)
    q_inconsistent = np.empty_like(uncertainty, dtype=float)

    dogmatic = uncertainty <= eps
    non_dogmatic = ~dogmatic

    if np.any(non_dogmatic):
        alpha_beta = (
            2.0 * belief[non_dogmatic] / uncertainty[non_dogmatic]
            + 2.0 * prior_ok
        )
        beta_beta = (
            2.0 * disbelief[non_dogmatic] / uncertainty[non_dogmatic]
            + 2.0 * (1.0 - prior_ok)
        )

        q_inconsistent[non_dogmatic] = beta_dist.cdf(
            tau_ok,
            alpha_beta,
            beta_beta,
        )
        q_consistent[non_dogmatic] = 1.0 - q_inconsistent[non_dogmatic]

    if np.any(dogmatic):
        p_ok = belief[dogmatic] + prior_ok * uncertainty[dogmatic]
        q_consistent[dogmatic] = np.where(
            p_ok > tau_ok,
            1.0,
            np.where(p_ok < tau_ok, 0.0, 0.5),
        )
        q_inconsistent[dogmatic] = 1.0 - q_consistent[dogmatic]

    if q_consistent.ndim == 0:
        return float(q_consistent), float(q_inconsistent)

    return q_consistent, q_inconsistent


def classify_beta_credible_regions(
    q_consistent,
    q_inconsistent,
    eta=0.95,
):
    """
    Classifies each end-of-run opinion as

        +1: confidently consistent
         0: undecided
        -1: confidently inconsistent

    For eta > 0.5, the two confident regions are disjoint because
    q_consistent + q_inconsistent = 1 for the continuous beta model.
    """
    if not 0.5 < eta < 1.0:
        raise ValueError(
            "eta must satisfy 0.5 < eta < 1.0 so that the confident "
            "decision regions remain disjoint."
        )

    q_consistent = np.asarray(q_consistent, dtype=float)
    q_inconsistent = np.asarray(q_inconsistent, dtype=float)

    region = np.full(
        q_consistent.shape,
        REGION_UNDECIDED,
        dtype=np.int8,
    )
    region[q_consistent >= eta] = REGION_CONFIDENTLY_CONSISTENT
    region[q_inconsistent >= eta] = REGION_CONFIDENTLY_INCONSISTENT

    if region.ndim == 0:
        return int(region)

    return region


def evaluate_end_of_run_regions_from_sampler(
    sampler_fn,
    p_ok_threshold,
    W=7,
    n=35,
    N=10000,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
    seed=1,
    is_null=False,
    pbar=None,
):
    """
    Repeats a complete GoF run N times. For every run, the final
    binomial opinion is classified using the beta-credible decision
    regions.

    Under H0 (uniformity), the correct region is confidently consistent.
    Under an alternative, the correct region is confidently inconsistent.
    """
    rng = np.random.default_rng(seed)

    p_ok_values = np.empty(N, dtype=float)
    belief_values = np.empty(N, dtype=float)
    disbelief_values = np.empty(N, dtype=float)
    uncertainty_values = np.empty(N, dtype=float)

    for i in range(N):
        samples = sampler_fn(
            n=n,
            seed=rng.integers(0, 2**32 - 1),
        )

        components = binomial_opinion_components_from_samples_fast(
            samples,
            W=W,
            prior_ok=prior_ok,
        )

        belief_values[i] = components["belief"]
        disbelief_values[i] = components["disbelief"]
        uncertainty_values[i] = components["uncertainty"]
        p_ok_values[i] = components["projected_probability"]

        if pbar is not None:
            pbar.update(1)

    q_consistent, q_inconsistent = beta_credible_probabilities(
        belief=belief_values,
        disbelief=disbelief_values,
        uncertainty=uncertainty_values,
        prior_ok=prior_ok,
        tau_ok=tau_ok,
    )

    regions = classify_beta_credible_regions(
        q_consistent=q_consistent,
        q_inconsistent=q_inconsistent,
        eta=eta,
    )

    consistent_mask = regions == REGION_CONFIDENTLY_CONSISTENT
    undecided_mask = regions == REGION_UNDECIDED
    inconsistent_mask = regions == REGION_CONFIDENTLY_INCONSISTENT
    decided_mask = ~undecided_mask

    correct_region = (
        REGION_CONFIDENTLY_CONSISTENT
        if is_null
        else REGION_CONFIDENTLY_INCONSISTENT
    )
    incorrect_region = -correct_region

    correct_mask = regions == correct_region
    false_mask = regions == incorrect_region

    decision_coverage = float(np.mean(decided_mask))
    selective_accuracy = (
        float(np.mean(correct_mask[decided_mask]))
        if np.any(decided_mask)
        else float("nan")
    )

    credible_consistent_rate = float(np.mean(consistent_mask))
    credible_undecided_rate = float(np.mean(undecided_mask))
    credible_inconsistent_rate = float(np.mean(inconsistent_mask))

    legacy_rejection_rate = float(np.mean(p_ok_values < p_ok_threshold))

    result = {
        # Legacy calibrated P_OK test
        "legacy_rejection_rate": legacy_rejection_rate,

        # Beta-credible region occupancy
        "credible_consistent_rate": credible_consistent_rate,
        "credible_undecided_rate": credible_undecided_rate,
        "credible_inconsistent_rate": credible_inconsistent_rate,
        "not_confidently_consistent_rate": 1.0 - credible_consistent_rate,

        # Classification metrics with abstention
        "credible_correct_rate": float(np.mean(correct_mask)),
        "credible_false_classification_rate": float(np.mean(false_mask)),
        "decision_coverage": decision_coverage,
        "selective_accuracy": selective_accuracy,

        # Explicit GoF interpretation
        "credible_detection_power": (
            None if is_null else credible_inconsistent_rate
        ),
        "credible_false_alarm_rate": (
            credible_inconsistent_rate if is_null else None
        ),

        # Continuous summary values
        "mean_p_ok": float(np.mean(p_ok_values)),
        "std_p_ok": float(np.std(p_ok_values)),
        "q01_p_ok": float(np.quantile(p_ok_values, 0.01)),
        "q05_p_ok": float(np.quantile(p_ok_values, 0.05)),
        "q50_p_ok": float(np.quantile(p_ok_values, 0.50)),
        "q95_p_ok": float(np.quantile(p_ok_values, 0.95)),
        "mean_q_consistent": float(np.mean(q_consistent)),
        "mean_q_inconsistent": float(np.mean(q_inconsistent)),
        "q05_q_inconsistent": float(np.quantile(q_inconsistent, 0.05)),
        "q50_q_inconsistent": float(np.quantile(q_inconsistent, 0.50)),
        "q95_q_inconsistent": float(np.quantile(q_inconsistent, 0.95)),

        # Configuration
        "p_ok_threshold": float(p_ok_threshold),
        "W": int(W),
        "n": int(n),
        "prior_ok": float(prior_ok),
        "eta": float(eta),
        "tau_ok": float(tau_ok),
        "is_null": bool(is_null),
    }

    return result


def make_density_sampler(density_fn):
    """Wraps a density on [0, 1] as a sampler with the common API."""
    def sampler(n, seed=None):
        return sample_from_density(
            density_fn=density_fn,
            n=n,
            seed=seed,
        )

    return sampler


def run_table1_like_evaluation_with_regions(
    densities,
    W=20,
    n=200,
    alpha=0.05,
    N_calib=100000,
    N_power=100000,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
    seed=0,
    show_progress=True,
):
    """
    Extends the Table-1-like evaluation with an end-of-run
    beta-credible classification of the final binomial opinion.

    Results include the uniform null model because its consistent,
    undecided, and false-inconsistent rates are required to interpret
    the alternative-distribution results.
    """
    total_steps = N_calib + (len(densities) + 1) * N_power

    pbar = tqdm(
        total=total_steps,
        desc="Calibrating H0",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    ) if show_progress else None

    p_ok_threshold = calibrate_p_ok_threshold_uniform_singlebar(
        W=W,
        n=n,
        alpha=alpha,
        N=N_calib,
        prior_ok=prior_ok,
        seed=seed,
        pbar=pbar,
    )

    results = {}

    evaluation_sources = [
        ("Uniform (H0)", sample_uniform, True),
    ]
    evaluation_sources.extend(
        (name, make_density_sampler(density_fn), False)
        for name, density_fn in densities.items()
    )

    for j, (name, sampler_fn, is_null) in enumerate(evaluation_sources):
        if pbar is not None:
            pbar.set_description(f"Regions: {name}")

        result = evaluate_end_of_run_regions_from_sampler(
            sampler_fn=sampler_fn,
            p_ok_threshold=p_ok_threshold,
            W=W,
            n=n,
            N=N_power,
            prior_ok=prior_ok,
            eta=eta,
            tau_ok=tau_ok,
            seed=seed + 1000 + j,
            is_null=is_null,
            pbar=pbar,
        )
        result["alpha"] = float(alpha)
        results[name] = result

    if pbar is not None:
        pbar.set_description("Done")
        pbar.close()

    return p_ok_threshold, results


def run_fig3_like_evaluation_with_regions(
    samplers,
    W=50,
    n=200,
    alpha=0.05,
    N_calib=100000,
    N_power=100000,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
    seed=0,
    show_progress=True,
):
    """
    Extends the Figure-3-like evaluation with beta-credible
    end-of-run classification of the final binomial opinion.
    """
    total_steps = N_calib + len(samplers) * N_power

    pbar = tqdm(
        total=total_steps,
        desc="Calibrating H0",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    ) if show_progress else None

    p_ok_threshold = calibrate_p_ok_threshold_uniform_singlebar(
        W=W,
        n=n,
        alpha=alpha,
        N=N_calib,
        prior_ok=prior_ok,
        seed=seed,
        pbar=pbar,
    )

    results = {}

    for j, (name, sampler_fn) in enumerate(samplers.items()):
        if pbar is not None:
            pbar.set_description(f"Regions: {name}")

        is_null = name.lower().startswith("uniform")

        result = evaluate_end_of_run_regions_from_sampler(
            sampler_fn=sampler_fn,
            p_ok_threshold=p_ok_threshold,
            W=W,
            n=n,
            N=N_power,
            prior_ok=prior_ok,
            eta=eta,
            tau_ok=tau_ok,
            seed=seed + 1000 + j,
            is_null=is_null,
            pbar=pbar,
        )
        result["alpha"] = float(alpha)
        results[name] = result

    if pbar is not None:
        pbar.set_description("Done")
        pbar.close()

    return p_ok_threshold, results


def save_results_table_with_regions(
    results,
    output_dir="results",
    filename_prefix="gof_results_with_regions",
):
    """
    Saves the extended evaluation in JSON, CSV, and LaTeX formats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"
    tex_path = output_dir / f"{filename_prefix}.tex"

    with open(json_path, "w") as file:
        json.dump(results, file, indent=2, allow_nan=True)

    fieldnames = [
        "name",
        "is_null",
        "legacy_rejection_rate",
        "credible_consistent_rate",
        "credible_undecided_rate",
        "credible_inconsistent_rate",
        "credible_correct_rate",
        "credible_false_classification_rate",
        "decision_coverage",
        "selective_accuracy",
        "credible_detection_power",
        "credible_false_alarm_rate",
        "not_confidently_consistent_rate",
        "mean_p_ok",
        "std_p_ok",
        "q01_p_ok",
        "q05_p_ok",
        "q50_p_ok",
        "q95_p_ok",
        "mean_q_consistent",
        "mean_q_inconsistent",
        "q05_q_inconsistent",
        "q50_q_inconsistent",
        "q95_q_inconsistent",
        "p_ok_threshold",
        "W",
        "n",
        "alpha",
        "prior_ok",
        "eta",
        "tau_ok",
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for name, result in results.items():
            row = {"name": name}
            row.update(result)
            writer.writerow({key: row.get(key) for key in fieldnames})

    with open(tex_path, "w") as file:
        file.write("% Automatically generated GoF region-classification rows\n")
        file.write(
            "% Distribution & legacy rejection & consistent & undecided & "
            "inconsistent & coverage & selective accuracy \\\\\n"
        )

        for name, result in results.items():
            selective_accuracy = result["selective_accuracy"]
            selective_accuracy_text = (
                "--"
                if np.isnan(selective_accuracy)
                else f"{selective_accuracy:.3f}"
            )

            file.write(
                f"{name} & "
                f"{result['legacy_rejection_rate']:.3f} & "
                f"{result['credible_consistent_rate']:.3f} & "
                f"{result['credible_undecided_rate']:.3f} & "
                f"{result['credible_inconsistent_rate']:.3f} & "
                f"{result['decision_coverage']:.3f} & "
                f"{selective_accuracy_text} \\\\\n"
            )

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "tex": str(tex_path),
    }


def plot_beta_credible_region_rates(
    results,
    title,
    output_path=None,
):
    """
    Plots the end-of-run occupancy of the three beta-credible regions.
    """
    names = list(results.keys())
    consistent = np.array(
        [results[name]["credible_consistent_rate"] for name in names]
    )
    undecided = np.array(
        [results[name]["credible_undecided_rate"] for name in names]
    )
    inconsistent = np.array(
        [results[name]["credible_inconsistent_rate"] for name in names]
    )

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(names)), 5.2))
    ax.bar(
        x,
        consistent,
        color="#3A923A",
        label="Confidently consistent",
    )
    ax.bar(
        x,
        undecided,
        bottom=consistent,
        color="#A9A9A9",
        label="Undecided",
    )
    ax.bar(
        x,
        inconsistent,
        bottom=consistent + undecided,
        color="#C83E3E",
        label="Confidently inconsistent",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of Monte Carlo runs")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def print_region_evaluation(results):
    """Prints a compact summary of the extended GoF evaluation."""
    print(
        "\n"
        "Distribution                 | Legacy reject | Consistent | "
        "Undecided | Inconsistent | Coverage | Selective accuracy"
    )
    print("-" * 115)

    for name, result in results.items():
        selective_accuracy = result["selective_accuracy"]
        selective_text = (
            "   n/a"
            if np.isnan(selective_accuracy)
            else f"{selective_accuracy:7.3f}"
        )

        print(
            f"{name:28s} | "
            f"{result['legacy_rejection_rate']:13.3f} | "
            f"{result['credible_consistent_rate']:10.3f} | "
            f"{result['credible_undecided_rate']:9.3f} | "
            f"{result['credible_inconsistent_rate']:12.3f} | "
            f"{result['decision_coverage']:8.3f} | "
            f"{selective_text}"
        )




# ============================================================
# Barycentric SL triangle for Monte Carlo end-of-run opinions
# ============================================================

def opinion_to_cartesian(belief, disbelief, uncertainty):
    """Maps barycentric SL coordinates (b, d, u) to an equilateral triangle."""
    belief = np.asarray(belief, dtype=float)
    disbelief = np.asarray(disbelief, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    x = disbelief + 0.5 * uncertainty
    y = (np.sqrt(3.0) / 2.0) * uncertainty
    return x, y


def collect_end_of_run_opinions_from_sampler(
    sampler_fn,
    W=7,
    n=200,
    N=3000,
    prior_ok=0.5,
    seed=1,
):
    """Generates final binomial opinions for a Monte Carlo visualization."""
    rng = np.random.default_rng(seed)

    belief = np.empty(N, dtype=float)
    disbelief = np.empty(N, dtype=float)
    uncertainty = np.empty(N, dtype=float)
    projected_probability = np.empty(N, dtype=float)

    for i in range(N):
        samples = sampler_fn(
            n=n,
            seed=rng.integers(0, 2**32 - 1),
        )
        components = binomial_opinion_components_from_samples_fast(
            samples=samples,
            W=W,
            prior_ok=prior_ok,
        )
        belief[i] = components["belief"]
        disbelief[i] = components["disbelief"]
        uncertainty[i] = components["uncertainty"]
        projected_probability[i] = components["projected_probability"]

    return {
        "belief": belief,
        "disbelief": disbelief,
        "uncertainty": uncertainty,
        "projected_probability": projected_probability,
    }


def _decision_region_grid(
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
    resolution=100,
):
    """Creates a triangular lattice and classifies every lattice point."""
    beliefs = []
    disbeliefs = []
    uncertainties = []

    for i in range(resolution + 1):
        uncertainty = i / resolution
        remaining = 1.0 - uncertainty
        n_row = resolution - i

        if n_row == 0:
            beliefs.append(0.0)
            disbeliefs.append(0.0)
            uncertainties.append(1.0)
            continue

        for j in range(n_row + 1):
            disbelief = remaining * j / n_row
            belief = remaining - disbelief
            beliefs.append(belief)
            disbeliefs.append(disbelief)
            uncertainties.append(uncertainty)

    beliefs = np.asarray(beliefs, dtype=float)
    disbeliefs = np.asarray(disbeliefs, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    q_consistent, q_inconsistent = beta_credible_probabilities(
        belief=beliefs,
        disbelief=disbeliefs,
        uncertainty=uncertainties,
        prior_ok=prior_ok,
        tau_ok=tau_ok,
    )
    regions = classify_beta_credible_regions(
        q_consistent=q_consistent,
        q_inconsistent=q_inconsistent,
        eta=eta,
    )

    x, y = opinion_to_cartesian(
        belief=beliefs,
        disbelief=disbeliefs,
        uncertainty=uncertainties,
    )

    return {
        "belief": beliefs,
        "disbelief": disbeliefs,
        "uncertainty": uncertainties,
        "x": x,
        "y": y,
        "q_consistent": q_consistent,
        "q_inconsistent": q_inconsistent,
        "regions": regions,
    }


def plot_mc_opinions_in_sl_triangle(
    samplers,
    W=7,
    n=200,
    N_per_distribution=3000,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
    seed=1,
    title="Monte Carlo end-of-run opinions in the SL triangle",
    output_path=None,
):
    """
    Plots Monte Carlo end-of-run binomial opinions together with the
    beta-credible consistent, undecided, and inconsistent regions.

    For fixed W and n, all opinions have the same uncertainty

        u = W / (W + n),

    and therefore lie on a line parallel to the b-d edge.
    """
    grid = _decision_region_grid(
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
        resolution=120,
    )

    triangulation = mtri.Triangulation(grid["x"], grid["y"])

    fig, ax = plt.subplots(figsize=(10.5, 8.8))

    region_colors = ["#C95A5A", "#D8D8D8", "#65A765"]
    ax.tricontourf(
        triangulation,
        grid["regions"],
        levels=[-1.5, -0.5, 0.5, 1.5],
        colors=region_colors,
        alpha=0.22,
    )

    # Exact beta-credible decision boundaries.
    ax.tricontour(
        triangulation,
        grid["q_consistent"],
        levels=[1.0 - eta],
        colors=["darkred"],
        linewidths=2.0,
    )
    ax.tricontour(
        triangulation,
        grid["q_consistent"],
        levels=[eta],
        colors=["darkgreen"],
        linewidths=2.0,
    )

    palette = plt.get_cmap("tab10")
    all_opinions = {}

    for index, (name, sampler_fn) in enumerate(samplers.items()):
        opinions = collect_end_of_run_opinions_from_sampler(
            sampler_fn=sampler_fn,
            W=W,
            n=n,
            N=N_per_distribution,
            prior_ok=prior_ok,
            seed=seed + 1000 * index,
        )
        all_opinions[name] = opinions

        x, y = opinion_to_cartesian(
            belief=opinions["belief"],
            disbelief=opinions["disbelief"],
            uncertainty=opinions["uncertainty"],
        )

        color = palette(index % 10)
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.20,
            color=color,
            edgecolors="none",
            label=f"{name} (N={N_per_distribution})",
            rasterized=True,
        )

        # Median opinion as a clearly visible marker.
        median_b = float(np.median(opinions["belief"]))
        median_d = float(np.median(opinions["disbelief"]))
        median_u = float(np.median(opinions["uncertainty"]))
        median_x, median_y = opinion_to_cartesian(
            median_b,
            median_d,
            median_u,
        )
        ax.scatter(
            [median_x],
            [median_y],
            s=70,
            marker="x",
            linewidths=2.2,
            color=color,
            zorder=5,
        )

    # Triangle outline.
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]
    ax.plot(triangle_x, triangle_y, color="black", linewidth=1.8)

    # Constant-uncertainty line generated by fixed W and n.
    fixed_u = W / (W + n)
    fixed_y = (np.sqrt(3.0) / 2.0) * fixed_u
    ax.plot(
        [0.5 * fixed_u, 1.0 - 0.5 * fixed_u],
        [fixed_y, fixed_y],
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7,
        label=rf"fixed uncertainty $u=W/(W+n)={fixed_u:.3f}$",
    )

    ax.text(-0.025, -0.025, r"$b=1$", ha="right", va="top", fontsize=13)
    ax.text(1.025, -0.025, r"$d=1$", ha="left", va="top", fontsize=13)
    ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.025, r"$u=1$", ha="center", va="bottom", fontsize=13)

    ax.text(0.13, 0.17, "confidently\nconsistent", color="darkgreen", ha="center")
    ax.text(0.50, 0.52, "undecided", color="#505050", ha="center")
    ax.text(0.87, 0.17, "confidently\ninconsistent", color="darkred", ha="center")

    ax.set_title(
        title
        + "\n"
        + rf"$W={W}$, $n={n}$, $a={prior_ok:.2f}$, "
          rf"$\eta={eta:.2f}$, $\tau_{{\mathrm{{OK}}}}={tau_ok:.2f}$"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.07, np.sqrt(3.0) / 2.0 + 0.10)
    ax.axis("off")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, all_opinions


def _q_consistent_from_normalized_conflict(
    normalized_conflict,
    uncertainty,
    prior_ok=0.5,
    tau_ok=0.5,
):
    """Credible consistency probability as a function of normalized TV conflict."""
    committed_mass = 1.0 - uncertainty
    disbelief = committed_mass * normalized_conflict
    belief = committed_mass - disbelief
    q_consistent, _ = beta_credible_probabilities(
        belief=belief,
        disbelief=disbelief,
        uncertainty=uncertainty,
        prior_ok=prior_ok,
        tau_ok=tau_ok,
    )
    return float(q_consistent)


def normalized_conflict_boundaries_for_fixed_uncertainty(
    W,
    n,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
):
    """Returns normalized-conflict boundaries for fixed W and n."""
    uncertainty = W / (W + n)

    consistent_boundary = brentq(
        lambda delta: (
            _q_consistent_from_normalized_conflict(
                delta,
                uncertainty,
                prior_ok=prior_ok,
                tau_ok=tau_ok,
            )
            - eta
        ),
        0.0,
        1.0,
    )
    inconsistent_boundary = brentq(
        lambda delta: (
            _q_consistent_from_normalized_conflict(
                delta,
                uncertainty,
                prior_ok=prior_ok,
                tau_ok=tau_ok,
            )
            - (1.0 - eta)
        ),
        0.0,
        1.0,
    )

    return consistent_boundary, inconsistent_boundary


def print_region_mapping_diagnostics(
    W,
    n,
    legacy_p_ok_threshold,
    prior_ok=0.5,
    eta=0.95,
    tau_ok=0.5,
):
    """Explains the relation between the legacy threshold and credible regions."""
    uncertainty = W / (W + n)
    committed_mass = 1.0 - uncertainty

    consistent_delta, inconsistent_delta = (
        normalized_conflict_boundaries_for_fixed_uncertainty(
            W=W,
            n=n,
            prior_ok=prior_ok,
            eta=eta,
            tau_ok=tau_ok,
        )
    )

    # For a=0.5, P_OK = 0.5 + (1-u)(0.5-delta).
    # The general expression used below follows directly from the mapping.
    legacy_delta = (
        committed_mass + prior_ok * uncertainty - legacy_p_ok_threshold
    ) / committed_mass

    print("\nMapping diagnostics for fixed W and n")
    print("-------------------------------------")
    print(f"Fixed opinion uncertainty: u = W/(W+n) = {uncertainty:.6f}")
    print(
        "Legacy rejection begins at approximately normalized conflict "
        f"delta > {legacy_delta:.6f}."
    )
    print(
        "Beta-credible classification with the current eta and tau_OK gives:"
    )
    print(
        f"  confidently consistent for delta <= {consistent_delta:.6f}"
    )
    print(
        f"  undecided for {consistent_delta:.6f} < delta < "
        f"{inconsistent_delta:.6f}"
    )
    print(
        f"  confidently inconsistent for delta >= {inconsistent_delta:.6f}"
    )
    print(
        "The legacy GoF threshold and the beta-credible region boundary "
        "therefore answer different questions and are not expected to "
        "produce the same detection rates."
    )


# ============================================================
# Main evaluation
# ============================================================

def main():
    W = 100
    n = 200
    alpha = 0.05
    prior_ok = 0.5

    # Beta-credible decision-policy parameters
    eta = 0.95
    tau_ok = 0.50

    N_calib = 100000
    N_power = 100000
    seed = 42

    p_ok_threshold, results_fig1 = run_table1_like_evaluation_with_regions(
        densities=DENSITIES,
        W=W,
        n=n,
        alpha=alpha,
        N_calib=N_calib,
        N_power=N_power,
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
        seed=seed,
        show_progress=True,
    )

    print(f"\nCalibrated legacy P_OK threshold: {p_ok_threshold:.6f}")
    print(
        "Beta-credible region parameters: "
        f"eta={eta:.3f}, tau_OK={tau_ok:.3f}, prior={prior_ok:.3f}"
    )
    print_region_evaluation(results_fig1)
    print_region_mapping_diagnostics(
        W=W,
        n=n,
        legacy_p_ok_threshold=p_ok_threshold,
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
    )

    paths_fig1 = save_results_table_with_regions(
        results_fig1,
        output_dir="results",
        filename_prefix="fig1_like_gof_results_with_regions",
    )
    print("Saved Fig. 1-like extended results:", paths_fig1)

    plot_fig1_like_densities(n=n, bins=W, seed=1)
    plot_beta_credible_region_rates(
        results_fig1,
        title=(
            "End-of-run beta-credible classification: local "
            f"uniformity alternatives (eta={eta:.2f}, tau_OK={tau_ok:.2f})"
        ),
        output_path="results/fig1_like_beta_credible_regions.png",
    )

    fig1_triangle_samplers = {
        "Uniform (H0)": sample_uniform,
        **{
            name: make_density_sampler(density_fn)
            for name, density_fn in DENSITIES.items()
        },
    }
    plot_mc_opinions_in_sl_triangle(
        samplers=fig1_triangle_samplers,
        W=W,
        n=n,
        N_per_distribution=min(3000, N_power),
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
        seed=seed + 20000,
        title="Monte Carlo opinions: local uniformity alternatives",
        output_path="results/fig1_like_mc_opinions_sl_triangle.png",
    )

    p_ok_threshold, results_fig3 = run_fig3_like_evaluation_with_regions(
        samplers=FIG3_SAMPLERS,
        W=W,
        n=n,
        alpha=alpha,
        N_calib=N_calib,
        N_power=N_power,
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
        seed=seed,
        show_progress=True,
    )

    print(f"\nCalibrated legacy P_OK threshold: {p_ok_threshold:.6f}")
    print_region_evaluation(results_fig3)

    paths_fig3 = save_results_table_with_regions(
        results_fig3,
        output_dir="results",
        filename_prefix="fig3_like_gof_results_with_regions",
    )
    print("Saved Fig. 3-like extended results:", paths_fig3)

    plot_fig3_like_densities(n=n, bins=W, seed=1)
    plot_beta_credible_region_rates(
        results_fig3,
        title=(
            "End-of-run beta-credible classification: distributional "
            f"alternatives (eta={eta:.2f}, tau_OK={tau_ok:.2f})"
        ),
        output_path="results/fig3_like_beta_credible_regions.png",
    )

    plot_mc_opinions_in_sl_triangle(
        samplers=FIG3_SAMPLERS,
        W=W,
        n=n,
        N_per_distribution=min(3000, N_power),
        prior_ok=prior_ok,
        eta=eta,
        tau_ok=tau_ok,
        seed=seed + 40000,
        title="Monte Carlo opinions: distributional alternatives",
        output_path="results/fig3_like_mc_opinions_sl_triangle.png",
    )

    plt.show()


if __name__ == "__main__":
    main()