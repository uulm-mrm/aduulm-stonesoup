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

# PARAMETERS
W = 20
n = 200
alpha = 0.05
prior_ok = 0.5

tau, results = run_table1_like_evaluation(
    densities=DENSITIES,
    W=W,
    n=n,
    alpha=alpha,
    N_calib=100000,
    N_power=100000,
    prior_ok=prior_ok,
    seed=42,
    show_progress=True,
)

print("tau_0.05 =", tau)

for name, res in results.items():
    print(
        f"{name:15s} | "
        f"power={res['power']:.3f} | "
        f"mean P_OK={res['mean_p_ok']:.3f}"
    )

paths_fig1 = save_results_table(
    results,
    output_dir="results",
    filename_prefix="fig1_like_gof_results"
)

print("Saved Fig. 1-like results:", paths_fig1)

plot_fig1_like_densities(n=n, bins=W, seed=1)

tau, results = run_fig3_like_evaluation(
    samplers=FIG3_SAMPLERS,
    W=W,
    n=n,
    alpha=alpha,
    N_calib=100000,
    N_power=100000,
    prior_ok=prior_ok,
    seed=42,
    show_progress=True,
)

print("tau_0.05 =", tau)

for name, res in results.items():
    print(
        f"{name:22s} | "
        f"power={res['power']:.3f} | "
        f"mean P_OK={res['mean_p_ok']:.3f}"
    )

paths_fig3 = save_results_table(
    results,
    output_dir="results",
    filename_prefix="fig3_like_gof_results"
)

print("Saved Fig. 3-like results:", paths_fig3)

plot_fig3_like_densities(n=n, bins=W, seed=1)
plt.show()