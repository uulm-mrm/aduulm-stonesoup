import threading
import webbrowser

import numpy as np

from scipy.stats import beta
from scipy.optimize import brentq

import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, ctx, no_update


# ============================================================
# 0) Defaults
# ============================================================

DEFAULTS = {
    "eta": 0.95,
    "tau_ok": 0.50,
    "a_prior": 0.50,
    "b_ex": 0.65,
    "u_ex": 0.10,
    "visible_regions": ["consistent", "inconsistent", "undecided"],
}


# ============================================================
# 1) Opinion -> induced beta distribution and decision probabilities
# ============================================================

def beta_parameters_from_opinion(
    b: float,
    d: float,
    u: float,
    a_prior: float = 0.5,
) -> tuple[float, float]:
    """
    Converts a binomial subjective-logic opinion omega = (b, d, u, a)
    into the corresponding beta parameters.

    Standard binomial subjective logic uses a prior weight of W = 2.
    Therefore:

        alpha = 2 b / u + 2 a
        beta  = 2 d / u + 2 (1 - a)

    This function assumes u > 0.
    """
    alpha = 2.0 * b / u + 2.0 * a_prior
    beta_param = 2.0 * d / u + 2.0 * (1.0 - a_prior)

    return alpha, beta_param


def q_consistent(
    b: float,
    d: float,
    u: float,
    a_prior: float = 0.5,
    tau_ok: float = 0.5,
) -> float:
    """
    Computes

        q_OK(omega) = Pr(theta > tau_OK | omega)

    with the beta density induced by a binomial subjective-logic opinion
    omega = (b, d, u, a). The latent variable theta denotes the
    probability of the proposition "consistent".
    """
    eps = 1e-12

    # Dogmatic limit
    if u <= eps:
        p_ok = b + a_prior * u

        if p_ok > tau_ok:
            return 1.0
        if p_ok < tau_ok:
            return 0.0
        return 0.5

    b = np.clip(float(b), 0.0, 1.0)
    d = np.clip(float(d), 0.0, 1.0)
    u = float(u)

    alpha, beta_param = beta_parameters_from_opinion(
        b=b,
        d=d,
        u=u,
        a_prior=a_prior,
    )

    return 1.0 - beta.cdf(tau_ok, alpha, beta_param)


def q_inconsistent(
    b: float,
    d: float,
    u: float,
    a_prior: float = 0.5,
    tau_ok: float = 0.5,
) -> float:
    """
    Computes

        q_NOT_OK(omega) = Pr(theta < tau_OK | omega)

    with the beta density induced by a binomial subjective-logic opinion
    omega = (b, d, u, a). This is the credibility that the latent
    consistency probability is below tau_OK.
    """
    eps = 1e-12

    # Dogmatic limit
    if u <= eps:
        p_ok = b + a_prior * u

        if p_ok < tau_ok:
            return 1.0
        if p_ok > tau_ok:
            return 0.0
        return 0.5

    b = np.clip(float(b), 0.0, 1.0)
    d = np.clip(float(d), 0.0, 1.0)
    u = float(u)

    alpha, beta_param = beta_parameters_from_opinion(
        b=b,
        d=d,
        u=u,
        a_prior=a_prior,
    )

    return beta.cdf(tau_ok, alpha, beta_param)


# ============================================================
# 2) Feasible ranges and coordinate conversions
# ============================================================

def p_ok_feasible_interval(
    u: float,
    a_prior: float = 0.5,
) -> tuple[float, float]:
    """
    Returns the feasible projected-probability interval for a fixed
    uncertainty u:

        P_OK = b + a u.

    Since b in [0, 1-u], the feasible interval is

        [a u, 1 - (1-a) u].
    """
    p_low = a_prior * u
    p_high = 1.0 - (1.0 - a_prior) * u

    return p_low, p_high


def opinion_from_u_and_p_ok(
    u: float,
    p_ok: float,
    a_prior: float = 0.5,
) -> tuple[float, float, float]:
    """
    Converts fixed uncertainty u and projected probability P_OK into
    the corresponding opinion components (b, d, u).
    """
    b = p_ok - a_prior * u
    d = 1.0 - u - b

    b = float(np.clip(b, 0.0, 1.0))
    d = float(np.clip(d, 0.0, 1.0))
    u = float(np.clip(u, 0.0, 1.0))

    return b, d, u


def ternary_arrays_from_opinions(
    opinions: list[tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float]]:
    """
    Plotly ternary convention used in this app:

        a-axis = uncertainty u
        b-axis = disbelief d
        c-axis = belief b
    """
    a_values = [op[2] for op in opinions]
    b_values = [op[1] for op in opinions]
    c_values = [op[0] for op in opinions]

    return a_values, b_values, c_values


# ============================================================
# 3) Decision boundary curves
# ============================================================

def boundary_p_ok_for_u(
    u: float,
    eta: float = 0.95,
    tau_ok: float = 0.5,
    a_prior: float = 0.5,
    side: str = "consistent",
) -> float | str | None:
    """
    For fixed uncertainty u, finds the boundary projected probability
    P_OK for one of the two beta-credible decision regions.

    For side="consistent", the boundary is the minimal P_OK such that

        Pr(theta > tau_OK | omega) = eta.

    For side="inconsistent", the boundary is the maximal P_OK such that

        Pr(theta < tau_OK | omega) = eta.

    Returns
    -------
    float
        Boundary projected probability P_OK.
    "all"
        The complete u-slice already satisfies the requested decision.
    None
        No feasible opinion at this u satisfies the requested decision.
    """
    eps = 1e-10
    u = float(u)

    if u <= eps:
        return tau_ok

    p_low, p_high = p_ok_feasible_interval(
        u=u,
        a_prior=a_prior,
    )

    def eval_probability(p_ok: float) -> float:
        b, d, u_local = opinion_from_u_and_p_ok(
            u=u,
            p_ok=p_ok,
            a_prior=a_prior,
        )

        if side == "consistent":
            return q_consistent(
                b=b,
                d=d,
                u=u_local,
                a_prior=a_prior,
                tau_ok=tau_ok,
            )

        if side == "inconsistent":
            return q_inconsistent(
                b=b,
                d=d,
                u=u_local,
                a_prior=a_prior,
                tau_ok=tau_ok,
            )

        raise ValueError(f"Unknown side: {side}")

    f_low = eval_probability(p_low + eps) - eta
    f_high = eval_probability(p_high - eps) - eta

    if side == "consistent":
        # q_consistent increases with P_OK.
        if f_high < 0:
            return None
        if f_low >= 0:
            return "all"
        return brentq(
            lambda p: eval_probability(p) - eta,
            p_low + eps,
            p_high - eps,
        )

    # q_inconsistent decreases with P_OK.
    if f_low < 0:
        return None
    if f_high >= 0:
        return "all"
    return brentq(
        lambda p: eval_probability(p) - eta,
        p_low + eps,
        p_high - eps,
    )


def compute_decision_geometry(
    eta: float,
    tau_ok: float,
    a_prior: float,
    n_boundary: int = 220,
) -> dict[str, object]:
    """
    Computes ternary boundary curves and filled polygons for
    consistent, inconsistent, and undecided decision regions.
    """
    eps_area = 1e-8

    consistent_boundary: list[tuple[float, float, float]] = []
    inconsistent_boundary: list[tuple[float, float, float]] = []

    undecided_lower: list[tuple[float, float, float]] = []
    undecided_upper: list[tuple[float, float, float]] = []

    u_grid = np.linspace(0.0, 1.0, n_boundary)

    for u in u_grid:
        p_low, p_high = p_ok_feasible_interval(
            u=u,
            a_prior=a_prior,
        )

        # -------------------------------
        # Consistent boundary
        # -------------------------------
        p_cons_result = boundary_p_ok_for_u(
            u=u,
            eta=eta,
            tau_ok=tau_ok,
            a_prior=a_prior,
            side="consistent",
        )

        if p_cons_result is None:
            p_cons_for_undecided = p_high
        elif p_cons_result == "all":
            p_cons_for_undecided = p_low
            consistent_boundary.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_low,
                    a_prior=a_prior,
                )
            )
        else:
            p_cons_for_undecided = float(p_cons_result)
            consistent_boundary.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_cons_for_undecided,
                    a_prior=a_prior,
                )
            )

        # -------------------------------
        # Inconsistent boundary
        # -------------------------------
        p_inc_result = boundary_p_ok_for_u(
            u=u,
            eta=eta,
            tau_ok=tau_ok,
            a_prior=a_prior,
            side="inconsistent",
        )

        if p_inc_result is None:
            p_inc_for_undecided = p_low
        elif p_inc_result == "all":
            p_inc_for_undecided = p_high
            inconsistent_boundary.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_high,
                    a_prior=a_prior,
                )
            )
        else:
            p_inc_for_undecided = float(p_inc_result)
            inconsistent_boundary.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_inc_for_undecided,
                    a_prior=a_prior,
                )
            )

        # -------------------------------
        # Undecided interval for this u
        # -------------------------------
        if p_inc_for_undecided + eps_area < p_cons_for_undecided:
            undecided_lower.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_inc_for_undecided,
                    a_prior=a_prior,
                )
            )
            undecided_upper.append(
                opinion_from_u_and_p_ok(
                    u=u,
                    p_ok=p_cons_for_undecided,
                    a_prior=a_prior,
                )
            )

    # Polygon for the consistent region:
    # decision boundary + edge d = 0.
    consistent_polygon: list[tuple[float, float, float]] = []
    if consistent_boundary:
        consistent_polygon.extend(consistent_boundary)
        for b, d, u in reversed(consistent_boundary):
            consistent_polygon.append((1.0 - u, 0.0, u))

    # Polygon for the inconsistent region:
    # decision boundary + edge b = 0.
    inconsistent_polygon: list[tuple[float, float, float]] = []
    if inconsistent_boundary:
        inconsistent_polygon.extend(inconsistent_boundary)
        for b, d, u in reversed(inconsistent_boundary):
            inconsistent_polygon.append((0.0, 1.0 - u, u))

    # Polygon for the undecided region:
    # lower boundary + reversed upper boundary.
    undecided_polygon: list[tuple[float, float, float]] = []
    if undecided_lower and undecided_upper:
        undecided_polygon.extend(undecided_lower)
        undecided_polygon.extend(reversed(undecided_upper))

    return {
        "consistent_boundary": consistent_boundary,
        "inconsistent_boundary": inconsistent_boundary,
        "consistent_polygon": consistent_polygon,
        "inconsistent_polygon": inconsistent_polygon,
        "undecided_polygon": undecided_polygon,
        "n_consistent_boundary": len(consistent_boundary),
        "n_inconsistent_boundary": len(inconsistent_boundary),
        "n_undecided_polygon": len(undecided_polygon),
    }


def max_uncertainty_for_side(
    eta: float,
    tau_ok: float,
    a_prior: float,
    side: str,
) -> float | None:
    """
    Finds the largest uncertainty u for which at least one opinion in
    the u-slice satisfies the selected credible decision condition.
    """
    eps = 1e-10

    def best_probability(u: float) -> float:
        if side == "consistent":
            b = 1.0 - u
            d = 0.0
            return q_consistent(
                b=b,
                d=d,
                u=u,
                a_prior=a_prior,
                tau_ok=tau_ok,
            )

        if side == "inconsistent":
            b = 0.0
            d = 1.0 - u
            return q_inconsistent(
                b=b,
                d=d,
                u=u,
                a_prior=a_prior,
                tau_ok=tau_ok,
            )

        raise ValueError(f"Unknown side: {side}")

    f_low = best_probability(eps) - eta
    f_high = best_probability(1.0 - eps) - eta

    if f_high >= 0:
        return 1.0

    if f_low < 0:
        return None

    return brentq(
        lambda u: best_probability(u) - eta,
        eps,
        1.0 - eps,
    )


# ============================================================
# 4) Example opinion validity and decision state
# ============================================================

def compute_example_opinion(
    b_ex: float,
    u_ex: float,
    eps: float = 1e-12,
) -> tuple[float, float, float, bool, str]:
    """
    Computes

        d_X = 1 - b_X - u_X.

    Validity condition:

        b_X >= 0,
        u_X >= 0,
        d_X >= 0.
    """
    b_ex = float(b_ex)
    u_ex = float(u_ex)
    d_ex = 1.0 - b_ex - u_ex

    is_valid = (
        b_ex >= -eps
        and u_ex >= -eps
        and d_ex >= -eps
    )

    if not is_valid:
        message = (
            "Invalid example opinion:\n"
            f"b_X + u_X = {b_ex + u_ex:.3f} > 1.000\n"
            f"d_X = 1 - b_X - u_X = {d_ex:.3f} < 0"
        )

        return b_ex, d_ex, u_ex, False, message

    b_ex = float(np.clip(b_ex, 0.0, 1.0))
    u_ex = float(np.clip(u_ex, 0.0, 1.0))
    d_ex = float(np.clip(d_ex, 0.0, 1.0))

    message = (
        "Valid example opinion:\n"
        f"d_X = 1 - b_X - u_X = {d_ex:.3f}"
    )

    return b_ex, d_ex, u_ex, True, message


def classify_opinion(
    b: float,
    d: float,
    u: float,
    eta: float,
    tau_ok: float,
    a_prior: float,
) -> tuple[str, float, float]:
    """
    Classifies a valid opinion into one of the displayed decision regions.
    """
    q_ok = q_consistent(
        b=b,
        d=d,
        u=u,
        a_prior=a_prior,
        tau_ok=tau_ok,
    )
    q_not_ok = q_inconsistent(
        b=b,
        d=d,
        u=u,
        a_prior=a_prior,
        tau_ok=tau_ok,
    )

    if q_ok >= eta:
        return "confidently consistent", q_ok, q_not_ok

    if q_not_ok >= eta:
        return "confidently inconsistent", q_ok, q_not_ok

    return "undecided", q_ok, q_not_ok


# ============================================================
# 5) Figure construction
# ============================================================

def add_filled_region(
    fig: go.Figure,
    polygon: list[tuple[float, float, float]],
    fillcolor: str,
    linecolor: str,
    name: str,
    visible: bool,
) -> None:
    """
    Adds a filled ternary polygon if requested and if it has enough points.
    """
    if not visible or len(polygon) < 3:
        return

    a_values, b_values, c_values = ternary_arrays_from_opinions(polygon)

    fig.add_trace(
        go.Scatterternary(
            a=a_values,
            b=b_values,
            c=c_values,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=linecolor, width=1),
            name=name,
            hoverinfo="skip",
        )
    )


def add_boundary_curve(
    fig: go.Figure,
    boundary: list[tuple[float, float, float]],
    linecolor: str,
    dash: str,
    name: str,
    criterion: str,
    a_prior: float,
    visible: bool,
) -> None:
    """
    Adds a ternary decision boundary curve.
    """
    if not visible or len(boundary) < 2:
        return

    a_values, b_values, c_values = ternary_arrays_from_opinions(boundary)

    custom_boundary = np.array(
        [
            [
                b,
                d,
                u,
                b + a_prior * u,
            ]
            for b, d, u in boundary
        ]
    )

    fig.add_trace(
        go.Scatterternary(
            a=a_values,
            b=b_values,
            c=c_values,
            mode="lines",
            line=dict(color=linecolor, width=3, dash=dash),
            customdata=custom_boundary,
            hovertemplate=(
                "b = %{customdata[0]:.3f}<br>"
                "d = %{customdata[1]:.3f}<br>"
                "u = %{customdata[2]:.3f}<br>"
                "P<sub>OK</sub> = %{customdata[3]:.3f}<br>"
                f"{criterion}"
                "<extra></extra>"
            ),
            name=name,
        )
    )


def make_figure(
    eta: float,
    tau_ok: float,
    a_prior: float,
    b_ex_raw: float,
    u_ex_raw: float,
    visible_regions: list[str],
    n_boundary: int = 220,
) -> tuple[go.Figure, str]:
    """
    Builds the ternary Plotly figure.
    """
    show_consistent = "consistent" in visible_regions
    show_inconsistent = "inconsistent" in visible_regions
    show_undecided = "undecided" in visible_regions

    geometry = compute_decision_geometry(
        eta=eta,
        tau_ok=tau_ok,
        a_prior=a_prior,
        n_boundary=n_boundary,
    )

    u_max_ok = max_uncertainty_for_side(
        eta=eta,
        tau_ok=tau_ok,
        a_prior=a_prior,
        side="consistent",
    )
    u_max_not_ok = max_uncertainty_for_side(
        eta=eta,
        tau_ok=tau_ok,
        a_prior=a_prior,
        side="inconsistent",
    )

    fig = go.Figure()

    b_ex, d_ex, u_ex, opinion_valid, opinion_message = compute_example_opinion(
        b_ex=b_ex_raw,
        u_ex=u_ex_raw,
    )

    # --------------------------------------------------------
    # Filled regions: draw undecided first so decision regions
    # are visually on top of it.
    # --------------------------------------------------------

    add_filled_region(
        fig=fig,
        polygon=geometry["undecided_polygon"],
        fillcolor="rgba(120,120,120,0.16)",
        linecolor="rgba(120,120,120,0.35)",
        name="undecided region",
        visible=show_undecided,
    )

    add_filled_region(
        fig=fig,
        polygon=geometry["consistent_polygon"],
        fillcolor="rgba(0,150,0,0.18)",
        linecolor="rgba(0,100,0,0.55)",
        name="confidently consistent region",
        visible=show_consistent,
    )

    add_filled_region(
        fig=fig,
        polygon=geometry["inconsistent_polygon"],
        fillcolor="rgba(200,0,0,0.16)",
        linecolor="rgba(140,0,0,0.55)",
        name="confidently inconsistent region",
        visible=show_inconsistent,
    )

    # --------------------------------------------------------
    # Decision boundaries
    # --------------------------------------------------------

    add_boundary_curve(
        fig=fig,
        boundary=geometry["consistent_boundary"],
        linecolor="darkgreen",
        dash="solid",
        name="consistent boundary",
        criterion="Pr(θ &gt; τ<sub>OK</sub> | ω) = η",
        a_prior=a_prior,
        visible=show_consistent,
    )

    add_boundary_curve(
        fig=fig,
        boundary=geometry["inconsistent_boundary"],
        linecolor="darkred",
        dash="solid",
        name="inconsistent boundary",
        criterion="Pr(θ &lt; τ<sub>OK</sub> | ω) = η",
        a_prior=a_prior,
        visible=show_inconsistent,
    )

    # --------------------------------------------------------
    # Example opinion point, only if valid
    # --------------------------------------------------------

    if opinion_valid:
        p_ok_ex = b_ex + a_prior * u_ex

        decision_state, q_ok_ex, q_not_ok_ex = classify_opinion(
            b=b_ex,
            d=d_ex,
            u=u_ex,
            eta=eta,
            tau_ok=tau_ok,
            a_prior=a_prior,
        )

        if decision_state == "confidently consistent":
            marker_color = "darkgreen"
        elif decision_state == "confidently inconsistent":
            marker_color = "darkred"
        else:
            marker_color = "black"

        fig.add_trace(
            go.Scatterternary(
                a=[u_ex],
                b=[d_ex],
                c=[b_ex],
                mode="markers+text",
                marker=dict(
                    size=11,
                    color=marker_color,
                    symbol="circle",
                ),
                text=[r"$\omega_X$"],
                textposition="top center",
                customdata=np.array(
                    [[b_ex, d_ex, u_ex, p_ok_ex, q_ok_ex, q_not_ok_ex, decision_state]]
                ),
                hovertemplate=(
                    "b<sub>X</sub> = %{customdata[0]:.3f}<br>"
                    "d<sub>X</sub> = %{customdata[1]:.3f}<br>"
                    "u<sub>X</sub> = %{customdata[2]:.3f}<br>"
                    "P<sub>OK,X</sub> = %{customdata[3]:.3f}<br>"
                    "Pr(θ &gt; τ<sub>OK</sub> | ω<sub>X</sub>) = %{customdata[4]:.3f}<br>"
                    "Pr(θ &lt; τ<sub>OK</sub> | ω<sub>X</sub>) = %{customdata[5]:.3f}<br>"
                    "Decision: %{customdata[6]}"
                    "<extra></extra>"
                ),
                name=f"omega_X: {decision_state}",
            )
        )

        example_status = (
            f"{opinion_message}\n"
            f"omega_X = (b_X={b_ex:.3f}, d_X={d_ex:.3f}, u_X={u_ex:.3f})\n"
            f"P_OK,X = b_X + a_prior * u_X = {p_ok_ex:.3f}\n"
            f"Pr(theta > tau_OK | omega_X) = {q_ok_ex:.3f}\n"
            f"Pr(theta < tau_OK | omega_X) = {q_not_ok_ex:.3f}\n"
            f"Decision state: {decision_state}"
        )

    else:
        example_status = (
            f"{opinion_message}\n"
            "The example point is not drawn."
        )

    title = (
        r"$\text{Beta-credible decision regions for binomial opinions}"
        rf"\quad "
        rf"\left("
        rf"\eta={eta:.3f},\ "
        rf"\tau_{{\mathrm{{OK}}}}={tau_ok:.3f},\ "
        rf"a={a_prior:.3f}"
        rf"\right)$"
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=14),
        showlegend=True,

        # Robust legend placement below the ternary plot
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.12,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            font=dict(size=12),
        ),

        margin=dict(l=40, r=40, t=110, b=150),

        ternary=dict(
            sum=1,
            bgcolor="white",
            aaxis=dict(
                title=dict(text=r"$u$"),
                min=0.0,
                linewidth=2,
                linecolor="black",
                gridcolor="rgba(0,0,0,1)",
                showgrid=True,
                showline=True,
                showticklabels=False,
                ticks="",
            ),
            baxis=dict(
                title=dict(text=r"$d$"),
                min=0.0,
                linewidth=2,
                linecolor="black",
                gridcolor="rgba(0,0,0,1)",
                showgrid=True,
                showline=True,
                showticklabels=False,
                ticks="",
            ),
            caxis=dict(
                title=dict(text=r"$b$"),
                min=0.0,
                linewidth=2,
                linecolor="black",
                gridcolor="rgba(0,0,0,1)",
                showgrid=True,
                showline=True,
                showticklabels=False,
                ticks="",
            ),
        ),
    )

    u_max_ok_text = "None" if u_max_ok is None else f"{u_max_ok:.6f}"
    u_max_not_ok_text = "None" if u_max_not_ok is None else f"{u_max_not_ok:.6f}"

    status = (
        f"eta = {eta:.3f}, tau_OK = {tau_ok:.3f}, a_prior = {a_prior:.3f}\n"
        f"u_max for confidently consistent decision = {u_max_ok_text}\n"
        f"u_max for confidently inconsistent decision = {u_max_not_ok_text}\n"
        f"consistent boundary points = {geometry['n_consistent_boundary']}\n"
        f"inconsistent boundary points = {geometry['n_inconsistent_boundary']}\n"
        f"undecided polygon points = {geometry['n_undecided_polygon']}\n"
        "\n"
        f"{example_status}"
    )

    return fig, status


# ============================================================
# 6) Dash app
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    style={
        "maxWidth": "1150px",
        "margin": "0 auto",
        "fontFamily": "Arial, sans-serif",
        "padding": "24px",
    },
    children=[
        dcc.Markdown(
            r"""
## Interactive Opinion Triangle

The ternary plot shows beta-credible decision regions for a binomial
subjective-logic opinion

$$
\omega=(b,d,u,a),
\qquad b+d+u=1.
$$

The consistent decision region is

$$
\mathcal{C}_{\eta,\tau_{\mathrm{OK}},a}
=
\left\{
\omega
\;\middle|\;
\Pr\!\left(\theta>\tau_{\mathrm{OK}}\mid\omega\right)
\geq
\eta
\right\},
$$

and the inconsistent decision region is

$$
\mathcal{I}_{\eta,\tau_{\mathrm{OK}},a}
=
\left\{
\omega
\;\middle|\;
\Pr\!\left(\theta<\tau_{\mathrm{OK}}\mid\omega\right)
\geq
\eta
\right\}.
$$

The remaining part of the triangle is the undecided region. The ternary axes are

$$
a_{\mathrm{axis}} = u,\qquad
b_{\mathrm{axis}} = d,\qquad
c_{\mathrm{axis}} = b.
$$
            """,
            mathjax=True,
        ),

        # ----------------------------------------------------
        # Region sliders and display options ABOVE the triangle
        # ----------------------------------------------------

        html.Div(
            style={
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "padding": "16px",
                "marginBottom": "16px",
                "background": "#fafafa",
            },
            children=[
                html.H4("Region parameters"),

                html.Label("η"),
                dcc.Slider(
                    id="eta-slider",
                    min=0.50,
                    max=0.999,
                    step=0.001,
                    value=DEFAULTS["eta"],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={
                        0.50: "0.50",
                        0.75: "0.75",
                        0.95: "0.95",
                        0.999: "0.999",
                    },
                ),

                html.Br(),

                html.Label(
                    children=[
                        "τ",
                        html.Sub("OK"),
                    ]
                ),
                dcc.Slider(
                    id="tau-ok-slider",
                    min=0.001,
                    max=0.999,
                    step=0.001,
                    value=DEFAULTS["tau_ok"],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={
                        0.001: "0.001",
                        0.25: "0.25",
                        0.50: "0.50",
                        0.75: "0.75",
                        0.999: "0.999",
                    },
                ),

                html.Br(),

                html.Label(
                    children=[
                        "a",
                        html.Sub("prior"),
                    ]
                ),
                dcc.Slider(
                    id="a-prior-slider",
                    min=0.001,
                    max=0.999,
                    step=0.001,
                    value=DEFAULTS["a_prior"],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={
                        0.001: "0.001",
                        0.25: "0.25",
                        0.50: "0.50",
                        0.75: "0.75",
                        0.999: "0.999",
                    },
                ),

                html.Br(),

                html.H4("Displayed regions"),
                dcc.Checklist(
                    id="region-toggle",
                    options=[
                        {
                            "label": "Show confidently consistent region",
                            "value": "consistent",
                        },
                        {
                            "label": "Show confidently inconsistent region",
                            "value": "inconsistent",
                        },
                        {
                            "label": "Show undecided region",
                            "value": "undecided",
                        },
                    ],
                    value=DEFAULTS["visible_regions"],
                    inputStyle={"marginRight": "8px"},
                    labelStyle={"display": "block", "marginBottom": "6px"},
                ),

                html.Button(
                    "Reset parameters",
                    id="reset-button",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "padding": "8px 14px",
                        "borderRadius": "6px",
                        "border": "1px solid #999",
                        "background": "white",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        # ----------------------------------------------------
        # Triangle
        # ----------------------------------------------------

        dcc.Graph(
            id="opinion-triangle",
            mathjax=True,
            style={"height": "860px"},
            config={
                "responsive": True,
                "displaylogo": False,
            },
        ),

        # ----------------------------------------------------
        # Example opinion sliders BELOW the triangle
        # ----------------------------------------------------

        html.Div(
            style={
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "padding": "16px",
                "marginTop": "16px",
                "marginBottom": "16px",
                "background": "#fafafa",
            },
            children=[
                html.H4("Example opinion ω_X"),

                dcc.Markdown(
                    r"""
The example opinion is defined by

$$
\omega_X = (b_X,d_X,u_X,a),
\qquad
 d_X = 1 - b_X - u_X.
$$

Validity condition:

$$
b_X \geq 0,
\qquad
u_X \geq 0,
\qquad
b_X+u_X \leq 1.
$$
                    """,
                    mathjax=True,
                ),

                html.Label(
                    children=[
                        "b",
                        html.Sub("X"),
                    ]
                ),
                dcc.Slider(
                    id="b-ex-slider",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    value=DEFAULTS["b_ex"],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={
                        0.0: "0.0",
                        0.25: "0.25",
                        0.50: "0.50",
                        0.75: "0.75",
                        1.0: "1.0",
                    },
                ),

                html.Br(),

                html.Label(
                    children=[
                        "u",
                        html.Sub("X"),
                    ]
                ),
                dcc.Slider(
                    id="u-ex-slider",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    value=DEFAULTS["u_ex"],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={
                        0.0: "0.0",
                        0.25: "0.25",
                        0.50: "0.50",
                        0.75: "0.75",
                        1.0: "1.0",
                    },
                ),

                html.Pre(
                    id="status-box",
                    style={
                        "fontFamily": "monospace",
                        "marginTop": "18px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ],
        ),
    ],
)


# ============================================================
# 7) Callback
# ============================================================

@app.callback(
    Output("opinion-triangle", "figure"),
    Output("status-box", "children"),

    Output("eta-slider", "value"),
    Output("tau-ok-slider", "value"),
    Output("a-prior-slider", "value"),
    Output("b-ex-slider", "value"),
    Output("u-ex-slider", "value"),

    Input("eta-slider", "value"),
    Input("tau-ok-slider", "value"),
    Input("a-prior-slider", "value"),
    Input("b-ex-slider", "value"),
    Input("u-ex-slider", "value"),
    Input("region-toggle", "value"),
    Input("reset-button", "n_clicks"),
)
def update_opinion_triangle(
    eta: float,
    tau_ok: float,
    a_prior: float,
    b_ex: float,
    u_ex: float,
    visible_regions: list[str],
    reset_clicks: int,
):
    triggered = ctx.triggered_id

    if triggered == "reset-button":
        eta = DEFAULTS["eta"]
        tau_ok = DEFAULTS["tau_ok"]
        a_prior = DEFAULTS["a_prior"]
        b_ex = DEFAULTS["b_ex"]
        u_ex = DEFAULTS["u_ex"]

        fig, status = make_figure(
            eta=eta,
            tau_ok=tau_ok,
            a_prior=a_prior,
            b_ex_raw=b_ex,
            u_ex_raw=u_ex,
            visible_regions=visible_regions,
        )

        return (
            fig,
            status,
            eta,
            tau_ok,
            a_prior,
            b_ex,
            u_ex,
        )

    fig, status = make_figure(
        eta=eta,
        tau_ok=tau_ok,
        a_prior=a_prior,
        b_ex_raw=b_ex,
        u_ex_raw=u_ex,
        visible_regions=visible_regions,
    )

    return (
        fig,
        status,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    )


# ============================================================
# 8) App start
# ============================================================

def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:8050/")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()

    app.run(
        host="127.0.0.1",
        port=8050,
        debug=False,
        use_reloader=False,
    )