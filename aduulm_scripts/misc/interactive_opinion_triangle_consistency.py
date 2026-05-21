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
}


# ============================================================
# 1) Opinion -> probability that latent p_OK exceeds tau_OK
# ============================================================

def q_consistent(
    b: float,
    d: float,
    u: float,
    a_prior: float = 0.5,
    tau_ok: float = 0.5,
) -> float:
    """
    Computes

        q(omega) = Pr(p_OK > tau_OK | omega)

    with the beta density induced by a binomial subjective-logic
    opinion omega = (b, d, u, a).

    Parameters
    ----------
    b : float
        Belief mass.
    d : float
        Disbelief mass.
    u : float
        Uncertainty mass.
    a_prior : float
        Base rate a.
    tau_ok : float
        Threshold tau_OK.

    Returns
    -------
    float
        Probability Pr(p_OK > tau_OK | omega).
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

    alpha = 2.0 * b / u + 2.0 * a_prior
    beta_param = 2.0 * d / u + 2.0 * (1.0 - a_prior)

    return 1.0 - beta.cdf(tau_ok, alpha, beta_param)


# ============================================================
# 2) Maximal uncertainty u_max
# ============================================================

def find_u_max(
    eta: float = 0.95,
    tau_ok: float = 0.5,
    a_prior: float = 0.5,
) -> float | None:
    """
    Finds the largest uncertainty u such that there still exists
    an opinion satisfying

        Pr(p_OK > tau_OK | omega) >= eta.

    The most favorable opinion for fixed uncertainty u is

        b = 1 - u, d = 0.
    """
    eps = 1e-10

    def f(u: float) -> float:
        b = 1.0 - u
        d = 0.0

        return q_consistent(
            b=b,
            d=d,
            u=u,
            a_prior=a_prior,
            tau_ok=tau_ok,
        ) - eta

    f_low = f(eps)
    f_high = f(1.0 - eps)

    # Even at maximal uncertainty, the condition is fulfilled.
    if f_high >= 0:
        return 1.0

    # Even the nearly dogmatic maximal-belief opinion is insufficient.
    if f_low < 0:
        return None

    return brentq(f, eps, 1.0 - eps)


# ============================================================
# 3) Boundary curve q(omega) = eta
# ============================================================

def p_ok_boundary_for_u(
    u: float,
    eta: float = 0.95,
    tau_ok: float = 0.5,
    a_prior: float = 0.5,
) -> float | str | None:
    """
    For fixed uncertainty u, finds the minimal projected probability

        P_OK = b + a u

    such that

        Pr(p_OK > tau_OK | omega) = eta.

    Returns
    -------
    float
        Boundary projected probability P_OK.
    "all"
        The complete u-slice is already consistent.
    None
        No feasible opinion at this u satisfies the condition.
    """
    eps = 1e-10
    u = float(u)

    if u <= eps:
        return tau_ok

    def f(p_ok: float) -> float:
        b = p_ok - a_prior * u
        d = 1.0 - u - b

        b = np.clip(b, 0.0, 1.0)
        d = np.clip(d, 0.0, 1.0)

        return q_consistent(
            b=b,
            d=d,
            u=u,
            a_prior=a_prior,
            tau_ok=tau_ok,
        ) - eta

    # Feasible projected-probability interval for fixed u
    p_low = a_prior * u
    p_high = 1.0 - (1.0 - a_prior) * u

    f_low = f(p_low + eps)
    f_high = f(p_high - eps)

    if f_high < 0:
        return None

    if f_low >= 0:
        return "all"

    return brentq(f, p_low + eps, p_high - eps)


# ============================================================
# 4) Example opinion validity
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

    b_ex = np.clip(b_ex, 0.0, 1.0)
    u_ex = np.clip(u_ex, 0.0, 1.0)
    d_ex = np.clip(d_ex, 0.0, 1.0)

    message = (
        "Valid example opinion:\n"
        f"d_X = 1 - b_X - u_X = {d_ex:.3f}"
    )

    return b_ex, d_ex, u_ex, True, message


# ============================================================
# 5) Figure construction
# ============================================================

def make_figure(
    eta: float,
    tau_ok: float,
    a_prior: float,
    b_ex_raw: float,
    u_ex_raw: float,
    n_boundary: int = 180,
) -> tuple[go.Figure, str]:
    """
    Builds the ternary Plotly figure.
    """
    u_max = find_u_max(
        eta=eta,
        tau_ok=tau_ok,
        a_prior=a_prior,
    )

    fig = go.Figure()

    b_ex, d_ex, u_ex, opinion_valid, opinion_message = compute_example_opinion(
        b_ex=b_ex_raw,
        u_ex=u_ex_raw,
    )

    if u_max is None:
        title = (
            r"$\text{No consistent region exists for }"
            rf"\eta={eta:.3f},\ "
            rf"\tau_{{\mathrm{{OK}}}}={tau_ok:.3f},\ "
            rf"a={a_prior:.3f}$"
        )

        status = (
            "u_max = None\n"
            "No consistent region exists.\n\n"
            f"{opinion_message}"
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            template="plotly_white",
            showlegend=True,
            margin=dict(l=40, r=40, t=110, b=150),
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
        )

        return fig, status

    u_boundary = []
    d_boundary = []
    b_boundary = []

    for u in np.linspace(0.0, u_max, n_boundary):
        result = p_ok_boundary_for_u(
            u=u,
            eta=eta,
            tau_ok=tau_ok,
            a_prior=a_prior,
        )

        if result is None:
            continue

        if result == "all":
            # Boundary is effectively the left edge b = 0.
            b = 0.0
            d = 1.0 - u
        else:
            p_boundary = result
            b = p_boundary - a_prior * u
            d = 1.0 - u - b

        u_boundary.append(u)
        d_boundary.append(d)
        b_boundary.append(b)

    u_boundary = np.asarray(u_boundary)
    d_boundary = np.asarray(d_boundary)
    b_boundary = np.asarray(b_boundary)

    # Polygon for the consistent region:
    # boundary curve + right edge d = 0
    poly_a = list(u_boundary)
    poly_b = list(d_boundary)
    poly_c = list(b_boundary)

    edge_a = list(reversed(u_boundary))
    edge_b = [0.0] * len(edge_a)
    edge_c = [1.0 - u for u in edge_a]

    poly_a = poly_a + edge_a
    poly_b = poly_b + edge_b
    poly_c = poly_c + edge_c

    # Custom hover data:
    # [belief, disbelief, uncertainty, projected probability]
    custom_boundary = np.column_stack(
        [
            b_boundary,
            d_boundary,
            u_boundary,
            b_boundary + a_prior * u_boundary,
        ]
    )

    # --------------------------------------------------------
    # Filled consistent region
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatterternary(
            a=poly_a,
            b=poly_b,
            c=poly_c,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0,0,0,0.08)",
            line=dict(color="black", width=1),
            name="consistent region",
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # Boundary line
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatterternary(
            a=u_boundary,
            b=d_boundary,
            c=b_boundary,
            mode="lines",
            line=dict(color="black", width=3),
            customdata=custom_boundary,
            hovertemplate=(
                "b = %{customdata[0]:.3f}<br>"
                "d = %{customdata[1]:.3f}<br>"
                "u = %{customdata[2]:.3f}<br>"
                "P_OK = %{customdata[3]:.3f}"
                "<extra></extra>"
            ),
            name="boundary",
        )
    )

    # --------------------------------------------------------
    # Example opinion point, only if valid
    # --------------------------------------------------------

    if opinion_valid:
        p_ok_ex = b_ex + a_prior * u_ex

        q_ex = q_consistent(
            b=b_ex,
            d=d_ex,
            u=u_ex,
            a_prior=a_prior,
            tau_ok=tau_ok,
        )

        is_consistent = q_ex >= eta

        fig.add_trace(
            go.Scatterternary(
                a=[u_ex],
                b=[d_ex],
                c=[b_ex],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="black" if is_consistent else "red",
                    symbol="circle",
                ),
                text=[r"$\omega_X$"],
                textposition="top center",
                customdata=np.array([[b_ex, d_ex, u_ex, p_ok_ex, q_ex]]),
                hovertemplate=(
                    "b_X = %{customdata[0]:.3f}<br>"
                    "d_X = %{customdata[1]:.3f}<br>"
                    "u_X = %{customdata[2]:.3f}<br>"
                    "P_OK,X = %{customdata[3]:.3f}<br>"
                    "q_X = Pr(p_OK > tau_OK | omega_X) = %{customdata[4]:.3f}"
                    "<extra></extra>"
                ),
                name=rf"omega_X: q_X={q_ex:.3f}",
            )
        )

        example_status = (
            f"{opinion_message}\n"
            f"omega_X = (b_X={b_ex:.3f}, d_X={d_ex:.3f}, u_X={u_ex:.3f})\n"
            f"P_OK,X = b_X + a_prior * u_X = {p_ok_ex:.3f}\n"
            f"q_X = Pr(p_OK > tau_OK | omega_X) = {q_ex:.3f}\n"
            f"consistent: {is_consistent}"
        )

    else:
        example_status = (
            f"{opinion_message}\n"
            "The example point is not drawn."
        )

    title = (
        r"$\Pr\!\left(p_{\mathrm{OK}} > "
        rf"\tau_{{\mathrm{{OK}}}}\mid\omega\right) \geq \eta"
        rf"\quad "
        rf"\left("
        rf"\eta={eta:.3f},\ "
        rf"\tau_{{\mathrm{{OK}}}}={tau_ok:.3f},\ "
        rf"a={a_prior:.3f},\ "
        rf"u_{{\max}}={u_max:.3f}"
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

    status = (
        f"u_max = {u_max:.6f}\n"
        f"Boundary points = {len(u_boundary)}\n"
        f"eta = {eta:.3f}, tau_OK = {tau_ok:.3f}, a_prior = {a_prior:.3f}\n"
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
## Interaktives Opinion Triangle

Die dargestellte Menge ist

$$
\mathcal{C}_{\eta,\tau_{\mathrm{OK}},a}
=
\left\{
\omega=(b,d,u,a)
\;\middle|\;
\Pr\!\left(p_{\mathrm{OK}}>\tau_{\mathrm{OK}}\mid\omega\right)
\geq
\eta
\right\}.
$$

Die ternären Achsen sind

$$
a_{\mathrm{axis}} = u,\qquad
b_{\mathrm{axis}} = d,\qquad
c_{\mathrm{axis}} = b.
$$
            """,
            mathjax=True,
        ),

        # ----------------------------------------------------
        # Region sliders ABOVE the triangle
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
b_X \geq 0,\qquad u_X \geq 0,\qquad b_X+u_X \leq 1.
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
    Input("reset-button", "n_clicks"),
)
def update_opinion_triangle(
    eta: float,
    tau_ok: float,
    a_prior: float,
    b_ex: float,
    u_ex: float,
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
