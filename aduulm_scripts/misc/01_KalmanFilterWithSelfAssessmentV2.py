import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta
import subjective_logic as sl

# -------------------------------
# STONESOUP SETUP
# -------------------------------
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

np.random.seed(0)

start_time = datetime.now()

transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.05),
    ConstantVelocity(0.05)
])

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[5, 0], [0, 5]])
)

# Ground truth erzeugen
truth = GroundTruthPath([
    GroundTruthState([0, 1, 0, 1], timestamp=start_time)
])

timesteps = [start_time]
for k in range(1, 20):
    t = start_time + timedelta(seconds=k)
    timesteps.append(t)
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True,
                                  time_interval=timedelta(seconds=1)),
        timestamp=t))

# Measurements
measurements = []
for state in truth:
    meas = measurement_model.function(state, noise=True)
    measurements.append(Detection(meas, timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# Filter
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

prior = GaussianState([[0], [1], [0], [1]],
                      np.diag([1.5, 0.5, 1.5, 0.5]),
                      timestamp=start_time)

track = Track()

# -------------------------------
# SUBJECTIVE LOGIC + DIRICHLET
# -------------------------------
alphas = np.array([1.0, 1.0])
evidence_per_step = 1.0
m = 2

opinions = []
dirichlet_pdfs = []
x_pdf = np.linspace(0.001, 0.999, 500)

track_x = []
track_y = []

for i, measurement in enumerate(measurements):

    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)

    track.append(post)
    prior = post

    # Track speichern
    track_x.append(post.state_vector[0, 0])
    track_y.append(post.state_vector[2, 0])

    # ---------------------------
    # Evidence aus Mahalanobis
    # ---------------------------
    meas_pred = hypothesis.measurement_prediction
    delta = (measurement.state_vector - meas_pred.state_vector).reshape(-1, 1)
    S = meas_pred.covar

    d2 = (delta.T @ np.linalg.inv(S) @ delta).item()

    score = np.exp(-0.5 * (d2 - m))

    e_alpha = evidence_per_step * score
    e_beta = evidence_per_step * (1 / score)

    alphas += np.array([e_alpha, e_beta])

    # ---------------------------
    # Dirichlet + Opinion
    # ---------------------------
    dist = sl.DirichletDistribution2d(alphas)

    pdf = list(map(dist.evaluate, x_pdf))
    dirichlet_pdfs.append(pdf)

    op = dist.as_opinion()
    opinions.append((op.belief(), op.disbelief(), op.uncertainty()))

# -------------------------------
# PLOTLY FIGURE
# -------------------------------
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"colspan": 2}, None],
        [{"type": "ternary"}, {"type": "xy"}]
    ],
    subplot_titles=("Tracking", "Opinion Triangle", "Beta Distribution"),
    vertical_spacing=0.12
)

# -------------------------------
# INITIAL TRACES
# -------------------------------

# Track
fig.add_trace(
    go.Scatter(
        x=[track_x[0]],
        y=[track_y[0]],
        mode='lines+markers',
        line=dict(color='black'),
        name='Track'
    ),
    row=1, col=1
)

# Triangle
b0, d0, u0 = opinions[0]
fig.add_trace(
    go.Scatterternary(
        a=[u0], b=[d0], c=[b0],
        mode='markers',
        marker=dict(size=14, color='red'),
        hovertemplate="b: %{c:.2f}<br>d: %{b:.2f}<br>u: %{a:.2f}<extra></extra>"
    ),
    row=2, col=1
)

# Beta
fig.add_trace(
    go.Scatter(
        x=x_pdf,
        y=dirichlet_pdfs[0],
        mode='lines',
        line=dict(color='blue')
    ),
    row=2, col=2
)

# -------------------------------
# FRAMES (ALLES SYNCHRON!)
# -------------------------------
frames = []

for i in range(len(opinions)):
    b, d, u = opinions[i]

    frame = go.Frame(
        data=[
            # Track
            go.Scatter(
                x=track_x[:i+1],
                y=track_y[:i+1]
            ),

            # Triangle
            go.Scatterternary(
                a=[u], b=[d], c=[b]
            ),

            # Beta
            go.Scatter(
                x=x_pdf,
                y=dirichlet_pdfs[i]
            )
        ],
        name=str(i)
    )
    frames.append(frame)

fig.frames = frames

# -------------------------------
# SLIDER
# -------------------------------
sliders = [dict(
    steps=[dict(method='animate',
                args=[[str(k)], dict(frame=dict(duration=800, redraw=True),
                                     transition=dict(duration=0),
                                     mode='immediate')],
                label=str(k)) for k in range(len(frames))]
)]

# -------------------------------
# LAYOUT
# -------------------------------
fig.update_layout(
    template="plotly_white",
    height=900,

    sliders=sliders,

    updatemenus=[dict(
        type='buttons',
        buttons=[
            dict(label='Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=800, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True,
                                  mode='immediate')]),
            dict(label='Pause',
                 method='animate',
                 args=[[None],
                       dict(frame=dict(duration=0),
                            mode='immediate')])
        ]
    )],

    ternary=dict(
        sum=1,
        aaxis=dict(title='uncertainty', showgrid=True, gridcolor='black',
                   ticks='', showticklabels=False),
        baxis=dict(title='disbelief', showgrid=True, gridcolor='black',
                   ticks='', showticklabels=False),
        caxis=dict(title='belief', showgrid=True, gridcolor='black',
                   ticks='', showticklabels=False),
    )
)

fig.update_xaxes(showgrid=True, gridcolor='black', row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor='black', row=1, col=1)

fig.update_xaxes(title_text="x", row=2, col=2)
fig.update_yaxes(title_text="Density", row=2, col=2)

# -------------------------------
# SHOW
# -------------------------------
fig.show(renderer="browser")