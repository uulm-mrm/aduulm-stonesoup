# Transition-Focused Evaluation of the Subjective-Logic Parameter Sweep

## 1. Objective

The self-assessment output at time step $k$ is assigned to one of three decision regions:

$$
R_k \in \{\mathrm{C},\mathrm{U},\mathrm{I}\},
$$

where

- $\mathrm{C}$ denotes **confidently consistent**,
- $\mathrm{U}$ denotes **undecided**,
- $\mathrm{I}$ denotes **confidently inconsistent**.

The desired temporal behavior is:

1. stable nominal intervals should be classified as $\mathrm{C}$,
2. stable disturbed intervals should be classified as $\mathrm{I}$,
3. $\mathrm{U}$ should occur mainly during the transition from $\mathrm{C}$ to $\mathrm{I}$ or from $\mathrm{I}$ to $\mathrm{C}$,
4. confidently wrong decisions should remain rare.

The previous ranking emphasized false alarms and fault detection but did not sufficiently penalize persistent undecided states. The transition-focused evaluation therefore separates **steady-state intervals** from **transition windows** and explicitly evaluates where undecided samples occur.

---

## 2. Evaluation sets

Let $\mathcal{K}$ denote all evaluated time steps after the warm-up period. The simulation provides the nominal and disturbed sets

$$
\mathcal{N} = \{k \in \mathcal{K} \mid \text{no disturbance is active}\},
$$

$$
\mathcal{F} = \{k \in \mathcal{K} \mid \text{a disturbance is active}\}.
$$

For every known disturbance onset and end, a fixed transition window is introduced. Let $t_j$ be such a boundary. With $n_{\mathrm{pre}}$ samples before and $n_{\mathrm{post}}$ samples after the boundary, the corresponding transition window is

$$
\mathcal{T}_j
=
\{k \mid t_j-n_{\mathrm{pre}} \le k < t_j+n_{\mathrm{post}}\}.
$$

The complete transition set is

$$
\mathcal{T}=\bigcup_j \mathcal{T}_j.
$$

The steady-state set is then

$$
\mathcal{S}=\mathcal{K}\setminus\mathcal{T}.
$$

It is further divided into

$$
\mathcal{S}_{\mathrm{N}}=\mathcal{S}\cap\mathcal{N},
\qquad
\mathcal{S}_{\mathrm{F}}=\mathcal{S}\cap\mathcal{F}.
$$

The transition windows are kept **fixed for all parameter combinations**. Otherwise, a larger LTST buffer could be rewarded simply by receiving a larger admissible transition interval.

The default implementation uses

$$
n_{\mathrm{pre}}=5,
\qquad
n_{\mathrm{post}}=35.
$$

These values are evaluation-design parameters and should be reported together with the sweep results.

---

## 3. Steady-state region rates

### 3.1 Nominal steady state

The desired nominal state is $\mathrm{C}$. The following rates are calculated only over $\mathcal{S}_{\mathrm{N}}$:

$$
r_{\mathrm{C}\mid\mathrm{N,S}}
=
\frac{\#\{k\in\mathcal{S}_{\mathrm{N}}\mid R_k=\mathrm{C}\}}
{|\mathcal{S}_{\mathrm{N}}|},
$$

$$
r_{\mathrm{U}\mid\mathrm{N,S}}
=
\frac{\#\{k\in\mathcal{S}_{\mathrm{N}}\mid R_k=\mathrm{U}\}}
{|\mathcal{S}_{\mathrm{N}}|},
$$

$$
r_{\mathrm{I}\mid\mathrm{N,S}}
=
\frac{\#\{k\in\mathcal{S}_{\mathrm{N}}\mid R_k=\mathrm{I}\}}
{|\mathcal{S}_{\mathrm{N}}|}.
$$

The corresponding CSV columns are:

- `steady_nominal_consistent_rate`,
- `steady_nominal_undecided_rate`,
- `steady_nominal_false_inconsistent_rate`.

The last quantity is a steady-state false-alarm rate.

### 3.2 Disturbed steady state

The desired disturbed state is $\mathrm{I}$. The corresponding rates over $\mathcal{S}_{\mathrm{F}}$ are

$$
r_{\mathrm{C}\mid\mathrm{F,S}},
\qquad
r_{\mathrm{U}\mid\mathrm{F,S}},
\qquad
r_{\mathrm{I}\mid\mathrm{F,S}}.
$$

The corresponding CSV columns are:

- `steady_fault_false_consistent_rate`,
- `steady_fault_undecided_rate`,
- `steady_fault_inconsistent_rate`.

The most critical quantity is often

$$
r_{\mathrm{C}\mid\mathrm{F,S}},
$$

because it measures how often the monitor confidently supports consistency during a persistent assumption violation.

---

## 4. Primary undecided metric

The central metric for the new selection objective is the undecided rate outside transition windows:

$$
r_{\mathrm{U,S}}
=
\frac{\#\{k\in\mathcal{S}\mid R_k=\mathrm{U}\}}
{|\mathcal{S}|}.
$$

It is stored as:

```text
steady_undecided_rate
```

A low value means that the monitor rarely remains undecided once the system is in a stable nominal or disturbed phase.

This metric is preferred over the previous whole-interval undecided rates because samples during expected state changes are no longer treated as undesirable.

---

## 5. Balanced steady-state correctness

A parameter combination should not reduce undecided merely by making the same confident decision everywhere. Therefore, correct occupancy in nominal and disturbed steady states is summarized by

$$
B_{\mathrm{S}}
=
\frac{1}{2}
\left(
    r_{\mathrm{C}\mid\mathrm{N,S}}
    +
    r_{\mathrm{I}\mid\mathrm{F,S}}
\right).
$$

It is stored as:

```text
steady_balanced_correct_rate
```

The balanced form gives equal importance to nominal and disturbed intervals even if their numbers of samples differ.

---

## 6. Localization of undecided samples

### 6.1 Undecided localization precision

The fraction of all undecided samples that lies inside transition windows is

$$
L_{\mathrm{U}}
=
\frac{
\#\{k\in\mathcal{T}\mid R_k=\mathrm{U}\}
}{
\#\{k\in\mathcal{K}\mid R_k=\mathrm{U}\}
}.
$$

It is stored as:

```text
undecided_localization_precision
```

A value close to one means that undecided is strongly localized to expected state changes.

This metric must not be optimized alone. A classifier that is never undecided would have an undefined localization precision and could still switch directly between confident regions. It is therefore evaluated jointly with the mediated-transition rate.

### 6.2 Undecided sample rate inside transition windows

The fraction of transition-window samples classified as undecided is

$$
r_{\mathrm{U}\mid\mathrm{T}}
=
\frac{
\#\{k\in\mathcal{T}\mid R_k=\mathrm{U}\}
}{|\mathcal{T}|}.
$$

It is stored as:

```text
transition_undecided_sample_rate
```

This is a descriptive metric. It is **not maximized**, because that would reward unnecessarily long undecided intervals.

---

## 7. Transition-path metrics

For every disturbance onset and recovery, a target region is defined:

- onset: $\mathrm{C}\rightarrow\mathrm{I}$,
- recovery: $\mathrm{I}\rightarrow\mathrm{C}$.

Let $g_j$ denote the target region of transition $j$. With a persistence requirement of $p$ samples, the first persistent target-state index is

$$
\hat{t}_j
=
\min
\left\{
 t\ge t_j
 \;\middle|\;
 R_t=R_{t+1}=\dots=R_{t+p-1}=g_j
\right\}.
$$

### 7.1 Transition completion rate

A transition is completed if $\hat{t}_j$ exists inside its admissible transition window. The completion rate is

$$
r_{\mathrm{comp}}
=
\frac{\#\{j\mid \hat{t}_j\text{ exists}\}}
{J},
$$

where $J$ is the number of evaluated onset and recovery transitions.

CSV column:

```text
transition_completion_rate
```

### 7.2 Mediated-transition rate

A completed transition is considered mediated by undecided if

$$
\exists k\in[t_j,\hat{t}_j)
\quad\text{such that}\quad
R_k=\mathrm{U}.
$$

The rate is

$$
r_{\mathrm{med}}
=
\frac{
\#\{j\mid \text{transition }j\text{ is completed through }\mathrm{U}\}
}{J}.
$$

CSV column:

```text
mediated_transition_rate
```

This metric directly captures the intended three-state behavior.

### 7.3 Direct-transition rate

A direct switch reaches the target region without an intermediate undecided sample:

$$
r_{\mathrm{direct}}
=
\frac{
\#\{j\mid \text{transition }j\text{ is completed without }\mathrm{U}\}
}{J}.
$$

CSV column:

```text
direct_transition_rate
```

For the intended behavior, $r_{\mathrm{med}}$ should be high and $r_{\mathrm{direct}}$ low.

### 7.4 Settling delay

For each completed transition,

$$
\Delta_j=\hat{t}_j-t_j.
$$

The script reports the mean and median values:

- `mean_transition_settling_delay_steps`,
- `median_transition_settling_delay_steps`.

The number of undecided samples before reaching the target is reported as:

```text
mean_undecided_steps_per_transition
```

A desirable parameter combination uses undecided as a short intermediate state rather than as a long-term outcome.

---

## 8. Selective-classification metrics

Outside transition windows, the decision coverage is

$$
C_{\mathrm{S}}
=
\frac{\#\{k\in\mathcal{S}\mid R_k\neq\mathrm{U}\}}
{|\mathcal{S}|}.
$$

CSV column:

```text
steady_decision_coverage
```

The selective accuracy among confident decisions is

$$
A_{\mathrm{sel,S}}
=
\frac{
\#\{k\in\mathcal{S}\mid R_k\text{ is confident and correct}\}
}{
\#\{k\in\mathcal{S}\mid R_k\neq\mathrm{U}\}
}.
$$

CSV column:

```text
steady_selective_accuracy
```

Coverage must be high, but not at the expense of confidently wrong decisions.

---

## 9. Continuous credibility metrics

Before applying the threshold $\eta$, the inconsistent credibility

$$
q_{\mathrm{I},k}
=
P\!\left(\theta_k<\tau_{\mathrm{OK}}\mid\omega_k\right)
$$

is evaluated as a continuous score. The sweep retains:

- ROC-AUC: `q_inconsistent_roc_auc`,
- average precision: `q_inconsistent_average_precision`,
- mean separation between disturbed and nominal intervals: `q_inconsistent_mean_separation`.

These metrics evaluate the mapping itself independently of the selected decision threshold $\eta$.

---

## 10. Feasibility constraints

A parameter set is marked as `design_feasible` only if all configured requirements are fulfilled. With the current defaults:

$$
r_{\mathrm{I}\mid\mathrm{N,S}} \le 0.01,
$$

$$
r_{\mathrm{C}\mid\mathrm{F,S}} \le 0.05,
$$

$$
r_{\mathrm{U,S}} \le 0.10,
$$

$$
B_{\mathrm{S}} \ge 0.85,
$$

$$
r_{\mathrm{med}} \ge 0.80,
$$

$$
r_{\mathrm{event\ detection}} \ge 0.80.
$$

These are application-specific design targets, not universal statistical constants.

The CSV additionally contains:

- `constraint_violation_count`,
- `normalized_constraint_violation`,
- `design_feasible`.

For compatibility, `safety_feasible` contains the same Boolean value. It must not be interpreted as a proof of safety.

---

## 11. Ranking logic

The ranking is lexicographic rather than based on one arbitrary weighted score.

The priorities are:

1. satisfy all hard design constraints,
2. for infeasible candidates, minimize the number of violated constraints,
3. minimize the normalized magnitude of constraint violations,
4. minimize `steady_undecided_rate_mean`,
5. maximize `steady_balanced_correct_rate_mean`,
6. maximize `undecided_localization_precision_mean`,
7. maximize `mediated_transition_rate_mean`,
8. minimize steady-state confidently wrong decisions,
9. maximize event detection,
10. minimize transition settling delay,
11. minimize excessive region switching.

This ordering implements the intended behavior:

> Undecided is acceptable as a short transition state, but not as a persistent steady-state output.

---

## 12. New result files

The transition-focused script writes:

```text
parameter_sweep_run_metrics.csv
parameter_sweep_event_metrics.csv
parameter_sweep_transition_metrics.csv
parameter_sweep_summary.csv
parameter_sweep_ranked.csv
parameter_sweep_pareto_front.csv
parameter_sweep_top_candidates.csv
parameter_sweep_undecided_tradeoff.png
```

The additional file `parameter_sweep_transition_metrics.csv` contains one row per onset and recovery transition and Monte-Carlo run.

---

## 13. Recommended initial command

For the first transition-focused sweep with fixed decision-policy parameters:

```bash
python3 02_KalmanFilterSelfAssessmentParameterSweep_transition_focused.py sweep \
    --cache-glob "sweep_caches/*.npz" \
    --output-dir sweep_results_transition_focused \
    --num-bins 5,7,10 \
    --short-window-sizes 20,25,35,50 \
    --discounts 0.90,0.95,0.98,0.99 \
    --base-rates 0.5 \
    --etas 0.95 \
    --tau-ok-values 0.50 \
    --transition-window-before 5 \
    --transition-window-after 35 \
    --max-steady-undecided 0.10 \
    --min-steady-balanced-correct 0.85 \
    --min-mediated-transition-rate 0.80
```

Because the previous top candidates showed very large whole-interval undecided rates, shorter windows and lower discount factors should be included. They reduce memory persistence and may allow the opinion to leave the undecided region more rapidly. The fixed transition window then penalizes parameter combinations that remain undecided for too long.
