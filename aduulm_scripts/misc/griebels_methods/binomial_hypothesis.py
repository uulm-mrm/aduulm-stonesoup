from scipy.stats import chi2
from collections import deque
import numpy as np

class GriebelBinomialOpinion:
    """
    Binomial SL opinion for X = {H0 accepted, H0 rejected}.
    Evidence convention:
        r_pos = evidence for H0 accepted
        r_neg = evidence for H0 rejected
    Base rates:
        a_pos = 1 - alpha
        a_neg = alpha
    """

    def __init__(self, alpha=0.05, window_length=35, W=2.0):
        self.alpha = alpha
        self.window_length = window_length
        self.W = W
        self.evidence_window = deque(maxlen=window_length)

    def add_test_result(self, accept_h0: bool):
        self.evidence_window.append(1 if accept_h0 else 0)

    def opinion(self):
        r_pos = float(sum(self.evidence_window))
        r_neg = float(len(self.evidence_window) - r_pos)
        R = r_pos + r_neg

        if R == 0:
            return {
                "belief": 0.0,
                "disbelief": 0.0,
                "uncertainty": 1.0,
                "base_rate": 1.0 - self.alpha,
                "projected_probability": 1.0 - self.alpha,
                "n_evidence": 0,
            }

        belief = r_pos / (R + self.W)
        disbelief = r_neg / (R + self.W)
        uncertainty = self.W / (R + self.W)
        base_rate = 1.0 - self.alpha
        projected_probability = belief + base_rate * uncertainty

        return {
            "belief": belief,
            "disbelief": disbelief,
            "uncertainty": uncertainty,
            "base_rate": base_rate,
            "projected_probability": projected_probability,
            "n_evidence": int(R),
        }

class GriebelInnovationTest:
    """
    Griebel-style NIS hypothesis test for single-object, no clutter.

    H0:
        The innovation gamma_k is consistent with innovation covariance S_k.

    Statistic:
        eps = gamma.T @ inv(S) @ gamma

    Decision:
        accept H0 iff eps in [r1, r2].
    """

    def __init__(self, dim_meas: int, alpha=0.05, window_length=35,
                 two_sided=True, mapping=None):
        self.dim_meas = dim_meas
        self.alpha = alpha
        self.window_length = window_length
        self.two_sided = two_sided
        self.mapping = mapping
        self.mapper = GriebelBinomialOpinion(alpha=alpha, window_length=window_length)

        if two_sided:
            self.r1 = chi2.ppf(alpha / 2.0, dim_meas)
            self.r2 = chi2.ppf(1.0 - alpha / 2.0, dim_meas)
        else:
            self.r1 = 0.0
            self.r2 = chi2.ppf(1.0 - alpha, dim_meas)

        self.stat_history = []
        self.accept_history = []
        self.score_history = []
        self.opinion_history = []

    def assess(self, z, z_pred, S):
        S = np.asarray(S, dtype=float)
        S = np.atleast_2d(S)

        z = np.asarray(z, dtype=float).reshape(-1, 1)
        z_pred = np.asarray(z_pred, dtype=float).reshape(-1, 1)

        m = S.shape[0]

        if S.shape != (m, m):
            raise ValueError(f"S must be square, got shape {S.shape}")

        if z.shape[0] != m:
            if self.mapping is not None:
                z = z[list(self.mapping), :]
            else:
                raise ValueError(
                    f"z has dimension {z.shape[0]}, but S is {m}x{m}."
                )

        if z_pred.shape[0] != m:
            if self.mapping is not None:
                z_pred = z_pred[list(self.mapping), :]
            else:
                raise ValueError(
                    f"z_pred has dimension {z_pred.shape[0]}, but S is {m}x{m}."
                )

        gamma = z - z_pred

        # Robust NIS scalar extraction
        eps_arr = gamma.T @ np.linalg.solve(S, gamma)
        eps = float(np.asarray(eps_arr).reshape(-1)[0])

        accept_h0 = bool(self.r1 <= eps <= self.r2)

        self.mapper.add_test_result(accept_h0)
        op = self.mapper.opinion()

        self.stat_history.append(eps)
        self.accept_history.append(accept_h0)
        self.score_history.append(op["projected_probability"])
        self.opinion_history.append(op)

        return {
            "eps": eps,
            "accept_h0": accept_h0,
            "score": op["projected_probability"],
            "opinion": op,
            "r1": self.r1,
            "r2": self.r2,
        }