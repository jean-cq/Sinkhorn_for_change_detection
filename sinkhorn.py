from __future__ import annotations
import numpy as np

# ============================================================
# 0) Sinkhorn
# ============================================================

def _logsumexp(A: np.ndarray, axis: int) -> np.ndarray:
    """Stable log(sum(exp(A))) along axis."""
    Amax = np.max(A, axis=axis, keepdims=True)
    out = Amax + np.log(np.sum(np.exp(A - Amax), axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)

class SinkhornOT:
    """
    Log-domain Sinkhorn for entropic OT.
    - balanced: exact marginals P1=a, P^T1=b
    - unbalanced (KL-relaxed): penalize mismatch with tau_a, tau_b
    """
    # barycenter?
    # eps: need to be smaller, e.g. 0.001/0.01
    # eye power to check its correctness
    # is there any quantitative correctness measurement?
    def __init__(
        self,
        eps: float = 0.01,
        n_iters: int = 2000,
        tol: float = 1e-9,
        unbalanced: bool = False,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
    ):
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.unbalanced = bool(unbalanced)
        self.tau_a = float(tau_a)
        self.tau_b = float(tau_b)

    def solve(self, a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)

        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("a,b must be nonnegative")

        n, m = C.shape
        if a.shape != (n,) or b.shape != (m,):
            raise ValueError("Shape mismatch: a is (n,), b is (m,), C is (n,m).")

        if not self.unbalanced:
            a = a / (a.sum() + 1e-300)
            b = b / (b.sum() + 1e-300)

        logK = -C / self.eps
        loga = np.log(a + 1e-300)
        logb = np.log(b + 1e-300)

        if self.unbalanced:
            ka = self.tau_a / (self.tau_a + self.eps)
            kb = self.tau_b / (self.tau_b + self.eps)
        else:
            ka = 1.0
            kb = 1.0

        f = np.zeros(n, dtype=np.float64)
        g = np.zeros(m, dtype=np.float64)

        for _ in range(self.n_iters):
            f_prev = f.copy()

            f = ka * self.eps * (loga - _logsumexp(logK + g[None, :] / self.eps, axis=1))
            g = kb * self.eps * (logb - _logsumexp(logK + f[:, None] / self.eps, axis=0))

            if np.linalg.norm(f - f_prev, 1) < self.tol:
                break

        P = np.exp((f[:, None] + g[None, :] - C) / self.eps)

        if not self.unbalanced:
            P = P * (a / (P.sum(axis=1) + 1e-300))[:, None]
            P = P * (b / (P.sum(axis=0) + 1e-300))[None, :]

        cost = float(np.sum(P * C))
        return P, cost


# ============================================================
# 1) Utilities: distances, normalization, gating-aware cost
# ============================================================

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X: (n,d), Y: (m,d)
    returns: (n,m) squared Euclidean distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True).T
    return XX + YY - 2.0 * (X @ Y.T)

def l2_normalize_rows(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (often helpful for embeddings)."""
    F = np.asarray(F, dtype=np.float64)
    norms = np.sqrt(np.sum(F * F, axis=1, keepdims=True))
    return F / np.maximum(norms, eps)

# normolise by the max C_ij/entries to make sure the cost calculation
def normalize_xy(XY: np.ndarray, method: str = "unit_box") -> np.ndarray:
    """
    Normalize XY so geometry cost scale is stable.
    method:
      - "unit_box": map to approx [0,1] using min/max per axis
      - "diag": divide by image diagonal (if XY already in pixel coords, you can pass diag yourself instead)
    """
    XY = np.asarray(XY, dtype=np.float64)
    if method == "unit_box":
        mn = XY.min(axis=0, keepdims=True)
        mx = XY.max(axis=0, keepdims=True)
        return (XY - mn) / (mx - mn + 1e-12)
    elif method == "diag":
        # If you use this, you should pre-divide XY by your diag externally for clarity.
        return XY
    else:
        raise ValueError(f"Unknown method: {method}")

def build_local_candidates(XY1: np.ndarray, XY2: np.ndarray, gate_radius: float):
    """
    For each source patch i, find target patches j within gate_radius.
    Returns:
      candidates: list of numpy arrays, candidates[i] = array of valid j indices
    """
    XY1 = np.asarray(XY1, dtype=np.float64)
    XY2 = np.asarray(XY2, dtype=np.float64)

    candidates = []
    r2 = gate_radius ** 2

    for i in range(len(XY1)):
        d2 = np.sum((XY2 - XY1[i]) ** 2, axis=1)
        js = np.where(d2 <= r2)[0]
        candidates.append(js)

    return candidates


def build_candidate_cost(
    XY1: np.ndarray,
    XY2: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    candidates,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gate_cost: float = 50.0,
):
    """
    Build one final dense cost matrix C, but only compute true costs for local candidates.
    Non-candidate pairs are assigned gate_cost.

    Returns:
      C: (n,m)
    """
    n = len(XY1)
    m = len(XY2)
    C = np.full((n, m), gate_cost, dtype=np.float64)

    for i, js in enumerate(candidates):
        if len(js) == 0:
            continue

        geo = np.sum((XY2[js] - XY1[i]) ** 2, axis=1)
        feat = np.sum((F2[js] - F1[i]) ** 2, axis=1)

        C[i, js] = alpha * geo + beta * feat

    return C
def make_gated_cost(
    XY1: np.ndarray,
    XY2: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gate_radius: float = 0.05,
    gate_cost: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build C = alpha*C_geo + beta*C_feat with gating.

    XY1, XY2: (n,2), (m,2) in comparable units (recommend normalized to ~[0,1]).
    F1, F2: (n,d), (m,d) embeddings (recommend L2-normalized).
    gate_radius: max allowed spatial shift (in XY units).
    gate_cost: cost value assigned to gated-out pairs.
      If None, a safe default is chosen relative to alpha/beta scales.

    Returns:
      C:      (n,m) final cost
      C_geo:  (n,m) geo squared distance
      C_feat: (n,m) feat squared distance
    """
    C_geo = pairwise_sq_dists(XY1, XY2)
    C_feat = pairwise_sq_dists(F1, F2)

    C = alpha * C_geo + beta * C_feat

    # Gating mask uses geo distance only
    mask = C_geo > (gate_radius ** 2)

    if gate_cost is None:
        # "large but not insane": aim for exp(-gate_cost/eps) ~ tiny in solver,
        # but keep numbers reasonable for stability.
        # We'll set it to a high quantile of current (non-gated) costs, then add margin.
        finite_vals = C[~mask]
        if finite_vals.size == 0:
            # If everything is gated, your gate_radius is too small.
            gate_cost = 50.0
        else:
            q = float(np.quantile(finite_vals, 0.99))
            gate_cost = q + 10.0

    C = np.where(mask, gate_cost, C)
    return C, C_geo, C_feat


# ============================================================
# 2) Patch-level change scores and helpers
# ============================================================

def patch_change_scores_expected_cost(P: np.ndarray, C: np.ndarray, out_mass: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Score_i = (sum_j P_ij * C_ij) / (sum_j P_ij)
    Higher => more "changed" (harder/more expensive to match).
    If out_mass is tiny => treat as highly changed (will blow up unless we clamp).
    """
    P = np.asarray(P, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    out_mass = np.asarray(out_mass, dtype=np.float64)

    numer = np.sum(P * C, axis=1)
    denom = np.maximum(out_mass, eps)
    return numer / denom

def patch_unmatched_score(out_mass: np.ndarray, *, a: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    """
    A simpler change score: how much mass "disappeared" from each source patch.
    If you provide a (source weights), score = 1 - out_mass/a (clipped).
    """
    out_mass = np.asarray(out_mass, dtype=np.float64)
    if a is None:
        # interpret relative to max out_mass for stability
        denom = np.maximum(out_mass.max(), eps)
        return 1.0 - np.clip(out_mass / denom, 0.0, 1.0)

    a = np.asarray(a, dtype=np.float64)
    frac = out_mass / np.maximum(a, eps)
    return 1.0 - np.clip(frac, 0.0, 1.0)
# ============================================================
# 3) End-to-end: run patch OT change detection
# ============================================================
def sinkhorn_patch_change(
    XY1: np.ndarray,
    F1: np.ndarray,
    XY2: np.ndarray,
    F2: np.ndarray,
    *,
    # cost weights + gating
    alpha: float = 1.0,
    beta: float = 1.0,
    gate_radius: float = 0.03,
    gate_cost: float | None = None,
    # solver params
    eps: float = 0.2,
    tau_a: float = 0.5,
    tau_b: float = 0.5,
    n_iters: int = 500,
    tol: float = 1e-6,
    # preprocessing
    normalize_xy_method: str = "unit_box",
    l2norm_features: bool = True,
    # weights
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> dict:
    """
    Best workflow:
      - normalize XY
      - (optional) L2-normalize embeddings
      - build gated cost C
      - run YOUR log-domain unbalanced Sinkhorn
      - return plan + costs + per-patch change scores

    Returns dict with:
      P, ot_cost, out_mass, in_mass,
      score_expected_cost, score_unmatched,
      C, C_geo, C_feat
    """
    XY1 = np.asarray(XY1, dtype=np.float64)
    XY2 = np.asarray(XY2, dtype=np.float64)
    F1 = np.asarray(F1, dtype=np.float64)
    F2 = np.asarray(F2, dtype=np.float64)

    # normalize XY to stable scale (highly recommended)
    XY1n = normalize_xy(XY1, method=normalize_xy_method)
    XY2n = normalize_xy(XY2, method=normalize_xy_method)

    # normalize embeddings (often helps)
    if l2norm_features:
        F1n = l2_normalize_rows(F1)
        F2n = l2_normalize_rows(F2)
    else:
        F1n, F2n = F1, F2

    # weights
    n = XY1n.shape[0]
    m = XY2n.shape[0]
    if a is None:
        a = np.ones(n, dtype=np.float64)
    if b is None:
        b = np.ones(m, dtype=np.float64)

    # build cost matrix (with gating)
    # C, C_geo, C_feat = make_gated_cost(
    #     XY1n, XY2n, F1n, F2n,
    #     alpha=alpha, beta=beta,
    #     gate_radius=gate_radius,
    #     gate_cost=gate_cost
    # )

    # local-candidate cost construction
    candidates = build_local_candidates(XY1n, XY2n, gate_radius=gate_radius)

    if gate_cost is None:
        gate_cost = 50.0

    C = build_candidate_cost(
        XY1n, XY2n, F1n, F2n,
        candidates,
        alpha=alpha,
        beta=beta,
        gate_cost=gate_cost
    )
    C_geo = None
    C_feat = None

    print(f"[INFO] OT problem size: {len(XY1)} x {len(XY2)} = {len(XY1) * len(XY2):,}")

    # solve unbalanced OT (log-domain)
    solver = SinkhornOT(
        eps=eps,
        n_iters=n_iters,
        tol=tol,
        unbalanced=True,
        tau_a=tau_a,
        tau_b=tau_b,
    )
    P, ot_cost = solver.solve(a, b, C)

    out_mass = P.sum(axis=1)
    in_mass = P.sum(axis=0)

    # change scores
    score_expected = patch_change_scores_expected_cost(P, C, out_mass)
    score_unmatch = patch_unmatched_score(out_mass, a=a)

    return {
        "P": P,
        "ot_cost": ot_cost,
        "out_mass": out_mass,
        "in_mass": in_mass,
        "score_expected_cost": score_expected,
        "score_unmatched": score_unmatch,
        "C": C,
        "C_geo": C_geo,
        "C_feat": C_feat,
        "XY1_norm": XY1n,
        "XY2_norm": XY2n,
        "F1_norm": F1n,
        "F2_norm": F2n,
    }
# ============================================================
# 4) Object-level cost helpers
# ============================================================

def standardize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Column-wise standardization:
      X_std = (X - mean) / std
    Useful for shape / metadata vectors whose scales differ a lot.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    return (X - mu) / np.maximum(sd, eps)


def make_object_cost(
    XY1: np.ndarray,
    XY2: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    gate_radius: float = 0.05,
    gate_cost: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build object-level cost matrix

      C = alpha * C_geo + beta * C_feat + gamma * C_shape

    with gating based only on geometry.

    Returns:
      C, C_geo, C_feat, C_shape
    """
    C_geo = pairwise_sq_dists(XY1, XY2)
    C_feat = pairwise_sq_dists(F1, F2)
    C_shape = pairwise_sq_dists(S1, S2)

    C = alpha * C_geo + beta * C_feat + gamma * C_shape

    mask = C_geo > (gate_radius ** 2)

    if gate_cost is None:
        finite_vals = C[~mask]
        if finite_vals.size == 0:
            gate_cost = 50.0
        else:
            q = float(np.quantile(finite_vals, 0.99))
            gate_cost = q + 10.0

    C = np.where(mask, gate_cost, C)
    return C, C_geo, C_feat, C_shape


# ============================================================
# 5) Object-level change scores
# ============================================================

def object_change_scores_expected_cost(
    P: np.ndarray,
    C: np.ndarray,
    mass: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Same idea as patch expected-cost score, but for objects.
    """
    P = np.asarray(P, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    numer = np.sum(P * C, axis=1)
    denom = np.maximum(mass, eps)
    return numer / denom
def object_change_scores_expected_cost_tgt(
    P: np.ndarray,
    C: np.ndarray,
    mass: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Target-side expected transport cost, one score per target object.
    """
    P = np.asarray(P, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    numer = np.sum(P * C, axis=0)
    denom = np.maximum(mass, eps)
    return numer / denom

def object_unmatched_score(
    mass: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Same as patch unmatched score, but for objects.
    """
    mass = np.asarray(mass, dtype=np.float64)

    if weights is None:
        denom = np.maximum(mass.max(), eps)
        return 1.0 - np.clip(mass / denom, 0.0, 1.0)

    weights = np.asarray(weights, dtype=np.float64)
    frac = mass / np.maximum(weights, eps)
    return 1.0 - np.clip(frac, 0.0, 1.0)


# ============================================================
# 6) End-to-end: run object OT change detection
# ============================================================

def sinkhorn_object_change(
    XY1: np.ndarray,
    F1: np.ndarray,
    S1: np.ndarray,
    XY2: np.ndarray,
    F2: np.ndarray,
    S2: np.ndarray,
    *,
    # cost weights
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    gate_radius: float = 0.03,
    gate_cost: float | None = None,
    # solver params
    eps: float = 0.2,
    tau_a: float = 0.5,
    tau_b: float = 0.5,
    n_iters: int = 500,
    tol: float = 1e-6,
    # preprocessing
    normalize_xy_method: str = "unit_box",
    l2norm_features: bool = True,
    standardize_shape: bool = True,
    # weights
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> dict:
    """
    Object-based OT change detection.

    Inputs:
      XY1, XY2 : object centroids / coordinates, shape (n,2), (m,2)
      F1, F2   : object appearance features / embeddings
      S1, S2   : object metadata / shape features

    Cost:
      C_ij = alpha ||x_i - y_j||^2
           + beta  ||f_i - g_j||^2
           + gamma ||s_i - s'_j||^2

    Returns:
      P, ot_cost, out_mass, in_mass,
      score_expected_cost, score_unmatched,
      C, C_geo, C_feat, C_shape,
      normalized inputs
    """
    XY1 = np.asarray(XY1, dtype=np.float64)
    XY2 = np.asarray(XY2, dtype=np.float64)
    F1 = np.asarray(F1, dtype=np.float64)
    F2 = np.asarray(F2, dtype=np.float64)
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)

    # --------------------------------------------------------
    # normalize geometry
    # --------------------------------------------------------
    XY1n = normalize_xy(XY1, method=normalize_xy_method)
    XY2n = normalize_xy(XY2, method=normalize_xy_method)

    # --------------------------------------------------------
    # normalize appearance features
    # --------------------------------------------------------
    if l2norm_features:
        F1n = l2_normalize_rows(F1)
        F2n = l2_normalize_rows(F2)
    else:
        F1n, F2n = F1, F2

    # --------------------------------------------------------
    # normalize shape/meta features
    # --------------------------------------------------------
    if standardize_shape:
        S1n = standardize_rows(S1)
        S2n = standardize_rows(S2)
    else:
        S1n, S2n = S1, S2

    # --------------------------------------------------------
    # weights
    # --------------------------------------------------------
    n = XY1n.shape[0]
    m = XY2n.shape[0]

    if a is None:
        a = np.ones(n, dtype=np.float64)
    if b is None:
        b = np.ones(m, dtype=np.float64)

    # --------------------------------------------------------
    # cost matrix
    # --------------------------------------------------------
    C, C_geo, C_feat, C_shape = make_object_cost(
        XY1n, XY2n, F1n, F2n, S1n, S2n,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        gate_radius=gate_radius,
        gate_cost=gate_cost,
    )

    print(f"[INFO] Object OT problem size: {n} x {m} = {n * m:,}")

    # --------------------------------------------------------
    # solve unbalanced OT
    # --------------------------------------------------------
    solver = SinkhornOT(
        eps=eps,
        n_iters=n_iters,
        tol=tol,
        unbalanced=True,
        tau_a=tau_a,
        tau_b=tau_b,
    )
    P, ot_cost = solver.solve(a, b, C)

    out_mass = P.sum(axis=1)
    in_mass = P.sum(axis=0)

    # --------------------------------------------------------
    # scores
    # --------------------------------------------------------
    score_expected = object_change_scores_expected_cost(P, C, out_mass)
    score_unmatch = object_unmatched_score(out_mass, weights=a)

    # target-side / bidirectional scores
    score_expected_tgt = object_change_scores_expected_cost_tgt(P, C, in_mass)
    score_unmatch_tgt = object_unmatched_score(in_mass, weights=b)

    return {
        "P": P,
        "ot_cost": ot_cost,
        "out_mass": out_mass,
        "in_mass": in_mass,
        "score_expected_cost_src": score_expected,
        "score_unmatched_src": score_unmatch,
        "score_expected_cost_tgt": score_expected_tgt,
        "score_unmatched_tgt": score_unmatch_tgt,
        "C": C,
        "C_geo": C_geo,
        "C_feat": C_feat,
        "C_shape": C_shape,
        "XY1_norm": XY1n,
        "XY2_norm": XY2n,
        "F1_norm": F1n,
        "F2_norm": F2n,
        "S1_norm": S1n,
        "S2_norm": S2n,
    }