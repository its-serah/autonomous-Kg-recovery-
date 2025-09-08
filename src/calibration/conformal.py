from typing import Dict, List, Tuple
import numpy as np

# Conformal prediction for binary classification (positive class acceptance)
# Mondrian (per-relation) inductive conformal classifier thresholds.
# We assume a probabilistic model that outputs P(y=1|x) for each (x, r).
# For each relation r, compute nonconformity scores for positives: a = 1 - p_pos.
# Set threshold tau_r as the (1 - alpha) quantile of these scores.
# At test time, accept positive if (1 - p_pos) <= tau_r  <=>  p_pos >= 1 - tau_r.

def _quantile(scores: np.ndarray, q: float) -> float:
    if scores.size == 0:
        return 0.5
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q))

def compute_conformal_thresholds(
    probs: List[float], labels: List[int], relations: List[str],
    alpha: float = 0.1, min_per_rel: int = 20
) -> Dict[str, float]:
    """Return per-relation probability thresholds using conformal calibration.
    probs: model P(y=1|x)
    labels: 0/1 ground-truth
    relations: relation id for each example
    alpha: target miscoverage for positive class (class-conditional)
    min_per_rel: minimum positives needed for Mondrian; otherwise fallback to global.
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    assert probs.shape[0] == labels.shape[0] == len(relations)

    # Nonconformity for positives: a = 1 - p
    rel2pos_scores: Dict[str, List[float]] = {}
    global_pos_scores: List[float] = []

    for p, y, r in zip(probs, labels, relations):
        if y == 1:
            s = 1.0 - float(p)
            rel2pos_scores.setdefault(r, []).append(s)
            global_pos_scores.append(s)

    # Global threshold fallback
    global_tau = 1.0 - _quantile(np.array([1.0 - s for s in global_pos_scores], dtype=np.float32), 1 - alpha) if global_pos_scores else 0.5

    rel2thr: Dict[str, float] = {}
    for r, scores in rel2pos_scores.items():
        if len(scores) >= min_per_rel:
            # tau on nonconformity => prob threshold = 1 - tau
            tau_nc = _quantile(np.array(scores, dtype=np.float32), 1 - alpha)
            rel2thr[r] = float(1.0 - tau_nc)
        else:
            rel2thr[r] = float(global_tau)

    # Ensure all relations have a threshold
    unique_relations = set(relations)
    for r in unique_relations:
        if r not in rel2thr:
            rel2thr[r] = float(global_tau)

    return rel2thr

