from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

class EdgeAnomalyDetector:
    """Unsupervised anomaly detector for edges using graph-structural features."""
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
        self.fitted = False

    def _edge_features(self, triples: List[Tuple[int,int,str]], rel: str, h: int, t: int, doc_size: int) -> List[float]:
        # Compute degrees and relation stats within the doc graph
        deg = {}
        rel_counts = {}
        rels_by_node = {}
        for (hh,tt,rr) in triples:
            deg[hh] = deg.get(hh, 0) + 1
            deg[tt] = deg.get(tt, 0) + 1
            rel_counts[rr] = rel_counts.get(rr, 0) + 1
            rels_by_node.setdefault(hh, set()).add(rr)
            rels_by_node.setdefault(tt, set()).add(rr)
        deg_h = deg.get(h, 0)
        deg_t = deg.get(t, 0)
        rel_count = rel_counts.get(rel, 0)
        inv_exists = 1.0 if (t, h, rel) in triples else 0.0
        rel_div_h = len(rels_by_node.get(h, set()))
        rel_div_t = len(rels_by_node.get(t, set()))
        return [deg_h, deg_t, rel_count, inv_exists, rel_div_h, rel_div_t, doc_size]

    def fit_on_ground_truth(self, ground_truth: Dict[str, List[Tuple[int,int,str]]]):
        X = []
        for did, triples in ground_truth.items():
            doc_size = len(triples)
            for (h,t,r) in triples:
                X.append(self._edge_features(triples, r, h, t, doc_size))
        if not X:
            return
        X = np.array(X, dtype=np.float32)
        self.model.fit(X)
        self.fitted = True

    def predict_is_outlier(self, doc_triples: List[Tuple[int,int,str]], edge: Tuple[int,int,str]) -> bool:
        if not self.fitted:
            return False
        (h,t,r) = edge
        feat = np.array([self._edge_features(doc_triples, r, h, t, len(doc_triples))], dtype=np.float32)
        pred = self.model.predict(feat)[0]  # -1 outlier, 1 inlier
        return pred == -1

