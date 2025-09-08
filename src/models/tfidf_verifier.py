from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import numpy as np

@dataclass
class TripleExample:
    text: str
    relation: str
    label: int  # 1 positive, 0 negative

class DocLevelVerifier:
    """TF-IDF + Logistic Regression verifier with relation token as feature."""

    def __init__(self):
        self.pipeline: Pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=200000, min_df=2)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.is_fit = False

    def _encode(self, text: str, relation: str) -> str:
        # Inject relation token
        return f"REL_{relation} \n {text}"

    def fit(self, examples: List[TripleExample]):
        X = [self._encode(ex.text, ex.relation) for ex in examples]
        y = [ex.label for ex in examples]
        self.pipeline.fit(X, y)
        self.is_fit = True

    def predict_proba(self, texts_relations: List[Tuple[str,str]]) -> np.ndarray:
        assert self.is_fit, "Model not fitted"
        X = [self._encode(t, r) for t, r in texts_relations]
        proba = self.pipeline.predict_proba(X)[:, 1]
        return proba

    def calibrate_thresholds(self, dev_examples: List[TripleExample]) -> Dict[str, float]:
        # Per-relation thresholds maximizing F1 on dev; fallback to 0.5
        rel2data: Dict[str, List[Tuple[float,int]]] = {}
        for ex in dev_examples:
            p = self.predict_proba([(ex.text, ex.relation)])[0]
            rel2data.setdefault(ex.relation, []).append((p, ex.label))
        rel2thr: Dict[str, float] = {}
        for r, pairs in rel2data.items():
            if not pairs:
                rel2thr[r] = 0.5
                continue
            scores, labels = zip(*pairs)
            scores = np.array(scores)
            labels = np.array(labels)
            # scan thresholds on unique scores
            uniq = np.unique(scores)
            best_f1, best_t = -1.0, 0.5
            for t in np.linspace(0.1, 0.9, 17):
                preds = (scores >= t).astype(int)
                f1 = f1_score(labels, preds) if (preds.sum() and labels.sum()) else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            rel2thr[r] = float(best_t)
        return rel2thr

