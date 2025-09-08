from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
import numpy as np

class StackingEnsemble:
    """Logistic regression meta-classifier to combine features.
    Features per triple: [text_prob, kge_score, rule_flag_*]
    """
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.clf.predict_proba(X)[:,1]

