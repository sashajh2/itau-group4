import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def get_oof_linear_probe_probs(embeddings: np.ndarray, labels: np.ndarray, folds: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Run k-fold cross-validated logistic regression and return out-of-fold probabilities for each sample.
    Args:
        embeddings: (n_samples, n_features) array
        labels: (n_samples,) array
        folds: number of cross-validation folds
        random_state: random seed
    Returns:
        oof_probs: (n_samples,) array of predicted probabilities (for class 1)
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(labels))
    for train_idx, test_idx in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train = labels[train_idx]
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        oof_probs[test_idx] = clf.predict_proba(X_test)[:, 1]
    return oof_probs 