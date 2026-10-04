"""Zufällige Instanzen gegen scikit-learn, Gleichstand-bewusst (ergänzt die festen Fälle aus test_algorithm.py).

Instanzen, in denen irgendein Knoten zwei Splits mit exakt gleichem Gain hat, werden übersprungen: scikit-learn löst Gleichstände zufällig auf, die Demo über das kleinste Merkmal/die kleinste Schwelle.
Verglichen werden Blattzahl, Vorhersagen, Wichtigkeit, AUC/Log-Loss, der Beschneidungspfad (alphas und Gesamt-Unreinheit) sowie der Rauschanteil der Splits (cart_evaluation.split_shares)
gegen eine Auszählung auf scikit-learns Baum.
"""

import numpy as np
import pytest
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import cart_algorithm as alg
import cart_evaluation as ev


def _data(rng, task, n, d):
    X = rng.normal(size=(n, d))
    sig = X[:, 0] + 0.7 * np.sin(2 * X[:, min(1, d - 1)])
    y = (sig + rng.normal(0, 1.6, n) > 0).astype(float) if task == "class" else sig * 3 + 10 + rng.normal(0, 1.0, n)
    return X, y


def _from_sklearn(ref, task, n_total):
    t = ref.tree_
    val = t.value[:, 0, 1] / t.value[:, 0].sum(axis=1) if task == "class" else t.value[:, 0, 0]
    leaf = t.children_left < 0
    depth = np.zeros(t.node_count, dtype=int)
    for i in range(t.node_count):
        if not leaf[i]:
            depth[t.children_left[i]] = depth[t.children_right[i]] = depth[i] + 1
    return alg.Tree(np.where(leaf, -1, t.feature), np.where(leaf, np.nan, t.threshold), np.where(leaf, -1, t.children_left), np.where(leaf, -1, t.children_right), val, t.impurity.copy(),
                    t.n_node_samples.copy(), depth, task, "gini" if task == "class" else "variance", n_total, t.n_features)


def test_random_trees_equal_scikit_learn_where_no_split_ties(monkeypatch):
    original = alg.best_split
    ties = []

    def spy(X, y, criterion, min_leaf):
        if len(y) >= 2 * min_leaf and len(y) >= 2:
            g, _, _ = alg.gain_matrix(X, y, criterion, min_leaf)
            v = np.sort(g[np.isfinite(g)])[::-1]
            if len(v) >= 2 and abs(v[0] - v[1]) <= 1e-12:
                ties.append(1)
        return original(X, y, criterion, min_leaf)

    monkeypatch.setattr(alg, "best_split", spy)
    rng = np.random.default_rng(99)
    compared = 0
    for it in range(150):
        task = "class" if it % 3 else "reg"
        crit = str(rng.choice(["gini", "entropy"])) if task == "class" else "variance"
        n, d = int(rng.integers(80, 400)), int(rng.integers(1, 7))
        X, y = _data(rng, task, n, d)
        depth, leaf = int(rng.integers(1, 5)), int(rng.integers(8, 40))
        ties.clear()
        ours = alg.grow(X, y, task, crit, depth, leaf)
        if ties:
            continue
        ref = (DecisionTreeClassifier(criterion=crit, max_depth=depth, min_samples_leaf=leaf, random_state=0) if task == "class"
               else DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0)).fit(X, y)
        Xt = rng.normal(size=(100, d))
        assert ours.n_leaves == ref.get_n_leaves(), it
        assert np.allclose(alg.predict_value(ours, Xt), ref.predict_proba(Xt)[:, 1] if task == "class" else ref.predict(Xt)), it
        assert np.allclose(alg.importances(ours), ref.feature_importances_, atol=1e-9), it
        if task == "class":
            v = alg.predict_value(ours, X)
            assert alg.auc(y, v) == pytest.approx(metrics.roc_auc_score(y, v), abs=1e-12)
            assert alg.log_loss(y, v) == pytest.approx(metrics.log_loss(y, v), abs=1e-9)
        compared += 1
    assert compared >= 100


def test_pruning_path_and_split_shares_equal_scikit_learn_on_sklearn_trees():
    rng = np.random.default_rng(5)
    for it in range(40):
        task = "class" if it % 2 else "reg"
        n, d = int(rng.integers(100, 300)), int(rng.integers(3, 8))
        n_real = int(rng.integers(1, d))
        X, y = _data(rng, task, n, d)
        cls = DecisionTreeClassifier if task == "class" else DecisionTreeRegressor
        ref = cls(random_state=0, min_samples_leaf=int(rng.integers(1, 10))).fit(X, y)
        tree = _from_sklearn(ref, task, n)
        # Beschneidungspfad
        pth = ref.cost_complexity_pruning_path(X, y)
        mine = alg.pruning_path(tree)
        assert np.allclose([a for a, *_ in mine], pth.ccp_alphas, atol=1e-10) and np.allclose([r for _, _, r, _ in mine], pth.impurities, atol=1e-10), it
        # Rauschanteil der Splits je Ebene, von Hand auf scikit-learns Knotenlisten gezählt
        t = ref.tree_
        depth, stack, per = {0: 0}, [0], {}
        while stack:
            i = stack.pop()
            if t.children_left[i] >= 0:
                per.setdefault(depth[i], []).append(t.feature[i] >= n_real)
                for c in (t.children_left[i], t.children_right[i]):
                    depth[c] = depth[i] + 1
                    stack.append(c)
        sh = ev.split_shares(tree, n_real)
        total = sum(len(v) for v in per.values())
        assert sh["total"] == total
        assert sh["noise"] == pytest.approx(sum(sum(v) for v in per.values()) / total if total else 0.0)
        assert [(lv, k) for lv, k, _ in sh["by_level"]] == [(lv, len(per[lv])) for lv in sorted(per)]
        assert all(fr == pytest.approx(np.mean(per[lv])) for lv, _, fr in sh["by_level"])
