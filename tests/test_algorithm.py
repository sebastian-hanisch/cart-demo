"""CART gegen unabhängige Referenzen: Brute-Force-Split-Suche, scikit-learn (dort exakt, wo es keine Gleichstände gibt), Grenzfälle."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import cart_algorithm as alg
import cart_scenario as S


def _continuous(n=400, d=5, seed=0, task="class", noise=1.6):
    """Stetige Merkmale und stark verrauschte Ziele: so gibt es keine zufälligen Gleichstände zwischen Splits (reine oder fast reine Knoten hätten sie: ein einzelner Ausreißer lässt sich auf mehrere Arten abspalten)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.7 * np.sin(2 * X[:, 1]) + 0.5 * (X[:, 2] > 0.3) * X[:, 3]
    y = (signal + rng.normal(0, noise, n) > 0).astype(float) if task == "class" else signal * 3 + 10 + rng.normal(0, 1.0, n)
    return X, y


def _from_sklearn(ref, task, n_total):
    """Ein scikit-learn-Baum in unserem Feldformat, damit das Beschneiden unabhängig vom Wachsen (und seinen Gleichständen) geprüft wird."""
    t = ref.tree_
    val = t.value[:, 0, 1] / t.value[:, 0].sum(axis=1) if task == "class" else t.value[:, 0, 0]
    leaf = t.children_left < 0
    depth = np.zeros(t.node_count, dtype=int)
    for i in range(t.node_count):
        if not leaf[i]:
            depth[t.children_left[i]] = depth[t.children_right[i]] = depth[i] + 1
    return alg.Tree(np.where(leaf, -1, t.feature), np.where(leaf, np.nan, t.threshold), np.where(leaf, -1, t.children_left), np.where(leaf, -1, t.children_right), val, t.impurity.copy(),
                    t.n_node_samples.copy(), depth, task, "gini" if task == "class" else "variance", n_total, t.n_features)


def _brute(X, y, criterion, min_leaf):
    """Alle Schwellen aller Merkmale einzeln, Unreinheit der Kinder direkt aus den Beispielen."""
    best = (None, None, -np.inf)
    parent = alg.node_impurity(y, criterion)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        for a, b in zip(vals[:-1], vals[1:]):
            thr = (a + b) / 2.0
            L = X[:, f] <= thr
            if L.sum() < min_leaf or (~L).sum() < min_leaf:
                continue
            gain = parent - (L.sum() * alg.node_impurity(y[L], criterion) + (~L).sum() * alg.node_impurity(y[~L], criterion)) / len(y)
            if gain > best[2] + 1e-12:
                best = (f, thr, gain)
    return best


@pytest.mark.parametrize("criterion,task", [("gini", "class"), ("entropy", "class"), ("variance", "reg")])
@pytest.mark.parametrize("seed", range(4))
def test_best_split_equals_brute_force(criterion, task, seed):
    X, y = _continuous(60, 4, seed, task)
    f, thr, gain = alg.best_split(X, y, criterion, 1)
    bf, bthr, bgain = _brute(X, y, criterion, 1)
    assert (f, thr) == (bf, pytest.approx(bthr)) and gain == pytest.approx(bgain, abs=1e-9)


@pytest.mark.parametrize("criterion,task", [("gini", "class"), ("entropy", "class"), ("variance", "reg")])
def test_best_split_with_ties_and_min_leaf(criterion, task):
    rng = np.random.default_rng(3)
    X = rng.integers(0, 5, size=(50, 3)).astype(float)                       # viele gleiche Werte
    y = (X[:, 0] + rng.integers(0, 2, 50) > 3).astype(float) if task == "class" else X[:, 1] * 2 + rng.normal(size=50)
    for leaf in (1, 5, 12):
        f, thr, gain = alg.best_split(X, y, criterion, leaf)
        bf, bthr, bgain = _brute(X, y, criterion, leaf)
        assert gain == pytest.approx(bgain, abs=1e-9), (leaf,)
        L = X[:, f] <= thr
        assert L.sum() >= leaf and (~L).sum() >= leaf


def test_ties_prefer_the_lowest_feature_then_the_lowest_threshold():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])           # zwei gleiche Spalten, y = 0 0 1 1: der beste Split in der Mitte ist eindeutig
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert alg.best_split(X, y, "gini", 1)[:2] == (0, 1.5)
    y2 = np.array([0.0, 1.0, 0.0, 1.0])                                       # Gleichstand zwischen zwei Schwellen derselben Spalte
    f, thr, _ = alg.best_split(X[:, :1], y2, "gini", 1)
    assert (f, thr) == (0, 0.5)


CASES = [("gini", "class", DecisionTreeClassifier), ("entropy", "class", DecisionTreeClassifier), ("variance", "reg", DecisionTreeRegressor)]
TIE_FREE = {"class": [(2, 10), (3, 20), (4, 25)], "reg": [(3, 20), (4, 10), (6, 5), (None, 8)]}


def _fit_pair(criterion, task, cls, depth, leaf, n=500):
    X, y = _continuous(n, 6, 1, task)
    Xt, _ = _continuous(300, 6, 2, task)
    kw = {"criterion": criterion} if task == "class" else {"criterion": "squared_error"}
    return alg.grow(X, y, task, criterion, depth, leaf), cls(max_depth=depth, min_samples_leaf=leaf, random_state=0, **kw).fit(X, y), Xt


@pytest.mark.parametrize("criterion,task,cls", CASES)
def test_tree_equals_scikit_learn_where_no_splits_tie(criterion, task, cls):
    """Gleiche Bäume, Vorhersagen und Blattwerte. Ausgenommen sind Bäume mit sehr kleinen Blättern: dort gibt es echte Gleichstände (ein Knoten mit zwei Beispielen lässt sich mit jedem Merkmal gleich gut teilen),
    und scikit-learn wählt dann zufällig, wir das kleinste Merkmal."""
    for depth, leaf in TIE_FREE[task]:
        ours, ref, Xt = _fit_pair(criterion, task, cls, depth, leaf)
        assert ours.n_leaves == ref.get_n_leaves() and ours.max_depth == ref.get_depth(), (depth, leaf)
        assert np.allclose(alg.predict(ours, Xt), ref.predict(Xt)), (depth, leaf)
        assert np.allclose(alg.predict_value(ours, Xt), ref.predict_proba(Xt)[:, 1] if task == "class" else ref.predict(Xt)), (depth, leaf)


@pytest.mark.parametrize("criterion,task,cls", CASES[:2])
def test_deep_classification_trees_differ_from_scikit_learn_only_through_ties(criterion, task, cls):
    for depth, leaf in [(6, 5), (None, 15), (None, 3)]:
        ours, ref, Xt = _fit_pair(criterion, task, cls, depth, leaf)
        assert abs(ours.n_leaves - ref.get_n_leaves()) <= max(2, 0.1 * ref.get_n_leaves()), (depth, leaf)
        assert np.mean(alg.predict(ours, Xt) == ref.predict(Xt)) > 0.9, (depth, leaf)


@pytest.mark.parametrize("criterion,task,cls", CASES)
def test_importances_formula_equals_scikit_learn_on_the_same_tree(criterion, task, cls):
    """Die Wichtigkeit auf scikit-learns eigenem Baum: unabhängig davon, wie Gleichstände beim Wachsen ausgehen."""
    for depth, leaf in [(None, 1), (6, 5), (3, 20)]:
        X, y = _continuous(500, 6, 1, task)
        kw = {"criterion": criterion} if task == "class" else {"criterion": "squared_error"}
        ref = cls(max_depth=depth, min_samples_leaf=leaf, random_state=0, **kw).fit(X, y)
        assert np.allclose(alg.importances(_from_sklearn(ref, task, 500)), ref.feature_importances_, atol=1e-12), (depth, leaf)


def test_scikit_learn_agreement_on_the_delivery_data():
    """Auf den Lieferdaten (mit ganzzahligen Merkmalen) stimmen die Blattwerte überein, solange die Blätter groß genug sind, dass keine Gleichstände auftreten (Seed 7)."""
    ds = S.generate_dataset(800, 3, 10, 7)
    for task, cls, crit in (("class", DecisionTreeClassifier, "gini"), ("reg", DecisionTreeRegressor, "squared_error")):
        Xtr, ytr, Xte, _ = S.split(ds, task)
        for depth, leaf in ((3, 20), (4, 25)):
            ours = alg.grow(Xtr, ytr, task, None, depth, leaf)
            ref = cls(criterion=crit, max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(Xtr, ytr)
            assert np.allclose(alg.predict_value(ours, Xte), ref.predict_proba(Xte)[:, 1] if task == "class" else ref.predict(Xte)), (task, depth, leaf)


@pytest.mark.parametrize("task", ["class", "reg"])
def test_cost_complexity_pruning_equals_scikit_learn(task):
    """Der Pfad der alpha und die beschnittenen Bäume stimmen mit scikit-learn überein; die Bäume kommen von scikit-learn selbst, damit Gleichstände beim Wachsen nichts verdecken."""
    X, y = _continuous(300, 5, 5, task, noise=0.6)
    cls = DecisionTreeClassifier if task == "class" else DecisionTreeRegressor
    crit = "gini" if task == "class" else "squared_error"
    path = cls(criterion=crit, random_state=0).cost_complexity_pruning_path(X, y)
    full = _from_sklearn(cls(criterion=crit, random_state=0).fit(X, y), task, len(y))
    mine = alg.pruning_path(full)
    assert np.allclose([a for a, *_ in mine], path.ccp_alphas, atol=1e-12) and np.allclose([r for _, _, r, _ in mine], path.impurities, atol=1e-12)
    Xt, _ = _continuous(200, 5, 6, task, noise=0.6)
    for k in (1, len(path.ccp_alphas) // 3, len(path.ccp_alphas) // 2, len(path.ccp_alphas) - 2):
        alpha = float(path.ccp_alphas[k]) * 1.0000001
        pruned = alg.prune(full, alpha)
        ref = cls(criterion=crit, random_state=0, ccp_alpha=alpha).fit(X, y)
        assert pruned.n_leaves == ref.get_n_leaves(), k
        assert np.allclose(alg.predict_value(pruned, Xt), ref.predict_proba(Xt)[:, 1] if task == "class" else ref.predict(Xt)), k


def test_pruning_our_own_regression_tree_equals_scikit_learn():
    X, y = _continuous(300, 5, 9, "reg")
    ours = alg.grow(X, y, "reg", None, None, 1)
    path = DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X, y)
    assert np.allclose([a for a, *_ in alg.pruning_path(ours)], path.ccp_alphas, atol=1e-12)


def test_pruning_path_properties():
    X, y = _continuous(300, 5, 8, "class")
    full = alg.grow(X, y, "class", None, None, 1)
    path = alg.pruning_path(full)
    alphas = [a for a, *_ in path]
    leaves = [l for _, l, _, _ in path]
    total = [r for _, _, r, _ in path]
    assert alphas[0] == 0.0 and alphas == sorted(alphas) and leaves[-1] == 1 and leaves[0] == full.n_leaves
    assert all(l1 < l0 for l0, l1 in zip(leaves, leaves[1:])) and total == sorted(total)
    assert alg.prune(full, 0.0).n_leaves == full.n_leaves and alg.prune(full, 1e9).n_leaves == 1


# --- Grenzfälle ---------------------------------------------------------------------------------------------------------------------------------

def test_depth_zero_is_a_single_leaf_with_the_mean():
    X, y = _continuous(100, 4, 0, "reg")
    t = alg.grow(X, y, "reg", None, 0, 1)
    assert t.n_nodes == 1 and t.n_leaves == 1 and t.value[0] == pytest.approx(y.mean()) and np.all(alg.predict(t, X) == pytest.approx(y.mean()))
    assert alg.importances(t).sum() == 0


def test_min_leaf_equal_to_n_and_pure_nodes_are_not_split():
    X, y = _continuous(40, 4, 0, "class")
    assert alg.grow(X, y, "class", None, None, 40).n_nodes == 1
    assert alg.grow(X, y, "class", None, None, 21).n_nodes == 1                # n < 2 * min_leaf: kein zulässiger Split
    assert alg.grow(X, np.ones(40), "class", None, None, 1).n_nodes == 1        # reiner Knoten


def test_a_constant_feature_is_never_used_and_all_constant_gives_a_leaf():
    X, y = _continuous(200, 4, 1, "class")
    X[:, 0] = 5.0
    t = alg.grow(X, y, "class", None, None, 1)
    assert 0 not in set(t.feature[t.feature >= 0]) and alg.importances(t)[0] == 0
    assert alg.grow(np.full((30, 2), 3.0), (np.arange(30) % 2).astype(float), "class", None, None, 1).n_nodes == 1


def test_full_tree_fits_the_training_data_when_no_two_points_collide():
    X, y = _continuous(300, 4, 3, "class")
    t = alg.grow(X, y, "class", None, None, 1)
    assert np.array_equal(alg.predict(t, X), y.astype(int)) and t.n[t.feature < 0].sum() == 300
    assert np.all(t.impurity[t.feature < 0] < 1e-12)


def test_structure_invariants_and_breadth_first_order():
    X, y = _continuous(300, 4, 3, "class")
    t = alg.grow(X, y, "class", None, 5, 3)
    inner = t.internal_nodes()
    assert np.all(t.n[t.left[inner]] + t.n[t.right[inner]] == t.n[inner]) and np.all(t.depth[t.left[inner]] == t.depth[inner] + 1)
    assert np.all(np.diff(t.depth) >= 0)                                        # Breitenreihenfolge: Tiefe wächst mit der Nummer
    assert np.all(t.n[t.feature < 0] >= 3) and t.n[0] == 300 and t.value[0] == pytest.approx(y.mean())
    assert np.array_equal(np.sort(alg.apply(t, X)), np.sort(np.repeat(np.nonzero(t.feature < 0)[0], t.n[t.feature < 0])))


def test_tree_after_k_splits_grows_one_split_at_a_time():
    X, y = _continuous(300, 4, 3, "class")
    t = alg.grow(X, y, "class", None, 4, 3)
    prev = 0
    for k in range(0, alg.n_splits(t) + 1):
        tk = alg.tree_after_splits(t, k)
        assert alg.n_splits(tk) == k and tk.n_leaves == k + 1
        if k:
            assert tk.n_leaves == prev + 1
        prev = tk.n_leaves
    full = alg.tree_after_splits(t, alg.n_splits(t))
    assert np.array_equal(alg.predict_value(full, X), alg.predict_value(t, X))
    assert alg.tree_after_splits(t, 0).n_nodes == 1


def test_decision_path_agrees_with_apply():
    X, y = _continuous(200, 4, 3, "class")
    t = alg.grow(X, y, "class", None, 5, 2)
    leaf = alg.apply(t, X)
    assert all(alg.decision_path(t, X[i])[-1] == leaf[i] for i in range(0, 200, 7))


# --- Gütemaße ------------------------------------------------------------------------------------------------------------------------------------

def test_metrics_against_scikit_learn():
    from sklearn import metrics
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(int)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1)
    p[:20] = 0.5                                                                # Gleichstände
    assert alg.auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p)) and alg.log_loss(y, p) == pytest.approx(metrics.log_loss(y, p))
    assert alg.accuracy(y, p > 0.5) == pytest.approx(metrics.accuracy_score(y, p > 0.5))
    z = rng.normal(size=300) * 4 + 30
    q = z + rng.normal(size=300) * 2
    assert alg.rmse(z, q) == pytest.approx(metrics.mean_squared_error(z, q) ** 0.5) and alg.mae(z, q) == pytest.approx(metrics.mean_absolute_error(z, q)) and alg.r2(z, q) == pytest.approx(metrics.r2_score(z, q))
