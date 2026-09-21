"""Messungen an CART: Fehler gegen Tiefe, Beschneiden, Instabilität gegen kleine Datenänderungen, Anteil der Schnitte auf Rauschmerkmalen."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

import cart_algorithm as alg
import cart_constants as C
import cart_scenario as S


def fit(ds, task, criterion=None, max_depth=None, min_leaf=1):
    Xtr, ytr, _, _ = S.split(ds, task)
    return alg.grow(Xtr, ytr, task, criterion, max_depth, min_leaf)


def error_of(tree, ds, part):
    Xtr, ytr, Xte, yte = S.split(ds, tree.task)
    return alg.scores(tree, *((Xtr, ytr) if part == "train" else (Xte, yte)))["error"]


def baseline_error(ds, task):
    """Fehler ohne Baum: Klassifikation = immer die häufigere Klasse des Trainings raten, Regression = immer der Trainings-Mittelwert."""
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def depth_rows(ds, task, criterion=None, min_leaf=1, depths=range(0, C.DEPTH_MAX + 1)):
    """Für jede Tiefe: Blätter, Trainings- und Testfehler."""
    rows = []
    for d in depths:
        t = fit(ds, task, criterion, d, min_leaf)
        rows.append({"depth": d, "leaves": t.n_leaves, "train": error_of(t, ds, "train"), "test": error_of(t, ds, "test")})
    return rows


def pruning_rows(ds, task, criterion=None, min_leaf=1, max_depth=None, max_rows=60):
    """Der Beschneidungspfad des (voll gewachsenen) Baums: je Schritt alpha, Blätter, Trainings- und Testfehler (höchstens `max_rows` gleichmäßig verteilte Schritte)."""
    full = fit(ds, task, criterion, max_depth, min_leaf)
    path = alg.pruning_path(full)
    pick = np.unique(np.linspace(0, len(path) - 1, min(max_rows, len(path))).round().astype(int))
    rows = []
    for i in pick:
        t = alg.prune_to_leaves(full, path[i][1], path)
        rows.append({"alpha": path[i][0], "leaves": t.n_leaves, "train": error_of(t, ds, "train"), "test": error_of(t, ds, "test")})
    return rows


def best_pruned(rows):
    """Zeile mit dem kleinsten Testfehler; bei Gleichstand der kleinere Baum."""
    return min(rows, key=lambda r: (r["test"], r["leaves"]))


def fingerprint(tree, levels=2):
    """Die Merkmale der ersten Ebenen als Text (z. B. 'Distanz | Verkehr Wetter'): dieselbe Zeichenkette = derselbe Aufbau oben im Baum."""
    parts = []
    for lv in range(levels):
        ids = [t for t in range(tree.n_nodes) if tree.depth[t] == lv and tree.feature[t] >= 0]
        parts.append(" ".join(str(int(tree.feature[t])) for t in ids))
    return " | ".join(parts)


def _disagreement(preds, task):
    """Mittlere paarweise Abweichung der Vorhersagen: Anteil verschiedener Klassen bzw. Wurzel des mittleren quadratischen Unterschieds."""
    vals = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            vals.append(np.mean(preds[i] != preds[j]) if task == "class" else np.sqrt(np.mean((preds[i] - preds[j]) ** 2)))
    return float(np.mean(vals))


def instability(ds, task, criterion=None, max_depth=None, min_leaf=1, n_boot=C.BOOTSTRAPS, seed=0, drop=0.05):
    """Wie sehr hängt der Baum an den Trainingsdaten? Zwei Störungen: Bootstrap-Stichproben (mit Zurücklegen) und das Weglassen von `drop` der Zeilen.
    Gemessen wird: Anteil der Bäume mit dem häufigsten Wurzelmerkmal, Zahl verschiedener Aufbauten der ersten zwei Ebenen, mittlere Abweichung der Testvorhersagen (paarweise) und die Streuung der Testfehler."""
    Xtr, ytr, Xte, yte = S.split(ds, task)
    rng = np.random.default_rng(seed)
    out = {}
    for label, draw in (("bootstrap", lambda: rng.integers(0, len(ytr), len(ytr))), ("drop", lambda: np.sort(rng.permutation(len(ytr))[: int(round(len(ytr) * (1.0 - drop)))]))):
        trees = [alg.grow(Xtr[idx], ytr[idx], task, criterion, max_depth, min_leaf) for idx in (draw() for _ in range(n_boot))]
        roots = Counter(int(t.feature[0]) for t in trees)
        prints = Counter(fingerprint(t) for t in trees)
        preds = [alg.predict(t, Xte) for t in trees]
        errs = [alg.scores(t, Xte, yte)["error"] for t in trees]
        out[label] = {"root_share": roots.most_common(1)[0][1] / n_boot, "root_feature": roots.most_common(1)[0][0], "n_roots": len(roots), "n_prints": len(prints),
                      "print_share": prints.most_common(1)[0][1] / n_boot, "disagree": _disagreement(preds, task), "err_mean": float(np.mean(errs)), "err_sd": float(np.std(errs)),
                      "leaves_mean": float(np.mean([t.n_leaves for t in trees])), "roots": roots}
    return out


def split_shares(tree, n_real=C.N_BASE):
    """Anteil der Schnitte auf Rauschmerkmalen (Spalten ab `n_real`), insgesamt und je Ebene [(Ebene, Schnitte, Anteil Rauschen)]."""
    inner = tree.internal_nodes()
    noisy = tree.feature[inner] >= n_real
    by_level = []
    for lv in range(int(tree.depth.max()) + 1):
        m = tree.depth[inner] == lv
        if m.any():
            by_level.append((lv, int(m.sum()), float(noisy[m].mean())))
    return {"total": int(len(inner)), "noise": float(noisy.mean()) if len(inner) else 0.0, "by_level": by_level}


def cv_alpha(ds, task, criterion=None, max_depth=None, min_leaf=1, folds=5, seed=0):
    """Wählt alpha des Beschneidens ohne die Testdaten anzufassen: fünffache Kreuzvalidierung auf dem Training. Kandidaten sind die alpha des Pfads des Baums auf dem ganzen Training;
    gewählt wird das alpha mit dem kleinsten mittleren Fehler auf den zurückgehaltenen Teilen, bei Gleichstand das größere (der kleinere Baum). Rückgabe: (alpha, [(alpha, mittlerer Fehler)])."""
    Xtr, ytr, _, _ = S.split(ds, task)
    full = alg.grow(Xtr, ytr, task, criterion, max_depth, min_leaf)
    alphas = sorted({a for a, *_ in alg.pruning_path(full)})
    if len(alphas) > C.CV_GRID:                                                  # höchstens CV_GRID Kandidaten, gleichmäßig über den Pfad verteilt
        alphas = [alphas[i] for i in np.unique(np.linspace(0, len(alphas) - 1, C.CV_GRID).round().astype(int))]
    order = np.random.default_rng(seed).permutation(len(ytr))
    parts = np.array_split(order, folds)
    err = np.zeros((folds, len(alphas)))
    for k, val in enumerate(parts):
        fit_idx = np.setdiff1d(order, val)
        t = alg.grow(Xtr[fit_idx], ytr[fit_idx], task, criterion, max_depth, min_leaf)
        path = alg.pruning_path(t)
        for j, a in enumerate(alphas):
            err[k, j] = alg.scores(alg.prune(t, a, path), Xtr[val], ytr[val])["error"]
    mean = err.mean(axis=0)
    best = max(j for j in range(len(alphas)) if mean[j] <= mean.min() + 1e-12)
    return alphas[best], list(zip(alphas, mean.tolist()))


# --- Ein Lauf für die App -------------------------------------------------------------------------------------------------------------------------

@dataclass
class Analysis:
    ds: object
    task: str
    criterion: str
    leaf: int
    full: object                 # Baum vor dem Beschneiden
    path: list                   # Beschneidungspfad des vollen Baums
    tree: object                 # der gezeigte Baum (nach dem Beschneiden)
    prune_mode: str
    alpha: float                 # bei Kreuzvalidierung das gewählte alpha, sonst 0
    train: dict
    test: dict
    baseline: float
    verdict: str
    imp: np.ndarray


def analyse(task, criterion, depth, leaf, prune_mode, prune_leaves, n, n_noise, label_noise, seed):
    """Daten erzeugen, Baum wachsen lassen, beschneiden (aus | Kreuzvalidierung | von Hand auf höchstens `prune_leaves` Blätter), messen."""
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    criterion = criterion if criterion in C.CRITERIA[task] else C.DEFAULT_CRITERION[task]
    Xtr, ytr, Xte, yte = S.split(ds, task)
    full = alg.grow(Xtr, ytr, task, criterion, depth, leaf)
    path = alg.pruning_path(full)
    alpha = 0.0
    if prune_mode == "cv" and full.n_leaves > 1:
        alpha, _ = cv_alpha(ds, task, criterion, depth, leaf)
        tree = alg.prune(full, alpha, path)
    elif prune_mode == "manual":
        tree = alg.prune_to_leaves(full, prune_leaves, path)
    else:
        tree = full
    a = Analysis(ds, task, criterion, leaf, full, path, tree, prune_mode, alpha, alg.scores(tree, Xtr, ytr), alg.scores(tree, Xte, yte), baseline_error(ds, task), "", alg.importances(tree))
    a.verdict = verdict(a)
    return a


def verdict(a):
    """'stump' (ein Blatt), 'overfit' (Test viel schlechter als Training), 'underfit' (kaum besser als Raten), sonst 'ok'."""
    tr, te = a.train["error"], a.test["error"]
    if a.tree.n_leaves == 1:
        return "stump"
    over = (te - tr > C.OVERFIT_GAP_CLASS) if a.task == "class" else (te > C.OVERFIT_RATIO_REG * max(tr, 1e-9))
    if over:
        return "overfit"
    return "underfit" if te > C.UNDERFIT_SHARE * a.baseline else "ok"


def next_split(a, k):
    """Der Schnitt Nummer k + 1 (Breitenreihenfolge) des gezeigten Baums, so wie ihn die Suche sieht: Knoten, Zahl der Lieferungen, bester Gewinn und beste Schwelle je Merkmal, gewähltes Merkmal.
    None, wenn der Baum nach k Schnitten fertig ist."""
    tree = a.tree
    inner = tree.internal_nodes()
    if k >= len(inner):
        return None
    node = int(inner[k])
    Xtr, ytr, _, _ = S.split(a.ds, a.task)
    members = alg.apply(alg.tree_after_splits(tree, k), Xtr) == node
    Xm, ym = Xtr[members], ytr[members]
    gain, thr, _ = alg.gain_matrix(Xm, ym, a.criterion, a.leaf)
    j = gain.argmax(axis=0)
    cols = np.arange(gain.shape[1])
    return {"node": node, "n": int(members.sum()), "impurity": tree.impurity[node], "best_gain": gain[j, cols], "best_thr": thr[j, cols], "feature": int(tree.feature[node]), "threshold": float(tree.threshold[node]),
            "gain": gain, "thr": thr, "candidates": int(np.isfinite(gain).sum())}


STABILITY_CONFIGS = (("Tiefe 3", 3, 5), ("Tiefe 6", 6, 5), ("voll gewachsen", C.DEPTH_MAX, 1))


def stability_rows(ds, task, criterion=None):
    """Instabilität für drei Baumgrößen, je mit Bootstrap-Stichproben und 5 % weniger Zeilen."""
    rows = []
    for label, d, leaf in STABILITY_CONFIGS:
        r = instability(ds, task, criterion, d, leaf, C.BOOTSTRAPS, 0)
        rows.append({"label": label, "bootstrap": r["bootstrap"]["disagree"], "drop": r["drop"]["disagree"], "prints_bootstrap": r["bootstrap"]["n_prints"], "prints_drop": r["drop"]["n_prints"],
                     "err_sd": r["bootstrap"]["err_sd"], "leaves": r["bootstrap"]["leaves_mean"], "root_share": min(r["bootstrap"]["root_share"], r["drop"]["root_share"])})
    return rows


def set_rows(task, criterion, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS):
    """Voll gewachsen, flach (Tiefe 4, Blatt >= 5) und per Kreuzvalidierung beschnitten auf mehreren Datensätzen (nur der Seed ändert sich)."""
    rows = []
    for i, sd in enumerate(seeds):
        ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, sd)
        full = fit(ds, task, criterion, None, 1)
        alpha, _ = cv_alpha(ds, task, criterion, None, 1)
        pruned = alg.prune(full, alpha)
        rows.append({"label": f"Datensatz {i + 1}", "seed": sd, "full": error_of(full, ds, "test"), "shallow": error_of(fit(ds, task, criterion, 4, 5), ds, "test"), "pruned": error_of(pruned, ds, "test"),
                     "full_leaves": full.n_leaves, "pruned_leaves": pruned.n_leaves})
    return rows
