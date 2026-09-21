"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Die Daten kommen aus festen Seeds und ihre Bäume aus reiner numpy-Rechnung; Fehlerquoten sind exakte Brüche
und stehen fest. Wo etwas von Gleichständen abhängen könnte (Reihenfolge gleicher Schnitte), steht ein Rand. Laufzeiten stehen in der App nicht und werden nie geprüft."""

import functools

import numpy as np
import pytest

import cart_algorithm as alg
import cart_constants as C
import cart_evaluation as ev
import cart_scenario as S

PRESET = {"flat": "🌳 Flacher Baum", "full": "🌲 Voll gewachsen", "pruned": "✂️ Beschneiden", "noisy": "🏷️ Falsche Etiketten", "reg": "📈 Regression"}
SIX = (7,) + C.SWEEP_SEEDS                      # "sechs Datensätze"


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["criterion"], p["depth"], p["leaf"], p["prune"], p["prune_leaves"], p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _ds(seed, n_noise=3, label_noise=0, n=1200):
    return S.generate_dataset(n, n_noise, label_noise, seed)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_flat_preset():
    a = _preset("flat")
    assert (a.tree.n_leaves, a.tree.max_depth, a.verdict) == (8, 3, "ok")
    assert (a.train["error"], a.test["error"], a.baseline) == pytest.approx((0.158, 0.172, 0.464), abs=0.0005)
    assert a.tree.feature[0] == 1 and a.tree.threshold[0] == pytest.approx(798, abs=1) and a.ds.names[a.tree.feature[0]] == "Ladegewicht"
    assert [a.ds.names[f] for f in a.tree.feature[1:3]] == ["Verkehr", "Verkehr"] and a.test["auc"] == pytest.approx(0.882, abs=0.0005) and a.test["logloss"] == pytest.approx(0.416, abs=0.0005)
    assert "Distanz" in {a.ds.names[f] for f in np.argsort(-a.imp)[:3]} and a.ds.names[int(np.argsort(-a.imp)[1])] == "Verkehr"
    _help("flat", "8 Blätter", "15.8 %", "17.2 %", "46.4 %", "798 kg")


def test_full_preset_overfits_but_accuracy_looks_fine():
    a, flat = _preset("full"), _preset("flat")
    assert (a.tree.n_leaves, a.verdict, a.train["error"]) == (100, "overfit", 0.0)
    assert a.test["error"] == pytest.approx(0.167, abs=0.0005) and a.test["accuracy"] == pytest.approx(0.833, abs=0.0005)
    assert abs(a.test["error"] - flat.test["error"]) < 0.01                                        # Fehlerquote: kaum anders als der flache Baum
    assert (a.test["auc"], flat.test["auc"]) == pytest.approx((0.830, 0.882), abs=0.0005) and (a.test["logloss"], flat.test["logloss"]) == pytest.approx((5.76, 0.416), abs=0.005)
    assert a.test["logloss"] > 2                                                                    # die App zeigt den Log-Loss-Hinweis ab 2
    _help("full", "100 Blätter", "0 %", "16.7 %", "0.830", "0.882", "5.76", "0.42")


def test_pruned_preset():
    a, full = _preset("pruned"), _preset("full")
    assert (a.tree.n_leaves, a.full.n_leaves, a.verdict) == (17, 100, "ok") and a.alpha == pytest.approx(0.004, abs=0.0005)
    assert (a.train["error"], a.test["error"]) == pytest.approx((0.104, 0.164), abs=0.0005) and a.test["auc"] == pytest.approx(0.890, abs=0.0005) and a.test["logloss"] == pytest.approx(0.58, abs=0.005)
    assert a.test["error"] <= full.test["error"] and a.test["logloss"] < full.test["logloss"] / 5
    _help("pruned", "17 Blätter", "0.004", "10.4 %", "16.4 %", "0.890", "0.58")


def test_noisy_labels_preset():
    a, full = _preset("noisy"), _preset("full")
    assert (a.tree.n_leaves, a.verdict, a.train["error"]) == (125, "overfit", 0.0) and a.test["error"] == pytest.approx(0.236, abs=0.0005)
    assert a.test["error"] > full.test["error"] + 0.05
    flat = ev.analyse("class", "gini", 4, 5, "off", 12, 1200, 3, 10, 7)
    assert flat.test["error"] < a.test["error"] - 0.05                                              # "ein flacher Baum wäre kaum betroffen"
    _help("noisy", "10 %", "125 Blätter", "23.6 %", "16.7 %")


def test_regression_preset():
    a = _preset("reg")
    assert (a.tree.n_leaves, a.tree.max_depth, a.verdict) == (16, 4, "ok")
    assert (a.train["error"], a.test["error"], a.baseline, a.test["r2"]) == pytest.approx((12.26, 14.38, 27.15, 0.719), abs=0.005)
    top = int(np.argmax(a.imp))
    assert a.ds.names[top] == "Distanz" and a.imp[top] == pytest.approx(0.655, abs=0.0005)
    _help("reg", "16 Blätter", "12.3 min", "14.4 min", "27.2 min", "0.719", "65.5 %")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["criterion"] in C.CRITERIA[p["task"]] and C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX and C.LEAF_MIN <= p["leaf"] <= C.LEAF_MAX
        assert C.N_MIN <= p["n"] <= C.N_MAX and p["prune"] in C.PRUNE_MODES and 0 <= p["fx"] < C.N_BASE + p["n_noise"] and 0 <= p["fy"] < C.N_BASE + p["n_noise"] and name in C.PRESET_HELP


# --- Sidebar-Hilfen -------------------------------------------------------------------------------------------------------------------------------

def test_depth_help_numbers():
    flat, full = _preset("flat"), _preset("full")
    assert (flat.tree.n_leaves, full.tree.n_leaves) == (8, 100) and (flat.train["error"], flat.test["error"], full.train["error"], full.test["error"]) == pytest.approx((0.158, 0.172, 0.0, 0.167), abs=0.0005)


def test_criterion_help_numbers():
    same, agree = 0, []
    for sd in SIX:
        ds = _ds(sd)
        g, e = ev.fit(ds, "class", "gini", 4, 5), ev.fit(ds, "class", "entropy", 4, 5)
        same += int(np.array_equal(g.feature[:7], e.feature[:7]))
        Xte = ds.X[ds.test]
        agree.append(np.mean(alg.predict(g, Xte) == alg.predict(e, Xte)))
    assert same == 2 and min(agree) == pytest.approx(0.92, abs=0.005) and max(agree) == pytest.approx(0.98, abs=0.005)


def test_label_noise_help_numbers():
    full, shallow = [], []
    for ln in (0, 10, 20):
        f, s = [], []
        for sd in SIX:
            ds = _ds(sd, 3, ln)
            f.append(ev.error_of(ev.fit(ds, "class", None, None, 1), ds, "test"))
            s.append(ev.error_of(ev.fit(ds, "class", None, 4, 5), ds, "test"))
        full.append(np.mean(f))
        shallow.append(np.mean(s))
    assert full == pytest.approx([0.204, 0.254, 0.350], abs=0.0006) and shallow == pytest.approx([0.171, 0.171, 0.218], abs=0.0006)


def test_noise_feature_help_numbers():
    share, first4 = {}, {}
    for nn in (3, 8):
        shares, f4 = [], 0
        for sd in SIX:
            r = ev.split_shares(ev.fit(_ds(sd, nn), "class", None, None, 1))
            shares.append(r["noise"])
            f4 += sum(int(round(n * s)) for lv, n, s in r["by_level"] if lv < 4)
        share[nn], first4[nn] = np.mean(shares), f4
    assert (share[3], share[8]) == pytest.approx((0.16, 0.29), abs=0.005) and (first4[3], first4[8]) == (5, 8)
    assert ev.split_shares(ev.fit(_ds(7, 0), "class", None, None, 1))["noise"] == 0.0


# --- Texte der Grafiken und Experimente -------------------------------------------------------------------------------------------------------------

def test_readme_depth_gap_numbers():
    """README: der Testfehler steigt mit der Tiefe nur mäßig (voller Baum gegen die am Test gewählte beste Tiefe, Mittel über sechs Datensätze)."""
    for task, full, best, lo, hi in (("class", 0.204, 0.155, 3, 6), ("reg", 14.6, 13.3, 5, 8)):
        rows_all = [ev.depth_rows(_ds(sd), task, None, 1) for sd in SIX]
        te = [[r["test"] for r in rows] for rows in rows_all]
        assert np.mean([t[-1] for t in te]) == pytest.approx(full, abs=0.0006 if task == "class" else 0.05)
        assert np.mean([min(t) for t in te]) == pytest.approx(best, abs=0.0006 if task == "class" else 0.05)
        assert lo <= min(int(np.argmin(t)) for t in te) and max(int(np.argmin(t)) for t in te) <= hi


def test_pruning_curve_sentence():
    rows = ev.pruning_rows(_ds(7), "class", "gini", 1, None)
    best = ev.best_pruned(rows)
    assert best["leaves"] == 34 and best["test"] == pytest.approx(0.139, abs=0.0005) and rows[0]["leaves"] == 100 and rows[0]["test"] == pytest.approx(0.167, abs=0.0005)
    assert best["test"] < rows[0]["test"] and best["test"] < rows[-1]["test"] and 1 < best["leaves"] < 100


def test_instability_sentences():
    """'Ein Baum der Tiefe 3 ist stabiler' und 'die Wurzel ist stabil': in allen sechs Datensätzen, für Klassifikation (Gini, Entropie) und Regression."""
    for task, crit in (("class", "gini"), ("class", "entropy"), ("reg", "variance")):
        for sd in SIX:
            rows = ev.stability_rows(_ds(sd), task, crit)
            assert rows[0]["bootstrap"] < rows[-1]["bootstrap"] and rows[0]["drop"] < rows[-1]["drop"] and all(r["root_share"] == 1.0 for r in rows), (task, crit, sd)
            assert rows[-1]["drop"] < rows[-1]["bootstrap"] and rows[0]["prints_bootstrap"] >= 2, (task, crit, sd)


def test_full_tree_disagreement_is_large():
    """Voll gewachsen: bei einem großen Teil der Testlieferungen andere Vorhersage (Klassifikation: mehr als jede zehnte, Bootstrap mehr als jede sechste)."""
    for sd in SIX:
        r = ev.stability_rows(_ds(sd), "class", "gini")[-1]
        assert r["bootstrap"] > 0.16 and r["drop"] > 0.08, sd


def test_cross_validated_pruning_helps_on_average():
    rows = ev.set_rows("class", "gini", 1200, 3, 0)
    assert np.mean([r["pruned"] for r in rows]) < np.mean([r["full"] for r in rows]) - 0.02 and sum(r["pruned"] <= r["full"] for r in rows) >= 4
    assert all(r["pruned_leaves"] < r["full_leaves"] for r in rows)
    reg = ev.set_rows("reg", "variance", 1200, 3, 0)
    assert np.mean([r["pruned"] for r in reg]) < np.mean([r["full"] for r in reg])


def test_pruning_selection_never_touches_the_test_data():
    """Die Kreuzvalidierung sieht nur das Training: ändert man die Testetiketten, bleibt alpha gleich."""
    ds = _ds(7)
    a1, _ = ev.cv_alpha(ds, "class", "gini", None, 1)
    changed = S.Dataset(ds.X, ds.y_reg, ds.y_cls, 1 - ds.y_true, ds.names, ds.train, ds.test, ds.n_noise, ds.label_noise, ds.seed)
    a2, _ = ev.cv_alpha(changed, "class", "gini", None, 1)
    assert a1 == a2


# --- Erzeuger ---------------------------------------------------------------------------------------------------------------------------------------

def test_generator_is_deterministic_and_appends_late_columns_last():
    a, b = S.generate_dataset(500, 3, 0, 7), S.generate_dataset(500, 3, 0, 7)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y_reg, b.y_reg)
    c = S.generate_dataset(500, 8, 0, 7)                                                            # mehr Rauschmerkmale: die echten Spalten und die Dauer bleiben
    assert np.array_equal(a.X[:, :11], c.X[:, :11]) and np.array_equal(a.y_reg, c.y_reg) and np.array_equal(a.y_true, c.y_true)
    d = S.generate_dataset(500, 3, 15, 7)                                                           # Etiketten-Rauschen ändert nur die Trainingsetiketten
    assert np.array_equal(a.X, d.X) and np.array_equal(a.y_true, d.y_true) and 0.10 < np.mean(d.y_cls != d.y_true) < 0.20


def test_test_labels_are_clean():
    ds = _ds(7, 3, 20)
    _, ytr, _, yte = S.split(ds, "class")
    assert np.array_equal(yte, ds.y_true[ds.test]) and np.mean(ytr != ds.y_true[ds.train]) > 0.1


def test_split_is_70_30_and_disjoint():
    ds = _ds(7)
    assert (len(ds.train), len(ds.test)) == (840, 360) and not set(ds.train) & set(ds.test) and len(set(ds.train) | set(ds.test)) == ds.n


def test_default_dataset_facts():
    ds = _ds(7)
    assert ds.X.shape == (1200, 11) and ds.y_cls.mean() == pytest.approx(0.43, abs=0.005) and ds.names[:2] == ("Distanz", "Ladegewicht")
