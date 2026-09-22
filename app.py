"""CART - der Entscheidungsbaum - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - CART, den Entscheidungsbaum - und lässt stattdessen das Beispiel wachsen.
Erstes Stück der Baumbasierten Linie der "Konzepte"-Reihe und die erste Demo des Portfolios mit überwachtem Lernen: es gibt Beispiele mit bekanntem Ergebnis, aus denen ein Modell lernt.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import cart_algorithm as alg
import cart_constants as C
import cart_evaluation as ev
from cart_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cart_visualization import (
    build_depth_curve,
    build_gain_curve,
    build_importance,
    build_instability,
    build_map,
    build_pruning_curve,
    build_sets,
    build_split_search,
    build_tree,
    feature_label,
    value_text,
)

st.set_page_config(page_title="CART – Sebastian Hanisch", layout="wide")


def _pct(x):
    return f"{x * 100:.1f} %"


def _err(task, x):
    return _pct(x) if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=24)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _depth_rows(task, criterion, leaf, n, n_noise, label_noise, seed):
    ds = _analysis(task, criterion, C.DEPTH_MAX, leaf, "off", C.DEFAULT_PRUNE_LEAVES, n, n_noise, label_noise, seed).ds
    return ev.depth_rows(ds, task, criterion, leaf)


@st.cache_data(show_spinner=False, max_entries=8)
def _pruning_rows(task, criterion, depth, leaf, n, n_noise, label_noise, seed):
    ds = _analysis(task, criterion, depth, leaf, "off", C.DEFAULT_PRUNE_LEAVES, n, n_noise, label_noise, seed).ds
    return ev.pruning_rows(ds, task, criterion, leaf, depth)


@st.cache_data(show_spinner=False, max_entries=8)
def _stability(task, criterion, n, n_noise, label_noise, seed):
    ds = _analysis(task, criterion, C.DEPTH_MAX, 1, "off", C.DEFAULT_PRUNE_LEAVES, n, n_noise, label_noise, seed).ds
    return ev.stability_rows(ds, task, criterion)


@st.cache_data(show_spinner=False, max_entries=8)
def _sets(task, criterion, n, n_noise, label_noise):
    return ev.set_rows(task, criterion, n, n_noise, label_noise)


st.title("🌳 CART – der Entscheidungsbaum")
st.markdown(
    """
Aus vielen Lieferungen, von denen man weiß, ob sie pünktlich waren (oder wie lange sie gedauert haben), soll ein Modell lernen, das das für **neue** Lieferungen vorhersagt. Ein **Entscheidungsbaum** tut das mit lauter Ja-Nein-Fragen an ein einzelnes Merkmal ("Ladegewicht ≤ 800 kg?"): jede Antwort schickt die Lieferung nach links oder rechts, am Ende steht ein **Blatt** mit der Vorhersage.
**CART** (Classification And Regression Trees) baut den Baum **gierig**: an jedem Knoten probiert es jede Schwelle jedes Merkmals aus und nimmt den Split, der die Lieferungen am besten sortiert - gemessen an der **Unreinheit** (Gini oder Entropie bei Klassen, Varianz bei Zahlen). Was danach kommt, sieht es nicht.
Das ist schnell und lesbar, hat aber zwei bekannte Schwächen, die die ganze Linie antreibt: ein Baum, der weiterwächst, **lernt das Rauschen mit** (Überanpassung), und **eine kleine Änderung der Daten kann einen ganz anderen Baum ergeben** (Instabilität).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - erstes Stück der Baumbasierten Linie der \"Konzepte\"-Reihe und **erste Demo mit überwachtem Lernen** (bisher: Kombinatorik, Graphen, Clustering, Anomalieerkennung) - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Breiman, Friedman, Olshen und Stone (1984) zurück; alle Lieferungen, Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten."
)
st.caption(
    "**Bezug zu OR:** Die vorhergesagte Lieferdauer ist Eingabe der Planung - für Zeitfenster und Touren (VRP mit Zeitfenstern) - und ein Baum liefert dazu Regeln, die man Disponenten zeigen kann. Lernen und Optimieren ergänzen sich; diese Demo zeigt den Lernteil für sich."
)

with st.expander("So funktioniert CART", expanded=True):
    st.markdown(
        """
1. **Wurzel:** alle Trainingslieferungen (70 % der Daten; die übrigen 30 % sind der **Test**, den das Verfahren nie sieht). Der Knoten hat eine Unreinheit $I$: Gini $2p(1-p)$ oder Entropie bei zwei Klassen, Varianz bei Zahlen.
2. **Split-Suche:** für jedes Merkmal werden die Lieferungen sortiert; jede Mitte zwischen zwei benachbarten Werten ist eine Schwelle. Mit kumulativen Summen ist der **Gain** (Unreinheit vorher minus gewichtete Unreinheit der beiden Kinder) für alle Schwellen in einem Zug bekannt. Der beste Split gewinnt.
3. **Rekursion:** dasselbe in den beiden Kindern, Ebene für Ebene, bis ein Knoten rein ist, zu wenige Lieferungen hat (**Mindestblattgröße**) oder die **Tiefe** erreicht ist. Der Blattwert ist der Anteil "zu spät" bzw. der Mittelwert der Dauer.
4. **Beschneiden:** ein voll gewachsener Baum passt das Rauschen an. **Kosten-Komplexität** kappt nacheinander den Teilbaum, dessen Wegfall den Trainingsfehler pro gesparte Blätter am wenigsten erhöht - es entsteht eine Kette immer kleinerer Bäume, aus der man mit **Kreuzvalidierung** (nur auf dem Training) den besten wählt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
if st.session_state["criterion_select"] not in ("gini", "entropy"):
    st.session_state["criterion_select"] = "gini"

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k],
                        help="Klassifikation: das Blatt sagt, mit welcher Wahrscheinlichkeit die Lieferung zu spät kommt. Regression: das Blatt sagt die Dauer in Minuten. Dieselben Lieferungen und Merkmale, nur das Ziel wechselt; es ändern sich Split-Kriterium und Blattwert, sonst nichts.")
    if task == "class":
        crit = st.selectbox("Split-Kriterium", C.CRITERIA["class"], key="criterion_select", format_func=lambda k: C.CRITERION_LABELS[k],
                            help="Gini (2p(1-p)) und Entropie messen beide, wie gemischt ein Knoten ist; sie liegen fast immer beieinander. Auf sechs Datensätzen wählen sie in 2 dieselben Merkmale für die ersten sieben Splits, die Vorhersagen (Tiefe 4, Blatt ≥ 5) stimmen auf den Testlieferungen zu 92 bis 98 % überein.")
        st.session_state[KEPT["criterion_select"]] = crit
    else:
        crit = "variance"
        st.caption("Split-Kriterium: Varianz (Summe der Fehlerquadrate) - bei einem Zahlenziel gibt es keine Wahl.")
    depth = st.slider("Maximale Tiefe", *bounds("depth_slider"), key="depth_slider",
                      help="Höchstzahl der Ebenen unter der Wurzel. Klassifikation im Standarddatensatz: Tiefe 3 -> 8 Blätter, Trainingsfehler 15.8 %, Testfehler 17.2 %; ohne Grenze (Tiefe 16, Blatt 1) 100 Blätter, Trainingsfehler 0 %, Testfehler 16.7 % - der große Baum ist auf dem Training perfekt und auf neuen Lieferungen nicht besser.")
    leaf = st.slider("Mindestgröße eines Blatts", *bounds("leaf_slider"), key="leaf_slider",
                     help="Ein Split ist nur erlaubt, wenn beide Kinder mindestens so viele Lieferungen behalten. 1 = erlaubt Blätter mit einer einzigen Lieferung (das Rauschen wird mitgelernt); größere Werte halten den Baum klein und stabil.")
    prune_mode = st.selectbox("Beschneiden (Kosten-Komplexität)", C.PRUNE_MODES, key="prune_select", format_func=lambda k: C.PRUNE_LABELS[k],
                              help="Kappt den gewachsenen Baum von unten. 'alpha per Kreuzvalidierung' wählt die Stärke mit fünffacher Kreuzvalidierung auf den Trainingsdaten (die Testdaten bleiben unberührt); 'von Hand' lässt die Blätterzahl wählen.")
    prune_slot = st.container()
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100, help="Zahl der erzeugten Lieferungen; 70 % davon zum Lernen, 30 % zum Testen.")
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider",
                        help="Zusätzliche Merkmale ohne jeden Bezug zum Ziel (Zufallszahlen). Ein voll gewachsener Baum benutzt sie trotzdem - Klassifikation, Mittel über sechs Datensätze: mit 3 Rauschmerkmalen liegen 16 % seiner Splits auf Rauschen, mit 8 sind es 29 %; nahe der Wurzel kaum (in den ersten vier Ebenen zusammen 5 bzw. 8 Splits über alle sechs Datensätze).")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider",
                                help="Anteil der Trainingslieferungen, deren Etikett (pünktlich / zu spät) vertauscht ist; der Test bleibt sauber. Klassifikation, Mittel über sechs Datensätze: der voll gewachsene Baum verliert stark (Testfehler 20.4 % bei 0, 25.4 % bei 10, 35.0 % bei 20 % falschen Etiketten), ein flacher Baum (Tiefe 4, Blatt ≥ 5) kaum (17.1 %, 17.1 %, 21.8 %).")
        st.session_state[KEPT["label_noise_slider"]] = label_noise
    else:
        label_noise = int(st.session_state.get(KEPT["label_noise_slider"], C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Erzeugt einen anderen Datensatz mit denselben Regeln.")
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed.")

base_params = (task, crit, int(depth), int(leaf))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    full_a = _analysis(*base_params, "off", C.DEFAULT_PRUNE_LEAVES, *data_params)
full_leaves = full_a.full.n_leaves
with prune_slot:
    if prune_mode == "manual":
        if full_leaves > 2:
            pl = st.slider("Höchstens so viele Blätter", 2, full_leaves, key="prune_leaves_slider",
                           help="Der Baum auf dem Beschneidungspfad mit den meisten Blättern, die höchstens so viele sind wie hier gewählt. Ganz rechts: der volle Baum.")
            st.session_state[KEPT["prune_leaves_slider"]] = pl
        else:
            pl = 2
            st.caption("Der Baum hat höchstens zwei Blätter - es gibt nichts zu beschneiden.")
    else:
        pl = int(st.session_state.get(KEPT["prune_leaves_slider"], C.DEFAULT_PRUNE_LEAVES))
pl = int(min(max(pl, 2), max(full_leaves, 2)))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, prune_mode, pl if prune_mode == "manual" else C.DEFAULT_PRUNE_LEAVES, *data_params)
ds, tree = a.ds, a.tree
names = ds.names
n_feat = len(names)
n_test = len(ds.test)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f), help="Die Karte zeigt die Vorhersage des Baums über zwei Merkmale; alle anderen Merkmale stehen dabei auf ihrem Median im Training.")
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider", help="Eine Lieferung aus dem Test: ihr Weg durch den Baum ist orange, in der Karte steht sie als Stern.")
sync_query_params({"task_select": task, "criterion_select": crit if task == "class" else st.session_state.get(KEPT["criterion_select"], "gini"), "depth_slider": int(depth), "leaf_slider": int(leaf), "prune_select": prune_mode,
                   "prune_leaves_slider": int(pl), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

n_splits = alg.n_splits(tree)
view_key = (base_params, prune_mode, pl if prune_mode == "manual" else 0, data_params)
if st.session_state.get("cart_owner") != view_key:
    st.session_state["cart_owner"] = view_key
    st.session_state["cart_step"] = n_splits

# --- CART in Aktion --------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 CART in Aktion")
st.caption("Der Baum wächst Split für Split in der Reihenfolge, in der CART ihn baut (Ebene für Ebene). Links der Baum: **weiße Knoten** sind Splits, **Blätter** sind nach ihrem Wert gefärbt (Größe = Zahl der Lieferungen), ein **oranger Rand** markiert einen Split auf einem Rauschmerkmal, der **schwarze Ring** den nächsten Split. Rechts die Vorhersage über zwei Merkmale.")
if n_splits > 0:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Split", 0, n_splits, key="cart_step", help="Wie viele Splits schon gemacht sind: 0 = nur die Wurzel (ein Blatt mit dem Gesamtmittel), ganz rechts der fertige Baum.")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 0, False
    st.info("ℹ️ Der Baum besteht nur aus der Wurzel - ein Blatt. Größere Tiefe, kleinere Mindestblattgröße oder weniger Beschneiden lassen ihn wachsen.")
view_slot = st.empty()
sample_x = ds.X[ds.test][sample_idx]
sample_y = ds.y_true[ds.test][sample_idx] if task == "class" else ds.y_reg[ds.test][sample_idx]


def _render(current):
    tk = alg.tree_after_splits(tree, current) if current < n_splits else tree
    info = ev.next_split(a, current)
    path = alg.decision_path(tk, sample_x)
    with view_slot.container():
        c1, c2 = st.columns([5, 4])
        c1.plotly_chart(build_tree(tk, names, ds.y(task)[ds.train], next_node=info["node"] if info else None, path=path), width="stretch", key=f"tree_chart_{current}")
        c2.plotly_chart(build_map(tk, ds, fx, fy, sample_x), width="stretch", key=f"map_chart_{current}")
        leaf_node = path[-1]
        conditions = [f"{names[tk.feature[t]]} {'≤' if (tk.left[t] == path[i + 1]) else '>'} {tk.threshold[t]:.3g}" for i, t in enumerate(path[:-1])]
        pred = value_text(task, tk.value[leaf_node])
        truth = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.0f} min"
        st.markdown(f"**Testlieferung {sample_idx}** ({', '.join(f'{names[j]} {ds.X[ds.test][sample_idx][j]:.3g}' for j in range(C.N_BASE))}): "
                    + (f"Weg: {' → '.join(conditions)} → " if conditions else "") + f"Blatt mit **{pred}** ({tk.n[leaf_node]} Trainingslieferungen); tatsächlich: **{truth}**.")
        if info:
            st.markdown(f"**Split {current + 1} von {n_splits}:** Knoten mit {info['n']} Lieferungen, Unreinheit {info['impurity']:.3f}. {info['candidates']} zulässige Schwellen geprüft; gewählt: **{names[info['feature']]} ≤ {info['threshold']:.4g}** (Gain {info['best_gain'][info['feature']]:.4f}).")
            g1, g2 = st.columns(2)
            g1.plotly_chart(build_split_search(names, info["best_gain"], info["feature"]), width="stretch", key=f"search_chart_{current}")
            g2.plotly_chart(build_gain_curve(names, info["feature"], info["thr"][:, info["feature"]], info["gain"][:, info["feature"]], info["threshold"]), width="stretch", key=f"gain_chart_{current}")
        else:
            st.success("✅ Der Baum ist fertig: kein weiterer Split ist erlaubt oder nötig.")


if auto_play:
    for kk in range(n_splits + 1):
        _render(kk)
        time.sleep(min(0.9, 6.0 / (n_splits + 1)))
    step = n_splits
else:
    _render(step)

st.markdown("---")

# --- Was der Baum gelernt hat -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was der Baum gelernt hat – und wie gut er auf neuen Lieferungen ist")
st.caption("**Training** = die Lieferungen, aus denen der Baum gebaut wurde; **Test** = die zurückgehaltenen 30 %, die er nie gesehen hat. Nur der Testfehler sagt, wie gut der Baum auf neue Lieferungen passt. **Raten** ist der Fehler ohne Baum (immer die häufigere Klasse bzw. der Trainings-Mittelwert).")
tr, te = a.train, a.test
m1, m2, m3, m4 = st.columns(4)
m1.metric("Blätter / Tiefe", f"{tree.n_leaves} / {tree.max_depth}", delta=f"{full_a.full.n_leaves} vor dem Beschneiden" if prune_mode != "off" else None, delta_color="off", help="Zahl der Blätter und größte Tiefe des gezeigten Baums.")
m2.metric("Trainingsfehler", _err(task, tr["error"]), help="Fehlerquote (Klassifikation) bzw. RMSE in Minuten (Regression) auf den Lieferungen, aus denen der Baum gebaut wurde.")
m3.metric("Testfehler", _err(task, te["error"]), delta=f"Raten: {_err(task, a.baseline)}", delta_color="off", help="Dieselbe Größe auf den zurückgehaltenen Lieferungen.")
if task == "class":
    m4.metric("Test: AUC / Log-Loss", f"{te['auc']:.3f} / {te['logloss']:.2f}", help="AUC: Wahrscheinlichkeit, dass eine zufällige verspätete Lieferung einen höheren Wert bekommt als eine pünktliche (0.5 = Raten, 1 = perfekt). Log-Loss: bestraft Vorhersagen, die sich sicher irren - ein Blatt mit 0 % oder 100 % 'zu spät' kostet bei einem Fehlgriff sehr viel.")
else:
    m4.metric("Test: R² / MAE", f"{te['r2']:.3f} / {te['mae']:.1f} min", help="R²: Anteil der Streuung der Dauer, den der Baum erklärt (0 = Mittelwert raten). MAE: mittlerer absoluter Fehler.")
if prune_mode == "cv" and full_leaves > 1:
    st.caption(f"Die Kreuzvalidierung wählte alpha = {a.alpha:.5f}: der volle Baum ({full_leaves} Blätter) wird auf {tree.n_leaves} Blätter beschnitten.")
code = a.verdict
if code == "stump":
    st.info("ℹ️ Der Baum ist ein einziges Blatt: er sagt für jede Lieferung dasselbe voraus (Mittelwert bzw. häufigere Klasse).")
elif code == "overfit":
    st.warning(f"⚠️ **Überanpassung:** Trainingsfehler {_err(task, tr['error'])}, Testfehler {_err(task, te['error'])}. Der Baum hat ({tree.n_leaves} Blätter) Besonderheiten der Trainingslieferungen gelernt, die auf neuen nicht wiederkehren."
               + (f" Auffällig: die Genauigkeit sieht mit {te['accuracy']:.1%} noch gut aus, aber der Log-Loss ist {te['logloss']:.2f} - die Blätter sind überzuversichtlich." if task == "class" and te["logloss"] > 2 else "") + " Kleinere Tiefe, größere Blätter oder Beschneiden helfen.")
elif code == "underfit":
    st.info(f"ℹ️ **Zu einfach:** Der Testfehler ({_err(task, te['error'])}) ist kaum kleiner als beim Raten ({_err(task, a.baseline)}). Der Baum ist zu klein, um die Struktur zu fassen - mehr Tiefe oder kleinere Blätter.")
else:
    st.success(f"✅ **Ausgewogen:** Trainingsfehler {_err(task, tr['error'])}, Testfehler {_err(task, te['error'])}, Raten {_err(task, a.baseline)}. Der Abstand zwischen Training und Test ist klein.")

c1, c2 = st.columns(2)
c1.markdown("**Wichtigkeit der Merkmale**")
c1.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
c1.caption("Anteil an der gesamten Abnahme der Unreinheit, die Splits dieses Merkmals erzielen. Orange = Rauschmerkmale. Die Wichtigkeit eines Einzelbaums ist selbst instabil (siehe Experiment unten).")
with c2:
    st.markdown("**Fehler gegen Tiefe**")
    rows = _depth_rows(task, crit, int(leaf), *data_params)
    st.plotly_chart(build_depth_curve(rows, task, a.baseline, int(depth)), width="stretch", key="depth_chart")
    st.caption("Mit jeder Ebene sinkt der Trainingsfehler; der Testfehler hört früh auf zu sinken. Die gestrichelte Senkrechte ist die eingestellte Tiefe (Mindestblattgröße wie in der Seitenleiste, ohne Beschneiden).")
st.markdown("**Beschneiden: kleinere Bäume auf dem Pfad der Kosten-Komplexität**")
prows = _pruning_rows(task, crit, int(depth), int(leaf), *data_params)
st.plotly_chart(build_pruning_curve(prows, task, a.baseline, tree.n_leaves), width="stretch", key="pruning_chart")
st.caption("Jeder Punkt ist ein Baum der Kette: von links (ein Blatt) bis rechts (voller Baum). Bei einem voll gewachsenen Baum (Schnellstart 🌲) liegt die Testkurve in der Mitte am tiefsten (im Standarddatensatz bei 34 Blättern, Testfehler 13.9 % gegen 16.7 % beim vollen Baum): dort ist der Baum klein genug, das Rauschen nicht mitzulernen, und groß genug für die Struktur. Diese Kurve auf den Testdaten zum Auswählen zu benutzen wäre geschummelt; die Kreuzvalidierung wählt mit dem Training allein.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stabil ist der Baum?")
if st.button("Instabilität messen (dauert einen Moment)", key="stab_start"):
    st.session_state["stab_on"] = True
if st.session_state.get("stab_on"):
    with st.spinner("Wachse 3 Baumgrößen × 2 Störungen × 30 Bäume ..."):
        srows = _stability(task, crit, *data_params)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_instability(srows, task), width="stretch", key="stab_chart")
    c2.table({"Baum": [r["label"] for r in srows], "Blätter (Mittel)": [f"{r['leaves']:.0f}" for r in srows], "Aufbau oben, Bootstrap": [f"{r['prints_bootstrap']} verschiedene" for r in srows]})
    full_r = srows[-1]
    unit = (lambda x: f"{x:.0%}") if task == "class" else (lambda x: f"{x:.1f} min")
    st.caption(f"Mit den Einstellungen der Seitenleiste (Aufgabe, Kriterium, Daten) wurden je 30 Bäume auf Bootstrap-Stichproben (mit Zurücklegen gezogen) und auf 95 % der Trainingslieferungen gebaut; verglichen wird, wie stark sich ihre Vorhersagen auf den Testlieferungen unterscheiden (Mittel über alle Paare). "
               f"Der voll gewachsene Baum ({full_r['leaves']:.0f} Blätter im Mittel) sagt für {unit(full_r['bootstrap'])} der Testlieferungen etwas anderes voraus, je nachdem, welche Stichprobe er sah; schon das Weglassen von 5 % der Zeilen ändert {unit(full_r['drop'])}. Ein Baum der Tiefe 3 ist stabiler ({unit(srows[0]['bootstrap'])} bzw. {unit(srows[0]['drop'])}), aber auch er wechselt: in den oberen zwei Ebenen "
               f"entstehen bei den 30 Bootstrap-Bäumen {srows[0]['prints_bootstrap']} verschiedene Aufbauten. "
               + ("Die Wurzel dagegen ist stabil - ein Merkmal ist so deutlich der beste erste Split, dass alle Bäume mit ihm beginnen. " if min(r["root_share"] for r in srows) == 1.0 else f"Auch die Wurzel wechselt: nur {min(r['root_share'] for r in srows):.0%} der Bäume beginnen mit demselben Merkmal. ")
               + "Genau diese Streuung nutzt das nächste Stück der Linie (Bagging).")

st.markdown("---")

st.subheader("🔬 Beschneiden auf fünf Datensätzen")
if st.button("Voll gewachsen, flach und beschnitten vergleichen (dauert einen Moment)", key="sets_start"):
    st.session_state["sets_on"] = True
if st.session_state.get("sets_on"):
    with st.spinner("Wachse und beschneide auf 5 Datensätzen ..."):
        rows5 = _sets(task, crit, int(n), int(n_noise), int(label_noise))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_sets(rows5, task), width="stretch", key="sets_chart")
    mean = {k: float(np.mean([r[k] for r in rows5])) for k in ("full", "shallow", "pruned")}
    c2.table({"Baum": ["voll gewachsen", "Tiefe 4, Blatt ≥ 5", "beschnitten"], "Testfehler (Mittel)": [_err(task, mean[k]) for k in ("full", "shallow", "pruned")], "Blätter (Mittel)": [f"{np.mean([r['full_leaves'] for r in rows5]):.0f}", "-", f"{np.mean([r['pruned_leaves'] for r in rows5]):.0f}"]})
    better = sum(r["pruned"] <= r["full"] for r in rows5)
    st.caption(f"Fünf Datensätze mit denselben Regeln (nur der Seed ändert sich), Daten wie in der Seitenleiste. Der Baum, der nach der Kreuzvalidierung beschnitten wurde, ist in {better} von 5 Datensätzen mindestens so gut wie der voll gewachsene: im Mittel {_err(task, mean['pruned'])} gegen {_err(task, mean['full'])}, mit {np.mean([r['pruned_leaves'] for r in rows5]):.0f} statt {np.mean([r['full_leaves'] for r in rows5]):.0f} Blättern. "
               f"Ein fest eingestellter flacher Baum (Tiefe 4, Blatt ≥ 5) liegt bei {_err(task, mean['shallow'])} - ohne Rechnung für die Wahl der Größe.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gieriger Split genügt** | Jeder Split ist nur für den nächsten Schritt der beste; zwei Splits, die erst zusammen etwas bringen (Wechselwirkung), findet die Suche nur, wenn schon der erste allein etwas bringt. | tieferer Baum, Ensemble |
| **Genug Daten für jede Frage** | Weiter unten im Baum entscheiden immer weniger Lieferungen: ein Blatt mit fünf Lieferungen hat sein Mittel mit großer Unsicherheit. Der voll gewachsene Baum ist auf dem Training perfekt und auf neuen Lieferungen nicht besser (siehe oben). | Beschneiden, Mindestblattgröße |
| **Ein Baum ist stabil** | Andere Stichprobe, anderer Baum: bei einem voll gewachsenen Baum ändert sich die Vorhersage bei einem großen Teil der Lieferungen (siehe Experiment). | Bagging und Random Forest (nächste Stücke) |
| **Die Welt ist stufig** | Ein Baum liefert eine Stufenfunktion. Eine gerade Steigung (Dauer wächst mit der Distanz) wird zur Treppe, für die er viele Blätter braucht; über die Trainingsdaten hinaus (längere Strecken als je gesehen) sagt er den Wert des äußersten Blatts voraus. | Boosting, lineare Modelle |
| **Splits an einem Merkmal genügen** | Alle Splits stehen senkrecht zu einer Achse. Eine schräge Grenze (Verhältnis von zwei Merkmalen) wird zur Treppe. | abgeleitete Merkmale, Ensembles |
| **Wichtigkeit = Ursache** | Die Wichtigkeit im Einzelbaum ist instabil und begünstigt Merkmale mit vielen möglichen Schwellen; ein Rauschmerkmal, das der Baum tief unten benutzt, bekommt Wichtigkeit. | Permutationswichtigkeit (Random Forest) |
"""
)
st.caption("Die Wochentage stehen als Zahl 0 bis 6 im Baum und werden mit '≤' geteilt (wie jedes Merkmal); Merkmale mit vielen ungeordneten Stufen (etwa 24 Depots) behandelt dieses Verfahren nicht gesondert - dazu später CatBoost.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Trainingsdaten $(x_i,y_i)_{i=1}^N$, $x_i\in\mathbb{R}^d$. Ein Baum $T$ teilt den Merkmalsraum durch Splits $x_j\le s$ in Rechtecke (Blätter $\ell$) und sagt in jedem Blatt einen Wert $\hat y_\ell$ voraus: Anteil $p_\ell$ der Klasse "zu spät" bzw. Mittelwert der $y_i$ im Blatt.

**Unreinheit** eines Knotens mit Beispielmenge $t$ (bei zwei Klassen, $p$ = Anteil "1"): Gini $I(t)=2p(1-p)$, Entropie $I(t)=-p\log_2 p-(1-p)\log_2(1-p)$, Varianz $I(t)=\frac1{|t|}\sum_{i\in t}(y_i-\bar y_t)^2$.

**Gain eines Splits** $(j,s)$ mit Kindern $t_L=\{x_j\le s\}$, $t_R=\{x_j>s\}$:
$$\Delta(j,s)=I(t)-\frac{|t_L|}{|t|}I(t_L)-\frac{|t_R|}{|t|}I(t_R).$$
Sortiert man $t$ nach $x_j$, liefern die kumulativen Summen von $y$ (bei Klassen: Zahl der "1", bei Varianz zusätzlich $\sum y^2$) die Werte $I(t_L)$ und $I(t_R)$ für alle Schwellen in $O(|t|)$; mit dem Sortieren kostet ein Knoten $O(d\,|t|\log|t|)$. Kandidaten sind die Mitten zwischen verschiedenen benachbarten Werten, zulässig sind nur Splits mit $|t_L|,|t_R|\ge m$ (Mindestblattgröße). Bei Gleichstand gewinnt das kleinste Merkmal, dann die kleinste Schwelle.

**Wichtigkeit** des Merkmals $j$: $\sum_{t:\,\text{Split auf }j}\big(\tfrac{|t|}{N}I(t)-\tfrac{|t_L|}{N}I(t_L)-\tfrac{|t_R|}{N}I(t_R)\big)$, auf Summe 1 normiert.

**Kosten-Komplexität.** $R(T)=\sum_{\ell}\frac{|\ell|}{N}I(\ell)$ ist die gewichtete Unreinheit der Blätter, $|T|$ die Zahl der Blätter, $R_\alpha(T)=R(T)+\alpha|T|$. Für einen inneren Knoten $t$ mit Teilbaum $T_t$ ist das effektive $\alpha_t=\dfrac{R(t)-R(T_t)}{|T_t|-1}$: ab diesem $\alpha$ lohnt es sich, $T_t$ durch ein Blatt zu ersetzen.
Man kappt immer den Teilbaum mit dem kleinsten $\alpha_t$; das ergibt eine Kette $T_0\supset T_1\supset\dots\supset\{\text{Wurzel}\}$ und eine wachsende Folge $\alpha_0=0<\alpha_1\le\dots$; zu jedem $\alpha$ gehört genau ein kleinster Baum, der $R_\alpha$ minimiert. **Kreuzvalidierung:** in fünf Teilen des Trainings jeweils einen Baum wachsen lassen, ihn zu jedem $\alpha$ der Kette beschneiden und auf dem zurückgehaltenen Teil messen; gewählt wird das $\alpha$ mit dem kleinsten mittleren Fehler (bei Gleichstand das größere).

**Gütemaße** (auf dem Test, von Hand): Fehlerquote, AUC über die Rangsumme, Log-Loss $-\frac1n\sum\big(y\ln p+(1-y)\ln(1-p)\big)$ mit $p$ auf $[10^{-15},1-10^{-15}]$, RMSE, MAE, $R^2$.

Implementiert in `cart_algorithm.py` (Split-Suche, Wachsen, Vorhersage, Wichtigkeit, Beschneiden, Gütemaße), `cart_scenario.py` (Lieferdaten), `cart_evaluation.py` (Kennzahlen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
