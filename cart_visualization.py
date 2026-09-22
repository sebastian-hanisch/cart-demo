"""Plotly-Darstellungen: Baumdiagramm, Karte über zwei Merkmale, Split-Suche, Wichtigkeit, Fehlerkurven und Experimente. Alle Achsen sind gesperrt (Touch-Scrollen)."""

import numpy as np
import plotly.graph_objects as go

import cart_algorithm as alg
import cart_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]        # pünktlich (grün) -> zu spät (rot)
REG_SCALE = "Viridis"
NOISE = C.COLORS["noise"]


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "")
    return f"{names[f]} [{unit}]" if unit else names[f]


def value_text(task, v):
    return f"{v:.0%} zu spät" if task == "class" else f"{v:.0f} min"


# --- Baumdiagramm ------------------------------------------------------------------------------------------------------------------------------------

def tree_layout(tree):
    """x = Blätter von links nach rechts durchnummeriert, innere Knoten mittig über ihren Kindern; y = -Tiefe."""
    x = np.zeros(tree.n_nodes)
    counter = 0
    stack = [(0, False)]
    while stack:
        t, done = stack.pop()
        if tree.feature[t] < 0:
            x[t] = counter
            counter += 1
        elif done:
            x[t] = (x[tree.left[t]] + x[tree.right[t]]) / 2.0
        else:
            stack += [(t, True), (int(tree.right[t]), False), (int(tree.left[t]), False)]
    return x, -tree.depth.astype(float)


def build_tree(tree, names, ds_y, next_node=None, path=None, height=460):
    """Baum mit Splits (Knoten) und Blättern (gefärbt nach Blattwert). `next_node` = Knoten, der als Nächster geteilt wird (schwarzer Ring); `path` = Knoten eines Beispiels (orange Linie).
    Knoten mit Splits auf Rauschmerkmalen haben einen orangen Rand."""
    x, y = tree_layout(tree)
    inner = tree.feature >= 0
    fig = go.Figure()
    ex, ey = [], []
    for t in np.nonzero(inner)[0]:
        for c in (tree.left[t], tree.right[t]):
            ex += [x[t], x[c], None]
            ey += [y[t], y[c], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    if path:
        px = [x[t] for t in path]
        py = [y[t] for t in path]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color=NOISE, width=5), hoverinfo="skip", showlegend=False, opacity=0.8))
    task = tree.task
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ds_y)), float(np.max(ds_y)))
    real = C.N_BASE
    hover = []
    text = []
    label_all = tree.n_nodes <= 15
    label_leaves = tree.n_leaves <= 12
    for t in range(tree.n_nodes):
        if inner[t]:
            desc = f"{names[tree.feature[t]]} ≤ {tree.threshold[t]:.3g}"
            hover.append(f"Knoten {t}: {desc}?<br>{tree.n[t]} Lieferungen, Unreinheit {tree.impurity[t]:.3f}<br>Blattwert wäre {value_text(task, tree.value[t])}")
            text.append(desc if label_all or tree.depth[t] <= 1 else "")
        else:
            hover.append(f"Blatt {t}: {value_text(task, tree.value[t])}<br>{tree.n[t]} Lieferungen, Unreinheit {tree.impurity[t]:.3f}")
            text.append(f"{value_text(task, tree.value[t]).replace(' zu spät', '')}<br>n={tree.n[t]}" if label_leaves else "")
    size = 9 + 20 * np.sqrt(tree.n / tree.n_total)
    line_c = ["#111111" if not inner[t] else (NOISE if tree.feature[t] >= real else "#555555") for t in range(tree.n_nodes)]
    line_w = [1 if not inner[t] else (3 if tree.feature[t] >= real else 1.5) for t in range(tree.n_nodes)]
    color = np.where(inner, np.nan, tree.value)
    fig.add_trace(go.Scatter(x=x[~inner], y=y[~inner], mode="markers+text", text=[text[t] for t in np.nonzero(~inner)[0]], textposition="bottom center", textfont=dict(size=9),
                             marker=dict(size=size[~inner], color=color[~inner], colorscale=CLASS_SCALE if task == "class" else REG_SCALE, cmin=vmin, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[hover[t] for t in np.nonzero(~inner)[0]], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=x[inner], y=y[inner], mode="markers+text", text=[text[t] for t in np.nonzero(inner)[0]], textposition="top center", textfont=dict(size=9),
                             marker=dict(size=size[inner], color="#ffffff", line=dict(color=[line_c[t] for t in np.nonzero(inner)[0]], width=[line_w[t] for t in np.nonzero(inner)[0]])),
                             hovertext=[hover[t] for t in np.nonzero(inner)[0]], hoverinfo="text", showlegend=False))
    if next_node is not None:
        fig.add_trace(go.Scatter(x=[x[next_node]], y=[y[next_node]], mode="markers", marker=dict(size=size[next_node] + 14, color="rgba(0,0,0,0)", line=dict(color="#111111", width=3)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Karte -----------------------------------------------------------------------------------------------------------------------------------------

def build_map(tree, ds, fx, fy, sample=None, height=430):
    """Vorhersage des Baums über zwei Merkmale (die übrigen auf ihrem Median im Training) mit den Trainingspunkten darüber."""
    task = tree.task
    Xtr = ds.X[ds.train]
    ytr = ds.y(task)[ds.train]
    med = np.median(Xtr, axis=0)
    gx = np.round(np.linspace(Xtr[:, fx].min(), Xtr[:, fx].max(), 60), 4)
    gy = np.round(np.linspace(Xtr[:, fy].min(), Xtr[:, fy].max(), 60), 4)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.tile(med, (XX.size, 1))
    grid[:, fx], grid[:, fy] = XX.ravel(), YY.ravel()
    z = np.round(alg.predict_value(tree, grid), 3).reshape(XX.shape)
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ytr)), float(np.max(ytr)))
    scale = CLASS_SCALE if task == "class" else REG_SCALE
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale=scale, zmin=vmin, zmax=vmax, opacity=0.55, showscale=False, hovertemplate="%{z:.2f}<extra></extra>"))
    if task == "class":
        fig.add_trace(go.Contour(x=gx, y=gy, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#111111", width=2), showscale=False, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.round(Xtr[:, fx], 4), y=np.round(Xtr[:, fy], 4), mode="markers", marker=dict(size=5, color=ytr, colorscale=scale, cmin=vmin, cmax=vmax, line=dict(color="#333333", width=0.5)),
                             hovertemplate="%{x:.3g} / %{y:.3g}<extra></extra>", showlegend=False))
    if sample is not None:
        fig.add_trace(go.Scatter(x=[sample[fx]], y=[sample[fy]], mode="markers", marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#111111", width=2)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=feature_label(ds.names, fx))
    fig.update_yaxes(title=feature_label(ds.names, fy))
    return lock_axes(fig, height)


# --- Split-Suche und Wichtigkeit ---------------------------------------------------------------------------------------------------------------------

def build_split_search(names, best_gain, chosen, height=330):
    """Bester Gain je Merkmal für den Knoten, der als Nächster geteilt wird; das gewählte Merkmal ist schwarz."""
    d = len(names)
    colors = ["#111111" if f == chosen else (NOISE if f >= C.N_BASE else "#1f77b4") for f in range(d)]
    gains = [g if np.isfinite(g) else 0.0 for g in best_gain]
    fig = go.Figure(go.Bar(x=gains, y=list(names), orientation="h", marker_color=colors, text=[f"{g:.4f}" if np.isfinite(b) else "kein zulässiger Split" for g, b in zip(gains, best_gain)], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="bester Gain (Abnahme der Unreinheit)", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=80, t=30, b=10))


def build_gain_curve(names, f, thr, gain, chosen_thr, height=280):
    """Gain über alle zulässigen Schwellen eines Merkmals."""
    ok = np.isfinite(gain)
    fig = go.Figure(go.Scatter(x=thr[ok], y=gain[ok], mode="lines", line=dict(color="#1f77b4", width=2), hovertemplate="Schwelle %{x:.4g}: Gain %{y:.4f}<extra></extra>"))
    if chosen_thr is not None:
        fig.add_vline(x=chosen_thr, line=dict(color="#111111", dash="dash"))
    fig.update_xaxes(title=f"Schwelle {feature_label(names, f)}")
    fig.update_yaxes(title="Gain", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False)


def build_importance(names, imp, height=330):
    order = np.argsort(-imp, kind="stable")
    colors = [NOISE if f >= C.N_BASE else "#1f77b4" for f in order]
    fig = go.Figure(go.Bar(x=imp[order], y=[names[f] for f in order], orientation="h", marker_color=colors, text=[f"{imp[f]:.1%}" for f in order], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (Anteil an der Abnahme der Unreinheit)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=60, t=30, b=10))


# --- Kurven ------------------------------------------------------------------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_depth_curve(rows, task, baseline, current, height=340):
    d = [r["depth"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=[r["train"] for r in rows], mode="lines+markers", name="Training", line=dict(color=C.COLORS["train"]), customdata=[r["leaves"] for r in rows], hovertemplate="Tiefe %{x}: %{y:.3f}, %{customdata} Blätter<extra>Training</extra>"))
    fig.add_trace(go.Scatter(x=d, y=[r["test"] for r in rows], mode="lines+markers", name="Test", line=dict(color=C.COLORS["test"]), customdata=[r["leaves"] for r in rows], hovertemplate="Tiefe %{x}: %{y:.3f}, %{customdata} Blätter<extra>Test</extra>"))
    fig.add_hline(y=baseline, line=dict(color="#888888", dash="dash"), annotation_text="ohne Baum (Raten)", annotation_position="top right")
    fig.add_vline(x=current, line=dict(color="#111111", dash="dot"))
    fig.update_xaxes(title="max. Tiefe", dtick=2)
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


def build_pruning_curve(rows, task, baseline, current_leaves, height=340):
    lv = [r["leaves"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lv, y=[r["train"] for r in rows], mode="lines+markers", name="Training", line=dict(color=C.COLORS["train"]), hovertemplate="%{x} Blätter: %{y:.3f}<extra>Training</extra>"))
    fig.add_trace(go.Scatter(x=lv, y=[r["test"] for r in rows], mode="lines+markers", name="Test", line=dict(color=C.COLORS["test"]), hovertemplate="%{x} Blätter: %{y:.3f}<extra>Test</extra>"))
    fig.add_hline(y=baseline, line=dict(color="#888888", dash="dash"), annotation_text="ohne Baum (Raten)", annotation_position="top right")
    fig.add_vline(x=current_leaves, line=dict(color="#111111", dash="dot"))
    fig.update_xaxes(title="Blätter nach dem Beschneiden", type="log")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


def build_instability(rows, task, height=320):
    """rows: [{"label", "bootstrap", "drop"}] - mittlere paarweise Abweichung der Vorhersagen."""
    fig = go.Figure()
    fmt = ".1%" if task == "class" else ".1f"
    fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r["bootstrap"] for r in rows], name="Bootstrap-Stichproben", marker_color="#1f77b4", text=[format(r["bootstrap"], fmt) for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r["drop"] for r in rows], name="5 % der Zeilen weg", marker_color="#ff7f0e", text=[format(r["drop"], fmt) for r in rows], textposition="outside"))
    fig.update_yaxes(title="Vorhersagen anders (Paare von Bäumen)" if task == "class" else "Abweichung der Vorhersagen [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))
    return lock_axes(fig, height, barmode="group", legend=dict(orientation="h", y=1.12))


def build_sets(rows, task, height=340):
    """rows: [{"label", "full", "shallow", "pruned"}] - Testfehler je Datensatz."""
    fig = go.Figure()
    fmt = ".1%" if task == "class" else ".1f"
    for key, name, color in (("full", "voll gewachsen", "#d62728"), ("shallow", "Tiefe 4, Blatt ≥ 5", "#1f77b4"), ("pruned", "beschnitten (Kreuzvalidierung)", "#2ca02c")):
        fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r[key] for r in rows], name=name, marker_color=color, text=[format(r[key], fmt) for r in rows], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Testfehler" if task == "class" else "Test-RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))
    return lock_axes(fig, height, barmode="group", legend=dict(orientation="h", y=1.12))
