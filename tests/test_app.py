"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Aufgabenwechsel, ausgeblendete Regler, Beschneiden, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cart_constants as C
from cart_presets import KEPT, PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
VERDICT = {"🌳 Flacher Baum": "success", "🌲 Voll gewachsen": "warning", "✂️ Beschneiden": "success", "🏷️ Falsche Etiketten": "warning", "📈 Regression": "success"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]
    at.session_state["criterion_select"] = p["criterion"] if p["criterion"] in ("gini", "entropy") else "gini"


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def _leaves(at):
    return int(next(m.value.split(" / ")[0] for m in at.metric if m.label == "Blätter / Tiefe"))


def test_default_renders_without_exception_and_gives_one_verdict():
    at = _run()
    assert any("CART in Aktion" in m.value for m in at.markdown) and not at.error
    assert any("Ausgewogen" in s.value for s in at.success) and not at.warning


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_a_verdict_of_the_expected_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    # Der fertige Baum meldet zusätzlich "Der Baum ist fertig" (success) im Schritt-Bereich
    assert (len(at.warning) == 1) == (VERDICT[name] == "warning") and len(at.success) >= 1 and not at.error


def test_the_task_switch_hides_the_criterion_and_the_label_noise():
    at = _run()
    assert {"Split-Kriterium", "Falsche Etiketten im Training [%]"} <= _labels(at)
    at.session_state["task_select"] = "reg"
    at.run()
    assert not at.exception and not {"Split-Kriterium", "Falsche Etiketten im Training [%]"} & _labels(at)
    assert any("Varianz" in c.value for c in at.sidebar.caption) and any("nur bei der Klassifikation" in c.value for c in at.sidebar.caption)
    assert any(m.label == "Test: R² / MAE" for m in at.metric)


def test_hidden_label_noise_comes_back_when_the_task_returns():
    # Die erste Sicht muss die Klassifikation sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run()
    at.session_state["label_noise_slider"] = 10
    at.run()
    at.session_state["task_select"] = "reg"
    at.run()
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception and at.slider(key="label_noise_slider").value == 10


def test_the_manual_pruning_slider_only_appears_for_manual_pruning_and_a_big_tree():
    at = _run()
    assert "Höchstens so viele Blätter" not in _labels(at)
    at.session_state["prune_select"] = "manual"
    at.run()
    assert not at.exception and "Höchstens so viele Blätter" in _labels(at)
    s = at.slider(key="prune_leaves_slider")
    assert s.min == 2 and s.max > 2
    at.session_state["depth_slider"] = 1                                                             # zwei Blätter: nichts zu beschneiden, kein Regler mit min == max
    at.run()
    assert not at.exception and "Höchstens so viele Blätter" not in _labels(at) and any("nichts zu beschneiden" in c.value for c in at.sidebar.caption)


def test_manual_pruning_limits_the_leaves_and_cv_reports_alpha():
    at = _run(lambda a: _apply(a, C.PRESETS["🌲 Voll gewachsen"]))
    assert _leaves(at) == 100
    at.session_state["prune_select"] = "manual"
    at.session_state["prune_leaves_slider"] = 10
    at.run()
    assert not at.exception and 2 <= _leaves(at) <= 10
    at.session_state["prune_select"] = "cv"
    at.run()
    assert not at.exception and any("Kreuzvalidierung wählte alpha" in c.value for c in at.caption)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_slider"] = C.N_MIN
        at.session_state["depth_slider"] = C.DEPTH_MIN
        at.session_state["leaf_slider"] = C.LEAF_MAX
        at.session_state["n_noise_slider"] = 0

    def big(at):
        at.session_state["n_slider"] = C.N_MAX
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["leaf_slider"] = 1
        at.session_state["n_noise_slider"] = C.NOISE_MAX
        at.session_state["label_noise_slider"] = C.LABEL_NOISE_MAX

    def reg(at):
        at.session_state["task_select"] = "reg"
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["leaf_slider"] = 1
        at.session_state["prune_select"] = "cv"
    for setup in (small, big, reg):
        at = _run(setup)
        assert at.slider(key="cart_step").value == at.slider(key="cart_step").max


def test_map_features_beyond_the_columns_fall_back_after_fewer_noise_features():
    at = _run()
    at.session_state["map_x_select"] = 10
    at.run()
    at.session_state["n_noise_slider"] = 0
    at.run()
    assert not at.exception and at.selectbox(key="map_x_select").value == C.DEFAULT_MAP[0]


def test_the_test_delivery_slider_survives_a_smaller_data_set():
    at = _run()
    at.session_state["sample_slider"] = 350
    at.run()
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    assert not at.exception and at.slider(key="sample_slider").value == 0


def test_step_slider_returns_to_the_last_cut_when_the_settings_change():
    at = _run()
    at.slider(key="cart_step").set_value(3)
    at.run()
    assert at.slider(key="cart_step").value == 3
    at.session_state["depth_slider"] = 6
    at.run()
    assert not at.exception and at.slider(key="cart_step").value == at.slider(key="cart_step").max


def test_every_step_of_a_small_tree_renders():
    at = _run(lambda a: _apply(a, C.PRESETS["🌳 Flacher Baum"]))
    for k in range(0, int(at.slider(key="cart_step").max) + 1):
        at.slider(key="cart_step").set_value(k)
        at.run()
        assert not at.exception, k
        if k == 0:
            assert any("Split 1 von" in m.value for m in at.markdown)


def test_play_renders_several_frames_without_duplicate_keys(monkeypatch):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _run(lambda a: _apply(a, C.PRESETS["🌳 Flacher Baum"]))                                       # acht Bilder; die Bildfolge im AppTest ist langsam, im Browser dauert sie 6 s
    _play(at)
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped_and_unknown_choices_fall_back():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["d"] = "99"
    at.query_params["leaf"] = "-4"
    at.query_params["task"] = "forest"
    at.query_params["crit"] = "variance"
    at.query_params["fx"] = "99"
    at.run()
    assert not at.exception
    assert at.slider(key="depth_slider").value == C.DEPTH_MAX and at.slider(key="leaf_slider").value == C.LEAF_MIN and at.selectbox(key="task_select").value == C.DEFAULT_TASK
    assert at.selectbox(key="criterion_select").value == "gini"                                     # "variance" gehört nicht zur Klassifikation


def test_the_address_bar_mirrors_the_settings():
    at = _run(lambda a: _apply(a, C.PRESETS["🏷️ Falsche Etiketten"]))
    assert str(at.query_params["ln"]) in ("10", "['10']") and str(at.query_params["d"]) in ("16", "['16']")


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Bootstrap-Stichproben" in c.value for c in at.caption)
    for key in ("stab_start", "sets_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Bootstrap-Stichproben", "Die Wurzel dagegen ist stabil", "nach der Kreuzvalidierung beschnitten"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert sorted(keys) == sorted(["tree_chart", "map_chart", "search_chart", "gain_chart", "importance_chart", "depth_chart", "pruning_chart", "stab_chart", "sets_chart"]), keys
    looped = [c for c in calls if 'key=f"' in c]
    assert len(looped) == 4 and all('_{current}"' in c for c in looped)                              # die Bilder der Abspiel-Schleife tragen den Schritt
    viz = (ROOT / "cart_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("lock_axes(fig") >= 9


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_kept_values_cover_the_conditionally_hidden_controls():
    assert set(KEPT) == {"criterion_select", "label_noise_slider", "prune_leaves_slider"}
