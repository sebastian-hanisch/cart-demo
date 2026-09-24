"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cart_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


ALL_CRITERIA = ("gini", "entropy", "variance")
N_FEATURES_MAX = C.N_BASE + C.NOISE_MAX

SETTING_SPECS = {
    "task_select": SettingSpec("task", _choice(C.TASKS), C.DEFAULT_TASK),
    "criterion_select": SettingSpec("crit", _choice(ALL_CRITERIA), "gini"),
    "depth_slider": SettingSpec("d", int, C.DEFAULT_DEPTH, C.DEPTH_MIN, C.DEPTH_MAX),
    "leaf_slider": SettingSpec("leaf", int, C.DEFAULT_LEAF, C.LEAF_MIN, C.LEAF_MAX),
    "prune_select": SettingSpec("prune", _choice(C.PRUNE_MODES), C.DEFAULT_PRUNE),
    "prune_leaves_slider": SettingSpec("pl", int, C.DEFAULT_PRUNE_LEAVES, 2, 3000),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "n_noise_slider": SettingSpec("nn", int, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "label_noise_slider": SettingSpec("ln", int, C.DEFAULT_LABEL_NOISE, C.LABEL_NOISE_MIN, C.LABEL_NOISE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "map_x_select": SettingSpec("fx", int, C.DEFAULT_MAP[0], 0, N_FEATURES_MAX - 1),
    "map_y_select": SettingSpec("fy", int, C.DEFAULT_MAP[1], 0, N_FEATURES_MAX - 1),
}
PRESET_KEYS = {"task": "task_select", "depth": "depth_slider", "leaf": "leaf_slider", "prune": "prune_select", "prune_leaves": "prune_leaves_slider", "n": "n_slider", "n_noise": "n_noise_slider",
               "label_noise": "label_noise_slider", "seed": "seed_input", "fx": "map_x_select", "fy": "map_y_select"}
# Regler, die je nach Einstellung ausgeblendet sind (Kriterium bei Regression, Etiketten-Rauschen bei Regression, Blätterzahl ohne Beschneiden von Hand): der Wert bleibt hier erhalten
KEPT = {"criterion_select": "_criterion_kept", "label_noise_slider": "_label_noise_kept", "prune_leaves_slider": "_prune_leaves_kept"}


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; ausgeblendete Regler kehren zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in KEPT and state_key not in st.session_state:       # ausblendbare Regler: siehe seed_widget
            st.session_state[state_key] = spec.default


def seed_widget(state_key):
    """Vor dem Zeichnen eines ausblendbaren Reglers: fehlt sein Zustand, kommt der zuletzt gewählte (oder der Standard-) Wert.
    Ein Wert, der in einem Lauf ohne den Regler in den Zustand des Reglers geschrieben wird, erscheint später als Mindestwert im Regler, während die App mit dem geschriebenen Wert rechnet."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def stash_kept_widget_state():
    """Permalink und Preset legen den Wert eines ausblendbaren Reglers nur in KEPT ab (der Regler holt ihn sich mit `seed_widget`, sobald er gezeichnet wird)."""
    for state_key, kept in KEPT.items():
        if state_key in st.session_state:
            st.session_state[kept] = st.session_state.pop(state_key)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
                if state_key in KEPT:
                    st.session_state[KEPT[state_key]] = value
            except (ValueError, TypeError):
                pass
    stash_kept_widget_state()
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = p[key]
    st.session_state["criterion_select"] = p["criterion"] if p["criterion"] in ("gini", "entropy") else st.session_state.get("_criterion_kept", "gini")
    for state_key, kept in KEPT.items():
        st.session_state[kept] = st.session_state[state_key]
    stash_kept_widget_state()


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
