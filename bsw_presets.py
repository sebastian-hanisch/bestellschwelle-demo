"""Permalink (Begrenzen und Einrasten), Presets und aktives Preset (Standardmuster aus dem Demo-Portfolio)."""

from dataclasses import dataclass

import streamlit as st

import bsw_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None  # Auswahlregler: erlaubte Werte (der Permalink rastet auf den nächsten ein)


SETTING_SPECS = {
    "k_slider": SettingSpec("k", float, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "l_slider": SettingSpec("l", int, C.DEFAULT_L, C.L_MIN, C.L_MAX),
    "ph_select": SettingSpec("ph", int, C.DEFAULT_PH, min(C.PH_OPTIONS), max(C.PH_OPTIONS), C.PH_OPTIONS),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "hor_select": SettingSpec("hor", int, C.DEFAULT_HORIZON, min(C.HORIZON_OPTIONS), max(C.HORIZON_OPTIONS), C.HORIZON_OPTIONS),
}
PRESET_KEYS = {"k": "k_slider", "l": "l_slider", "ph": "ph_select", "q": "q_slider", "hor": "hor_select"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec: SettingSpec, raw):
    """Wert aus Adresszeile oder Preset: in den Typ wandeln, auf die Grenzen begrenzen, bei Auswahlreglern auf den nächsten erlaubten Wert einrasten."""
    value = spec.caster(raw)
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.options is not None:
        value = min(spec.options, key=lambda o: (abs(o - value), o))
    elif spec.caster is float and spec is SETTING_SPECS["k_slider"]:
        value = round(value / C.K_STEP) * C.K_STEP  # K nur in Schritten von 0,5
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt, eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                st.session_state[state_key] = clean_value(spec, qp[spec.url_param])
            except (ValueError, TypeError):
                continue
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos, jeder Regler aus SETTING_SPECS)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
