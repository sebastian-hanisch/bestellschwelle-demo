"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen, Einrasten), je Preset das erwartete Verdikt."""

import pytest

import bsw_constants as C
import bsw_evaluation as ev
import bsw_presets as P


def _settings(preset: dict) -> ev.Settings:
    return ev.Settings(K=preset["k"], L=preset["l"], ph=preset["ph"], Q=preset["q"], horizon=preset["hor"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert C.K_MIN <= preset["k"] <= C.K_MAX and (preset["k"] / C.K_STEP) % 1 == 0, name
        assert C.L_MIN <= preset["l"] <= C.L_MAX and preset["ph"] in C.PH_OPTIONS, name
        assert C.Q_MIN <= preset["q"] <= C.Q_MAX and preset["hor"] in C.HORIZON_OPTIONS, name


def test_default_preset_is_first_and_equals_the_default_settings():
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET
    assert _settings(C.PRESETS[C.DEFAULT_PRESET]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_match_their_values():
    """Namen gegen die tatsächlichen Werte (Beschriftungs-Falle)."""
    assert C.PRESETS["Kein Unterschied (K = 0,5, Lieferzeit 1)"]["k"] == 0.5 and C.PRESETS["Kein Unterschied (K = 0,5, Lieferzeit 1)"]["l"] == 1
    assert C.PRESETS["Takt eine Periode (K = 1,5)"]["k"] == C.H * C.MEAN / 2  # K = h · Mittel / 2
    assert C.PRESETS["Teure Bestellung (K = 20)"]["k"] == 20.0
    p = C.PRESETS["Lange Lieferzeit (K = 10, Lieferzeit 3)"]
    assert (p["k"], p["l"]) == (10.0, 3)
    assert C.PRESETS["Billige Fehlmenge (p/h = 1)"]["ph"] == 1
    p = C.PRESETS["Von Hand (Handbeispiel)"]
    assert (p["k"], p["l"], p["ph"]) == (2.0, 0, 4)  # wie im Handbeispiel (K = 2, Lieferzeit 0, p = 4, h = 1)


def test_each_preset_shows_the_expected_verdict_line():
    """Ein Preset ohne Effekt darf keinen Effekt melden (DEMO-PLAYBOOK §11a, Punkt 7): je Preset die erwartete Zeile."""
    expect = {
        "Kein Unterschied (K = 0,5, Lieferzeit 1)": ("info", "noch optimal"),
        C.DEFAULT_PRESET: ("success", "kostet **22,0 %** mehr"),
        "Takt eine Periode (K = 1,5)": ("success", "kostet **3,6 %** mehr"),
        "Teure Bestellung (K = 20)": ("success", "kostet **94,0 %** mehr"),
        "Lange Lieferzeit (K = 10, Lieferzeit 3)": ("success", "kostet **46,1 %** mehr"),
        "Billige Fehlmenge (p/h = 1)": ("success", "kostet **50,5 %** mehr"),
        "Von Hand (Handbeispiel)": ("success", "mehr als die DP-Politik"),
    }
    for name, (kind, snippet) in expect.items():
        k, text = ev.verdict_gap(ev.analyse(_settings(C.PRESETS[name])))
        assert k == kind and snippet in text, (name, text)


def test_clean_value_clamps_and_snaps():
    spec = P.SETTING_SPECS
    assert P.clean_value(spec["k_slider"], "99") == 30.0 and P.clean_value(spec["k_slider"], "-5") == 0.0
    assert P.clean_value(spec["k_slider"], "1.3") == 1.5 and P.clean_value(spec["k_slider"], "1.2") == 1.0
    assert P.clean_value(spec["ph_select"], "5") in (4, 9) and P.clean_value(spec["ph_select"], "100") == 39
    assert P.clean_value(spec["ph_select"], "0") == 1
    assert P.clean_value(spec["hor_select"], "7") in (5, 10) and P.clean_value(spec["hor_select"], "1000") == 60
    assert P.clean_value(spec["l_slider"], "9") == 3 and P.clean_value(spec["q_slider"], "1") == 10
    with pytest.raises(ValueError):
        P.clean_value(spec["l_slider"], "abc")
