"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset, fährt jeden Regler an seine Grenzen und prüft alle sechs Tabs.

Geprüft wird, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Diagrammen mit gleichem Inhalt und Fehler bei Reglern mit
berechneten Grenzen) und dass jeder Tab und jedes Experiment erscheint."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import bsw_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
TABS = ["Von Hand", "Politik", "Wertfunktion", "Über K", "Alle (s,S)", "EOQ"]


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _expander(at, label):
    return next(e for e in at.expander if e.label == label)  # nie per Index: neue Expander verschieben ihn


def test_app_loads_with_six_tabs_and_five_experiments():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == TABS
    subs = " | ".join(s.value for s in at.subheader)
    for n in range(1, 6):
        assert f"Experiment {n}:" in subs
    assert "🚧 Wo die Annahmen enden" in subs
    assert at.title[0].value.startswith("📦")
    assert len(at.get("plotly_chart")) >= 14


def test_every_preset_runs_and_shows_a_verdict():
    at = _fresh_app()
    for name in C.PRESETS:
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        texts = " ".join(m.value for m in list(at.success) + list(at.info))
        assert "Ein-Schwellen-Regel" in texts or "Base-Stock" in texts, name


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert len(labels) >= 4  # K, Lieferzeit, Bestellgrenze, Faktor f
    for index, label in enumerate(labels):
        at = _fresh_app()
        for edge in ("min", "max"):
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,options", [("ph_select", (1, 4, 39)), ("hor_select", (0, 1, 2, 10, 60))])
def test_selection_sliders_run_with_their_edge_and_middle_options(key, options):
    at = _fresh_app()
    for option in options:
        at.sidebar.select_slider(key=key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_corner_settings_run_in_all_tabs():
    """Drei ausgewählte Eckeinstellungen des Reglerraums: kleinstes K mit längster Lieferzeit und kleinstem p/h, größtes K mit Lieferzeit 0 und größtem p/h, Bestellgrenze 10."""
    for k, l_, ph, q, hor in ((0.0, 3, 1, 40, 0), (30.0, 0, 39, 10, 1), (30.0, 3, 1, 10, 60)):
        at = _fresh_app()
        at.sidebar.slider(key="k_slider").set_value(k)
        at.sidebar.slider(key="l_slider").set_value(l_)
        at.sidebar.select_slider(key="ph_select").set_value(ph)
        at.sidebar.slider(key="q_slider").set_value(q)
        at.sidebar.select_slider(key="hor_select").set_value(hor)
        at.run(timeout=TIMEOUT)
        assert not at.exception, ((k, l_, ph, q, hor), [str(e) for e in at.exception])


def test_assumptions_expander_equals_the_assumptions_text():
    at = _fresh_app()
    assert C.ASSUMPTIONS_MARKDOWN.strip() == _expander(at, "⚠️ Annahmen, Grenzen und bekannte Probleme").markdown[0].value.strip()


def test_math_expander_has_one_block_per_display_equation():
    at = _fresh_app()
    text = [m.value for m in _expander(at, "📐 Mathematische Formulierung").markdown if "$$" in m.value][0]  # der Lesehilfe-Block steht davor
    assert text.count("$$") % 2 == 0 and text.count("$$") // 2 == 6  # Kosten, Bellman, (s,S), K-Konvexität, EOQ, Zyklus: je ein Block


def test_permalink_is_clamped_and_snapped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["k"] = "99"
    at.query_params["ph"] = "5"
    at.query_params["hor"] = "7"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="k_slider").value == 30.0
    assert at.sidebar.select_slider(key="ph_select").value in (4, 9)
    assert at.sidebar.select_slider(key="hor_select").value in (5, 10)


def test_assumption_texts_in_the_app_say_what_is_not_measured():
    at = _fresh_app()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Messbefund" in text and "Vermutung" in text and "Volltext gelesen" in text
