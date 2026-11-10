"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Die Analyse wird für den Test von Hand gebaut."""

import numpy as np

import bsw_evaluation as ev
import bsw_policy as P


def _analysis(K, g, g_base, freq, freq_base, S_base=9, s=5, S=16, pure=True):
    return ev.Analysis(ev.Settings(K=K), g, np.zeros(3, dtype=int), np.zeros(3), np.zeros(3), np.zeros(3), P.Structure(s, S, pure), freq, g_base, S_base, freq_base)


# ---- Kernfrage ---------------------------------------------------------------------------------------------------------------------------


def test_verdict_gap_names_the_decomposition_in_cost_units_and_points():
    """K = 20, DP 12 (Bestellanteil 0,25), Base-Stock 24 (0,95): Fixkosten 5 gegen 19, Rest 7 gegen 5; Lücke 100 %."""
    a = _analysis(20.0, 12.0, 24.0, 0.25, 0.95)
    assert a.fixed_saved == 14.0 and a.rest_extra == 2.0 and round(a.gap, 6) == 100.0
    kind, text = ev.verdict_gap(a)
    assert kind == "success"
    assert "**100,0 %**" in text and "S = 9" in text and "(s, S) = (5, 16)" in text
    assert "24,000 statt 12,000" in text and "in 25,0 % der Perioden" in text and "in 95,0 %" in text
    assert "spart 14,00 Fixkosten" in text and "zahlt 2,00 mehr" in text
    assert "116,7 Fixkosten" in text and "-16,7 Lager und Fehlmengen" in text  # 14 / 12 und -2 / 12 in Punkten


def test_verdict_gap_threshold_one_percent_is_inclusive():
    assert ev.verdict_gap(_analysis(2.0, 100.0, 101.0, 0.6, 0.95))[0] == "success"   # genau 1,0 %
    kind, text = ev.verdict_gap(_analysis(2.0, 100.0, 100.99, 0.6, 0.95))             # 0,99 %
    assert kind == "info"


def test_verdict_gap_base_stock_still_optimal():
    kind, text = ev.verdict_gap(_analysis(0.5, 5.0, 5.0, 0.95, 0.95))
    assert kind == "info" and "noch optimal" in text and "Lücke 0,00 %" in text and "in 95,0 % der Perioden" in text


def test_verdict_gap_without_fixed_costs():
    kind, text = ev.verdict_gap(_analysis(0.0, 4.6, 4.6, 0.95, 0.95, S_base=9))
    assert kind == "info" and "Ohne Fixkosten" in text and "bis S = 9" in text


def test_verdict_gap_impure_form_is_named():
    kind, text = ev.verdict_gap(_analysis(20.0, 12.0, 24.0, 0.25, 0.95, pure=False))
    assert "mit nicht reiner (s, S)-Form (die Bestellgrenze greift)" in text and "(s, S) = (5, 16)" not in text


# ---- Schwellen ---------------------------------------------------------------------------------------------------------------------------


def test_verdict_threshold_names_each_percent_and_kstar():
    kind, text = ev.verdict_threshold({1.0: 1.0, 5.0: 2.0, 10.0: 3.0}, 0.5553, 0.6083, 1, 9)
    assert kind == "info"
    assert "Lieferzeit 1 und p/h = 9" in text and "ab K = 1,0 bei mindestens 1 %" in text and "ab K = 2,0 bei 5 %" in text and "ab K = 3,0 bei 10 %" in text
    assert "K* = 0,555 (EOQ-Takt 0,61 Perioden)" in text


def test_verdict_threshold_missing_values():
    _, text = ev.verdict_threshold({1.0: None, 5.0: None, 10.0: None}, None, None, 0, 1)
    assert text.count("nie (auf dem Raster bis K = 30)") == 3 and "liegt über K = 4" in text


# ---- EOQ ---------------------------------------------------------------------------------------------------------------------------------


def test_verdict_eoq_counts_the_band_inclusively():
    kind, text = ev.verdict_eoq([0.8, 1.0, 1.25])
    assert kind == "success" and "in 3 von 3 K-Werten" in text and "kleinstes Verhältnis 0,80, größtes 1,25" in text
    kind, text = ev.verdict_eoq([0.79, 1.0, 1.26, 1.1])
    assert kind == "info" and "in 2 von 4 K-Werten" in text and "kleinstes Verhältnis 0,79, größtes 1,26" in text
    assert "Das ist ein Band, keine Gleichheit." in text


def test_verdict_eoq_without_values():
    kind, text = ev.verdict_eoq([])
    assert kind == "info" and "K < 2" in text


# ---- Konvexität --------------------------------------------------------------------------------------------------------------------------


def test_verdict_convexity_branches():
    kind, text = ev.verdict_convexity(0.0, [], None, 0.0, [])
    assert kind == "success" and "konvex" in text and "K = 0,0" in text
    kind, text = ev.verdict_convexity(10.0, [6], 6, 8.24, [])
    assert kind == "info" and "an 1 Stelle nicht konvex (darunter z = s)" in text and "8,24 (0,82 K)" in text and "W(y) sind konvex" in text
    kind, text = ev.verdict_convexity(10.0, [0, 1, 2, 3], 0, 7.59, [0, 1])
    assert "an 4 Stellen nicht konvex" in text and "Auch W(y) ist nicht konvex (2 Stellen" in text
    _, text = ev.verdict_convexity(10.0, [7, 8], 6, 1.0, [])
    assert "z = s ist nicht darunter" in text


# ---- Zyklus, Los, Hand -------------------------------------------------------------------------------------------------------------------


def test_verdict_cycle():
    r = ev.CycleResult(K=5.0, g_dp=7.68, g_cycle=8.48, T=2, S=12, gap_cycle=10.43, gap_base=21.97, takt=1.83)
    kind, text = ev.verdict_cycle(r)
    assert kind == "info" and "alle T = 2 Perioden bis S = 12" in text and "10,4 % mehr als die DP" in text and "Base-Stock 22,0 %" in text
    r1 = ev.CycleResult(K=1.0, g_dp=5.49, g_cycle=5.56, T=1, S=9, gap_cycle=1.27, gap_base=1.27, takt=0.8)
    _, text = ev.verdict_cycle(r1)
    assert "T = 1" in text and "Base-Stock-Politik" in text and "1,3 % über der DP" in text


def test_verdict_cycle_gives_no_ranking_when_the_order_limit_binds():
    """Regression: Bei Q = 10, K = 30, Lieferzeit 3, p/h = 1 greift die Bestellgrenze nur in der DP, die Zyklus-Politik ist ohne Grenze gerechnet
    und kostet dann weniger als die DP. Das Verdikt schrieb „kostet -16,9 % mehr als die DP ... nicht so gut wie die Schwelle s“."""
    res = ev.cycle_vs_dp(ev.Settings(K=30.0, L=3, ph=1, Q=10))
    assert res.gap_cycle < 0
    kind, text = ev.verdict_cycle(res)
    assert "nicht so gut wie die Schwelle s" not in text and "mehr als die DP" not in text
    assert kind == "info" and "Bestellgrenze greift" in text and "nicht fair" in text and res.capped
    free = ev.cycle_vs_dp(ev.Settings(K=30.0, L=3, ph=1, Q=40))
    assert not free.capped and free.gap_cycle > 0
    assert "nicht so gut wie die Schwelle s" in ev.verdict_cycle(free)[1]


def test_verdict_lot_names_all_four_numbers():
    lot = {"s": 6, "S": 14, "lot": 8, "half": {"lot": 4, "pct": 11.57}, "double": {"lot": 16, "pct": 15.32},
           "param_x2": {"pct": 2.56}, "param_half": {"pct": 1.51}}
    kind, text = ev.verdict_lot(lot)
    assert kind == "info" and "(s, S) = (6, 14), Losgröße 8" in text
    assert "halbe Losgröße (4) 11,6 %" in text and "doppelte (16) 15,3 %" in text and "2 K kostet 2,6 %" in text and "K / 2 1,5 %" in text


def test_verdict_hand():
    kind, text = ev.verdict_hand(((0, 2), "2"), ((1, 2), "5/2"), 25.0, 0.45, 0.75)
    assert kind == "success" and "(s, S) = (0, 2) mit g = 2" in text and "({sb}".format(sb=1) in text
    assert "(1, 2) mit g = 5/2: **25,0 %** mehr" in text and "in 75 % der Perioden" in text and "nur in 45 %" in text


def test_verdict_gap_says_weniger_when_the_dp_pays_less_for_the_rest():
    kind, text = ev.verdict_gap(_analysis(2.0, 100.0, 101.0, 0.6, 0.95))  # Rest DP 98,8 gegen Base-Stock 99,1
    assert "spart 0,70 Fixkosten und zahlt 0,30 weniger für Lager und Fehlmengen" in text
