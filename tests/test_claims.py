"""Jede Zahl der README, an der Quelle nachgerechnet (Tabelle „Zahl → Test“ in der README).

Zwei Arten von Prüfung:
  * **live:** die Zahl wird hier mit den Funktionen der Demo neu berechnet (Band oder exakt, wie angegeben);
  * **Raster:** Zählungen über alle 1.464 Zellen stehen in data/messraster.csv (erzeugt von scripts/messraster.py, unter einer Minute). Der Test zählt
    an dieser Datei und rechnet zusätzlich eine Stichprobe von Zellen live nach, damit die Datei nicht unbemerkt veraltet.
Messungen, die nur in der README stehen und nicht hier geprüft werden, sind dort als „Messung dieser README, nicht im Test“ gekennzeichnet."""

import csv
from pathlib import Path

import numpy as np
import pytest

import bsw_constants as C
import bsw_demand as DEM
import bsw_dp as D
import bsw_eoq as E
import bsw_evaluation as ev
import bsw_policy as P

ROOT = Path(__file__).resolve().parent.parent
MU = C.MEAN


@pytest.fixture(scope="module")
def raster():
    with open(ROOT / "data" / "messraster.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    cols = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}
    cols["n"] = len(rows)
    return cols


def cell(raster_, L, ph, K):
    i = np.nonzero((raster_["L"] == L) & (raster_["ph"] == ph) & (raster_["K"] == K))[0]
    assert len(i) == 1
    return int(i[0])


# ---------------------------------------------------------------------------------------------------------------- Raster: Form und Ränder


def test_raster_has_1464_cells(raster):
    assert raster["n"] == 4 * 6 * 61 == 1464


def test_dp_has_pure_sS_form_in_all_1464_cells(raster):
    assert int(raster["pure"].sum()) == 1464


def test_boundary_masses_in_the_whole_raster(raster):
    assert raster["qcap"].max() < 1e-13 and raster["qcapBase"].max() == 0.0          # Anteil der Zeit an der Bestellgrenze Q = 40
    assert max(raster["mass_lo"].max(), raster["mass_hi"].max()) < 3e-16             # Masse an den Zustandsgrenzen -40 und 60
    assert max(raster["massBase_lo"].max(), raster["massBase_hi"].max()) == 0.0
    assert raster["qmax"].max() == 38                                               # größte bestellte Menge einer erreichten Position


def test_raster_sample_is_reproduced_live(raster):
    """Jede 61. Zelle (24 Zellen) neu gerechnet: g, Base-Stock, (s, S) stimmen mit der Datei überein."""
    for i in range(0, raster["n"], 61):
        L, ph, K = int(raster["L"][i]), int(raster["ph"][i]), float(raster["K"][i])
        a = ev.analyse(ev.Settings(K=K, L=L, ph=ph))
        assert a.g == pytest.approx(raster["g"][i], abs=1e-8) and a.g_base == pytest.approx(raster["gBase"][i], abs=1e-8)
        assert (a.structure.s, a.structure.S, a.S_base) == (raster["s"][i], raster["S"][i], raster["SBase"][i])
    for (L, ph, K) in ((0, 9, 2.0), (1, 9, 5.0), (3, 1, 30.0), (2, 39, 0.5), (3, 9, 20.0)):
        i = cell(raster, L, ph, K)
        a = ev.analyse(ev.Settings(K=K, L=L, ph=ph))
        assert a.gap == pytest.approx(raster["gap_pct"][i], abs=1e-4)


# ---------------------------------------------------------------------------------------------------------------- K = 0: Newsvendor


def test_k_zero_dp_is_base_stock_at_the_quantile_in_24_of_24_cells(raster):
    ok = 0
    for L in range(4):
        for ph in C.PH_OPTIONS:
            i = cell(raster, L, ph, 0.0)
            S_q = DEM.critical_quantile(ev.model_for(L, ph, 40).dn, float(ph), 1.0)
            ok += (raster["S"][i], raster["s"][i], raster["SBase"][i]) == (S_q, S_q - 1, S_q)
    assert ok == 24


def test_S_at_k_zero_for_p_over_h_9_is_5_9_13_17(raster):
    assert [int(raster["S"][cell(raster, L, 9, 0.0)]) for L in range(4)] == [5, 9, 13, 17]


# ---------------------------------------------------------------------------------------------------------------- Lücke über K


GAP_TABLE = {  # p/h = 9, Lieferzeit 0 / 1 / 2 / 3 (Prozent der DP-Kosten, eine Stelle genau)
    0.5: (0.00, 0.00, 0.53, 0.91), 1.0: (0.33, 1.27, 1.49, 2.41), 2.0: (5.87, 5.67, 5.59, 6.15),
    5.0: (20.8, 22.0, 20.9, 21.5), 20.0: (95.8, 94.0, 91.1, 88.6), 30.0: (134.8, 130.8, 128.1, 125.4),
}


def test_gap_table_for_p_over_h_9_live():
    for K, expected in GAP_TABLE.items():
        got = [ev.analyse(ev.Settings(K=K, L=L, ph=9)).gap for L in range(4)]
        assert got == pytest.approx(list(expected), abs=0.006 if K <= 2.0 else 0.06), K


def test_gap_counts_over_the_24_cells_per_K(raster):
    expected = {0.5: (8, 0, 0), 1.0: (17, 4, 0), 1.5: (24, 11, 4), 2.0: (24, 16, 8), 3.0: (24, 24, 16), 4.5: (24, 24, 24)}  # >= 1 %, >= 5 %, >= 10 %
    for K, counts in expected.items():
        m = raster["K"] == K
        got = tuple(int((raster["gap_pct"][m] >= t).sum()) for t in (1.0, 5.0, 10.0))
        assert got == counts, K


def test_gap_range_at_takt_one_period_k_1_5(raster):
    m = raster["K"] == 1.5
    assert raster["gap_pct"][m].min() == pytest.approx(1.90, abs=0.005) and raster["gap_pct"][m].max() == pytest.approx(12.23, abs=0.005)
    assert [round(raster["gap_pct"][cell(raster, L, 9, 1.5)], 2) for L in range(4)] == [3.26, 3.60, 2.86, 4.15]


def test_nine_cells_with_k_above_zero_have_zero_gap(raster):
    m = (raster["K"] > 0) & (raster["gap_pct"] < 0.005)
    assert int(m.sum()) == 9 and set(raster["K"][m]) == {0.5}


def test_largest_gap_is_212_8_percent(raster):
    assert raster["gap_pct"].max() == pytest.approx(212.76, abs=0.005)
    i = int(np.argmax(raster["gap_pct"]))
    assert (raster["L"][i], raster["ph"][i], raster["K"][i]) == (1, 1, 30.0)


def test_gap_is_larger_for_small_p_over_h(raster):
    """K = 5, Lieferzeit 1: 50,5 % bei p/h = 1 gegen 15,0 % bei p/h = 39."""
    assert raster["gap_pct"][cell(raster, 1, 1, 5.0)] == pytest.approx(50.5, abs=0.05)
    assert raster["gap_pct"][cell(raster, 1, 39, 5.0)] == pytest.approx(15.0, abs=0.05)


# ---------------------------------------------------------------------------------------------------------------- Ursache der großen Lücke


def test_base_stock_orders_in_95_percent_and_costs_are_linear_in_K(raster):
    assert 1 - np.exp(-MU) == pytest.approx(0.9502, abs=1e-4)
    assert raster["freqBase"].min() == pytest.approx(1 - np.exp(-MU), abs=1e-6) and raster["freqBase"].max() == pytest.approx(1 - np.exp(-MU), abs=1e-6)
    same_S = linear = 0
    for L in range(4):
        for ph in C.PH_OPTIONS:
            m0 = cell(raster, L, ph, 0.0)
            for K in C.K_GRID[1:]:
                i = cell(raster, L, ph, K)
                same_S += raster["SBase"][i] == raster["SBase"][m0]
                linear += abs(raster["gBase"][i] - raster["gBase"][m0] - K * (1 - np.exp(-MU))) < 1e-5
    assert same_S == 1440 and linear == 1440


def test_decomposition_for_lead_time_one_p_over_h_9_k_20():
    a = ev.analyse(ev.Settings(K=20.0, L=1, ph=9))
    assert (a.g, a.fixed_dp, a.rest_dp) == pytest.approx((12.18, 4.80, 7.38), abs=0.005)       # 12,18 = 4,80 + 7,38
    assert (a.g_base, a.fixed_base, a.rest_base) == pytest.approx((23.62, 19.00, 4.61), abs=0.005)
    assert a.freq == pytest.approx(0.24, abs=5e-4) and a.freq_base == pytest.approx(0.95, abs=5e-4)
    assert a.fixed_saved == pytest.approx(14.2, abs=0.05) and a.rest_extra == pytest.approx(2.8, abs=0.05)


def test_slope_of_dp_costs_over_K_is_the_order_frequency(raster):
    """Der Differenzenquotient von g über K liegt zwischen den Bestellanteilen der beiden Rasterpunkte (1.440 von 1.440 Schritten)."""
    ok = 0
    for L in range(4):
        for ph in C.PH_OPTIONS:
            for K in C.K_GRID[:-1]:
                a, b = cell(raster, L, ph, K), cell(raster, L, ph, K + 0.5)
                slope = (raster["g"][b] - raster["g"][a]) / 0.5
                ok += raster["freq"][b] - 1e-8 <= slope <= raster["freq"][a] + 1e-8
    assert ok == 1440


def test_s_never_rises_S_never_falls_costs_and_gap_never_fall_over_K(raster):
    bad = dict(s=0, S=0, g=0, gap=0)
    for L in range(4):
        for ph in C.PH_OPTIONS:
            idx = [cell(raster, L, ph, K) for K in C.K_GRID]
            for a, b in zip(idx[:-1], idx[1:]):
                bad["s"] += raster["s"][b] > raster["s"][a]
                bad["S"] += raster["S"][b] < raster["S"][a]
                bad["g"] += raster["g"][b] < raster["g"][a] - 1e-8
                bad["gap"] += raster["gap_pct"][b] < raster["gap_pct"][a] - 1e-5
    assert bad == dict(s=0, S=0, g=0, gap=0)


def test_s_and_S_over_K_for_p_over_h_9():
    for K, expected in ((0.0, [(4, 5), (8, 9), (12, 13), (16, 17)]), (30.0, [(1, 15), (4, 18), (8, 22), (11, 26)])):
        got = []
        for L in range(4):
            a = ev.analyse(ev.Settings(K=K, L=L, ph=9))
            got.append((a.structure.s, a.structure.S))
        assert got == expected


# ---------------------------------------------------------------------------------------------------------------- Schwelle K*


def test_kstar_range_over_the_24_cells_and_non_monotone():
    stars = {(L, ph): ev.k_star(L, ph) for L in range(4) for ph in C.PH_OPTIONS}
    vals = list(stars.values())
    assert min(vals) == pytest.approx(0.0136, abs=5e-4) and max(vals) == pytest.approx(0.8918, abs=5e-4)
    assert [stars[(L, 9)] for L in range(4)] == pytest.approx([0.8918, 0.5553, 0.2550, 0.0136], abs=6e-4)
    # nicht monoton: in Lieferzeit (p/h = 4: 0,51 / 0,29 / 0,10 / 0,15) und in p/h (Lieferzeit 0: 0,16 / 0,06 / 0,51 / ...)
    seq_L = [stars[(L, 4)] for L in range(4)]
    seq_ph = [stars[(0, ph)] for ph in C.PH_OPTIONS]
    assert not (all(np.diff(seq_L) >= 0) or all(np.diff(seq_L) <= 0))
    assert not (all(np.diff(seq_ph) >= 0) or all(np.diff(seq_ph) <= 0))


def test_gap_at_twice_kstar_is_small():
    gaps = []
    for L in range(4):
        for ph in C.PH_OPTIONS:
            ks = ev.k_star(L, ph)
            gaps.append(ev.analyse(ev.Settings(K=2 * ks, L=L, ph=ph)).gap)
    assert min(gaps) == pytest.approx(0.03, abs=0.005) and max(gaps) == pytest.approx(4.78, abs=0.005)


# ---------------------------------------------------------------------------------------------------------------- EOQ als Band


def test_lot_over_eoq_quantity_band(raster):
    m = raster["K"] >= 2.0
    ratio = raster["lot"][m] / np.sqrt(2 * raster["K"][m] * MU / C.H)
    assert int(m.sum()) == 1368
    assert ratio.min() == pytest.approx(0.52, abs=0.005) and ratio.max() == pytest.approx(1.65, abs=0.005) and ratio.mean() == pytest.approx(1.11, abs=0.005)
    assert int(((ratio >= 0.8) & (ratio <= 1.25)).sum()) == 1089
    m9 = m & (raster["ph"] == 9)
    r9 = raster["lot"][m9] / np.sqrt(2 * raster["K"][m9] * MU / C.H)
    assert r9.min() == pytest.approx(0.71, abs=0.005) and r9.max() == pytest.approx(1.31, abs=0.005)


# ---------------------------------------------------------------------------------------------------------------- Textfalle EOQ


def test_deterministic_eoq_numbers():
    assert E.extra_cost_lot_factor(2.0) == pytest.approx(25.0)
    assert E.extra_cost_parameter_factor(2.0) == pytest.approx(6.07, abs=0.005)
    assert E.extra_cost_parameter_factor(4.0) == pytest.approx(25.0)
    assert E.extra_cost_lot_factor(np.sqrt(2)) == pytest.approx(6.07, abs=0.005)


def test_lot_and_parameter_errors_in_the_model_of_the_demo():
    """20 Zellen (Lieferzeit 0 bis 3, K = 2, 5, 10, 20, 30, p/h = 9): halbe Losgröße +2,4 bis +19,5 %, doppelte +5,1 bis +23,8 %, Politik für 2 K
    +1,1 bis +6,7 %, für K / 2 0,0 bis 6,5 %, bezogen auf die Mehrkosten durch Fixkosten (Politik für 2 K) +3,9 bis +11,0 %."""
    half, dbl, x2, h2, extra = [], [], [], [], []
    for L in range(4):
        for K in (2.0, 5.0, 10.0, 20.0, 30.0):
            r = ev.lot_experiment(L, 9, K)
            half.append(r["half"]["pct"])
            dbl.append(r["double"]["pct"])
            x2.append(r["param_x2"]["pct"])
            h2.append(r["param_half"]["pct"])
            extra.append(r["param_x2"]["pct_of_extra"])
    assert (min(half), max(half)) == pytest.approx((2.41, 19.47), abs=0.01)
    assert (min(dbl), max(dbl)) == pytest.approx((5.09, 23.76), abs=0.01)
    assert (min(x2), max(x2)) == pytest.approx((1.13, 6.72), abs=0.01)
    assert (min(h2), max(h2)) == pytest.approx((0.00, 6.48), abs=0.01)
    assert (min(extra), max(extra)) == pytest.approx((3.87, 10.98), abs=0.01)


def test_cost_over_lot_size_is_flat_around_the_optimum():
    """Lieferzeit 1, K = 10, p/h = 9: Losgröße 7 und 9 höchstens 0,1 % über dem Optimum, 5: +5,0 %, 11: +2,6 %, 3: +20,9 % (ungerade Losgrößen)."""
    model = ev.model_for(1, 9, 40)
    g = D.rvi(model, 10.0)[0]
    costs = ev.lot_costs(model, 10.0, lots=[3, 5, 7, 9, 11])
    pct = {lot: 100 * (v[0] / g - 1) for lot, v in costs.items()}
    assert pct[7] < 0.1 and pct[9] < 0.15 and pct[7] >= -1e-9
    assert pct[5] == pytest.approx(5.0, abs=0.05) and pct[11] == pytest.approx(2.6, abs=0.05) and pct[3] == pytest.approx(20.9, abs=0.05)


# ---------------------------------------------------------------------------------------------------------------- Zyklus-Politik


def test_cycle_policy_is_7_to_18_percent_above_the_dp_for_k_5_and_more():
    gaps, Ts = [], {}
    for L in range(4):
        table = E.cycle_table(ev.model_for(L, 9, 40))
        for K in (5.0, 10.0, 20.0, 30.0):
            g_dp = ev.analyse(ev.Settings(K=K, L=L, ph=9)).g
            gc, T, S = table.best(K)
            gaps.append(ev.gap_percent(gc, g_dp))
            Ts[(L, K)] = T
    assert (min(gaps), max(gaps)) == pytest.approx((7.38, 17.45), abs=0.01)
    assert [Ts[(0, K)] for K in (5.0, 10.0, 20.0, 30.0)] == [2, 3, 4, 5]            # Lieferzeit 0: T = 2, 3, 4, 5


def test_cycle_policy_equals_base_stock_at_k_one_and_two():
    for L in range(4):
        table = E.cycle_table(ev.model_for(L, 9, 40))
        for K in (1.0, 2.0):
            assert table.best(K)[1] == 1


# ---------------------------------------------------------------------------------------------------------------- Konvexität


def test_convexity_counts(raster):
    k0 = raster["K"] == 0.0
    kp = ~k0
    assert int((raster["nonconv_v"][k0] == 0).sum()) == 24                          # K = 0: v konvex in 24 von 24 Zellen
    assert int((raster["nonconv_v"][kp] > 0).sum()) == 1440                        # K > 0: v nicht konvex in 1.440 von 1.440
    assert int(raster["nonconv_at_s"][kp].sum()) == 1440                           # die Stelle z = s ist immer darunter
    assert raster["hull_over_K"].max() == pytest.approx(0.975, abs=0.0005)         # Abweichung von der unteren Hülle höchstens 0,975 K
    assert np.nanmax(raster["kviol"]) < 1e-12                                       # K-Konvexität (Steigungsform) erfüllt bis auf Rundung


def test_W_is_convex_for_p_over_h_9_19_39_and_not_in_all_cells_below(raster):
    kp = raster["K"] > 0
    got = {ph: int((raster["nonconv_W"][kp & (raster["ph"] == ph)] > 0).sum()) for ph in C.PH_OPTIONS}
    assert got == {1: 208, 2: 156, 4: 28, 9: 0, 19: 0, 39: 0}


def test_value_function_of_the_standard_cell_is_not_convex_exactly_at_s():
    a = ev.analyse(ev.Settings(K=10.0, L=1, ph=9))
    cell_ = ev.convexity_cell(1, 9, 10.0)
    assert cell_["nonconvex_v"] == [6] and a.structure.s == 6 and cell_["nonconvex_W"] == []
    assert cell_["hull_defect"] == pytest.approx(8.24, abs=0.005)


# ---------------------------------------------------------------------------------------------------------------- Aufzählung


def test_enumeration_pairs_near_the_optimum_over_40_cells():
    cells = [(L, 9.0, K) for L in range(4) for K in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)] + [(L, p, K) for L in (0, 3) for p in (1.0, 39.0) for K in (5.0, 20.0)]
    n01, n1, n5 = [], [], []
    parts = {}
    for L, ph, K in cells:
        model = ev.model_for(L, int(ph), 40)
        if (L, ph) not in parts:
            parts[(L, ph)] = P.sS_parts(model)
        en = ev.enumeration(model, K, parts[(L, ph)])
        assert abs(en["best_g"] - D.rvi(model, K)[0]) < 1e-9  # die Aufzählung trifft die DP in allen 40 Zellen
        n01.append(en["n_within"][0.1])
        n1.append(en["n_within"][1.0])
        n5.append(en["n_within"][5.0])
    assert len(cells) == 40
    assert (min(n01), max(n01)) == (1, 3) and (min(n1), max(n1)) == (1, 8) and (min(n5), max(n5)) == (4, 47)


# ---------------------------------------------------------------------------------------------------------------- Bestellgrenze, Horizont


def test_smallest_order_limit_with_boundary_mass_below_1e_4():
    """K = 30: die kleinste Bestellgrenze mit Anteil der Zeit an der Grenze unter 1e-4 (Schritt 1)."""
    expected = {(9, 0): 23, (9, 1): 23, (9, 2): 23, (9, 3): 24, (1, 0): 27, (1, 1): 27, (1, 2): 27, (1, 3): 28}
    for (ph, L), q_exp in expected.items():
        found = None
        for Qc in range(15, 41):
            a = ev.analyse(ev.Settings(K=30.0, L=L, ph=ph, Q=Qc))
            if a.masses["qcap"] < 1e-4:
                found = Qc
                break
        assert found == q_exp, (ph, L)


def test_larger_bounds_do_not_change_the_costs():
    for L, ph, K in ((3, 1, 30.0), (3, 39, 30.0), (0, 1, 30.0), (2, 4, 20.0)):
        small = D.rvi(ev.model_for(L, ph, 40), K)[0]
        big = D.rvi(D.build_model(D.ModelSpec(L=L, p=float(ph), Q=80, zmin=-80, ymax=100)), K)[0]
        assert big == pytest.approx(small, abs=1e-9)


def test_finite_horizon_keeps_the_sS_form_in_2160_pairs():
    """36 Zellen (Lieferzeit 0 bis 3, p/h = 1, 9, 39, K = 2, 10, 30) mal n = 1 bis 60 Restperioden: Form (s, S) im Fenster z >= -10."""
    total = pure = no_order = 0
    for L in range(4):
        for ph in (9, 1, 39):
            for K in (2.0, 10.0, 30.0):
                for st in ev.finite_path(ev.Settings(K=K, L=L, ph=ph)):
                    total += 1
                    pure += st.pure
                    no_order += st.s is None
    assert total == 2160 and pure == 2160
    # p/h = 1, K = 30: bei n = 1 (Lieferzeit 0 auch bei n = 2) wird im Fenster nicht bestellt (in den Zellen mit kleinem p/h und hohem K)
    assert no_order >= 4
    assert ev.finite_path(ev.Settings(K=30.0, L=0, ph=1))[0].s is None and ev.finite_path(ev.Settings(K=30.0, L=0, ph=1))[1].s is None


def test_finite_horizon_converges_to_the_stationary_policy_for_standard_cells():
    stat = ev.analyse(ev.Settings(K=2.0, L=0, ph=9)).structure
    path = ev.finite_path(ev.Settings(K=2.0, L=0, ph=9))
    assert (path[1].s, path[1].S) == (stat.s, stat.S) == (3, 6)  # ab n = 2 gleich der stationären Politik
    assert (path[0].s, path[0].S) == (3, 5)


# ---------------------------------------------------------------------------------------------------------------- Rechenzeit


def test_one_dp_cell_takes_a_few_milliseconds():
    import time
    model = ev.model_for(2, 9, 40)
    t = time.perf_counter()
    for _ in range(20):
        D.rvi(model, 5.0)
    assert (time.perf_counter() - t) / 20 < 0.25  # großzügige Schranke (einige ms, unter Last mehr)


# ---------------------------------------------------------------------------------------------------------------- Texte


def test_assumptions_text_of_the_app_is_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert C.ASSUMPTIONS_MARKDOWN.strip() in readme

