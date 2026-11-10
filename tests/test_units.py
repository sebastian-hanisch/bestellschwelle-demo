"""Einheiten, je Funktion eine Regel, an der Mini-Instanz (Handbeispiel, Handrechnung in tests/conftest.py)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import bsw_convex as CV
import bsw_cost as COST
import bsw_demand as DEM
import bsw_dp as D
import bsw_eoq as E
import bsw_hand as H
import bsw_policy as P

from conftest import HAND_K, HAND_SPEC

PMF = [0.25, 0.5, 0.25]


# ---- Nachfrage ---------------------------------------------------------------------------------------------------------------------------


def test_poisson_pmf_sums_to_one_and_has_the_right_values():
    pmf = DEM.poisson_pmf(3.0, 30)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-15)
    assert pmf[0] == pytest.approx(math.exp(-3.0), rel=1e-14)
    assert pmf[3] == pytest.approx(27.0 * math.exp(-3.0) / 6.0, rel=1e-13)
    assert (np.arange(31) * pmf).sum() == pytest.approx(3.0, abs=1e-12)


def test_poisson_pmf_edge_cases():
    assert list(DEM.poisson_pmf(0.0, 3)) == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        DEM.poisson_pmf(-1.0, 5)


def test_lead_demand_of_the_mini_instance():
    """Zwei Perioden mit ¼, ½, ¼: (1, 4, 6, 4, 1) / 16."""
    assert DEM.lead_demand(PMF, 2) == pytest.approx([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])
    assert list(DEM.lead_demand(PMF, 0)) == [1.0]
    assert DEM.lead_demand(PMF, 1) == pytest.approx(PMF)
    with pytest.raises(ValueError):
        DEM.lead_demand(PMF, -1)


def test_critical_quantile_of_the_mini_instance():
    """F = ¼, ¾, 1; p/(p + h) = 4/5: das kleinste y mit F(y) >= 0,8 ist 2; bei p/(p+h) = 1/2 ist es 1."""
    assert DEM.critical_quantile(PMF, 4.0, 1.0) == 2
    assert DEM.critical_quantile(PMF, 1.0, 1.0) == 1
    assert DEM.critical_quantile(PMF, 1.0, 3.0) == 0  # F(0) = 1/4 >= 1/4 (inklusive Grenze)


# ---- Kosten einer Periode ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("y,expected", [(-1, 8.0), (0, 4.0), (1, 1.25), (2, 1.0), (3, 2.0), (4, 3.0)])
def test_period_cost_G_of_the_mini_instance(y, expected):
    assert COST.period_cost_G(y, PMF, 1.0, 4.0) == pytest.approx(expected)


def test_period_cost_G_accepts_arrays():
    assert COST.period_cost_G(np.array([0, 1, 2]), PMF, 1.0, 4.0) == pytest.approx([4.0, 1.25, 1.0])


# ---- Rückwärtsschritt --------------------------------------------------------------------------------------------------------------------


def test_backup_reproduces_C1(hand_W1):
    """Positionen -3 ... 4, K = 2: bestellen bis y = 2 bei x <= 0 (Kosten 3), sonst nicht."""
    v, q = D.backup(np.array([float(w) for w in hand_W1]), HAND_K, 7)
    assert v == pytest.approx([3, 3, 3, 3, 1.25, 1, 2, 3])
    assert list(q) == [5, 4, 3, 2, 0, 0, 0, 0]


def test_backup_respects_the_order_limit_and_the_upper_end():
    W = np.array([16.0, 12.0, 8.0, 4.0, 1.25, 1.0, 2.0, 3.0])
    _, q = D.backup(W, 2.0, 2)  # höchstens 2 Stück je Bestellung: bei x = -3 geht es nur bis -1, bei x = 0 bis 2
    assert list(q) == [2, 2, 2, 2, 0, 0, 0, 0]
    v, q = D.backup(W, 2.0, 7)
    assert q[-1] == 0 and v[-1] == 3.0  # am oberen Ende gibt es nichts zu bestellen


def test_backup_ties_prefer_not_to_order_and_the_smallest_quantity():
    W = np.array([3.0, 1.0, 1.0, 1.0])
    v, q = D.backup(W, 2.0, 3)  # Bestellen kostet 2 + 1 = 3 = W[0]: Gleichstand, nicht bestellen
    assert q[0] == 0 and v[0] == 3.0
    v, q = D.backup(np.array([4.0, 1.0, 1.0, 1.0]), 2.0, 3)  # Bestellen lohnt; drei gleich gute Ziele: das nächste (Menge 1)
    assert q[0] == 1 and v[0] == 3.0


def test_finite_horizon_reproduces_C2(hand_model):
    v, q, W = D.finite_horizon(hand_model, HAND_K, 2)
    assert v == pytest.approx([37 / 8] * 4 + [61 / 16, 21 / 8, 53 / 16, 5.0])
    assert W[3:] == pytest.approx([7.0, 61 / 16, 21 / 8, 53 / 16, 5.0])  # W₂(y), y = 0 ... 4
    assert list(q[:4]) == [5, 4, 3, 2] and not q[4:].any()  # bestelle bis 2 bei x <= 0
    with pytest.raises(ValueError):
        D.finite_horizon(hand_model, HAND_K, 0)


def test_rvi_of_the_mini_instance_gives_two(hand_model):
    g, q, h, W, it = D.rvi(hand_model, HAND_K)
    assert g == pytest.approx(2.0, abs=1e-9)
    assert list(q[:4]) == [5, 4, 3, 2] and not q[4:].any()  # (0, 2): bestelle bis 2, sobald z <= 0
    assert h[hand_model.ref] == 0.0 and it < 200


def test_model_rejects_invalid_specs():
    with pytest.raises(ValueError):
        D.Model(D.ModelSpec(zmin=1, ymax=5))
    with pytest.raises(ValueError):
        D.Model(D.ModelSpec(Q=0))
    with pytest.raises(ValueError):
        D.Model(D.ModelSpec(pmf=(0.5, 0.2)))


# ---- Politik -----------------------------------------------------------------------------------------------------------------------------


def test_evaluate_the_two_policies_of_the_mini_instance(hand_model):
    g02, pi = P.evaluate(hand_model, HAND_K, P.pol_sS(hand_model, 0, 2))
    assert g02 == pytest.approx(2.0, abs=1e-12)
    # wiederkehrende Positionen -1, 0, 1, 2 mit 1/10, 7/20, 2/5, 3/20
    assert {int(z): round(float(p), 12) for z, p in zip(hand_model.z, pi) if p > 1e-12} == {-1: 0.1, 0: 0.35, 1: 0.4, 2: 0.15}
    g12, _ = P.evaluate(hand_model, HAND_K, P.pol_base_stock(hand_model, 2))
    assert g12 == pytest.approx(2.5, abs=1e-12)


def test_evaluate_parts_splits_fixed_and_rest_costs(hand_model):
    g_ops, f, _ = P.evaluate_parts(hand_model, P.pol_sS(hand_model, 0, 2))
    assert f == pytest.approx(9 / 20)           # Bestellung bei z <= 0: 1/10 + 7/20
    assert g_ops == pytest.approx(1.1)          # 3/5 · 1 + 2/5 · 5/4
    assert g_ops + HAND_K * f == pytest.approx(2.0)


def test_structure_of_the_optimal_mini_policy(hand_model):
    g, q, *_ = D.rvi(hand_model, HAND_K)
    _, pi = P.evaluate(hand_model, HAND_K, q)
    st = P.structure(hand_model, q, P.reached(pi))
    assert (st.s, st.S, st.pure) == (0, 2, True)


def test_structure_detects_impure_and_idle_policies(hand_model):
    z = hand_model.z
    q = P.pol_sS(hand_model, 0, 2)
    all_states = np.ones(len(z), dtype=bool)
    q_bad = q.copy()
    q_bad[list(z).index(-3)] = 3  # anderes Ziel (0) bei -3
    assert P.structure(hand_model, q_bad, all_states).pure is False
    q_gap = q.copy()
    q_gap[list(z).index(-1)] = 0  # bei -1 nicht bestellt, bei 0 aber doch: keine Schwelle
    assert P.structure(hand_model, q_gap, all_states).pure is False
    idle = np.zeros(len(z), dtype=int)
    assert P.structure(hand_model, idle, all_states) == P.Structure(None, None, True)


def test_base_stock_table_of_the_mini_instance(hand_model):
    tab = P.base_stock_table(hand_model, range(-2, 4))
    g, S, f = tab.best(HAND_K)
    assert (S, f) == (2, pytest.approx(0.75)) and g == pytest.approx(2.5)
    assert tab.best(0.0)[1] == 2  # ohne Fixkosten: Quantil


def test_sS_parts_are_linear_in_K(hand_model):
    s, S, g_ops, f = P.sS_parts(hand_model, s_lo=-1, s_hi=2, S_lo=0, S_hi=3)
    i = [k for k in range(len(s)) if (s[k], S[k]) == (0, 2)][0]
    assert g_ops[i] + HAND_K * f[i] == pytest.approx(2.0, abs=1e-12)
    assert len(s) == 10  # alle Paare -1 <= s < S <= 3 mit s <= 2 (1 + 2 + 3 + 4)


def test_boundary_masses_are_small_for_the_demo_model():
    import bsw_evaluation as ev
    a = ev.analyse(ev.Settings(K=5.0, L=1, ph=9))
    assert a.masses["qcap"] < 1e-12 and a.masses["mass_lo"] < 1e-12 and a.masses["mass_hi"] < 1e-12
    assert a.masses["z_lo"] >= -26 and a.masses["z_hi"] <= 28


def test_pol_sS_is_clipped_at_the_upper_state_limit(hand_model):
    q = P.pol_sS(hand_model, 10, 10)  # Schwelle und Ziel über der oberen Grenze 4
    assert (hand_model.z + q).max() == 4


# ---- Konvexität --------------------------------------------------------------------------------------------------------------------------


C2 = [37 / 8] * 4 + [61 / 16, 21 / 8, 53 / 16, 5.0]
Z = list(range(-3, 5))


def test_second_difference_of_the_mini_C2():
    d2 = CV.second_differences(C2)
    assert d2[2] == pytest.approx(-13 / 16)  # an der Stelle x = 0
    assert CV.nonconvex_points(Z, C2) == [0, 1]  # bei x = 1: 21/8 - 2 · 61/16 + 37/8 = -3/8
    assert CV.nonconvex_points(Z, [0, 1, 4, 9]) == []  # konvex


def test_lower_hull_and_defect_of_the_mini_C2():
    hull = CV.lower_hull(C2)
    assert all(h <= c + 1e-12 for h, c in zip(hull, C2))
    d, where = CV.hull_defect(Z, C2)
    assert where == 0 and 0 < d <= HAND_K  # die Abweichung bleibt unter K
    assert CV.hull_defect([0, 1, 2], [0.0, 1.0, 4.0]) == (0.0, None)  # konvex: keine Abweichung


def test_k_convexity_of_the_mini_C2():
    assert CV.k_convex_violation(C2, HAND_K) <= 1e-12     # K-konvex mit K = 2
    assert CV.k_convex_violation(C2, 0.0) > 0.1           # nicht konvex (K = 0)
    assert CV.k_convex_violation([1.0, 2.0], 0.0) == 0.0  # weniger als drei Stellen


# ---- EOQ und Zyklus ----------------------------------------------------------------------------------------------------------------------


def test_eoq_formulas():
    assert E.eoq_quantity(2.0, 3.0, 1.0) == pytest.approx(math.sqrt(12.0))
    assert E.eoq_cost(6.0, 5.0, 3.0, 1.0) == pytest.approx(5.5)       # 5·3/6 + 6/2
    assert E.eoq_cycle_periods(1.5, 3.0, 1.0) == pytest.approx(1.0)   # K = h·mu/2: Takt eine Periode
    assert E.flatness(2.0) == pytest.approx(1.25)
    assert E.extra_cost_lot_factor(2.0) == pytest.approx(25.0)
    assert E.extra_cost_lot_factor(0.5) == pytest.approx(25.0)        # symmetrisch in f und 1/f
    assert E.extra_cost_parameter_factor(2.0) == pytest.approx(100 * ((math.sqrt(2) + 1 / math.sqrt(2)) / 2 - 1))
    assert E.extra_cost_parameter_factor(4.0) == pytest.approx(25.0)  # Parameter ×4 = Losgröße ×2


def test_cycle_table_of_the_mini_instance(hand_model):
    tab = E.cycle_table(hand_model, T_max=3, S_range=range(-2, 5))
    # T = 1, S = 2: K · P(D > 0) + G(2) = 2 · 3/4 + 1 = 5/2 (das ist die Base-Stock-Politik (1, 2))
    assert tab.cost(HAND_K, 1, 2) == pytest.approx(2.5)
    # T = 2, S = 2 (von Hand): [K · (1 - 1/16) + G(2) + E G(2 - D)] / 2 = [15/8 + 1 + 15/8] / 2 = 19/8, besser als jede Base-Stock-Politik
    assert tab.cost(HAND_K, 2, 2) == pytest.approx(19 / 8)
    g, T, S = tab.best(HAND_K)
    assert (T, S) == (2, 2) and g == pytest.approx(19 / 8)
    assert tab.freqfac[0] == pytest.approx(0.75) and tab.freqfac[1] == pytest.approx((1 - 1 / 16) / 2)


# ---- Handbeispiel in Brüchen -------------------------------------------------------------------------------------------------------------


def test_hand_module_matches_the_hand_calculation(hand_W1):
    assert [H.period_cost(y) for y in range(-3, 5)] == hand_W1
    (C1, _, _), (C2h, tgt2, W2) = H.backward(2)
    assert [C1[x] for x in H.STATES] == [F(3)] * 4 + [F(5, 4), F(1), F(2), F(3)]
    assert [W2[y] for y in range(0, 5)] == [F(7), F(61, 16), F(21, 8), F(53, 16), F(5)]
    assert [C2h[x] for x in H.STATES] == [F(37, 8)] * 4 + [F(61, 16), F(21, 8), F(53, 16), F(5)]
    assert tgt2[0] == 2 and tgt2[1] == 1
    assert H.second_difference(C2h, 0) == F(-13, 16)


def test_hand_stationary_costs():
    g, pi = H.stationary_cost(0, 2)
    assert g == F(2) and pi == {-1: F(1, 10), 0: F(7, 20), 1: F(2, 5), 2: F(3, 20)}
    assert H.stationary_cost(1, 2)[0] == F(5, 2)
    pairs = H.all_pairs()
    assert len(pairs) == 10
    assert H.best_pair(pairs) == ((0, 2), F(2)) and H.best_base_stock(pairs) == ((1, 2), F(5, 2))
    assert H.order_frequency(0, 2) == F(9, 20) and H.order_frequency(1, 2) == F(3, 4)
    assert H.frac_text(F(4, 2)) == "2" and H.frac_text(F(61, 16)) == "61/16"
