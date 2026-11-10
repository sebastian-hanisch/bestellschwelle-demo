"""Randfälle (DEMO-PLAYBOOK §12): K = 0, Lieferzeit 0, kleinstes und größtes p/h, Bestellgrenze 10, kleinste Zustandsräume, nie bestellen."""

import numpy as np
import pytest

import bsw_constants as C
import bsw_demand as DEM
import bsw_dp as D
import bsw_evaluation as ev
import bsw_policy as P


def test_without_fixed_costs_dp_is_base_stock_at_the_newsvendor_quantile():
    """K = 0: in allen 24 Zellen ist die DP-Politik Base-Stock (s = S - 1) mit S = kleinstes y mit F(y) >= p/(p + h) der Nachfrage über L + 1 Perioden."""
    for L in range(4):
        for ph in C.PH_OPTIONS:
            model = ev.model_for(L, ph, 40)
            a = ev.analyse(ev.Settings(K=0.0, L=L, ph=ph))
            S_expected = DEM.critical_quantile(model.dn, float(ph), 1.0)
            assert (a.structure.s, a.structure.S) == (S_expected - 1, S_expected), (L, ph)
            assert a.gap == pytest.approx(0.0, abs=1e-8)


def test_exactly_two_states():
    """Zwei Positionen {0, 1}: der kleinste Raum, in dem es etwas zu entscheiden gibt (kein Zweig n < 2)."""
    model = D.Model(D.ModelSpec(L=0, Q=1, zmin=0, ymax=1, pmf=(0.5, 0.5), p=4.0))
    g, q, h, W, it = D.rvi(model, 1.0)
    assert np.isfinite(g) and q.shape == (2,) and it < 5000
    gg, _ = P.evaluate(model, 1.0, q)
    assert gg == pytest.approx(g, abs=1e-9)


def test_lead_time_zero_and_three_are_different_cells():
    a0 = ev.analyse(ev.Settings(K=10.0, L=0, ph=9))
    a3 = ev.analyse(ev.Settings(K=10.0, L=3, ph=9))
    assert (a0.structure.s, a0.structure.S) != (a3.structure.s, a3.structure.S)  # Zweig-Test: die Lieferzeit ändert s und S
    assert a3.g > a0.g


def test_ratio_p_over_h_moves_S():
    low = ev.analyse(ev.Settings(K=5.0, L=1, ph=1))
    high = ev.analyse(ev.Settings(K=5.0, L=1, ph=39))
    assert high.structure.S > low.structure.S and high.structure.s > low.structure.s


def test_fixed_costs_change_the_policy_against_k_zero():
    zero = ev.analyse(ev.Settings(K=0.0, L=1, ph=9))
    some = ev.analyse(ev.Settings(K=5.0, L=1, ph=9))
    assert (zero.structure.S - zero.structure.s) == 1 and (some.structure.S - some.structure.s) > 1
    assert some.freq < zero.freq


def test_order_limit_ten_binds_at_high_fixed_costs():
    """Zweig-Test: Q = 10 ändert die Kosten bei K = 30 deutlich (die Grenze greift: Anteil der Zeit an der Grenze über 1e-4), bei K = 2 um weniger als 0,001."""
    free = ev.analyse(ev.Settings(K=30.0, L=1, ph=9, Q=40))
    capped = ev.analyse(ev.Settings(K=30.0, L=1, ph=9, Q=10))
    assert capped.g > free.g + 0.01 and capped.masses["qcap"] > 1e-4
    assert capped.structure.pure is False and free.structure.pure is True  # die greifende Grenze zerstört die reine Form
    assert free.masses["qcap"] < 1e-12
    low_free = ev.analyse(ev.Settings(K=2.0, L=1, ph=9, Q=40)).g
    low_cap = ev.analyse(ev.Settings(K=2.0, L=1, ph=9, Q=10)).g
    assert 0 <= low_cap - low_free < 1e-3  # bei K = 2 greift die Grenze kaum (Mehrkosten unter 0,001)


def test_finite_horizon_changes_the_policy_at_n_equals_one():
    stat = ev.analyse(ev.Settings(K=10.0, L=1, ph=9)).structure
    fin = ev.finite_policy(ev.Settings(K=10.0, L=1, ph=9, horizon=1)).structure
    assert (fin.s, fin.S) != (stat.s, stat.S)
    assert (fin.s, fin.S) == (4, 9)  # n = 1, Lieferzeit 1, K = 10, p/h = 9 (Messung der Vorabrechnung)


def test_finite_horizon_with_huge_fixed_costs_never_orders_in_the_window():
    fp = ev.finite_policy(ev.Settings(K=1e6, L=0, ph=9, horizon=1))
    assert fp.structure == P.Structure(None, None, True)


def test_huge_fixed_costs_stationary_policy_runs():
    """K = 1e6: Bestellen lohnt nur ganz unten; die Rechnung darf nicht abstürzen, und die Kosten bleiben endlich."""
    model = ev.model_for(1, 9, 40)
    g, q, *_ = D.rvi(model, 1e6)
    assert np.isfinite(g)


def test_zero_demand_model_has_no_costs_when_stock_is_zero():
    model = D.Model(D.ModelSpec(L=0, Q=3, zmin=-2, ymax=3, pmf=(1.0,), p=4.0))
    g, q, *_ = D.rvi(model, 5.0)
    assert g == pytest.approx(0.0, abs=1e-9)


def test_k_grid_and_options_are_consistent():
    assert len(C.K_GRID) == 61 and C.K_GRID[0] == 0.0 and C.K_GRID[-1] == 30.0
    assert C.DEFAULT_PH in C.PH_OPTIONS and C.DEFAULT_HORIZON in C.HORIZON_OPTIONS
    assert C.Q_MIN <= C.DEFAULT_Q <= C.Q_MAX


def test_threshold_helpers_with_missing_thresholds():
    sw = ev.sweep_K(1, 9)
    assert ev.threshold_K(sw, 1e9) is None
    assert ev.threshold_K(sw, 1.0) == 1.0
    assert ev.k_star(1, 9, hi=0.0001) is None  # K* liegt über der Obergrenze
