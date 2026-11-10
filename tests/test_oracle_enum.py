"""Orakel B: Aufzählung aller (s, S) gegen die DP.

Die DP (Wertiteration) und die Aufzählung (Auswertung jedes Paares über die stationäre Verteilung) sind zwei Rechenwege zum selben Optimum, wenn die
optimale Politik die (s, S)-Form hat. Zusätzlich ein dritter, unabhängiger Weg: eigene Kette mit scipy-Poisson und Potenziermethode statt der
Faltung und des linearen Gleichungssystems der Demo (kleines Fenster um das Optimum)."""

import numpy as np
import pytest
from scipy.stats import poisson

import bsw_dp as D
import bsw_evaluation as ev
import bsw_policy as P

CELLS = [(L, 9.0, K) for L in (0, 1, 2, 3) for K in (0.0, 1.0, 5.0, 20.0)] + [(L, p, K) for L in (0, 3) for p in (1.0, 39.0) for K in (5.0, 20.0)]


@pytest.fixture(scope="module")
def parts_cache():
    cache = {}

    def get(L, ph):
        if (L, ph) not in cache:
            cache[(L, ph)] = P.sS_parts(ev.model_for(L, int(ph), 40))
        return cache[(L, ph)]

    return get


def test_enumeration_hits_the_dp_optimum(parts_cache):
    worst = 0.0
    for L, ph, K in CELLS:
        model = ev.model_for(L, int(ph), 40)
        g = D.rvi(model, K)[0]
        en = ev.enumeration(model, K, parts_cache(L, ph))
        assert en["n_pairs"] == 1378
        worst = max(worst, abs(en["best_g"] - g))
    assert worst < 1e-9


def test_k_free_parts_equal_direct_evaluation_with_K(parts_cache):
    """Die schnelle Auswertung g = g_ops + K·f stimmt mit der direkten Auswertung der Kette mit K im Bestellschritt überein."""
    model = ev.model_for(1, 9, 40)
    s, S, g_ops, f = parts_cache(1, 9.0)
    for K in (0.0, 3.5, 30.0):
        for i in range(0, len(s), 97):
            g_direct, _ = P.evaluate(model, K, P.pol_sS(model, int(s[i]), int(S[i])))
            assert g_ops[i] + K * f[i] == pytest.approx(g_direct, abs=1e-12)


def independent_cost(L, p, K, s, S, mean=3.0, h=1.0, zmin=-40, ymax=60, iters=6000):
    """Eigener Aufbau: Kosten je Position aus Poisson(3 (L+1)) (scipy), Übergänge aus Poisson(3), Potenziermethode für die stationäre Verteilung."""
    z = np.arange(zmin, ymax + 1)
    n = len(z)
    k = np.arange(0, 400)
    lead = poisson.pmf(k, mean * (L + 1))
    one = poisson.pmf(np.arange(0, 60), mean)
    P_ = np.zeros((n, n))
    cost = np.zeros(n)
    for i, zi in enumerate(z):
        y = S if zi <= s else zi
        cost[i] = float((lead * (h * np.maximum(y - k, 0) + p * np.maximum(k - y, 0))).sum()) + (K if zi <= s else 0.0)
        for d, pr in enumerate(one):
            P_[i, max(y - d, zmin) - zmin] += pr
    pi = np.zeros(n)
    pi[-zmin + S] = 1.0
    for _ in range(iters):
        pi = 0.5 * pi + 0.5 * pi @ P_
    return float(pi @ cost)


@pytest.mark.parametrize("L,ph,K", [(0, 9.0, 5.0), (1, 9.0, 20.0), (2, 1.0, 5.0), (3, 39.0, 10.0)])
def test_dp_policy_is_best_among_independently_evaluated_pairs(L, ph, K):
    model = ev.model_for(L, int(ph), 40)
    g, q, *_ = D.rvi(model, K)
    _, pi = P.evaluate(model, K, q)
    st = P.structure(model, q, P.reached(pi))
    g_ind = independent_cost(L, ph, K, st.s, st.S)
    assert g_ind == pytest.approx(g, abs=1e-6)
    for ds, dS in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 2), (2, -2)):
        s2, S2 = st.s + ds, st.S + dS
        if s2 < S2:
            assert independent_cost(L, ph, K, s2, S2) >= g - 1e-6
