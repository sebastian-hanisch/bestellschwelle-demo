"""Orakel G: das Handbeispiel in einem unabhängigen Rechenweg.

Die Demo rechnet das Handbeispiel (bsw_hand.py) mit Gauß-Jordan in Brüchen und die Zahlenrechnung (bsw_dp.py) mit Gleitkomma. Dieser Test rechnet
ohne beides: die Restkosten über alle Politiken per Aufzählung der Zustandsfolgen (Fractions, rekursiv) und die stationären Kosten jedes Paares durch
Potenzieren der Übergangsmatrix. Die erwarteten Werte stehen von Hand gerechnet im Plan (Brüche)."""

from fractions import Fraction as F

import numpy as np
import pytest

import bsw_dp as D
import bsw_hand as H
import bsw_policy as P

from conftest import HAND_K

PMF = {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}
XMIN, XMAX = -3, 4


def L(y):
    return sum(p * (max(y - d, 0) + 4 * max(d - y, 0)) for d, p in PMF.items())


def C(n, x, K=F(2)):
    """Restkosten bei n Perioden: rekursiv über alle Ziele y >= x (kein Tabellenaufbau, andere Struktur als bsw_hand)."""
    if n == 0:
        return F(0)
    options = []
    for y in range(x, XMAX + 1):
        fixed = K if y > x else F(0)
        options.append(fixed + L(y) + sum(p * C(n - 1, max(y - d, XMIN)) for d, p in PMF.items()))
    return min(options)


def test_two_period_costs_by_recursion_equal_the_hand_values():
    assert [C(1, x) for x in range(-3, 5)] == [3, 3, 3, 3, F(5, 4), 1, 2, 3]
    assert [C(2, x) for x in range(-3, 5)] == [F(37, 8)] * 4 + [F(61, 16), F(21, 8), F(53, 16), 5]


def stationary_float(s, S, iters=4000):
    states = list(range(XMIN, S + 1))
    n = len(states)
    P_ = np.zeros((n, n))
    cost = np.zeros(n)
    for i, z in enumerate(states):
        y = S if z <= s else z
        cost[i] = float(L(y)) + (2.0 if z <= s else 0.0)
        for d, p in PMF.items():
            P_[i, states.index(max(y - d, XMIN))] += float(p)
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        pi = 0.5 * pi + 0.5 * pi @ P_  # träge Kette: aperiodisch
    return float(pi @ cost)


@pytest.mark.parametrize("s,S", [(s, S) for S in range(0, 4) for s in range(-1, S)])
def test_each_pair_equals_power_iteration(s, S):
    assert float(H.stationary_cost(s, S)[0]) == pytest.approx(stationary_float(s, S), abs=1e-9)


def test_best_pair_and_the_25_percent_gap_by_enumeration():
    costs = {(s, S): stationary_float(s, S) for S in range(0, 4) for s in range(-1, S)}
    best = min(costs, key=costs.get)
    base = min((k for k in costs if k[0] == k[1] - 1), key=costs.get)
    assert best == (0, 2) and costs[best] == pytest.approx(2.0, abs=1e-9)
    assert base == (1, 2) and costs[base] == pytest.approx(2.5, abs=1e-9)
    assert 100 * (costs[base] / costs[best] - 1) == pytest.approx(25.0, abs=1e-6)


def test_base_stock_orders_in_three_quarters_of_the_periods_by_hand():
    """Bestellung, wenn die Nachfrage der Vorperiode > 0 war: 1 - 1/4 = 3/4; Kosten: 3/2 Fixkosten + G(2) = 1 gleich 5/2."""
    assert 1 - PMF[0] == F(3, 4)
    assert F(3, 4) * 2 + L(2) == F(5, 2)


def test_numeric_dp_agrees_with_the_fractions(hand_model):
    g, q, *_ = D.rvi(hand_model, HAND_K)
    assert g == pytest.approx(2.0, abs=1e-9)
    v, qn, _ = D.finite_horizon(hand_model, HAND_K, 2)
    assert list(v) == pytest.approx([float(C(2, x)) for x in range(-3, 5)])
    g_policy, _ = P.evaluate(hand_model, HAND_K, P.pol_sS(hand_model, 0, 2))
    assert g_policy == pytest.approx(2.0, abs=1e-12)


def test_value_iteration_over_finite_horizons_converges_to_two(hand_model):
    """C_n(1) - C_(n-1)(1) strebt gegen die Durchschnittskosten 2."""
    v39 = D.finite_horizon(hand_model, HAND_K, 39)[0]
    v40 = D.finite_horizon(hand_model, HAND_K, 40)[0]
    assert v40[4] - v39[4] == pytest.approx(2.0, abs=1e-8)  # Position 1
