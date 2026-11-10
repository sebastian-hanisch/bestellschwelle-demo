"""Orakel E und F: EOQ-Formel und Zyklus-Politik gegen unabhängige Rechenwege.

E: die Formel Q* = sqrt(2 K D / h) gegen Aufzählung der Kurve (Schritt 0,01) und die Sägezahn-Kosten durch numerische Integration des Bestandsverlaufs;
   die Flachheit (f + 1/f) / 2 gegen direktes Einsetzen.
F: die Zyklus-Formel gegen eine Phasenkette (Zustand = Phase im Zyklus und Position) und T = 1 gegen die Base-Stock-Auswertung."""

import numpy as np
import pytest

import bsw_eoq as E
import bsw_evaluation as ev
import bsw_policy as P

D_, H_ = 3.0, 1.0


@pytest.mark.parametrize("K", [0.5, 1, 2, 3, 5, 8, 12, 20, 30])
def test_eoq_formula_equals_enumeration(K):
    Q = np.arange(0.01, 80.0, 0.01)
    cost = K * D_ / Q + H_ * Q / 2
    assert Q[cost.argmin()] == pytest.approx(E.eoq_quantity(K, D_, H_), abs=0.006)


def test_saw_tooth_integration_equals_the_closed_form():
    for K in (2.0, 5.0, 20.0):
        for Q in (2.0, 5.0, 9.0, 20.0):
            T = Q / D_
            t = np.linspace(0.0, T, 200001)
            inv = Q - D_ * t
            integral = np.sum((inv[1:] + inv[:-1]) / 2 * np.diff(t))  # Trapezregel
            assert (K + H_ * integral) / T == pytest.approx(E.eoq_cost(Q, K, D_, H_), abs=1e-9)


def test_flatness_equals_direct_substitution():
    K = 5.0
    qs = E.eoq_quantity(K, D_, H_)
    for f in (0.5, 1.1, 1.5, np.sqrt(2), 2.0, 4.0):
        direct = E.eoq_cost(f * qs, K, D_, H_) / E.eoq_cost(qs, K, D_, H_)
        assert direct == pytest.approx(E.flatness(f), rel=1e-12)
    # Parameter K/h um den Faktor c daneben: Losgröße für c·K, bewertet mit K
    for c in (2.0, 0.5, 4.0):
        qp = E.eoq_quantity(c * K, D_, H_)
        assert 100 * (E.eoq_cost(qp, K, D_, H_) / E.eoq_cost(qs, K, D_, H_) - 1) == pytest.approx(E.extra_cost_parameter_factor(c), rel=1e-9)


def phase_chain_cost(model, K, T, S):
    """Phasenkette: Zustand (Phase im Zyklus, Position); in Phase 0 wird bis S bestellt, wenn z < S."""
    n = model.n
    zmin = model.spec.zmin
    P_ = np.zeros((n * T, n * T))
    c = np.zeros(n * T)
    for ph in range(T):
        for i, z in enumerate(model.z):
            if ph == 0 and z < S:
                y, cost = S, model.G[S - zmin] + K
            else:
                y, cost = z, model.G[i]
            c[ph * n + i] = cost
            nxt = (ph + 1) % T
            P_[ph * n + i, nxt * n:(nxt + 1) * n] = model.B[y - zmin]
    A = np.eye(n * T) - P_.T
    A[-1, :] = 1.0
    b = np.zeros(n * T)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return float(pi @ c)


@pytest.mark.parametrize("L,K,T,S", [(1, 5.0, 2, 12), (1, 5.0, 3, 14), (3, 20.0, 2, 24)])
def test_cycle_formula_equals_the_phase_chain(L, K, T, S):
    model = ev.model_for(L, 9, 40)
    table = E.cycle_table(model)
    assert table.cost(K, T, S) == pytest.approx(phase_chain_cost(model, K, T, S), abs=1e-9)


@pytest.mark.parametrize("L", [0, 1, 2, 3])
def test_cycle_with_T_one_is_base_stock(L):
    model = ev.model_for(L, 9, 40)
    table = E.cycle_table(model)
    base = ev.base_table(L, 9, 40)
    for K in (0.0, 2.0, 10.0, 30.0):
        g_b, S_b, _ = base.best(K)
        assert table.cost(K, 1, S_b) == pytest.approx(g_b, abs=1e-9)
        assert min(table.cost(K, 1, int(S)) for S in table.S) == pytest.approx(g_b, abs=1e-9)
