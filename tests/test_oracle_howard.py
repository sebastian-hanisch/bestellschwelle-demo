"""Orakel C: Politik-Iteration (Howard) und eine allgemeine relative Wertiteration über ALLE Aktionen gegen die DP der Demo.

Die DP der Demo nutzt die Struktur „bestellen oder nicht“ (gleitendes Minimum über W). Hier dagegen ein eigener Aufbau ohne diese Struktur: je Aktion q
eine volle Übergangsmatrix und ein Kostenvektor (Poisson aus scipy), dann (a) Politik-Iteration mit linearem Gleichungssystem, (b) allgemeine Wertiteration.
Beide dürfen weder die Form (s, S) noch die Faltung der Demo benutzen."""

import numpy as np
import pytest
from scipy.stats import poisson

import bsw_dp as D
import bsw_evaluation as ev

CELLS = [(0, 2.0), (0, 30.0), (1, 5.0), (2, 10.0), (3, 0.0), (3, 20.0)]
ZMIN, YMAX, QMAX = -40, 60, 40


def build(L, K, p=9.0, h=1.0, mean=3.0):
    z = np.arange(ZMIN, YMAX + 1)
    n = len(z)
    ks = np.arange(0, 31)
    one = poisson.pmf(ks, mean)
    one[-1] += 1.0 - one.sum()
    ln = np.arange(0, 400)
    lead = poisson.pmf(ln, mean * (L + 1))
    G = np.array([(lead * (h * np.maximum(y - ln, 0) + p * np.maximum(ln - y, 0))).sum() for y in z])
    cost = np.full((n, QMAX + 1), 1e9)
    trans = []
    for a in range(QMAX + 1):
        T = np.zeros((n, n))
        for i in range(n):
            y = z[i] + a
            if y > YMAX:
                T[i, i] = 1.0  # unzulässig, hoher Preis
                continue
            cost[i, a] = G[y - ZMIN] + (K if a > 0 else 0.0)
            for d, pr in zip(ks, one):
                T[i, max(y - d, ZMIN) - ZMIN] += pr
        trans.append(T)
    return n, cost, trans


def howard(n, cost, trans, ref, start):
    pol = start.copy()
    for _ in range(100):
        P = np.array([trans[pol[i]][i] for i in range(n)])
        c = cost[np.arange(n), pol]
        M = np.eye(n) - P
        M[:, ref] = 1.0  # unbekannt: relative Werte h mit h(ref) = 0 und g in der Referenzspalte
        x = np.linalg.solve(M, c)
        g = x[ref]
        h = x.copy()
        h[ref] = 0.0
        Qm = np.array([cost[:, a] + trans[a] @ h for a in range(len(trans))]).T
        best = Qm.min(axis=1)
        cur = Qm[np.arange(n), pol]
        new = np.where(cur <= best + 1e-10, pol, Qm.argmin(axis=1))
        if np.array_equal(new, pol):
            return g, pol
        pol = new
    raise AssertionError("keine Konvergenz")


def general_rvi(n, cost, trans, ref, tol=1e-11):
    h = np.zeros(n)
    for _ in range(20000):
        Qm = np.array([cost[:, a] + trans[a] @ h for a in range(len(trans))]).T
        v = Qm.min(axis=1)
        g = v[ref]
        hn = v - g
        if np.max(np.abs(hn - h)) < tol:
            return g
        h = hn
    raise AssertionError("keine Konvergenz")


@pytest.mark.parametrize("L,K", CELLS)
def test_howard_and_general_rvi_equal_the_dp(L, K):
    model = ev.model_for(L, 9, 40)
    g_dp = D.rvi(model, K)[0]
    n, cost, trans = build(L, K)
    ref = -ZMIN
    start = np.minimum(np.maximum(30 - np.arange(ZMIN, YMAX + 1), 0), QMAX)  # Start: Base-Stock S = 30 (schlecht)
    g_h, pol = howard(n, cost, trans, ref, start)
    assert g_h == pytest.approx(g_dp, abs=1e-8)
    assert general_rvi(n, cost, trans, ref) == pytest.approx(g_dp, abs=1e-8)
    # die Politik von Howard bestellt auf den Positionen der DP genauso
    q_dp = D.rvi(model, K)[1]
    z = np.arange(ZMIN, YMAX + 1)
    mid = (z >= -15) & (z <= 30)
    assert np.array_equal(pol[mid], q_dp[mid])
