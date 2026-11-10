"""Orakel A: Rechnung auf der Bestandsposition gegen die Rechnung mit vollem Pipeline-Zustand.

Der volle Zustand ist (Bestand x, Bestellungen der letzten L - 1 Perioden); die Bestellung der Vorperiode kommt am Ende der Periode an. Aufbau hier
mit dünnen Matrizen und einer allgemeinen relativen Wertiteration über alle Aktionen, ohne die Position als Zustand zu benutzen. Verglichen werden die
Durchschnittskosten und die Aktion je erreichter Position (die Aktion hängt im vollen Modell nur an der Position, nicht an der Verteilung auf die Pipeline)."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import poisson

import bsw_dp as D

DMAX = 14
XMIN, XMAX = -25, 30
CELLS = [(0, 0.0, 10), (0, 10.0, 10), (1, 2.0, 10), (1, 10.0, 10), (2, 5.0, 10), (3, 5.0, 6)]  # (Lieferzeit, K, Bestellgrenze)


def full_model(L, K, Q, p=9.0, h=1.0, mean=3.0):
    ks = np.arange(DMAX + 1)
    pd = poisson.pmf(ks, mean)
    pd[-1] += 1.0 - pd.sum()
    npipe = max(L - 1, 0)
    dims = [XMAX - XMIN + 1] + [Q + 1] * npipe
    S = int(np.prod(dims))
    grid = np.indices(dims).reshape(len(dims), -1)
    x = grid[0] + XMIN
    pipe = grid[1:]
    ip = x + (pipe.sum(axis=0) if npipe else 0)

    def idx(xn, pn):
        i = np.clip(xn, XMIN, XMAX) - XMIN
        for k in range(npipe):
            i = i * (Q + 1) + pn[k]
        return i

    cost = np.zeros((S, Q + 1))
    mats = []
    for a in range(Q + 1):
        rows, cols, vals = [], [], []
        y = x + a if L == 0 else x
        ec = np.zeros(S)
        for d, pr in zip(ks, pd):
            ec += pr * (h * np.maximum(y - d, 0) + p * np.maximum(d - y, 0))
            if L == 0:
                xn, pn = y - d, []
            elif L == 1:
                xn, pn = x - d + a, []
            else:
                xn, pn = x - d + pipe[0], list(pipe[1:]) + [np.full(S, a)]
            rows.append(np.arange(S))
            cols.append(idx(xn, pn))
            vals.append(np.full(S, pr))
        mats.append(sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S)))
        cost[:, a] = ec + (K if a > 0 else 0.0)
    return S, ip, cost, mats


def stationary(S, mats, pol, start):
    rows = []
    for a in np.unique(pol):
        mask = (pol == a).astype(float)
        rows.append(sp.diags(mask) @ mats[a])
    P = sum(rows).tocsr()
    pi = np.zeros(S)
    pi[start] = 1.0
    PT = ((P + sp.identity(S)) * 0.5).T.tocsr()
    for _ in range(4000):
        pi = PT @ pi
    return pi / pi.sum()


@pytest.mark.parametrize("L,K,Q", CELLS)
def test_position_dp_equals_the_full_pipeline_dp(L, K, Q):
    S, ip, cost, mats = full_model(L, K, Q)
    ref = (-XMIN) * (Q + 1) ** max(L - 1, 0)  # Bezugszustand: x = 0, leere Pipeline
    h = np.zeros(S)
    for _ in range(5000):
        Qm = np.empty_like(cost)
        for a, M in enumerate(mats):
            Qm[:, a] = cost[:, a] + M @ h
        v = Qm.min(axis=1)
        g_full = v[ref]
        hn = v - g_full
        done = np.max(np.abs(hn - h)) < 1e-10
        h = hn
        if done:
            break
    Qm = np.empty_like(cost)
    for a, M in enumerate(mats):
        Qm[:, a] = cost[:, a] + M @ h
    pol_full = Qm.argmin(axis=1)

    spec = D.ModelSpec(L=L, h=1.0, p=9.0, Q=Q, zmin=-40, ymax=60, dmax=DMAX)
    model = D.build_model(spec)
    g_pos, q_pos, *_ = D.rvi(model, K)
    assert g_full == pytest.approx(g_pos, abs=1e-6)

    # Aktion je erreichter Position: im vollen Modell nur eine Aktion je Position?
    start = ref
    pi = stationary(S, mats, pol_full, start)
    reached = pi > 1e-9
    checked = 0
    for pos in np.unique(ip[reached]):
        acts = np.unique(pol_full[reached & (ip == pos)])
        if len(acts) == 1 and -26 < pos < 30:
            assert acts[0] == q_pos[pos - spec.zmin], (L, K, pos)
            checked += 1
    assert checked >= 10
