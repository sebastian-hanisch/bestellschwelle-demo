"""Orakel D: Simulation des vollen Systems mit der (s, S)-Politik der DP gegen die exakten Kosten.

Die Simulation kennt weder die Bestandsposition als Zustand noch die stationäre Verteilung: Sie führt Bestand und Pipeline einzeln, würfelt die Nachfrage
und mittelt die Kosten über die Läufe. Vier Zufalls-Seeds je Zelle; verglichen wird in Standardfehlern (Mittel über die Läufe eines Seeds). Die Zufallsströme von
numpy sind zwischen Versionen nicht stabil, deshalb gilt eine statistische Schranke statt Gleichheit."""

import numpy as np
import pytest

import bsw_dp as D
import bsw_evaluation as ev
import bsw_policy as P

SEEDS = (0, 1, 2, 3)
RUNS, PERIODS, BURN = 1500, 2500, 300


def simulate(L, K, s, S, seed, mean=3.0, h=1.0, p=9.0, Q=40):
    rng = np.random.default_rng(seed)
    x = np.full(RUNS, float(S))
    pipe = np.zeros((RUNS, max(L - 1, 0)))
    total = np.zeros(RUNS)
    for t in range(PERIODS + BURN):
        ip = x + pipe.sum(axis=1)
        a = np.where(ip <= s, np.minimum(S - ip, Q), 0.0)
        d = rng.poisson(mean, RUNS)
        if L == 0:
            y = x + a
            c = h * np.maximum(y - d, 0) + p * np.maximum(d - y, 0) + K * (a > 0)
            x = y - d
        elif L == 1:
            c = h * np.maximum(x - d, 0) + p * np.maximum(d - x, 0) + K * (a > 0)
            x = x - d + a
        else:
            c = h * np.maximum(x - d, 0) + p * np.maximum(d - x, 0) + K * (a > 0)
            x = x - d + pipe[:, 0]
            pipe = np.concatenate([pipe[:, 1:], a[:, None]], axis=1)
        if t >= BURN:
            total += c
    return total / PERIODS


@pytest.mark.parametrize("L,K", [(0, 10.0), (1, 5.0), (2, 10.0), (3, 30.0)])
def test_simulation_agrees_with_the_exact_costs(L, K):
    model = ev.model_for(L, 9, 40)
    g, q, *_ = D.rvi(model, K)
    _, pi = P.evaluate(model, K, q)
    st = P.structure(model, q, P.reached(pi))
    g_exact, _ = P.evaluate(model, K, P.pol_sS(model, st.s, st.S))
    means = []
    for seed in SEEDS:
        per = simulate(L, K, st.s, st.S, seed)
        se = per.std(ddof=1) / np.sqrt(len(per))
        assert abs(per.mean() - g_exact) < 4.5 * se, (seed, per.mean(), g_exact, se)  # 4,5 Standardfehler: Fehlalarm unter 1e-4
        means.append(per.mean())
    assert abs(np.mean(means) - g_exact) < 0.02  # das Mittel der vier Seeds liegt nahe am exakten Wert (Streuung der Seed-Mittel etwa 0,002)
