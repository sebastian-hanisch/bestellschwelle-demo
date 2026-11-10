"""Nachfrage: Poisson-Verteilung und die Summe über mehrere Perioden (Faltung).

Die Wiederbeschaffungszeit ist die Lieferzeit L plus die Periode der Bestellung: Eine Bestellung bis y kostet (Lager und Fehlmengen)
erst L + 1 Perioden später, und zwar im Mittel G(y) mit der Nachfrage über genau diese L + 1 Perioden (`lead_demand`).
"""

from __future__ import annotations

import numpy as np


def poisson_pmf(mean: float, dmax: int) -> np.ndarray:
    """Poisson(mean) auf 0 bis dmax; die Restwahrscheinlichkeit oberhalb von dmax liegt auf dmax (bei Mittel 3 und dmax 30 unter 1e-19).

    mean = 0 ergibt die Nachfrage 0 mit Sicherheit."""
    if mean < 0 or dmax < 0:
        raise ValueError("mean und dmax müssen nicht negativ sein")
    pmf = np.zeros(dmax + 1)
    pmf[0] = np.exp(-mean)
    for k in range(1, dmax + 1):
        pmf[k] = pmf[k - 1] * mean / k
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


def lead_demand(pmf, periods: int) -> np.ndarray:
    """Verteilung der Summe der Nachfrage über `periods` unabhängige Perioden (periods-fache Faltung); periods = 0 ist die Nachfrage 0."""
    if periods < 0:
        raise ValueError("periods darf nicht negativ sein")
    out = np.array([1.0])
    base = np.asarray(pmf, dtype=float)
    for _ in range(periods):
        out = np.convolve(out, base)
    return out


def critical_quantile(pmf, p: float, h: float) -> int:
    """Kleinstes y mit F(y) >= p / (p + h): die Base-Stock-Stufe bei K = 0 (Newsvendor-Quantil der Nachfrage `pmf`)."""
    cdf = np.cumsum(np.asarray(pmf, dtype=float))
    target = p / (p + h)
    idx = int(np.searchsorted(cdf, target - 1e-12, side="left"))
    return min(idx, len(cdf) - 1)
