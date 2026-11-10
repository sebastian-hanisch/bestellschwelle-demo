"""Kosten einer Periode (Lager und Fehlmengen) bei gegebenem Bestand nach der Bestellung."""

from __future__ import annotations

import numpy as np


def period_cost_G(y, pmf, h: float, p: float):
    """G(y) = E[h (y - D)^+ + p (D - y)^+] für die Nachfrage D mit Verteilung `pmf` (Index = Nachfragewert).

    y darf eine ganze Zahl oder ein Feld sein (auch negativ: Rückstand). h = Lagerkosten je Stück und Periode, p = Fehlmengenkosten je Stück."""
    pmf = np.asarray(pmf, dtype=float)
    k = np.arange(len(pmf))
    yy = np.asarray(y, dtype=float)
    scalar = yy.ndim == 0
    yy = np.atleast_1d(yy)[:, None]
    out = (pmf[None, :] * (h * np.maximum(yy - k[None, :], 0.0) + p * np.maximum(k[None, :] - yy, 0.0))).sum(axis=1)
    return float(out[0]) if scalar else out
