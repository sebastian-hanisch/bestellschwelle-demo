"""Die EOQ-Orientierung (deterministische Losgröße) und die Zyklus-Politik.

EOQ (Lehrbuchformel, hier nachgerechnet: C'(Q) = 0): bei der konstanten Nachfrage D je Periode kostet eine Losgröße Q je Periode
C(Q) = K D / Q + h Q / 2, minimal bei Q* = sqrt(2 K D / h). Ein Los, das um den Faktor f daneben liegt, kostet (f + 1/f) / 2 mal so viel wie das Optimum.
Das gilt für diese deterministische Kurve, nicht für das Modell mit zufälliger Nachfrage.

Zyklus-Politik (R,S): alle T Perioden wird bis S bestellt (nur wenn seit der letzten Bestellung Nachfrage aufgetreten ist), dazwischen nie.
Je Periode kostet sie g = [ K P(D_T > 0) + sum_{t=0}^{T-1} E G(S - D_t) ] / T mit D_t = Nachfrage über t Perioden. Ohne Bestellgrenze gerechnet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsw_cost import period_cost_G
from bsw_demand import lead_demand
from bsw_dp import Model


def eoq_quantity(K: float, D: float, h: float) -> float:
    """Q* = sqrt(2 K D / h) (Menge je Bestellung)."""
    return float(np.sqrt(2.0 * K * D / h))


def eoq_cost(Q: float, K: float, D: float, h: float) -> float:
    """Kosten je Periode der deterministischen Kurve: K D / Q + h Q / 2."""
    return K * D / Q + h * Q / 2.0


def eoq_cycle_periods(K: float, D: float, h: float) -> float:
    """Dauer eines EOQ-Zyklus in Perioden: Q* / D = sqrt(2 K / (h D))."""
    return float(np.sqrt(2.0 * K / (h * D)))


def flatness(f: float) -> float:
    """Kosten(f Q*) / Kosten(Q*) = (f + 1/f) / 2 (deterministische Kurve)."""
    return (f + 1.0 / f) / 2.0


def extra_cost_lot_factor(f: float) -> float:
    """Mehrkosten in Prozent, wenn die LOSGRÖSSE um den Faktor f vom EOQ abweicht."""
    return 100.0 * (flatness(f) - 1.0)


def extra_cost_parameter_factor(c: float) -> float:
    """Mehrkosten in Prozent, wenn der PARAMETER K/h um den Faktor c falsch ist (die Losgröße weicht dann um sqrt(c) ab)."""
    return 100.0 * (flatness(float(np.sqrt(c))) - 1.0)


@dataclass(frozen=True)
class CycleTable:
    """K-freie Teile der Zyklus-Politik für alle (T, S): g = ops[T-1, S-S0] + K * freqfac[T-1]."""

    S: np.ndarray
    ops: np.ndarray       # Form (T_max, len(S)): Kosten für Lager und Fehlmengen je Periode
    freqfac: np.ndarray   # (1 - P(D_1 = 0)^T) / T: Anteil der Perioden mit Bestellung

    def best(self, K: float) -> tuple[float, int, int]:
        """(g, T, S) der besten Zyklus-Politik bei Fixkosten K."""
        g = self.ops + K * self.freqfac[:, None]
        i, j = np.unravel_index(int(np.argmin(g)), g.shape)
        return float(g[i, j]), int(i + 1), int(self.S[j])

    def cost(self, K: float, T: int, S: int) -> float:
        j = int(np.where(self.S == S)[0][0])
        return float(self.ops[T - 1, j] + K * self.freqfac[T - 1])


def cycle_table(model: Model, T_max: int = 12, S_range=range(-15, 60)) -> CycleTable:
    """Rechnet die K-freien Teile aller Zyklus-Politiken (T = 1 ... T_max, S aus S_range); T = 1 ist die Base-Stock-Politik."""
    S = np.array(list(S_range))
    pmf = model.pmf
    # D_t = Nachfrage über t Perioden; zu lange Schwänze (Masse < 1e-18) abschneiden
    dts = []
    for t in range(T_max):
        d = lead_demand(pmf, t)
        keep = np.nonzero(d > 1e-18)[0]
        dts.append(d[: keep[-1] + 1])
    kmax = max(len(d) for d in dts)
    y = np.arange(int(S.min()) - kmax, int(S.max()) + 1)
    Gy = period_cost_G(y, model.dn, model.spec.h, model.spec.p)
    ops = np.zeros((T_max, len(S)))
    running = np.zeros(len(S))
    for t in range(T_max):
        d = dts[t]
        # E G(S - D_t) = sum_k d[k] G(S - k)
        vals = np.zeros(len(S))
        for k, pk in enumerate(d):
            vals += pk * Gy[S - k - y[0]]
        running = running + vals
        ops[t] = running / (t + 1)
    p0 = float(pmf[0])
    freqfac = np.array([(1.0 - p0 ** (t + 1)) / (t + 1) for t in range(T_max)])
    return CycleTable(S, ops, freqfac)
