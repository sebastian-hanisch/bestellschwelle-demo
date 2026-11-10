"""Politiken auf der Bestandsposition, ihre exakte Bewertung und ihre Form.

Eine Politik ist hier ein Feld q (Bestellmenge je Position z). Ihre Durchschnittskosten je Periode folgen aus der stationären Verteilung der Position
(lineares Gleichungssystem, kein Simulationsrauschen). Für jede feste Politik gilt g = g_ops + K * f: g_ops sind die Kosten für Lager und Fehlmengen,
f ist der Anteil der Perioden mit Bestellung; die Kette selbst hängt nicht von K ab. Darauf beruht die schnelle Auswertung über viele K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsw_dp import Model


def pol_sS(model: Model, s: int, S: int) -> np.ndarray:
    """(s,S)-Politik: bestelle bis S, sobald die Position z <= s ist (höchstens Q Stück und höchstens bis zur oberen Zustandsgrenze), sonst nichts."""
    z = model.z
    q = np.where(z <= s, np.minimum(S - z, model.spec.Q), 0)
    q = np.where(z + q > model.spec.ymax, model.spec.ymax - z, q)
    return np.maximum(q, 0).astype(int)


def pol_base_stock(model: Model, S: int) -> np.ndarray:
    """Base-Stock-Politik: bestelle in jeder Periode bis S (der Sonderfall s = S - 1)."""
    return pol_sS(model, S - 1, S)


_PI_CACHE: dict = {}


def stationary_distribution(model: Model, q: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung der Position unter der Politik q (eine rückkehrende Klasse; die Wahrscheinlichkeit transienter Positionen ist 0).

    Sie hängt nur vom Übergang der Position ab (Nachfrage je Periode, Zustandsgrenzen), nicht von Lieferzeit und Kosten; deshalb wird sie je
    (Übergang, Politik) einmal berechnet und für alle Lieferzeiten und p/h wiederverwendet."""
    spec = model.spec
    key = (spec.mean, spec.dmax, spec.pmf, spec.zmin, spec.ymax, q.tobytes())
    hit = _PI_CACHE.get(key)
    if hit is not None:
        return hit
    y_idx = np.arange(model.n) + q
    P = model.B[y_idx]
    A = np.eye(model.n) - P.T
    A[-1, :] = 1.0
    b = np.zeros(model.n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    pi.setflags(write=False)
    if len(_PI_CACHE) > 20000:
        _PI_CACHE.clear()
    _PI_CACHE[key] = pi
    return pi


def evaluate_parts(model: Model, q: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(g_ops, f, pi): Kosten für Lager und Fehlmengen je Periode, Anteil der Perioden mit Bestellung, stationäre Verteilung."""
    pi = stationary_distribution(model, q)
    y_idx = np.arange(model.n) + q
    return float(pi @ model.G[y_idx]), float(pi[q > 0].sum()), pi


def evaluate(model: Model, K: float, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Durchschnittskosten je Periode g = g_ops + K f der Politik q (exakt) und die stationäre Verteilung."""
    g_ops, f, pi = evaluate_parts(model, q)
    return g_ops + K * f, pi


@dataclass(frozen=True)
class BaseStockTable:
    """Alle Base-Stock-Stufen S mit g_ops(S) und f(S); die Kosten bei K sind g_ops + K f (für jede feste Politik exakt)."""

    S: np.ndarray
    g_ops: np.ndarray
    freq: np.ndarray

    def best(self, K: float) -> tuple[float, int, float]:
        """(g, S, f) der besten Base-Stock-Politik bei Fixkosten K; bei Gleichstand die kleinste Stufe."""
        g = self.g_ops + K * self.freq
        i = int(np.argmin(g))
        return float(g[i]), int(self.S[i]), float(self.freq[i])


def base_stock_table(model: Model, S_range=None) -> BaseStockTable:
    """Bewertet alle Base-Stock-Stufen (Standard: von zmin/2 bis ymax - 6, das sind -20 bis 54 im Raster der Demo)."""
    if S_range is None:
        S_range = range(model.spec.zmin // 2, model.spec.ymax - 5)
    Ss, gs, fs = [], [], []
    for S in S_range:
        g_ops, f, _ = evaluate_parts(model, pol_base_stock(model, S))
        Ss.append(S)
        gs.append(g_ops)
        fs.append(f)
    return BaseStockTable(np.array(Ss), np.array(gs), np.array(fs))


@dataclass(frozen=True)
class Structure:
    """Form einer Politik auf den betrachteten Positionen.

    s: höchste Position mit Bestellung; S: das häufigste Ziel z + q; pure: genau EIN Ziel, und bestellt wird genau bei z <= s (reine (s,S)-Form);
    s und S sind None, wenn auf den betrachteten Positionen nie bestellt wird."""

    s: int | None
    S: int | None
    pure: bool


def structure(model: Model, q: np.ndarray, mask: np.ndarray) -> Structure:
    """Form der Politik q auf den Positionen, die `mask` markiert (stationär: Positionen mit Wahrscheinlichkeit > 1e-12; endlicher Horizont: ein Fenster)."""
    z = model.z
    order = q > 0
    if not order[mask].any():
        return Structure(None, None, True)
    s = int(z[mask & order].max())
    targets = (z + q)[mask & order]
    vals, cnt = np.unique(targets, return_counts=True)
    S = int(vals[cnt.argmax()])
    pure = bool(len(vals) == 1 and np.all(order[mask & (z <= s)]) and not order[mask & (z > s)].any())
    return Structure(s, S, pure)


def reached(pi: np.ndarray, thr: float = 1e-12) -> np.ndarray:
    """Positionen, die im stationären Betrieb vorkommen (Wahrscheinlichkeit > thr)."""
    return pi > thr


def boundary_masses(model: Model, q: np.ndarray, pi: np.ndarray) -> dict:
    """Randmassen der Rechnung: Anteil der Zeit an der Bestellgrenze Q, Masse an der unteren und der oberen Zustandsgrenze, größte bestellte Menge."""
    rec = reached(pi)
    return {
        "qcap": float(pi[q == model.spec.Q].sum()),
        "mass_lo": float(pi[0]),
        "mass_hi": float(pi[-1]),
        "qmax": int(q[rec].max()) if rec.any() else 0,
        "z_lo": int(model.z[rec].min()),
        "z_hi": int(model.z[rec].max()),
    }


def sS_parts(model: Model, s_lo: int = -14, s_hi: int = 39, S_lo: int = -13, S_hi: int = 45):
    """Alle Paare (s,S) mit s < S: g_ops und f je Paar (K-frei). Rückgabe: Felder s, S, g_ops, f (gleich lang)."""
    ss, SS, gs, fs = [], [], [], []
    for S in range(S_lo, S_hi + 1):
        for s in range(s_lo, min(S - 1, s_hi) + 1):
            g_ops, f, _ = evaluate_parts(model, pol_sS(model, s, S))
            ss.append(s)
            SS.append(S)
            gs.append(g_ops)
            fs.append(f)
    return np.array(ss), np.array(SS), np.array(gs), np.array(fs)
