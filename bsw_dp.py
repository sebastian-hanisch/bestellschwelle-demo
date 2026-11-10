"""Die exakte Rückwärtsrechnung (dynamische Programmierung) auf der Bestandsposition.

Modell: ein Artikel, Rückstand (nicht gedeckte Nachfrage wird nachgeliefert), Nachfrage je Periode i. i. d. (Poisson, Mittel 3), Lagerkosten h
je Stück und Periode, Fehlmengenkosten p je fehlendem Stück und Periode, Bestellkosten je Stück 0, Fixkosten K je Bestellung, Lieferzeit L.

Zustand ist die Bestandsposition z (Bestand + Bestelltes - Rückstand) vor der Bestellung, Aktion die Bestellmenge q (0 bis Q). Das ist eine
Rechenidentität, kein Satz aus der Literatur: Die Kosten einer Bestellung bis y fallen L + 1 Perioden später an, im Mittel G(y) mit der Summe der Nachfrage
über L + 1 Perioden. Die Rechnung mit dem vollen Pipeline-Zustand (tests/test_oracle_voll.py) liefert dieselben Kosten und dieselbe Aktion je Position.

Nach der Bestellung steht die Position bei y = z + q; eine Periode später bei y - D. Mit W(y) = G(y) + E[v(y - D)] gilt

    v(z) = min( W(z),  K + min_{z < y <= z + Q} W(y) )      (Bestellen oder nicht)

Unterhalb der unteren Zustandsgrenze liegende Positionen werden auf die Grenze gelegt, bestellt wird höchstens bis zur oberen Grenze.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bsw_cost import period_cost_G
from bsw_demand import lead_demand, poisson_pmf

TIE = 1e-12  # bestellt wird nur, wenn Bestellen um mehr als diesen Betrag billiger ist (Gleichstand: nicht bestellen)


@dataclass(frozen=True)
class ModelSpec:
    """Modellparameter; alle als Schlüsselwörter (nie positional). `pmf` ersetzt die Poisson-Nachfrage (Handbeispiel)."""

    L: int = 1
    h: float = 1.0
    p: float = 9.0
    Q: int = 40
    zmin: int = -40
    ymax: int = 60
    mean: float = 3.0
    dmax: int = 30
    pmf: tuple | None = None


class Model:
    """Zustandsraum z = zmin ... ymax mit G (Kosten je Position nach der Bestellung) und B (Übergang y -> y - D)."""

    def __init__(self, spec: ModelSpec):
        if spec.L < 0 or spec.Q < 1 or spec.ymax <= spec.zmin:
            raise ValueError("ungültiges Modell")
        if not spec.zmin <= 0 <= spec.ymax:
            raise ValueError("die Position 0 muss im Raster liegen (Bezugszustand der relativen Wertiteration)")
        self.spec = spec
        self.z = np.arange(spec.zmin, spec.ymax + 1)
        self.n = len(self.z)
        pmf = np.array(spec.pmf, dtype=float) if spec.pmf is not None else poisson_pmf(spec.mean, spec.dmax)
        if abs(pmf.sum() - 1.0) > 1e-9:
            raise ValueError("pmf muss sich zu 1 summieren")
        self.pmf = pmf
        self.dn = lead_demand(pmf, spec.L + 1)
        self.G = period_cost_G(self.z, self.dn, spec.h, spec.p)
        # B[i, j] = P(y_i - D = z_j); Masse unterhalb der unteren Grenze liegt auf der Grenze
        B = np.zeros((self.n, self.n))
        for d, pr in enumerate(pmf):
            if pr == 0.0:
                continue
            cols = np.maximum(self.z - d, spec.zmin) - spec.zmin
            np.add.at(B, (np.arange(self.n), cols), pr)
        self.B = B
        self.ref = -spec.zmin  # Index der Position 0

    @property
    def mu(self) -> float:
        return float((np.arange(len(self.pmf)) * self.pmf).sum())


@lru_cache(maxsize=64)
def build_model(spec: ModelSpec) -> Model:
    return Model(spec)


def backup(W: np.ndarray, K: float, Q: int) -> tuple[np.ndarray, np.ndarray]:
    """Eine Bestell-Entscheidung je Position. v[i] = min(W[i], K + min_{1 <= q <= Q} W[i + q]); q[i] = 0, wenn nicht bestellt wird.

    Bestellt wird nur, wenn es um mehr als TIE billiger ist; bei gleich guten Zielen gilt die kleinste Menge. Über das obere Ende hinaus wird nicht bestellt."""
    W = np.asarray(W, dtype=float)
    n = len(W)
    padded = np.concatenate([W[1:], np.full(Q, np.inf)])
    windows = sliding_window_view(padded, Q)[:n]
    j = windows.argmin(axis=1)
    c = K + windows[np.arange(n), j]
    order = c < W - TIE
    v = np.where(order, c, W)
    q = np.where(order, j + 1, 0)
    return v, q.astype(int)


def finite_step(model: Model, K: float, v_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ein Rückwärtsschritt: aus den Restkosten v_prev (n - 1 Perioden übrig) die Restkosten v (n Perioden), die Bestellmenge q und W."""
    W = model.G + model.B @ v_prev
    v, q = backup(W, K, model.spec.Q)
    return v, q, W


def finite_horizon(model: Model, K: float, n_periods: int):
    """Endlicher Horizont: n_periods Rückwärtsschritte ab Endkosten 0. Rückgabe v_n, q_n, W_n (die Kosten fallen zum Bestellzeitpunkt an)."""
    if n_periods < 1:
        raise ValueError("n_periods >= 1")
    v = np.zeros(model.n)
    q = np.zeros(model.n, dtype=int)
    W = model.G.copy()
    for _ in range(n_periods):
        v, q, W = finite_step(model, K, v)
    return v, q, W


def rvi(model: Model, K: float, tol: float = 1e-11, maxit: int = 50000):
    """Relative Wertiteration auf die Durchschnittskosten je Periode g. Rückgabe (g, q, h, W, Iterationen); h sind die relativen Restkosten (h(0) = 0)."""
    h = np.zeros(model.n)
    g = 0.0
    it = 0
    for it in range(1, maxit + 1):
        W = model.G + model.B @ h
        v, _ = backup(W, K, model.spec.Q)
        g_new = float(v[model.ref])
        h_new = v - g_new
        diff = float(np.max(np.abs(h_new - h)))
        h, g = h_new, g_new
        if diff < tol:
            break
    W = model.G + model.B @ h
    v, q = backup(W, K, model.spec.Q)
    return g, q, h, W, it
