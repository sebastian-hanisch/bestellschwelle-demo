"""Das Handbeispiel in exakten Brüchen (kein Gleitkomma): Nachfrage 0, 1, 2 mit 1/4, 1/2, 1/4; h = 1, p = 4, K = 2, Lieferzeit 0, Rückstand.

Alles hier ist Bruchrechnung mit `fractions.Fraction`; die Tabelle der App „Von Hand“ zeigt diese Werte, und der Orakel-Test rechnet sie unabhängig nach.
Zustände x = -3 ... 4 (Bestandsposition vor der Bestellung); bestellt wird höchstens bis 4, Positionen unter -3 zählen als -3.
"""

from __future__ import annotations

from fractions import Fraction as F

PMF = {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}
H, P, K = F(1), F(4), F(2)
XMIN, XMAX = -3, 4
STATES = list(range(XMIN, XMAX + 1))


def period_cost(y: int) -> F:
    """L(y) = E[h (y - D)^+ + p (D - y)^+]: erwartete Kosten einer Periode bei Bestand y nach der Bestellung."""
    return sum(pr * (H * max(y - d, 0) + P * max(d - y, 0)) for d, pr in PMF.items())


def backward_step(C_prev: dict[int, F]) -> tuple[dict[int, F], dict[int, int], dict[int, F]]:
    """Ein Rückwärtsschritt: aus C_{n-1} die Restkosten C_n, das Bestellziel (= x, wenn nicht bestellt wird) und W_n(y) = L(y) + E C_{n-1}(y - D)."""
    W = {y: period_cost(y) + sum(pr * C_prev[max(y - d, XMIN)] for d, pr in PMF.items()) for y in STATES}
    C, target = {}, {}
    for x in STATES:
        best, tgt = W[x], x
        for y in STATES:
            if y > x and K + W[y] < best:
                best, tgt = K + W[y], y
        C[x], target[x] = best, tgt
    return C, target, W


def backward(n_periods: int):
    """Restkosten bei n Restperioden: Liste [(C_n, Ziel_n, W_n)] für n = 1 ... n_periods (Endkosten 0)."""
    C = {x: F(0) for x in STATES}
    out = []
    for _ in range(n_periods):
        C, target, W = backward_step(C)
        out.append((C, target, W))
    return out


def second_difference(C: dict[int, F], x: int) -> F:
    return C[x + 1] - 2 * C[x] + C[x - 1]


def stationary_cost(s: int, S: int) -> tuple[F, dict[int, F]]:
    """Durchschnittskosten je Periode der (s,S)-Politik und ihre stationäre Verteilung (exakt: Gauß-Jordan mit Brüchen)."""
    states = list(range(XMIN, S + 1))
    n = len(states)
    ix = {z: i for i, z in enumerate(states)}
    T = [[F(0)] * n for _ in range(n)]
    cost = []
    for z in states:
        y = S if z <= s else z
        cost.append(period_cost(y) + (K if z <= s else 0))
        for d, pr in PMF.items():
            T[ix[z]][ix[max(y - d, XMIN)]] += pr
    A = [[(F(int(i == j)) - T[j][i]) for j in range(n)] + [F(0)] for i in range(n)]
    A[-1] = [F(1)] * n + [F(1)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    pi = [A[i][n] for i in range(n)]
    g = sum(a * b for a, b in zip(pi, cost))
    return g, {z: pi[ix[z]] for z in states if pi[ix[z]] != 0}


def order_frequency(s: int, S: int) -> F:
    """Anteil der Perioden mit Bestellung der (s,S)-Politik (stationäre Wahrscheinlichkeit der Positionen z <= s)."""
    _, pi = stationary_cost(s, S)
    return sum((p for z, p in pi.items() if z <= s), F(0))


def all_pairs(s_lo: int = -1, S_hi: int = 3) -> dict[tuple[int, int], F]:
    """Durchschnittskosten aller Paare (s, S) mit s_lo <= s < S <= S_hi."""
    return {(s, S): stationary_cost(s, S)[0] for S in range(s_lo + 1, S_hi + 1) for s in range(s_lo, S)}


def best_pair(pairs: dict[tuple[int, int], F]) -> tuple[tuple[int, int], F]:
    key = min(pairs, key=pairs.get)
    return key, pairs[key]


def best_base_stock(pairs: dict[tuple[int, int], F]) -> tuple[tuple[int, int], F]:
    base = {k: v for k, v in pairs.items() if k[0] == k[1] - 1}
    key = min(base, key=base.get)
    return key, base[key]


def frac_text(x: F) -> str:
    """Bruch als Text: ganze Zahl ohne Nenner, sonst z/n."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
