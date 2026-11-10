"""Konvexität und K-Konvexität der Restkosten.

Ohne Fixkosten (K = 0) sind die Restkosten v(z) konvex in der Position z; mit Fixkosten nicht mehr. Statt konvex sind sie K-konvex (Scarf): für alle
Positionen i < k < j gilt v(k) <= (1 - w) v(i) + w v(j) + w K mit w = (k - i) / (j - i) (Steigungsform, eigene Formulierung nach der Definition bei
Scarf/Feinberg). Eine konvexe Funktion hat die zweiten Differenzen >= 0 überall; eine K-konvexe hat sie an wenigen Stellen < 0.
"""

from __future__ import annotations

import numpy as np

TOL = 1e-8  # zweite Differenzen über -TOL gelten als nicht negativ (Rundung der Wertiteration)


def second_differences(f) -> np.ndarray:
    """Zweite Differenzen f[k+1] - 2 f[k] + f[k-1] für k = 1 ... n - 2 (Länge n - 2)."""
    f = np.asarray(f, dtype=float)
    return f[2:] - 2.0 * f[1:-1] + f[:-2]


def nonconvex_points(z, f, tol: float = TOL) -> list[int]:
    """Positionen z, an denen die zweite Differenz kleiner als -tol ist (dort ist f nicht konvex)."""
    d2 = second_differences(f)
    z = np.asarray(z)
    return [int(z[i + 1]) for i in range(len(d2)) if d2[i] < -tol]


def lower_hull(f) -> np.ndarray:
    """Untere konvexe Hülle von f auf den Stellen 0 ... n - 1 (der größte konvexe Unterschätzer), Werte an allen Stellen."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    stack: list[int] = []
    for i in range(n):
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            # b liegt über oder auf der Sehne a-i: dann gehört b nicht zur unteren Hülle
            if (f[b] - f[a]) * (i - a) >= (f[i] - f[a]) * (b - a):
                stack.pop()
            else:
                break
        stack.append(i)
    return np.interp(np.arange(n), stack, f[stack])


def hull_defect(z, f) -> tuple[float, int | None]:
    """Größte Abweichung von f über der unteren konvexen Hülle und die Position (None, wenn f konvex ist: Abweichung 0)."""
    f = np.asarray(f, dtype=float)
    dev = f - lower_hull(f)
    i = int(np.argmax(dev))
    if dev[i] <= 1e-12:
        return 0.0, None
    return float(dev[i]), int(np.asarray(z)[i])


def k_convex_violation(f, K: float) -> float:
    """Größte Verletzung der K-Konvexität (Steigungsform) über alle Positionen i < k < j; <= 0 heißt K-konvex. Bei weniger als 3 Stellen: 0."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    if n < 3:
        return 0.0
    i = np.arange(n)[:, None, None]
    k = np.arange(n)[None, :, None]
    j = np.arange(n)[None, None, :]
    valid = (i < k) & (k < j)
    w = (k - i) / np.maximum(j - i, 1)
    excess = f[k] - ((1.0 - w) * f[i] + w * f[j]) - w * K
    return float(np.max(np.where(valid, excess, -np.inf)))
