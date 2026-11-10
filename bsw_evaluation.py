"""Kennzahlen, Sweeps über K und Verdikte (reine Funktionen: `verdict_*(...) -> (Art, Text)`).

Die Sweeps hängen nur von (Lieferzeit, p/h, Bestellgrenze) ab, nicht von K: sie rechnen das ganze K-Raster 0 bis 30 in Schritten von 0,5 und werden von
der App je Kombination einmal berechnet. Für jede feste Politik sind die Kosten linear in K (g = g_ops + K f), deshalb braucht die beste Base-Stock-Politik
je K keine neue Auswertung, nur die exakte DP (Wertiteration) wird je K gerechnet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import bsw_constants as C
import bsw_convex as CV
import bsw_dp as D
import bsw_eoq as E
import bsw_policy as P


@dataclass(frozen=True)
class Settings:
    """Einstellungen der Seitenleiste (immer mit Schlüsselwörtern)."""

    K: float = C.DEFAULT_K
    L: int = C.DEFAULT_L
    ph: int = C.DEFAULT_PH
    Q: int = C.DEFAULT_Q
    horizon: int = C.DEFAULT_HORIZON  # 0 = stationär, sonst Zahl der Restperioden


def spec_for(L: int, ph: int, Q: int) -> D.ModelSpec:
    """Modell der Demo: Poisson-Nachfrage Mittel 3, h = 1, p = p/h (je fehlendem Stück), Rückstand, Lieferzeit L, Bestellgrenze Q."""
    return D.ModelSpec(L=int(L), h=1.0, p=float(ph), Q=int(Q), zmin=C.ZMIN, ymax=C.YMAX, mean=C.MEAN, dmax=C.DMAX)


def model_for(L: int, ph: int, Q: int) -> D.Model:
    return D.build_model(spec_for(L, ph, Q))


def gap_percent(g_base: float, g_dp: float) -> float:
    """Mehrkosten der Base-Stock-Politik in Prozent der DP-Kosten."""
    return 0.0 if g_dp <= 0 else 100.0 * (g_base / g_dp - 1.0)


# ---------------------------------------------------------------------------------------------------------------- eine Zelle (stationär)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    g: float
    q: np.ndarray
    pi: np.ndarray
    h: np.ndarray            # relative Restkosten v(z)
    W: np.ndarray            # Kosten nach der Bestellung W(y)
    structure: P.Structure
    freq: float              # Anteil der Perioden mit Bestellung (DP)
    g_base: float
    S_base: int
    freq_base: float
    masses: dict = field(default_factory=dict)
    iterations: int = 0

    @property
    def gap(self) -> float:
        return gap_percent(self.g_base, self.g)

    @property
    def fixed_dp(self) -> float:
        return self.settings.K * self.freq

    @property
    def fixed_base(self) -> float:
        return self.settings.K * self.freq_base

    @property
    def rest_dp(self) -> float:
        return self.g - self.fixed_dp

    @property
    def rest_base(self) -> float:
        return self.g_base - self.fixed_base

    @property
    def fixed_saved(self) -> float:
        """Fixkosten je Periode, die die DP gegenüber Base-Stock spart."""
        return self.fixed_base - self.fixed_dp

    @property
    def rest_extra(self) -> float:
        """Mehrkosten der DP für Lager und Fehlmengen gegenüber Base-Stock (meist positiv: die DP hält die Bestellung zurück)."""
        return self.rest_dp - self.rest_base

    @property
    def fixed_points(self) -> float:
        return 0.0 if self.g <= 0 else 100.0 * self.fixed_saved / self.g

    @property
    def rest_points(self) -> float:
        return 0.0 if self.g <= 0 else -100.0 * self.rest_extra / self.g


def analyse(settings: Settings, table: P.BaseStockTable | None = None) -> Analysis:
    """Exakte DP, beste Base-Stock-Politik und Randmassen für eine Einstellung (stationär; `horizon` wird hier nicht verwendet)."""
    model = model_for(settings.L, settings.ph, settings.Q)
    g, q, h, W, it = D.rvi(model, settings.K)
    _, pi = P.evaluate(model, settings.K, q)
    st = P.structure(model, q, P.reached(pi))
    g_ops, f, _ = P.evaluate_parts(model, q)
    if table is None:
        table = base_table(settings.L, settings.ph, settings.Q)
    gb, Sb, fb = table.best(settings.K)
    return Analysis(settings, g, q, pi, h, W, st, f, gb, Sb, fb, P.boundary_masses(model, q, pi), it)


_TABLES: dict = {}


def base_table(L: int, ph: int, Q: int) -> P.BaseStockTable:
    """Base-Stock-Tabelle (alle Stufen S) je Modell, im Prozess zwischengespeichert."""
    key = (int(L), int(ph), int(Q))
    if key not in _TABLES:
        _TABLES[key] = P.base_stock_table(model_for(*key))
    return _TABLES[key]


# ---------------------------------------------------------------------------------------------------------------- endlicher Horizont


@dataclass(frozen=True)
class FinitePolicy:
    n: int
    q: np.ndarray
    v: np.ndarray
    W: np.ndarray
    structure: P.Structure


def finite_policy(settings: Settings, n: int | None = None) -> FinitePolicy:
    """Politik bei n Restperioden (Standard: settings.horizon), Form auf dem Fenster z >= C.WINDOW_LO."""
    n = settings.horizon if n is None else n
    model = model_for(settings.L, settings.ph, settings.Q)
    v, q, W = D.finite_horizon(model, settings.K, n)
    mask = model.z >= C.WINDOW_LO
    return FinitePolicy(n, q, v, W, P.structure(model, q, mask))


def finite_path(settings: Settings, n_max: int = C.HORIZON_MAX) -> list[P.Structure]:
    """(s_n, S_n, rein?) für n = 1 ... n_max Restperioden (ein Durchlauf)."""
    model = model_for(settings.L, settings.ph, settings.Q)
    mask = model.z >= C.WINDOW_LO
    v = np.zeros(model.n)
    out = []
    for _ in range(n_max):
        v, q, _W = D.finite_step(model, settings.K, v)
        out.append(P.structure(model, q, mask))
    return out


# ---------------------------------------------------------------------------------------------------------------- Sweep über K


@dataclass(frozen=True)
class KSweep:
    """Ergebnisse über das K-Raster für feste (L, p/h, Q); alle Felder haben die Länge len(K)."""

    L: int
    ph: int
    Q: int
    K: np.ndarray
    g: np.ndarray
    s: np.ndarray          # -9999, wenn nie bestellt wird (nur K = 0 mit Q-Rand: kommt im Raster nicht vor)
    S: np.ndarray
    pure: np.ndarray
    freq: np.ndarray
    g_base: np.ndarray
    S_base: np.ndarray
    freq_base: np.ndarray
    qcap: np.ndarray
    nonconv_v: np.ndarray   # Zahl der Stellen mit zweiter Differenz < -1e-8 im Fenster
    nonconv_at_s: np.ndarray  # ist z = s eine dieser Stellen?
    nonconv_W: np.ndarray
    hull_over_K: np.ndarray  # Abweichung der Restkosten von der unteren Hülle, geteilt durch K (K = 0: 0)

    @property
    def gap(self) -> np.ndarray:
        return 100.0 * (self.g_base / self.g - 1.0)

    @property
    def lot(self) -> np.ndarray:
        return self.S - self.s


def window_arrays(model: D.Model, *arrays):
    i0 = C.HULL_LO - model.spec.zmin
    i1 = C.HULL_HI - model.spec.zmin + 1
    return (model.z[i0:i1],) + tuple(a[i0:i1] for a in arrays)


def sweep_K(L: int, ph: int, Q: int = C.DEFAULT_Q, K_grid=C.K_GRID) -> KSweep:
    model = model_for(L, ph, Q)
    table = base_table(L, ph, Q)
    cols = {k: [] for k in ("g", "s", "S", "pure", "freq", "g_base", "S_base", "freq_base", "qcap", "nv", "nvs", "nW", "hull")}
    for K in K_grid:
        g, q, h, W, _ = D.rvi(model, K)
        g_ops, f, pi = P.evaluate_parts(model, q)
        st = P.structure(model, q, P.reached(pi))
        gb, Sb, fb = table.best(K)
        z, v, Wv = window_arrays(model, h, W)
        bad = CV.nonconvex_points(z, v)
        badW = CV.nonconvex_points(z, Wv)
        hd, _ = CV.hull_defect(z, v)
        cols["g"].append(g)
        cols["s"].append(-9999 if st.s is None else st.s)
        cols["S"].append(-9999 if st.S is None else st.S)
        cols["pure"].append(st.pure)
        cols["freq"].append(f)
        cols["g_base"].append(gb)
        cols["S_base"].append(Sb)
        cols["freq_base"].append(fb)
        cols["qcap"].append(float(pi[q == model.spec.Q].sum()))
        cols["nv"].append(len(bad))
        cols["nvs"].append(st.s in bad)
        cols["nW"].append(len(badW))
        cols["hull"].append(0.0 if K == 0 else hd / K)
    a = {k: np.array(v) for k, v in cols.items()}
    return KSweep(int(L), int(ph), int(Q), np.array(list(K_grid), dtype=float), a["g"], a["s"], a["S"], a["pure"], a["freq"], a["g_base"], a["S_base"],
                  a["freq_base"], a["qcap"], a["nv"], a["nvs"], a["nW"], a["hull"])


def threshold_K(sweep: KSweep, percent: float) -> float | None:
    """Kleinstes K auf dem Raster, bei dem die Lücke der Base-Stock-Politik mindestens `percent` % beträgt (None, wenn sie nie so groß wird)."""
    hit = np.nonzero(sweep.gap >= percent)[0]
    return float(sweep.K[hit[0]]) if len(hit) else None


def k_star(L: int, ph: int, Q: int = C.DEFAULT_Q, hi: float = 4.0, iterations: int = 30, tol: float = 1e-8) -> float | None:
    """Schwelle K*: größtes K (Bisektion auf [0, hi]), bei dem die beste Base-Stock-Politik die DP noch bis auf `tol` Kosteneinheiten trifft.

    Die Lücke g_Base - g_DP ist in K nicht fallend (Base-Stock-Kosten sind linear in K, die DP-Kosten konkav). None, wenn die Lücke bei K = hi
    noch höchstens tol beträgt (dann ist K* > hi)."""
    model = model_for(L, ph, Q)
    table = base_table(L, ph, Q)

    def excess(K: float) -> float:
        g = D.rvi(model, K)[0]
        return table.best(K)[0] - g

    if excess(hi) <= tol:
        return None
    lo, up = 0.0, float(hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + up)
        if excess(mid) > tol:
            up = mid
        else:
            lo = mid
    return 0.5 * (lo + up)


# ---------------------------------------------------------------------------------------------------------------- Konvexität, Los und Zyklus je Zelle


def convexity_cell(L: int, ph: int, K: float, Q: int = C.DEFAULT_Q) -> dict:
    """Konvexität der Restkosten v und der Kosten nach der Bestellung W im Fenster z = -30 ... 55 für eine Zelle."""
    model = model_for(L, ph, Q)
    g, q, h, W, _ = D.rvi(model, K)
    _, pi = P.evaluate(model, K, q)
    st = P.structure(model, q, P.reached(pi))
    z, v, Wv = window_arrays(model, h, W)
    bad = CV.nonconvex_points(z, v)
    hd, where = CV.hull_defect(z, v)
    return {
        "K": K, "L": L, "ph": ph, "s": st.s, "S": st.S, "nonconvex_v": bad, "nonconvex_W": CV.nonconvex_points(z, Wv),
        "hull_defect": hd, "hull_at": where, "k_violation": CV.k_convex_violation(v, K) if K > 0 else None, "g": g,
    }


def lot_costs(model: D.Model, K: float, lots=range(1, C.LOT_MAX + 1), s_lo: int = -14, s_hi: int = 39) -> dict[int, tuple[float, int]]:
    """Bestes s je Losgröße Q = S - s: {Q: (g, s)} (exakte Auswertung jeder (s, s + Q)-Politik)."""
    out = {}
    for lot in lots:
        best = (float("inf"), 0)
        for s in range(s_lo, s_hi):
            g, _ = P.evaluate(model, K, P.pol_sS(model, s, s + lot))
            if g < best[0]:
                best = (g, s)
        out[lot] = best
    return out


def lot_parts(model: D.Model, lots=range(1, C.LOT_MAX + 1), s_lo: int = -14, s_hi: int = 39):
    """K-freie Teile aller (s, s + Q) mit Q aus `lots`: Felder lot, s, g_ops, f."""
    ls, ss, gs, fs = [], [], [], []
    for lot in lots:
        for s in range(s_lo, s_hi):
            g_ops, f, _ = P.evaluate_parts(model, P.pol_sS(model, s, s + lot))
            ls.append(lot)
            ss.append(s)
            gs.append(g_ops)
            fs.append(f)
    return np.array(ls), np.array(ss), np.array(gs), np.array(fs)


def best_s_per_lot(parts, K: float) -> dict[int, tuple[float, int]]:
    """Aus `lot_parts` das beste s je Losgröße bei Fixkosten K: {Q: (g, s)}."""
    ls, ss, gs, fs = parts
    g = gs + K * fs
    out = {}
    for lot in np.unique(ls):
        m = ls == lot
        i = int(np.argmin(np.where(m, g, np.inf)))
        out[int(lot)] = (float(g[i]), int(ss[i]))
    return out


def lot_experiment(L: int, ph: int, K: float, Q: int = C.DEFAULT_Q) -> dict:
    """Textfalle im Modell der Demo: Die Losgröße S - s der DP um den Faktor 2 daneben (halbiert, verdoppelt; je bestes s) und der Parameter K um den
    Faktor 2 daneben (Politik für 2 K bzw. K / 2, bewertet mit dem wahren K). Prozent der DP-Kosten; dazu bezogen auf die Mehrkosten durch Fixkosten."""
    model = model_for(L, ph, Q)
    g, q, *_ = D.rvi(model, K)
    _, pi = P.evaluate(model, K, q)
    st = P.structure(model, q, P.reached(pi))
    lot = st.S - st.s
    g0 = D.rvi(model, 0.0)[0]
    out = {"g": g, "s": st.s, "S": st.S, "lot": lot, "g0": g0}
    for name, size in (("half", max(1, int(round(lot / 2)))), ("double", 2 * lot)):
        gg, s_best = lot_costs(model, K, lots=[size])[size]
        out[name] = {"lot": size, "pct": 100.0 * (gg / g - 1.0)}
    for name, fac in (("param_x2", 2.0), ("param_half", 0.5)):
        g2, q2, *_ = D.rvi(model, K * fac)
        _, pi2 = P.evaluate(model, K * fac, q2)
        st2 = P.structure(model, q2, P.reached(pi2))
        gtrue, _ = P.evaluate(model, K, P.pol_sS(model, st2.s, st2.S))
        out[name] = {"s": st2.s, "S": st2.S, "pct": 100.0 * (gtrue / g - 1.0), "pct_of_extra": 100.0 * (gtrue - g) / (g - g0) if g > g0 else 0.0}
    return out


@dataclass(frozen=True)
class CycleResult:
    K: float
    g_dp: float
    g_cycle: float
    T: int
    S: int
    gap_cycle: float
    gap_base: float
    takt: float
    capped: bool = False  # die Bestellgrenze greift in der DP (Anteil der Zeit an der Grenze über 1e-4); die Zyklus-Politik ist ohne Grenze gerechnet


def cycle_vs_dp(settings: Settings, table: E.CycleTable | None = None) -> CycleResult:
    model = model_for(settings.L, settings.ph, settings.Q)
    a = analyse(settings)
    if table is None:
        table = E.cycle_table(model)
    gc, T, S = table.best(settings.K)
    takt = E.eoq_cycle_periods(settings.K, model.mu, model.spec.h) if settings.K > 0 else 0.0
    return CycleResult(settings.K, a.g, gc, T, S, gap_percent(gc, a.g), a.gap, takt, capped=bool(a.masses["qcap"] > 1e-4))


# ---------------------------------------------------------------------------------------------------------------- Aufzählung aller (s, S)


def enumeration(model: D.Model, K: float, parts=None) -> dict:
    """Aufzählung aller (s, S) im Fenster s in [-12, S - 1], S in [-11, 40] (1.378 Paare): bestes Paar, Zahl der Paare nahe dem Optimum."""
    if parts is None:
        parts = P.sS_parts(model)
    s, S, g_ops, f = parts
    g = g_ops + K * f
    win = (s >= C.ENUM_S_LO) & (S <= C.ENUM_S_HI) & (S >= C.ENUM_S_LO + 1)
    gw = np.where(win, g, np.inf)
    i = int(np.argmin(gw))
    best = float(gw[i])
    return {
        "s": s, "S": S, "g": g, "window": win, "best_pair": (int(s[i]), int(S[i])), "best_g": best, "n_pairs": int(win.sum()),
        "n_within": {pct: int(((g <= best * (1 + pct / 100.0)) & win).sum()) for pct in (0.1, 1.0, 5.0)},
    }


# ---------------------------------------------------------------------------------------------------------------- Verdikte


def _de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def verdict_gap(a: Analysis) -> tuple[str, str]:
    """Kernfrage „Ab wann lohnt eine zweite Schwelle?“: Mehrkosten der Ein-Schwellen-Regel (beste Base-Stock-Politik) gegen die DP, in Punkten zerlegt."""
    K = a.settings.K
    if K <= 0:
        return "info", ("Ohne Fixkosten (K = 0) ist Base-Stock die optimale Politik: bestellt wird in jeder Periode bis S = "
                        f"{a.S_base}. Die zweite Schwelle hat nichts zu tun (Lücke {_de(a.gap, 2)} %).")
    s, S = a.structure.s, a.structure.S
    if a.gap < 0.005:
        return "info", (f"Bei K = {_de(K)} ist Base-Stock noch optimal (Lücke {_de(a.gap, 2)} %): Die DP bestellt in {_de(100 * a.freq)} % der Perioden, "
                        f"Base-Stock in {_de(100 * a.freq_base)} %. Die zweite Schwelle kommt erst bei höherem K ins Spiel.")
    kind = "success" if a.gap >= 1.0 else "info"
    form = f"(s, S) = ({s}, {S})" if a.structure.pure else "mit nicht reiner (s, S)-Form (die Bestellgrenze greift)"
    return kind, (
        f"Die Ein-Schwellen-Regel (beste Base-Stock-Politik, S = {a.S_base}) kostet **{_de(a.gap)} %** mehr als die DP-Politik {form}: "
        f"{_de(a.g_base, 3)} statt {_de(a.g, 3)} je Periode. Die DP bestellt in {_de(100 * a.freq)} % der Perioden, Base-Stock in {_de(100 * a.freq_base)} %: "
        f"sie spart {_de(a.fixed_saved, 2)} Fixkosten und zahlt {_de(abs(a.rest_extra), 2)} {'mehr' if a.rest_extra >= 0 else 'weniger'} für Lager und Fehlmengen "
        f"(in Punkten der DP-Kosten: {_de(a.fixed_points)} Fixkosten, {_de(a.rest_points)} Lager und Fehlmengen)."
    )


def verdict_threshold(thresholds: dict[float, float | None], k_star_value: float | None, takt_star: float | None, L: int, ph: int) -> tuple[str, str]:
    """Über K: ab welchem K die Lücke 1, 5, 10 % erreicht (Raster 0,5) und die genaue Schwelle K*, ab der Base-Stock nicht mehr optimal ist."""

    def at(pct):
        v = thresholds.get(pct)
        return "nie (auf dem Raster bis K = 30)" if v is None else f"K = {_de(v)}"

    star = "liegt über K = 4" if k_star_value is None else f"liegt bei K* = {_de(k_star_value, 3)} (EOQ-Takt {_de(takt_star, 2)} Perioden)"
    return "info", (f"Bei Lieferzeit {L} und p/h = {ph} liegt die Lücke ab {at(1.0)} bei mindestens 1 %, ab {at(5.0)} bei 5 % und ab {at(10.0)} bei 10 %. "
                    f"Die genaue Schwelle, ab der Base-Stock nicht mehr optimal ist, {star}.")


def verdict_eoq(ratios: list[float], lo: float = C.EOQ_BAND_LO, hi: float = C.EOQ_BAND_HI) -> tuple[str, str]:
    """EOQ-Menge gegen S - s: in wie vielen K-Werten (K >= 2, Raster 0,5) liegt das Verhältnis (S - s) / EOQ-Menge im Band lo bis hi."""
    if not ratios:
        return "info", "Für K < 2 ist der Vergleich mit der EOQ-Menge nicht aussagekräftig (S - s ist dort 1 bis 3 Stück, ganzzahlig gerundet)."
    inside = sum(1 for r in ratios if lo <= r <= hi)
    kind = "success" if inside == len(ratios) else "info"
    return kind, (f"S - s liegt in {inside} von {len(ratios)} K-Werten (K >= 2, Schritt 0,5) zwischen dem {_de(lo, 1)}- und dem {_de(hi, 2)}-fachen der EOQ-Menge "
                  f"√(2 K μ / h); kleinstes Verhältnis {_de(min(ratios), 2)}, größtes {_de(max(ratios), 2)}. Das ist ein Band, keine Gleichheit.")


def verdict_convexity(K: float, nonconvex: list[int], s: int | None, hull_defect: float, W_nonconvex: list[int]) -> tuple[str, str]:
    """Wertfunktion: ist v(z) konvex? Wo nicht, wie weit von der unteren Hülle?"""
    if not nonconvex:
        return "success", f"Bei K = {_de(K)} ist die Restkostenfunktion v(z) im Fenster z = -30 ... 55 konvex (keine zweite Differenz unter -1e-8)."
    k = f"{len(nonconvex)} Stelle" + ("" if len(nonconvex) == 1 else "n")
    at_s = "darunter z = s" if s in nonconvex else "z = s ist nicht darunter"
    wtxt = ("Die Kosten nach der Bestellung W(y) sind konvex." if not W_nonconvex
            else f"Auch W(y) ist nicht konvex ({len(W_nonconvex)} Stellen, bei kleinem p/h).")
    return "info", (f"Bei K = {_de(K)} ist v(z) an {k} nicht konvex ({at_s}); die größte Abweichung von der unteren konvexen Hülle beträgt "
                    f"{_de(hull_defect, 2)} ({_de(hull_defect / K, 2) if K > 0 else '0'} K). {wtxt}")


def verdict_cycle(r: CycleResult) -> tuple[str, str]:
    """Zyklus-Politik (alle T Perioden bis S bestellen) gegen die DP und gegen Base-Stock. Greift die Bestellgrenze, ist der Vergleich nicht fair: kein Verdikt."""
    if r.capped:
        return "info", ("Die Bestellgrenze greift: Die DP rechnet mit ihr, die Zyklus-Politik ohne. Der Vergleich ist deshalb nicht fair, "
                        "ein Verdikt über die Reihenfolge entfällt (Regler Q größer stellen).")
    if r.T == 1:
        return "info", (f"Die beste Zyklus-Politik hat T = 1 (jede Periode, S = {r.S}) und ist damit die Base-Stock-Politik: {_de(r.gap_cycle)} % über der DP.")
    return "info", (f"Die beste Zyklus-Politik (alle T = {r.T} Perioden bis S = {r.S}) kostet {_de(r.gap_cycle)} % mehr als die DP, Base-Stock {_de(r.gap_base)} %. "
                    f"Eine feste Zykluslänge ist besser als jede Periode, aber nicht so gut wie die Schwelle s.")


def verdict_lot(lot: dict) -> tuple[str, str]:
    """Textfalle im Modell der Demo: Losgröße S - s halbiert/verdoppelt und Parameter K halbiert/verdoppelt (Prozent der DP-Kosten)."""
    return "info", (f"Im Modell der Demo (Politik (s, S) = ({lot['s']}, {lot['S']}), Losgröße {lot['lot']}) kostet die halbe Losgröße ({lot['half']['lot']}) "
                    f"{_de(lot['half']['pct'])} % und die doppelte ({lot['double']['lot']}) {_de(lot['double']['pct'])} % mehr; die Politik für 2 K kostet "
                    f"{_de(lot['param_x2']['pct'])} %, die für K / 2 {_de(lot['param_half']['pct'])} % mehr (je bestes s, bewertet mit dem wahren K). "
                    "Das gilt für diese Zelle, nicht für die deterministische Kurve.")


def verdict_hand(best: tuple[tuple[int, int], str], base: tuple[tuple[int, int], str], gap: float, freq_best: float, freq_base: float) -> tuple[str, str]:
    """Handbeispiel: bestes Paar (s, S) und beste Base-Stock-Politik mit den exakten Kosten (als Bruchtext), der Lücke in Prozent und den Bestellanteilen (0 bis 1)."""
    (s, S), g = best
    (sb, Sb), gb = base
    return "success", (f"Bestes Paar (s, S) = ({s}, {S}) mit g = {g}; beste Base-Stock-Politik ({sb}, {Sb}) mit g = {gb}: **{_de(gap)} %** mehr. "
                       f"Base-Stock bestellt in {_de(100 * freq_base, 0)} % der Perioden, ({s}, {S}) nur in {_de(100 * freq_best, 0)} %: "
                       f"Die Politik wartet, bis die Position auf {s} oder darunter gefallen ist.")
