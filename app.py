"""Bestellschwelle: aus Base-Stock wird (s,S) - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Artikel, Poisson-Nachfrage (Mittel 3), Rückstand, Fixkosten K je Bestellung, Lieferzeit 0 bis 3. Die exakte Rückwärtsrechnung (DP) auf der
Bestandsposition setzt zwei Schwellen (s, S); die beste Ein-Schwellen-Regel (Base-Stock) kostet mehr, sobald K > 0 ist.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import numpy as np
import streamlit as st

import bsw_constants as C
import bsw_convex as CV
import bsw_eoq as E
import bsw_evaluation as ev
import bsw_hand as H
import bsw_policy as P
import bsw_visualization as V
from bsw_layout import dunkles_design, kopf_css
from bsw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Bestellschwelle – Sebastian Hanisch", layout="wide")


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def tausend(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{n:,}".replace(",", ".")


def frac(x: Fraction) -> str:
    return H.frac_text(x)


# ---- zwischengespeicherte Rechnungen: hängen nur von (Lieferzeit, p/h, Bestellgrenze) ab, nicht von K ---------------------------------------


@st.cache_data(show_spinner=False)
def _sweep(L, ph, Q):
    return ev.sweep_K(L, ph, Q)


@st.cache_data(show_spinner=False)
def _kstar(L, ph, Q):
    return ev.k_star(L, ph, Q)


@st.cache_data(show_spinner=False)
def _sS_parts(L, ph, Q):
    return P.sS_parts(ev.model_for(L, ph, Q))


@st.cache_data(show_spinner=False)
def _lot_parts(L, ph, Q):
    return ev.lot_parts(ev.model_for(L, ph, Q))


@st.cache_data(show_spinner=False)
def _cycle_table(L, ph, Q):
    return E.cycle_table(ev.model_for(L, ph, Q))


@st.cache_data(show_spinner=False)
def _hand():
    pairs = H.all_pairs()
    best, base = H.best_pair(pairs), H.best_base_stock(pairs)
    return {
        "backward": H.backward(2),
        "pairs": pairs,
        "best": best,
        "base": base,
        "f_best": H.order_frequency(*best[0]),
        "f_base": H.order_frequency(*base[0]),
    }


# ---- Kopf ------------------------------------------------------------------------------------------------------------------------------------

st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Bestellschwelle: aus Base-Stock wird (s,S)")
st.markdown(
    """
**Base-Stock** kennt nur eine Schwelle: Fällt die **Bestandsposition** unter S, wird bis S bestellt, in fast jeder Periode. Kostet jede Bestellung **Fixkosten K**, ist das oberhalb einer Schwelle K-Stern (im Raster 0,014 bis 0,89) zu oft.
Die exakte Rückwärtsrechnung (**dynamische Programmierung**) setzt eine zweite Schwelle: Bestelle erst, wenn die Position auf **s** oder darunter gefallen ist, dann bis **S**.
Die Demo zeigt, ab welchem K die zweite Schwelle sich lohnt, wie die Politik aussieht, wie die Restkosten dabei nicht mehr konvex werden und was die EOQ-Formel (Losgröße) davon trifft.

Weiter unten: die **Kernfrage** mit Verdikt, die Experimente in sechs Tabs, **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Bestellschwelle", expanded=True):
    st.markdown(
        """
1. **Zustand:** die **Bestandsposition** z = Bestand + Bestelltes (noch unterwegs) − Rückstand. Sie fasst Lager und Pipeline zusammen; die Bestellung hängt im Modell der Demo nur an z (gegen die Rechnung mit vollem Pipeline-Zustand geprüft).
2. **Rückwärtsrechnung:** Für jede Position vergleicht die DP „nicht bestellen“ mit „bis y bestellen“ (Fixkosten K plus die erwarteten Folgekosten W(y)) und nimmt das Billigere.
3. **Ergebnis:** In allen 1.464 Zellen des Messrasters (Lieferzeit 0 bis 3, p/h in sechs Stufen, K = 0 bis 30) hat die DP-Politik genau die Form (s, S): bestelle genau dann, wenn z ≤ s ist, und dann bis S. Das ist ein **Messbefund im Modell der Demo**, kein Satz (siehe ⚠️).
4. **Preis der einen Schwelle:** Base-Stock (s = S − 1) bestellt in 95 % der Perioden (Nachfrage über 0 mit Wahrscheinlichkeit 1 − e⁻³). Das sind bei großem K überwiegend Fixkosten, die die zweite Schwelle spart.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kosten und Lieferzeit**")
    K = st.slider(
        "Fixkosten K je Bestellung", *bounds("k_slider"), step=C.K_STEP, key="k_slider",
        help="Kosten jeder Bestellung, unabhängig von der Menge (Lagerkosten h = 1 je Stück und Periode). EOQ-Takt eine Periode entspricht K = 1,5.",
    )
    L = st.slider("Lieferzeit (Perioden)", *bounds("l_slider"), key="l_slider", help="Perioden zwischen Bestellung und Eintreffen der Ware.")
    st.markdown("**Fehlmenge**")
    ph = st.select_slider(
        "Strafverhältnis p/h", options=C.PH_OPTIONS, key="ph_select",
        help="Kosten eines fehlenden Stücks (je Periode, Rückstand) geteilt durch die Lagerkosten eines Stücks.",
    )
    st.markdown("**Für Fachleute**")
    Q = st.slider(
        "Bestellgrenze Q (Stück je Bestellung)", *bounds("q_slider"), key="q_slider",
        help="Größte Bestellmenge je Periode. Bei 40 greift sie im Modell der Demo nie (Anteil der Zeit höchstens 9e-14 im ganzen Raster); bei 10 und hohem K greift sie.",
    )
    horizon = st.select_slider(
        "Horizont (Restperioden)", options=C.HORIZON_OPTIONS, key="hor_select", format_func=lambda n: "stationär" if n == 0 else str(n),
        help="stationär = unendlicher Horizont (Durchschnittskosten). Eine Zahl rechnet n Perioden rückwärts mit Endkosten 0. Wirkt auf die Treppe im Tab Politik und auf die Wertfunktion; alle anderen Zahlen rechnen stationär.",
    )

sync_query_params({"k_slider": K, "l_slider": int(L), "ph_select": int(ph), "q_slider": int(Q), "hor_select": int(horizon)})

settings = ev.Settings(K=float(K), L=int(L), ph=int(ph), Q=int(Q), horizon=int(horizon))
model = ev.model_for(settings.L, settings.ph, settings.Q)
a = ev.analyse(settings)
# Marken für s und S nur bei reiner (s, S)-Form (bei greifender Bestellgrenze gibt es kein einzelnes Ziel)
s_stat, S_stat = (a.structure.s, a.structure.S) if a.structure.pure else (None, None)

# ---- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis (stationär)")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Kosten je Periode, DP-Politik", de(a.g, 3), delta="exakt", delta_color="off")
m2.metric("Kosten je Periode, Base-Stock", de(a.g_base, 3), delta=f"{de(a.g_base - a.g, 3)} gegen DP", delta_color="inverse")
m3.metric("Mehrkosten der einen Schwelle", f"{de(a.gap, 2)} %", delta="in % der DP-Kosten", delta_color="off")
m4.metric("DP-Politik (s, S)", f"({a.structure.s}, {a.structure.S})" if a.structure.pure else "nicht rein",
          delta=f"Bestellanteil {de(100 * a.freq)} %", delta_color="off")
mass = a.masses
if mass["qcap"] > 1e-4:
    st.warning(
        f"Die Bestellgrenze Q = {settings.Q} greift: Die DP-Politik liegt {de(100 * mass['qcap'], 2)} % der Zeit an der Grenze. Die Randmasse ist nicht klein; "
        "alle Aussagen gelten für diese Grenze (bei Q = 40 ist der Anteil höchstens 9e-14)."
    )
else:
    st.caption(
        f"Rechengrenzen: Anteil der Zeit an der Bestellgrenze {mass['qcap']:.1e}, Masse an den Zustandsgrenzen {max(mass['mass_lo'], mass['mass_hi']):.1e} "
        f"(unter 1e-4); erreichte Positionen {mass['z_lo']} bis {mass['z_hi']}, größte bestellte Menge {mass['qmax']}."
    )

# ---- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ab wann lohnt eine zweite Schwelle?")
kind, text = ev.verdict_gap(a)
getattr(st, kind)(text)
st.caption("Verglichen wird mit der besten Base-Stock-Politik (jede Stufe S exakt bewertet), nicht mit einer schlechten Stufe. Der große Prozentsatz bei hohem K ist zum größten Teil Arithmetik: Fixkosten je Bestellung mal Bestellhäufigkeit.")

st.markdown("---")

tab_hand, tab_pol, tab_val, tab_k, tab_enum, tab_eoq = st.tabs(["Von Hand", "Politik", "Wertfunktion", "Über K", "Alle (s,S)", "EOQ"])

# ================================================================================================================ Tab: Von Hand
with tab_hand:
    st.subheader("🔬 Experiment 1: Von Hand zwei Perioden rückwärts")
    st.markdown(
        "Eigene Mini-Instanz, unabhängig von den Reglern: Nachfrage 0, 1, 2 mit Wahrscheinlichkeit ¼, ½, ¼; Lagerkosten h = 1, Fehlmengenkosten p = 4, Fixkosten K = 2, Lieferzeit 0, Rückstand. "
        "Alle Zahlen sind exakte Brüche."
    )
    hd = _hand()
    (C1, tgt1, W1), (C2, tgt2, W2) = hd["backward"]
    st.markdown("**Schritt 1: Kosten einer Periode** bei Bestand y nach der Bestellung, L(y) = E[h·(y−D)⁺ + p·(D−y)⁺].")
    st.dataframe([{"y": y, "L(y)": frac(H.period_cost(y)), "Dezimal": de(float(H.period_cost(y)), 3)} for y in range(-1, 5)], hide_index=True)
    st.markdown(
        "**Schritt 2: eine Periode übrig.** Bestellen lohnt, wenn K + min L = 2 + 1 = 3 unter L(x) liegt. **Schritt 3: zwei Perioden übrig.** "
        "W₂(y) = L(y) + E[C₁(y − D)], bestelle bis zum kleinsten W₂ (hier y = 2), wenn K + W₂(y) unter W₂(x) liegt."
    )
    rows = []
    for x in range(-1, 5):
        rows.append({
            "Position x": x, "C₁(x)": frac(C1[x]), "W₂(x)": frac(W2[x]), "C₂(x)": frac(C2[x]), "C₂ dezimal": de(float(C2[x]), 3),
            "Entscheidung bei 2 Perioden": "bestelle bis " + str(tgt2[x]) if tgt2[x] != x else "nicht bestellen",
        })
    st.dataframe(rows, hide_index=True)
    xs = list(range(-1, 5))
    st.plotly_chart(V.build_hand_curve(xs, [float(C1[x]) for x in xs], [float(C2[x]) for x in xs]), width="stretch", key="hand_curve")
    d2 = H.second_difference(C2, 0)
    st.markdown(
        f"**Nicht konvex:** zweite Differenz von C₂ bei x = 0: C₂(1) − 2·C₂(0) + C₂(−1) = {frac(C2[1])} − 2·{frac(C2[0])} + {frac(C2[-1])} = **{frac(d2)}** < 0. "
        "Links der Schwelle ist C₂ flach (bestellt wird immer bis 2, die Restkosten sind dort gleich), dann fällt sie: konvex wäre das nicht, **K-konvex** schon."
    )
    st.markdown("**Schritt 4: unendlicher Horizont.** Alle Paare (s, S) mit −1 ≤ s < S ≤ 3, exakt bewertet über die stationäre Verteilung der Position:")
    pairs = hd["pairs"]
    st.dataframe(
        [{"(s, S)": f"({s}, {S})", "g (Bruch)": frac(g), "g dezimal": de(float(g), 4), "Base-Stock (s = S − 1)": "ja" if s == S - 1 else ""}
         for (s, S), g in sorted(pairs.items(), key=lambda kv: kv[1])],
        hide_index=True,
    )
    best, base = hd["best"], hd["base"]
    gap_hand = 100.0 * (float(base[1] / best[1]) - 1.0)
    kind, text = ev.verdict_hand((best[0], frac(best[1])), (base[0], frac(base[1])), gap_hand, float(hd["f_best"]), float(hd["f_base"]))
    getattr(st, kind)(text)

# ================================================================================================================ Tab: Politik
with tab_pol:
    st.subheader("Die Politik der DP")
    if settings.horizon == 0:
        st.markdown("**Treppe der Bestellmenge** über der Bestandsposition (stationär, nur erreichte Positionen).")
        st.plotly_chart(V.build_staircase(model.z, a.q, a.pi, s_stat, S_stat), width="stretch", key="stair_stat")
    else:
        fp = ev.finite_policy(settings)
        st.markdown(f"**Treppe bei {settings.horizon} Restperioden** (Endkosten 0, die Kosten fallen zum Bestellzeitpunkt an); Fenster z ≥ {C.WINDOW_LO}.")
        st.plotly_chart(V.build_staircase(model.z, fp.q, None, fp.structure.s, fp.structure.S, finite=True), width="stretch", key="stair_fin")
        if fp.structure.s is None:
            st.info(f"Bei {settings.horizon} Restperioden wird im Fenster z ≥ {C.WINDOW_LO} nie bestellt (s liegt darunter).")
        else:
            st.caption(f"(s, S) bei {settings.horizon} Restperioden: ({fp.structure.s}, {fp.structure.S}); reine (s, S)-Form im Fenster: {'ja' if fp.structure.pure else 'nein'}. "
                       f"Stationär: {'(' + str(s_stat) + ', ' + str(S_stat) + ')' if s_stat is not None else 'keine reine Form'}.")
        path = ev.finite_path(settings)
        st.plotly_chart(V.build_horizon_path(path, s_stat, S_stat), width="stretch", key="horizon_path")
        pure_all = sum(1 for p_ in path if not p_.pure)
        st.caption(f"Über n = 1 bis {C.HORIZON_MAX} Restperioden hat die Politik im Fenster in {C.HORIZON_MAX - pure_all} von {C.HORIZON_MAX} Fällen reine (s, S)-Form (diese Zelle).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kosten je Periode: Fixkosten gegen Lager und Fehlmengen** (stationär)")
        st.plotly_chart(V.build_decomposition(a), width="stretch", key="decomp")
    with c2:
        st.markdown("**Wo hält sich die Position auf?** (stationär, DP-Politik)")
        st.plotly_chart(V.build_distribution(model.z, a.pi, s_stat, S_stat), width="stretch", key="distribution")
    st.caption(f"Die DP bestellt in {de(100 * a.freq)} % der Perioden und spart gegenüber Base-Stock {de(a.fixed_saved, 2)} Fixkosten je Periode; "
               f"bei Lager und Fehlmengen zahlt sie {de(abs(a.rest_extra), 2)} {'mehr' if a.rest_extra >= 0 else 'weniger'}.")

# ================================================================================================================ Tab: Wertfunktion
with tab_val:
    st.subheader("🔬 Experiment 2: K-Konvexität der Restkosten")
    st.markdown(
        "**Ohne Fixkosten** sind die Restkosten v(z) konvex in der Position, und die einfache Schwelle (Base-Stock) ist optimal. **Mit Fixkosten** entsteht an der Schwelle s ein Knick nach unten: "
        "v ist nicht mehr konvex, nur noch **K-konvex** (die Abweichung von der unteren konvexen Hülle bleibt unter K). Daraus folgt die (s, S)-Form (Scarf; unter Voraussetzungen, siehe ⚠️)."
    )
    if settings.horizon == 0:
        v_full, W_full, s_use, S_use = a.h, a.W, s_stat, S_stat
        label = "stationär (relative Restkosten)"
    else:
        fp = ev.finite_policy(settings)
        v_full, W_full, s_use, S_use = fp.v, fp.W, fp.structure.s, fp.structure.S
        label = f"{settings.horizon} Restperioden"
    z, v, Wv = ev.window_arrays(model, v_full, W_full)
    bad = CV.nonconvex_points(z, v)
    badW = CV.nonconvex_points(z, Wv)
    hull = CV.lower_hull(v)
    defect, _where = CV.hull_defect(z, v)
    lo = max(C.HULL_LO, (s_use if s_use is not None else 0) - 12)
    hi = min(C.HULL_HI, (S_use if S_use is not None else 10) + 12)
    sel = (z >= lo) & (z <= hi)
    st.caption(f"Ansicht: {label}; Fenster z = {C.HULL_LO} bis {C.HULL_HI} für die Prüfung (Randeffekte der Zustandsgrenzen bleiben draußen), gezeichnet z = {lo} bis {hi}.")
    st.plotly_chart(V.build_value(z[sel], v[sel], Wv[sel], s_use, [p_ for p_ in bad if lo <= p_ <= hi], hull[sel], S_use), width="stretch", key="value_fn")
    kind, text = ev.verdict_convexity(settings.K, bad, s_use, defect, badW)
    getattr(st, kind)(text)
    st.markdown("**Abweichung von der unteren Hülle über K** (diese Lieferzeit und dieses p/h, stationär):")
    sw = _sweep(settings.L, settings.ph, settings.Q)
    st.plotly_chart(V.build_hull_over_K(sw), width="stretch", key="hull_K")
    st.markdown(f"**Dasselbe Prüfmuster für alle p/h** bei Lieferzeit {settings.L} und K = {de(settings.K)} (stationär):")
    rows = []
    for ph_i in C.PH_OPTIONS:
        cell = ev.convexity_cell(settings.L, ph_i, settings.K, settings.Q)
        rows.append({
            "p/h": ph_i, "(s, S)": f"({cell['s']}, {cell['S']})", "v nicht konvex bei z": ", ".join(map(str, cell["nonconvex_v"])) or "nirgends",
            "W nicht konvex (Stellen)": len(cell["nonconvex_W"]), "Abweichung / K": de(cell["hull_defect"] / settings.K, 2) if settings.K > 0 else "-",
            "K-Konvexität verletzt?": ("nein" if cell["k_violation"] <= 1e-9 else "ja") if cell["k_violation"] is not None else "-",
        })
    st.dataframe(rows, hide_index=True)
    st.caption("Messbefund: Die Kosten nach der Bestellung W(y) sind für p/h = 9, 19, 39 konvex; bei kleinem p/h (1, 2, 4) sind sie es nicht immer. Die Ursache ist nicht untersucht.")

# ================================================================================================================ Tab: Über K
with tab_k:
    st.subheader("🔬 Experiment 3: Wie groß ist die Lücke über K und Lieferzeit?")
    with st.spinner("Rechnet das K-Raster für alle vier Lieferzeiten (unter einer Sekunde, dann zwischengespeichert) ..."):
        sweeps = [_sweep(l_, settings.ph, settings.Q) for l_ in range(C.L_MIN, C.L_MAX + 1)]
        stars = [_kstar(l_, settings.ph, settings.Q) for l_ in range(C.L_MIN, C.L_MAX + 1)]
    st.markdown(f"Mehrkosten der besten Base-Stock-Politik gegenüber der DP in %, bei p/h = {settings.ph} (Bestellgrenze {settings.Q}).")
    st.plotly_chart(V.build_gap_heatmap(list(sweeps[0].K), list(range(C.L_MIN, C.L_MAX + 1)), [list(s_.gap) for s_ in sweeps]), width="stretch", key="gap_heat")
    sw = sweeps[settings.L]
    st.plotly_chart(V.build_gap_over_K(sw, model.mu, model.spec.h), width="stretch", key="gap_K")
    rows = []
    for l_, s_, ks in zip(range(C.L_MIN, C.L_MAX + 1), sweeps, stars):
        thr = {pct: ev.threshold_K(s_, pct) for pct in (1.0, 5.0, 10.0)}
        rows.append({
            "Lieferzeit": l_, "Lücke ≥ 1 % ab K": de(thr[1.0]) if thr[1.0] is not None else "-", "≥ 5 % ab K": de(thr[5.0]) if thr[5.0] is not None else "-",
            "≥ 10 % ab K": de(thr[10.0]) if thr[10.0] is not None else "-", "Schwelle K*": de(ks, 3) if ks is not None else "> 4",
            "EOQ-Takt bei K* (Perioden)": de(E.eoq_cycle_periods(ks, model.mu, model.spec.h), 2) if ks is not None else "-",
        })
    st.dataframe(rows, hide_index=True)
    thr_sel = {pct: ev.threshold_K(sw, pct) for pct in (1.0, 5.0, 10.0)}
    ks_sel = stars[settings.L]
    kind, text = ev.verdict_threshold(thr_sel, ks_sel, E.eoq_cycle_periods(ks_sel, model.mu, model.spec.h) if ks_sel is not None else None, settings.L, settings.ph)
    getattr(st, kind)(text)
    st.caption("K* ist das größte K, bei dem Base-Stock noch bis auf 1e-8 Kosteneinheiten optimal ist (Bisektion). Es liegt in den 24 Zellen unregelmäßig zwischen 0,014 und 0,89 und ist in Lieferzeit und p/h nicht durchgehend monoton (Beispiele: siehe README). "
               "Eine Deutung über die Ganzzahligkeit ist eine Vermutung, nicht gemessen.")

    st.subheader("🔬 Experiment 4: Passt S − s zur EOQ-Menge?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**s und S der DP** über K")
        st.plotly_chart(V.build_sS_over_K(sw), width="stretch", key="sS_K")
    with c2:
        st.markdown("**Losgröße S − s gegen EOQ-Menge** √(2 K μ / h)")
        st.plotly_chart(V.build_lot_vs_eoq(sw, model.mu, model.spec.h), width="stretch", key="lot_eoq")
    mask = sw.K >= C.EOQ_MIN_K
    ratios = list(sw.lot[mask] / np.sqrt(2 * sw.K[mask] * model.mu / model.spec.h))
    kind, text = ev.verdict_eoq(ratios)
    getattr(st, kind)(text)

# ================================================================================================================ Tab: Alle (s,S)
with tab_enum:
    st.subheader("Alle Paare (s, S) im Vergleich")
    with st.spinner("Bewertet 1.755 Paare (s, S) exakt, etwa eine halbe Sekunde, dann zwischengespeichert ..."):
        parts = _sS_parts(settings.L, settings.ph, settings.Q)
    en = ev.enumeration(model, settings.K, parts)
    st.markdown(
        f"Jedes Paar (s, S) mit s ≥ {C.ENUM_S_LO} und S ≤ {C.ENUM_S_HI} ({tausend(en['n_pairs'])} Paare) wird über die stationäre Verteilung exakt bewertet; die Karte zeigt die Mehrkosten gegenüber dem besten Paar, "
        "ausgeschnitten um dieses Paar (Farbskala bis 25 %)."
    )
    st.plotly_chart(V.build_enum_heatmap(en, settings.K), width="stretch", key="enum_heat")
    n = en["n_within"]
    diff = en["best_g"] - a.g
    if a.structure.pure:
        dp_txt = f"({a.structure.s}, {a.structure.S})"
        note = " (dasselbe Paar)" if (a.structure.s, a.structure.S) == en["best_pair"] else (" (ein gleich gutes Paar)" if abs(diff) < 1e-9 else "")
    else:
        dp_txt = "keine reine (s, S)-Form (die Bestellgrenze greift)"
        note = f"; das beste Paar ist um {de(diff, 4)} teurer als die DP-Politik"
    st.markdown(
        f"- Bestes Paar der Aufzählung: **{en['best_pair']}** mit {de(en['best_g'], 4)}; Politik der DP: {dp_txt} mit {de(a.g, 4)}{note}.\n"
        f"- Paare höchstens 0,1 % über dem Optimum: **{n[0.1]}**, höchstens 1 %: **{n[1.0]}**, höchstens 5 %: **{n[5.0]}** von {tausend(en['n_pairs'])}."
    )
    st.caption("Die Kostenfläche ist um das Optimum flach: Mehrere Nachbarpaare kosten kaum mehr. Die Aufzählung ist ein zweiter Rechenweg zum Optimum (jedes Paar einzeln bewertet); der Orakel-Test rechnet sie noch einmal mit eigenem Aufbau nach.")

# ================================================================================================================ Tab: EOQ
with tab_eoq:
    st.subheader("Die EOQ-Formel: Orientierung, keine Politik")
    st.markdown("**Deterministische Kurve** (konstante Nachfrage 3 je Periode, h = 1): C(Q) = K·D/Q + h·Q/2, Optimum Q* = √(2·K·D/h).")
    f = st.slider("Faktor f", 0.25, 4.0, 2.0, 0.25, key="f_eoq", help="Um diesen Faktor liegt die Losgröße (links) bzw. der Parameter K/h (rechts) daneben.")
    t1, t2 = st.columns(2)
    t1.metric(f"Losgröße um Faktor {de(f, 2)} daneben", f"+{de(E.extra_cost_lot_factor(f))} %", delta="deterministische Kurve", delta_color="off")
    t2.metric(f"Parameter K/h um Faktor {de(f, 2)} daneben (Losgröße × {de(float(np.sqrt(f)), 2)})", f"+{de(E.extra_cost_parameter_factor(f))} %", delta="deterministische Kurve", delta_color="off")
    st.caption("Bezugsgröße beachten: „Faktor 2“ bei der **Losgröße** kostet 25,0 % mehr, „Faktor 2“ beim **Parameter K/h** nur 6,1 % mehr, weil die Losgröße dann nur um √2 abweicht. Beides gilt für die deterministische Kurve.")
    if settings.K > 0:
        st.plotly_chart(V.build_eoq_curve(settings.K, model.mu, model.spec.h, f), width="stretch", key="eoq_curve")
    else:
        st.info("Bei K = 0 gibt es keine EOQ-Menge (Q* = 0); die Kurve wird erst ab K > 0 gezeichnet.")

    st.subheader("Im Modell der Demo (zufällige Nachfrage)")
    if settings.K > 0:
        with st.spinner("Bewertet alle Losgrößen S − s = 1 bis 30 mit bestem s (etwa eine halbe Sekunde, dann zwischengespeichert) ..."):
            lot_p = _lot_parts(settings.L, settings.ph, settings.Q)
        best_lot = ev.best_s_per_lot(lot_p, settings.K)
        g_best = min(v_[0] for v_ in best_lot.values())
        lots = sorted(best_lot)
        st.plotly_chart(V.build_lot_curve(lots, [100.0 * (best_lot[l_][0] / g_best - 1.0) for l_ in lots]), width="stretch", key="lot_curve")
        st.caption(f"Mehrkosten je Losgröße S − s (bestes s je Losgröße) gegenüber der besten Losgröße; EOQ-Menge √(2 K μ / h) = {de(float(np.sqrt(2 * settings.K * model.mu / model.spec.h)), 2)}. "
                   "Um das Optimum ist die Kurve flach, nach rechts steigt sie nur langsam.")
        lot = ev.lot_experiment(settings.L, settings.ph, settings.K, settings.Q)
        kind, text = ev.verdict_lot(lot)
        getattr(st, kind)(text)
    else:
        st.info("Bei K = 0 ist die Losgröße S − s = 1 (Base-Stock); der Vergleich beginnt bei K > 0.")

    st.subheader("🔬 Experiment 5: Feste Zykluslänge gegen die Schwelle s")
    st.markdown("Die **Zyklus-Politik** (R, S) bestellt alle T Perioden bis S (nur wenn seither Nachfrage aufgetreten ist), dazwischen nie. T = 1 ist Base-Stock. Beste Wahl von T und S je K, ohne Bestellgrenze gerechnet.")
    ct = _cycle_table(settings.L, settings.ph, settings.Q)
    gaps_c = [ev.gap_percent(ct.best(k_)[0], g_) for k_, g_ in zip(sw.K, sw.g)]
    st.plotly_chart(V.build_gap_over_K(sw, model.mu, model.spec.h, extra={"beste Zyklus-Politik (T, S frei)": gaps_c}), width="stretch", key="cycle_K")
    res = ev.cycle_vs_dp(settings, ct)
    kind, text = ev.verdict_cycle(res)
    getattr(st, kind)(text)

st.markdown("---")

# ---- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nachfrage i. i. d., Poisson, Mittel 3** | Schwankt die Nachfrage mit der Zeit oder gibt es Trend, ist die Position kein ausreichender Zustand mehr; nicht gemessen. | Nachfolger-Stück der Reihe (Zustand wächst) |
| **Rückstand statt verlorener Verkäufe** | Bei verlorenen Verkäufen ändert sich die Kostenfunktion; in dieser Demo nicht gerechnet. | Stück der Reihe zu verlorenen Verkäufen |
| **Bestellgrenze greift praktisch nie** | Bei Q = 10 und hohem K liegt die Politik an der Grenze; die reine (s, S)-Form gilt dann nicht mehr überall (der Regler zeigt es). | Regler „Bestellgrenze“ |
| **(s, S) als Messbefund** | Nach Feinberg ist (s, S) unter Rückstand, Lieferzeit null, konvexen Kosten und unbeschränkter Bestellmenge optimal; bei endlichem Horizont und begrenzter Kapazität im Allgemeinen nicht. Hier gemessen, nicht bewiesen. | Regler „Horizont“, „Bestellgrenze“ |
| **EOQ-Zahlen gelten deterministisch** | Im Modell der Demo weichen sie ab (Tab EOQ). | Tab EOQ |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben:** Nachfrage $D$ je Periode (Poisson, Mittel $\mu = 3$), Lagerkosten $h$, Fehlmengenkosten $p$ je Stück und Periode, Fixkosten $K$ je Bestellung, Lieferzeit $L$, Bestellgrenze $Q$. **Gesucht:** die Bestellmenge $q(z)$ je Bestandsposition $z$, die die Durchschnittskosten je Periode minimiert.

| Symbol | Bedeutung |
|---|---|
| $z$ | Bestandsposition vor der Bestellung (Bestand + Bestelltes − Rückstand) |
| $y = z + q$ | Position nach der Bestellung |
| $D_{L+1}$ | Nachfrage über $L + 1$ Perioden (Summe von $L + 1$ unabhängigen Poisson-Größen) |
| $G(y)$ | erwartete Kosten für Lager und Fehlmengen einer Bestellung bis $y$ |
| $v(z)$ | relative Restkosten vor der Bestellung, $g$ die Durchschnittskosten je Periode |
| $s$, $S$ | Bestellschwelle und Bestellziel der Politik |

**Kosten einer Bestellung bis $y$** (fallen $L + 1$ Perioden später an):

$$G(y) = \mathbb{E}\left[\, h\,(y - D_{L+1})^{+} + p\,(D_{L+1} - y)^{+} \right]$$

**Bellman-Gleichung mit Bestell-Minimum** (relative Wertiteration, $W(y) = G(y) + \mathbb{E}\, v(y - D)$):

$$g + v(z) = \min\Big( W(z),\; K + \min_{z < y \le z + Q} W(y) \Big)$$

**(s,S)-Form:** Es gibt $s < S$ mit

$$q(z) = \begin{cases} S - z & z \le s \\ 0 & z > s \end{cases}$$

**K-Konvexität** (Scarf): für alle Positionen $i < k < j$ mit $w = (k-i)/(j-i)$ gilt

$$v(k) \le (1 - w)\, v(i) + w\, v(j) + w\,K$$

**EOQ** (deterministische Nachfrage $D$ je Periode):

$$C(Q) = \frac{K D}{Q} + \frac{h Q}{2}, \qquad Q^{*} = \sqrt{\frac{2 K D}{h}}, \qquad \frac{C(f\,Q^{*})}{C(Q^{*})} = \frac{f + 1/f}{2}$$

**Zyklus-Politik** (alle $T$ Perioden bis $S$ bestellen, $D_t$ = Nachfrage über $t$ Perioden):

$$g_{T,S} = \frac{1}{T}\left( K\,\Pr(D_T > 0) + \sum_{t=0}^{T-1} \mathbb{E}\, G(S - D_t) \right)$$

**Literatur.** Feinberg, E. A.: Optimality Conditions for Inventory Control, arXiv:1606.00957 (2016; Volltext gelesen, Abschnitte 1 und 4): Satz von Scarf unter Rückstand, Lieferzeit null, i. i. d. Nachfrage, konvexen Kosten und unbeschränkter Bestellmenge. Scarf (1959/60) selbst ist nicht gelesen, nur über Feinberg zitiert. Die EOQ-Formel ist eine Lehrbuchformel, hier durch Ableiten und Aufzählung nachgerechnet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html)."
)
