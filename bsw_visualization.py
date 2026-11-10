"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import bsw_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_staircase(z, q, pi, s, S, finite=False, z_lo=None, z_hi=None):
    """Bestellmenge über der Bestandsposition: die Treppe der Politik; Marken bei s und S. Bei endlichem Horizont ohne Verteilung."""
    z = np.asarray(z)
    lo = z_lo if z_lo is not None else (int(z[pi > 1e-12].min()) - 2 if not finite else -10)
    hi = z_hi if z_hi is not None else (int(z[pi > 1e-12].max()) + 2 if not finite else 30)
    m = (z >= lo) & (z <= hi)
    fig = go.Figure(go.Bar(x=z[m], y=np.asarray(q)[m], marker_color=C.COLOR_DP, name="Bestellmenge", hovertemplate="Position %{x}: bestelle %{y}<extra></extra>"))
    if s is not None:
        fig.add_vline(x=s + 0.5, line_dash="dash", line_color=C.COLOR_BASE, annotation_text=f"s = {s}", annotation_position="top left")
        fig.add_vline(x=S, line_dash="dot", line_color=C.COLOR_EOQ, annotation_text=f"S = {S}", annotation_position="top right")
    fig.update_xaxes(title_text="Bestandsposition z vor der Bestellung (Bestand + Bestelltes - Rückstand)")
    fig.update_yaxes(title_text="Bestellmenge")
    return _base(fig)


def build_distribution(z, pi, s, S):
    """Stationäre Verteilung der Position unter der DP-Politik (nur erreichte Positionen)."""
    z = np.asarray(z)
    m = pi > 1e-12
    fig = go.Figure(go.Bar(x=z[m], y=pi[m], marker_color=C.COLOR_GREY, hovertemplate="Position %{x}: %{y:.3f}<extra></extra>", name="Anteil der Zeit"))
    if s is not None:
        fig.add_vline(x=s + 0.5, line_dash="dash", line_color=C.COLOR_BASE)
    fig.update_xaxes(title_text="Bestandsposition z vor der Bestellung")
    fig.update_yaxes(title_text="Anteil der Perioden")
    return _base(fig, 280)


def build_decomposition(a):
    """Kostenzerlegung je Periode: Fixkosten und Lager/Fehlmengen, DP gegen Base-Stock."""
    names = ["DP-Politik (s, S)", "Base-Stock (beste Stufe)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[a.fixed_dp, a.fixed_base], name="Fixkosten", marker_color=C.COLOR_BASE, hovertemplate="%{x}: %{y:.2f}<extra>Fixkosten</extra>"))
    fig.add_trace(go.Bar(x=names, y=[a.rest_dp, a.rest_base], name="Lager und Fehlmengen", marker_color=C.COLOR_DP, hovertemplate="%{x}: %{y:.2f}<extra>Lager und Fehlmengen</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Kosten je Periode")
    return _base(fig, 300)


def build_value(z, v, W, s, nonconvex, hull, S, title_v="Restkosten v(z) vor der Bestellung"):
    """v(z) und W(y), beide relativ zu ihrem Wert bei S, mit der unteren konvexen Hülle von v und den nicht konvexen Stellen."""
    z = np.asarray(z)
    ref = float(np.interp(S, z, v)) if S is not None else 0.0
    refW = float(np.interp(S, z, W)) if S is not None else 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=np.asarray(v) - ref, mode="lines+markers", name=title_v, line=dict(color=C.COLOR_DP, width=3), marker=dict(size=4),
                             hovertemplate="z = %{x}: %{y:.3f}<extra>v</extra>"))
    fig.add_trace(go.Scatter(x=z, y=np.asarray(W) - refW, mode="lines", name="Kosten nach der Bestellung W(y)", line=dict(color=C.COLOR_EOQ, width=2, dash="dot"),
                             hovertemplate="y = %{x}: %{y:.3f}<extra>W</extra>"))
    fig.add_trace(go.Scatter(x=z, y=np.asarray(hull) - ref, mode="lines", name="untere konvexe Hülle von v", line=dict(color=C.COLOR_GREY, width=2),
                             hovertemplate="z = %{x}: %{y:.3f}<extra>Hülle</extra>"))
    if nonconvex:
        idx = [int(np.where(z == p)[0][0]) for p in nonconvex]
        fig.add_trace(go.Scatter(x=[z[i] for i in idx], y=[np.asarray(v)[i] - ref for i in idx], mode="markers", name="nicht konvex (zweite Differenz < 0)",
                                 marker=dict(color=C.COLOR_BASE, size=12, symbol="diamond"), hovertemplate="z = %{x}<extra>nicht konvex</extra>"))
    v_rel = np.asarray(v) - ref
    top = float(v_rel.max())
    fig.update_xaxes(title_text="Position")
    # W steigt links steil an und würde v flachdrücken: die Achse folgt v (W wird oben abgeschnitten)
    fig.update_yaxes(title_text="Kosten relativ zum Wert bei S", range=[float(v_rel.min()) - 0.05 * top - 0.2, top * 1.12 + 0.2])
    return _base(fig, 380)


def build_hull_over_K(sweep):
    """Abweichung von der unteren Hülle geteilt durch K, über K (höchstens 1: K-Konvexität)."""
    m = sweep.K > 0
    fig = go.Figure(go.Scatter(x=sweep.K[m], y=sweep.hull_over_K[m], mode="lines+markers", line=dict(color=C.COLOR_DP, width=3), marker=dict(size=4),
                               hovertemplate="K = %{x}: %{y:.3f}<extra></extra>", name="Abweichung / K"))
    fig.add_hline(y=1.0, line_dash="dash", line_color=C.COLOR_GREY, annotation_text="Schranke 1 · K")
    fig.update_xaxes(title_text="Fixkosten K")
    fig.update_yaxes(title_text="Abweichung / K", range=[0, 1.1])
    return _base(fig, 300)


def build_sS_over_K(sweep):
    """s und S der DP-Politik über K sowie die beste Base-Stock-Stufe."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.S, mode="lines", name="S (DP)", line=dict(color=C.COLOR_DP, width=3), line_shape="hv"))
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.s, mode="lines", name="s (DP)", line=dict(color=C.COLOR_BASE, width=3), line_shape="hv"))
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.S_base, mode="lines", name="S der besten Base-Stock-Politik", line=dict(color=C.COLOR_GREY, width=2, dash="dot"), line_shape="hv"))
    fig.update_xaxes(title_text="Fixkosten K")
    fig.update_yaxes(title_text="Position")
    return _base(fig, 320)


def build_lot_vs_eoq(sweep, mu, h):
    """Losgröße S - s der DP gegen die EOQ-Menge sqrt(2 K mu / h)."""
    K = sweep.K
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K, y=np.sqrt(2 * K * mu / h), mode="lines", name="EOQ-Menge √(2 K μ / h)", line=dict(color=C.COLOR_EOQ, width=3)))
    fig.add_trace(go.Scatter(x=K, y=sweep.lot, mode="lines+markers", name="S - s der DP", line=dict(color=C.COLOR_DP, width=2), marker=dict(size=4), line_shape="hv"))
    fig.update_xaxes(title_text="Fixkosten K")
    fig.update_yaxes(title_text="Menge je Bestellung")
    return _base(fig, 320)


def build_gap_over_K(sweep, mu, h, extra=None):
    """Lücke der Base-Stock-Politik über K, mit senkrechten Marken beim EOQ-Takt 1, 2, 3, 4 Perioden (K = h mu T^2 / 2). `extra`: {Name: Lückenfeld}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.gap, mode="lines+markers", name="Base-Stock", line=dict(color=C.COLOR_BASE, width=3), marker=dict(size=4),
                             hovertemplate="K = %{x}: %{y:.2f} %<extra>Base-Stock</extra>"))
    for name, vals in (extra or {}).items():
        fig.add_trace(go.Scatter(x=sweep.K, y=vals, mode="lines+markers", name=name, line=dict(color=C.COLOR_CYCLE, width=2), marker=dict(size=4),
                                 hovertemplate="K = %{x}: %{y:.2f} %<extra>" + name + "</extra>"))
    for T in C.TAKT_STEPS:
        Kt = h * mu * T * T / 2.0
        if Kt <= sweep.K.max():
            fig.add_vline(x=Kt, line_dash="dot", line_color=C.COLOR_GREY, annotation_text=f"Takt {T}", annotation_position="top")
    fig.update_xaxes(title_text="Fixkosten K (senkrechte Marken: EOQ-Takt in Perioden)")
    fig.update_yaxes(title_text="Mehrkosten gegenüber der DP (%)")
    return _base(fig, 360)


def build_gap_heatmap(Ks, Ls, gaps):
    """Lücke in % über K (x) und Lieferzeit (y)."""
    fig = go.Figure(go.Heatmap(x=Ks, y=[f"Lieferzeit {l}" for l in Ls], z=gaps, colorscale="YlOrRd", colorbar=dict(title=dict(text="Lücke %", side="top"), nticks=4),
                               hovertemplate="K = %{x}, %{y}: %{z:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Fixkosten K")
    return _base(fig, 280)


def build_enum_heatmap(en, K, window_half=8):
    """Mehrkosten in % gegenüber dem besten Paar über (S, s), ausgeschnitten um das beste Paar; das Paar der DP ist markiert."""
    s, S, g, win = en["s"], en["S"], en["g"], en["window"]
    s0, S0 = en["best_pair"]
    s_axis = np.arange(s0 - window_half, s0 + window_half + 1)
    S_axis = np.arange(S0 - window_half, S0 + window_half + 1)
    grid = np.full((len(s_axis), len(S_axis)), np.nan)
    lookup = {(int(a), int(b)): float(c) for a, b, c, w in zip(s, S, g, win) if w}
    for i, sv in enumerate(s_axis):
        for j, Sv in enumerate(S_axis):
            val = lookup.get((int(sv), int(Sv)))
            if val is not None:
                grid[i, j] = 100.0 * (val / en["best_g"] - 1.0)
    fig = go.Figure(go.Heatmap(x=S_axis, y=s_axis, z=np.minimum(grid, 25.0), colorscale="Viridis_r", zmin=0, zmax=25,
                               colorbar=dict(title=dict(text="Mehrkosten % (bis 25)", side="top"), nticks=4), hovertemplate="S = %{x}, s = %{y}: +%{z:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[S0], y=[s0], mode="markers", marker=dict(symbol="x", size=14, color="white", line=dict(color="black", width=2)), name="bestes Paar"))
    fig.update_xaxes(title_text="Ziel S")
    fig.update_yaxes(title_text="Schwelle s")
    return _base(fig, 420)


def build_eoq_curve(Kv, mu, h, f):
    """Deterministische EOQ-Kurve C(Q) = K D / Q + h Q / 2 mit dem Optimum und dem Punkt bei Losgröße f · Q*."""
    qs = np.sqrt(2 * Kv * mu / h)
    Q = np.linspace(0.25 * qs, 4.0 * qs, 200)
    C_ = Kv * mu / Q + h * Q / 2
    fig = go.Figure(go.Scatter(x=Q, y=C_, mode="lines", line=dict(color=C.COLOR_EOQ, width=3), name="C(Q)", hovertemplate="Q = %{x:.1f}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[qs], y=[Kv * mu / qs + h * qs / 2], mode="markers", marker=dict(size=12, color=C.COLOR_DP), name="Optimum Q*"))
    fig.add_trace(go.Scatter(x=[f * qs], y=[Kv * mu / (f * qs) + h * f * qs / 2], mode="markers", marker=dict(size=12, color=C.COLOR_BASE, symbol="diamond"),
                             name=f"Losgröße {f:.2f} · Q*".replace(".", ",")))
    fig.update_xaxes(title_text="Losgröße Q")
    fig.update_yaxes(title_text="Kosten je Periode (deterministisch)")
    return _base(fig, 320)


def build_lot_curve(lots, pct):
    """Mehrkosten gegenüber der besten Politik je Losgröße S - s (bestes s je Q), Modell der Demo."""
    fig = go.Figure(go.Scatter(x=lots, y=pct, mode="lines+markers", line=dict(color=C.COLOR_DP, width=3), marker=dict(size=5),
                               hovertemplate="Losgröße %{x}: +%{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Losgröße S - s")
    fig.update_yaxes(title_text="Mehrkosten gegenüber der DP (%)", rangemode="tozero")
    return _base(fig, 300)


def build_hand_curve(xs, C1, C2):
    """Handbeispiel: Restkosten bei 1 und 2 Restperioden über der Position."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=C1, mode="lines+markers", name="C₁ (eine Periode übrig)", line=dict(color=C.COLOR_GREY, width=2)))
    fig.add_trace(go.Scatter(x=xs, y=C2, mode="lines+markers", name="C₂ (zwei Perioden übrig)", line=dict(color=C.COLOR_DP, width=3)))
    fig.update_xaxes(title_text="Position x", dtick=1)
    fig.update_yaxes(title_text="Restkosten")
    return _base(fig, 300)


def build_horizon_path(path, s_stat, S_stat):
    """(s_n, S_n) über die Zahl der Restperioden n gegen die stationäre Politik (nur n mit Bestellung im Fenster)."""
    ns = [i + 1 for i, st in enumerate(path) if st.s is not None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[path[n - 1].S for n in ns], mode="lines+markers", name="S_n", line=dict(color=C.COLOR_DP, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=ns, y=[path[n - 1].s for n in ns], mode="lines+markers", name="s_n", line=dict(color=C.COLOR_BASE, width=2), marker=dict(size=4)))
    if s_stat is not None:
        fig.add_hline(y=S_stat, line_dash="dot", line_color=C.COLOR_DP, annotation_text="S stationär", annotation_position="bottom right")
        fig.add_hline(y=s_stat, line_dash="dot", line_color=C.COLOR_BASE, annotation_text="s stationär", annotation_position="bottom right")
    fig.update_xaxes(title_text="Restperioden n")
    fig.update_yaxes(title_text="Position")
    return _base(fig, 300)
