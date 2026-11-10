"""Messraster der Demo: Lieferzeit 0 bis 3, p/h in {1, 2, 4, 9, 19, 39}, K = 0 bis 30 in Schritten von 0,5 (1.464 Zellen).

Je Zelle: exakte Kosten g der DP, ihre Form (s, S), beste Base-Stock-Politik (exakt bewertet), Lücke, Bestellanteile, Randmassen und die Konvexität der
Restkosten. Das Ergebnis liegt als data/messraster.csv im Repo (Semikolon, Dezimalpunkt); tests/test_claims.py zählt daran die Aussagen der README
und rechnet eine Stichprobe von Zellen live nach.

Aufruf (im Repo, unter einer Minute): python scripts/messraster.py
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bsw_constants as C  # noqa: E402
import bsw_convex as CV  # noqa: E402
import bsw_dp as D  # noqa: E402
import bsw_evaluation as ev  # noqa: E402
import bsw_policy as P  # noqa: E402

FIELDS = ["L", "ph", "K", "g", "gBase", "SBase", "gap_pct", "s", "S", "lot", "pure", "freq", "freqBase", "qmax", "mass_lo", "mass_hi", "qcap", "qcapBase",
          "massBase_lo", "massBase_hi", "nonconv_v", "nonconv_at_s", "nonconv_W", "hull_over_K", "kviol"]


def cell_rows(L: int, ph: int):
    model = ev.model_for(L, ph, C.DEFAULT_Q)
    table = ev.base_table(L, ph, C.DEFAULT_Q)
    for K in C.K_GRID:
        g, q, h, W, _ = D.rvi(model, K)
        g_ops, f, pi = P.evaluate_parts(model, q)
        st = P.structure(model, q, P.reached(pi))
        masses = P.boundary_masses(model, q, pi)
        gb, Sb, fb = table.best(K)
        qb = P.pol_base_stock(model, Sb)
        _, _, pib = P.evaluate_parts(model, qb)
        z, v, Wv = ev.window_arrays(model, h, W)
        bad = CV.nonconvex_points(z, v)
        hd, _ = CV.hull_defect(z, v)
        yield {
            "L": L, "ph": ph, "K": K, "g": f"{g:.9f}", "gBase": f"{gb:.9f}", "SBase": Sb, "gap_pct": f"{100 * (gb / g - 1):.6f}", "s": st.s, "S": st.S,
            "lot": st.S - st.s, "pure": int(st.pure), "freq": f"{f:.9f}", "freqBase": f"{fb:.9f}", "qmax": masses["qmax"], "mass_lo": f"{masses['mass_lo']:.3e}",
            "mass_hi": f"{masses['mass_hi']:.3e}", "qcap": f"{masses['qcap']:.3e}", "qcapBase": f"{float(pib[qb == model.spec.Q].sum()):.3e}",
            "massBase_lo": f"{float(pib[0]):.3e}", "massBase_hi": f"{float(pib[-1]):.3e}", "nonconv_v": len(bad), "nonconv_at_s": int(st.s in bad),
            "nonconv_W": len(CV.nonconvex_points(z, Wv)), "hull_over_K": f"{0.0 if K == 0 else hd / K:.6f}",
            "kviol": "" if K == 0 else f"{CV.k_convex_violation(v, K):.3e}",
        }


def main():
    out = ROOT / "data" / "messraster.csv"
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter=";")
        w.writeheader()
        n = 0
        for L in range(C.L_MIN, C.L_MAX + 1):
            for ph in C.PH_OPTIONS:
                for row in cell_rows(L, ph):
                    w.writerow(row)
                    n += 1
            print(f"Lieferzeit {L} fertig ({time.time() - t0:.0f} s)", flush=True)
    print(f"{n} Zellen nach {out}")


if __name__ == "__main__":
    main()
