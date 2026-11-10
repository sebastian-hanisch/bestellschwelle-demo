"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# ---- Modell (fest) -----------------------------------------------------------------------------------------------------------------------
MEAN = 3.0                  # Mittel der Poisson-Nachfrage je Periode
H = 1.0                     # Lagerkosten je Stück und Periode (p/h ist der Regler, p = p/h)
DMAX = 30                   # Nachfrage wird bei 30 abgeschnitten (Restmasse unter 1e-19 auf 30 gelegt)
ZMIN, YMAX = -40, 60        # Bestandsposition: untere und obere Zustandsgrenze (101 Zustände)

# ---- Regler ------------------------------------------------------------------------------------------------------------------------------
K_MIN, K_MAX, K_STEP, DEFAULT_K = 0.0, 30.0, 0.5, 5.0
L_MIN, L_MAX, DEFAULT_L = 0, 3, 1
PH_OPTIONS = (1, 2, 4, 9, 19, 39)
DEFAULT_PH = 9
Q_MIN, Q_MAX, DEFAULT_Q = 10, 40, 40
HORIZON_OPTIONS = (0, 1, 2, 3, 5, 10, 20, 40, 60)  # 0 = stationär (unendlicher Horizont)
DEFAULT_HORIZON = 0
HORIZON_MAX = 60

K_GRID = tuple(0.5 * i for i in range(61))  # 0 bis 30 in Schritten von 0,5

# ---- Auswertung --------------------------------------------------------------------------------------------------------------------------
WINDOW_LO = -10             # Fenster der Formprüfung bei endlichem Horizont (z >= -10)
HULL_LO, HULL_HI = -30, 55  # Fenster der Konvexitätsprüfung (Randeffekte der Zustandsgrenzen bleiben draußen)
ENUM_S_LO, ENUM_S_HI = -12, 40  # Fenster der Aufzählung aller (s,S): s >= -12, S <= 40 (1.378 Paare)
LOT_MAX = 30                # größte Losgröße S - s im Tab EOQ
EOQ_BAND_LO, EOQ_BAND_HI = 0.8, 1.25
EOQ_MIN_K = 2.0             # Vergleich S - s gegen EOQ-Menge erst ab K = 2
TAKT_STEPS = (1, 2, 3, 4)   # EOQ-Takt in Perioden, als senkrechte Marken im Diagramm

COLOR_DP = "#2F6B65"
COLOR_BASE = "#c0504d"
COLOR_CYCLE = "#e0a030"
COLOR_EOQ = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.55)"

DEFAULT_PRESET = "Standard (K = 5, Lieferzeit 1, p/h = 9)"
PRESETS = {
    DEFAULT_PRESET: {"k": 5.0, "l": 1, "ph": 9, "q": 40, "hor": 0},
    "Von Hand (Handbeispiel)": {"k": 2.0, "l": 0, "ph": 4, "q": 40, "hor": 0},
    "Kein Unterschied (K = 0,5, Lieferzeit 1)": {"k": 0.5, "l": 1, "ph": 9, "q": 40, "hor": 0},
    "Takt eine Periode (K = 1,5)": {"k": 1.5, "l": 1, "ph": 9, "q": 40, "hor": 0},
    "Teure Bestellung (K = 20)": {"k": 20.0, "l": 1, "ph": 9, "q": 40, "hor": 0},
    "Lange Lieferzeit (K = 10, Lieferzeit 3)": {"k": 10.0, "l": 3, "ph": 9, "q": 40, "hor": 0},
    "Billige Fehlmenge (p/h = 1)": {"k": 5.0, "l": 1, "ph": 1, "q": 40, "hor": 0},
}
PRESET_HELP = {
    DEFAULT_PRESET: "Fixkosten K = 5, Lieferzeit 1, Fehlmenge neunmal so teuer wie ein Stück im Lager: die Voreinstellung der Demo.",
    "Von Hand (Handbeispiel)": "K = 2, Lieferzeit 0, p/h = 4 wie im Handbeispiel (Tab „Von Hand“, Nachfrage 0, 1, 2). Die übrigen Tabs rechnen weiter mit Poisson-Nachfrage, Mittel 3.",
    "Kein Unterschied (K = 0,5, Lieferzeit 1)": "Fixkosten so klein, dass die beste Ein-Schwellen-Regel (Base-Stock) die exakte Politik noch trifft (Lücke 0,00 %).",
    "Takt eine Periode (K = 1,5)": "K = h · Mittel / 2: Die EOQ-Formel rät zu einer Bestellung je Periode. Trotzdem liegt Base-Stock in diesem Fall schon über der exakten Politik.",
    "Teure Bestellung (K = 20)": "Jede Bestellung kostet 20: Base-Stock bestellt fast jede Periode und zahlt vor allem Fixkosten.",
    "Lange Lieferzeit (K = 10, Lieferzeit 3)": "Vier Perioden Wiederbeschaffungszeit: s und S liegen höher als bei Lieferzeit 1; die Lücke der einen Schwelle beträgt 46,1 %.",
    "Billige Fehlmenge (p/h = 1)": "Eine fehlende Einheit kostet nur so viel wie eine gelagerte: Die Politik nimmt Rückstand in Kauf, s liegt niedrig.",
}

# Dieselbe Fassung steht in der README (tests/test_claims.py vergleicht sie); Zahlen: data/messraster.csv und tests/test_claims.py.
ASSUMPTIONS_MARKDOWN = """
- **Modell:** ein Artikel, Poisson-Nachfrage mit Mittel 3 je Periode (unabhängig und gleichverteilt über die Zeit), Rückstand (nicht gedeckte Nachfrage wird nachgeliefert, nie verloren), Lagerkosten h = 1 je Stück und Periode, Fehlmengenkosten p = p/h je fehlendem Stück und Periode (p/h = 1, 2, 4, 9, 19, 39), Bestellkosten je Stück 0, Fixkosten K je Bestellung (0 bis 30 in Schritten von 0,5), Lieferzeit 0 bis 3 Perioden. Alle Parameter sind erfunden; es gibt keine Daten.
- **Zustand = Bestandsposition:** Bestand plus Bestelltes minus Rückstand. Das ist eine Rechenidentität (die Kosten einer Bestellung fallen L + 1 Perioden später an), kein Satz. Gegen die Rechnung mit vollem Pipeline-Zustand geprüft: 16 Zellen (Lieferzeit 0 bis 3, K = 0, 2, 5, 10, Bestellgrenze 10), Abweichung der Kosten höchstens 1,8e-7; die Abweichung ist ein Randeffekt der unteren Zustandsgrenze des vollen Modells und verschwindet, wenn man sie weiter nach unten legt.
- **Zustandsgrenzen und Bestellgrenze:** Positionen von −40 bis 60 (101 Zustände), Bestellmenge höchstens Q = 40. Im Messraster (1.464 Zellen) liegt die DP-Politik höchstens 9e-14 der Zeit an der Bestellgrenze, die Masse an den Zustandsgrenzen ist höchstens 3e-16, erreicht werden Positionen von −26 bis 28. Mit dem Regler Q unter etwa 23 bis 28 (K = 30, je nach p/h und Lieferzeit) greift die Grenze; die App zeigt dann den Anteil, und alle Aussagen gelten für diese Grenze.
- **(s, S) ist hier ein Messbefund, kein Satz.** Nach Feinberg (arXiv 1606.00957, Volltext gelesen, Abschnitte 1 und 4) ist die (s, S)-Form (Scarf; für den unendlichen Horizont Iglehart sowie Veinott und Wagner) optimal unter Rückstand, Lieferzeit null, unabhängiger gleichverteilter Nachfrage, konvexen Lager- und Fehlmengenkosten und unbeschränkter Bestellmenge; bei endlichem Horizont nur mit Zusatzbedingungen (etwa einer Wachstumsbedingung), bei begrenzter Kapazität im Allgemeinen nicht. Scarf (1959/60) ist nicht gelesen, nur über Feinberg zitiert. Dass die DP in der Demo mit Lieferzeit bis 3 und Bestellgrenze 40 reine (s, S)-Form hat, ist gemessen: in 1.464 von 1.464 Zellen auf den erreichten Positionen. Die Bestellgrenze greift dort praktisch nie (Anteil der Zeit höchstens 9e-14), die Ausnahme „begrenzte Kapazität“ wird also nicht widerlegt, sondern nicht berührt.
- **Endlicher Horizont:** n Rückwärtsschritte ab Endkosten 0, die Kosten fallen zum Bestellzeitpunkt an. Die Form (s, S) ist im Fenster z ≥ −10 geprüft: in 2.160 (Zelle, n)-Paaren (36 Zellen, n = 1 bis 60) hatte sie reine Form; bei p/h = 1, K = 30 liegt s bei n = 1 unter dem Fenster (bei Lieferzeit 0 auch bei n = 2); dort wird im Fenster nicht bestellt.
- **Schwelle K-Stern** (das größte K, bei dem Base-Stock noch optimal ist) ist mit der Toleranz 1e-8 Kosteneinheiten durch Bisektion bestimmt und liegt in den 24 Zellen unregelmäßig zwischen 0,014 und 0,89. Eine Erklärung über die Ganzzahligkeit der Nachfrage ist eine Vermutung und nicht gemessen.
- **Nicht untersucht:** die Ursache der Nichtkonvexität der Kosten nach der Bestellung W(y) bei kleinem p/h und die Ursache nicht konvexer Stellen am unteren Fensterrand; die Optimalität von (s, S) außerhalb der gemessenen Zellen.
- **EOQ:** Die Zahlen +25,0 % (Losgröße um Faktor 2 daneben) und +6,1 % (Parameter K/h um Faktor 2 daneben) gelten für die deterministische Kurve, nicht für das Modell der Demo (Tab EOQ zeigt die Zahlen dort). Die EOQ-Formel ist eine Lehrbuchformel, hier nachgerechnet.
- **Zyklus-Politik** (alle T Perioden bis S bestellen) ist ohne Bestellgrenze gerechnet; bei Q = 40 ist der Einfluss der Grenze auf die besten T (bei p/h = 9 bis 5) vernachlässigbar; bei kleinerem Q ist der Vergleich nicht fair, die App zeigt dann statt des Verdikts einen Hinweis.
- **Rechenzeit:** alles wird live gerechnet (Wertiteration auf 101 Zuständen, wenige Millisekunden je Zelle); das K-Raster und die Aufzählung aller Paare werden je Kombination von Lieferzeit, p/h und Bestellgrenze einmal berechnet und zwischengespeichert. Zeitangaben sind Größenordnungen, keine Zusage.
"""
