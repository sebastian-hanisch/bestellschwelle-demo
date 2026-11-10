# Bestellschwelle: aus Base-Stock wird (s,S) – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bestellschwelle-demo.streamlit.app/)**

Demo der „Konzepte“-Reihe zu Bestandspolitiken für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie zeigt, was passiert, wenn jede Bestellung **Fixkosten K** kostet: Die Politik **Base-Stock** („bestelle bis S, sobald die Bestandsposition unter S fällt“) kennt keine Fixkosten und bestellt bei Fixkosten oberhalb einer Schwelle K\* zu oft. Die exakte Rückwärtsrechnung (dynamische Programmierung, DP) setzt eine zweite Schwelle: **bestelle erst, wenn die Position auf s oder darunter gefallen ist, dann bis S**. Kante in der Reihe: Base-Stock hat die Schwäche „keine Fixkosten“, diese Demo behebt sie. Die **Bestandsposition** z (Bestand plus Bestelltes minus Rückstand) wird als gegeben übernommen und nur erklärt.

Vehikel: ein Artikel, Poisson-Nachfrage mit Mittel 3 je Periode, Lagerkosten h = 1, Fehlmengenkosten p = p/h (Strafverhältnis 1 bis 39), Rückstand, Lieferzeit 0 bis 3, Bestellkosten je Stück 0.

## Warum dieses Problem

Fixkosten je Bestellung sind ein wichtiger Grund, warum „immer bis S auffüllen“ in der Praxis nicht gilt. Die Kernfrage der Demo: **Ab wann lohnt eine zweite Schwelle, und wo hilft die EOQ-Formel (Losgröße) dabei?** Der Befund hat zwei Ehrlichkeitspunkte: Der große Prozentsatz bei hohem K ist zum größten Teil Arithmetik (Fixkosten je Bestellung mal Bestellhäufigkeit), und die Schwelle, ab der Base-Stock nicht mehr optimal ist, liegt unregelmäßig.

## Modell

Zustand ist die Bestandsposition z vor der Bestellung, Aktion die Bestellmenge q (0 bis Q). Mit der Nachfrage über L + 1 Perioden und G(y) = E[h (y − D)⁺ + p (D − y)⁺] gilt für W(y) = G(y) + E v(y − D):

g + v(z) = min( W(z), K + min über z < y ≤ z + Q von W(y) ).

Das ist eine Rechenidentität (die Kosten einer Bestellung fallen L + 1 Perioden später an), kein Satz. Der Rest steht im 📐-Expander der App: (s,S)-Form, K-Konvexität nach Scarf, EOQ und Zyklus-Politik. Die Politik wird auf dem begrenzten Zustandsraum exakt bewertet (stationäre Verteilung der Position, lineares Gleichungssystem, kein Simulationsrauschen).

## Methodik

- **Exakte DP:** relative Wertiteration auf 101 Positionen (−40 bis 60), Bestellgrenze 40; wenige Millisekunden je Zelle.
- **Beste Ein-Schwellen-Regel:** jede Base-Stock-Stufe S exakt bewertet, die beste genommen. Für jede feste Politik sind die Kosten linear in K (g = Kosten für Lager und Fehlmengen + K · Bestellanteil); darauf beruht die schnelle Auswertung über K.
- **Aufzählung aller (s, S):** 1.378 Paare (s ≥ −12, S ≤ 40), jedes exakt bewertet; zweiter Rechenweg neben der DP.
- **Konvexität:** zweite Differenzen, untere konvexe Hülle und K-Konvexität (Steigungsform) der Restkosten im Fenster z = −30 bis 55.
- **EOQ:** deterministische Kurve K·D/Q + h·Q/2 und dieselbe Frage im Modell der Demo (bestes s je Losgröße S − s; Politik für den falschen Parameter K).
- **Zyklus-Politik:** alle T Perioden bis S bestellen, geschlossene Formel, beste T und S je K.
- **Schwelle K\*:** größtes K, bei dem Base-Stock die DP noch bis auf 1e-8 Kosteneinheiten trifft (Bisektion auf [0, 4]).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten im Modell der Demo: Poisson-Nachfrage Mittel 3, h = 1, Rückstand, Bestellgrenze 40, Positionen −40 bis 60, Rastergrößen wie angegeben. „Lücke“ sind die Mehrkosten der besten Base-Stock-Politik in Prozent der DP-Kosten. Die Spalte „Test“ nennt den Test in `tests/test_claims.py`, der die Zahl prüft (Raster = Zählung an `data/messraster.csv`).

| Frage | Befund | Test |
|---|---|---|
| Form der DP-Politik | In **1.464 von 1.464** Zellen (Lieferzeit 0 bis 3, p/h = 1, 2, 4, 9, 19, 39, K = 0 bis 30 in Schritten von 0,5) hat die DP-Politik auf den erreichten Positionen die reine Form (s, S). Das ist ein Messbefund, kein Satz. | `test_dp_has_pure_sS_form_in_all_1464_cells` (Raster) |
| Ohne Fixkosten | Bei K = 0 ist die DP Base-Stock mit S gleich dem kleinsten y mit F(y) ≥ p/(p+h) der Nachfrage über L + 1 Perioden: 24 von 24 Zellen; bei p/h = 9 ist S = 5, 9, 13, 17 für Lieferzeit 0 bis 3. | `test_k_zero_dp_is_base_stock_at_the_quantile_in_24_of_24_cells`, `test_S_at_k_zero_for_p_over_h_9_is_5_9_13_17` |
| Lücke über K (p/h = 9, Lieferzeit 0 / 1 / 2 / 3) | K = 0,5: 0,00 / 0,00 / 0,53 / 0,91 %; K = 1: 0,33 / 1,27 / 1,49 / 2,41 %; K = 2: 5,87 / 5,67 / 5,59 / 6,15 %; K = 5: 20,8 / 22,0 / 20,9 / 21,5 %; K = 20: 95,8 / 94,0 / 91,1 / 88,6 %; K = 30: 134,8 / 130,8 / 128,1 / 125,4 % | `test_gap_table_for_p_over_h_9_live` |
| Wie viele der 24 Zellen (Lieferzeit × p/h) haben Lücke ≥ 1 % / ≥ 5 % / ≥ 10 %? | K = 0,5: 8 / 0 / 0; K = 1: 17 / 4 / 0; K = 1,5: 24 / 11 / 4; K = 2: 24 / 16 / 8; K = 3: 24 / 24 / 16; K = 4,5: 24 / 24 / 24 | `test_gap_counts_over_the_24_cells_per_K` (Raster) |
| EOQ-Takt eine Periode (K = h·μ/2 = 1,5) | Lücke 1,90 bis 12,23 % über die 24 Zellen; bei p/h = 9: 3,26 / 3,60 / 2,86 / 4,15 % (Lieferzeit 0 bis 3). Bei K = 1,5 ist die Lücke also schon in allen 24 Zellen mindestens 1 %. | `test_gap_range_at_takt_one_period_k_1_5` (Raster) |
| Kein Unterschied | In 9 Zellen mit K > 0 ist die Lücke 0,00 % (alle bei K = 0,5: Base-Stock ist dort noch optimal). Größte Lücke im Raster: 212,8 % (Lieferzeit 1, p/h = 1, K = 30). | `test_nine_cells_with_k_above_zero_have_zero_gap`, `test_largest_gap_is_212_8_percent` (Raster) |
| Abhängigkeit von p/h | Für K ≥ 2 ist die Lücke bei kleinerem p/h größer (K = 5, Lieferzeit 1: 50,5 % bei p/h = 1 gegen 15,0 % bei p/h = 39); bei K ≤ 1,5 ist das nicht durchgehend so. | `test_gap_is_larger_for_small_p_over_h` (Raster) |
| Warum die Lücke so groß ist | Base-Stock bestellt in 1 − e⁻³ = 95,0 % der Perioden; seine Kosten sind exakt linear in K (in 1.440 von 1.440 Zellen mit K > 0 bleibt die beste Stufe S wie bei K = 0, Abweichung unter 1e-5). Lieferzeit 1, p/h = 9, K = 20: DP 12,18 = 4,80 Fixkosten + 7,38 Lager und Fehlmengen (Bestellanteil 0,24); Base-Stock 23,62 = 19,00 + 4,61. Die DP spart 14,2 Fixkosten und zahlt 2,8 mehr für Lager und Fehlmengen. | `test_base_stock_orders_in_95_percent_and_costs_are_linear_in_K` (Raster), `test_decomposition_for_lead_time_one_p_over_h_9_k_20` |
| Steigung der DP-Kosten über K | Der Differenzenquotient von g über K liegt in 1.440 von 1.440 Schritten zwischen den Bestellanteilen der beiden Rasterpunkte. | `test_slope_of_dp_costs_over_K_is_the_order_frequency` (Raster) |
| s und S über K | Auf dem Raster (1.440 Schritte, K wächst um 0,5) steigt s nicht, S fällt nicht, Kosten und Lücke fallen nicht (je 0 Verletzungen). p/h = 9, Lieferzeit 0 / 1 / 2 / 3: bei K = 0 (s, S) = (4, 5), (8, 9), (12, 13), (16, 17); bei K = 30 (1, 15), (4, 18), (8, 22), (11, 26). | `test_s_never_rises_S_never_falls_costs_and_gap_never_fall_over_K` (Raster), `test_s_and_S_over_K_for_p_over_h_9` |
| Schwelle K\* (ab der Base-Stock nicht mehr optimal ist) | Zwischen **0,014 und 0,89** über die 24 Zellen, **nicht monoton** in Lieferzeit (p/h = 4: 0,51 / 0,29 / 0,10 / 0,15) und in p/h (Lieferzeit 0). p/h = 9: 0,892 / 0,555 / 0,255 / 0,014. Bei K = 2·K\* beträgt die Lücke nur 0,03 bis 4,78 %. Warum K\* so unregelmäßig ist, ist nicht untersucht (eine Deutung über die Ganzzahligkeit der Nachfrage wäre eine Vermutung). | `test_kstar_range_over_the_24_cells_and_non_monotone`, `test_gap_at_twice_kstar_is_small` |
| EOQ-Menge gegen S − s | Das Verhältnis (S − s) / √(2·K·μ/h) liegt in den 1.368 Zellen mit K ≥ 2 zwischen 0,52 und 1,65 (Mittel 1,11), in **1.089** zwischen 0,8 und 1,25; bei p/h = 9: 0,71 bis 1,31. Ein Band, keine Gleichheit. | `test_lot_over_eoq_quantity_band` (Raster) |
| Textfalle EOQ, deterministische Kurve | Losgröße um Faktor 2 daneben: +25,0 %. **Parameter K/h** um Faktor 2 daneben (Losgröße × √2): +6,07 %; um Faktor 4: +25 %. Die Bezugsgröße ist jeweils die Losgröße bzw. der Parameter. | `test_deterministic_eoq_numbers` |
| Dieselbe Frage im Modell der Demo (Lieferzeit 0 bis 3, K = 2, 5, 10, 20, 30, p/h = 9; 20 Zellen) | Halbe Losgröße S − s: +2,4 bis +19,5 %, doppelte: +5,1 bis +23,8 % (je bestes s). Politik für 2·K statt K, bewertet mit dem wahren K: +1,1 bis +6,7 %; für K/2: 0,0 bis +6,5 %. Bezogen auf die Mehrkosten durch Fixkosten (g − g bei K = 0) kostet die Politik für 2·K +3,9 bis +11,0 %. Die Zahlen +25 % und +6,1 % gelten **nicht** im Modell der Demo. | `test_lot_and_parameter_errors_in_the_model_of_the_demo` |
| Flachheit um das Optimum | Lieferzeit 1, K = 10, p/h = 9: Losgröße 7 und 9 etwa 0,1 % über dem Optimum (+0,04 % und +0,11 %), 5: +5,0 %, 11: +2,6 %, 3: +20,9 %. | `test_cost_over_lot_size_is_flat_around_the_optimum` |
| Feste Zykluslänge (alle T Perioden bis S) | Bei K = 5, 10, 20, 30 (Lieferzeit 0 bis 3, p/h = 9; 16 Zellen) kostet die beste Zyklus-Politik 7,4 bis 17,5 % mehr als die DP (Lieferzeit 0: T = 2, 3, 4, 5); bei K = 1 und 2 ist T = 1, also Base-Stock. | `test_cycle_policy_is_7_to_18_percent_above_the_dp_for_k_5_and_more`, `test_cycle_policy_equals_base_stock_at_k_one_and_two` |
| Konvexität der Restkosten v(z) | K = 0: konvex in 24 von 24 Zellen. K > 0: **nicht konvex in 1.440 von 1.440** Zellen, und zwar immer an der Stelle z = s (1.440 von 1.440). v ist K-konvex (Steigungsform, größte Verletzung unter 1e-12); die Abweichung von der unteren konvexen Hülle ist im Fenster z = −30 bis 55 höchstens 0,975·K (unter der Schranke K aus der K-Konvexität; der Wert hängt von Fenster und Bestellgrenze ab). | `test_convexity_counts` (Raster) |
| Kosten nach der Bestellung W(y) | Für p/h = 9, 19, 39 ist W in allen 720 Zellen mit K > 0 konvex; bei p/h = 1, 2, 4 in einem Teil der Zellen nicht (208, 156, 28 von je 240). Ursache nicht untersucht. | `test_W_is_convex_for_p_over_h_9_19_39_and_not_in_all_cells_below` (Raster) |
| Aufzählung aller (s, S) | Trifft die DP in 40 Zellen (Abweichung unter 1e-9). Je Zelle liegen 1 bis 3 Paare höchstens 0,1 %, 1 bis 8 höchstens 1 %, 4 bis 47 von 1.378 höchstens 5 % über dem Optimum (die Kostenfläche ist flach). | `test_enumeration_pairs_near_the_optimum_over_40_cells`, `test_enumeration_hits_the_dp_optimum` (Orakel) |
| Bestellgrenze | Kleinste Bestellgrenze mit Anteil der Zeit an der Grenze unter 1e-4 bei K = 30 (Schritt 1): 23, 23, 23, 24 (p/h = 9) bzw. 27, 27, 27, 28 (p/h = 1) für Lieferzeit 0 bis 3; 40 ist die Reserve. Größere Grenzen (Q = 80, Positionen −80 bis 100) ändern die Kosten in 4 Zellen nicht. Im ganzen Raster: Anteil an der Bestellgrenze höchstens 9e-14, Masse an den Zustandsgrenzen unter 3e-16, erreichte Positionen bis zur größten Bestellmenge 38. | `test_smallest_order_limit_with_boundary_mass_below_1e_4`, `test_larger_bounds_do_not_change_the_costs`, `test_boundary_masses_in_the_whole_raster` |
| Endlicher Horizont | In 2.160 (Zelle, n)-Paaren (36 Zellen, n = 1 bis 60 Restperioden, Endkosten 0, Fenster z ≥ −10) hat die Politik reine (s, S)-Form; bei p/h = 1, K = 30 wird im Fenster bei n = 1 nicht bestellt (bei Lieferzeit 0 auch bei n = 2). Lieferzeit 0, p/h = 9, K = 2: (3, 5) bei n = 1, ab n = 2 gleich der stationären (3, 6). | `test_finite_horizon_keeps_the_sS_form_in_2160_pairs`, `test_finite_horizon_converges_to_the_stationary_policy_for_standard_cells` |
| Handbeispiel (Nachfrage 0, 1, 2 mit ¼, ½, ¼; h = 1, p = 4, K = 2, Lieferzeit 0) | L(y) = 4, 5/4, 1, 2, 3 für y = 0 bis 4; C₁ = 3, 3, 5/4, 1, 2, 3; C₂ = 37/8, 37/8, 61/16, 21/8, 53/16, 5 für x = −1 bis 4; zweite Differenz von C₂ bei x = 0: −13/16. Bestes Paar (0, 2) mit g = 2 (Bestellanteil 9/20), beste Base-Stock-Politik (1, 2) mit g = 5/2: **25,0 %** mehr. | `test_hand_module_matches_the_hand_calculation`, `test_hand_stationary_costs` (`tests/test_units.py`), `tests/test_oracle_hand.py` |
| Rechenzeit | Eine DP-Zelle braucht einige Millisekunden (Test: unter 250 ms, großzügige Schranke); keine Zusage, unter Last mehr. | `test_one_dp_cell_takes_a_few_milliseconds` |

**Messung dieser README, nicht im Test:** (a) Die Vormessung der Zustandsreduktion (Rechnung auf der Position gegen volles Pipeline-Modell, Bestellgrenze 10, 16 Zellen) fand Abweichungen der Kosten bis 1,8e-7 (Randeffekt der unteren Zustandsgrenze des vollen Modells, nach Verlegung der Grenze 2e-13); der Test `tests/test_oracle_voll.py` prüft 6 Zellen auf unter 1e-6. (b) Die Vormessung der Simulation (2.000 Läufe × 6.000 Perioden, vier Zufalls-Seeds 0 bis 3) lag höchstens 2,4 Standardfehler vom exakten Wert; der Test prüft kleinere Läufe auf 4,5 Standardfehler. (c) Die Rasterzählungen stehen in `data/messraster.csv`; der Test zählt an der Datei und rechnet 24 Zellen live nach, die übrigen Zellen werden von `scripts/messraster.py` erzeugt und sind nicht einzeln im Test neu gerechnet.

## Annahmen, Grenzen und bekannte Probleme

(Derselbe Text steht im Ausklapper „⚠️ Annahmen, Grenzen und bekannte Probleme“ der App; `tests/test_app.py` vergleicht beide Fassungen.)

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

## Verifikation

Gesamt **185 Tests** (Laufzeit seriell rund eine Minute, mit der Test-Umgebung des Portfolios gemessen; unter Last mehr).

- **Einheiten** (`tests/test_units.py`): Nachfrage, Kosten einer Periode, Rückwärtsschritt, Politik, Konvexität, EOQ und Zyklus je Funktion an der Mini-Instanz (Handbeispiel); Handrechnung in `tests/conftest.py`.
- **Orakel:** `test_oracle_voll.py` (volle Pipeline-Rechnung gegen Rechnung auf der Position, 6 Zellen, Aktion je Position), `test_oracle_enum.py` (Aufzählung aller (s, S) gegen DP in 24 Zellen, dazu ein unabhängiger Aufbau mit scipy-Poisson und Potenziermethode), `test_oracle_howard.py` (Politik-Iteration und allgemeine Wertiteration über alle Aktionen, 6 Zellen), `test_oracle_sim.py` (Simulation des vollen Systems, vier Zufalls-Seeds je Zelle, 4 Zellen), `test_oracle_eoq.py` (EOQ-Formel gegen Aufzählung und numerische Integration, Zyklus-Formel gegen Phasenkette, T = 1 gegen Base-Stock), `test_oracle_hand.py` (Handbeispiel in Brüchen, rekursiv und per Potenziermethode).
- **Zweig-Tests:** jede Variante weicht auf mindestens einer Instanz ab (Lieferzeit ändert s und S; p/h ändert S; K > 0 ändert die Politik gegenüber K = 0; Bestellgrenze 10 ändert die Kosten bei K = 30; endlicher Horizont ändert (s, S) bei n = 1); `tests/test_edge_cases.py`. Dass die Lücke bei K = 0,5 in 9 Zellen 0,00 % ist, ist kein toter Zweig, sondern Base-Stock gleich Optimum (K unter K\*).
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten; je Preset die erwartete Verdikt-Zeile (`tests/test_presets.py`).
- **Randfälle** (`tests/test_edge_cases.py`): K = 0, Lieferzeit 0 und 3, kleinster Zustandsraum (zwei Positionen), nie bestellen, Nachfrage 0, K = 1e6.
- **Oberfläche:** AppTest-Rauchtest aller sechs Tabs (jeder Preset, jeder Regler an Minimum und Maximum, drei ausgewählte Eckeinstellungen des Reglerraums, Permalink), Footer, Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Regleränderung hebt die Markierung auf), Laufzeit-Importe.
- **Zahlen der README:** `tests/test_claims.py` (Tabelle „Befunde“ oben).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfrage, sechs Tabs (Von Hand, Politik, Wertfunktion, Über K, Alle (s,S), EOQ), fünf Experimente |
| `bsw_demand.py` | Poisson-Nachfrage, Summe über mehrere Perioden, Newsvendor-Quantil |
| `bsw_cost.py` | Kosten einer Periode G(y) |
| `bsw_dp.py` | Zustandsraum, Rückwärtsschritt, relative Wertiteration, endlicher Horizont |
| `bsw_policy.py` | (s,S)- und Base-Stock-Politik, exakte Bewertung, Form, Randmassen, Aufzählung |
| `bsw_convex.py` | zweite Differenzen, untere konvexe Hülle, K-Konvexität |
| `bsw_eoq.py` | EOQ-Formeln, Zyklus-Politik |
| `bsw_hand.py` | Handbeispiel in exakten Brüchen |
| `bsw_evaluation.py` | Einstellungen, Kennzahlen, Sweeps über K, Schwelle K\*, Verdikte (reine Funktionen) |
| `bsw_constants.py` | Regler-Grenzen, feste Annahmen, Presets, Text „Annahmen, Grenzen und bekannte Probleme“ |
| `bsw_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) |
| `bsw_layout.py` | Fester Kopf (Stylesheet) |
| `bsw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/messraster.csv` | Messraster 1.464 Zellen (rund 250 KB), erzeugt von `scripts/messraster.py` |
| `scripts/messraster.py` | erzeugt das Messraster (unter einer Minute) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Einheiten, Orakel, Zahlen der README, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Verlorene Verkäufe statt Rückstand, zeitlich veränderliche Nachfrage, mehrere Artikel und Preisabschläge für Mengen.
- Der Nachweis der (s, S)-Form für das Modell der Demo (hier nur gemessen).
- Eine Erklärung, warum K\* unregelmäßig liegt, und die Ursache der Nichtkonvexität von W(y) bei kleinem p/h.
- Zyklus-Politik mit Bestellgrenze (die Formel rechnet ohne).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Messraster neu erzeugen: `python scripts/messraster.py`.

Gebaut mit Streamlit, Plotly und NumPy (SciPy nur für die Gegenproben der Tests).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html).
