# Trennschärfe und globales Fehlerbudget (08.11.2026)

Umsetzung der Punkte 1 und 2 aus der Analyse vom 08.11.: **Effizienter testen** und **ein globales Fehlerbudget**.

Nur Research. Keine Schwelle wurde gelockert. ERR-d6695229 und alle bereits vergebenen Slots bleiben unverändert.

## 1. Korrektur der eigenen Analyse: Die Alternierung ist kein Effizienzverlust

In der Analyse stand, die Alternierung verwerfe die Hälfte der Fenster. Ein Test über alle 24 Fenster hätte demnach 26,6 % statt 7,6 % Power. **Das war falsch.** Die Rechnung setzte unabhängige Fenster voraus. Benachbarte Haltefenster teilen aber Renditezeiträume.

Geprüft wurde mit einer Simulation, die die Überlappung nachbildet:
- Tages-IC = μ + a · Mittel der nächsten h Marktschocks + b · eigenes Rauschen;
- h = 5, Fenster mit je h Stichtagen, 200 000 Wiederholungen je Zelle.

**Testgröße unter H0** (Vielfaches des nominalen α, Überlappung a = b = 1):

| Test | α = 0,05, n = 24 | α = 0,01, n = 24 | α = 0,002, n = 8 | α = 0,002, n = 24 |
|---|---|---|---|---|
| A Alternierung (bisher) | 1,01× | 1,01× | 1,08× | 1,01× |
| B alle Fenster, HAC (Lag 1), df = n−1 | 1,21× | 1,50× | **1,82×** | **1,98×** |
| C alle Fenster, Schranke Var ≤ 2s²/n, df = n−1 | 0,51× | 0,35× | 0,36× | 0,23× |
| D alle Fenster, HAC, df = n/2−1 | 1,05× | 0,99× | 0,04× | 0,82× |

Bei starker Überlappung ohne Eigenrauschen (a = 1, b = 0) bestätigt B bis 2,4-fach zu oft.

**Power** (α = 0,002, n = 24, μ = 0,3 in Einheiten der Rauschstreuung):

| A | B | C | D |
|---|---|---|---|
| 0,03 | 0,07 | 0,02 | 0,04 |

**Folgen:**
- Der einzige Test mit deutlich mehr Power (B) ist **antikonservativ**.
- Die gültige Schranken-Variante (C) hat *weniger* Power als die Alternierung.
- D ist an der Grenze (bis 1,10×) und bringt kaum etwas.

**Die Alternierung bleibt.** Sie ist nahezu exakt und bei den gegebenen Abhängigkeiten effizient. Die Information wächst mit der Kalenderzeit (unabhängige Marktschock-Perioden), nicht mit der Zahl überlappender Stichtage.

Echte Hebel für Power sind deshalb:
- mehr unabhängige Perioden (kürzere Horizonte, mehr Zeit);
- weniger gemeinsames Rauschen (breiterer Querschnitt);
- **mehr Alpha je sinnvollem Test**: weniger aussichtslose Tests (Power-Gate) und ein besser verteiltes Budget (Ledger).

Das setzen die folgenden Punkte um.

## 2. Power-Gate vor der Registrierung (`research_lab.design_power`, Protokoll error-research-v3)

**Design-Power:** Simuliert wird exakt die Bestätigungsregel (`look_outcome`):
- Fenstermittel ≤ 0 verwirft;
- einseitig p < Teil-Alpha bestätigt;
- Prüfungen bei 4, 8 und 12 Fenstern;
- Teil-Alphas nach O'Brien-Fleming aus der Ledger-Zuteilung, die die Hypothese erhielte;
- fester Seed.

Unter H0 bestätigt der Plan höchstens mit seinem Alpha (Test `test_design_power_holds_size_under_h0_and_grows_with_effect`).

**Gate:**
- Die Streuung wird aus den *gesehenen* unabhängigen Fenstern geschätzt (Entdeckung und Walk-Forward-Screen, mindestens 3 Fenster).
- Der Effekt ist 0,5 × Entdeckungsmittel, als Korrektur für den Gewinnerfluch.
- Registriert wird nur bei einer Power ≥ 0,5.
- Sonst gibt es eine Datenlücke `underpowered` mit MDE (kleinstem nachweisbarem Effekt) und Streuung. **Kein Alpha wird verbraucht.**
- Alle Werte sind vorab festgelegt und nicht an Ergebnissen gewählt.

**Nebenbefund behoben:** v2 vergab Slots auch an Hypothesen, die schon im Walk-Forward verworfen waren und nie forward getestet werden. Unter v3 erhalten sie kein Alpha.

## 3. Globales Fehlerbudget (`config/alpha_ledger.yaml`, `option_learner.alpha_allocate`)

**Vorher:** Drei getrennte Budgets mit je 0,05 (Options-Varianten, Alpha-to-Options-Pläne, fehlergetriebene Forschung). Systemweit ergab der Union-Bound **bis zu 0,15**.

**Jetzt:** ein append-only Ledger mit Hash-Kette für alle neuen Registrierungen der drei Teilsysteme.
- **Altlast:** wird bei der Einrichtung aus den alten Registern gelesen und eingefroren. Echter Stand: 6 Options-Varianten zu je 0,0025 plus ERR-d6695229 mit 0,00208, zusammen 0,0171. Daraus folgt W0 = 0,0329.
- **Klassen:**
  - Alpha-Hypothesen (Fehlerforschung, Alpha-to-Options): 70 % von W0;
  - Modellvarianten: 30 %;
  - je höchstens 60 Registrierungen.
- **Online-Bonferroni mit fester Folge:** Der k-te Test einer Klasse erhält α_k = Anteil · W0 · k^−1,5 / ζ(1,5). Die Folge hängt nur von der Reihenfolge ab. Die Garantie gilt bei beliebiger Abhängigkeit: Altlast + Σα ≤ 0,05 über alle je möglichen Zuteilungen.
- **Fail closed:**
  - verletzte Hash-Kette → keine Zuteilung;
  - unbekanntes Teilsystem → keine Zuteilung;
  - Registrierungsgrenze → keine Zuteilung;
  - Slot in einem alten Register ohne Zuteilung → Befund (Umgehung).
- **Keine Rückgabe von Budget nach Entdeckungen:** Für asynchrone, abhängige Gruppen-Sequenztests ist sie ohne eigene Closed-Testing-Konstruktion nicht nachweislich gültig, deshalb bewusst weggelassen.

**Ergebnis mit echtem Stand:**

| k | Alpha-Hypothese | Modellvariante | bisheriger Slot |
|---|---|---|---|
| 1 | 0,0088 | 0,0038 | 0,0021 / 0,0025 |
| 2 | 0,0031 | 0,0013 | |
| 3 | 0,0017 | 0,0007 | |
| 10 | 0,0003 | 0,0001 | |

Die ersten beiden Alpha-Hypothesen erhalten mehr als bisher, spätere weniger. Das ist der ehrliche Preis dafür, dass systemweit jetzt höchstens 0,05 statt bis zu 0,15 gilt. Zusammen mit dem Power-Gate geht das Budget an entscheidbare Tests.

**Noch nicht im Ledger** (eigene Regeln, offen): ML-Promotion (`research_protocol.yaml`), Factory- und Promotion-Controller-Forward-Gate, Faktor-Forward.

## Tests

- `tests/test_alpha_ledger.py` (10):
  - Konfiguration gepinnt, Invarianten;
  - Altlast eingefroren, Union-Bound über alle möglichen Zuteilungen;
  - idempotent, geordnet, fail closed;
  - Manipulation und Umgehung sind kritisch;
  - gemeinsames Ledger der Teilsysteme;
  - echter Stand;
  - Design-Power (Testgröße unter H0, Monotonie, MDE);
  - v3-Registrierung nur für entscheidbare Hypothesen, wiederholter Lauf ohne zweite Zuteilung;
  - unterpowerte Hypothesen verbrauchen nichts;
  - ERR-d6695229 unverändert.
- Bestehende Tests der Slot-Mechanik prüfen sie weiter im Legacy-Pfad, denn sie gilt für die Altlast.
