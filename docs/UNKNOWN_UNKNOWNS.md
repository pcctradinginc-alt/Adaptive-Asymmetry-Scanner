# Unknown-Unknown-Detektor

Modul: `modules/blind_spots.py`. Protokoll: `config/next_protocol.yaml`
(`blind_spots`).

## Methode
1. **Große Fehlprognosen:** Top-Dezil-Positionen des Champions, deren
   Netto-Relativrendite im schlechtesten Dezil liegt.
2. **Eigenschaften:**
   - Sektor, Volatilität, Liquidität, Momentum;
   - extreme 5-Tage-Bewegung, Nähe zum 52W-Hoch;
   - Beta, Lotterie-Profil;
   - VIX-Regime und Trend.

   Geprüft werden alle einzelnen Eigenschaften und alle Paare.
3. **Lift gegen alle Top-Dezil-Positionen:** Signifikanz per Binomial-z mit
   Benjamini-Hochberg (q = 0,10). Ein Cluster erfordert Lift ≥ 1,5, n ≥ 30 und
   Signifikanz.
4. **Ausgabe je Cluster** `UNKNOWN_CLUSTER_###`:
   - N, typischer Fehler, gemeinsame Eigenschaften;
   - Abdeckung, d.h. ob ein Modell-Failure-Profil das Segment bereits kennt;
   - Empfehlung.
5. **Rückfluss:** Die Cluster gehen in den Research Director (Hypothese
   „Segment meiden“) und in den High-Confidence-Scanner (Unknown-Risk führt zur
   Ablehnung).

## Nutzen-Messung
Walk-Forward:
- Cluster werden nur aus Basis-OOS-Zeilen gebildet, deren Label **vor** dem
  Testjahr endet.
- Im Folgejahr werden Positionen in diesen Segmenten gemieden, und die
  Expectancy wird mit dem Champion verglichen (Bootstrap).
- KEEP nur, wenn die Untergrenze > 0 ist **und** die Trefferquote nicht sinkt.

Ergebnis: `docs/NEXT_INTELLIGENCE_VALIDATION.md`.

**Ergebnis (Lauf 2026-09-29): REJECT als Filter, KEEP als Diagnose.**
- Je Fold 15 Cluster, z.B. Energie + 12M-Verlierer oder Lotterie-Profil.
- Der Filter schließt 38–87 % des Top-Dezils aus.
- Expectancy: 0,27 % → 0,06 % je Position (Monats-Δ CI [−0,84 %, +0,61 %]).
  Locked: 1,9 % → 0,5 %.
- Der MaxDD verbessert sich (−18,8 % → −10,2 %), aber nur durch weniger
  Exposure.
- Die Cluster speisen weiter den Research Director: Daraus entstanden 5
  „Segment meiden“-Hypothesen, die das Lab einzeln mit BH prüft. Im HC-Scanner
  dienen sie als Unknown-Risk-Prüfung.
