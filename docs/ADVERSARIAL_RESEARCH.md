# Adversarial Self-Play

Modul: `modules/research_lab.py` (`adversarial_review`). Jede getestete
Hypothese erhält ein strukturiertes Rollen-Review. Alle Aussagen werden aus
Messwerten gebildet; es gibt keine freie LLM-Argumentation.

| Rolle | Prüft (gemessen) |
|---|---|
| Researcher | Netto-Evidenz, t, Rank-IC, Asymmetrie Top-Dezil vs. Universum |
| Skeptic | Hälften, Anteil positiver Jahre, Regime mit Versagen |
| Statistician | Stichprobe (Kohorten/Monate), p-Wert, BH über alle Tests, Locked einmalig |
| Leakage Auditor | nur PIT-Merkmale (Whitelist der Signal-Sprache), Test-Labels vor Locked, Survivorship-Hinweis; Stacking-Leakage prüft `meta_learning.leakage_checks` |
| Regime Agent | Regime-Urteile (VIX, Trend) |
| Execution Agent | Stresskosten 25 bp, Max-Drawdown; Liquidität/Turnover im HC-Scanner und in der Decision Intelligence |
| Failure Agent | schlechteste Phase (Max-DD), Anteil positiver Jahre; Fehlercluster im Unknown-Unknown-Detektor |

**Keine Rolle trifft die Produktionsentscheidung.** Die Entscheidung folgt
allein der vorab definierten Prüfkette bzw. dem Promotion-Gate.

**Messbarer Nutzen:** Das Review ändert keine Entscheidung, weil das Gate
entscheidet. Die Ablation „ohne Self-Play“ ist deshalb per Konstruktion
identisch. Wert: Transparenz und Fehlersuche. **MODIFY:** nur Dokumentation,
keine Score-Wirkung.
