#!/usr/bin/env python3
"""
scripts/build_faf_exposure.py – manuelles/einmaliges Build-Skript für
config/faf_exposure.yaml (FHWA Freight Analysis Framework Exposure-Map).

WICHTIG: Dieses Skript wird NICHT im laufenden Scanner-Pipeline-Betrieb
aufgerufen. Die FAF-Rohdaten (FAF5 Regionaldatenbank) sind mehrere
Gigabyte groß; ein automatischer Download in jedem Scanner-Run wäre falsch.
Stattdessen: ein Operator lädt die offiziellen FAF5-CSV/DB-Extrakte manuell
von

    https://ops.fhwa.dot.gov/freight/freight_analysis/faf/faf5/

herunter, prüft Lizenz/Aktualität und lässt dieses Skript LOKAL laufen, um
eine kompakte Commodity(SCTG) -> Region -> Sektor-Tabelle abzuleiten, die
danach von Hand kuratiert wird (GICS-Sektor-Zuordnung ist eine analytische
Entscheidung, kein reiner Datenabruf).

Nutzung (Platzhalter-Gerüst):

    python3 scripts/build_faf_exposure.py \
        --faf-csv /pfad/zu/FAF5_regional_flows.csv \
        --out config/faf_exposure.yaml

Status: NICHT implementiert (kein FAF5-Extrakt in dieser Sandbox verfügbar,
kein Netzzugriff). Dieses Gerüst dokumentiert die erwartete Struktur und
Aufrufkonvention, damit ein Operator die eigentliche Ableitung später
nachziehen kann. Bis dahin bleibt config/faf_exposure.yaml ein klar
markierter Platzhalter ("manual/derived, not yet built") und die Registry
(config/external_sources/road_freight.yaml, source_id fhwa_faf) trägt
status_override: DEFERRED.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faf-csv", required=False,
                         help="Pfad zu einem lokal heruntergeladenen FAF5-Regionalflow-Extrakt (CSV).")
    parser.add_argument("--out", default="config/faf_exposure.yaml",
                         help="Zieldatei für die abgeleitete Exposure-Tabelle.")
    args = parser.parse_args()

    if not args.faf_csv:
        print(
            "Kein --faf-csv angegeben. Dieses Skript lädt bewusst KEINE FAF-Daten "
            "automatisch herunter (zu groß für Routine-Runs; Lizenz/Region-Codes "
            "müssen vor Verwendung geprüft werden).\n\n"
            "Vorgehen:\n"
            "  1. FAF5-Extrakt manuell laden: "
            "https://ops.fhwa.dot.gov/freight/freight_analysis/faf/faf5/\n"
            "  2. python3 scripts/build_faf_exposure.py --faf-csv <pfad> "
            "--out config/faf_exposure.yaml\n"
            "  3. Ergebnis von Hand kuratieren (GICS-Sektor-Mapping prüfen).\n"
            "  4. Registry-Eintrag fhwa_faf.status_override erst nach Review "
            "von DEFERRED entfernen.",
            file=sys.stderr,
        )
        return 1

    # Absichtlich nicht implementiert: die eigentliche CSV->Exposure-Ableitung
    # hängt vom konkreten FAF5-Extraktschema ab (Spalten je Release-Version
    # leicht unterschiedlich) und muss gegen den tatsächlichen Download
    # verifiziert werden, nicht gegen Annahmen aus dieser Sandbox.
    print(
        "FAF5-Parsing ist in dieser Umgebung nicht implementiert (kein "
        "Netzzugriff/kein Referenzextrakt verfügbar, um das aktuelle "
        "FAF5-CSV-Schema zu verifizieren). Bitte manuell gegen den "
        "tatsächlichen Extrakt vervollständigen.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
