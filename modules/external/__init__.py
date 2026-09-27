"""
modules/external – Point-in-Time External-Data Factory (Road/Maritime Freight,
Weather, Real Economy).

Reine Observability im Modus `external_context.mode: shadow` (Default):
externe Daten werden archiviert, PIT-validiert, als Features an Kandidaten
gehängt und im Candidate Ledger eingefroren — sie verändern KEINE
Produktionsentscheidung. Produktionswirkung nur nach Challenger-Validierung
und menschlich gemergtem PR (siehe README "External Context").
"""
