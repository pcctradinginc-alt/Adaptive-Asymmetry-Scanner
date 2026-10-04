#!/usr/bin/env bash
# scripts/ci_stage_check.sh – Stufen-Bilanz am Ende eines Workflows (Audit 2026-10-04).
# Stufen mit continue-on-error lassen den Lauf weiterlaufen (Commit der übrigen Ergebnisse),
# melden sonst aber "grün", obwohl eine zentrale Stufe ausgefallen ist. Dieser Schritt läuft mit
# if: always() NACH dem Commit:
#   CRITICAL="name=outcome ..."  -> failure => Lauf ROT (::error::)
#   DEGRADED="name=outcome ..."  -> failure => ::warning:: (Lauf bleibt grün, Ausfall sichtbar)
set -uo pipefail
rc=0
for kv in ${CRITICAL:-}; do
  name="${kv%%=*}"; outcome="${kv#*=}"
  if [ "$outcome" = "failure" ]; then echo "::error::Zentrale Stufe '$name' ausgefallen"; rc=1; fi
done
for kv in ${DEGRADED:-}; do
  name="${kv%%=*}"; outcome="${kv#*=}"
  if [ "$outcome" = "failure" ]; then echo "::warning::Stufe '$name' ausgefallen (DEGRADED, nicht zentral)"; fi
done
[ "$rc" = 0 ] && echo "Stufen-Bilanz: alle zentralen Stufen ok"
exit $rc
