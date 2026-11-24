#!/usr/bin/env bash
# scripts/ci_push.sh "<Commit-Nachricht>" <Pfad> [<Pfad> ...]
#
# Gemeinsamer Commit/Push-Schritt aller schreibenden Workflows (Audit 2026-10-04).
# Mehrere Workflows (unterschiedliche concurrency-Gruppen) schreiben parallel auf main.
#  - Append-only Ledger (*.jsonl): Union-Merge über .gitattributes – beide Seiten bleiben erhalten.
#  - Abgeleitete, jederzeit neu berechenbare Zustandsdateien (outputs/state/*.json,
#    outputs/health/*.json, outputs/research/improvement_tasks.json – abgeleitet aus dem append-only
#    improvement_tasks.jsonl; outputs/research/autonomous_alpha/option_link.json – von option_link.coverage()
#    aus den append-only Journalen signals/decisions/outcomes neu berechnet, Schreiber: options_capture.yml
#    UND ml_research.yml in verschiedenen concurrency-Gruppen): bei Konflikt gilt die Version DIESES Laufs
#    (wird ohnehin neu abgeleitet) – ein Report-Konflikt darf nie den Commit des Quote-Archivs kosten.
#  - Jeder andere Konflikt: Abbruch, Lauf ROT (fail-closed, nie still eine Seite verwerfen).
#  - Push-Ablehnung (main hat sich bewegt): bis zu 5 Versuche mit Backoff.
set -uo pipefail
msg="$1"; shift
# Schutz (2026-10-15): Läufe auf anderen Branches (workflow_dispatch zum Testen, z. B. Preflight auf einem
# Feature-Branch) dürfen nie nach main pushen – sonst landete ungeprüfter Branch-Code per Rebase in main.
if [ "${GITHUB_REF:-}" != "refs/heads/main" ]; then
  echo "::notice::Lauf auf ${GITHUB_REF:-unset} (nicht main) – kein Commit/Push nach main"
  exit 0
fi
# Defense in depth: this bot is permitted to persist observational outputs and
# the three explicitly inventoried generated research/config state files only.
# Never accept arbitrary repo code/workflows from CI or path traversal.
# A Ruleset with review requirements is still needed to protect main itself.
ci_allowed_path() {
  local path="$1"
  case "$path" in
    ""|/*|../*|*/../*|*/..|*//*|./*|*/./*|*/.) return 1 ;;
  esac
  case "$path" in
    outputs|outputs/*|challengers.yaml|config/faf_exposure.yaml|config/challenger_proposals_auto.yaml)
      return 0 ;;
  esac
  return 1
}
if [ "$#" -eq 0 ]; then
  echo "::error::ci_push requires at least one allowlisted data path"
  exit 2
fi
for path in "$@"; do
  if ! ci_allowed_path "$path"; then
    echo "::error::ci_push refuses non-data or traversal path: $path"
    exit 2
  fi
done
git config user.name  "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
for p in "$@"; do
  if [ -e "$p" ]; then git add -- "$p"; fi
done
if git diff --cached --quiet; then
  # Nichts Neues – aber ein früherer Schritt kann lokal committet und nicht gepusht haben (Push gescheitert).
  # Dann trotzdem pushen, statt die Daten mit dem kurzlebigen Runner zu verlieren.
  if [ -z "$(git log --oneline origin/main..HEAD 2>/dev/null)" ]; then echo "Keine Änderungen"; exit 0; fi
  echo "Keine neuen Änderungen – pushe ausstehende lokale Commits"
else
  git commit -q -m "$msg"
fi
git stash --include-untracked -q >/dev/null 2>&1 || true      # nicht committete Nebenprodukte
remote="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

resolve_rebase() {
  local guard=0
  while [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; do
    guard=$((guard + 1))
    if [ "$guard" -gt 20 ]; then echo "::error::Rebase-Auflösung hängt"; git rebase --abort; return 1; fi
    local conflicted
    conflicted=$(git diff --name-only --diff-filter=U)
    for f in $conflicted; do
      case "$f" in
        outputs/research/paper_account/account.json)
          # Preserve the newer generated report; immutable bookings remain in the archive.
          local ours theirs choice
          ours=$(git show ":2:$f" | python -c 'import json,sys; from datetime import datetime; print(datetime.fromisoformat(json.load(sys.stdin)["generated_at"]).timestamp())') || { git rebase --abort; return 1; }
          theirs=$(git show ":3:$f" | python -c 'import json,sys; from datetime import datetime; print(datetime.fromisoformat(json.load(sys.stdin)["generated_at"]).timestamp())') || { git rebase --abort; return 1; }
          choice=ours
          if python -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$theirs" "$ours"; then choice=theirs; fi
          git checkout --"$choice" -- "$f" && git add -- "$f" || { git rebase --abort; return 1; } ;;
        outputs/state/*.json|outputs/health/*.json|outputs/research/autonomous_alpha/option_link.json|outputs/research/improvement_tasks.json)
          echo "Konflikt in abgeleiteter Datei $f -> Version dieses Laufs"
          git checkout --theirs -- "$f" && git add -- "$f" ;;
        *)
          echo "::error::Push-Konflikt in $f (nicht abgeleitet) – Abbruch, nichts wird verworfen"
          git rebase --abort; return 1 ;;
      esac
    done
    GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || true
  done
  return 0
}

for attempt in 1 2 3 4 5; do
  if ! git pull --rebase -q origin main; then
    resolve_rebase || exit 1
  fi
  # Validate the ACTUAL outgoing diff, including any pending local commits.
  # Checking requested paths alone is insufficient when a previous step
  # created an unpushed commit. NUL-safe paths avoid shell tokenization.
  # A failed diff must never look like an empty (safe) diff.
  outgoing_paths=$(mktemp) || exit 2
  if ! git diff --name-only -z origin/main HEAD >"$outgoing_paths"; then
    rm -f "$outgoing_paths"
    echo "::error::ci_push could not verify outgoing commit paths"
    exit 2
  fi
  while IFS= read -r -d '' changed; do
    if ! ci_allowed_path "$changed"; then
      rm -f "$outgoing_paths"
      echo "::error::ci_push refuses non-data commit diff: $changed"
      exit 2
    fi
  done < "$outgoing_paths"
  rm -f "$outgoing_paths"
  if git push -q "$remote" HEAD:main; then
    git update-ref refs/remotes/origin/main HEAD 2>/dev/null || true   # Push per URL aktualisiert origin/main nicht
    echo "Push ok (Versuch $attempt)"; exit 0
  fi
  echo "Push abgelehnt (Versuch $attempt) – erneuter Rebase"
  sleep $((attempt * 7))
done
echo "::error::Push nach 5 Versuchen fehlgeschlagen"
exit 1
