#!/usr/bin/env bash
# scripts/ci_push.sh "<Commit-Nachricht>" <Pfad> [<Pfad> ...]
#
# Gemeinsamer Commit/Push-Schritt aller schreibenden Workflows (Audit 2026-10-04).
# Mehrere Workflows (unterschiedliche concurrency-Gruppen) schreiben parallel auf main.
#  - Append-only Ledger (*.jsonl): Union-Merge über .gitattributes – beide Seiten bleiben erhalten.
#  - Abgeleitete, jederzeit neu berechenbare Zustandsdateien (outputs/state/*.json,
#    outputs/health/*.json): bei Konflikt gilt die Version DIESES Laufs (wird ohnehin neu abgeleitet).
#  - Jeder andere Konflikt: Abbruch, Lauf ROT (fail-closed, nie still eine Seite verwerfen).
#  - Push-Ablehnung (main hat sich bewegt): bis zu 5 Versuche mit Backoff.
set -uo pipefail
msg="$1"; shift
# Schutz (2026-10-15): Läufe auf anderen Branches (workflow_dispatch zum Testen, z. B. Preflight auf einem
# Feature-Branch) dürfen nie nach main pushen – sonst landete ungeprüfter Branch-Code per Rebase in main.
if [ -n "${GITHUB_REF:-}" ] && [ "${GITHUB_REF}" != "refs/heads/main" ]; then
  echo "::notice::Lauf auf ${GITHUB_REF} (nicht main) – kein Commit/Push nach main"
  exit 0
fi
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
        outputs/state/*.json|outputs/health/*.json)
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
  if git push -q "$remote" HEAD:main; then
    git update-ref refs/remotes/origin/main HEAD 2>/dev/null || true   # Push per URL aktualisiert origin/main nicht
    echo "Push ok (Versuch $attempt)"; exit 0
  fi
  echo "Push abgelehnt (Versuch $attempt) – erneuter Rebase"
  sleep $((attempt * 7))
done
echo "::error::Push nach 5 Versuchen fehlgeschlagen"
exit 1
