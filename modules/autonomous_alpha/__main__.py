"""CLI entry point for the experiment journal and fixed research pipeline."""
import argparse
import json
import sys
from pathlib import Path

from . import registry as reg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment registry and research cycle")
    ap.add_argument("command", choices=["run", "ingest", "status", "leaderboard", "verify"])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--panel-cache", type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "run":
            from .orchestrator import run_cycle
            result = run_cycle(root, panel_cache=args.panel_cache, timeout_seconds=args.timeout)
            code = 0 if result["success"] else 1
        elif args.command == "ingest":
            result = reg.ingest(root, strict=args.strict)
            code = 0
        else:
            events = reg.read_journal(root / reg.JOURNAL)
            if args.command == "verify":
                conn = reg.build_sqlite(events)
                n = conn.execute("SELECT count(*) FROM experiments").fetchone()[0]
                conn.close()
                result = {"journal_valid": True, "n_unique_events": len(events), "sqlite_records": n}
            elif args.command == "status":
                result = reg.summary(events)
            else:
                result = {"historical_research_only": True, "rows": reg.leaderboard(events, args.limit)}
            code = 0
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return code
    except (OSError, ValueError, KeyError) as exc:
        print(f"Experiment registry error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
