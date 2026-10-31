"""CLI entry point for the experiment journal and fixed research pipeline."""
import argparse
import json
import sys
from pathlib import Path

from . import registry as reg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment registry and research cycle")
    ap.add_argument("command", choices=["run", "ingest", "explore", "option-forward", "factor-forward", "option-link",
                                        "option-link-correct", "reflect", "status", "leaderboard", "verify"])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--panel-cache", type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--limit", type=int, default=15)
    # option-link-correct (human use only): versioned, never deletes or re-labels
    ap.add_argument("--kind", choices=["signals", "decisions", "outcomes"])
    ap.add_argument("--target")
    ap.add_argument("--action", choices=["EXCLUDE_DATA_ERROR", "DOWNGRADE_EVIDENCE"])
    ap.add_argument("--reason")
    ap.add_argument("--by")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "option-forward":
            from .forward_options import evaluate
            result = evaluate(root)
            code = 0
        elif args.command == "factor-forward":
            from .factor_forward import process
            if args.panel_cache is None:
                raise ValueError("--panel-cache from the current trusted CI job is required")
            result = process(root, cache=args.panel_cache)
            code = 0
        elif args.command == "option-link":
            from .option_link import run as link_run
            result = link_run(root, cache=args.panel_cache if args.panel_cache and args.panel_cache.is_file() else None)
            # Phase B: research-only economics on the same journals (no network, no orders)
            from .option_economics import evaluate as economics
            eco = economics(root)
            result["economics"] = {"status": eco["status"], "evidence_classes": eco["evidence_classes"],
                                   "proposals": {h: p["structure"] for h, p in eco["proposals"].items()}}
            code = 0
        elif args.command == "option-link-correct":
            from .option_link import record_correction
            result = record_correction(root, kind=args.kind, target_id=args.target, action=args.action,
                                       reason=args.reason, recorded_by=args.by)
            code = 0
        elif args.command == "reflect":
            from .reflection import reflect
            result = reflect(root)
            code = 0
        elif args.command == "explore":
            from .explorer import register
            result = register(root)
            code = 0
        elif args.command == "run":
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
