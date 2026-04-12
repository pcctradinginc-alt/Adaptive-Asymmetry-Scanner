"""
Reporter: Speichert tägliche Trade-Vorschläge als JSON + Markdown
- outputs/daily_reports/YYYY-MM-DD.json
- outputs/daily_reports/YYYY-MM-DD.md (lesbar)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class Reporter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, today: str, proposals: list[dict], history: dict) -> None:
        self._save_json(today, proposals)
        self._save_markdown(today, proposals, history)

    def _save_json(self, today: str, proposals: list[dict]) -> None:
        path = self.reports_dir / f"{today}.json"
        with open(path, "w") as f:
            json.dump(
                {"date": today, "proposals": proposals},
                f, indent=2, default=str
            )
        log.info(f"Report gespeichert: {path}")

    def _save_markdown(self, today: str, proposals: list[dict], history: dict) -> None:
        path = self.reports_dir / f"{today}.md"
        lines = [
            f"# Adaptive Asymmetry-Scanner – {today}",
            "",
            f"**Generiert:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Trade-Vorschläge:** {len(proposals)}",
            "",
        ]

        if not proposals:
            lines.append("_Kein Signal heute. Alle Gates haben blockiert._")
        else:
            for i, p in enumerate(proposals, 1):
                da     = p.get("deep_analysis", {})
                sim    = p.get("simulation", {})
                feat   = p.get("features", {})
                option = p.get("option", {})

                lines += [
                    f"---",
                    f"## {i}. {p['ticker']} – {p.get('strategy', '')}",
                    "",
                    f"**Richtung:** {p.get('direction', '')}  ",
                    f"**FinalScore:** `{p.get('final_score', 0):.4f}`  ",
                    f"**IV-Rank:** {p.get('iv_rank', 'N/A')}  ",
                    "",
                    "### Asymmetry-Analyse",
                    f"- **Impact:** {feat.get('impact', 'N/A')}/10",
                    f"- **Surprise:** {feat.get('surprise', 'N/A')}/10",
                    f"- **Mismatch-Score:** {feat.get('mismatch', 'N/A')}",
                    f"- **Z-Score (48h):** {feat.get('z_score', 'N/A')}",
                    f"- **EPS-Drift:** {feat.get('eps_drift', 'N/A')} ({feat.get('bin_eps_drift', '')})",
                    "",
                    f"**Mispricing-Logik:**  ",
                    f"> {da.get('mispricing_logic', 'N/A')}",
                    "",
                    f"**Katalysator:** {da.get('catalyst', 'N/A')}  ",
                    f"**Time-to-Materialization:** {da.get('time_to_materialization', 'N/A')}  ",
                    "",
                    "### Bear Case",
                    f"> {da.get('bear_case', 'N/A')}  ",
                    f"**Severity:** {da.get('bear_case_severity', 'N/A')}/10",
                    "",
                    "### Monte-Carlo Simulation",
                    f"- **Hit-Rate:** {sim.get('hit_rate', 0):.1%} ({sim.get('n_paths', 0):,} Pfade)",
                    f"- **Target-Preis:** ${sim.get('target_price', 0):.2f}",
                    f"- **Aktueller Preis:** ${sim.get('current_price', 0):.2f}",
                    f"- **σ (adj.):** {sim.get('sigma_adj', 0):.4f}",
                    "",
                ]

                if option:
                    lines += [
                        "### Options-Vorschlag",
                        f"- **Expiry:** {option.get('expiry', 'N/A')} ({option.get('dte', 'N/A')} DTE)",
                        f"- **Strike:** ${option.get('strike', 0):.2f}",
                        f"- **Bid/Ask:** ${option.get('bid', 0):.2f} / ${option.get('ask', 0):.2f}",
                        f"- **Impl. Vol.:** {option.get('implied_vol', 0):.1%}",
                        f"- **Open Interest:** {option.get('open_interest', 0):,}",
                        f"- **Bid-Ask-Ratio:** {option.get('spread_ratio', 0):.2%}",
                        "",
                    ]
                    if p.get("strategy") == "BULL_CALL_SPREAD" and option.get("spread_leg"):
                        sl = option["spread_leg"]
                        lines += [
                            f"- **Short Strike:** ${sl.get('strike', 0):.2f}  ",
                            f"- **Short Bid/Ask:** ${sl.get('bid', 0):.2f} / ${sl.get('ask', 0):.2f}",
                            "",
                        ]

        # Modell-Gewichte
        weights = history.get("model_weights", {})
        lines += [
            "---",
            "## Modell-Gewichte (aktuell)",
            f"- Impact: `{weights.get('impact', 0.35):.2f}`",
            f"- Mismatch: `{weights.get('mismatch', 0.45):.2f}`",
            f"- EPS-Drift: `{weights.get('eps_drift', 0.20):.2f}`",
            "",
            "_Automatisch generiert durch Adaptive Asymmetry-Scanner v3.5_",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info(f"Markdown-Report gespeichert: {path}")
