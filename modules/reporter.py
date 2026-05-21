"""
Reporter: Speichert tägliche Trade-Vorschläge als JSON + Markdown
- outputs/daily_reports/YYYY-MM-DD.json
- outputs/daily_reports/YYYY-MM-DD.md (lesbar)

v8.2: Exit-Regeln in jedem Trade-Vorschlag:
    - Take-Profit:  +50% Options-Wert → Position (oder Hälfte) schliessen
    - Stop-Loss:    -40% Options-Wert → Gesamte Position liquidieren
    - Time-Exit:    50% DTE verstrichen und < +20% → schliessen (Theta-Decay)
    - Profit-Dollar: Konkrete $-Beträge basierend auf Entry-Preis
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def compute_exit_rules(proposal: dict) -> dict:
    option    = proposal.get("option", {})
    strategy  = proposal.get("strategy", "")
    is_spread = "SPREAD" in strategy

    if is_spread and option.get("net_debit"):
        entry = float(option.get("net_debit", 0))
    else:
        entry = float(option.get("ask", 0))

    if entry <= 0:
        return _empty_exit_rules()

    dte = int(option.get("dte", 0) or 0)

    if is_spread:
        sl           = option.get("spread_leg") or {}
        long_k       = float(option.get("strike", 0))
        short_k      = float(sl.get("strike", 0))
        width        = round(short_k - long_k, 2) if short_k > long_k else 0
        max_profit_s = round(width - entry, 2) if width > 0 else 0

        take_profit_pct   = 70
        full_profit_pct   = 90
        take_profit_price = round(entry + max_profit_s * 0.70, 2) if max_profit_s > 0 else round(entry * 1.50, 2)
        full_profit_price = round(entry + max_profit_s * 0.90, 2) if max_profit_s > 0 else round(entry * 2.00, 2)
        stop_loss_pct     = -50
        stop_loss_price   = round(entry * 0.50, 2)
        time_exit_dte     = max(int(dte * 0.45), 1)   # ~55% DTE abgelaufen
        vol_crush_pts     = 40
        delta_exit        = None
        underlying_be     = round(long_k + entry, 2) if long_k > 0 else 0
    else:
        take_profit_pct   = 50
        full_profit_pct   = 100
        take_profit_price = round(entry * 1.50, 2)
        full_profit_price = round(entry * 2.00, 2)
        stop_loss_pct     = -45
        stop_loss_price   = round(entry * 0.55, 2)
        time_exit_dte     = max(dte // 2, 1)           # 50% DTE abgelaufen
        vol_crush_pts     = 30
        delta_exit        = 0.80
        underlying_be     = 0

    try:
        expiry_str = option.get("expiry", "")
        if expiry_str:
            expiry_dt      = datetime.strptime(expiry_str, "%Y-%m-%d")
            time_exit_date = (expiry_dt - timedelta(days=time_exit_dte)).strftime("%Y-%m-%d")
        else:
            time_exit_date = "N/A"
    except Exception:
        time_exit_date = "N/A"

    return {
        "entry_cost":               entry,
        "is_spread":                is_spread,
        "take_profit_pct":          take_profit_pct,
        "take_profit_price":        take_profit_price,
        "full_profit_pct":          full_profit_pct,
        "full_profit_price":        full_profit_price,
        "stop_loss_pct":            stop_loss_pct,
        "stop_loss_price":          stop_loss_price,
        "time_exit_dte_remaining":  time_exit_dte,
        "time_exit_date":           time_exit_date,
        "time_exit_min_profit_pct": 20,
        "vol_crush_pts":            vol_crush_pts,
        "delta_exit":               delta_exit,
        "underlying_breakeven":     underlying_be,
    }


def _empty_exit_rules() -> dict:
    return {
        "entry_cost": 0, "is_spread": False,
        "take_profit_pct": 50, "take_profit_price": 0,
        "full_profit_pct": 100, "full_profit_price": 0,
        "stop_loss_pct": -45, "stop_loss_price": 0,
        "time_exit_dte_remaining": 0, "time_exit_date": "N/A",
        "time_exit_min_profit_pct": 20,
        "vol_crush_pts": 30, "delta_exit": 0.80, "underlying_breakeven": 0,
    }


class Reporter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, today: str, proposals: list[dict], history: dict) -> None:
        # v8.2: Exit-Regeln zu jedem Proposal hinzufügen
        for p in proposals:
            if "exit_rules" not in p:
                p["exit_rules"] = compute_exit_rules(p)

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
                ts     = p.get("trade_score", {})

                lines += [
                    f"---",
                    f"## {i}. {p['ticker']} – {p.get('strategy', '')}",
                    "",
                    f"**Richtung:** {p.get('direction', '')}  ",
                    f"**FinalScore:** `{p.get('final_score', 0):.4f}`  ",
                    f"**IV-Rank:** {p.get('iv_rank', 'N/A')}  ",
                    f"**Trade-Score:** {ts.get('total', 'N/A')}/100 — {ts.get('grade', '')}  ",
                    "",
                    "### Asymmetry-Analyse",
                    f"- **Impact:** {feat.get('impact', 'N/A')}/10",
                    f"- **Surprise:** {feat.get('surprise', 'N/A')}/10",
                    f"- **Mismatch-Score:** {feat.get('mismatch', 'N/A')}",
                    f"- **Z-Score (48h):** {feat.get('z_score', 'N/A')}",
                    f"- **48h-Move:** {feat.get('price_move_48h', 'N/A')}",
                    f"- **EPS-Drift:** {feat.get('eps_drift', 'N/A')} ({feat.get('bin_eps_drift', '')})",
                    "",
                    f"**Asymmetry-Reasoning:**  ",
                    f"> {da.get('asymmetry_reasoning', da.get('mispricing_logic', 'N/A'))}",
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
                    f"- **σ:** {sim.get('sigma', sim.get('sigma_adj', 0)):.4f}",
                    f"- **α (Signal-Drift):** {sim.get('alpha', 0):.5f}",
                    "",
                ]

                if option:
                    _is_spread   = p.get("strategy") == "BULL_CALL_SPREAD"
                    _action_line = (
                        f"- **Buy to Open:** ${option.get('strike', 0):.2f} CALL @ ask ${option.get('ask', 0):.2f}"
                        if not _is_spread else
                        f"- **Strike (Long Leg):** ${option.get('strike', 0):.2f} CALL"
                    )
                    lines += [
                        "### Options-Vorschlag",
                        f"- **Expiry:** {option.get('expiry', 'N/A')} ({option.get('dte', 'N/A')} DTE)",
                        _action_line,
                        f"- **Bid/Ask:** ${option.get('bid', 0):.2f} / ${option.get('ask', 0):.2f}",
                        f"- **IV %:** {option.get('implied_vol', 0):.1%}",
                        f"- **Open Interest:** {option.get('open_interest', 0):,}",
                        f"- **Bid-Ask-Ratio:** {option.get('spread_ratio', 0):.2%}",
                        "",
                    ]
                    if p.get("strategy") == "BULL_CALL_SPREAD" and option.get("spread_leg"):
                        sl         = option["spread_leg"]
                        long_k     = float(option.get("strike", 0))
                        short_k    = float(sl.get("strike", 0))
                        net_debit  = float(option.get("net_debit", 0))
                        width      = round(short_k - long_k, 2)
                        max_profit = round((width - net_debit) * 100, 2)
                        max_loss   = round(net_debit * 100, 2)
                        bep_price  = round(long_k + net_debit, 2)
                        max_roi    = round((width - net_debit) / net_debit, 4) if net_debit > 0 else 0
                        lines += [
                            f"- **Buy to Open:**  ${long_k:.2f} CALL @ ask ${option.get('ask', 0):.2f}",
                            f"- **Sell to Open:** ${short_k:.2f} CALL @ bid ${sl.get('bid', 0):.2f}",
                            f"- **Net Debit:** ${net_debit:.2f} pro Kontrakt (Max Verlust: ${max_loss:.0f})",
                            f"- **Spread-Breite:** ${width:.2f}",
                            f"- **Break-even:** ${bep_price:.2f} (+{(bep_price / long_k - 1) * 100:.1f}% über Buy-Strike)",
                            f"- **Max Gewinn:** ${max_profit:.0f} pro Kontrakt ({max_roi:.1%} ROI auf Debit)",
                            "",
                        ]

                # ── Exit-Regeln (strategie-spezifisch) ──────────────────────
                exit_r = p.get("exit_rules", {})
                if exit_r and exit_r.get("entry_cost", 0) > 0:
                    _is_sp      = exit_r.get("is_spread", False)
                    _strat_lbl  = "Bull Call Spread" if _is_sp else "Long Call"
                    _tp_lbl     = f"+{exit_r['take_profit_pct']}% Max-Gewinn" if _is_sp else f"+{exit_r['take_profit_pct']}% ROI"
                    _tp2_lbl    = f"+{exit_r['full_profit_pct']}% Max-Gewinn" if _is_sp else f"+{exit_r['full_profit_pct']}% ROI"
                    _sl_action  = "→ Sell to Close Spread (Limit)" if _is_sp else "→ Buy to Close (Limit)"
                    _warn = (
                        f"- **Warnschwellen:** IV-Crush >{exit_r.get('vol_crush_pts', 40)} Pkte | "
                        f"Kurs < Break-even + 1% (${exit_r.get('underlying_breakeven', 0):.2f})"
                        if _is_sp else
                        f"- **Warnschwellen:** IV-Crush >{exit_r.get('vol_crush_pts', 30)} Pkte | "
                        f"Delta > {exit_r.get('delta_exit', 0.80):.2f} (ITM-Roll prüfen)"
                    )
                    lines += [
                        f"### 🚪 Exit-Regeln — {_strat_lbl}",
                        f"- **Entry-Kosten:** ${exit_r['entry_cost']:.2f} pro Kontrakt",
                        f"- **TP1 ({_tp_lbl}):** bei ${exit_r['take_profit_price']:.2f} → Hälfte schliessen",
                        f"- **TP2 ({_tp2_lbl}):** bei ${exit_r['full_profit_price']:.2f} → Rest schliessen",
                        f"- **Stop-Loss ({exit_r['stop_loss_pct']}%):** bei ${exit_r['stop_loss_price']:.2f} {_sl_action}",
                        f"- **Time-Exit:** ab {exit_r['time_exit_date']} "
                        f"({exit_r['time_exit_dte_remaining']} DTE Rest) "
                        f"— schliessen wenn Gewinn < +{exit_r['time_exit_min_profit_pct']}%",
                        _warn,
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
            "_Automatisch generiert durch Adaptive Asymmetry-Scanner v8.2_",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info(f"Markdown-Report gespeichert: {path}")
