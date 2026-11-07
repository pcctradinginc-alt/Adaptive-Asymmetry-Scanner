"""Weekly German owner digest for the GitHub-only paper account.

Combines existing research evidence with the separate account ledger.
Plain-text SMTP via already configured Gmail application credentials.
No LLM, broker, orders or paid data.
"""
from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from .paper_account import REPORT as ACCOUNT_REPORT
from .registry import canonical, digest

ROOT = Path("outputs/research/paper_account")
ECONOMICS = Path("outputs/research/autonomous_alpha/option_economics.json")
VALIDATION = Path("outputs/research/autonomous_alpha/option_validation.json")
LEARNER = Path("outputs/research/option_learner")
ALPHA_QUALITY = Path("outputs/research/autonomous_alpha/data_quality.json")
HYP_DB = Path("outputs/research/hypothesis_db.json")
FACTORY_PLAN = Path("outputs/research/factory_plan.json")


def _json(root: Path, path: Path) -> dict:
    full = root / path
    try:
        value = json.loads(full.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}


def _usd(v) -> str:
    return "nicht bewertbar" if v is None else f"{float(v):,.2f} USD"


def _pct(v) -> str:
    return "nicht bewertbar" if v is None else f"{float(v) * 100:.2f} %"


def render(account: dict, economics: dict, validation: dict,
           learner5: dict, learner20: dict, *, week: str,
           data_quality: dict | None = None, hypothesis_db: dict | None = None,
           factory_plan: dict | None = None) -> str:
    """Separate observed facts, exploratory hypotheses and confirmation."""
    lines = [f"GITHUB-PAPER-DEPOT – Wochenbericht {week}",
             "Ausschließlich virtuelle Optionsgeschäfte. Keine Brokeranbindung, keine echten Orders.", "",
             "A. Virtuelles 100.000-USD-Depot (gemessene Buchungen)",
             f"Kontostatus: {account.get('status', 'NOT_ACTIVATED')}",
             f"Startkapital: {_usd(account.get('starting_cash_usd', 100000))}",
             f"Cash: {_usd(account.get('cash_usd'))}",
             f"Gebundenes maximales Risiko: {_usd(account.get('open_risk_usd'))}",
             f"Offen / geschlossen: {account.get('open_trades', 0)} / {account.get('closed_trades', 0)}",
             f"Netto-P&L realisiert (BASIS): {_usd(account.get('net_pnl_base_usd'))}",
             f"Netto-P&L realisiert (ADVERSE): {_usd(account.get('net_pnl_adverse_usd'))}",
             f"Mark-to-Market NAV: {_usd(account.get('nav_market_usd'))}",
             f"Bewertungsstatus: {account.get('nav_market_status', 'NO_MARKS')}",
             f"Reiner Buchwert-NAV (kein Marktwert): {_usd(account.get('nav_at_cost_usd'))}",
             f"Trefferquote geschlossene Trades: {_pct(account.get('winrate_closed'))}",
             f"Profitfaktor: {account.get('profit_factor_closed') or 'noch nicht bewertbar'}",
             f"Drawdown geschlossene Buchungen (nicht Intraday): {_pct(account.get('max_drawdown_closed_book'))}",
             "Hinweis: Fehlende Quotes werden nicht durch theoretische Optionspreise ersetzt.", "",
             "B. Optionsstrategien (getrennte Forschungspopulation)"]
    counts = {k: {"n": 0, "pnl": 0.0, "periods": 0} for k in ("LONG_CALL", "BULL_CALL_SPREAD")}
    for per in (economics.get("hypotheses") or {}).values():
        for st, report in per.items():
            if st not in counts:
                continue
            m = (report.get("metrics") or {}).get("ADVERSE") or {}
            counts[st]["n"] += int(m.get("n") or 0)
            counts[st]["pnl"] += float(m.get("total_pnl_usd") or 0)
            counts[st]["periods"] = max(counts[st]["periods"], int(m.get("independent_periods") or 0))
    for st, v in counts.items():
        lines.append(f"{st}: {v['n']} verifizierte Varianten-Ergebnisse, "
                     f"{_usd(v['pnl'])} aggregiert, höchstens {v['periods']} unabhängige Fenster je Hypothese")
    lines.append("Die Varianten sind nicht automatisch unabhängig oder direkt gepaart; daraus folgt kein Sieger.")
    lines += ["", "C. Autonomes Lernen und statistischer Status",
              f"Alpha-Validierung: {validation.get('status', 'NEED_MORE_DATA')}",
              f"Alpha-Ökonomie: {economics.get('status', 'NEED_MORE_DATA')}",
              "5-Tage-Lernen: " + str(learner5.get("learning_status") or "NEED_MORE_DATA")[:220],
              "20-Tage-Lernen: " + str(learner20.get("learning_status") or "NEED_MORE_DATA")[:220],
              "Diese Forschungsergebnisse verwenden bestehende getrennte Outcome-Journale; "
              "Depotbuchungen erzeugen keine doppelten Trainingslabels.",
              "Ein verbesserter Depotwert allein beweist keinen Lernfortschritt.", "",
              "D. Alpha-Forschung (Hypothesen, keine bestätigten Tatsachen)"]
    learning = validation.get("learning") or {}
    ideas = learning.get("suggested_hypotheses") or []
    if ideas:
        lines += ["Forschungsfragen (ungeprüft): " + str(idea.get("idea") or idea)[:220] for idea in ideas[:3]]
    else:
        lines.append("Keine neuen, methodisch belastbar registrierten Hypothesen aus dem Konto.")
    lines += [f"Validierungspläne: {validation.get('n_plans', 0)}",
              f"Validierungsklassen: {json.dumps(validation.get('classification_counts') or {}, ensure_ascii=False)}",
              "", "E. Systemgesundheit"]
    missing = learning.get("missing_data") or {}
    top = sorted(missing.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))[:3]
    lines.append("Wichtigste Quellen-/Quote-Lücken: " +
                 (", ".join(f"{k}: {v}" for k, v in top) if top else "keine Angaben"))
    lines.append("Verifizierte Alpha-Options-Outcomes: " +
                 str((economics.get("evidence_classes") or {}).get("QUOTE_VERIFIED_PAPER", 0)))
    lines.append("Kontostatus unmarkiert: " +
                 str(sum(m.get("status") != "OBSERVED" for m in (account.get("marks") or {}).values())))
    lines += ["", "F. Maximal drei nächste Schritte"]
    candidates = [f"Abdeckung prüfen: {k} ({v} Fälle)" for k, v in top]
    candidates += ["Prospektive, unabhängige Handelsfenster sammeln",
                   "Keine Produktionsänderung ohne Owner-Review und statistische Bestätigung"]
    lines += ["- " + suggestion for suggestion in candidates[:3]]
    lines += ["", "FAKT: nur archivierte und überprüfte Quote-/Buchungsdaten.",
              "HYPOTHESE: Forschungsansatz ohne bestätigte Überlegenheit.",
              "BESTÄTIGT: nur mit vorregistriertem, unabhängigen Forward-Test und Fehlerbudget.",
              "Keine Trading-Empfehlung; kein automatischer Produktionswechsel."]
    quality, db, factory = data_quality or {}, hypothesis_db or {}, factory_plan or {}
    verdicts = [p.get("status") for p in (validation.get("plans") or [])
                if isinstance(p, dict)]
    h5 = (quality.get("existing_option_learner") or {}).get("5") or {}
    h20 = (quality.get("existing_option_learner") or {}).get("20") or {}
    gaps = (quality.get("largest_observed_count_gaps") or [])
    lines += ["", "G. Autonomous Alpha Discovery 2.0 – Status der Evidenz",
              f"Neue Ideen im letzten Fabrik-Plan (nicht bestätigt): {factory.get('n_ideas', 'unbekannt')}",
              f"Historisch geprüfte Hypothesen kumulativ: {db.get('n_tested_total', 'unbekannt')}",
              f"EXPLORATORY: {((db.get('status_counts') or {}).get('EXPLORATORY', 0))} historische Kandidaten",
              f"PENDING_FORWARD: {sum(v not in ('CONFIRMED', 'FALSIFIED', 'NOT_CONFIRMED') for v in verdicts)} registrierte Optionspläne ohne endgültiges Urteil",
              f"CONFIRMED (unabhängiger Prospektivtest): {verdicts.count('CONFIRMED')}",
              f"FALSIFIED: {verdicts.count('FALSIFIED')}",
              f"NEED_MORE_DATA: {sum(int(v or 0) for v in (quality.get('blocked_hypotheses_by_structure') or {}).values())} Struktur-Hypothesen-Bewertungen",
              f"5-Tage-Quote-Belege (historische Archivstichprobe): {h5.get('verified_quote_labels', 'unbekannt')}; "
              f"archiviert {h5.get('archive_sample_verified_fraction', 'unbekannt')}",
              f"20-Tage-Quote-Belege (historische Archivstichprobe): {h20.get('verified_quote_labels', 'unbekannt')}; "
              f"archiviert {h20.get('archive_sample_verified_fraction', 'unbekannt')}",
              "Unabhängige Forward-Fenster: keine belastbare Zahl allein aus historischen Quotentagen ableitbar",
              "Netto-Alpha nach Optionskosten: erst nach verifizierter, unabhängiger Validierung entscheidbar",
              "Größtes beobachtetes Datenproblem (Anzahl, nicht Alpha-Nutzen): " +
              (", ".join(gaps) if gaps else "noch nicht bewertet"),
              "Nächste Forschungspriorität: zukünftige echte Entry-/Exit-Quotes und "
              "unabhängige Forward-Fenster sammeln, nur danach neue Hypothesen wirtschaftlich validieren",
              "Status-Definition: EXPLORATORY (historisch), PENDING_FORWARD (vorgemerkt), "
              "CONFIRMED/FALSIFIED (prospektiv), NEED_MORE_DATA (noch nicht bewertbar).",
              "Historische ACCEPTED-/ROBUST-Marker aus dem kontaminierten Locked-Holdout sind KEIN Alpha-Beweis."]
    return "\n".join(lines) + "\n"


def generate(root: Path = Path("."), *, now: datetime | None = None) -> tuple[str, str]:
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    week = f"{now.isocalendar().year}-W{now.isocalendar().week:02}"
    content = render(_json(root, ACCOUNT_REPORT), _json(root, ECONOMICS),
                     _json(root, VALIDATION),
                     _json(root, LEARNER / "status_5d.json"),
                     _json(root, LEARNER / "status_20d.json"), week=week,
                     data_quality=_json(root, ALPHA_QUALITY),
                     hypothesis_db=_json(root, HYP_DB),
                     factory_plan=_json(root, FACTORY_PLAN))
    return week, content


def send(root: Path = Path("."), *, now: datetime | None = None) -> dict:
    """SMTP at-least-once; a committed SENT receipt deduplicates later jobs.

    A provider success followed by Git push failure can resend once. The stable
    Message-ID and stored week receipt make this diagnosable, not 'exactly once'.
    """
    root = Path(root)
    week, body = generate(root, now=now)
    dest = root / ROOT / "mail" / (week + ".json")
    if dest.exists():
        record = json.loads(dest.read_text(encoding="utf-8"))
        payload = record.get("payload") or {}
        if record.get("sha256") != digest(payload) or payload.get("status") != "SENT":
            raise ValueError("corrupt paper-email delivery receipt")
        return {"status": "ALREADY_SENT", "week": week}
    sender = os.getenv("GMAIL_SENDER", "").strip()
    password = os.getenv("GMAIL_APP_PW", "").strip()
    recipient = os.getenv("NOTIFY_EMAIL", "").strip()
    if not sender or not password or not recipient:
        raise ValueError("missing configured Gmail sender, app password or recipient")
    msg = EmailMessage()
    msg["From"], msg["To"] = sender, recipient
    msg["Subject"] = f"GitHub Paper-Trading – Wochenbericht {week}"
    msg["Message-ID"] = f"<github-paper-account-{week}@weekly.local>"
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    payload = {"schema": "github-paper-mail-v1", "week": week,
               "status": "SENT", "body_sha256": digest(body), "recipient": recipient,
               "message_id": str(msg["Message-ID"]),
               "sent_at": (now or datetime.now(timezone.utc)).isoformat()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    wrapped = canonical({"sha256": digest(payload), "payload": payload}) + "\n"
    with dest.open("x", encoding="utf-8") as f:
        f.write(wrapped)
        f.flush()
        os.fsync(f.fileno())
    return {"status": "SENT", "week": week, "message_id": payload["message_id"]}
