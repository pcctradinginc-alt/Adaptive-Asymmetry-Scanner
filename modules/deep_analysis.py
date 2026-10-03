"""
modules/deep_analysis.py v9.0

Änderungen v9.0:
    #10 catalyst_confidence (0-10) als neues JSON-Feld.
        Separater Score für "Wie sicher ist der Catalyst einzutreten?"
        Unabhängig vom Gesamt-Impact — ein Catalyst kann materialreich sein
        (Impact=8) aber unsicher (confidence=3). Wird in RL-Observation
        und Email ausgegeben.

    #12 Mega-Cap-Filter im System-Prompt.
        Bei Marktkapitalisierung > $200 Mrd. expliziter Hinweis auf
        Informationseffizienz. Impact > 6 bei Mega-Caps erfordert
        besonders starke Begründung. Marktkapitalisierung wird im
        Analyse-Template angezeigt.

    #3  mc_hit_rate wird im returned result dict gespeichert,
        damit options_designer.py darauf zugreifen kann ohne
        erneuten quick_mc-Lookup.

Änderungen v8.2:
    - FIX: _get_48h_move() nutzt period='10d' und vergleicht volle Handelstage.

Änderungen v8.1:
    - Analysedatum im SYSTEM_PROMPT + ANALYSIS_TEMPLATE
    - asymmetry_reasoning: Genau 3 Sätze, max 500 Zeichen
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import yfinance as yf

from modules                import analysis_cache, cost_telemetry, model_routing
from modules.config        import cfg
from modules.macro_context import get_macro_context

log = logging.getLogger(__name__)

# v9.0 #12: Mega-Cap-Schwelle in USD
MEGA_CAP_THRESHOLD = 200_000_000_000  # $200 Mrd.

SYSTEM_PROMPT = """Du bist ein skeptischer Quant-Analyst mit Fokus auf mittelfristige Optionsstrategien (2-6 Monate). Das aktuelle Jahr ist {current_year}. Alle Jahreszahlen in deinen Analysen müssen ≥ {current_year} sein. Ignoriere historische Jahreszahlen aus deinen Trainingsdaten — orientiere dich ausschließlich am Analysedatum im Prompt.

PFLICHT-ABLAUF — in dieser Reihenfolge, keine Ausnahme:

SCHRITT 1 — RED TEAM (zuerst immer):
Finde die 3 stärksten Argumente GEGEN diesen Trade.
Denke wie ein Short-Seller. Was könnte das Signal zerstören?
Typische Red Flags: Überbewertung, Sektor-Gegenwind, fragliche Datenqualität,
Makro-Risiko, IV-Crush, Katalysator bereits eingepreist.

SCHRITT 2 — STATISTIK-CHECK:
Ist die MC Hit-Rate realistisch gegeben historischer Volatilität?
Warnung wenn Hit-Rate > 80% (Modell möglicherweise zu optimistisch).

SCHRITT 3 — MAKRO-KONTEXT:
Passt das Signal zum aktuellen Zinsumfeld?
Rezessives Umfeld (invertierte Kurve) → erhöhte Skepsis bei BULLISH-Signalen.

SCHRITT 4 — MEGA-CAP-CHECK (wenn Marktkapitalisierung > $200 Mrd.):
Mega-Cap-Warnung: Bei Aktien mit Marktkapitalisierung > $200 Mrd. ist die
Informationseffizienz extrem hoch. Große institutionelle Desk-Coverage bedeutet,
dass öffentliche Informationen innerhalb von Minuten eingepreist werden.
Strukturelle Underreactions bei Mega-Caps sind deutlich seltener als bei Mid-Caps.
Impact > 6 bei Mega-Caps erfordert einen sehr konkreten, nicht-öffentlichen Informationsvorsprung.
Im Zweifel: Impact auf 5 begrenzen wenn nur öffentliche Informationen vorliegen.

SCHRITT 5 — ERST JETZT: Finale Bewertung.
Im Zweifel BEARISH. Nur eindeutige strukturelle Signale verdienen Impact > 7.

SCORING-KALIBRIERUNG — lies diese Skala bevor du Impact und Surprise vergibst:

Impact (fundamentale Bedeutung der Information):
  1-2  → Marginales Rauschen, kein nachweisbarer Fundamentaleffekt
  3-4  → Solide News, kleine bis mittlere fundamentale Auswirkung
  5-6  → Klares strukturelles Signal, messbar für Earnings oder Marktanteil
  7-8  → Starker Katalysator, signifikante Neubewertung gerechtfertigt
  9-10 → Nur für bahnbrechende, nicht-öffentliche Informationsvorsprünge (sehr selten)

Surprise (wie unerwartet ist das Signal für den Konsensus?):
  1-2  → Vollständig erwartet, im Markt bereits eingepreist
  3-4  → Leicht überraschend, Konsensus hatte es teilweise antizipiert
  5-6  → Genuiner Überraschungseffekt, Markt hat klar unterreagiert
  7-8  → Stark unerwartet, fundamentaler Paradigmenwechsel
  9-10 → Black-Swan-Charakter (extrem selten)

WICHTIG: Impact=5, Surprise=4 als Reflex-Standardwert ist VERBOTEN.
Vergib nur Werte die du im asymmetry_reasoning KONKRET begründen kannst.
Erwartete Verteilung: ~60% der Signale bei Impact 3-5, Surprise 2-4.

Antworte ausschließlich mit validem JSON."""

ANALYSIS_TEMPLATE = """=== ANALYSEDATUM: {analysis_date} (WICHTIG: Alle Jahreszahlen müssen ≥ {analysis_year} sein) ===

=== MAKRO-KONTEXT ===
{macro_context}

=== TICKER: {ticker} ===
Aktueller Preis: ${current_price:.2f}
Marktkapitalisierung: {market_cap_str}{mega_cap_flag}
Sektor: {sector}
Haiku-Prescreening: {prescreen_reason} [Kategorie: {prescreen_category}]
WICHTIG: Wenn deine Bewertung der Direction von der Haiku-Einschätzung abweicht,
erkläre explizit warum im asymmetry_reasoning.
EPS (yfinance): {forward_eps} | EPS (SEC EDGAR): {sec_eps}
EPS-Abweichung: {eps_deviation}
48h-Preisbewegung: {move_48h}

=== QUICK MONTE CARLO (Vorfilter) ===
Hit-Rate: {mc_hit_rate:.1%} ({mc_paths} Pfade, {mc_days}d)
Interpretation: {mc_interpretation}

=== NEWS (letzte 48h) ===
{news_text}

{data_anomaly_warning}

=== DEINE AUFGABE ===
Folge dem Pflicht-Ablauf: Red Team → Statistik → Makro → {mega_cap_step}→ Finale Bewertung.

Antworte NUR mit diesem JSON:
{{
    "red_team": {{
        "argument_1": "<Stärkstes Argument gegen den Trade — Min 2 vollständige Sätze, mindestens 200 Zeichen, konkret>",
        "argument_2": "<Zweitstärkstes Argument>",
        "argument_3": "<Drittstärkstes Argument>",
        "red_team_verdict": "VETO" oder "PASSIERT"
    }},
    "stats_check": {{
        "mc_assessment": "<Ist {mc_hit_rate:.0%} realistisch?>",
        "concern_level": "low" oder "medium" oder "high"
    }},
    "impact": <0-10 | 1-2=Rauschen, 3-4=solide, 5-6=klar strukturell, 7-8=stark, 9-10=selten>,
    "surprise": <0-10 | 1-2=erwartet, 3-4=leicht überraschend, 5-6=echte Überraschung, 7+=Black-Swan>,
    "direction": "BULLISH" oder "BEARISH",
    "bear_case_severity": <0-10>,
    "time_to_materialization": "4-8 Wochen" oder "2-3 Monate" oder "6 Monate",
    "catalyst_confidence": <0-10>,
    "asymmetry_reasoning": "<Genau 3 vollständige Sätze — warum der Markt unterreagiert hat. Maximal 500 Zeichen. Kein Satz darf abgebrochen werden>",
    "catalyst": "<Spezifischer Katalysator>",
    "bear_case": "<Stärkstes Gegenargument>",
    "macro_assessment": "<Bewertung im aktuellen Makro-Umfeld>",
    "data_confidence": "high" oder "medium" oder "low"
}}"""


def _format_market_cap(market_cap: Optional[int]) -> str:
    if not market_cap or market_cap <= 0:
        return "Unbekannt"
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap/1_000_000_000_000:.1f}B (Trillion)"
    if market_cap >= 1_000_000_000:
        return f"${market_cap/1_000_000_000:.1f} Mrd."
    return f"${market_cap/1_000_000:.0f} Mio."


# Prompt-Version für den Analyse-Cache: jede Änderung an den Vorlagen invalidiert alle Einträge.
PROMPT_VERSION = analysis_cache.prompt_version(SYSTEM_PROMPT, ANALYSIS_TEMPLATE)


class DeepAnalysis:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._macro = get_macro_context()
        if self._macro["data_available"]:
            log.info(
                f"Makro: {self._macro['macro_regime']} | "
                f"YC={self._macro.get('yield_curve_desc', 'n/a')}"
            )

    def run(self, shortlist: list[dict], deadline: float | None = None) -> list[dict]:
        """deadline: time.monotonic()-Wert, ab dem keine weitere Analyse mehr
        beginnt (Laufzeitbudget der Pipeline); übersprungene -> skipped_for_time."""
        import time as _time
        analyses = []
        self.skipped_for_time = []
        routing = self._routing() if getattr(self, "client", None) is not None else {}
        pre = self._prefetch_batch(shortlist, deadline, routing)
        for idx, candidate in enumerate(shortlist):
            if idx in pre and pre[idx][0] == "done":
                analysis = pre[idx][1]
            else:
                if deadline is not None and _time.monotonic() > deadline:
                    # nur echte Lücken: per Batch bereits beantwortete Kandidaten laufen weiter
                    rest = [j for j in range(idx, len(shortlist))
                            if not (j in pre and pre[j][0] == "done")]
                    self.skipped_for_time = [shortlist[j].get("ticker") for j in rest]
                    for j in rest:
                        pre[j] = ("done", None)
                    continue
                analysis = (self._analyze(candidate, pre[idx][1]) if idx in pre
                            else self._analyze(candidate))
            if candidate.get("prescreen_direction"):
                # gepaarte Messung für den Bearish-Vorfilter (Prescreen-Richtung vs. Deep Analysis)
                g = getattr(cfg, "gates", None)
                cost_telemetry.record(model_routing.prefilter_pair(
                    candidate.get("ticker"), candidate.get("prescreen_direction"), analysis,
                    impact_min=int(getattr(g, "impact_min", 4)), surprise_min=int(getattr(g, "surprise_min", 3))))
            if not analysis:
                continue

            red_team = analysis.get("red_team", {})
            arg1 = (red_team.get("argument_1", "") or "").lower()
            narrativ_mismatch = any(w in arg1 for w in [
                "narrativ-mismatch", "narrative mismatch", "trifft das geschäftsmodell",
                "falsches narrativ", "datenfehler in der vorselektion",
                "grundlegendes missverständnis", "trifft nicht zu"
            ])
            if narrativ_mismatch and red_team.get("red_team_verdict") != "VETO":
                log.warning(
                    f"  [{candidate['ticker']}] AUTO-VETO: Narrativ-Mismatch erkannt → "
                    f"'{arg1[:60]}'"
                )
                red_team["red_team_verdict"] = "VETO"

            if red_team.get("red_team_verdict") == "VETO":
                log.info(
                    f"  [{candidate['ticker']}] RED TEAM VETO → verworfen. "
                    f"Grund: {red_team.get('argument_1', 'n/a')}"
                )
                continue

            stats = analysis.get("stats_check", {})
            if stats.get("concern_level") == "high" and analysis.get("impact", 0) > 6:
                original = analysis["impact"]
                analysis["impact"] = 6
                log.info(
                    f"  [{candidate['ticker']}] Stats-Concern HIGH: "
                    f"Impact {original} → 6 gedeckelt"
                )

            haiku_reason  = candidate.get("prescreen_reason", "").lower()
            sonnet_dir    = analysis.get("direction", "")
            haiku_bullish = any(w in haiku_reason for w in
                ["positiv", "erhöht", "wachstum", "deal", "akquisition",
                 "expansion", "gewinn", "stieg", "prognose"])
            haiku_bearish = (
                any(w in haiku_reason for w in
                    ["warnung", "verlust", "rückgang", "verfehlt", "kürzung",
                     "rechtsstreit", "rückruf", "gegenwind", "sinkt", "enttäuscht"])
                or candidate.get("prescreen_category") == "bearish_reversal"
            )
            if haiku_bullish and sonnet_dir == "BEARISH":
                log.warning(
                    f"  [{candidate['ticker']}] ⚠️ WIDERSPRUCH: "
                    f"Haiku=BULLISH aber Sonnet=BEARISH → Impact≤6"
                )
                if analysis.get("impact", 0) > 6:
                    analysis["impact"] = 6
                analysis["direction_conflict"] = True
            elif haiku_bearish and sonnet_dir == "BULLISH":
                log.warning(
                    f"  [{candidate['ticker']}] ⚠️ WIDERSPRUCH: "
                    f"Haiku=BEARISH aber Sonnet=BULLISH → Impact≤6"
                )
                if analysis.get("impact", 0) > 6:
                    analysis["impact"] = 6
                analysis["direction_conflict"] = True
            else:
                analysis["direction_conflict"] = False

            cat_conf = analysis.get("catalyst_confidence")
            log.info(
                f"  [{candidate['ticker']}] "
                f"Impact={analysis['impact']} "
                f"Surprise={analysis['surprise']} "
                f"Direction={analysis['direction']} "
                f"CatalystConf={cat_conf}/10 "
                f"TTM={analysis.get('time_to_materialization','?')} "
                f"RedTeam={red_team.get('red_team_verdict', '?')} "
                f"{'⚠️ KONFLIKT' if analysis.get('direction_conflict') else '✅'}"
            )

            analyses.append({**candidate, "deep_analysis": analysis})

        return analyses

    def _prefetch_batch(self, shortlist: list[dict], deadline: float | None, routing: dict | None = None) -> dict:
        """Batch-Pfad (-50 %, gleiches Modell/Prompt): {idx: ("done", Analyse|None) | ("ctx", ctx)}.
        "ctx" = nicht per Batch beantwortet -> run() holt den Call synchron nach."""
        client = getattr(self, "client", None)
        conf = cost_telemetry.batch_settings("deep_analysis")
        if client is None or conf is None or len(shortlist) < conf["min_requests"]:
            return {}
        pre: dict = {}
        requests, tickers = {}, {}
        routing = routing or {}
        for idx, cand in enumerate(shortlist):
            try:
                ctx = self._prepare(cand)
            except Exception as e:  # noqa: BLE001 – Fallback: _analyze baut den Kontext neu
                log.warning(f"  [{cand.get('ticker')}] Prompt-Aufbau fehlgeschlagen: {e}")
                continue
            cached = self._lookup_cache(ctx)
            if cached is not None:
                pre[idx] = ("done", cached)
                continue
            pre[idx] = ("ctx", ctx)
            cid = f"da-{idx}"
            requests[cid], tickers[cid] = ctx["params"], ctx["ticker"]
        if len(requests) < conf["min_requests"]:
            return pre
        # A/B-Stichprobe: dieselben Prompts zusätzlich mit dem jeweils anderen Modell (nur Messung)
        ab = {}
        if routing.get("other"):
            for cid in model_routing.sample(list(requests), routing["n"]):
                ab["ab-" + cid.split("-", 1)[1]] = {**requests[cid], "model": routing["other"]}
        results = cost_telemetry.run_batch(client, workflow="deep_analysis", requests={**requests, **ab},
                                           tickers={**tickers, **{k: tickers["da-" + k[3:]] for k in ab}},
                                           max_wait_s=conf["max_wait_s"], poll_s=conf["poll_s"], deadline=deadline)
        for cid, msg in results.items():
            if not cid.startswith("da-"):
                continue
            idx = int(cid.split("-", 1)[1])
            ctx = {**pre[idx][1], "batch": True}
            raw_ref = None
            if ("ab-%d" % idx) in results:
                try:
                    raw_ref = self._parse_json(msg, ctx["ticker"])
                except Exception:  # noqa: BLE001 – unlesbar: _complete verwirft den Kandidaten ohnehin
                    raw_ref = None
            pre[idx] = ("done", self._complete(ctx, msg))
            other_msg = results.get("ab-%d" % idx)
            if other_msg is not None:
                self._record_pair(ctx, raw_ref, other_msg, routing)
        return pre

    def _routing(self) -> dict:
        """Aktives Modell (A/B-validiert) + Gegenmodell für die Stichprobe dieses Laufs."""
        champion = cfg.models.deep_analysis
        dec = model_routing.decision("deep_analysis", champion)
        self._active_model = dec.get("active", champion)
        s = model_routing.settings("deep_analysis") or {}
        if not s:
            return {"active": self._active_model}
        if self._active_model == champion:
            other, n = s["challenger"], int(s.get("sample_per_run", 6))
        else:
            other, n = champion, int(s.get("monitor_sample_per_run", 2))
        log.info(f"Modell-Routing deep_analysis: aktiv={self._active_model} ({dec.get('status')}), "
                 f"Stichprobe {n}x {other}")
        return {"active": self._active_model, "other": other, "n": n, "status": dec.get("status")}

    def _record_pair(self, ctx: dict, ref: Optional[dict], other_msg, routing: dict) -> None:
        try:
            other = self._parse_json(other_msg, ctx["ticker"])
        except Exception:  # noqa: BLE001 – unlesbare Challenger-Antwort zählt als Fehlentscheidung
            other = None
        g = getattr(cfg, "gates", None)
        cost_telemetry.record(model_routing.pair_row(
            "deep_analysis", ctx["ticker"], ctx["params"]["model"], routing["other"], ref, other,
            impact_min=int(getattr(g, "impact_min", 4)), surprise_min=int(getattr(g, "surprise_min", 3))))

    def _prepare(self, candidate: dict) -> dict:
        """Baut Prompt, Cache-Schlüssel und Kontext (ohne LLM-Call)."""
        ticker  = candidate.get("ticker", "")
        info    = candidate.get("info", {})
        news    = candidate.get("news", [])

        current_price = float(
            info.get("currentPrice") or
            info.get("regularMarketPrice") or 0
        )
        forward_eps  = info.get("forwardEps") or info.get("trailingEps") or 0.0
        sector       = info.get("sector", "Unknown")
        market_cap   = info.get("marketCap") or 0
        move_48h     = self._get_48h_move(ticker)

        eps_check     = candidate.get("data_validation", {}).get("eps_cross_check", {})
        sec_eps       = eps_check.get("sec_eps", "n/a")
        dev_pct       = eps_check.get("deviation_pct")
        eps_deviation = f"{dev_pct:.1%}" if dev_pct is not None else "n/a"

        data_anomaly    = candidate.get("data_anomaly", False)
        anomaly_warning = ""
        if data_anomaly:
            anomaly_warning = (
                "⚠️ DATA ANOMALY: EPS-Daten weichen >20% ab. "
                "Red Team sollte Datenqualität als Argument 1 nennen. "
                "data_confidence muss 'low' sein."
            )

        qmc         = candidate.get("quick_mc", {})
        mc_hit_rate = qmc.get("hit_rate", 0.0)
        mc_paths    = qmc.get("n_paths", 0)
        mc_days     = qmc.get("n_days", 30)

        if mc_hit_rate == 0:
            mc_interpretation = "Kein Quick MC durchgeführt — keine Statistik verfügbar."
        elif mc_hit_rate > 0.80:
            mc_interpretation = "WARNUNG: >80% Hit-Rate ist ungewöhnlich hoch — Modell möglicherweise zu optimistisch."
        elif mc_hit_rate > 0.60:
            mc_interpretation = "Solide statistische Basis — realistisch für 2-6M Horizont."
        else:
            mc_interpretation = f"Nur {mc_hit_rate:.0%} — knapp über Minimum-Gate, erhöhte Vorsicht."

        # v9.0 #12: Mega-Cap-Hinweis
        market_cap_str = _format_market_cap(market_cap)
        is_mega_cap    = market_cap >= MEGA_CAP_THRESHOLD
        mega_cap_flag  = (
            f"\n⚠️ MEGA-CAP: Informationseffizienz sehr hoch — Impact > 6 erfordert konkreten Edge!"
            if is_mega_cap else ""
        )
        mega_cap_step = "Mega-Cap-Check → " if is_mega_cap else ""

        news_text = "\n".join(f"- {h}" for h in news[:8]) if news else "Keine News."

        macro_text = (
            self._macro.get("claude_context", "Makro: nicht verfügbar")
            if self._macro.get("data_available")
            else "Makro-Kontext: FRED nicht erreichbar."
        )

        prescreen_reason   = candidate.get("prescreen_reason", "n/a")
        prescreen_category = candidate.get("prescreen_category", "n/a")

        from datetime import datetime as _dt
        _today = _dt.now()
        prompt = ANALYSIS_TEMPLATE.format(
            analysis_date      = _today.strftime("%d.%m.%Y"),
            analysis_year      = _today.year,
            macro_context      = macro_text,
            ticker             = ticker,
            current_price      = current_price,
            market_cap_str     = market_cap_str,
            mega_cap_flag      = mega_cap_flag,
            mega_cap_step      = mega_cap_step,
            sector             = sector,
            prescreen_reason   = prescreen_reason,
            prescreen_category = prescreen_category,
            forward_eps        = forward_eps,
            sec_eps            = sec_eps,
            eps_deviation      = eps_deviation,
            move_48h           = (f"{move_48h:+.1%}" if move_48h is not None
                                  else "n/a (keine Kursdaten – NICHT als 0 % interpretieren)"),
            mc_hit_rate        = mc_hit_rate,
            mc_paths           = mc_paths,
            mc_days            = mc_days,
            mc_interpretation  = mc_interpretation,
            news_text          = news_text,
            data_anomaly_warning = anomaly_warning,
        )

        model     = getattr(self, "_active_model", None) or cfg.models.deep_analysis
        cache_key = analysis_cache.deep_analysis_key(
            ticker=ticker, news=news, move_48h=move_48h, mc_hit_rate=mc_hit_rate,
            eps_deviation=eps_deviation,
            earnings_date=(candidate.get("alpha_signals") or {}).get("earnings_date"),
            macro_text=macro_text, sector=sector, prescreen_category=prescreen_category,
            is_mega_cap=is_mega_cap, data_anomaly=bool(data_anomaly),
            prompt_version=PROMPT_VERSION, model=model,
        )
        # System-Prompt als cachebarer Block (Prompt-Caching; unter der Mindestlänge
        # verarbeitet die API ohne Cache und ohne Aufpreis -> messbar via Telemetrie).
        params = {
            "model":      model,
            "max_tokens": 1600,
            "system":     [{"type": "text", "text": SYSTEM_PROMPT.format(current_year=_today.year),
                            "cache_control": {"type": "ephemeral"}}],
            "messages":   [{"role": "user", "content": prompt}],
        }
        return {"ticker": ticker, "cache_key": cache_key, "params": params, "mc_hit_rate": mc_hit_rate,
                "is_mega_cap": is_mega_cap, "market_cap": market_cap}

    def _lookup_cache(self, ctx: dict) -> Optional[dict]:
        cached = analysis_cache.lookup(ctx["cache_key"], workflow="deep_analysis", ticker=ctx["ticker"])
        if cached is None:
            return None
        log.info(f"  [{ctx['ticker']}] Analyse-Cache-Treffer (identische Eingaben) → kein Sonnet-Call")
        return self._finalize(cached, ctx["mc_hit_rate"], ctx["is_mega_cap"], ctx["market_cap"])

    def _analyze(self, candidate: dict, ctx: Optional[dict] = None) -> Optional[dict]:
        ticker = candidate.get("ticker", "")
        try:
            ctx = ctx or self._prepare(candidate)
            cached = self._lookup_cache(ctx)
            if cached is not None:
                return cached
            response = cost_telemetry.tracked_create(
                self.client, workflow="deep_analysis", ticker=ticker, **ctx["params"])
            return self._complete(ctx, response)
        except Exception as e:
            log.error(f"  [{ticker}] Deep Analysis Fehler: {e}")
            return None

    @staticmethod
    def _parse_json(response, ticker: str) -> Optional[dict]:
        """Rohtext -> JSON (inkl. Reparatur abgeschnittener Antworten); None = verworfen."""
        raw = response.content[0].text.strip()

        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        if not raw.startswith("{"):
            idx = raw.find("{")
            if idx != -1:
                raw = raw[idx:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError as je:
            log.warning(f"  [{ticker}] JSON teilweise abgeschnitten: {je} → Reparatur-Versuch")
            last_comma = raw.rfind('",')
            cutoff = max(last_comma, 0)
            if cutoff > 100:
                raw_fixed = raw[:cutoff] + '"}'
                try:
                    result = json.loads(raw_fixed)
                    log.info(f"  [{ticker}] JSON repariert (gekürzt auf {cutoff} Zeichen)")
                    return result
                except Exception:
                    # Kein erfundener BULLISH/PASSIERT-Datensatz mehr: der
                    # wäre (bei gelockerten Gates) als echte Analyse in
                    # Trades und Lerndaten gelandet (Review 2026-09-27).
                    log.warning(f"  [{ticker}] JSON nicht reparierbar → Kandidat verworfen")
                    return None
            raise

    def _complete(self, ctx: dict, response) -> Optional[dict]:
        """JSON-Parsing, Cache, Anreicherung – identisch für Batch- und Einzel-Antworten."""
        ticker, cache_key, model = ctx["ticker"], ctx["cache_key"], ctx["params"]["model"]
        mc_hit_rate, is_mega_cap, market_cap = ctx["mc_hit_rate"], ctx["is_mega_cap"], ctx["market_cap"]
        try:
            result = self._parse_json(response, ticker)
            if result is None:
                return None

            analysis_cache.compare_observed(cache_key, result, workflow="deep_analysis", ticker=ticker)
            analysis_cache.store(
                cache_key, result, model=model,
                cost_usd=cost_telemetry.compute_cost(model, cost_telemetry.usage_from_response(response),
                                                     batch=bool(ctx.get("batch"))))
            return self._finalize(result, mc_hit_rate, is_mega_cap, market_cap)

        except Exception as e:
            log.error(f"  [{ticker}] Deep Analysis Fehler: {e}")
            return None

    def _finalize(self, result: dict, mc_hit_rate, is_mega_cap: bool, market_cap) -> dict:
        """Deterministische Anreicherung des LLM-Rohergebnisses (auch für Cache-Treffer)."""
        # Sicherstellen dass catalyst_confidence vorhanden ist (Fallback: 5)
        if "catalyst_confidence" not in result:
            result["catalyst_confidence"] = 5

        result["macro_regime"]  = self._macro.get("macro_regime", "unknown")
        result["macro_context"] = {
            "yield_curve": self._macro.get("yield_curve_spread"),
            "regime":      self._macro.get("macro_regime"),
        }
        # v9.0 #3: mc_hit_rate im Result speichern für options_designer
        result["mc_hit_rate"]  = mc_hit_rate
        result["is_mega_cap"]  = is_mega_cap
        result["market_cap"]   = market_cap

        return result

    # ── FIX v8.2: 48h-Move Timing ────────────────────────────────────────────
    def _get_48h_move(self, ticker: str) -> float | None:
        """
        Berechnet die Preisbewegung der letzten 2 vollen Handelstage.

        FIX v8.2: Nutzt period='10d' und vergleicht volle Handelstage:
          close[-2] = gestriger Close (letzter abgeschlossener Tag)
          close[-4] = vor-3-Tage-Close (48h-Fenster)
        """
        try:
            hist = yf.Ticker(ticker).history(period="10d")
            close = hist["Close"]
            if hasattr(close, "iloc"):
                close = close.squeeze()
            if len(close) < 5:
                return None
            base = float(close.iloc[-4])
            return float((close.iloc[-2] - base) / base) if base > 0 else None
        except Exception as e:
            # Nie 0.0: "+0.0%" läse das LLM als "Markt hat nicht reagiert"
            # (= maximale Unterreaktion), obwohl nur Daten fehlen.
            log.warning(f"  [{ticker}] 48h-Move nicht berechenbar: {e}")
            return None
