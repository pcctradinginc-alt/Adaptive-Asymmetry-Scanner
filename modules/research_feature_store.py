"""PIT research features from ALFRED liquidity and previously archived release schedules.

Research only: never attaches features to production proposals. A macro value is
eligible only after its documented availability; date-only ALFRED availability
receives a full-day embargo. Calendar data MUST come from historical snapshots.
No backfilled *current* calendar is ever treated as historically known.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from modules.external.pit import AvailabilityPrecision
from modules.external.sources.calendars import schedule_as_of

SOURCES = (
    "fred_regime_macro", "fred_liquidity",
    "fomc_calendar", "bls_release_calendar", "bea_release_calendar",
)
CALENDARS = ("fomc_calendar", "bls_release_calendar", "bea_release_calendar")
FEATURES = (
    "fd3_net_liquidity_bn", "fd3_net_liquidity_delta_20d_bn",
    "fd3_us_curve_10y_2y", "fd3_fomc_days", "fd3_bls_days", "fd3_bea_days",
    "fd3_liquidity_x_beta", "fd3_fomc_near_x_beta",
    "fd3_bls_near_x_beta", "fd3_bea_near_x_beta",
)


def signal_time(day: date | str) -> datetime:
    """As-of just after US close, prior to the next trading day's entry.

    Midnight UTC on the following calendar day is after US regular close
    even across DST; it is never used as a proxy for a real intraday scan.
    """
    if isinstance(day, str):
        day = date.fromisoformat(day[:10])
    return datetime.combine(day + timedelta(days=1), time(0), tzinfo=timezone.utc)


class PITResearchStore:
    """Load each source once, then perform all historical joins in memory."""

    def __init__(self, archive):
        self.observations = {sid: archive.load(sid) for sid in SOURCES}

    def load(self, source_id):
        """Adapter for the repository's schedule_as_of() snapshot reader."""
        return self.observations.get(source_id, [])

    def latest(self, source: str, metric: str, as_of: datetime, age_days: int):
        # A date-only publication might have occurred AFTER the US close.
        # Require one complete calendar day of availability in that case.
        valid = []
        for obs in self.observations[source]:
            if obs.metric != metric or obs.value is None:
                continue
            if obs.observation_time > as_of or obs.available_at > as_of:
                continue
            if obs.availability_precision == AvailabilityPrecision.EXACT_DATE:
                if obs.available_at > as_of - timedelta(days=1):
                    continue
            if not obs.is_confirmatory() or obs.quality_status in ("INVALID", "MISSING"):
                continue
            if as_of - obs.observation_time > timedelta(days=age_days):
                continue
            valid.append(obs)
        if not valid:
            return None
        # Most recent observed period; for a revised period, newest known
        # vintage only (no lookahead into subsequent corrections).
        chosen = max(valid, key=lambda o: (
            o.observation_time, o.vintage_time or o.available_at, o.available_at,
        ))
        return float(chosen.value)

    def net_liquidity(self, t: datetime):
        assets = self.latest("fred_regime_macro", "fed_total_assets", t, 14)
        tga = self.latest("fred_liquidity", "us_treasury_general_account", t, 14)
        rrp = self.latest("fred_liquidity", "us_on_rrp", t, 7)
        if any(v is None for v in (assets, tga, rrp)):
            return None
        # Assets/TGA: USD millions. RRP: USD billions.
        return assets / 1000 - tga / 1000 - rrp

    def event_distance(self, source: str, t: datetime):
        plan = schedule_as_of(source, t, archive=self)
        if plan is None:
            return None
        future = [
            (e["time"] - t).total_seconds() / 86400.0
            for e in plan if e["time"] >= t
        ]
        return min(future) if future else None

    def date_features(self, day: date | str) -> dict:
        t = signal_time(day)
        net = self.net_liquidity(t)
        prev = self.net_liquidity(t - timedelta(days=20))
        d = {
            "fd3_net_liquidity_bn": net,
            "fd3_net_liquidity_delta_20d_bn": net - prev if net is not None and prev is not None else None,
            "fd3_us_curve_10y_2y": self.latest("fred_liquidity", "us_curve_10y_2y", t, 7),
        }
        for name, source in (("fomc", CALENDARS[0]), ("bls", CALENDARS[1]), ("bea", CALENDARS[2])):
            d[f"fd3_{name}_days"] = self.event_distance(source, t)
        return d

    def enrich_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Research-only frame. Original panel/production inputs stay untouched."""
        out = panel.copy()
        dates = sorted({str(pd.Timestamp(d).date()) for d in out["date"]})
        rows = {d: self.date_features(d) for d in dates}
        for field in FEATURES[:6]:
            out[field] = out["date"].map(lambda d: rows[str(pd.Timestamp(d).date())][field])
        beta = pd.to_numeric(out["beta_126"], errors="coerce")
        out["fd3_liquidity_x_beta"] = out["fd3_net_liquidity_delta_20d_bn"] * beta
        for name in ("fomc", "bls", "bea"):
            days = out[f"fd3_{name}_days"]
            # Unknown schedule remains NaN, never "no event".
            near = days.map(lambda x: float(0 <= x <= 2) if pd.notna(x) else float("nan"))
            out[f"fd3_{name}_near_x_beta"] = near * beta
        return out
