"""AUD-001 (Audit 2026-11-26): TTM-EPS aus SEC-companyfacts nur aus eindeutigen, nicht überlappenden Perioden."""
from datetime import date

from modules import data_validator as dv


def _u(start, end, val, form="10-Q", filed=None):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed or end[:8] + "28"}


# Kalenderjahr-Unternehmen: so liefert companyfacts ein 10-Q für Q3 2026 (Quartal UND 9-Monats-Wert mit
# demselben Periodenende, dazu Vorjahresvergleiche) plus den letzten 10-K.
UNITS = [
    _u("2025-01-01", "2025-12-31", 8.0, form="10-K", filed="2026-02-20"),
    _u("2025-07-01", "2025-09-30", 1.9, filed="2025-10-30"),
    _u("2025-01-01", "2025-09-30", 5.9, filed="2025-10-30"),
    _u("2026-01-01", "2026-03-31", 2.1, filed="2026-04-30"),
    _u("2026-04-01", "2026-06-30", 2.2, filed="2026-07-30"),
    _u("2026-01-01", "2026-06-30", 4.3, filed="2026-07-30"),
    _u("2026-07-01", "2026-09-30", 2.3, filed="2026-10-30"),
    _u("2026-01-01", "2026-09-30", 6.6, filed="2026-10-30"),
    # Vergleichswert desselben Vorjahres-Quartals, erneut im 10-Q 2026 eingereicht (Duplikat)
    _u("2025-07-01", "2025-09-30", 1.9, filed="2026-10-30"),
]


def _old_sum(units):
    """Bisherige Logik (zum Nachweis des Fehlers)."""
    q = [u for u in units if u.get("form") in ("10-Q", "10-K") and u.get("val") is not None]
    q.sort(key=lambda x: x.get("end", ""), reverse=True)
    return sum(u["val"] for u in q[:4])


def test_old_logic_double_counts_ytd_values():
    assert round(_old_sum(UNITS), 4) == 15.4          # Q3 + 9M + Q2 + H1 – fast das Doppelte des Jahres-EPS


def test_ttm_is_fiscal_year_plus_ytd_minus_prior_ytd():
    eps, method = dv.ttm_eps_from_units(UNITS, as_of=date(2026, 11, 26))
    assert eps == round(8.0 + 6.6 - 5.9, 4)
    assert "YTD" in method


def test_latest_fiscal_year_is_used_directly():
    units = UNITS + [_u("2026-01-01", "2026-12-31", 9.1, form="10-K", filed="2027-02-20")]
    assert dv.ttm_eps_from_units(units, as_of=date(2027, 3, 1)) == (9.1, "Geschäftsjahr")


def test_point_in_time_ignores_later_filings_and_uses_latest_revision():
    # Am 2026-08-15 war das Q3-10-Q noch nicht eingereicht: H1 2026 ist der jüngste Stand.
    eps, _ = dv.ttm_eps_from_units(UNITS + [_u("2025-01-01", "2025-06-30", 3.8, filed="2025-07-30")],
                                   as_of=date(2026, 8, 15))
    assert eps == round(8.0 + 4.3 - 3.8, 4)
    # Revision des Jahreswerts (10-K/A) gilt erst ab ihrer Einreichung.
    revised = UNITS + [_u("2025-01-01", "2025-12-31", 7.5, form="10-K/A", filed="2026-11-01")]
    assert dv.ttm_eps_from_units(revised, as_of=date(2026, 10, 31))[0] == round(8.0 + 6.6 - 5.9, 4)
    assert dv.ttm_eps_from_units(revised, as_of=date(2026, 11, 26))[0] == round(7.5 + 6.6 - 5.9, 4)


def test_four_consecutive_quarters_fallback():
    units = [_u("2025-10-01", "2025-12-31", 1.0), _u("2026-01-01", "2026-03-31", 1.1),
             _u("2026-04-01", "2026-06-30", 1.2), _u("2026-07-01", "2026-09-30", 1.3)]
    assert dv.ttm_eps_from_units(units, as_of=date(2026, 11, 26)) == (4.6, "4 Einzelquartale")


def test_unclear_data_returns_none_instead_of_a_wrong_number():
    assert dv.ttm_eps_from_units([_u("2026-07-01", "2026-09-30", 2.3)], as_of=date(2026, 11, 26)) is None
    assert dv.ttm_eps_from_units([{"end": "2026-09-30", "val": 2.3, "form": "10-Q"}]) is None   # ohne start
    assert dv.ttm_eps_from_units([_u("2026-07-01", "2026-09-30", 2.3, form="8-K")]) is None
    assert dv.ttm_eps_from_units([]) is None


def test_fetch_uses_the_corrected_ttm(monkeypatch):
    class _Resp:
        def __init__(self, payload):
            self.status_code, self._p = 200, payload

        def json(self):
            return self._p

    monkeypatch.setattr(dv, "_sec_fail_streak", 0)
    monkeypatch.setattr(dv, "_get_cik", lambda t: "0000000001")
    monkeypatch.setattr(dv, "ttm_eps_from_units", lambda units, as_of=None: (8.7, "test"))
    payload = {"facts": {"us-gaap": {"EarningsPerShareBasic": {"units": {"USD/shares": UNITS}}}}}
    monkeypatch.setattr(dv.requests, "get", lambda url, **kw: _Resp(payload))
    assert dv.fetch_eps_sec_edgar("XYZ") == 8.7
