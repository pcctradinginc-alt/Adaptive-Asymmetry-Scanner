"""
tests/test_nyse_calendar.py – Tests for NYSE calendar functions
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from modules.market_snapshot import (
    nyse_holidays,
    nyse_early_closes,
    is_trading_day,
    market_close_time,
    next_trading_day,
    us_market_session,
    MARKET_CLOSE,
    EARLY_CLOSE,
)

NY_TZ = ZoneInfo("America/New_York")


class TestNyseHolidays2026:
    """Test NYSE holidays for 2026."""

    def test_2026_holidays(self):
        """Test all 2026 holidays."""
        holidays = nyse_holidays(2026)

        # New Year's Day (Jan 1, Thursday)
        assert date(2026, 1, 1) in holidays

        # MLK Day (3rd Monday January = Jan 19)
        assert date(2026, 1, 19) in holidays

        # Washington's Birthday (3rd Monday February = Feb 16)
        assert date(2026, 2, 16) in holidays

        # Good Friday (Apr 3, 2026; Easter is Apr 5)
        assert date(2026, 4, 3) in holidays

        # Memorial Day (last Monday May = May 25)
        assert date(2026, 5, 25) in holidays

        # Juneteenth (June 19, Friday)
        assert date(2026, 6, 19) in holidays

        # Independence Day observed (Jul 3, Friday; Jul 4 is Saturday)
        assert date(2026, 7, 3) in holidays
        assert date(2026, 7, 4) not in holidays

        # Labor Day (1st Monday September = Sep 7)
        assert date(2026, 9, 7) in holidays

        # Thanksgiving (4th Thursday November = Nov 26)
        assert date(2026, 11, 26) in holidays

        # Christmas (Dec 25, Friday)
        assert date(2026, 12, 25) in holidays


class TestNyseEarlycloses2026:
    """Test NYSE early closes for 2026."""

    def test_2026_early_closes(self):
        """Test all 2026 early closes."""
        early_closes = nyse_early_closes(2026)

        # Day after Thanksgiving (Nov 27, Friday)
        assert date(2026, 11, 27) in early_closes

        # Christmas Eve (Dec 24, Thursday)
        assert date(2026, 12, 24) in early_closes

        # Jul 3 is a holiday, so NOT an early close
        assert date(2026, 7, 3) not in early_closes


class TestNyseHolidays2027:
    """Test NYSE holidays for 2027."""

    def test_2027_holidays(self):
        """Test all 2027 holidays."""
        holidays = nyse_holidays(2027)

        # New Year's Day (Jan 1, Friday)
        assert date(2027, 1, 1) in holidays

        # MLK Day (3rd Monday January = Jan 18)
        assert date(2027, 1, 18) in holidays

        # Washington's Birthday (3rd Monday February = Feb 15)
        assert date(2027, 2, 15) in holidays

        # Good Friday (Mar 26, 2027; Easter is Mar 28)
        assert date(2027, 3, 26) in holidays

        # Memorial Day (last Monday May = May 31)
        assert date(2027, 5, 31) in holidays

        # Juneteenth observed (June 18, Friday; June 19 is Saturday)
        assert date(2027, 6, 18) in holidays
        assert date(2027, 6, 19) not in holidays

        # Independence Day observed (July 5, Monday; July 4 is Sunday)
        assert date(2027, 7, 5) in holidays
        assert date(2027, 7, 4) not in holidays

        # Labor Day (1st Monday September = Sep 6)
        assert date(2027, 9, 6) in holidays

        # Thanksgiving (4th Thursday November = Nov 25)
        assert date(2027, 11, 25) in holidays

        # Christmas observed (Dec 24, Friday; Dec 25 is Saturday)
        assert date(2027, 12, 24) in holidays
        assert date(2027, 12, 25) not in holidays


class TestNyseEarlycloses2027:
    """Test NYSE early closes for 2027."""

    def test_2027_early_closes(self):
        """Test 2027 early closes."""
        early_closes = nyse_early_closes(2027)

        # Day after Thanksgiving (Nov 26, Friday)
        assert date(2027, 11, 26) in early_closes

        # Christmas Eve Dec 24 is a holiday (Christmas observed), so NOT an early close
        assert date(2027, 12, 24) not in early_closes


class TestNewYearSaturdayRule:
    """Test special NYSE rule for New Year's Day on Saturday."""

    def test_2022_new_year_saturday(self):
        """Test that Jan 1, 2022 (Saturday) is NOT observed on Dec 31, 2021."""
        holidays_2021 = nyse_holidays(2021)
        holidays_2022 = nyse_holidays(2022)

        # Dec 31, 2021 should NOT be a holiday
        assert date(2021, 12, 31) not in holidays_2021

        # Jan 1, 2022 (Saturday) should be a holiday
        assert date(2022, 1, 1) in holidays_2022


class TestIsTradingDay:
    """Test is_trading_day function."""

    def test_weekdays_are_trading_days(self):
        """Weekdays without holidays should be trading days."""
        # Jan 2, 2026 is a Friday (not a holiday)
        assert is_trading_day(date(2026, 1, 2)) is True

    def test_weekend_not_trading_days(self):
        """Weekends should not be trading days."""
        # Jan 3-4, 2026 is Saturday-Sunday
        assert is_trading_day(date(2026, 1, 3)) is False
        assert is_trading_day(date(2026, 1, 4)) is False

    def test_holidays_not_trading_days(self):
        """Holidays should not be trading days."""
        # Jan 1, 2026 is a holiday
        assert is_trading_day(date(2026, 1, 1)) is False

        # Jul 3, 2026 is a holiday (Independence Day observed)
        assert is_trading_day(date(2026, 7, 3)) is False


class TestMarketCloseTime:
    """Test market_close_time function."""

    def test_regular_close_time(self):
        """Regular trading days close at 16:00 ET."""
        # Jan 2, 2026 is a Friday (regular trading day, not early close)
        assert market_close_time(date(2026, 1, 2)) == MARKET_CLOSE

    def test_early_close_time(self):
        """Early close days close at 13:00 ET."""
        # Nov 27, 2026 is an early close day
        assert market_close_time(date(2026, 11, 27)) == EARLY_CLOSE

        # Dec 24, 2026 is an early close day
        assert market_close_time(date(2026, 12, 24)) == EARLY_CLOSE


class TestNextTradingDay:
    """Test next_trading_day function."""

    def test_weekday_to_next_weekday(self):
        """Next trading day after a weekday (if no holiday) is next weekday."""
        # Jan 2, 2026 (Friday) -> Jan 5, 2026 (Monday)
        assert next_trading_day(date(2026, 1, 2)) == date(2026, 1, 5)

    def test_skip_weekend(self):
        """Next trading day skips weekends."""
        # Jan 3, 2026 (Saturday) -> Jan 5, 2026 (Monday)
        assert next_trading_day(date(2026, 1, 3)) == date(2026, 1, 5)

    def test_skip_holiday(self):
        """Next trading day skips holidays."""
        # Dec 31, 2025 (Wednesday) -> Jan 2, 2026 (Friday, skip Jan 1 holiday)
        assert next_trading_day(date(2025, 12, 31)) == date(2026, 1, 2)


class TestUsMarketSessionBasic:
    """Test us_market_session with basic cases."""

    def test_pre_session(self):
        """Pre-market session (before 9:30 ET)."""
        # 2026-01-05 08:00 ET = 2026-01-05 13:00 UTC
        ts = datetime(2026, 1, 5, 13, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "pre"

    def test_regular_session(self):
        """Regular trading session (9:30-16:00 ET)."""
        # 2026-01-05 10:00 ET = 2026-01-05 15:00 UTC
        ts = datetime(2026, 1, 5, 15, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "regular"

    def test_post_session(self):
        """Post-market session (after 16:00 ET)."""
        # 2026-01-05 17:00 ET = 2026-01-05 22:00 UTC
        ts = datetime(2026, 1, 5, 22, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "post"

    def test_weekend_closed(self):
        """Weekend should be closed."""
        # 2026-01-03 (Saturday)
        ts = datetime(2026, 1, 3, 14, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "closed"


class TestUsMarketSessionHoliday:
    """Test us_market_session on holidays."""

    def test_holiday_is_closed(self):
        """On a holiday, market is closed regardless of time."""
        # 2026-01-01 (New Year's Day) at 14:00 ET should be "closed"
        ts = datetime(2026, 1, 1, 19, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "closed"

    def test_holiday_evening_is_closed(self):
        """On a holiday, even evening should be closed."""
        # 2026-07-03 (Independence Day observed) at 20:00 ET should be "closed"
        # 20:00 ET = 00:00 UTC next day (July 4)
        ts = datetime(2026, 7, 4, 0, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "closed"


class TestUsMarketSessionEarlyClose:
    """Test us_market_session on early close days."""

    def test_early_close_regular_session(self):
        """On early close day, regular session is 9:30-13:00 ET."""
        # 2026-11-27 (early close) at 11:00 ET = 16:00 UTC
        ts = datetime(2026, 11, 27, 16, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "regular"

    def test_early_close_post_session_before_1300(self):
        """On early close day at 13:00 ET exactly, should be post."""
        # 2026-11-27 (early close) at 13:00 ET = 18:00 UTC
        ts = datetime(2026, 11, 27, 18, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "post"

    def test_early_close_post_session_after_1300(self):
        """On early close day after 13:00 ET, should be post."""
        # 2026-11-27 (early close) at 14:00 ET = 19:00 UTC
        ts = datetime(2026, 11, 27, 19, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "post"


class TestUsMarketSessionDST:
    """Test us_market_session around DST boundaries."""

    def test_spring_forward_dst(self):
        """Test around spring forward DST boundary."""
        # 2026-03-08 02:00 EST becomes 03:00 EDT
        # After DST, 9:30 EDT = 13:30 UTC

        # 2026-03-09 10:00 EDT = 2026-03-09 14:00 UTC (in regular session)
        ts = datetime(2026, 3, 9, 14, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "regular"

    def test_fall_back_dst(self):
        """Test around fall back DST boundary."""
        # 2026-11-01 02:00 EDT becomes 01:00 EST
        # Before DST, 15:00 EDT = 20:00 UTC
        # After DST, 15:00 EST = 20:00 UTC

        # 2026-11-02 10:00 EST = 2026-11-02 15:00 UTC (in regular session)
        ts = datetime(2026, 11, 2, 15, 0, 0, tzinfo=timezone.utc)
        assert us_market_session(ts) == "regular"


class TestUsMarketSessionEdgeCases:
    """Test us_market_session edge cases."""

    def test_midnight_utc_conversion(self):
        """Test timestamp at midnight UTC."""
        # 2026-01-05 00:00 UTC = 2026-01-04 19:00 EST (previous day in ET, post-market)
        ts = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
        # ET is UTC-5 in January, so 00:00 UTC = 19:00 EST previous day
        # Monday pre-market should be after Sunday, so closed
        assert us_market_session(ts) == "closed"

    def test_timestamp_without_tzinfo(self):
        """Test that timestamp without tzinfo is treated as UTC."""
        # Should not raise, should treat as UTC
        ts = datetime(2026, 1, 5, 15, 0, 0)  # No tzinfo
        result = us_market_session(ts)
        assert result in ("pre", "regular", "post", "closed")


class TestEarlyCloseNotOnHolidays:
    """Test that early closes don't occur on holidays."""

    def test_july_3_2026_is_holiday_not_early_close(self):
        """July 3, 2026 (Independence Day observed) is a holiday, not an early close."""
        holidays = nyse_holidays(2026)
        early_closes = nyse_early_closes(2026)

        # Jul 3 should be a holiday
        assert date(2026, 7, 3) in holidays
        # Jul 3 should NOT be an early close
        assert date(2026, 7, 3) not in early_closes

    def test_christmas_eve_2027_is_holiday_not_early_close(self):
        """Dec 24, 2027 (Christmas observed) is a holiday, not an early close."""
        holidays = nyse_holidays(2027)
        early_closes = nyse_early_closes(2027)

        # Dec 24 should be a holiday
        assert date(2027, 12, 24) in holidays
        # Dec 24 should NOT be an early close
        assert date(2027, 12, 24) not in early_closes


class TestMarketCloseTimeConsistency:
    """Test consistency of market_close_time with early_closes."""

    def test_early_close_days_have_correct_time(self):
        """Days in early_closes should have EARLY_CLOSE as their close time."""
        early_closes = nyse_early_closes(2026)
        for d in early_closes:
            assert market_close_time(d) == EARLY_CLOSE

    def test_regular_days_have_correct_time(self):
        """Regular trading days should have MARKET_CLOSE as their close time."""
        # Jan 5, 2026 is a regular trading day
        assert market_close_time(date(2026, 1, 5)) == MARKET_CLOSE
