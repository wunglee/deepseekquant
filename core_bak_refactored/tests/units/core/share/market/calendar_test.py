

from core_bak_refactored.core.share.market.calendar import is_market_holiday


def test_is_market_holiday_checks_common_holidays():
    assert is_market_holiday(datetime(2024, 1, 1)) is True  # New Year's Day
    assert is_market_holiday(datetime(2024, 7, 4)) is True  # Independence Day
    assert is_market_holiday(datetime(2024, 12, 25)) is True  # Christmas
    assert is_market_holiday(datetime(2024, 3, 15)) is False  # Regular day
