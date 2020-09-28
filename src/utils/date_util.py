from datetime import datetime
import arrow
from enum import Enum
import os


class TimeZones(object):
    US_CA = 'US/Pacific'    # UTC
    US_NY = 'US/Eastern'    # EDT
    CN = 'Asia/Shanghai'


default_time_zone = TimeZones.US_NY
timezone_str = os.getenv('PY_TIMEZONE', 'US_NY')
default_time_zone = getattr(TimeZones, timezone_str, 'US/Eastern')

DEFAULT_DATE_FORMAT = 'YYYY-MM-DDTHH:mm:ssSS(ZZ)'
DEFAULT_SHORT_DATE_FORMAT = 'YYYY-MM-DDTHH:mm:ssZZZ'


def get_date_now(time_zone=default_time_zone, output_str=False, output_format=DEFAULT_DATE_FORMAT):
    if not output_str:
        return arrow.utcnow().to(time_zone)
    else:
        return date_to_str(arrow.utcnow().to(time_zone), output_format)


def date_to_str(date, format_str=DEFAULT_DATE_FORMAT):
    return date.format(format_str)


def str_to_date(date_str: str, format_str=DEFAULT_DATE_FORMAT):
    return arrow.get(date_str, format_str)


if __name__ == '__main__':
    # print(default_time_zone == 'US/Eastern')
    # print(arrow.utcnow().to(default_time_zone))

    date_now = get_date_now()
    print(date_now)
    date_now_str = date_to_str(date_now)
    print(date_now_str)
    # print(str_to_date(date_to_str(date_now, DEFAULT_SHORT_DATE_FORMAT), DEFAULT_SHORT_DATE_FORMAT))
    print(str_to_date(date_now_str))

    new_date_now = get_date_now()
    print(new_date_now - str_to_date(date_now_str))