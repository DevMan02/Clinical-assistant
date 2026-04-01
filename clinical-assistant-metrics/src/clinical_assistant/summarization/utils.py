import asyncio
import calendar
from datetime import date
from typing import Awaitable, Iterable


async def _bounded[T](coro: Awaitable[T], sem: asyncio.Semaphore) -> T:
    async with sem:
        return await coro


async def gather_bounded[T](
    sem: asyncio.Semaphore,
    coros: Iterable[Awaitable[T]],
) -> list[T]:
    return await asyncio.gather(*[_bounded(c, sem) for c in coros])


def format_dates(start: date, end: date) -> str:
    if start.month == end.month and start.year == end.year:
        return f"{calendar.month_abbr[start.month]} {start.year}"
    elif start.year == end.year:
        return f"{calendar.month_abbr[start.month]}-{calendar.month_abbr[end.month]} {start.year}"
    else:
        return f"{calendar.month_abbr[start.month]} {start.year} - {calendar.month_abbr[end.month]} {end.year}"