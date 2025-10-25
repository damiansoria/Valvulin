"""Async scheduler used by the trading engine."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

Callback = Callable[[], Awaitable[None]]


@dataclass
class ScheduledJob:
    name: str
    interval: float
    callback: Callback
    warmup: float = 0.0


class AsyncScheduler:
    """Simple asyncio based scheduler."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, logger: Optional[logging.Logger] = None) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.logger = logger or logging.getLogger(__name__)
        self._jobs: Dict[str, ScheduledJob] = {}
        self._tasks: Dict[str, "asyncio.Task[None]"] = {}
        self._stop_event = asyncio.Event()
        self._started = False

    def add_job(self, name: str, interval: float, callback: Callback, warmup: float = 0.0) -> None:
        if name in self._jobs:
            raise ValueError(f"Job '{name}' already registered")
        self._jobs[name] = ScheduledJob(name=name, interval=interval, callback=callback, warmup=warmup)
        if self._started:
            self._tasks[name] = self.loop.create_task(self._job_runner(self._jobs[name]))
            self.logger.debug("Started scheduled job '%s' immediately after registration", name)

    async def start(self) -> None:
        if self._started:
            return
        self.logger.debug("Starting %d scheduled jobs", len(self._jobs))
        self._stop_event.clear()
        for job in self._jobs.values():
            self._tasks[job.name] = self.loop.create_task(self._job_runner(job))
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self.logger.debug("Stopping scheduler")
        self._stop_event.set()
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._started = False

    async def _job_runner(self, job: ScheduledJob) -> None:
        try:
            if job.warmup > 0:
                await asyncio.sleep(job.warmup)
            while not self._stop_event.is_set():
                try:
                    await job.callback()
                except Exception:  # pragma: no cover - logging side effect
                    self.logger.exception("Error while executing job '%s'", job.name)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=job.interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover
            pass
