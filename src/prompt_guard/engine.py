"""Single inference worker and a bounded queue.

A request that expires in the queue is discarded without running the model.
Cancellation and the deadline both stop further windows. The in-flight window
stops early only when the backend can abort it.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from prompt_guard.errors import (
    FatalConfigError,
    ScanError,
    internal_error,
    not_ready,
    queue_full,
    scan_timeout,
)
from prompt_guard.scanner import ScanSuccess

log = logging.getLogger(__name__)

Loader = Callable[[], tuple[object, object]]


@dataclass
class Job:
    text: str = field(repr=False)
    deadline: float = 0.0
    future: asyncio.Future[ScanSuccess] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)


class Engine:
    def __init__(
        self,
        *,
        queue_size: int,
        scanner: object | None = None,
        info: object | None = None,
        loader: Loader | None = None,
    ) -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=queue_size)
        self._scanner = scanner
        self._info = info
        self._loader = loader
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._failed = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name="prompt-guard-inference",
            daemon=True,
        )
        self._started = False
        self._start_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._scanner is not None and self._info is not None

    @property
    def info(self) -> object | None:
        if not self.ready:
            return None
        return self._info

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            if self._loader is None and self._scanner is not None and self._info is not None:
                self._ready.set()
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    async def submit(self, text: str, deadline: float) -> ScanSuccess:
        if not self.ready:
            raise not_ready()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ScanSuccess] = loop.create_future()
        job = Job(text=text, deadline=deadline, future=future, loop=loop)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            raise queue_full() from None
        try:
            return await future
        except asyncio.CancelledError:
            job.cancelled.set()
            job.deadline = time.monotonic()
            raise

    def _thread_main(self) -> None:
        if self._loader is not None:
            loaded = self._load_with_retry()
            if loaded is None:
                return
            self._scanner, self._info = loaded
            self._ready.set()
        self._serve()

    def _load_with_retry(self) -> tuple[object, object] | None:
        delay = 2.0
        assert self._loader is not None
        while not self._stop.is_set():
            try:
                return self._loader()
            except FatalConfigError:
                log.exception("model configuration is invalid; not retrying")
                self._failed = True
                return None
            except Exception:
                log.exception("model load failed; retrying in %.0f seconds", delay)
                self._failed = True
                if self._stop.wait(delay):
                    return None
                delay = min(delay * 2, 60)
        return None

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        if job.cancelled.is_set():
            return
        if time.monotonic() >= job.deadline:
            self._resolve(job, scan_timeout())
            return
        scanner = self._scanner
        if scanner is None:
            self._resolve(job, not_ready())
            return

        def should_stop() -> bool:
            return job.cancelled.is_set() or time.monotonic() >= job.deadline

        try:
            result = scanner.scan(job.text, should_stop)  # type: ignore[attr-defined]
        except ScanError as exc:
            self._resolve(job, exc)
            return
        except Exception:
            log.exception("inference failed")
            self._resolve(job, internal_error())
            return
        if job.cancelled.is_set() or time.monotonic() >= job.deadline:
            self._resolve(job, scan_timeout())
            return
        self._resolve(job, result)

    def _resolve(self, job: Job, value: ScanSuccess | ScanError) -> None:
        if job.cancelled.is_set() or job.future is None or job.loop is None:
            return

        def set_result() -> None:
            if job.future is None or job.future.done():
                return
            if isinstance(value, Exception):
                job.future.set_exception(value)
            else:
                job.future.set_result(value)

        job.loop.call_soon_threadsafe(set_result)
