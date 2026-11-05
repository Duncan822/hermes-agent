"""Shutdown must drain queued memory sync before on_session_end.

One-shot (and any other caller of ``AIAgent.shutdown_memory_provider``)
used to invoke ``on_session_end`` while ``sync_all`` was still queued on
the manager's single worker. Providers that commit from that hook —
OpenViking only after the turn upload is registered — then saw no turn.
``shutdown_all`` drains the worker only afterwards, and one-shot
``os._exit`` skips atexit, so the upload landed with no commit.

The hook also has to run once. This tree has no
``MemoryManager.claim_teardown``; repeating teardown must not commit
twice, and a drain timeout or error must not skip the hook.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

import run_agent
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from run_agent import AIAgent


class _RecordingProvider(MemoryProvider):
    """Records sync, session-end, and shutdown on the real manager worker."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.saw_session_end = threading.Event()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "shutdown-order"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs) -> None:
        return None

    def get_tool_schemas(self):
        return []

    def sync_turn(
        self,
        user_content,
        assistant_content,
        *,
        session_id: str = "",
        messages=None,
    ) -> None:
        self._mark("sync")

    def on_session_end(self, messages) -> None:
        self._mark("session_end")

    def shutdown(self) -> None:
        self._mark("shutdown")

    def _mark(self, name: str) -> None:
        with self._lock:
            self.events.append(name)
        if name == "session_end":
            self.saw_session_end.set()


def _agent(mgr: MemoryManager, compressor: list | None = None) -> SimpleNamespace:
    sink = compressor if compressor is not None else []

    def _on_session_end(session_id, messages):
        sink.append((session_id, list(messages or [])))

    return SimpleNamespace(
        _memory_manager=mgr,
        session_id="sess-shutdown",
        context_compressor=SimpleNamespace(on_session_end=_on_session_end),
        _compressor_sink=sink,
    )


def _hold_worker(mgr: MemoryManager):
    """Block the single memory worker until the returned event is set."""
    hold = threading.Event()
    entered = threading.Event()

    def _gate() -> None:
        entered.set()
        assert hold.wait(5), "memory worker gate was not released"

    mgr._submit_background(_gate)
    assert entered.wait(2), "memory worker did not enter the gate"
    return hold


def test_shutdown_drains_queued_sync_before_session_end():
    """A sync still queued behind the worker must finish before on_session_end.

    The worker is held so the turn cannot start on its own. Shutdown runs
    on another thread. Before the drain-first fix, session end is observed
    while that turn is still blocked. After the fix, the drain is entered
    first and the turn lands before the hook.
    """
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    hold = _hold_worker(mgr)
    mgr.sync_all(
        "remember the pilot fact",
        "the pending token count was 157",
        session_id="sess-shutdown",
    )

    entered_flush = threading.Event()
    flush_timeouts: list[object] = []
    real_flush = mgr.flush_pending

    def _wrapped_flush(timeout=None):
        flush_timeouts.append(timeout)
        entered_flush.set()
        return real_flush(timeout=timeout)

    mgr.flush_pending = _wrapped_flush

    messages = [{"role": "user", "content": "remember the pilot fact"}]
    agent = _agent(mgr)
    errors: list[BaseException] = []

    def _shutdown() -> None:
        try:
            AIAgent.shutdown_memory_provider(agent, messages)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=_shutdown, name="memory-shutdown-order")
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if entered_flush.is_set() or provider.saw_session_end.is_set():
                break
            time.sleep(0.005)
        else:
            pytest.fail("shutdown did not reach the sync drain or session end")

        assert not provider.saw_session_end.is_set(), (
            "on_session_end ran while the queued sync_turn was still blocked"
        )
        assert entered_flush.is_set()
    finally:
        hold.set()
        thread.join(5)

    assert not thread.is_alive()
    assert errors == []
    assert provider.events == ["sync", "session_end", "shutdown"]
    assert flush_timeouts == [run_agent._MEMORY_SHUTDOWN_FLUSH_TIMEOUT_S]
    assert agent._compressor_sink == [("sess-shutdown", messages)]

    AIAgent.shutdown_memory_provider(agent, messages)
    assert provider.events == ["sync", "session_end", "shutdown"]
    assert flush_timeouts == [run_agent._MEMORY_SHUTDOWN_FLUSH_TIMEOUT_S]
    assert agent._compressor_sink == [("sess-shutdown", messages)]


def test_flush_timeout_still_runs_session_end_once(monkeypatch, caplog):
    """A wedged worker must not block teardown or skip the end hook.

    The drain budget is shortened. The hook still runs, a warning is
    logged, and a second teardown does not commit again.
    """
    monkeypatch.setattr(run_agent, "_MEMORY_SHUTDOWN_FLUSH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(
        "agent.memory_manager._SYNC_DRAIN_TIMEOUT_S",
        0.05,
    )

    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    hold = _hold_worker(mgr)
    mgr.sync_all(
        "remember the pilot fact",
        "still queued",
        session_id="sess-shutdown",
    )
    agent = _agent(mgr)
    messages = [{"role": "user", "content": "remember the pilot fact"}]

    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING, logger="run_agent"):
            AIAgent.shutdown_memory_provider(agent, messages)
            AIAgent.shutdown_memory_provider(agent, messages)
    finally:
        hold.set()

    assert time.monotonic() - started < 2
    assert "Memory sync drain timed out" in caplog.text
    assert provider.events[0] == "session_end"
    assert "shutdown" in provider.events
    assert provider.events.count("session_end") == 1
    assert provider.events.count("shutdown") == 1
    assert agent._compressor_sink == [("sess-shutdown", messages)]


def test_drain_and_session_end_errors_still_shut_down_once(caplog):
    """Drain and hook failures are logged; provider shutdown still runs once."""
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)

    def _flush_boom(timeout=None):
        provider.events.append("flush")
        raise RuntimeError("drain broke")

    def _end_boom(messages):
        provider.events.append("session_end")
        raise RuntimeError("end broke")

    mgr.flush_pending = _flush_boom
    mgr.on_session_end = _end_boom
    agent = _agent(mgr)
    messages = [{"role": "user", "content": "remember the pilot fact"}]

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        AIAgent.shutdown_memory_provider(agent, messages)
        AIAgent.shutdown_memory_provider(agent, messages)

    assert provider.events == ["flush", "session_end", "shutdown"]
    assert "drain broke" in caplog.text
    assert "end broke" in caplog.text
    assert agent._compressor_sink == [("sess-shutdown", messages)]
