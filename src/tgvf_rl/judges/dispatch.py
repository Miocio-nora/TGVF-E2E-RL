"""Bounded execution boundary for synchronous Policy reward judges.

The canonical Qwen2.5-72B provider is synchronous because its fail-closed HTTP
client is also used by source-only audits.  Policy rollout, however, must not
block the asyncio agent loop on that HTTP call.  This module provides one
explicit bridge: a separately owned thread pool with bounded concurrency and
both synchronous and awaitable entry points.

The dispatcher never catches provider failures and never selects another
model.  In particular, using a smaller judge is a semantic intervention that
requires an explicit alternate route, exact model name and identity, and an
acknowledgement flag.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import threading
from types import TracebackType

from tgvf_rl.contracts.identity import ArtifactIdentity

from .base import JudgeProvider, JudgeRequest, JudgeResult


JUDGE_DISPATCH_SCHEMA_VERSION = "tgvf-judge-dispatch-v1"
JUDGE_DISPATCH_MAXIMUM_CONCURRENCY = 256
_QWEN25_72B_SERVED_NAMES = frozenset(
    {
        "qwen2.5-72b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen2.5-72b-instruct",
    }
)


class JudgeDispatchMode(str, Enum):
    """Ownership of blocking judge calls."""

    INLINE = "inline"
    DEDICATED_THREAD_POOL = "dedicated_thread_pool"


class JudgeModelRoute(str, Enum):
    """Reward semantics selected by the runtime binding."""

    QWEN25_72B = "qwen2.5_72b"
    EXPLICIT_ALTERNATE = "explicit_alternate"


@dataclass(frozen=True, slots=True)
class JudgeDispatchConfig:
    """Typed process-local execution and model-route binding.

    ``QWEN25_72B`` is the only zero-configuration route.  An alternate judge
    needs all three opt-in fields so a performance setting cannot silently
    change the reward function.
    """

    mode: JudgeDispatchMode = JudgeDispatchMode.INLINE
    maximum_concurrency: int = 1
    model_route: JudgeModelRoute = JudgeModelRoute.QWEN25_72B
    alternate_model_name: str | None = None
    alternate_model_identity: ArtifactIdentity | None = None
    alternate_semantics_acknowledged: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, JudgeDispatchMode):
            raise TypeError("judge dispatch mode must be JudgeDispatchMode")
        if (
            type(self.maximum_concurrency) is not int
            or not 1 <= self.maximum_concurrency <= JUDGE_DISPATCH_MAXIMUM_CONCURRENCY
        ):
            raise ValueError("judge maximum_concurrency must lie in [1,256]")
        if not isinstance(self.model_route, JudgeModelRoute):
            raise TypeError("judge model_route must be JudgeModelRoute")
        if type(self.alternate_semantics_acknowledged) is not bool:
            raise TypeError("alternate judge acknowledgement must be bool")

        alternate_fields_present = (
            self.alternate_model_name is not None,
            self.alternate_model_identity is not None,
            self.alternate_semantics_acknowledged,
        )
        if self.model_route is JudgeModelRoute.QWEN25_72B:
            if any(alternate_fields_present):
                raise ValueError(
                    "default Qwen2.5-72B route cannot carry alternate judge fields"
                )
            return

        if not isinstance(self.alternate_model_name, str) or not (
            self.alternate_model_name.strip()
        ):
            raise ValueError("explicit alternate judge requires its model name")
        if not isinstance(self.alternate_model_identity, ArtifactIdentity):
            raise ValueError("explicit alternate judge requires its model identity")
        if not self.alternate_semantics_acknowledged:
            raise ValueError(
                "explicit alternate judge requires semantic-change acknowledgement"
            )

    def validate_bound_model(
        self,
        *,
        model_name: str,
        model_identity: ArtifactIdentity,
    ) -> None:
        """Fail closed if the provider differs from the selected route."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("bound judge model name must be non-empty")
        if not isinstance(model_identity, ArtifactIdentity):
            raise TypeError("bound judge model identity must be ArtifactIdentity")
        if self.model_route is JudgeModelRoute.QWEN25_72B:
            if model_name.strip().casefold() not in _QWEN25_72B_SERVED_NAMES:
                raise ValueError(
                    "default reward judge must remain Qwen2.5-72B; "
                    "use explicit_alternate for another model"
                )
            return

        assert self.alternate_model_name is not None
        assert self.alternate_model_identity is not None
        if model_name != self.alternate_model_name:
            raise ValueError("bound alternate judge model name differs")
        if model_identity != self.alternate_model_identity:
            raise ValueError("bound alternate judge model identity differs")

    @property
    def semantic_fallback_route(self) -> str:
        """Audit route that names the actual selected reward semantics."""

        if self.model_route is JudgeModelRoute.QWEN25_72B:
            return "qwen2.5_72b_semantic_fallback"
        assert self.alternate_model_identity is not None
        identity = self.alternate_model_identity
        return (
            "explicit_alternate_semantic_fallback:"
            f"{identity.namespace}/{identity.name}@{identity.version}:"
            f"{identity.sha256}"
        )


class BoundedJudgeDispatcher:
    """Expose one judge through a bounded, optionally dedicated executor.

    Provider exceptions cross this boundary unchanged.  Consequently the
    caller's existing sample-level failure policy remains authoritative and a
    transport, identity, or permanent service failure still aborts scoring.
    """

    def __init__(
        self,
        provider: JudgeProvider,
        *,
        config: JudgeDispatchConfig = JudgeDispatchConfig(),
        bound_model_name: str,
        bound_model_identity: ArtifactIdentity,
    ) -> None:
        if not callable(getattr(provider, "judge", None)):
            raise TypeError("judge dispatcher provider must implement judge()")
        if not isinstance(config, JudgeDispatchConfig):
            raise TypeError("judge dispatcher config must be JudgeDispatchConfig")
        config.validate_bound_model(
            model_name=bound_model_name,
            model_identity=bound_model_identity,
        )
        self.provider = provider
        self.config = config
        self.bound_model_name = bound_model_name
        self.bound_model_identity = bound_model_identity
        self.semantic_fallback_route = config.semantic_fallback_route
        self._slots = threading.BoundedSemaphore(config.maximum_concurrency)
        self._state_lock = threading.Lock()
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(
                max_workers=config.maximum_concurrency,
                thread_name_prefix="tgvf-reward-judge",
            )
            if config.mode is JudgeDispatchMode.DEDICATED_THREAD_POOL
            else None
        )

    def judge(self, request: JudgeRequest) -> JudgeResult:
        """Synchronous compatibility entry point with the same hard failures."""

        self._validate_request(request)
        if self._executor is None:
            self._ensure_open()
            return self._execute(request)
        return self._submit(request).result()

    async def judge_async(self, request: JudgeRequest) -> JudgeResult:
        """Await one blocking provider call without blocking the event loop."""

        self._validate_request(request)
        if self._executor is None:
            self._ensure_open()
            return await asyncio.to_thread(self._execute, request)
        return await asyncio.wrap_future(self._submit(request))

    def _submit(self, request: JudgeRequest) -> Future[JudgeResult]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("judge dispatcher is closed")
            executor = self._executor
            if executor is None:
                raise RuntimeError("inline judge dispatcher has no dedicated executor")
            return executor.submit(self._execute, request)

    def _execute(self, request: JudgeRequest) -> JudgeResult:
        with self._slots:
            result = self.provider.judge(request)
        if not isinstance(result, JudgeResult):
            raise TypeError("judge provider must return JudgeResult")
        return result

    @staticmethod
    def _validate_request(request: JudgeRequest) -> None:
        if not isinstance(request, JudgeRequest):
            raise TypeError("judge dispatcher requires JudgeRequest")

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("judge dispatcher is closed")

    def close(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Close the owned pool; repeated closes are harmless."""

        if type(wait) is not bool or type(cancel_futures) is not bool:
            raise TypeError("judge dispatcher close flags must be bool")
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> BoundedJudgeDispatcher:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    async def __aenter__(self) -> BoundedJudgeDispatcher:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await asyncio.to_thread(self.close)


__all__ = [
    "JUDGE_DISPATCH_MAXIMUM_CONCURRENCY",
    "JUDGE_DISPATCH_SCHEMA_VERSION",
    "BoundedJudgeDispatcher",
    "JudgeDispatchConfig",
    "JudgeDispatchMode",
    "JudgeModelRoute",
]
