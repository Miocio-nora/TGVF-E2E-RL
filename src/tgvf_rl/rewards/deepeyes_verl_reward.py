"""Executable veRL reward-loop adapter for the official DeepEyes control.

Pinned veRL 0.9 calls ``RewardManagerBase.run_single`` concurrently for all
rows assigned to one reward worker.  PRL13 deliberately uses one reward
worker and bounds the OpenRouter requests inside that worker with an asyncio
semaphore.  This keeps the mandatory 3,152 visual judgements per stratified
step concurrent without selecting the legacy, unused v0 ``BatchRewardManager``.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from tgvf_rl.judges.base import JudgeUsage
from tgvf_rl.rewards.deepeyes_batch import (
    JudgeGlobalFailure,
    JudgeSampleOutputError,
)
from tgvf_rl.rewards.deepeyes_official import (
    DEEPEYES_BINARY_JUDGE_MAX_TOKENS,
    DEEPEYES_BINARY_JUDGE_TOP_P,
    DEEPEYES_THINKLITE_ANSWER_WEIGHT,
    DEEPEYES_THINKLITE_FORMAT_WEIGHT,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    DEEPEYES_THINKLITE_JUDGE_TEMPERATURE,
    DEEPEYES_VISUAL_ANSWER_LIMIT,
    DEEPEYES_VISUAL_ANSWER_WEIGHT,
    DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT,
    DEEPEYES_VISUAL_FORMAT_WEIGHT,
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    DEEPEYES_VISUAL_JUDGE_TEMPERATURE,
    DeepEyesBinaryJudgeRequest,
    extract_thinklite_answer,
    extract_visual_answer,
    parse_binary_judge_output,
)


DEEPEYES_VERL_REWARD_SCHEMA = "tgvf.deepeyes-verl-reward-manager.v1"
DEEPEYES_VERL_REWARD_MANAGER_CLASS = (
    "tgvf_rl.rewards.deepeyes_verl_reward.DeepEyesOfficialRewardManager"
)
DEEPEYES_VERL_AUDIT_SEQUENCE_ENCODING = "canonical-json-v1"
_RAGGED_AUDIT_FIELDS = frozenset(
    {
        "crop_boxes",
        "crop_area_fractions",
        "crop_area",
        "crop_observation_token_spans",
    }
)
_OPTIONAL_NUMERIC_AUDIT_FIELDS = frozenset(
    {
        "crop_first_call_iou",
        "crop_best_call_iou",
        "crop_best_gt_coverage",
        "first_iou",
        "best_iou",
        "gt_coverage",
    }
)


def _plain(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, Mapping)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _required_text(value: object, name: str) -> str:
    value = _plain(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DeepEyes reward {name} must be non-empty text")
    return value


def _batch_safe_reward_extra_info(result: Mapping[str, object]) -> dict[str, object]:
    """Encode ragged audit sequences for upstream AgentLoop batching.

    Pinned veRL constructs each reward-extra column with ``np.array(values)``
    and therefore cannot batch variable-length list values.  The full native
    Crop audit remains losslessly available in rollout dumps, but crosses this
    boundary as canonical JSON text so direct-answer and multi-Crop samples
    share one scalar column shape.  Numeric reward/health metrics stay numeric.
    """

    encoded: dict[str, object] = {}
    for name, value in result.items():
        if name in _RAGGED_AUDIT_FIELDS:
            if not isinstance(value, list):
                raise TypeError(f"DeepEyes ragged audit field {name} is not a list")
            encoded[name] = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            continue
        if value is None:
            if name not in _OPTIONAL_NUMERIC_AUDIT_FIELDS:
                raise TypeError(
                    f"DeepEyes reward extra field {name} is unexpectedly null"
                )
            # Pinned veRL computes validation means for every non-string reward
            # extra.  Python None makes np.mean fail; NaN retains the numeric
            # schema while preserving the distinction from a real zero IoU.
            encoded[name] = math.nan
            continue
        if isinstance(value, (list, tuple, Mapping)):
            raise TypeError(
                f"DeepEyes reward extra field {name} is not scalar/batch-safe"
            )
        encoded[name] = value
    encoded["audit_sequence_encoding"] = DEEPEYES_VERL_AUDIT_SEQUENCE_ENCODING
    return encoded


@dataclass(frozen=True, slots=True)
class DeepEyesJudgeServiceConfig:
    file_sha256: str
    model: str
    base_url: str
    api_key_env: str
    provider_only: tuple[str, ...]
    allow_fallbacks: bool
    timeout_seconds: float
    maximum_concurrency: int
    maximum_attempts: int
    retry_backoff_seconds: float
    retry_maximum_seconds: float
    cache_max_entries: int
    transient_failure_window_size: int
    maximum_transient_failure_fraction: float

    def __post_init__(self) -> None:
        if len(self.file_sha256) != 64:
            raise ValueError("judge service file SHA-256 differs")
        if self.model != "qwen/qwen-2.5-72b-instruct":
            raise ValueError("judge service model differs from Qwen2.5-72B")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("judge service base_url must be HTTP(S)")
        if self.api_key_env != "OPENROUTER_API_KEY":
            raise ValueError("judge service must use OPENROUTER_API_KEY")
        if self.provider_only != ("deepinfra",) or self.allow_fallbacks:
            raise ValueError("judge service must pin DeepInfra without fallbacks")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("judge timeout must be positive")
        if not 1 <= self.maximum_concurrency <= 256:
            raise ValueError("judge maximum_concurrency must lie in [1,256]")
        if not 1 <= self.maximum_attempts <= 16:
            raise ValueError("judge maximum_attempts must lie in [1,16]")
        if (
            self.retry_backoff_seconds < 0
            or self.retry_maximum_seconds < self.retry_backoff_seconds
        ):
            raise ValueError("judge retry delays differ")
        if self.cache_max_entries < 0:
            raise ValueError("judge cache size must be non-negative")
        if self.transient_failure_window_size <= 0:
            raise ValueError("judge transient-failure window must be positive")
        if not 0.0 <= self.maximum_transient_failure_fraction <= 1.0:
            raise ValueError("judge transient-failure fraction differs")


def load_deepeyes_judge_service_config(
    path: str | Path,
    *,
    expected_file_sha256: str,
    require_launch_enabled: bool = True,
) -> DeepEyesJudgeServiceConfig:
    """Load the exact OpenRouter/DeepInfra service binding without a request."""

    source = Path(path)
    raw = source.read_bytes()
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_file_sha256:
        raise ValueError("judge service config SHA-256 differs")
    value = json.loads(raw)
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "tgvf.deepeyes-binary-judge-service.v1"
    ):
        raise ValueError("judge service schema differs")
    if require_launch_enabled and value.get("launch_enabled") is not True:
        raise ValueError("judge service binding is not launch-enabled")
    if type(value.get("launch_enabled")) is not bool:
        raise ValueError("judge service launch_enabled must be bool")
    service = value.get("service")
    failure = value.get("failure_policy")
    if not isinstance(service, Mapping) or not isinstance(failure, Mapping):
        raise ValueError("judge service/failure binding differs")
    if failure != {
        "transport_error": "bounded_retry_then_zero_with_threshold",
        "malformed_or_nonbinary": "zero_current_sample_after_audit",
        "auth_or_model_identity": "abort_run",
    }:
        raise ValueError("judge failure policy differs")
    return DeepEyesJudgeServiceConfig(
        file_sha256=actual_sha256,
        model=_required_text(value.get("model"), "service model"),
        base_url=_required_text(service.get("base_url"), "service base_url"),
        api_key_env=_required_text(service.get("api_key_env"), "API key env"),
        provider_only=tuple(service.get("provider_only", ())),
        allow_fallbacks=service.get("allow_fallbacks"),
        timeout_seconds=float(service.get("timeout_seconds")),
        maximum_concurrency=int(service.get("maximum_concurrency")),
        maximum_attempts=int(service.get("maximum_attempts")),
        retry_backoff_seconds=float(service.get("retry_backoff_seconds")),
        retry_maximum_seconds=float(service.get("retry_maximum_seconds")),
        cache_max_entries=int(service.get("cache_max_entries")),
        transient_failure_window_size=int(service.get("transient_failure_window_size")),
        maximum_transient_failure_fraction=float(
            service.get("maximum_transient_failure_fraction")
        ),
    )


@dataclass(frozen=True, slots=True)
class AsyncJudgeOutcome:
    verdict: bool
    calls: int
    retries: int
    cache_hit: int
    failure_kind: str | None
    latency_seconds: float
    usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if self.usage is not None and not isinstance(self.usage, JudgeUsage):
            raise TypeError("DeepEyes async judge usage must be JudgeUsage or None")


class _JudgeResponseIdentityError(JudgeSampleOutputError):
    """One completed response whose model metadata cannot be trusted."""


class AsyncDeepEyesOpenRouterJudge:
    """One-process bounded async transport used by the veRL reward worker."""

    def __init__(
        self,
        config: DeepEyesJudgeServiceConfig,
        *,
        request_json: Callable[..., Any] | None = None,
        sleeper: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._request_json_override = request_json
        self._sleeper = sleeper
        self._clock = clock
        self._semaphore = asyncio.Semaphore(config.maximum_concurrency)
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._failure_lock = asyncio.Lock()
        self._failure_windows: dict[int, list[int]] = {}
        self._next_failure_slot = 0
        self._session: object | None = None
        self._session_lock = asyncio.Lock()

    async def judge(self, request: DeepEyesBinaryJudgeRequest) -> AsyncJudgeOutcome:
        started = self._clock()
        async with self._cache_lock:
            cached = self._cache.get(request.request_id)
            if cached is not None:
                self._cache.move_to_end(request.request_id)
                return AsyncJudgeOutcome(
                    cached, 0, 0, 1, None, max(0.0, self._clock() - started)
                )

        attempts = 0
        last_transient_error: Exception | None = None
        failure_slot = await self._reserve_failure_slot()
        async with self._semaphore:
            while attempts < self.config.maximum_attempts:
                attempts += 1
                try:
                    payload = await self._request_json(request)
                    try:
                        verdict = self._parse_response(payload, request=request)
                    except _JudgeResponseIdentityError:
                        # The request itself remains pinned to the configured
                        # model and DeepInfra-only/no-fallback route.  A single
                        # completed response with inconsistent model metadata
                        # must not be consumed as reward, but neither should it
                        # terminate every other trajectory in the optimizer
                        # step.  Isolate it as a conservative zero and retain
                        # the billable usage for health/cost accounting.
                        usage = self._parse_response_usage(payload)
                        return AsyncJudgeOutcome(
                            False,
                            attempts,
                            attempts - 1,
                            0,
                            "completed_identity_mismatch",
                            max(0.0, self._clock() - started),
                            usage=usage,
                        )
                    except JudgeSampleOutputError:
                        # A completed response consumed real judge tokens even
                        # when its answer is malformed/non-binary.  Preserve
                        # that response's usage on the sample-local failure.
                        usage = self._parse_response_usage(payload)
                        return AsyncJudgeOutcome(
                            False,
                            attempts,
                            attempts - 1,
                            0,
                            "completed_invalid_output",
                            max(0.0, self._clock() - started),
                            usage=usage,
                        )
                    usage = self._parse_response_usage(payload)
                    async with self._cache_lock:
                        if self.config.cache_max_entries:
                            self._cache[request.request_id] = verdict
                            self._cache.move_to_end(request.request_id)
                            while len(self._cache) > self.config.cache_max_entries:
                                self._cache.popitem(last=False)
                    return AsyncJudgeOutcome(
                        verdict,
                        attempts,
                        attempts - 1,
                        0,
                        None,
                        max(0.0, self._clock() - started),
                        usage=usage,
                    )
                except JudgeSampleOutputError:
                    # No parseable completed response exists, so there is no
                    # truthful token usage to attach to this failed sample.
                    return AsyncJudgeOutcome(
                        False,
                        attempts,
                        attempts - 1,
                        0,
                        "completed_invalid_output",
                        max(0.0, self._clock() - started),
                    )
                except JudgeGlobalFailure:
                    raise
                except (TimeoutError, ConnectionError, RuntimeError) as error:
                    last_transient_error = error
                    if attempts == self.config.maximum_attempts:
                        break
                    delay = min(
                        self.config.retry_maximum_seconds,
                        self.config.retry_backoff_seconds * (2 ** (attempts - 1)),
                    )
                    await self._sleeper(delay)
        await self._record_transient_failure(
            failure_slot, last_error=last_transient_error
        )
        return AsyncJudgeOutcome(
            False,
            attempts,
            max(0, attempts - 1),
            0,
            "transient_exhausted",
            max(0.0, self._clock() - started),
        )

    async def _reserve_failure_slot(self) -> tuple[int, int]:
        async with self._failure_lock:
            slot = self._next_failure_slot
            self._next_failure_slot += 1
            window_size = self.config.transient_failure_window_size
            window, offset = divmod(slot, window_size)
            state = self._failure_windows.setdefault(window, [0, 0])
            state[0] += 1
            return window, offset

    async def _record_transient_failure(
        self,
        slot: tuple[int, int],
        *,
        last_error: Exception | None,
    ) -> None:
        window, _offset = slot
        async with self._failure_lock:
            state = self._failure_windows[window]
            state[1] += 1
            maximum = math.floor(
                self.config.transient_failure_window_size
                * self.config.maximum_transient_failure_fraction
            )
            if state[1] > maximum:
                detail = (
                    "unknown"
                    if last_error is None
                    else f"{type(last_error).__name__}: {last_error}"
                )
                raise JudgeGlobalFailure(
                    "DeepEyes transient judge failures exceed the bounded window; "
                    f"last_error={detail}"
                ) from last_error

    async def _request_json(
        self, request: DeepEyesBinaryJudgeRequest
    ) -> Mapping[str, object]:
        if self._request_json_override is not None:
            result = self._request_json_override(request, self._payload(request))
            if hasattr(result, "__await__"):
                result = await result
            if not isinstance(result, Mapping):
                raise JudgeSampleOutputError("judge HTTP body is not an object")
            return result
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise JudgeGlobalFailure(
                f"DeepEyes judge requires {self.config.api_key_env}"
            )
        try:
            import aiohttp
        except ImportError as error:  # pragma: no cover - pinned veRL owns aiohttp
            raise JudgeGlobalFailure("DeepEyes async judge requires aiohttp") from error
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }
        try:
            async with self._session_lock:
                session = self._session
                if session is None or getattr(session, "closed", True):
                    connector = aiohttp.TCPConnector(
                        limit=self.config.maximum_concurrency
                    )
                    session = aiohttp.ClientSession(
                        timeout=timeout, connector=connector
                    )
                    self._session = session
            async with session.post(
                self.config.base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=self._payload(request),
            ) as response:
                if response.status in {401, 403, 404}:
                    raise JudgeGlobalFailure(
                        f"DeepEyes judge global HTTP failure {response.status}"
                    )
                if response.status == 429 or response.status >= 500:
                    raise RuntimeError(
                        f"DeepEyes judge transient HTTP failure {response.status}"
                    )
                if response.status >= 400:
                    raise JudgeGlobalFailure(
                        f"DeepEyes judge permanent HTTP failure {response.status}"
                    )
                try:
                    value = await response.json()
                except (ValueError, TypeError) as error:
                    raise JudgeSampleOutputError(
                        "judge response is not JSON"
                    ) from error
        except asyncio.TimeoutError as error:
            raise TimeoutError("DeepEyes judge timeout") from error
        except aiohttp.ClientConnectionError as error:
            raise ConnectionError("DeepEyes judge connection failure") from error
        if not isinstance(value, Mapping):
            raise JudgeSampleOutputError("judge response is not an object")
        return value

    async def close(self) -> None:
        """Close the process-local pooled HTTP session during actor teardown."""

        async with self._session_lock:
            session, self._session = self._session, None
        if session is not None and not getattr(session, "closed", True):
            await session.close()

    def _payload(self, request: DeepEyesBinaryJudgeRequest) -> dict[str, object]:
        temperature = (
            DEEPEYES_VISUAL_JUDGE_TEMPERATURE
            if request.prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
            else DEEPEYES_THINKLITE_JUDGE_TEMPERATURE
        )
        return {
            "model": self.config.model,
            "messages": list(request.messages),
            "temperature": temperature,
            "top_p": DEEPEYES_BINARY_JUDGE_TOP_P,
            "max_tokens": DEEPEYES_BINARY_JUDGE_MAX_TOKENS,
            "provider": {
                "only": list(self.config.provider_only),
                "allow_fallbacks": self.config.allow_fallbacks,
            },
        }

    def _parse_response(
        self,
        value: Mapping[str, object],
        *,
        request: DeepEyesBinaryJudgeRequest,
    ) -> bool:
        if value.get("model") != self.config.model:
            raise _JudgeResponseIdentityError(
                "DeepEyes judge response model differs: "
                f"expected={self.config.model!r}, actual={value.get('model')!r}"
            )
        try:
            choices = value["choices"]
            first = choices[0]
            message = first["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise JudgeSampleOutputError("judge completion shape differs") from error
        try:
            return parse_binary_judge_output(content, prompt_kind=request.prompt_kind)
        except (TypeError, ValueError) as error:
            raise JudgeSampleOutputError("judge output is nonbinary") from error

    def _parse_response_usage(self, value: Mapping[str, object]) -> JudgeUsage:
        raw_usage = value.get("usage")
        if not isinstance(raw_usage, Mapping):
            raise JudgeGlobalFailure("DeepEyes judge response lacks required usage")
        try:
            cost = raw_usage["cost"]
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                raise TypeError("cost")
            usage = JudgeUsage(
                prompt_tokens=raw_usage["prompt_tokens"],
                completion_tokens=raw_usage["completion_tokens"],
                total_tokens=raw_usage["total_tokens"],
                cost_usd=float(cost),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise JudgeGlobalFailure(
                "DeepEyes judge response usage is invalid"
            ) from error
        if usage.prompt_tokens == 0:
            raise JudgeGlobalFailure(
                "DeepEyes completed judge response requires prompt tokens"
            )
        return usage


def _official_math_verify(reference_answer: str, candidate_answer: str) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError as error:  # pragma: no cover - formal environment gate
        raise RuntimeError("official ThinkLite route requires math-verify") from error
    return bool(verify(parse(reference_answer), parse(candidate_answer)))


def _config_value(
    config: object, path: tuple[str, ...], default: object = None
) -> object:
    current = config
    for part in path:
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            getter = getattr(current, "get", None)
            current = (
                getter(part, default)
                if callable(getter)
                else getattr(current, part, default)
            )
        if current is default:
            return default
    return current


def _crop_count(extra_info: Mapping[str, object], *, visual: bool) -> int:
    values = [
        _plain(extra_info[name])
        for name in ("crop_action_count", "native_crop_image_count")
        if name in extra_info
    ]
    if not values:
        if visual:
            raise ValueError("visual reward lacks native crop execution count")
        return 0
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("native crop execution count is invalid")
    if len(set(values)) != 1:
        raise ValueError("native crop execution counters disagree")
    if not visual and values[0] != 0:
        raise ValueError("ThinkLite reward unexpectedly contains a crop")
    return int(values[0])


_VISUAL_AUDIT_FIELDS = (
    "crop_call_count",
    "crop_action_count",
    "crop_boxes",
    "crop_area_fractions",
    "crop_first_call_iou",
    "crop_best_call_iou",
    "crop_best_gt_coverage",
    "crop_error_count",
    "crop_observation_token_spans",
    "native_original_image_count",
    "native_crop_image_count",
    "native_total_image_count",
    "response_token_count",
    "native_pixels_proven",
    "legacy_adapter_loaded",
    "observation_role",
    "observation_envelope",
)


def _json_list(value: object, name: str) -> list[object]:
    value = _plain(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"DeepEyes audit {name} must be a sequence")
    return list(value)


def _optional_unit_float(value: object, name: str) -> float | None:
    value = _plain(value)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DeepEyes audit {name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"DeepEyes audit {name} lies outside [0,1]")
    return result


def _trajectory_audit_fields(
    combined: Mapping[str, object],
    *,
    visual: bool,
    source: str,
    sample_id: str,
    task_kind: str,
    response_length: int,
    action_length: int,
) -> dict[str, object]:
    """Normalize the core-loop fields retained by upstream rollout dumps."""

    common: dict[str, object] = {
        "source": source,
        "sample_id": sample_id,
        "group_id": sample_id,
        "task_kind": task_kind,
        "response_length": response_length,
        "action_length": action_length,
    }
    if not visual:
        return {
            **common,
            "crop_call_count": 0,
            "crop_action_count": 0,
            "crop_boxes": [],
            "crop_area_fractions": [],
            "crop_area": [],
            "crop_first_call_iou": None,
            "crop_best_call_iou": None,
            "crop_best_gt_coverage": None,
            "first_iou": None,
            "best_iou": None,
            "gt_coverage": None,
            "crop_error_count": 0,
            "crop_observation_token_spans": [],
            "native_original_image_count": 0,
            "native_crop_image_count": 0,
            "native_total_image_count": 0,
            # This is the normalized generated-response count.  Visual rows
            # additionally prove it against the native loop sidecar below;
            # text-only rows derive the same value from the attention mask.
            "response_token_count": response_length,
            "native_pixels_proven": False,
            "legacy_adapter_loaded": False,
            "observation_role": "none",
            "observation_envelope": "none",
            "action_count": 0,
        }
    missing = [name for name in _VISUAL_AUDIT_FIELDS if name not in combined]
    if missing:
        raise ValueError(
            "visual reward lacks native trajectory audit fields: " + ",".join(missing)
        )
    integer_names = (
        "crop_call_count",
        "crop_action_count",
        "crop_error_count",
        "native_original_image_count",
        "native_crop_image_count",
        "native_total_image_count",
        "response_token_count",
    )
    integers = {name: _plain(combined[name]) for name in integer_names}
    if any(type(value) is not int or value < 0 for value in integers.values()):
        raise ValueError("native trajectory audit counters are invalid")
    if integers["response_token_count"] != response_length:
        raise ValueError("native/core response token counts disagree")
    if integers["native_original_image_count"] != 1:
        raise ValueError("native visual trajectory must retain one source image")
    if integers["native_crop_image_count"] != integers["crop_action_count"]:
        raise ValueError("native crop image/action counts disagree")
    if integers["native_total_image_count"] != (
        integers["native_original_image_count"] + integers["native_crop_image_count"]
    ):
        raise ValueError("native total image count differs")
    if _plain(combined["native_pixels_proven"]) is not True:
        raise ValueError("native pixel provenance was not proven")
    if _plain(combined["legacy_adapter_loaded"]) is not False:
        raise ValueError("legacy adapter unexpectedly appeared in PRL13")
    boxes = _json_list(combined["crop_boxes"], "crop_boxes")
    areas = _json_list(combined["crop_area_fractions"], "crop_area_fractions")
    spans = _json_list(
        combined["crop_observation_token_spans"],
        "crop_observation_token_spans",
    )
    call_count = integers["crop_call_count"]
    successful_actions = integers["crop_action_count"]
    error_count = integers["crop_error_count"]
    if successful_actions > call_count or error_count > call_count:
        raise ValueError("native Crop call/action/error counters disagree")
    if len(boxes) != successful_actions or len(areas) != successful_actions:
        raise ValueError("native successful-Crop audit sequence lengths disagree")
    # Spans describe rendered and policy-masked tool observations, not
    # successful crops.  Invalid bbox/parser responses therefore legitimately
    # add a span without adding a crop image/action.
    for index, span in enumerate(spans):
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= response_length
        ):
            raise ValueError(f"native Crop observation span {index} is invalid")
    first_iou = _optional_unit_float(
        combined["crop_first_call_iou"], "crop_first_call_iou"
    )
    best_iou = _optional_unit_float(
        combined["crop_best_call_iou"], "crop_best_call_iou"
    )
    coverage = _optional_unit_float(
        combined["crop_best_gt_coverage"], "crop_best_gt_coverage"
    )
    return {
        **common,
        **integers,
        "crop_boxes": boxes,
        "crop_area_fractions": areas,
        # ``crop_area`` intentionally aliases the complete per-action list;
        # no first/last/mean scalar is silently invented.
        "crop_area": areas,
        "crop_first_call_iou": first_iou,
        "crop_best_call_iou": best_iou,
        "crop_best_gt_coverage": coverage,
        "first_iou": first_iou,
        "best_iou": best_iou,
        "gt_coverage": coverage,
        "crop_observation_token_spans": spans,
        "native_pixels_proven": True,
        "legacy_adapter_loaded": False,
        "observation_role": _required_text(
            combined["observation_role"], "observation role"
        ),
        "observation_envelope": _required_text(
            combined["observation_envelope"], "observation envelope"
        ),
        "action_count": integers["crop_action_count"],
    }


try:
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ImportError:  # pragma: no cover - unit-only environment without veRL

    class RewardManagerBase:  # type: ignore[no-redef]
        def __init__(self, config: object, tokenizer: object, compute_score: object):
            self.config = config
            self.tokenizer = tokenizer
            self.compute_score = compute_score


class DeepEyesOfficialRewardManager(RewardManagerBase):
    """Hydra-importable reward manager for pinned veRL's async RewardLoop."""

    def __init__(
        self,
        config: object,
        tokenizer: object,
        compute_score: object = None,
        reward_router_address: str | None = None,
        reward_model_tokenizer: object | None = None,
        *,
        judge_transport: AsyncDeepEyesOpenRouterJudge | None = None,
        math_verify: Callable[[str, str], bool] = _official_math_verify,
        trajectory_id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        del reward_router_address, reward_model_tokenizer
        super().__init__(config, tokenizer, compute_score)
        if judge_transport is None:
            service_path = _required_text(
                _config_value(
                    config,
                    ("reward", "deepeyes_official", "judge_service_config_path"),
                ),
                "judge service path",
            )
            service_sha = _required_text(
                _config_value(
                    config,
                    ("reward", "deepeyes_official", "judge_service_config_sha256"),
                ),
                "judge service SHA-256",
            )
            service = load_deepeyes_judge_service_config(
                service_path, expected_file_sha256=service_sha
            )
            judge_transport = AsyncDeepEyesOpenRouterJudge(service)
        if not callable(math_verify) or not callable(trajectory_id_factory):
            raise TypeError("DeepEyes verifier/trajectory factory must be callable")
        self.judge_transport = judge_transport
        self.math_verify = math_verify
        self.trajectory_id_factory = trajectory_id_factory

    async def run_single(self, data: object) -> dict[str, object]:
        if len(data) != 1:
            raise ValueError("DeepEyes reward manager requires one trajectory")
        item = data[0]
        response_ids = item.batch["responses"]
        response_width = response_ids.shape[-1]
        valid_length = int(item.batch["attention_mask"][-response_width:].sum())
        response_mask = item.batch.get("response_mask")
        action_length = (
            valid_length
            if response_mask is None
            else int(response_mask[:valid_length].sum())
        )
        solution = self.tokenizer.decode(
            response_ids[:valid_length], skip_special_tokens=True
        )
        non_tensor = item.non_tensor_batch
        source = _required_text(non_tensor["data_source"], "data_source")
        reward_model = _plain(non_tensor["reward_model"])
        if not isinstance(reward_model, Mapping):
            raise ValueError("DeepEyes reward_model row differs")
        reference = _required_text(reward_model.get("ground_truth"), "ground truth")
        extra = _plain(non_tensor.get("extra_info", {}))
        tool_extra = _plain(non_tensor.get("tool_extra_fields", {}))
        if not isinstance(extra, Mapping) or not isinstance(tool_extra, Mapping):
            raise ValueError("DeepEyes reward extra fields differ")
        combined = {**extra, **tool_extra}
        question = _required_text(combined.get("question"), "question")
        sample_id = _required_text(combined.get("sample_id"), "sample_id")
        task_kind = _required_text(combined.get("task_kind"), "task_kind")
        trajectory_id = combined.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            # Pinned veRL does not forward rollout_n to RewardLoop.  A fresh
            # UUID prevents cross-trajectory cache dedup; it is returned in
            # reward_extra_info so the retained rollout can bind it later.
            trajectory_id = f"{sample_id}/{self.trajectory_id_factory()}"

        visual = source in {"vstar", "arxivqa"}
        if not visual and source != "thinklite":
            raise ValueError("DeepEyes reward source is unsupported")
        crop_count = _crop_count(combined, visual=visual)
        audit = _trajectory_audit_fields(
            combined,
            visual=visual,
            source=source,
            sample_id=sample_id,
            task_kind=task_kind,
            response_length=valid_length,
            action_length=action_length,
        )
        if visual:
            result = await self._score_visual(
                trajectory_id=trajectory_id,
                sample_id=sample_id,
                question=question,
                reference=reference,
                response=solution,
                task_kind=task_kind,
                crop_count=crop_count,
                audit=audit,
            )
        else:
            result = await self._score_thinklite(
                trajectory_id=trajectory_id,
                sample_id=sample_id,
                question=question,
                reference=reference,
                response=solution,
                task_kind=task_kind,
                audit=audit,
            )
        return {
            "reward_score": result["score"],
            "reward_extra_info": _batch_safe_reward_extra_info(result),
        }

    async def _score_visual(self, **value: object) -> dict[str, object]:
        response = str(value["response"])
        extraction = extract_visual_answer(response)
        request = DeepEyesBinaryJudgeRequest.build(
            trajectory_id=str(value["trajectory_id"]),
            sample_id=str(value["sample_id"]),
            question=str(value["question"]),
            reference_answer=str(value["reference"]),
            candidate_answer=extraction.answer or "[NO VALID FINAL ANSWER]",
            task_kind=str(value["task_kind"]),
            prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
        )
        judged = await self.judge_transport.judge(request)
        too_long = len(extraction.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
        accuracy = int(judged.verdict and not too_long)
        format_penalty = -1 if too_long else extraction.format_penalty
        crop_count = int(value["crop_count"])
        conditional_tool = int(accuracy == 1 and crop_count > 0)
        score = (
            DEEPEYES_VISUAL_ANSWER_WEIGHT * accuracy
            + DEEPEYES_VISUAL_FORMAT_WEIGHT * format_penalty
            + DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT * conditional_tool
        )
        if judged.failure_kind is not None:
            score, accuracy, conditional_tool = 0.0, 0, 0
        return self._result(
            score=score,
            accuracy=accuracy,
            format_penalty=format_penalty,
            conditional_tool=conditional_tool,
            answer=extraction.answer,
            crop_count=crop_count,
            trajectory_id=str(value["trajectory_id"]),
            judge=judged,
            visual_requested=1,
            thinklite_requested=0,
            route="qwen2.5_72b_every_visual_trajectory",
            audit=value["audit"],
        )

    async def _score_thinklite(self, **value: object) -> dict[str, object]:
        task_kind = str(value["task_kind"])
        if task_kind in {"open", "mcq"}:
            extraction = extract_thinklite_answer(str(value["response"]))
            request = DeepEyesBinaryJudgeRequest.build(
                trajectory_id=str(value["trajectory_id"]),
                sample_id=str(value["sample_id"]),
                question=str(value["question"]),
                reference_answer=str(value["reference"]),
                candidate_answer=extraction.answer or "[NO VALID FINAL ANSWER]",
                task_kind=task_kind,
                prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
            )
            judged = await self.judge_transport.judge(request)
            too_long = len(extraction.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
            accuracy = int(judged.verdict and not too_long)
            format_penalty = -1 if too_long else extraction.format_penalty
            score = (
                DEEPEYES_THINKLITE_ANSWER_WEIGHT * accuracy
                + DEEPEYES_THINKLITE_FORMAT_WEIGHT * format_penalty
            )
            if judged.failure_kind is not None:
                score, accuracy = 0.0, 0
            return self._result(
                score=score,
                accuracy=accuracy,
                format_penalty=format_penalty,
                conditional_tool=0,
                answer=extraction.answer,
                crop_count=0,
                trajectory_id=str(value["trajectory_id"]),
                judge=judged,
                visual_requested=0,
                thinklite_requested=1,
                route="thinklite_boxed_qwen2.5_72b",
                audit=value["audit"],
            )
        if task_kind != "math":
            raise ValueError("ThinkLite task_kind must be math, open, or mcq")
        extraction = extract_thinklite_answer(str(value["response"]))
        rule_correct = False
        if extraction.answer:
            try:
                rule_correct = self.math_verify(
                    str(value["reference"]), extraction.answer
                )
            except Exception:
                rule_correct = False
            if type(rule_correct) is not bool:
                raise TypeError("math_verify must return bool")
        judged: AsyncJudgeOutcome | None = None
        if rule_correct:
            accuracy, route = 1, "math_verify"
        elif extraction.answer:
            request = DeepEyesBinaryJudgeRequest.build(
                trajectory_id=str(value["trajectory_id"]),
                sample_id=str(value["sample_id"]),
                question=str(value["question"]),
                reference_answer=str(value["reference"]),
                candidate_answer=extraction.answer,
                task_kind=str(value["task_kind"]),
                prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
            )
            judged = await self.judge_transport.judge(request)
            accuracy = int(judged.verdict)
            route = "math_verify_then_qwen2.5_72b"
        else:
            accuracy, route = 0, "missing_boxed_answer"
        score = (
            DEEPEYES_THINKLITE_ANSWER_WEIGHT * accuracy
            + DEEPEYES_THINKLITE_FORMAT_WEIGHT * extraction.format_penalty
        )
        if judged is not None and judged.failure_kind is not None:
            score, accuracy = 0.0, 0
        return self._result(
            score=score,
            accuracy=accuracy,
            format_penalty=extraction.format_penalty,
            conditional_tool=0,
            answer=extraction.answer,
            crop_count=0,
            trajectory_id=str(value["trajectory_id"]),
            judge=judged,
            visual_requested=0,
            thinklite_requested=int(judged is not None),
            route=route,
            audit=value["audit"],
        )

    @staticmethod
    def _result(
        *,
        score: float,
        accuracy: int,
        format_penalty: int,
        conditional_tool: int,
        answer: str,
        crop_count: int,
        trajectory_id: str,
        judge: AsyncJudgeOutcome | None,
        visual_requested: int,
        thinklite_requested: int,
        route: str,
        audit: object,
    ) -> dict[str, object]:
        if not isinstance(audit, Mapping):
            raise TypeError("DeepEyes trajectory audit must be a mapping")
        requested = visual_requested + thinklite_requested
        return {
            **audit,
            "score": float(score),
            "acc": accuracy,
            "format_penalty": format_penalty,
            "conditional_tool": conditional_tool,
            "answer_length": len(answer),
            "successful_crop_count": crop_count,
            "judge_requested": requested,
            "visual_judge_requested": visual_requested,
            "thinklite_fallback_judge_requested": thinklite_requested,
            "judge_calls": 0 if judge is None else judge.calls,
            "judge_cache_hits": 0 if judge is None else judge.cache_hit,
            "judge_retries": 0 if judge is None else judge.retries,
            "judge_latency_seconds": 0.0 if judge is None else judge.latency_seconds,
            "judge_failures": int(judge is not None and judge.failure_kind is not None),
            "judge_failure_kind": "none"
            if judge is None or judge.failure_kind is None
            else judge.failure_kind,
            "judge_route": route,
            "trajectory_id": trajectory_id,
        }


__all__ = [
    "DEEPEYES_VERL_REWARD_MANAGER_CLASS",
    "DEEPEYES_VERL_REWARD_SCHEMA",
    "AsyncDeepEyesOpenRouterJudge",
    "AsyncJudgeOutcome",
    "DeepEyesJudgeServiceConfig",
    "DeepEyesOfficialRewardManager",
    "load_deepeyes_judge_service_config",
]
