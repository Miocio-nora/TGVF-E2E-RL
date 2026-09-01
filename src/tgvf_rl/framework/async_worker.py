"""Cancellation-safe boundaries for side-effecting synchronous worker calls."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar


_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_side_effecting_in_thread(
    function: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    """Reach a stable synchronous boundary before propagating cancellation.

    ``asyncio.to_thread`` cannot stop an already-running worker.  Callers use
    this helper only when the synchronous function can consume or mutate exact
    request state; releasing that state while the worker continues would race
    cleanup.  Pure or cancellable async waits should not use this helper.
    """

    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:
            if cancellation is None:
                raise
            break
    if cancellation is not None:
        if not worker.cancelled():
            worker_error = worker.exception()
            if worker_error is not None:
                cancellation.add_note(
                    "side-effecting worker also failed while draining cancellation: "
                    f"{worker_error!r}"
                )
        raise cancellation
    return worker.result()


__all__ = ["run_side_effecting_in_thread"]
