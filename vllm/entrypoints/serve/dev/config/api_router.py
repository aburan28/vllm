# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/update_engine_args")
async def update_engine_args(raw_request: Request) -> JSONResponse:
    """Hot-update allowlisted engine args at runtime.

    The JSON body maps field names to new values, e.g.
    ``{"max_num_seqs": 8, "long_prefill_token_threshold": 2048}``.
    Currently covers scheduler knobs (``max_num_seqs``,
    ``max_num_batched_tokens``, ``long_prefill_token_threshold``); capacity
    fields may only be lowered or restored, never raised past their startup
    values, since worker-side buffers were sized by them. Unknown fields and
    invalid values are rejected without applying anything.
    """

    try:
        overrides = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904
    if not isinstance(overrides, dict) or not overrides:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Body must be a non-empty JSON object of field overrides",
        )

    engine = engine_client(raw_request)

    try:
        applied = await engine.update_engine_args(overrides)
        return JSONResponse(
            content={"applied": applied},
            status_code=HTTPStatus.OK.value,
        )
    except ValueError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to update engine args")
        return JSONResponse(
            content={"error": f"Failed to update engine args: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


def attach_router(app: FastAPI):
    app.include_router(router)
