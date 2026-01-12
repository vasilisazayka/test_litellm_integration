from __future__ import annotations

"""
Smoke-test LLM models via LiteLLM — async orchestrator with parallel sync tests.

What it does
────────────
• Orchestrates with asyncio (no explicit ThreadPoolExecutor)
• Runs sync tests (completion, streaming, structured output, tool calling, multimodal) in parallel via asyncio.to_thread
• Runs async completion test too
• Async model discovery (httpx)
• Wall-clock timeouts via asyncio.wait_for; HTTP timeouts via LiteLLM/httpx
• Bounded concurrency; HTML summary with ⏱/❌/✅
"""

import asyncio
from html import escape
from typing import Dict, List, Tuple, Callable, Awaitable, Union, Optional

import httpx
import base64
import pandas as pd
from litellm import completion, acompletion
from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
from litellm.types.utils import ModelResponseStream


srv = GenAIHubOrchestrationConfig()

# ─────────────────────────────────────────  CONFIG  ──────────────────────────────────────────
HTTP_TIMEOUT = 110        # seconds; passed to LiteLLM via request_timeout
EXECUTION_TIMEOUT = 120   # wall-clock timeout per test case
MAX_CONCURRENCY = 8      # overall async concurrency cap

# ───────────────────────────────────  DISCOVER AVAILABLE MODELS  ─────────────────────────────
async def check_model_availability_async() -> Dict[str, List[str]]:
    url = srv.base_url + "/lm/scenarios/foundation-models/models"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(url, headers=srv.headers)
        resp.raise_for_status()
        data = resp.json()

    matching_models: Dict[str, List[str]] = {}
    for model in data.get("resources", []):
        if not any(s.get("executableId") == "orchestration" for s in model.get("allowedScenarios", [])):
            continue
        if model["model"] == "gpt-35-turbo" or model["model"] == "gpt-35-turbo-0125":
            continue
        for version in model.get("versions", []):
            if "text-generation" not in version.get("capabilities", []):
                continue
            matching_models.setdefault(model["model"], []).append(version["name"])
    return matching_models

# ───────────────────────────────────────────  MESSAGES  ──────────────────────────────────────
BASE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather like today in Berlin?"},
]



def encode_image_base64_from_url(image_url: str) -> str:
    url = srv.base_url + "/lm/scenarios/foundation-models/models"
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.get(image_url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')


image_url = "https://www.zastavki.com/pictures/originals/2023/_Beautiful_view_of_the_Eiffel_Tower__Paris._France_165775_.jpg"
image_base64 = encode_image_base64_from_url(image_url=image_url)

def make_multimodal_messages() -> List[dict]:
    # Minimal public image; replace with a relevant one within your network if needed.
    return [
        {"role": "system", "content": "You are a helpful assistant that can interpret images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the weather-related elements you can infer from this image, briefly."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        },
    ]

def _model_id(name: str, version: str) -> str:
    """Prepend 'sap/' to every model name (add version if your router needs it)."""
    return f"sap/{name}"

# ─────────────────────────────────────  SYNC TESTS (kept)  ───────────────────────────────────
def test_completion_sync(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        resp = completion(
            model=model,
            messages=BASE_MESSAGES,
            request_timeout=HTTP_TIMEOUT,
            custom_llm_provider="sap",
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)

def test_streaming_sync(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        stream = completion(
            model=model,
            messages=BASE_MESSAGES,
            stream=True,
            request_timeout=HTTP_TIMEOUT,
        )
        for chunk in stream:
            if not isinstance(chunk, ModelResponseStream):
                return False, f"Unexpected chunk type [{type(chunk)=}]"
        return True, None
    except Exception as exc:
        return False, str(exc)

def test_structured_output_sync(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temperature": {"type": "number"},
                    "condition": {"type": "string"},
                },
                "required": ["city", "temperature", "condition"],
                "additionalProperties": False,
            },
        },
    }
    try:
        resp = completion(
            model=model,
            messages=BASE_MESSAGES,
            response_format=response_format,
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a short weather summary for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]

def test_tool_calling_sync(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        resp = completion(
            model=model,
            messages=BASE_MESSAGES,
            tools=_TOOL_SCHEMAS,
            # tool_choice="auto",
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)

def test_multimodal_sync(name: str, version: str) -> Tuple[bool, Optional[str]]:
    """
    Multimodal input (image + text) via sync completion.
    If the model/route doesn't support vision, the provider may return an error — we surface it.
    """
    model = _model_id(name, version)
    try:
        resp = completion(
            model=model,
            messages=make_multimodal_messages(),
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload for multimodal"
    except Exception as exc:
        return False, str(exc)

# ─────────────────────────────────────  ASYNC TESTS  ─────────────────────────────────────────
async def test_acompletion_async(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        resp = await acompletion(
            model=model,
            messages=BASE_MESSAGES,
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)


async def test_streaming_async(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        stream = await acompletion(
            model=model,
            messages=BASE_MESSAGES,
            stream=True,
            request_timeout=HTTP_TIMEOUT,
        )
        async for chunk in stream:
            if not isinstance(chunk, ModelResponseStream):
                return False, f"Unexpected chunk type [{type(chunk)=}]"
        return True, None
    except Exception as exc:
        return False, str(exc)

async def test_structured_output_async(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temperature": {"type": "number"},
                    "condition": {"type": "string"},
                },
                "required": ["city", "temperature", "condition"],
                "additionalProperties": False,
            },
        },
    }
    try:
        resp = await acompletion(
            model=model,
            messages=BASE_MESSAGES,
            response_format=response_format,
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)

async def test_tool_calling_async(name: str, version: str) -> Tuple[bool, Optional[str]]:
    model = _model_id(name, version)
    try:
        resp = await acompletion(
            model=model,
            messages=BASE_MESSAGES,
            tools=_TOOL_SCHEMAS,
            # tool_choice="auto",
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload"
    except Exception as exc:
        return False, str(exc)

async def test_multimodal_async(name: str, version: str) -> Tuple[bool, Optional[str]]:
    """
    Multimodal input (image + text) via sync completion.
    If the model/route doesn't support vision, the provider may return an error — we surface it.
    """
    model = _model_id(name, version)
    try:
        resp = await acompletion(
            model=model,
            messages=make_multimodal_messages(),
            request_timeout=HTTP_TIMEOUT,
        )
        ok = bool(resp) and "choices" in resp and len(resp["choices"]) > 0
        return ok, None if ok else "Unexpected response payload for multimodal"
    except Exception as exc:
        return False, str(exc)


# Registry: sync tests run via to_thread; async tests awaited directly
TestFn = Callable[..., Union[Tuple[bool, Optional[str]], Awaitable[Tuple[bool, Optional[str]]]]]
TESTS: Dict[str, TestFn] = {
    # Sync tests (parallelized via asyncio.to_thread)
    "completion_sync": test_completion_sync,
    "acompletion_async": test_acompletion_async,
    "streaming_sync": test_streaming_sync,
    "streaming_async": test_streaming_async,
    "structured_output_sync": test_structured_output_sync,
    "structured_output_async": test_structured_output_async,
    "tool_calling_sync": test_tool_calling_sync,
    "tool_calling_async": test_tool_calling_async,
    "multimodal_sync": test_multimodal_sync,
    "multimodal_async": test_multimodal_async,
}

# ─────────────────────────────  TIMEOUT DETECTION HELPER  ────────────────────────────────────
def _is_timeout(exc_or_msg: object) -> bool:
    if isinstance(
        exc_or_msg,
        (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.TimeoutException,
            asyncio.TimeoutError,
        ),
    ):
        return True
    msg = str(exc_or_msg).lower()
    return any(k in msg for k in ("timeout", "timed out", "readtimeout", "connection aborted"))

# ───────────────────────────────────  EXECUTION WITH TIMEOUT  ────────────────────────────────
async def run_test_case_async(
    fn: TestFn,
    model: str,
    version: str,
    *,
    timeout: int = EXECUTION_TIMEOUT,
) -> Tuple[bool, Optional[str], bool]:
    """Run a sync or async test with a wall-clock timeout; sync fns run in threads via asyncio.to_thread."""
    async def _call():
        if asyncio.iscoroutinefunction(fn):
            return await fn(model, version)
        return await asyncio.to_thread(fn, model, version)

    try:
        success, error = await asyncio.wait_for(_call(), timeout=timeout)
        timed = (not success) and _is_timeout(error)
        return success, error, timed
    except asyncio.TimeoutError:
        return False, "Timed out", True
    except Exception as exc:
        return False, str(exc), _is_timeout(exc)

# ────────────────────────────────────  PARALLEL GATHER (ASYNC)  ──────────────────────────────
async def gather_results_async() -> pd.DataFrame:
    records: List[Dict[str, str | bool]] = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    MODEL_VERSIONS = await check_model_availability_async()

    async def run_one(tc_name: str, fn: TestFn, model: str, version: str):
        async with semaphore:
            success, error, timed = await run_test_case_async(fn, model, version)
            records.append(
                {
                    "model": model,
                    "version": version,
                    "test_case": tc_name,
                    "success": success,
                    "error": error if not success else None,
                    "timed_out": timed,
                }
            )

    tasks: List[asyncio.Task] = []
    for model, versions in MODEL_VERSIONS.items():
        for version in versions[:1]:  # only first version for now
            for tc_name, fn in TESTS.items():
                tasks.append(asyncio.create_task(run_one(tc_name, fn, model, version)))

    for t in asyncio.as_completed(tasks):
        await t

    return pd.DataFrame(records).sort_values(["model", "version"]).reset_index(drop=True)

# ─────────────────────────────────────  REPORT GENERATION  ───────────────────────────────────
def badge(success: bool, error: Optional[str], timed: bool) -> str:
    if timed:
        return "⏱"
    if success:
        return "✅"
    else:
        if error.startswith("litellm.UnsupportedParamsError"):
            return f'<span title="{escape(error or "")}">⚠️</span>'
        elif 'not support' in error or 'not supported' in error:
            return f'<span title="{escape(error or "")}">🔴</span>'
        return f'<span title="{escape(error or "")}">❌</span>'

def build_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No results.</p>"

    succ = df.pivot(index=["model", "version"], columns="test_case", values="success")
    err  = df.pivot(index=["model", "version"], columns="test_case", values="error")
    to   = df.pivot(index=["model", "version"], columns="test_case", values="timed_out")

    cases = list(TESTS.keys())
    headings = {
        "completion_sync": "Completion (sync)",
        "streaming_sync": "Streaming (sync)",
        "structured_output_sync": "Structured Output (sync)",
        "tool_calling_sync": "Tool Calling (sync)",
        "multimodal_sync": "Image Input (sync)",
        "acompletion_async": "Completion (async)",
        "streaming_async": "Streaming (async)",
        "structured_output_async": "Structured Output (async)",
        "tool_calling_async": "Tool Calling (async)",
        "multimodal_async": "Image Input (async)",
    }

    rows = []
    for (model, version), _ in succ.iterrows():
        row_cells = [
            escape(model),
            escape(version),
            *[
                badge(
                    bool(succ.loc[(model, version), c]),
                    err.loc[(model, version), c],
                    bool(to.loc[(model, version), c]),
                )
                for c in cases
            ],
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in row_cells) + "</tr>")

    return (
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead><tr><th>Model</th><th>Version</th>"
        + "".join(f"<th>{headings.get(c, c)}</th>" for c in cases)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

# ─────────────────────────────────────────────  MAIN  ────────────────────────────────────────
async def amain() -> None:
    df = await gather_results_async()
    html = build_html(df)
    with open("test_results.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("✓ Test run finished – open test_results.html to view the report.")

if __name__ == "__main__":
    asyncio.run(amain())
