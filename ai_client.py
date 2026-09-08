"""
Single entry point for every LLM call in the backend.

All AI features (chat, feasibility advisory, expense summaries, goal narratives,
expense-impact analysis, the health probe) go through `ai_complete()` /
`ai_complete_json()` here. Requests are sent to OpenRouter's chat-completions API,
which proxies Google's Gemini models on a quota pool separate from the direct
Google SDK (we hit the direct Gemini free-tier limit).

Response shape is OpenAI-style: choices[0].message.content
(NOT the Google SDK's candidates[...].content.parts[...]).
"""

import os
import json
import datetime

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 45


def _key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _primary_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")


def _free_model() -> str:
    # Used only as a rate-limit fallback. "" disables it.
    return os.getenv("OPENROUTER_FREE_MODEL", "google/gemini-2.0-flash-exp:free").strip()


def key_configured() -> bool:
    return len(_key()) > 10


class AIError(RuntimeError):
    """Carries a safe, non-secret reason code for the health endpoint / logs."""

    def __init__(self, message: str, *, status: int | None = None, reason: str = "network_failure"):
        super().__init__(message)
        self.status = status
        self.reason = reason


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
        # OpenRouter uses these for its dashboard / rankings; optional.
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Vyapar-Mitra"),
    }


def _one_call(model: str, messages: list, json_mode: bool, timeout: int) -> str:
    """One POST to OpenRouter. Returns the message text or raises AIError."""
    payload: dict = {"model": model, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(OPENROUTER_URL, headers=_headers(), data=json.dumps(payload), timeout=timeout)
    except requests.Timeout:
        raise AIError("request timed out", reason="provider_timeout")
    except requests.RequestException as e:
        raise AIError(f"network error: {e}", reason="network_failure")

    if resp.status_code in (401, 403):
        raise AIError("auth rejected by OpenRouter", status=resp.status_code, reason="invalid_credentials")
    if resp.status_code == 429:
        raise AIError("rate limited", status=429, reason="quota_exceeded")
    if resp.status_code == 404:
        raise AIError(f"model not found: {model}", status=404, reason="invalid_model")
    if resp.status_code == 400 and json_mode and "response_format" in resp.text:
        # This model/route doesn't accept response_format — signal caller to retry without it.
        raise AIError("response_format unsupported", status=400, reason="json_mode_unsupported")
    if not resp.ok:
        raise AIError(f"HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code, reason="provider_error")

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise AIError(f"malformed response: {e}", reason="malformed_response")

    if not text or not text.strip():
        raise AIError("empty completion", reason="malformed_response")
    return text


def ai_complete(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    allow_free_fallback: bool = True,
) -> tuple[str, str]:
    """
    Returns (text, model_used). Raises AIError on failure.

    On a 429 from the primary model, retries once on OPENROUTER_FREE_MODEL.
    If json_mode is rejected by the route (400), retries once without it.
    """
    if not key_configured():
        raise AIError("OPENROUTER_API_KEY not configured", reason="invalid_credentials")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    models = [_primary_model()]
    free = _free_model()
    if allow_free_fallback and free and free != models[0]:
        models.append(free)

    last: AIError | None = None
    for model in models:
        try:
            return _one_call(model, messages, json_mode, timeout), model
        except AIError as e:
            last = e
            if e.reason == "json_mode_unsupported":
                # retry the same model, no response_format
                try:
                    return _one_call(model, messages, False, timeout), model
                except AIError as e2:
                    last = e2
            if e.reason == "quota_exceeded":
                continue  # try the free model
            if e.reason == "invalid_credentials":
                raise  # no point retrying
            # other errors: try next model if any, else raise below
    raise last or AIError("all models failed", reason="network_failure")


def strip_json_fences(text: str) -> str:
    """Models often wrap JSON in ```json ... ``` even when asked not to."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t.lstrip("`")
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def ai_complete_json(
    prompt: str,
    *,
    system: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict, str]:
    """
    Returns (parsed_obj, model_used). Tries OpenRouter's response_format=json_object;
    always also strips code fences and, on a parse failure, does one repair retry
    with an explicit "JSON only" instruction.
    """
    text, model = ai_complete(prompt, system=system, json_mode=True, timeout=timeout)
    try:
        return json.loads(strip_json_fences(text)), model
    except json.JSONDecodeError:
        pass

    text, model = ai_complete(
        prompt + "\n\nReturn ONLY the raw JSON object. No markdown, no code fences, no commentary.",
        system=system,
        json_mode=False,
        timeout=timeout,
    )
    return json.loads(strip_json_fences(text)), model  # let a 2nd failure bubble up


def probe() -> dict:
    """Health check — exercises the real request path."""
    now = datetime.datetime.now().isoformat()
    base = {"provider": "openrouter", "model": _primary_model(), "checkedAt": now}
    if not key_configured():
        return {**base, "status": "not_configured", "safeReason": "invalid_credentials"}
    try:
        text, model = ai_complete("Reply with exactly: OK", allow_free_fallback=False, timeout=15)
        ok = "OK" in text.strip().upper() or len(text.strip()) > 0
        return {**base, "model": model, "status": "connected" if ok else "unavailable",
                "safeReason": "connected" if ok else "malformed_response"}
    except AIError as e:
        status = "not_configured" if e.reason == "invalid_credentials" else "unavailable"
        return {**base, "status": status, "safeReason": e.reason}
