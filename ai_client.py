"""
Single entry point for every LLM call in the backend.

Every AI feature (chat, feasibility advisory, expense summaries, goal narratives,
expense-impact analysis, the health probe) goes through `ai_complete()` /
`ai_complete_json()` here.

Provider is auto-selected from env, in this order:
  1. GEMINI_API_KEY      -> Google Generative Language REST API (direct)
  2. OPENROUTER_API_KEY  -> OpenRouter chat-completions (proxies Gemini on a separate pool)

Both are reached with plain `requests` — no vendor SDKs.
"""

import os
import json
import datetime

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_TIMEOUT = 45


# ── provider / key selection ────────────────────────────────────────────────
def _gemini_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def _openrouter_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _provider() -> str | None:
    if len(_gemini_key()) > 10:
        return "gemini"
    if len(_openrouter_key()) > 10:
        return "openrouter"
    return None


def key_configured() -> bool:
    return _provider() is not None


def _primary_model() -> str:
    if _provider() == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")


def _free_model() -> str:
    # OpenRouter-only rate-limit fallback. "" disables it.
    return os.getenv("OPENROUTER_FREE_MODEL", "google/gemini-2.0-flash-exp:free").strip()


class AIError(RuntimeError):
    """Carries a safe, non-secret reason code for the health endpoint / logs."""

    def __init__(self, message: str, *, status: int | None = None, reason: str = "network_failure"):
        super().__init__(message)
        self.status = status
        self.reason = reason


def _classify(status: int, body: str) -> str:
    if status in (401, 403):
        return "invalid_credentials"
    if status == 400 and "API_KEY_INVALID" in body:
        return "invalid_credentials"
    if status == 429:
        return "quota_exceeded"
    if status == 404:
        return "invalid_model"
    return "provider_error"


# ── OpenRouter ─────────────────────────────────────────────────────────────
def _openrouter_call(model: str, system: str | None, prompt: str, json_mode: bool, timeout: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict = {"model": model, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {_openrouter_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Vyapar-Mitra"),
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.Timeout:
        raise AIError("request timed out", reason="provider_timeout")
    except requests.RequestException as e:
        raise AIError(f"network error: {e}", reason="network_failure")

    if r.status_code == 400 and json_mode and "response_format" in r.text:
        raise AIError("response_format unsupported", status=400, reason="json_mode_unsupported")
    if not r.ok:
        raise AIError(f"HTTP {r.status_code}: {r.text[:200]}", status=r.status_code,
                      reason=_classify(r.status_code, r.text))
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise AIError(f"malformed response: {e}", reason="malformed_response")
    if not text or not text.strip():
        raise AIError("empty completion", reason="malformed_response")
    return text


# ── Google Gemini (direct REST) ────────────────────────────────────────────
def _gemini_call(model: str, system: str | None, prompt: str, json_mode: bool, timeout: int) -> str:
    url = GEMINI_URL.format(model=model)
    payload: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
    try:
        r = requests.post(url, params={"key": _gemini_key()},
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=timeout)
    except requests.Timeout:
        raise AIError("request timed out", reason="provider_timeout")
    except requests.RequestException as e:
        raise AIError(f"network error: {e}", reason="network_failure")

    if not r.ok:
        raise AIError(f"HTTP {r.status_code}: {r.text[:200]}", status=r.status_code,
                      reason=_classify(r.status_code, r.text))
    try:
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            fb = (data.get("promptFeedback") or {}).get("blockReason")
            raise AIError(f"no candidates (blockReason={fb})", reason="malformed_response")
        parts = cands[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise AIError(f"malformed response: {e}", reason="malformed_response")
    if not text or not text.strip():
        raise AIError("empty completion", reason="malformed_response")
    return text


def _one_call(model: str, system: str | None, prompt: str, json_mode: bool, timeout: int) -> str:
    if _provider() == "gemini":
        return _gemini_call(model, system, prompt, json_mode, timeout)
    return _openrouter_call(model, system, prompt, json_mode, timeout)


# ── public API ─────────────────────────────────────────────────────────────
def ai_complete(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    allow_free_fallback: bool = True,
) -> tuple[str, str]:
    """Returns (text, model_used). Raises AIError on failure."""
    prov = _provider()
    if prov is None:
        raise AIError("no AI key configured (set GEMINI_API_KEY or OPENROUTER_API_KEY)",
                      reason="invalid_credentials")

    models = [_primary_model()]
    if prov == "openrouter" and allow_free_fallback:
        free = _free_model()
        if free and free != models[0]:
            models.append(free)

    last: AIError | None = None
    for model in models:
        try:
            return _one_call(model, system, prompt, json_mode, timeout), model
        except AIError as e:
            last = e
            if e.reason == "json_mode_unsupported":
                try:
                    return _one_call(model, system, prompt, False, timeout), model
                except AIError as e2:
                    last = e2
            if e.reason == "quota_exceeded":
                continue  # try next model (openrouter free fallback)
            if e.reason == "invalid_credentials":
                raise
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
    """Returns (parsed_obj, model_used). JSON mode + fence-strip + one repair retry."""
    text, model = ai_complete(prompt, system=system, json_mode=True, timeout=timeout)
    try:
        return json.loads(strip_json_fences(text)), model
    except json.JSONDecodeError:
        pass
    text, model = ai_complete(
        prompt + "\n\nReturn ONLY the raw JSON object. No markdown, no code fences, no commentary.",
        system=system, json_mode=False, timeout=timeout,
    )
    return json.loads(strip_json_fences(text)), model


def probe() -> dict:
    """Health check — exercises the real request path."""
    now = datetime.datetime.now().isoformat()
    prov = _provider() or "none"
    base = {"provider": prov, "model": _primary_model() if prov != "none" else None, "checkedAt": now}
    if prov == "none":
        return {**base, "status": "not_configured", "safeReason": "invalid_credentials"}
    try:
        text, model = ai_complete("Reply with exactly: OK", allow_free_fallback=False, timeout=15)
        ok = len(text.strip()) > 0
        return {**base, "model": model, "status": "connected" if ok else "unavailable",
                "safeReason": "connected" if ok else "malformed_response"}
    except AIError as e:
        status = "not_configured" if e.reason == "invalid_credentials" else "unavailable"
        return {**base, "status": status, "safeReason": e.reason}
