from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import datetime
import asyncio

import ai_client

router = APIRouter()

def _key_configured() -> bool:
    return ai_client.key_configured()

def _generate(prompt: str, json_mode: bool = False):
    """Routes through OpenRouter (see ai_client). Returns (response_text, model_name)."""
    return ai_client.ai_complete(prompt, json_mode=json_mode)


# In-memory last health check
_last_ai_health = {
    "status": "not_configured",
    "provider": "openrouter",
    "model": None,
    "checkedAt": None,
    "safeReason": "Service not initialized"
}

def probe_gemini() -> dict:
    """Validate provider config + live reachability via the shared AI client. Never logs secrets."""
    global _last_ai_health
    _last_ai_health = ai_client.probe()
    return _last_ai_health


class AdvisoryRequest(BaseModel):
    business: Dict[str, Any]
    location: Optional[Dict[str, Any]] = None
    finance: Optional[Dict[str, Any]] = None
    competition: Optional[Dict[str, Any]] = None
    demandAnchors: Optional[List[Dict[str, Any]]] = None
    schemes: Optional[List[Dict[str, Any]]] = None
    sourceMetadata: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    location: Optional[Dict[str, Any]] = None
    businessDiscoveryResults: Optional[List[Dict[str, Any]]] = None
    selectedBusinesses: Optional[List[Dict[str, Any]]] = None
    comparison: Optional[Dict[str, Any]] = None
    financialPlan: Optional[Dict[str, Any]] = None
    schemeMatches: Optional[List[Dict[str, Any]]] = None
    sourceMetadata: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None

@router.post("/advisory")
async def generate_advisory(req: AdvisoryRequest):
    timestamp = datetime.datetime.now().isoformat()
    if not _key_configured():
        return {
            "status": "unavailable",
            "advisory": {
                "whyRecommended": [],
                "risksAndMitigations": [],
                "opportunities": [],
                "dataGaps": [],
                "confidence": None
            },
            "message": "AI service is unavailable. Retry after the service is restored.",
            "citations": [],
            "generatedAt": timestamp
        }

    prompt = f"""
You are an expert rural business advisor for Vyapar-Mitra (India).
You MUST output ONLY valid JSON matching this exact schema:
{{
  "advisory": {{
    "whyRecommended": ["string", "string"],
    "risksAndMitigations": [
      {{ "risk": "string", "mitigation": "string" }}
    ],
    "opportunities": ["string"],
    "dataGaps": ["string"],
    "confidence": "high" | "medium" | "low"
  }}
}}

INPUT DATA:
Business: {json.dumps(req.business)}
Location: {json.dumps(req.location)}
Finance: {json.dumps(req.finance)}
Competition: {json.dumps(req.competition)}
Demand Anchors: {json.dumps(req.demandAnchors)}
Schemes: {json.dumps(req.schemes)}

RULES:
- Base analysis ONLY on actual evidence provided.
- Do not hallucinate external schemes or financial values.
"""
    async def attempt_parse(p, attempt=1):
        response_text, _ = await asyncio.to_thread(_generate, p, attempt == 1)
        parsed = json.loads(ai_client.strip_json_fences(response_text))
        
        # Normalize response
        advisory = parsed.get("advisory", {}) if isinstance(parsed, dict) else {}
        
        def enforce_list(val):
            if isinstance(val, list): return val
            if isinstance(val, str): return [val]
            return []
            
        return {
            "status": "ready",
            "advisory": {
                "whyRecommended": enforce_list(advisory.get("whyRecommended")),
                "risksAndMitigations": enforce_list(advisory.get("risksAndMitigations")),
                "opportunities": enforce_list(advisory.get("opportunities")),
                "dataGaps": enforce_list(advisory.get("dataGaps")),
                "confidence": advisory.get("confidence", None)
            },
            "message": "",
            "citations": [],
            "generatedAt": timestamp
        }

    try:
        return await attempt_parse(prompt)
    except json.JSONDecodeError:
        try:
            # 1 retry with repair prompt
            repair_prompt = prompt + "\n\nWARNING: Your last output was not valid JSON. Please fix any trailing commas or unescaped quotes and return ONLY valid JSON."
            return await attempt_parse(repair_prompt, attempt=2)
        except Exception:
            pass # Fall through to fallback
    except Exception as e:
        print(f"Advisory Generation Error: {type(e).__name__} - {str(e)}")
        
    return {
        "status": "unavailable",
        "advisory": {
            "whyRecommended": [],
            "risksAndMitigations": [],
            "opportunities": [],
            "dataGaps": [],
            "confidence": None
        },
        "message": "AI service is temporarily unavailable. Please retry later.",
        "citations": [],
        "generatedAt": timestamp
    }


@router.post("/chat")
async def chat_with_advisor(req: ChatRequest):
    if not _key_configured():
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable. Retry after the service is restored.",
        )

    lang_note = {
        "en": "Respond in clear English.",
        "hi": "Respond in Hindi (Devanagari).",
        "te": "Respond in Telugu.",
    }.get(req.language, "Respond in clear English.")

    has_discovery = bool(req.businessDiscoveryResults)
    has_schemes = bool(req.schemeMatches)
    has_comparison = bool(req.comparison)
    has_finance = bool(req.financialPlan)

    prompt = f"""You are Vyapar-Mitra Assistant — a grounded financial advisor for rural micro-entrepreneurs in India.
{lang_note}
Keep answers practical and concise (3–8 sentences). Use Markdown. No raw HTML.

CRITICAL GROUNDING RULES — violation is not allowed:
1. Use ONLY the context JSON below. Do NOT invent nearby businesses, job listings, government scheme terms, eligibility, loan approvals, or financial calculations.
2. If a field is null/empty, say the data is not available in the user's session — do not invent it.
3. Never claim provider data exists when discovery results are empty.
4. SAFE LANGUAGE: Never say "You are eligible" or "You will get a loan". Use "You may be eligible" / "Verify with your bank/SCA".
5. When referencing schemes, only use schemeMatches or matchedSchemes present in context; cite officialUrl if present.
6. When comparing businesses, use comparison / selectedBusinesses / businessDiscoveryResults only.
7. Cite sources from sourceMetadata or provenance fields when available.
8. Always suggest a concrete next step (explore, compare, verify documents, visit bank).

CONTEXT AVAILABILITY:
- discovery: {has_discovery}
- schemes: {has_schemes}
- comparison: {has_comparison}
- financialPlan: {has_finance}

Context JSON:
Location: {json.dumps(req.location)}
Discovery Results: {json.dumps(req.businessDiscoveryResults)}
Selected Businesses: {json.dumps(req.selectedBusinesses)}
Comparison: {json.dumps(req.comparison)}
Financial Plan: {json.dumps(req.financialPlan)}
Scheme Matches: {json.dumps(req.schemeMatches)}
Source Metadata: {json.dumps(req.sourceMetadata)}
Other Context: {json.dumps(req.context)}

User Question: {req.message}
"""
    try:
        response_text, used_model = await asyncio.to_thread(_generate, prompt)
        return {
            "response": response_text.strip(),
            "status": "ok",
            "provider": "openrouter",
            "model": used_model,
        }
    except Exception as e:
        print(f"Chat Error: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable. Retry after the service is restored.",
        )
