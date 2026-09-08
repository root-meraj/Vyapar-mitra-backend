from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import datetime
import traceback

import api_location
import api_business
import api_schemes
import api_ai
import finance_engine

router = APIRouter()

class FeasibilityRequest(BaseModel):
    location: dict
    business: dict
    userProfile: dict
    budget: float
    selectedScenario: str = "expected"

# Simple memory cache: key -> { "data": dict, "expires_at": datetime }
_cache = {}

def get_cache_key(req: FeasibilityRequest):
    dist = req.location.get("district", "")
    cat = req.business.get("category", "")
    return f"{dist}-{cat}-{req.budget}"

@router.post("/report")
async def generate_feasibility_report(req: FeasibilityRequest):
    cache_key = get_cache_key(req)
    now = datetime.datetime.now()
    
    if cache_key in _cache:
        if now < _cache[cache_key]["expires_at"]:
            return _cache[cache_key]["data"]
            
    async def fetch_location():
        try:
            return await asyncio.wait_for(
                api_location.get_location_profile(req.location.get("district", ""), req.location.get("state", "")),
                timeout=4.0
            )
        except Exception:
            return None

    async def fetch_finance():
        try:
            import api_finance
            req_fin = api_finance.FinancialPlanRequest(
                projectCost=req.business.get("avgOperatingCost", 0) * 6,
                applicantMargin=req.budget,
                requiredCredit=max(0, (req.business.get("avgOperatingCost", 0) * 6) - req.budget),
                annualInterestRate=8.0, # Default for advisory
                tenureMonths=60, # Default for advisory
                monthlyRevenue=req.business.get("avgRevenue", 0),
                monthlyOperatingCost=req.business.get("avgOperatingCost", 0)
            )
            res = await api_finance.financial_plan(req_fin)
            return res
        except Exception as e:
            print("Finance error", e)
            return None
            
    async def fetch_schemes():
        try:
            scheme_req = api_schemes.SchemeMatchRequest(
                businessCategory=req.business.get("category", ""),
                userProfile=req.userProfile,
                location=req.location
            )
            res = await asyncio.wait_for(api_schemes.match_schemes(scheme_req), timeout=4.0)
            return res
        except asyncio.TimeoutError:
            return {"status": "unavailable", "matches": [], "message": "Scheme matching provider timed out.", "sources": [], "retrievedAt": now.isoformat(), "providerStatus": "unavailable"}
        except Exception:
            return {"status": "error", "matches": [], "message": "Scheme matching provider failed.", "sources": [], "retrievedAt": now.isoformat(), "providerStatus": "unavailable"}

    async def fetch_advisory(finance_data):
        try:
            advisory_req = api_ai.AdvisoryRequest(
                business=req.business,
                location=req.location,
                finance=finance_data or {}
            )
            res = await asyncio.wait_for(api_ai.generate_advisory(advisory_req), timeout=8.0)
            return res
        except asyncio.TimeoutError:
            return {"status": "unavailable", "advisory": {"whyRecommended": [], "risksAndMitigations": [], "opportunities": [], "dataGaps": [], "confidence": None}, "message": "AI advisory timed out.", "citations": [], "generatedAt": now.isoformat()}
        except Exception as e:
            return {"status": "unavailable", "advisory": {"whyRecommended": [], "risksAndMitigations": [], "opportunities": [], "dataGaps": [], "confidence": None}, "message": "AI advisory unavailable.", "citations": [], "generatedAt": now.isoformat()}

    loc_task = asyncio.create_task(fetch_location())
    fin_task = asyncio.create_task(fetch_finance())
    scheme_task = asyncio.create_task(fetch_schemes())
    
    fin_res = await fin_task
    adv_task = asyncio.create_task(fetch_advisory(fin_res))
    
    loc_res, scheme_res, adv_res = await asyncio.gather(loc_task, scheme_task, adv_task, return_exceptions=True)

    if isinstance(scheme_res, Exception):
        scheme_res = {"status": "error", "matches": [], "message": "Provider failed", "sources": [], "retrievedAt": now.isoformat(), "providerStatus": "unavailable"}
    if isinstance(adv_res, Exception):
        adv_res = {"status": "error", "advisory": {"whyRecommended": [], "risksAndMitigations": [], "opportunities": [], "dataGaps": [], "confidence": None}, "message": "Provider failed", "citations": [], "generatedAt": now.isoformat()}

    overall_status = "complete"
    if scheme_res.get("status") != "ready" or adv_res.get("status") != "ready":
        overall_status = "partial"

    import uuid
    report = {
        "status": overall_status,
        "reportId": str(uuid.uuid4()),
        "generatedAt": now.isoformat(),
        "summary": {},
        "financials": fin_res or {},
        "locationContext": loc_res or {},
        "strategicAdvisory": adv_res,
        "schemeMatching": scheme_res,
        "sources": scheme_res.get("sources") or adv_res.get("citations") or [],
        "dataGaps": [],
        "providerStatus": {
            "location": "ready" if loc_res else "unavailable",
            "finance": "ready" if fin_res else "unavailable",
            "schemes": scheme_res.get("status"),
            "advisory": adv_res.get("status")
        }
    }
    
    _cache[cache_key] = {
        "data": report,
        "expires_at": now + datetime.timedelta(minutes=30)
    }
    
    return report
