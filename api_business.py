"""
Business discovery & comparison — live map providers + deterministic scoring.
Primary: Google Places (if GOOGLE_MAPS_API_KEY set). Secondary: OSM Overpass.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import datetime
import os
import math
import requests

router = APIRouter()

# ── Request models (accept both new + legacy shapes) ─────────────────────────

class LocationBody(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    state: Optional[str] = ""
    district: Optional[str] = ""
    cityOrVillage: Optional[str] = ""
    block: Optional[str] = ""
    village: Optional[str] = ""
    radiusKm: Optional[int] = None
    coordinates: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ProfileBody(BaseModel):
    skills: Optional[List[str]] = []
    workPreference: Optional[str] = ""
    spaceStatus: Optional[str] = ""
    availability: Optional[str] = ""
    skillLevel: Optional[str] = ""
    workType: Optional[str] = ""
    isArtisan: Optional[bool] = False
    isWomenEnterprise: Optional[bool] = False

    class Config:
        extra = "allow"


class FiltersBody(BaseModel):
    category: Optional[str] = ""
    withinBudget: Optional[bool] = None
    schemeSupported: Optional[bool] = None

    class Config:
        extra = "allow"


class DiscoverRequest(BaseModel):
    location: Dict[str, Any]
    budget: Optional[float] = None
    profile: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    # Legacy fields
    radius: Optional[Union[str, int, float]] = None
    marginCapital: Optional[float] = None
    skillLevel: Optional[str] = ""
    workType: Optional[str] = ""
    isArtisan: Optional[bool] = False
    isWomenEnterprise: Optional[bool] = False


class CompareRequest(BaseModel):
    businesses: List[dict]
    budget: float
    location: Optional[Dict[str, Any]] = None
    schemeMatches: Optional[List[dict]] = None


# ── Category taxonomy queried from live providers ────────────────────────────

CATEGORY_QUERIES = {
    "grocery": {"osm_shop": ["convenience", "supermarket", "grocery", "general"], "places": "grocery store|kirana"},
    "dairy": {"osm_shop": ["dairy", "cheese"], "places": "dairy|milk"},
    "food": {"osm_amenity": ["restaurant", "cafe", "fast_food", "food_court"], "places": "restaurant|tiffin"},
    "tailoring": {"osm_shop": ["tailor", "clothes", "boutique", "fabric"], "osm_craft": ["tailor"], "places": "tailor|garment"},
    "printing": {"osm_shop": ["copyshop", "stationery", "books"], "places": "printing|stationery|xerox"},
    "repair": {"osm_shop": ["mobile_phone", "electronics", "computer", "car_repair", "motorcycle"], "places": "mobile repair|electronics repair"},
    "agriculture": {"osm_shop": ["agrarian", "farm", "garden_centre"], "places": "agricultural supplier|fertilizer"},
    "rental": {"osm_shop": ["rental", "hardware"], "places": "equipment rental|tool rental"},
    "transport": {"osm_amenity": ["taxi", "bus_station"], "osm_shop": ["car"], "places": "transport|logistics|tempo"},
    "handicrafts": {"osm_craft": ["handicraft", "pottery", "jeweller", "basket_maker"], "osm_shop": ["gift", "art"], "places": "handicraft|artisan"},
    "beauty": {"osm_shop": ["beauty", "hairdresser", "cosmetics"], "places": "salon|beauty parlor"},
    "digital": {"osm_amenity": ["internet_cafe"], "osm_shop": ["computer", "mobile_phone"], "places": "cyber cafe|CSC|documentation"},
}

DEMAND_ANCHORS = {
    "schools": {"osm_amenity": ["school", "college", "university", "kindergarten"]},
    "hospitals": {"osm_amenity": ["hospital", "clinic", "doctors"]},
    "markets": {"osm_amenity": ["marketplace", "bus_station", "townhall"], "osm_shop": ["mall", "marketplace"]},
    "industrial": {"osm_landuse": ["industrial", "commercial"]},
}


def _resolve_coords(location: dict) -> tuple:
    lat = location.get("latitude") or location.get("lat")
    lon = location.get("longitude") or location.get("lng")
    coords = location.get("coordinates") or {}
    if lat is None:
        lat = coords.get("lat") or coords.get("latitude")
    if lon is None:
        lon = coords.get("lng") or coords.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon), "provided"

    # Geocode via Nominatim when only place names exist
    place_parts = [
        location.get("cityOrVillage") or location.get("village") or "",
        location.get("district") or "",
        location.get("state") or "",
        "India",
    ]
    query = ", ".join([p for p in place_parts if p])
    if query.strip(", India"):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "ArthnitiBusinessDiscovery/1.0"},
                timeout=12,
            )
            if resp.ok and resp.json():
                hit = resp.json()[0]
                return float(hit["lat"]), float(hit["lon"]), f"nominatim:{query}"
        except Exception as e:
            print(f"Nominatim geocode error: {type(e).__name__}")

    # Nagpur center — last-resort known demo point (explicitly attributed)
    return 21.1458, 79.0882, "fallback:nagpur_center"


def _parse_radius_km(req: DiscoverRequest) -> int:
    loc = req.location or {}
    if loc.get("radiusKm") in (5, 10, 20):
        return int(loc["radiusKm"])
    if req.radius is not None:
        if isinstance(req.radius, str) and req.radius.endswith("km"):
            try:
                return max(1, int(req.radius.replace("km", "")))
            except ValueError:
                pass
        elif isinstance(req.radius, (int, float)):
            return max(1, int(req.radius))
    return 5


def _budget(req: DiscoverRequest) -> float:
    if req.budget is not None:
        return float(req.budget)
    if req.marginCapital is not None:
        return float(req.marginCapital)
    return 0.0


# ── Live providers ───────────────────────────────────────────────────────────

def query_google_places(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    api_key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "reason": "not_configured", "elements": [], "provider": "google_places"}

    # Nearby Search (legacy) — types covering our categories
    types = [
        "store", "supermarket", "restaurant", "cafe", "clothing_store",
        "electronics_store", "beauty_salon", "car_repair", "bus_station",
        "school", "hospital", "pharmacy", "bank", "laundry",
    ]
    elements = []
    errors = 0
    for t in types:
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params={
                    "location": f"{lat},{lon}",
                    "radius": min(radius_m, 50000),
                    "type": t,
                    "key": api_key,
                },
                timeout=15,
            )
            data = resp.json()
            status = data.get("status")
            if status not in ("OK", "ZERO_RESULTS"):
                errors += 1
                continue
            for p in data.get("results", []):
                elements.append({
                    "id": p.get("place_id"),
                    "tags": {
                        "name": p.get("name"),
                        "amenity": t if t in ("school", "hospital", "bus_station", "cafe", "restaurant", "bank") else "",
                        "shop": t if t not in ("school", "hospital", "bus_station", "cafe", "restaurant", "bank") else "",
                        "source": "google_places",
                        "types": p.get("types", []),
                    },
                    "lat": (p.get("geometry") or {}).get("location", {}).get("lat"),
                    "lon": (p.get("geometry") or {}).get("location", {}).get("lng"),
                })
        except Exception as e:
            errors += 1
            print(f"Places API error ({t}): {type(e).__name__}")

    if errors == len(types) and not elements:
        return {"ok": False, "reason": "provider_error", "elements": [], "provider": "google_places"}

    return {
        "ok": True,
        "reason": "ok",
        "elements": elements,
        "provider": "Google Places API",
        "count": len(elements),
    }


def query_overpass(lat: float, lon: float, radius_meters: int) -> Dict[str, Any]:
    # Cap query size for reliability; still covers required categories
    radius_meters = min(int(radius_meters), 20000)
    query = f"""
    [out:json][timeout:30];
    (
      node["shop"](around:{radius_meters},{lat},{lon});
      way["shop"](around:{radius_meters},{lat},{lon});
      node["amenity"~"school|college|university|kindergarten|hospital|clinic|doctors|restaurant|cafe|fast_food|food_court|marketplace|bus_station|townhall|internet_cafe|taxi"](around:{radius_meters},{lat},{lon});
      way["amenity"~"school|college|university|hospital|clinic|marketplace|bus_station"](around:{radius_meters},{lat},{lon});
      node["craft"](around:{radius_meters},{lat},{lon});
      node["office"](around:{radius_meters},{lat},{lon});
      way["landuse"~"industrial|commercial|retail"](around:{radius_meters},{lat},{lon});
    );
    out center tags 250;
    """
    headers = {
        "User-Agent": "ArthnitiBusinessDiscovery/1.0 (local-dev)",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_error = None
    for overpass_url in mirrors:
        try:
            response = requests.post(
                overpass_url, data={"data": query}, headers=headers, timeout=35
            )
            if response.status_code in (429, 504, 502, 406):
                last_error = f"http_{response.status_code}"
                continue
            response.raise_for_status()
            data = response.json()
            elements = data.get("elements") or []
            return {
                "ok": True,
                "reason": "ok" if elements else "empty_area",
                "elements": elements,
                "provider": f"OpenStreetMap Overpass API ({overpass_url.split('/')[2]})",
                "count": len(elements),
            }
        except Exception as e:
            last_error = type(e).__name__
            print(f"Overpass API Error ({overpass_url.split('/')[2]}): {last_error}")
            continue
    return {
        "ok": False,
        "reason": "provider_error",
        "elements": [],
        "provider": "OpenStreetMap Overpass API",
        "count": 0,
        "lastError": last_error,
    }


def analyze_elements(elements: List[dict]) -> Dict[str, int]:
    counts = {
        "grocery": 0, "dairy": 0, "food": 0, "tailoring": 0, "printing": 0,
        "repair": 0, "agriculture": 0, "rental": 0, "transport": 0,
        "handicrafts": 0, "beauty": 0, "digital": 0,
        "schools": 0, "hospitals": 0, "markets": 0, "industrial": 0,
        "total": len(elements),
    }

    for el in elements:
        tags = el.get("tags") or {}
        amenity = (tags.get("amenity") or "").lower()
        shop = (tags.get("shop") or "").lower()
        craft = (tags.get("craft") or "").lower()
        landuse = (tags.get("landuse") or "").lower()
        types = [str(t).lower() for t in (tags.get("types") or [])]
        blob = " ".join([amenity, shop, craft, landuse] + types)

        if amenity in ("school", "college", "university", "kindergarten") or "school" in types:
            counts["schools"] += 1
        if amenity in ("hospital", "clinic", "doctors") or "hospital" in types:
            counts["hospitals"] += 1
        if amenity in ("marketplace", "bus_station", "townhall") or shop in ("mall", "marketplace") or "bus_station" in types:
            counts["markets"] += 1
        if landuse in ("industrial", "commercial", "retail"):
            counts["industrial"] += 1

        if shop in ("convenience", "supermarket", "grocery", "general") or "supermarket" in types or "grocery" in blob:
            counts["grocery"] += 1
        if shop in ("dairy", "cheese") or "dairy" in blob or "milk" in blob:
            counts["dairy"] += 1
        if amenity in ("restaurant", "cafe", "fast_food", "food_court") or "restaurant" in types or "cafe" in types:
            counts["food"] += 1
        if shop in ("tailor", "clothes", "boutique", "fabric") or craft == "tailor" or "clothing" in blob:
            counts["tailoring"] += 1
        if shop in ("copyshop", "stationery", "books") or "stationery" in blob or "print" in blob:
            counts["printing"] += 1
        if shop in ("mobile_phone", "electronics", "computer", "car_repair", "motorcycle") or "electronics" in types or "repair" in blob:
            counts["repair"] += 1
        if shop in ("agrarian", "farm", "garden_centre") or "agricultur" in blob or "fertilizer" in blob:
            counts["agriculture"] += 1
        if shop in ("rental", "hardware") or "rental" in blob:
            counts["rental"] += 1
        if amenity in ("taxi", "bus_station") or "transport" in blob or "logistics" in blob:
            counts["transport"] += 1
        if craft in ("handicraft", "pottery", "jeweller", "basket_maker") or shop in ("gift", "art") or "handicraft" in blob:
            counts["handicrafts"] += 1
        if shop in ("beauty", "hairdresser", "cosmetics") or "beauty" in types or "salon" in blob:
            counts["beauty"] += 1
        if amenity == "internet_cafe" or "cyber" in blob or "documentation" in blob:
            counts["digital"] += 1

    return counts


def _density_label(comp_count: int, radius_km: float) -> str:
    """Absolute listing counts are more actionable than sparse OSM density ratios."""
    # Scale soft thresholds with radius (5km baseline)
    scale = max(1.0, radius_km / 5.0)
    if comp_count >= int(15 * scale):
        return "high"
    if comp_count >= int(5 * scale):
        return "medium"
    return "low"


def _match_schemes_for_idea(idea: dict, profile: dict, budget: float) -> List[dict]:
    """Lightweight verified-scheme gate using official curated rules (same sources as /api/schemes)."""
    matches = []
    is_artisan = bool(profile.get("isArtisan"))
    project_cost = idea.get("maxCapital") or idea.get("minCapital") or 0

    # PM MUDRA — official mudra.org.in
    if project_cost <= 1_000_000:
        matches.append({
            "schemeId": "pm_mudra",
            "name": "Pradhan Mantri MUDRA Yojana (PMMY)",
            "officialUrl": "https://www.mudra.org.in/",
            "agency": "Ministry of Finance",
        })

    # PM Vishwakarma — artisans only
    if is_artisan and project_cost <= 300_000 and idea.get("workType") in ("service", "manufacturing", "agriculture-linked"):
        if any(k in (idea.get("name") or "").lower() for k in ("tailor", "garment", "handicraft", "repair", "beauty")):
            matches.append({
                "schemeId": "pm_vishwakarma",
                "name": "PM Vishwakarma Yojana",
                "officialUrl": "https://pmvishwakarma.gov.in/",
                "agency": "Ministry of MSME",
            })

    return matches


def build_opportunities(
    counts: Dict[str, int],
    radius_km: float,
    budget: float,
    apply_budget: bool,
    profile: dict,
    provider_name: str,
    timestamp: str,
) -> tuple:
    """Build evidence-based ideas using a 5-factor deterministic scoring model. Returns (results, filtered_out)."""
    results = []
    filtered_out = []

    # Idea templates driven by evidence rules (not hardcoded fake cards)
    ideas = [
        {
            "id": "idea-grocery",
            "name": "Grocery / Kirana Store",
            "category": "Retail and Kirana",
            "workType": "retail",
            "skillLevel": "None",
            "minCapital": 25000,
            "maxCapital": 150000,
            "avgRevenue": 30000,
            "avgOperatingCost": 22000,
            "comp_key": "grocery",
            "anchor": counts["markets"] + counts["schools"] + counts["industrial"],
            "rule": "Dense residential + market anchors support daily-needs retail.",
            "prefer_when": lambda c: c["markets"] + c["schools"] >= 2,
        },
        {
            "id": "idea-dairy",
            "name": "Dairy & Milk Products",
            "category": "Dairy and Animal Husbandry",
            "workType": "agriculture-linked",
            "skillLevel": "Beginner",
            "minCapital": 30000,
            "maxCapital": 200000,
            "avgRevenue": 35000,
            "avgOperatingCost": 20000,
            "comp_key": "dairy",
            "anchor": counts["markets"] + counts["agriculture"],
            "rule": "Dairy recommended only when competition is not already dense.",
            "prefer_when": lambda c: c["dairy"] < 8 and (c["markets"] > 0 or c["agriculture"] > 0),
        },
        {
            "id": "idea-food",
            "name": "Food / Tiffin Service",
            "category": "Food & Beverage",
            "workType": "service",
            "skillLevel": "Beginner",
            "minCapital": 15000,
            "maxCapital": 50000,
            "avgRevenue": 25000,
            "avgOperatingCost": 12000,
            "comp_key": "food",
            "anchor": counts["hospitals"] + counts["markets"] + counts["schools"] + counts["industrial"],
            "rule": "Offices, schools, hospitals and markets create meal demand.",
            "prefer_when": lambda c: (c["hospitals"] + c["markets"] + c["schools"] + c["industrial"]) >= 2,
        },
        {
            "id": "idea-tailoring",
            "name": "Tailoring / Garment Unit",
            "category": "Tailoring, Garment, and Textile Work",
            "workType": "service",
            "skillLevel": "Experienced",
            "minCapital": 10000,
            "maxCapital": 80000,
            "avgRevenue": 25000,
            "avgOperatingCost": 10000,
            "comp_key": "tailoring",
            "anchor": counts["markets"] + counts["schools"],
            "rule": "Local garment demand near markets and residential clusters.",
            "prefer_when": lambda c: True,
        },
        {
            "id": "idea-printing",
            "name": "Printing & Stationery",
            "category": "Retail",
            "workType": "retail",
            "skillLevel": "Beginner",
            "minCapital": 15000,
            "maxCapital": 70000,
            "avgRevenue": 20000,
            "avgOperatingCost": 8000,
            "comp_key": "printing",
            "anchor": counts["schools"],
            "rule": "Schools + limited print shops → stationery/xerox opportunity.",
            "prefer_when": lambda c: c["schools"] >= 1 and c["printing"] <= 3,
        },
        {
            "id": "idea-repair",
            "name": "Mobile / Electronics Repair",
            "category": "Electronics and Repair",
            "workType": "service",
            "skillLevel": "Experienced",
            "minCapital": 20000,
            "maxCapital": 100000,
            "avgRevenue": 40000,
            "avgOperatingCost": 15000,
            "comp_key": "repair",
            "anchor": counts["markets"] + counts["schools"] + counts["hospitals"],
            "rule": "Service anchors support repair demand.",
            "prefer_when": lambda c: (c["markets"] + c["schools"] + c["hospitals"]) >= 2,
        },
        {
            "id": "idea-agri",
            "name": "Agricultural Inputs / Supplier",
            "category": "Agriculture Linked",
            "workType": "agriculture-linked",
            "skillLevel": "Beginner",
            "minCapital": 40000,
            "maxCapital": 250000,
            "avgRevenue": 45000,
            "avgOperatingCost": 30000,
            "comp_key": "agriculture",
            "anchor": counts["agriculture"] + counts["markets"] + counts["industrial"],
            "rule": "Agricultural activity near markets supports agri-input retail.",
            "prefer_when": lambda c: c["agriculture"] > 0 or c["industrial"] > 0 or c["markets"] > 0,
        },
        {
            "id": "idea-rental",
            "name": "Equipment Rental",
            "category": "Services",
            "workType": "service",
            "skillLevel": "Beginner",
            "minCapital": 50000,
            "maxCapital": 300000,
            "avgRevenue": 40000,
            "avgOperatingCost": 18000,
            "comp_key": "rental",
            "anchor": counts["agriculture"] + counts["industrial"] + counts["markets"],
            "rule": "Agri/industrial activity + low rental listings → equipment rental gap.",
            "prefer_when": lambda c: (c["agriculture"] + c["industrial"]) >= 1 and c["rental"] <= 2,
        },
        {
            "id": "idea-beauty",
            "name": "Beauty / Wellness Salon",
            "category": "Beauty and Wellness",
            "workType": "service",
            "skillLevel": "Beginner",
            "minCapital": 20000,
            "maxCapital": 120000,
            "avgRevenue": 30000,
            "avgOperatingCost": 14000,
            "comp_key": "beauty",
            "anchor": counts["markets"] + counts["schools"],
            "rule": "Residential + market clusters support local beauty services.",
            "prefer_when": lambda c: c["markets"] + c["schools"] >= 1,
        },
        {
            "id": "idea-digital",
            "name": "Digital / Documentation Centre (CSC-style)",
            "category": "Digital Services",
            "workType": "service",
            "skillLevel": "Beginner",
            "minCapital": 25000,
            "maxCapital": 100000,
            "avgRevenue": 28000,
            "avgOperatingCost": 12000,
            "comp_key": "digital",
            "anchor": counts["markets"] + counts["schools"] + counts["hospitals"] + counts["industrial"],
            "rule": "Service anchors create demand for documentation & digital access.",
            "prefer_when": lambda c: (c["markets"] + c["schools"] + c["hospitals"]) >= 2 and c["digital"] <= 3,
        },
        {
            "id": "idea-transport",
            "name": "Local Transport / Logistics",
            "category": "Transport and Logistics",
            "workType": "service",
            "skillLevel": "Beginner",
            "minCapital": 50000,
            "maxCapital": 400000,
            "avgRevenue": 50000,
            "avgOperatingCost": 28000,
            "comp_key": "transport",
            "anchor": counts["markets"] + counts["industrial"],
            "rule": "Market and industrial anchors create last-mile logistics demand.",
            "prefer_when": lambda c: c["markets"] + c["industrial"] >= 1,
        },
        {
            "id": "idea-handicrafts",
            "name": "Handicrafts / Artisan Products",
            "category": "Handicrafts",
            "workType": "manufacturing",
            "skillLevel": "Experienced",
            "minCapital": 10000,
            "maxCapital": 80000,
            "avgRevenue": 22000,
            "avgOperatingCost": 9000,
            "comp_key": "handicrafts",
            "anchor": counts["markets"] + counts["handicrafts"],
            "rule": "Local craft activity near markets supports artisan enterprise.",
            "prefer_when": lambda c: profile.get("isArtisan") or c["handicrafts"] > 0 or c["markets"] > 0,
        },
    ]

    confidence = "high" if counts["total"] > 25 else ("medium" if counts["total"] > 8 else "low")

    for idea in ideas:
        if not idea["prefer_when"](counts):
            if counts["total"] == 0:
                continue

        comp_count = counts.get(idea["comp_key"], 0)
        density = _density_label(comp_count, radius_km)
        schemes = _match_schemes_for_idea(idea, profile, budget)
        within_budget = budget <= 0 or budget >= idea["minCapital"]

        # Factor 1: Local Demand (0-30)
        demand_points = min(30, max(5, idea["anchor"] * 4))
        if idea["id"] == "idea-printing" and counts["schools"] >= 1: demand_points += 5
        if idea["id"] == "idea-rental" and (counts["agriculture"] + counts["industrial"]) >= 1: demand_points += 5
        demand_points = min(30, demand_points)

        # Factor 2: Competition density (0-25)
        if density == "high": comp_points = 5
        elif density == "medium": comp_points = 15
        else: comp_points = 25

        # Factor 3: Financial viability (0-25)
        fin_points = 25 if within_budget else 10
        margin = max(0, idea["avgRevenue"] - idea["avgOperatingCost"])
        if margin > 15000: fin_points = min(25, fin_points + 5)
        
        # Factor 4: User profile fit (0-10)
        profile_points = 5
        if (idea["workType"] == profile.get("workPreference") or not profile.get("workPreference")):
            profile_points += 2
        if idea["skillLevel"] == profile.get("skillLevel") or idea["skillLevel"] == "Beginner" or not profile.get("skillLevel"):
            profile_points += 3

        # Factor 5: Verified scheme fit (0-10)
        scheme_points = 10 if len(schemes) > 0 else 0

        total_score = int(demand_points + comp_points + fin_points + profile_points + scheme_points)
        # Cap at 95 unless it's a completely perfect match across the board, which is rare.
        total_score = min(98, total_score)

        item = {
            "id": idea["id"],
            "name": idea["name"],
            "category": idea["category"],
            "workType": idea["workType"],
            "skillLevel": idea["skillLevel"],
            "minCapital": idea["minCapital"],
            "maxCapital": idea["maxCapital"],
            "avgRevenue": idea["avgRevenue"],
            "avgOperatingCost": idea["avgOperatingCost"],
            "ownSpaceRequired": False,
            "isHomeBased": idea["workType"] in ("service", "manufacturing"),
            "isWomenFriendly": True,
            "competitorDensity": density,
            "competitorCount": comp_count,
            "demandProxyScore": total_score,
            "scoreBreakdown": {
                "demand": demand_points,
                "competition": comp_points,
                "finance": fin_points,
                "profile": profile_points,
                "schemes": scheme_points
            },
            "schemeSupported": len(schemes) > 0,
            "matchedSchemes": schemes,
            "radiusKm": radius_km,
            "signals": (
                f"{idea['rule']} Nearby: {comp_count} similar listings; "
                f"demand anchors — schools {counts['schools']}, markets {counts['markets']}, "
                f"hospitals {counts['hospitals']}, industrial/commercial {counts['industrial']} "
                f"(radius {radius_km} km)."
            ),
            "nearbySignals": {
                "competitorCount": comp_count,
                "schools": counts["schools"],
                "markets": counts["markets"],
                "hospitals": counts["hospitals"],
                "industrial": counts["industrial"],
                "categoryCount": counts.get(idea["comp_key"], 0),
            },
            "provenance": {
                "source": provider_name,
                "retrievedAt": timestamp,
                "dataType": "Live local map signals",
                "confidence": confidence,
            },
            "withinBudget": within_budget,
        }

        if apply_budget and not within_budget:
            filtered_out.append({
                "id": idea["id"],
                "name": idea["name"],
                "reason": f"Requires ₹{idea['minCapital']:,}–₹{idea['maxCapital']:,}; your budget is ₹{int(budget):,}.",
            })
            continue

        results.append(item)

    # Sort: prefer higher total score
    results.sort(key=lambda r: -r["demandProxyScore"])
    return results, filtered_out


@router.post("/discover")
async def discover_businesses(req: DiscoverRequest):
    timestamp = datetime.datetime.now().isoformat()
    lat, lon, coord_source = _resolve_coords(req.location or {})
    radius_km = _parse_radius_km(req)
    radius_m = radius_km * 1000
    budget = _budget(req)
    profile = req.profile or {}
    if req.skillLevel:
        profile.setdefault("skillLevel", req.skillLevel)
    if req.workType:
        profile.setdefault("workPreference", req.workType)
    if req.isArtisan:
        profile["isArtisan"] = True
    if req.isWomenEnterprise:
        profile["isWomenEnterprise"] = True

    filters = req.filters or {}
    # withinBudget: default True when budget > 0 unless explicitly false
    within_budget_flag = filters.get("withinBudget")
    if within_budget_flag is None:
        within_budget_flag = budget > 0
    apply_budget = bool(within_budget_flag) and budget > 0

    category_filter = (filters.get("category") or req.workType or profile.get("workPreference") or "").strip().lower()
    scheme_only = bool(filters.get("schemeSupported"))

    # Primary Places, secondary OSM
    places = query_google_places(lat, lon, radius_m)
    if places.get("ok") and places.get("elements"):
        provider_payload = places
    else:
        osm = query_overpass(lat, lon, radius_m)
        if places.get("reason") == "not_configured":
            provider_payload = osm
        elif osm.get("ok"):
            provider_payload = osm
            provider_payload["fallbackNote"] = "Google Places unavailable or empty; used OpenStreetMap."
        else:
            provider_payload = {
                "ok": False,
                "reason": "provider_unavailable",
                "elements": [],
                "provider": "none",
                "count": 0,
            }

    if not provider_payload.get("ok") or (
        provider_payload.get("reason") in ("provider_error", "provider_unavailable")
        and not provider_payload.get("elements")
    ):
        return {
            "status": "provider_unavailable",
            "results": [],
            "filteredOut": [],
            "meta": {
                "provider": provider_payload.get("provider", "none"),
                "providerStatus": "unavailable",
                "radiusKm": radius_km,
                "retrievedAt": timestamp,
                "latitude": lat,
                "longitude": lon,
                "coordSource": coord_source,
                "elementCount": 0,
                "safeMessage": "Live local-business data is unavailable for this area right now.",
                "jobsConnected": False,
            },
        }

    elements = provider_payload.get("elements") or []
    # empty_area with zero elements → still honest provider-unavailable-ish for that radius
    if len(elements) == 0:
        return {
            "status": "provider_unavailable",
            "results": [],
            "filteredOut": [],
            "meta": {
                "provider": provider_payload.get("provider"),
                "providerStatus": "empty",
                "radiusKm": radius_km,
                "retrievedAt": timestamp,
                "latitude": lat,
                "longitude": lon,
                "coordSource": coord_source,
                "elementCount": 0,
                "safeMessage": "Live local-business data is unavailable for this area right now.",
                "jobsConnected": False,
                "suggestions": ["Increase radius to 10 km or 20 km", "Edit location", "Add manual local observations"],
            },
        }

    counts = analyze_elements(elements)
    results, filtered_out = build_opportunities(
        counts, radius_km, budget, apply_budget, profile,
        provider_payload.get("provider", "OpenStreetMap"), timestamp,
    )

    # Category / workType filter
    if category_filter:
        before = results[:]
        results = [
            r for r in results
            if category_filter in (r.get("category") or "").lower()
            or category_filter in (r.get("name") or "").lower()
            or category_filter == (r.get("workType") or "").lower()
            or (category_filter == "agriculture-linked" and r.get("workType") == "agriculture-linked")
            or (category_filter == "manufacturing" and r.get("workType") == "manufacturing")
        ]
        for r in before:
            if r not in results:
                filtered_out.append({
                    "id": r["id"],
                    "name": r["name"],
                    "reason": f"Does not match category filter “{category_filter}”.",
                })

    if scheme_only:
        before = results[:]
        results = [r for r in results if r.get("schemeSupported")]
        for r in before:
            if r not in results:
                filtered_out.append({
                    "id": r["id"],
                    "name": r["name"],
                    "reason": "No verified official scheme match for this idea under current profile.",
                })

    # NEW LOGIC: Dynamic Deep Analysis using Gemini 2.5 Flash
    if results:
        import api_ai
        import asyncio
        import json
        
        prompt = f"""You are Arthniti AI, an expert rural business advisor.
We have found the following business opportunities. Based on the local statistics provided below, write a deep, 1-2 sentence analytical paragraph for EACH business explaining its viability and potential.
Local Statistics:
- Schools: {counts.get('schools', 0)}
- Markets: {counts.get('markets', 0)}
- Hospitals: {counts.get('hospitals', 0)}
- Industrial/Commercial: {counts.get('industrial', 0)}

Respond strictly in JSON format where keys are business IDs and values are the analytical paragraphs.
{{
  "idea-grocery": "analysis text...",
  ...
}}

Businesses:
"""
        for r in results:
            prompt += f"- ID: {r['id']}, Name: {r['name']}, Category: {r['category']}, Competition Count: {r.get('competitorCount')}\n"
            
        try:
            response_text, _ = await asyncio.to_thread(api_ai._generate, prompt)
            text = response_text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            parsed_analysis = json.loads(text.strip())
            
            for r in results:
                if r["id"] in parsed_analysis:
                    r["signals"] = parsed_analysis[r["id"]]
                    if isinstance(r.get("provenance"), dict):
                        r["provenance"]["dataType"] = "AI-Generated Deep Analysis (Gemini 2.5 Flash)"
        except Exception as e:
            print(f"Failed to generate deep analysis: {e}")

    status = "ok" if results else "no_suitable"
    return {
        "status": status,
        "results": results,
        "filteredOut": filtered_out,
        "counts": counts,
        "meta": {
            "provider": provider_payload.get("provider"),
            "providerStatus": "connected",
            "radiusKm": radius_km,
            "retrievedAt": timestamp,
            "latitude": lat,
            "longitude": lon,
            "coordSource": coord_source,
            "elementCount": len(elements),
            "budget": budget,
            "filtersApplied": {
                "withinBudget": apply_budget,
                "category": category_filter or None,
                "schemeSupported": scheme_only,
            },
            "jobsConnected": False,
            "fallbackNote": provider_payload.get("fallbackNote"),
        },
    }


@router.post("/compare")
async def compare_businesses(req: CompareRequest):
    if len(req.businesses) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 businesses to compare")

    comparisons = []
    missing_data = []

    for b in req.businesses:
        score = float(b.get("demandProxyScore") or 0)
        density = b.get("competitorDensity") or "medium"
        confidence = ((b.get("provenance") or {}).get("confidence")) or "low"
        has_signals = bool(b.get("signals") or b.get("nearbySignals") or b.get("competitorCount") is not None)

        if not has_signals:
            missing_data.append(b.get("name") or b.get("id") or "unknown")

        if density == "low":
            score += 15
            viability = "High"
            risk = "Low"
        elif density == "high":
            score -= 15
            viability = "Medium-Low"
            risk = "High"
        else:
            viability = "Medium"
            risk = "Medium"

        # Deterministic finance from retrieved capital bands + user budget
        min_cap = float(b.get("minCapital") or 0)
        max_cap = float(b.get("maxCapital") or min_cap)
        avg_rev = float(b.get("avgRevenue") or 0)
        avg_op = float(b.get("avgOperatingCost") or 0)
        surplus = avg_rev - avg_op
        shortfall = max(0, min_cap - req.budget)
        if shortfall > 0:
            score -= min(20, shortfall / max(min_cap, 1) * 20)
            risk = "High" if risk != "High" else risk

        scheme_bonus = 5 if b.get("schemeSupported") else 0
        score += scheme_bonus

        # Confidence dampening — never fabricate high scores without data
        if confidence == "low" or not has_signals:
            score = min(score, 55)
            viability = "Low-confidence"
            missing_data.append(f"{b.get('name')}: incomplete local signals")

        comparisons.append({
            "id": b.get("id"),
            "name": b.get("name"),
            "score": int(min(100, max(0, round(score)))),
            "viability": viability,
            "riskLevel": risk,
            "financialShortfall": shortfall,
            "monthlySurplusEstimate": surplus,
            "competitorDensity": density,
            "competitorCount": b.get("competitorCount"),
            "schemeSupported": bool(b.get("schemeSupported")),
            "matchedSchemes": b.get("matchedSchemes") or [],
            "confidence": confidence,
            "provenance": b.get("provenance"),
            "recommendedAction": (
                "Insufficient data — add manual observations"
                if confidence == "low" and not has_signals
                else ("Proceed with caution" if risk == "High" else "Strong candidate")
            ),
        })

    comparisons.sort(key=lambda x: x["score"], reverse=True)
    top = comparisons[0]
    low_confidence = any(c.get("confidence") == "low" for c in comparisons) or bool(missing_data)

    return {
        "comparisonList": comparisons,
        "topRecommendation": top["id"] if not low_confidence or top["score"] >= 50 else None,
        "lowConfidence": low_confidence,
        "missingData": list(dict.fromkeys(missing_data)),
        "summary": (
            f"Low-confidence comparison: missing or sparse live signals for {', '.join(missing_data[:3])}. "
            f"Scores are capped until more local observations are added."
            if low_confidence and missing_data
            else (
                f"Based on live market data, {top['name']} ranks highest "
                f"({top['score']}/100) with {top['competitorDensity']} competitor density "
                f"and {top['riskLevel']} risk. Scoring is deterministic from retrieved density, "
                f"demand anchors, budget fit, and verified scheme flags — not AI-generated."
            )
        ),
        "retrievedAt": datetime.datetime.now().isoformat(),
    }
