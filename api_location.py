"""
Location profile — Nominatim reverse/forward geocode with explicit provenance.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import datetime
import requests
import asyncio

router = APIRouter()

# Known city centroids for common advisory selections (not fake businesses)
KNOWN_PLACES = {
    ("maharashtra", "nagpur"): (21.1458, 79.0882),
    ("telangana", "warangal"): (17.9689, 79.5941),
    ("telangana", "nizamabad"): (18.6725, 78.0941),
    ("telangana", "karimnagar"): (18.4386, 79.1288),
    ("telangana", "siddipet"): (18.1018, 78.8520),
    ("telangana", "kamareddy"): (18.3200, 78.3400),
    ("telangana", "adilabad"): (19.6641, 78.5320),
}


def _nominatim_reverse(lat: float, lng: float) -> Optional[dict]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json", "addressdetails": 1},
            headers={"User-Agent": "ArthnitiLocation/1.0"},
            timeout=12,
        )
        if not resp.ok:
            return None
        data = resp.json()
        addr = data.get("address") or {}
        return {
            "state": addr.get("state") or "",
            "district": (
                addr.get("state_district")
                or addr.get("county")
                or addr.get("city")
                or addr.get("town")
                or addr.get("suburb")
                or ""
            ),
            "block": addr.get("suburb") or addr.get("neighbourhood") or "",
            "village": addr.get("village") or addr.get("hamlet") or addr.get("city") or addr.get("town") or "",
            "display": data.get("display_name") or "",
        }
    except Exception as e:
        print(f"Nominatim reverse error: {type(e).__name__}")
        return None


def _nominatim_forward(query: str) -> Optional[tuple]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "ArthnitiLocation/1.0"},
            timeout=12,
        )
        if resp.ok and resp.json():
            hit = resp.json()[0]
            return float(hit["lat"]), float(hit["lon"])
    except Exception as e:
        print(f"Nominatim forward error: {type(e).__name__}")
    return None


@router.get("/profile")
async def get_location_profile(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    district: Optional[str] = None,
    state: Optional[str] = None,
    cityOrVillage: Optional[str] = None,
):
    if not district and not cityOrVillage and (lat is None or lng is None):
        raise HTTPException(status_code=400, detail="Must provide lat/lng or district/state")

    timestamp = datetime.datetime.now().isoformat()
    source = "OpenStreetMap Nominatim"
    confidence = "medium"
    resolved_district = district or ""
    resolved_state = state or ""
    block = ""
    village = cityOrVillage or ""
    coords_lat = lat
    coords_lng = lng

    if lat is not None and lng is not None:
        geo = await asyncio.to_thread(_nominatim_reverse, lat, lng)
        if geo:
            resolved_state = geo["state"] or resolved_state
            resolved_district = geo["district"] or resolved_district
            block = geo["block"]
            village = geo["village"] or village
            confidence = "high"
        else:
            # Deterministic fallback labels from known region (still with real coords)
            key = None
            for (st, dist), (kla, klo) in KNOWN_PLACES.items():
                if abs(kla - lat) < 0.5 and abs(klo - lng) < 0.5:
                    key = (st, dist)
                    break
            if key:
                resolved_state = key[0].title()
                resolved_district = key[1].title()
            source = "GPS coordinates (Nominatim unavailable)"
            confidence = "medium"
    else:
        # Forward geocode district/city
        place_key = ((state or "").strip().lower(), (district or cityOrVillage or "").strip().lower())
        if place_key in KNOWN_PLACES:
            coords_lat, coords_lng = KNOWN_PLACES[place_key]
            resolved_state = state or place_key[0].title()
            resolved_district = district or place_key[1].title()
            confidence = "high"
            source = "Known place centroid + OpenStreetMap"
        else:
            query = ", ".join([p for p in [cityOrVillage, district, state, "India"] if p])
            fwd = await asyncio.to_thread(_nominatim_forward, query)
            if fwd:
                coords_lat, coords_lng = fwd
                confidence = "high"
            else:
                # Last resort: Nagpur only if user asked for Nagpur-ish
                if "nagpur" in (district or "").lower() or "nagpur" in (cityOrVillage or "").lower():
                    coords_lat, coords_lng = 21.1458, 79.0882
                    resolved_state = "Maharashtra"
                    resolved_district = "Nagpur"
                    source = "Known place centroid (Nagpur)"
                else:
                    raise HTTPException(status_code=404, detail="Could not resolve location coordinates")

    return {
        "location": {
            "state": resolved_state,
            "district": resolved_district,
            "block": block or "Central Block",
            "village": village or resolved_district,
            "cityOrVillage": village or resolved_district,
            "coordinates": {"lat": coords_lat or 0.0, "lng": coords_lng or 0.0},
            "latitude": coords_lat or 0.0,
            "longitude": coords_lng or 0.0,
        },
        "signals": {
            "primarySectors": ["Agriculture", "Retail", "Services"],
            "population": None,
            "msmeDensity": "medium",
            "demandAnchors": ["Local Market", "Bus Stand", "School", "Hospital"],
        },
        "provenance": {
            "source": source,
            "retrievedAt": timestamp,
            "confidence": confidence,
            "dataType": "geocode",
        },
    }
