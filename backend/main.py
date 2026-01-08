import os
import secrets
import httpx
import json
from math import radians, cos, sin, asin, sqrt
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.location",
    "places.shortFormattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.currentOpeningHours.openNow",
    "places.priceLevel",
])
PREFERENCE_DICT = {
    "study": "Pick cafes that are good for studying: quieter vibe, seating-friendly, reliable ratings. Prefer closer options if otherwise similar.",
    "friendly": "Pick cafes that feel welcoming/cozy and good for hanging out. Balance distance and quality.",
    "best": "Pick the highest overall quality cafes: prioritize high rating with high rating_count reliability. Distance is secondary.",
    "open": "Pick cafes that are open right now (open_now true). If many are open, prioritize closer + better rated.",
    "busy": "Pick popular/lively cafes: prioritize high rating_count (busy proxy), then rating, then distance.",
}
APP_BASIC_USER = os.getenv("APP_BASIC_USER", "")
APP_BASIC_PASS = os.getenv("APP_BASIC_PASS", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_PLACES_API_KEY:
    raise RuntimeError("GOOGLE_PLACES_API_KEY is not set")

elif not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

app = FastAPI()
security = HTTPBasic()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

Preference = Literal["study", "friendly", "best", "open", "busy"]

class RecommendationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    preference: Preference

class CafePick(BaseModel):
    place_id: str
    why: str
    tags: list[str]

class CafePicksResponse(BaseModel):
    picks: list[CafePick]\

def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, APP_BASIC_USER)
    correct_password = secrets.compare_digest(credentials.password, APP_BASIC_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def haversine_m(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r

def normalize_place(p: dict, user_lat: float, user_lng: float) -> dict | None:
    place_id = p.get("id")
    name = p.get("displayName", {}).get("text")
    address = p.get("shortFormattedAddress")
    location = p.get("location", {})
    lat = location.get("latitude")
    lng = location.get("longitude")

    #these must be present
    if not (place_id and name and address and lat and lng):
        return None
    
    rating = p.get("rating")
    rating_count = p.get("userRatingCount")
    open_now = p.get("currentOpeningHours", {}).get("openNow")
    price_level = p.get("priceLevel")

    distance_m = haversine_m(user_lat, user_lng, lat, lng)

    return {
        "place_id": place_id,
        "name": name,
        "address": address,
        "lat": lat,
        "lng": lng,
        "rating": rating,
        "rating_count": rating_count,
        "open_now": open_now,
        "price_level": price_level,
        "distance_m": distance_m,
    }

async def fetch_nearby_cafes(lat: float, lng: float) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }

    body = {
        "includedTypes": ["cafe"],
        "maxResultCount": 20,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lng
                },
                "radius": 5000.0
            }
        }
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            PLACES_SEARCH_URL,
            headers=headers,
            json=body
        )
        if response.status_code != 200:
            # Show Google’s real error message
            raise HTTPException(status_code=502, detail=response.text)

        data = response.json()

    return data.get("places", [])

async def openai_pick_5(preference: str, formatted_places: list[dict]) -> list[CafePick]:
    compacted_places = []
    for p in formatted_places:
        compacted_places.append({
            "place_id": p["place_id"],
            "name": p["name"],
            "address": p["address"],
            "rating": p.get("rating"),
            "rating_count": p.get("rating_count"),
            "open_now": p.get("open_now"),
            "price_level": p.get("price_level"),
            "distance_m": round(float(p["distance_m"]), 1),
        })
    
    rubric = PREFERENCE_DICT.get(preference, PREFERENCE_DICT["Best"])
    allowed_ids = [p["place_id"] for p in compacted_places]
    prompt = {
        "rubric": rubric,
        "rules": [
            "Return exactly 5 cafe picks from the provided list, if less than 5 cafes are provided, return as many as provided.",
            "Each pick must be one of the provided places.",
            "No duplicate place IDs, therfore no duplicate picks.",
            "keep the 'why' reasoning concise, ie 1-2 sentences.",
            "tags should only be a 4 short strings that capture key attributes of the cafe."
        ],
        "allowed_place_ids": allowed_ids,
        "places": compacted_places
    }

    response = openai_client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are selecting the best cafes from a provided list. Follow the rules strictly."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        text_format=CafePicksResponse,
    )

    return response.output_parsed

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommendations")
async def recommendations(req: RecommendationRequest, _user: str = Depends(require_basic_auth)):
    places = await fetch_nearby_cafes(req.lat, req.lng)

    formatted_places = []
    for p in places:
        normalized = normalize_place(p, req.lat, req.lng)
        if normalized:
            formatted_places.append(normalized)

    formatted_places.sort(key=lambda x: x["distance_m"])

    return {
        "preference": req.preference,
        "count": len(formatted_places),
        "preview": formatted_places
    }
