import os
import secrets
import httpx
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_PLACES_API_KEY:
    raise RuntimeError("GOOGLE_PLACES_API_KEY is not set")

elif not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

app = FastAPI()
security = HTTPBasic()

APP_BASIC_USER = os.getenv("APP_BASIC_USER", "")
APP_BASIC_PASS = os.getenv("APP_BASIC_PASS", "")

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

Preference = Literal["study", "friendly", "best", "open", "busy"]

class RecommendationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    preference: Preference

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommendations")
async def recommendations(req: RecommendationRequest, _user: str = Depends(require_basic_auth)):
    places = await fetch_nearby_cafes(req.lat, req.lng)

    return {
        "preference": req.preference,
        "count": len(places),
        "names": [p.get("displayName", {}).get("text") for p in places]
    }
