import os
import secrets
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

load_dotenv()

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

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/recommendations")
def recommendations(_user: str = Depends(require_basic_auth)):
    #dummy response
    return {"status": "authorized", "message": "Here are your recommendations."}