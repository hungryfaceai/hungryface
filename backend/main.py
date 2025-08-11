import os, json
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

# ---- Firebase Admin init (from env var) ----
# In Render, set an env var FIREBASE_SERVICE_ACCOUNT to the *entire* JSON
# of your Firebase service account (downloaded from Firebase console).
svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if not svc_json:
    raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var is not set")
cred = credentials.Certificate(json.loads(svc_json))
firebase_admin.initialize_app(cred)

app = FastAPI()

# ---- CORS (allow your GitHub Pages domain + local dev) ----
# Replace YOUR_GH_PAGES_USERNAME and YOUR_REPO with your actual values
# e.g. https://hungryfaceai.github.io/hungryface
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://hungryfaceai.github.io",        # org/user pages
    "https://hungryfaceai.github.io/hungryface",  # project pages (if applicable)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---- Dependency to verify Firebase ID token ----
async def require_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded  # contains uid, email, etc.
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/protected")
def protected(user=Depends(require_user)):
    email = user.get("email", "unknown")
    uid = user["uid"]
    return {"message": f"Hello {email}", "uid": uid}
