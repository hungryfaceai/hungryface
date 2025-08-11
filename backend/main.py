# main.py
import os, json, mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

# ===== Firebase Admin init (Render env var) =====
# In Render: Settings → Environment → add FIREBASE_SERVICE_ACCOUNT with the full JSON
svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if not svc_json:
    raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var is not set")
cred = credentials.Certificate(json.loads(svc_json))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

app = FastAPI()

# ===== CORS (allow only your site + local dev) =====
allowed_origins = [
    "https://hungryfaceai.github.io",  # your GitHub Pages origin
    "http://localhost:3000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ===== Auth dependency =====
async def require_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded  # contains uid, email, email_verified, etc.
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ===== Health check =====
@app.get("/healthz")
def healthz():
    return {"ok": True}

# ===== Example protected ping =====
@app.get("/protected")
def protected(user=Depends(require_user)):
    return {"message": f"Hello {user.get('email','unknown')}", "uid": user["uid"]}

# ===== Secure static assets under crydetector/private_assets =====
BASE_DIR = Path(__file__).parent
PRIVATE_ROOT = (BASE_DIR / "crydetector" / "private_assets").resolve()

def _safe_join(base: Path, subpath: str) -> Path:
    """Prevent path traversal and pin to PRIVATE_ROOT."""
    candidate = (base / subpath).resolve()
    if not str(candidate).startswith(str(base)):
        # e.g., attempts like ../../etc/passwd
        raise HTTPException(403, "Forbidden path")
    return candidate

@app.get("/assets/crydetector/{subpath:path}")
def get_crydetector_asset(subpath: str, user=Depends(require_user)):
    """
    Serve files ONLY if the caller is authenticated.
    Example URL:
      https://YOUR-SERVICE.onrender.com/assets/crydetector/model.onnx
      https://YOUR-SERVICE.onrender.com/assets/crydetector/app.bundle.js
    """
    target = _safe_join(PRIVATE_ROOT, subpath)

    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Not found")

    media_type, _ = mimetypes.guess_type(str(target))
    headers = {
        # You can raise the cache time once you're confident in auth flow.
        "Cache-Control": "private, max-age=60",
        "X-User-Uid": user["uid"],
    }
    return FileResponse(target, media_type=media_type, headers=headers)

# ===== Optional: directory listing for debugging (remove in prod) =====
@app.get("/assets/crydetector")
def list_crydetector_assets(user=Depends(require_user)):
    if not PRIVATE_ROOT.exists():
        return JSONResponse({"files": []})
    files = []
    for p in PRIVATE_ROOT.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(PRIVATE_ROOT)).replace("\\", "/"))
    return {"files": files}
