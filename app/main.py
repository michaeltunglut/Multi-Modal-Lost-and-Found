from fastapi import FastAPI
from .config import DB_PATH
from .database import init_db
from .email_notifications import send_pending_notifications
from .routes import auth, matches, pages, reports, sightings
app = FastAPI(title="Campus Lost & Found", version="1.1.0")
init_db()
send_pending_notifications()
app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(reports.router)
app.include_router(sightings.router)
app.include_router(matches.router)
@app.get("/api/health")
def health():
    return {"ok": True, "database": str(DB_PATH), "image_features": "RGB/grayscale histograms + perceptual hash", "text_features": "token and bigram cosine with synonym normalization"}
