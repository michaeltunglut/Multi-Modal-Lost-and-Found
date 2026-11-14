import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from ..config import ALLOWED, MATCH_MIN_SCORE, MAX_UPLOAD, UPLOAD_DIR
from ..database import db, now
from ..email_notifications import notify_sighting_owner
from ..features import cosine_text, extract_image_features, image_similarity
from ..security import optional_user, signed_in_user
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/reports/{report_id}/sightings", status_code=201)
async def create_sighting(
    report_id: int,
    background_tasks: BackgroundTasks,
    description: str = Form(...),
    location: str = Form(...),
    event_date: str = Form(default=""),
    contact_method: str = Form(...),
    image: UploadFile = File(...),
    user=Depends(optional_user),
):
    with db() as c:
        lost = c.execute("SELECT * FROM reports WHERE id=? AND report_type='LOST' AND status IN ('OPEN','UNDER_REVIEW')", (report_id,)).fetchone()
    if not lost:
        raise HTTPException(status_code=404, detail="Lost report not found.")
    if user and user["role"] != "ADMIN" and lost["reporter_id"] == user["id"]:
        raise HTTPException(status_code=400, detail="You cannot submit a sighting on your own lost report.")
    description = description.strip()
    location = location.strip()
    event_date = event_date.strip()
    contact_method = contact_method.strip()
    if not description or len(description) > 1200:
        raise HTTPException(status_code=400, detail="Describe what you saw (up to 1,200 characters).")
    if not location or len(location) > 200:
        raise HTTPException(status_code=400, detail="Enter where and when you saw the item.")
    if not 5 <= len(contact_method) <= 120:
        raise HTTPException(status_code=400, detail="Enter a phone number, email, or other contact method (5–120 characters).")
    if image.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="Use a JPEG, PNG, or WebP photo.")
    raw = await image.read(MAX_UPLOAD + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Upload a photo of the sighting.")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=400, detail="Image must be 10 MB or smaller.")
    try:
        features, processed = extract_image_features(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    lost_features = json.loads(lost["image_features"])
    lost_text = " ".join([lost["title"], lost["description"], lost["colour"], lost["brand"]])
    image_score = image_similarity(lost_features, features)
    text_score = cosine_text(lost_text, description)
    score = 0.60 * image_score + 0.40 * text_score
    if score < MATCH_MIN_SCORE:
        raise HTTPException(
            status_code=422,
            detail=f"This sighting scored {score:.0%} overall (image {image_score:.0%}; description {text_score:.0%}), below the 60% minimum. Add a clearer photo and describe the item’s identifying details.",
        )

    features_json = json.dumps(features)
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(timespec="seconds")
    reporter_id = user["id"] if user else None
    with db() as c:
        duplicate = c.execute("""SELECT id,similarity_score,image_score,text_score,status FROM sightings
                                WHERE lost_report_id=? AND description=? AND location=? AND event_date=?
                                  AND contact_method=? AND image_features=? AND created_at>=? AND reporter_id IS ?
                                ORDER BY created_at DESC LIMIT 1""",
                             (report_id,description,location,event_date,contact_method,features_json,cutoff,reporter_id)).fetchone()
    if duplicate:
        return {"id":duplicate["id"],"lost_report_id":report_id,"similarity_score":duplicate["similarity_score"],"image_score":duplicate["image_score"],"text_score":duplicate["text_score"],"status":duplicate["status"],"duplicate":True}

    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[image.content_type]
    filename = uuid.uuid4().hex + ext
    processed.save(UPLOAD_DIR / filename, format={".jpg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[ext], quality=90)
    try:
        with db() as c:
            cur = c.execute("""INSERT INTO sightings(lost_report_id,description,location,event_date,contact_method,image_path,image_features,similarity_score,image_score,text_score,created_at,reporter_id)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (report_id,description,location,event_date,contact_method,filename,features_json,round(score,4),round(image_score,4),round(text_score,4),now(),reporter_id))
            sighting_id = cur.lastrowid
    except Exception:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)
        raise
    try:
        notify_sighting_owner(report_id, sighting_id, round(score, 4), background_tasks)
    except Exception:
        logger.exception("Could not queue sighting email for lost report %s", report_id)
    return {"id": sighting_id, "lost_report_id": report_id, "similarity_score": round(score,4), "image_score": round(image_score,4), "text_score": round(text_score,4), "status": "PENDING"}

@router.get("/api/reports/{report_id}/sightings")
def list_sightings(report_id: int, user=Depends(signed_in_user)):
    with db() as c:
        lost = c.execute("SELECT id,reporter_id FROM reports WHERE id=? AND report_type='LOST'", (report_id,)).fetchone()
        if not lost or (user["role"] != "ADMIN" and lost["reporter_id"] != user["id"]):
            raise HTTPException(status_code=404, detail="Lost report not found.")
        rows = c.execute("SELECT * FROM sightings WHERE lost_report_id=? ORDER BY created_at DESC", (report_id,)).fetchall()
    return [{"id":r["id"],"description":r["description"],"location":r["location"],"event_date":r["event_date"],"contact_method":r["contact_method"],"image_url":f"/uploads/{r['image_path']}","similarity_score":r["similarity_score"],"image_score":r["image_score"],"text_score":r["text_score"],"status":r["status"],"created_at":r["created_at"]} for r in rows]

class SightingUpdate(BaseModel):
    status: str

@router.patch("/api/sightings/{sighting_id}")
def update_sighting(sighting_id: int, update: SightingUpdate, user=Depends(signed_in_user)):
    if update.status not in ("REVIEWED", "REJECTED", "RESOLVED"):
        raise HTTPException(status_code=400, detail="Invalid sighting action.")
    with db() as c:
        sighting = c.execute("""SELECT s.id,s.lost_report_id,l.reporter_id
                                FROM sightings s JOIN reports l ON l.id=s.lost_report_id
                                WHERE s.id=? AND l.report_type='LOST'""", (sighting_id,)).fetchone()
        if not sighting or (user["role"] != "ADMIN" and sighting["reporter_id"] != user["id"]):
            raise HTTPException(status_code=404, detail="Sighting not found.")
        if update.status == "RESOLVED":
            c.execute("UPDATE reports SET status='RESOLVED' WHERE id=?", (sighting["lost_report_id"],))
            c.execute("UPDATE sightings SET status='RESOLVED' WHERE lost_report_id=? AND status IN ('PENDING','REVIEWED')", (sighting["lost_report_id"],))
        else:
            c.execute("UPDATE sightings SET status=? WHERE id=?", (update.status,sighting_id))
    return {"ok":True,"status":update.status}
