import json
import logging
import re
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from ..config import ALLOWED, CATEGORIES, DB_PATH, MATCH_MIN_SCORE, MAX_UPLOAD, UPLOAD_DIR
from ..database import db, now
from ..email_notifications import notify_match_owner
from ..features import candidate_matches, extract_image_features, report_dict
from ..security import admin_user, optional_user, signed_in_user
router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/api/reports")
def list_reports(report_type: str | None = Query(default=None, alias="type"), user=Depends(optional_user)):
    kind = (report_type or "").upper()
    if kind == "FOUND" and (not user or user["role"] != "ADMIN"):
        raise HTTPException(status_code=403, detail="Found reports are private to matched lost-report owners and admins.")
    sql = "SELECT * FROM reports"
    args = []
    if kind in ("LOST", "FOUND"):
        sql += " WHERE report_type=?"; args.append(kind)
    elif not user or user["role"] != "ADMIN":
        sql += " WHERE report_type='LOST'"
    sql += " ORDER BY created_at DESC LIMIT 200"
    with db() as c:
        rows = c.execute(sql, args).fetchall()
    result=[]
    for row in rows:
        item = report_dict(row)
        item["can_view_matches"] = bool(user and (user["role"] == "ADMIN" or row["reporter_id"] == user["id"]))
        item["can_report_sighting"] = not (user and row["reporter_id"] == user["id"] and user["role"] != "ADMIN")
        result.append(item)
    return result

@router.delete("/api/reports/{report_id}")
def delete_report(report_id: int, user=Depends(admin_user)):
    with db() as c:
        report = c.execute("SELECT image_path FROM reports WHERE id=?", (report_id,)).fetchone()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found.")
        image_path = report["image_path"]
        sighting_paths = [row["image_path"] for row in c.execute("SELECT image_path FROM sightings WHERE lost_report_id=?", (report_id,)).fetchall()]
        c.execute("DELETE FROM reports WHERE id=?", (report_id,))
    # Cascades remove matches and sightings; remove their files as well.
    for path in [image_path, *sighting_paths]:
        if re.fullmatch(r"[a-f0-9]{32}\.(?:jpg|png|webp)", path):
            (UPLOAD_DIR / path).unlink(missing_ok=True)
    return {"ok": True, "deleted_report_id": report_id}

@router.get("/api/reports/{report_id}/matches")
def get_matches(report_id: int, user=Depends(signed_in_user)):
    with db() as c:
        report = c.execute("SELECT id,report_type,reporter_id FROM reports WHERE id=?", (report_id,)).fetchone()
    if not report or report["report_type"] != "LOST":
        raise HTTPException(status_code=404, detail="Lost report not found.")
    if user["role"] != "ADMIN" and report["reporter_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Lost report not found.")
    return candidate_matches(report_id)

@router.get("/uploads/{filename}")
def get_upload(filename: str, user=Depends(optional_user)):
    if not re.fullmatch(r"[a-f0-9]{32}\.(?:jpg|png|webp)", filename):
        raise HTTPException(status_code=404, detail="Image not found.")
    with db() as c:
        report = c.execute("SELECT id,report_type FROM reports WHERE image_path=?", (filename,)).fetchone()
        if report:
            if report["report_type"] == "FOUND":
                if not user:
                    raise HTTPException(status_code=404, detail="Image not found.")
                if user["role"] != "ADMIN":
                    linked = c.execute("""SELECT 1 FROM matches m JOIN reports lost ON lost.id=m.lost_report_id
                                         WHERE m.found_report_id=? AND lost.reporter_id=? AND m.final_score>=? LIMIT 1""", (report["id"],user["id"],MATCH_MIN_SCORE)).fetchone()
                    if not linked:
                        raise HTTPException(status_code=404, detail="Image not found.")
        else:
            sighting = c.execute("""SELECT s.id,l.reporter_id FROM sightings s
                                    JOIN reports l ON l.id=s.lost_report_id WHERE s.image_path=?""", (filename,)).fetchone()
            if not sighting or not user or (user["role"] != "ADMIN" and sighting["reporter_id"] != user["id"]):
                raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(UPLOAD_DIR / filename)

@router.post("/api/reports", status_code=201)
async def create_report(
    background_tasks: BackgroundTasks,
    report_type: str = Form(...), title: str = Form(...), description: str = Form(...), category: str = Form(...),
    image: UploadFile = File(...), colour: str = Form(default=""), brand: str = Form(default=""), contact_method: str = Form(default=""),
    location: str = Form(default=""), event_date: str = Form(default=""), user=Depends(optional_user),
):
    kind = report_type.strip().upper()
    if kind not in ("LOST", "FOUND"):
        raise HTTPException(status_code=400, detail="Choose whether the item is lost or found.")
    contact_method = contact_method.strip()
    if kind == "FOUND" and not 5 <= len(contact_method) <= 120:
        raise HTTPException(status_code=400, detail="Enter a phone number, email, or other contact method (5–120 characters).")
    if kind == "LOST":
        contact_method = ""
    if kind == "LOST" and not user:
        raise HTTPException(status_code=401, detail="Sign in before posting a lost report so its found matches stay private to you.")
    if not title.strip() or not description.strip():
        raise HTTPException(status_code=400, detail="Title and description are required.")
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="Choose a valid category.")
    if image.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="Use a JPEG, PNG, or WebP image.")
    raw = await image.read(MAX_UPLOAD + 1)
    if not raw: raise HTTPException(status_code=400, detail="Upload a photo of the item.")
    if len(raw) > MAX_UPLOAD: raise HTTPException(status_code=400, detail="Image must be 10 MB or smaller.")
    try:
        features, processed = extract_image_features(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    ext = {"image/jpeg":".jpg","image/png":".png","image/webp":".webp"}[image.content_type]
    safe_name = uuid.uuid4().hex + ext
    processed.save(UPLOAD_DIR / safe_name, format={".jpg":"JPEG",".png":"PNG",".webp":"WEBP"}[ext], quality=90)
    created = now()
    report_id = None
    try:
        with db() as c:
            cur = c.execute("""INSERT INTO reports(report_type,title,description,category,colour,brand,contact_method,location,event_date,image_path,image_features,created_at,reporter_id)
                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (kind,title.strip(),description.strip(),category,colour.strip(),brand.strip(),contact_method,location.strip(),event_date.strip(),safe_name,json.dumps(features),created,user["id"] if user else None))
            report_id = cur.lastrowid
        matches = candidate_matches(report_id)
        with db() as c:
            for match in matches:
                lost_id = report_id if kind == "LOST" else match["report"]["id"]
                found_id = match["report"]["id"] if kind == "LOST" else report_id
                c.execute("INSERT OR IGNORE INTO matches(lost_report_id,found_report_id,image_score,text_score,category_score,final_score,created_at) VALUES(?,?,?,?,?,?,?)", (lost_id,found_id,match["image_score"],match["text_score"],match["category_score"],match["final_score"],created))
        matching_reports = candidate_matches(report_id)
        for match in matching_reports:
            lost_id = report_id if kind == "LOST" else match["report"]["id"]
            found_id = match["report"]["id"] if kind == "LOST" else report_id
            try:
                notify_match_owner(lost_id, found_id, match["final_score"], background_tasks)
            except Exception:
                logger.exception("Could not queue match email for lost report %s", lost_id)
        private_matches = matching_reports if kind == "LOST" else []
        return {"report":{"id":report_id,"report_type":kind},"matches":private_matches}
    except Exception:
        if report_id is None: (UPLOAD_DIR / safe_name).unlink(missing_ok=True)
        raise
