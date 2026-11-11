from fastapi import APIRouter, Depends, HTTPException
from ..config import MATCH_MIN_SCORE
from ..database import db
from ..schemas import MatchUpdate
from ..security import signed_in_user
router = APIRouter()

@router.patch("/api/matches/{match_id}")
def update_match(match_id: int, update: MatchUpdate, user=Depends(signed_in_user)):
    if update.status not in ("REVIEWED","REJECTED","RESOLVED"):
        raise HTTPException(status_code=400, detail="Invalid match status.")
    with db() as c:
        match = c.execute("SELECT lost_report_id,final_score FROM matches WHERE id=?", (match_id,)).fetchone()
        if not match: raise HTTPException(status_code=404, detail="Match not found.")
        if user["role"] != "ADMIN" and match["final_score"] < MATCH_MIN_SCORE:
            raise HTTPException(status_code=404, detail="Match not found.")
        report = c.execute("SELECT reporter_id FROM reports WHERE id=?", (match["lost_report_id"],)).fetchone()
        if user["role"] != "ADMIN" and report["reporter_id"] != user["id"]:
            raise HTTPException(status_code=404, detail="Match not found.")
        c.execute("UPDATE matches SET status=? WHERE id=?", (update.status, match_id))
    return {"ok":True,"status":update.status}
