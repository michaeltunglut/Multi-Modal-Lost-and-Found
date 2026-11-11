from fastapi import APIRouter
from fastapi.responses import FileResponse
from ..config import WEB_DIR
router = APIRouter()

@router.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")

@router.get("/reports")
def reports_page():
    return FileResponse(WEB_DIR / "reports.html")

@router.get("/account")
def account_page():
    return FileResponse(WEB_DIR / "account.html")

@router.get("/sightings/new")
def new_sighting_page():
    return FileResponse(WEB_DIR / "sighting.html")
