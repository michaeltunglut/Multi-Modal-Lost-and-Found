#!/usr/bin/env python3
"""Compatibility entry point for `uvicorn main:app`."""
from app.main import app
from app.config import HOST, PORT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
