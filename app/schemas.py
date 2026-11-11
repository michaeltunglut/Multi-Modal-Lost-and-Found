from pydantic import BaseModel

class MatchUpdate(BaseModel):
    status: str
