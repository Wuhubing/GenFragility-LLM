from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal

class CandidateTriple(BaseModel):
    head: str
    relation: str
    tail: str
    tail_type: Literal["entity","literal"]
    question: Optional[str] = None # To support downstream probing
    as_of_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    evidence: List[str] = []
    generator: Optional[str] = None

    @validator("start_time","end_time","as_of_date")
    def _check_date(cls, v):
        if v and len(v) not in (4,7,10):  # YYYY / YYYY-MM / YYYY-MM-DD
            raise ValueError("bad date")
        return v
