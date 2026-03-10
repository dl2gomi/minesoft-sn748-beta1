from pydantic import BaseModel

class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str