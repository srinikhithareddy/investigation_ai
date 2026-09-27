from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    document_id: str
    title: str
    type: str
    date: Optional[str] = None
    version: Optional[str] = None
    content: str


class ContradictionItem(BaseModel):
    title: str
    description: str


class InvestigateResponse(BaseModel):
    answer: str
    confidence: str
    evidence: List[EvidenceItem] = []
    contradictions: List[ContradictionItem] = []
    trace: List[str] = []


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str] = Field(default_factory=dict)
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None
