from typing import List
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class Relationship(BaseModel):
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)


class KGExtraction(BaseModel):
    entities: List[Entity] = []
    relationships: List[Relationship] = []


class Source(BaseModel):
    source_type: str
    source_id: str


class RAGAnswer(BaseModel):
    answer: str = Field(min_length=1)
    sources: List[Source] = []