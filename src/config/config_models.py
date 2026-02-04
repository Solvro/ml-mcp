from __future__ import annotations

from pydantic import BaseModel


class Mcp(BaseModel):
    transport: str
    port: int
    host: str


class TopwrApi(BaseModel):
    host: str
    port: int
    cors_origins: str


class Servers(BaseModel):
    mcp: Mcp
    topwr_api: TopwrApi


class Rag(BaseModel):
    max_results: int
    enable_debug: bool


class DataPipeline(BaseModel):
    max_chunk_size: int
    chunk_overlap: int
    token_limit: int


class FastModel(BaseModel):
    name: str
    temperature: float


class AccurateModel(BaseModel):
    name: str
    temperature: int


class Clarin(BaseModel):
    name: str
    base_url: str


class Gemini(BaseModel):
    name: str


class Llm(BaseModel):
    fast_model: FastModel
    accurate_model: AccurateModel
    clarin: Clarin
    gemini: Gemini


class Langfuse(BaseModel):
    host: str
    secret_key: str
    public_key: str


class Observability(BaseModel):
    langfuse: Langfuse


class Database(BaseModel):
    name: str
    uri: str
    username: str
    password: str


class Prompts(BaseModel):
    final_answer: str
    cypher_insert: str
    cypher_search: str
    guardrails: str


class Model(BaseModel):
    servers: Servers
    rag: Rag
    data_pipeline: DataPipeline
    llm: Llm
    observability: Observability
    database: Database
    nodes: list[str]
    relations: list[str]
    prompts: Prompts
