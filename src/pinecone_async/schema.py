from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel

class Serverless(BaseModel):
    cloud: Literal["aws", "gcp", "azure"] = "aws"
    region: str

class PineconePod(BaseModel):
    environment: str
    replicas: Optional[int] = None
    shards: Optional[int] = None
    pod_type: Optional[str] = None

class IndexStatus(BaseModel):
    ready: bool
    state: str

class IndexResponse(BaseModel):
    name: str
    metric: str
    dimension: int
    status: Dict[str, Any]
    host: str
    spec: Dict[str, Any]
    deletion_protection: str

class SparseValues(BaseModel):
    indices: List[int]
    values: List[float]

class VectorMetadata(BaseModel):
    content: Optional[str] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None

class PineconeVector(BaseModel):
    id: str
    values: List[float]
    sparse_values: Optional[SparseValues] = None
    metadata: Optional[VectorMetadata] = None