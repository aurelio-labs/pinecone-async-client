from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel

# Base Models
class SparseValues(BaseModel):
    """Representation of sparse vector values"""
    indices: List[int]
    values: List[float]

class VectorMetadata(BaseModel):
    """Metadata associated with a vector"""
    content: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    token_count: Optional[int] = None

class PineconeVector(BaseModel):
    """A vector in Pinecone with its metadata"""
    id: str
    values: List[float]
    sparse_values: Optional[SparseValues] = None
    metadata: Optional[VectorMetadata] = None

# Control Plane Models
class Serverless(BaseModel):
    """Serverless configuration for Pinecone indexes"""
    cloud: Literal["aws", "gcp", "azure"] = "aws"
    region: str

class PineconePod(BaseModel):
    """Pod configuration for Pinecone indexes"""
    environment: str
    replicas: Optional[int] = None
    shards: Optional[int] = None
    pod_type: Optional[str] = None

class IndexStatus(BaseModel):
    """Status of a Pinecone index"""
    ready: bool
    state: str

class IndexResponse(BaseModel):
    """Response from index operations"""
    name: str
    metric: Literal["cosine", "euclidean", "dotproduct"]
    dimension: int
    status: Dict[str, Any]
    host: str
    spec: Dict[str, Any]
    deletion_protection: Literal["enabled", "disabled"]

# Data Plane Request Models
class UpsertRequest(BaseModel):
    """Request to upsert vectors"""
    vectors: List[PineconeVector]
    namespace: Optional[str] = None

class UpdateRequest(BaseModel):
    """Request to update a vector"""
    id: str
    values: Optional[List[float]] = None
    sparse_values: Optional[SparseValues] = None
    set_metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

class QueryRequest(BaseModel):
    """Request to query vectors"""
    vector: Optional[List[float]] = None
    id: Optional[str] = None
    sparse_vector: Optional[SparseValues] = None
    filter: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    top_k: int = 5
    include_values: bool = False
    include_metadata: bool = False

class ListRequest(BaseModel):
    """Request to list vectors"""
    prefix: Optional[str] = None
    limit: Optional[int] = None
    pagination_token: Optional[str] = None
    namespace: Optional[str] = None

class DeleteRequest(BaseModel):
    """Request to delete vectors"""
    ids: Optional[List[str]] = None
    delete_all: bool = False
    namespace: Optional[str] = None

class FetchRequest(BaseModel):
    """Request to fetch vectors"""
    ids: List[str]
    namespace: Optional[str] = None

# Data Plane Response Models
class UpsertResponse(BaseModel):
    """Response from upsert operation"""
    upsertedCount: int

class Match(BaseModel):
    """A matching vector from a query"""
    id: str
    score: float
    values: Optional[List[float]] = None
    sparse_values: Optional[SparseValues] = None
    metadata: Optional[VectorMetadata] = None

class QueryResponse(BaseModel):
    """Response from query operation"""
    matches: List[Match]
    namespace: Optional[str] = None

class VectorEntry(BaseModel):
    """A vector entry from list operation"""
    id: str

class ListResponse(BaseModel):
    """Response from list operation"""
    vectors: List[VectorEntry]
    namespace: Optional[str] = None

class FetchResponse(BaseModel):
    """Response from fetch operation"""
    vectors: Dict[str, PineconeVector]
    namespace: Optional[str] = None
    
class Document(BaseModel):
    """A document for reranking."""
    id: str
    text: str
    source: Optional[str] = None
    custom_field: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RerankParameters(BaseModel):
    """Parameters for reranking operation."""
    truncate: Optional[Literal["START", "END", "NONE"]] = "END"
    chunk_size: Optional[int] = None

class RerankRequest(BaseModel):
    """Request structure for reranking operation."""
    model: str
    query: str
    documents: List[Document]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True
    parameters: Optional[RerankParameters] = None
    rank_fields: Optional[List[str]] = None

class RerankResult(BaseModel):
    """Individual result from reranking operation."""
    index: int
    document: Optional[Document] = None
    score: float

class RerankUsage(BaseModel):
    rerank_units: int

class RerankResponse(BaseModel):
    data: list[RerankResult]
    usage: RerankUsage
