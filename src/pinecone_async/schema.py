from pydantic import BaseModel
from typing import Optional, Dict, Any

class Serverless(BaseModel):
    cloud: Optional[str] = None
    region: Optional[str] = None

class PineconePod(BaseModel):
    environment: str
    replicas: Optional[int] = None
    shards: Optional[int] = None
    pod_type: Optional[str] = None

class IndexResponse(BaseModel):
    database: Dict[str, Any]
    status: Dict[str, Any]
    host: str