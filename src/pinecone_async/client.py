from typing import Any, Dict, List, Literal, Optional
import os
import httpx
from pinecone_async.exceptions import IndexNotFoundError
from pinecone_async.schema import IndexResponse, PineconePod, Serverless


class PineconeClient:
    """Async client for Pinecone API."""
    
    RERANK_MODELS = {
        "cohere-rerank-3.5": "cohere-rerank-3.5",
        "bge-reranker-base": "bge-reranker-base",
        "bge-reranker-large": "bge-reranker-large",
        "bge-reranker-v2-m3": "bge-reranker-v2-m3"
    }
    DEFAULT_MODEL = "cohere-rerank-3.5"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.pinecone.io",
    ):
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "api_key is required, "
                "either pass it as an argument or set the PINECONE_API_KEY environment variable"
            )

        self.base_url = base_url
        self.client = httpx.AsyncClient(headers=self.headers)
        

    async def list_indexes(self) -> List[Dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}/indexes")
        if response.status_code != 200:
            raise Exception(
                f"Failed to list indexes: {response.status_code} : {response.text}"
            )
        return response.json()

    async def describe_index(self, index_name: str) -> IndexResponse:
        response = await self.client.get(f"{self.base_url}/indexes/{index_name}")
        match response.status_code:
            case 200:
                return IndexResponse(**response.json())
            case 404:
                raise IndexNotFoundError(f"Index `{index_name}` not found")
            case _:
                raise Exception(
                    f"Failed to describe index: {response.status_code} : {response.text}"
                )

    async def create_index(
        self,
        name: str,
        dimension: int,
        metric: Literal["cosine", "euclidean", "dotproduct"],
        spec: Serverless | PineconePod,
        deletion_protection: Literal["enabled", "disabled"] = "disabled",
    ) -> IndexResponse:
        try:
            match spec:
                case Serverless():
                    spec_dict = {"serverless": spec.model_dump(exclude_none=True)}
                case PineconePod():
                    spec_dict = {"pod": spec.model_dump(exclude_none=True)}
                case _:
                    raise ValueError("spec must be either Serverless or Pod")

            create_index_request = {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "spec": spec_dict,
                "deletion_protection": deletion_protection,
            }

            response = await self.client.post(
                f"{self.base_url}/indexes",
                json=create_index_request,
            )

            if response.status_code in [200, 201]:
                return IndexResponse(**response.json())
            else:
                raise Exception(
                    f"Failed to create index: {response.status_code} : {response.text}"
                )
        except Exception as e:
            raise e

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, str]],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = True,
        parameters: Optional[dict] = None,
        rank_fields: Optional[list[str]] = None
    ) -> RerankResponse:
        """
        Rerank documents based on their relevance to a query.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        # Use default model if none specified or validate provided model
        if model is None:
            model = self.DEFAULT_MODEL
        elif model not in self.RERANK_MODELS:
            raise ValueError(
                f"Invalid model. Supported models are: {', '.join(self.RERANK_MODELS.keys())}"
            )

        headers = {
            "Api-Key": self.headers["Api-Key"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Pinecone-API-Version": "2024-10"
        }

        request = RerankRequest(
            model=model,
            query=query,
            documents=[Document(**doc) for doc in documents],
            top_n=top_n,
            return_documents=return_documents,
            parameters=RerankParameters(**(parameters or {})),
            rank_fields=rank_fields
        )

        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.post(
                "https://api.pinecone.io/rerank",
                json=request.model_dump(exclude_none=True)
            )
            
            if response.status_code == 200:
                return RerankResponse(**response.json())
            else:
                raise Exception(f"Failed to rerank: {response.status_code} : {response.text}")

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """Returns a list of supported reranking models."""
        return list(cls.RERANK_MODELS.keys())
