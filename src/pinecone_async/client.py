import os
from typing import Any, Dict, List, Literal

import httpx

from pinecone_async.exceptions import IndexNotFoundError
from pinecone_async.schema import IndexResponse, PineconePod, Serverless


class PineconeClient:
    """Async client for Pinecone API."""

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
