from typing import Any, Literal

import aiohttp
from pinecone_async.exceptions import IndexNotFoundError
from pinecone_async.schema import IndexResponse, PineconePod, Serverless


class PineconeClient:
    """
    A client for the Pinecone API.
    """

    base_url: str = "https://api.pinecone.io"

    def __init__(self, api_key: str):
        self.base_url = "https://api.pinecone.io"
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }

    # ------------------
    # # Control Plane API
    # ------------------
    async def _get_index_host(self, index_name: str) -> str:
        index_response = await self.describe_index(index_name)
        return index_response.host

    async def list_indexes(self) -> list[dict[str, Any]]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f"{self.base_url}/indexes") as response:
                return await response.json(content_type=None)

    async def create_index(
        self,
        name: str,
        dimension: int,
        metric: Literal["cosine", "euclidean", "dotproduct"],
        spec: Serverless | PineconePod,
        deletion_protection: Literal["enabled", "disabled"] = "disabled",
    ) -> IndexResponse:
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

        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(
                f"{self.base_url}/indexes",
                json=create_index_request,
            ) as response:
                if response.status in [200, 201]:
                    return IndexResponse(**(await response.json()))
                else:
                    error_message = await response.text()
                    raise Exception(f"Failed to create index: {response.status} : {error_message}")

    async def describe_index(self, index_name: str) -> IndexResponse:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f"{self.base_url}/indexes/{index_name}") as response:
                if response.status == 200:
                    return IndexResponse(**(await response.json()))
                elif response.status == 404:
                    raise IndexNotFoundError(f"Index `{index_name}` not found")
                else:
                    error_message = await response.text()
                    raise Exception(
                        f"Failed to describe index: {response.status} : {error_message}"
                    )