from typing import Any, Dict, List, Literal, Optional
import httpx
from pinecone_async.exceptions import IndexNotFoundError
from pinecone_async.schema import (
    IndexResponse, PineconeVector, Serverless, PineconePod
)

class PineconeClient:
    """Async client for Pinecone API."""
    
    base_url: str = "https://api.pinecone.io"
    
    def __init__(self, api_key: str):
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }
        self.client = httpx.AsyncClient(headers=self.headers)

    async def list_indexes(self) -> List[Dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}/indexes")
        if response.status_code != 200:
            raise Exception(f"Failed to list indexes: {response.status_code} : {response.text}")
        return response.json()

    async def describe_index(self, index_name: str) -> IndexResponse:
        response = await self.client.get(f"{self.base_url}/indexes/{index_name}")
        if response.status_code == 200:
            return IndexResponse(**response.json())
        elif response.status_code == 404:
            raise IndexNotFoundError(f"Index `{index_name}` not found")
        else:
            raise Exception(f"Failed to describe index: {response.status_code} : {response.text}")

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
                print(response.json())
                return IndexResponse(**response.json())
            else:
                raise Exception(f"Failed to create index: {response.status_code} : {response.text}")
        except Exception as e:
            raise e

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

async def main():
    import os
    import uuid
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    async with PineconeClient(api_key=api_key) as client:
       try:
           
           print("\n=== Listing all indexes ===")
           indexes = await client.list_indexes()
           print(indexes)

           
           name = "test-index-" + str(uuid.uuid4())[:4]
           print("\n=== Creating new index ===")
           new_index = await client.create_index(
               name=name,
               dimension=1536,
               metric="cosine",
               spec=Serverless(
                   cloud="aws",
                   region="us-east-1"
               )
           )


           
           print("\n=== Describing index ===")
           index_details = await client.describe_index(name)
           print(index_details)

       except Exception as e:
           print(f"Error occurred: {e}")

if __name__ == "__main__":
   import asyncio
   from pinecone_async.schema import Serverless
   asyncio.run(main())
