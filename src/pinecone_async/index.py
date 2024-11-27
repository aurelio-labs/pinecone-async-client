from typing import Literal, Optional
import asyncio
import httpx
import os
from pinecone_async.exceptions import IndexNotFoundError
from pinecone_async.client import PineconeClient
from pinecone_async.schema import (
    DeleteRequest,
    FetchRequest,
    FetchResponse,
    IndexResponse,
    PineconeVector,
    QueryRequest,
    QueryResponse,
    Serverless,
    SparseValues,
    UpsertRequest,
    UpsertResponse,
    VectorMetadata,
)

class PineconeIndex:
    """
    A high-level interface for working with a specific Pinecone index.
    This class handles both control plane operations (through PineconeClient)
    and data plane operations (direct vector operations).
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        metric: Literal["cosine", "euclidean", "dotproduct"],
        dimensions: int,
        region: str,
        namespace: Optional[str] = None,
        deletion_protection: Literal["enabled", "disabled"] = "disabled",
    ):
        self.index_name = index_name
        self.metric = metric
        self.dimensions = dimensions
        self.region = region
        self.namespace = namespace
        self.deletion_protection = deletion_protection
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }
        self.client = PineconeClient(api_key=api_key)
        self.http_client = httpx.AsyncClient(headers=self.headers)
        self.index_host = None  # Set during initialization

    @classmethod
    async def create(
        cls,
        api_key: str,
        index_name: str,
        metric: Literal["cosine", "euclidean", "dotproduct"],
        dimensions: int,
        region: str,
        namespace: Optional[str] = None,
        deletion_protection: Literal["enabled", "disabled"] = "disabled",
    ):
        """Factory method to create and initialize a PineconeIndex instance."""
        self = cls(
            api_key=api_key,
            index_name=index_name,
            metric=metric,
            dimensions=dimensions,
            region=region,
            namespace=namespace,
            deletion_protection=deletion_protection,
        )
        await self._initialize_index()
        return self

    async def _initialize_index(self) -> IndexResponse:
        """Initialize or get existing index and set index_host."""
        try:
            index_response = await self.client.describe_index(self.index_name)
        except IndexNotFoundError:
            index_response = await self.client.create_index(
                name=self.index_name,
                dimension=self.dimensions,
                metric=self.metric,
                spec=Serverless(region=self.region),
                deletion_protection=self.deletion_protection,
            )

        self.index_host = index_response.host
        return index_response

    async def upsert(self, vectors: list[PineconeVector]) -> UpsertResponse:
        """Upsert vectors to the index."""
        response = await self.http_client.post(
            f"https://{self.index_host}/vectors/upsert",
            json=UpsertRequest(
                vectors=vectors,
                namespace=self.namespace
            ).model_dump(exclude_none=True),
        )
        
        if response.status_code == 200:
            return UpsertResponse(**response.json())
        else:
            raise Exception(f"Failed to upsert vectors: {response.status_code} : {response.text}")

    async def upsert_batch(
        self,
        vectors: list[PineconeVector],
        batch_size: int = 200,
        max_concurrency: int = 10,
    ):
        """Upsert vectors in batches with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def upsert_batch(batch: list[PineconeVector]):
            async with semaphore:
                return await self.upsert(batch)

        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
        tasks = [asyncio.create_task(upsert_batch(batch)) for batch in batches]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise Exception(f"Errors in batch upsert: {errors}")

    async def query(
        self,
        vector: Optional[list[float]] = None,
        vector_id: Optional[str] = None,
        sparse_vector: Optional[SparseValues] = None,
        filter: Optional[dict] = None,
        top_k: int = 5,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> QueryResponse:
        """Query the index for similar vectors."""
        response = await self.http_client.post(
            f"https://{self.index_host}/query",
            json=QueryRequest(
                vector=vector,
                id=vector_id,
                sparse_vector=sparse_vector,
                filter=filter,
                namespace=self.namespace,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
            ).model_dump(exclude_none=True),
        )
        
        if response.status_code == 200:
            return QueryResponse(**response.json())
        else:
            raise Exception(f"Failed to query index: {response.status_code} : {response.text}")

    async def fetch(self, ids: list[str]) -> FetchResponse:
        """Fetch vectors by their IDs."""
        response = await self.http_client.get(
            f"https://{self.index_host}/vectors/fetch",
            params=FetchRequest(
                ids=ids,
                namespace=self.namespace
            ).model_dump(exclude_none=True),
        )
        
        if response.status_code == 200:
            return FetchResponse(**response.json())
        else:
            raise Exception(f"Failed to fetch vectors: {response.status_code} : {response.text}")

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        delete_all: bool = False,
        filter: Optional[dict] = None,
    ) -> dict:
        """Delete vectors from the index."""
        if filter and not ids:
            # Get IDs from filter
            query_result = await self.query(
                vector=[0.0] * self.dimensions,
                filter=filter,
                top_k=10000
            )
            ids = [m.id for m in query_result.matches]
            if not ids:
                return {}

        response = await self.http_client.post(
            f"https://{self.index_host}/vectors/delete",
            json=DeleteRequest(
                ids=ids,
                delete_all=delete_all,
                namespace=self.namespace,
            ).model_dump(exclude_none=True),
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to delete vectors: {response.status_code} : {response.text}")

    async def close(self):
        """Close HTTP client connections."""
        await self.http_client.aclose()
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        

async def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    # Create and initialize index
    index = await PineconeIndex.create(
        api_key=api_key,
        index_name="test-index-simple",
        metric="cosine",
        dimensions=8,
        region="us-east-1"
    )

    try:
        # Print the index host to verify we're connected
        print(f"\n=== Index host ===")
        print(f"Host: {index.index_host}")

        test_vector = PineconeVector(
            id="test1",
            values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            metadata=VectorMetadata(content="test content")
        )
        
        print("\n=== Upserting vector ===")
        upsert_response = await index.upsert([test_vector])
        print(f"Upsert response: {upsert_response}")
        
        # Add a small delay to allow for propagation
        await asyncio.sleep(2)

        print("\n=== Querying vector ===")
        query_response = await index.query(
            vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            top_k=1,
            include_metadata=True
        )
        print(f"Query response: {query_response}")

        print("\n=== Fetching vector ===")
        fetch_response = await index.fetch(["test1"])
        print(f"Fetch response: {fetch_response}")

        print("\n=== Deleting vector ===")
        delete_response = await index.delete(ids=["test1"])
        print(f"Delete response: {delete_response}")

    finally:
        await index.close()
      
  
if __name__ == "__main__":
    asyncio.run(main())