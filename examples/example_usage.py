import asyncio
import os
from pinecone_async.client import PineconeClient

async def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    client = PineconeClient(api_key=api_key)
    
    indexes = await client.list_indexes()
    print(f"Indexes: {indexes}")


if __name__ == "__main__":
    asyncio.run(main())