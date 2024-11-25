import asyncio
import os
from pinecone_async import PineconeClient

async def main():
    api_key = os.getenv("PINECONE_API_KEY")
    client = PineconeClient(api_key=api_key)
    
    # List indexes
    indexes = await client.list_indexes()
    print(f"Indexes: {indexes}")
    
    # Create index
    # Add other operations you want to test

if __name__ == "__main__":
    asyncio.run(main())