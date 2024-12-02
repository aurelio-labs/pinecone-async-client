import asyncio
import os
from pinecone_async import PineconeClient

async def list_indexes_example():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    client = PineconeClient(api_key=api_key)
    
    indexes = await client.list_indexes()
    print(f"Indexes: {indexes}")

async def rerank_example():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    client = PineconeClient(api_key=api_key)
    
    result = await client.rerank(
        model="bge-reranker-v2-m3",
        query="The tech company Apple is known for its innovative products like the iPhone.",
        documents=[
            {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "vec2", "text": "Many people enjoy eating apples as a healthy snack."},
            {"id": "vec3", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
            {"id": "vec4", "text": "An apple a day keeps the doctor away, as the saying goes."},
        ],
        top_n=4,
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(rerank_example())
