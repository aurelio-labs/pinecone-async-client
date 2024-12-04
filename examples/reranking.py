import asyncio
import os
from pinecone_async.client import PineconeClient

async def reranking_example():
    # Initialize the client with your API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    client = PineconeClient(api_key=api_key)

    try:
        # Example documents about different topics
        documents = [
            {
                "id": "1",
                "text": "Python is a high-level, interpreted programming language known for its simplicity and readability."
            },
            {
                "id": "2",
                "text": "JavaScript is a programming language that enables interactive web pages and is an essential part of web applications."
            },
            {
                "id": "3",
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience."
            },
            {
                "id": "4",
                "text": "The Python programming language was created by Guido van Rossum and was first released in 1991."
            },
            {
                "id": "5",
                "text": "Deep learning is a type of machine learning based on artificial neural networks."
            }
        ]

        # Print available models
        print("\nAvailable reranking models:")
        print(PineconeClient.list_supported_models())

        # Example 1: Basic reranking with default model
        print("\nExample 1: Basic reranking with default model (cohere-rerank-3.5)")
        query = "What is Python programming?"
        results = await client.rerank(
            query=query,
            documents=documents
        )
        
        print(f"\nQuery: {query}")
        print("\nRanked results:")
        for result in results.data:
            print(f"Score: {result.score:.4f} - {result.document.text}")

        # Example 2: Reranking with a different model and top_n
        print("\nExample 2: Reranking with BGE model and top 3 results")
        query = "Tell me about artificial intelligence"
        results = await client.rerank(
            query=query,
            documents=documents,
            model="bge-reranker-v2-m3",
            top_n=3
        )
        
        print(f"\nQuery: {query}")
        print("\nTop 3 ranked results:")
        for result in results.data:
            print(f"Score: {result.score:.4f} - {result.document.text}")

        # Example 3: Reranking with parameters
        print("\nExample 3: Reranking with truncation parameter")
        results = await client.rerank(
            query=query,
            documents=documents,
            parameters={"truncate": "START"},
            return_documents=True
        )
        
        print(f"\nQuery: {query}")
        print("\nRanked results with START truncation:")
        for result in results.data:
            print(f"Score: {result.score:.4f} - {result.document.text}")

        # Print usage information
        print("\nUsage information:")
        print(f"Rerank units used: {results.usage.rerank_units}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(reranking_example())