from pinecone_async.client import PineconeClient


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
                spec=Serverless(cloud="aws", region="us-east-1"),
            )
            print(new_index)

            print("\n=== Describing index ===")
            index_details = await client.describe_index(name)
            print(index_details)

        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    import asyncio

    from pinecone_async.schema import Serverless

    asyncio.run(main())
