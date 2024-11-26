# Pinecone Async Client

An async Python client for Pinecone vector database.

## Installation

```bash
pip install pinecone-async-client
```

## Usage

```python
import asyncio
import os
from pinecone_async import PineconeClient, Serverless

async def main():
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    async with PineconeClient(api_key=api_key) as client:
        try:
            # List all indexes
            print("\n=== Listing all indexes ===")
            indexes = await client.list_indexes()
            print(indexes)

            # Create a new index
            print("\n=== Creating new index ===")
            new_index = await client.create_index(
                name="test-index",
                dimension=1536,
                metric="cosine",
                spec=Serverless(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(new_index)

            # Describe specific index
            print("\n=== Describing index ===")
            index_details = await client.describe_index("test-index")
            print(index_details)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- Full async support with context managers
- Type hints and Pydantic models
- Python 3.10+
- Simple API with clear error handling

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pinecone-async-client
cd pinecone-async-client

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## API Reference

### PineconeClient

```python
from pinecone_async import PineconeClient

client = PineconeClient(api_key="your-api-key")
```

#### Methods

- `list_indexes()`: List all available indexes
- `create_index(name, dimension, metric, spec)`: Create a new index
- `describe_index(index_name)`: Get details about a specific index

### Models

```python
from pinecone_async import Serverless, PineconePod

# Serverless configuration
serverless = Serverless(cloud="aws", region="us-east-1")

# Pod configuration
pod = PineconePod(
    environment="production",
    replicas=1,
    shards=1,
    pod_type="p1.x1"
)
```

## Requirements

- Python 3.10+
- httpx>=0.27.2
- pydantic>=2.0.0

## License

MIT License
