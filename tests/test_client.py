import pytest
from unittest.mock import patch, AsyncMock
import httpx
from pinecone_async import PineconeClient, Serverless, IndexNotFoundError

@pytest.mark.asyncio
async def test_client_initialization():
    client = PineconeClient(api_key="test-key")
    assert client.headers["Api-Key"] == "test-key"
    assert client.headers["X-Pinecone-API-Version"] == "2024-07"
    await client.close()

@pytest.mark.asyncio
async def test_list_indexes_success():
    mock_response = {
        'indexes': [{
            'name': 'test-index-a06c',
            'metric': 'cosine',
            'dimension': 1536,
            'status': {'ready': True, 'state': 'Ready'},
            'host': 'test-index-a06c-b0ed6df.svc.aped-4627-b74a.pinecone.io',
            'spec': {'serverless': {'region': 'us-east-1', 'cloud': 'aws'}},
            'deletion_protection': 'disabled'
        }]
    }
    response = httpx.Response(200, json=mock_response)
    
    with patch('httpx.AsyncClient.get', return_value=response):
        async with PineconeClient(api_key="test-key") as client:
            result = await client.list_indexes()
            assert result == mock_response

@pytest.mark.asyncio
async def test_list_indexes_error():
    error_response = httpx.Response(401, text="Unauthorized")
    
    with patch('httpx.AsyncClient.get', return_value=error_response):
        async with PineconeClient(api_key="wrong-key") as client:
            with pytest.raises(Exception) as exc_info:
                await client.list_indexes()
            assert "401" in str(exc_info.value)

@pytest.mark.asyncio
async def test_create_index_success():
    mock_response = {
        'name': 'test-index-776b',
        'metric': 'cosine',
        'dimension': 1536,
        'status': {'ready': False, 'state': 'Initializing'},
        'host': 'test-index-776b-b0ed6df.svc.aped-4627-b74a.pinecone.io',
        'spec': {'serverless': {'region': 'us-east-1', 'cloud': 'aws'}},
        'deletion_protection': 'disabled'
    }
    response = httpx.Response(201, json=mock_response)
    
    with patch('httpx.AsyncClient.post', return_value=response):
        async with PineconeClient(api_key="test-key") as client:
            result = await client.create_index(
                name="test-index-776b",
                dimension=1536,
                metric="cosine",
                spec=Serverless(cloud="aws", region="us-east-1")
            )
            assert result.name == mock_response['name']
            assert result.host == mock_response['host']
            assert result.status == mock_response['status']

@pytest.mark.asyncio
async def test_create_index_validates_spec():
    async with PineconeClient(api_key="test-key") as client:
        with pytest.raises(ValueError) as exc_info:
            await client.create_index(
                name="test",
                dimension=1536,
                metric="cosine",
                spec="invalid_spec"  # Should be Serverless or PineconePod
            )
        assert "must be either Serverless or Pod" in str(exc_info.value)

@pytest.mark.asyncio
async def test_describe_index_success():
    mock_response = {
        'name': 'test-index-776b',
        'metric': 'cosine',
        'dimension': 1536,
        'status': {'ready': False, 'state': 'Initializing'},
        'host': 'test-index-776b-b0ed6df.svc.aped-4627-b74a.pinecone.io',
        'spec': {'serverless': {'region': 'us-east-1', 'cloud': 'aws'}},
        'deletion_protection': 'disabled'
    }
    response = httpx.Response(200, json=mock_response)
    
    with patch('httpx.AsyncClient.get', return_value=response):
        async with PineconeClient(api_key="test-key") as client:
            result = await client.describe_index("test-index-776b")
            assert result.name == mock_response['name']
            assert result.host == mock_response['host']
            assert result.status == mock_response['status']

@pytest.mark.asyncio
async def test_describe_index_not_found():
    error_response = httpx.Response(404, text="Not Found")
    
    with patch('httpx.AsyncClient.get', return_value=error_response):
        async with PineconeClient(api_key="test-key") as client:
            with pytest.raises(IndexNotFoundError) as exc_info:
                await client.describe_index("non-existent-index")
            assert "not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_context_manager():
    client = PineconeClient(api_key="test-key")
    client.close = AsyncMock()
    
    async with client:
        pass
    
    client.close.assert_called_once()

@pytest.fixture
async def client():
    async with PineconeClient(api_key="test-key") as client:
        yield client