import pytest
import pytest_asyncio
pytestmark = pytest.mark.asyncio
from unittest.mock import patch
import httpx
from pinecone_async import PineconeIndex, PineconeVector, VectorMetadata

# Real response patterns we observed from Pinecone
MOCK_INDEX_RESPONSE = {
    'name': 'test-index-simple',
    'metric': 'cosine',
    'dimension': 8,
    'status': {'ready': True, 'state': 'Ready'},
    'host': 'test-index-simple-b0ed6df.svc.aped-4627-b74a.pinecone.io',
    'spec': {'serverless': {'region': 'us-east-1', 'cloud': 'aws'}},
    'deletion_protection': 'disabled'
}

MOCK_UPSERT_RESPONSE = {"upsertedCount": 1}

MOCK_QUERY_RESPONSE = {
    "matches": [{
        "id": "test1",
        "score": 0.999999881,
        "values": [],
        "metadata": {
            "content": "test content"
        }
    }],
    "namespace": ""
}

MOCK_FETCH_RESPONSE = {
    "vectors": {
        "test1": {
            "id": "test1",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "metadata": {
                "content": "test content"
            }
        }
    },
    "namespace": ""
}

MOCK_DELETE_RESPONSE = {}

@pytest_asyncio.fixture
async def index():
    with patch('httpx.AsyncClient.get', return_value=httpx.Response(200, json=MOCK_INDEX_RESPONSE)):
        return await PineconeIndex.create(
            api_key="test-key",
            index_name="test-index",
            metric="cosine",
            dimensions=8,
            region="us-east-1"
        )

@pytest.mark.asyncio
async def test_upsert_vectors(index):
    vector = PineconeVector(
        id="test1",
        values=[0.1] * 8,
        metadata=VectorMetadata(content="test content")
    )
    
    with patch('httpx.AsyncClient.post', return_value=httpx.Response(200, json=MOCK_UPSERT_RESPONSE)):
        response = await index.upsert([vector])
        assert response.upsertedCount == 1


@pytest.mark.asyncio
async def test_debug():
    
    with patch('httpx.AsyncClient.get', return_value=httpx.Response(200, json=MOCK_INDEX_RESPONSE)):
        index = await PineconeIndex.create(
            api_key="test-key",
            index_name="test-index",
            metric="cosine",
            dimensions=8,
            region="us-east-1"
        )
        assert hasattr(index, 'delete'), "Index doesn't have delete method"
        assert callable(index.delete), "Delete is not callable"
        
        
@pytest.mark.asyncio
async def test_index_initialization():
    with patch('httpx.AsyncClient.get', return_value=httpx.Response(200, json=MOCK_INDEX_RESPONSE)):
        index = await PineconeIndex.create(
            api_key="test-key",
            index_name="test-index",
            metric="cosine",
            dimensions=8,
            region="us-east-1"
        )
        assert index.index_name == "test-index"
        assert index.dimensions == 8
        assert index.index_host == MOCK_INDEX_RESPONSE['host']

@pytest.mark.asyncio
async def test_upsert_vectors(index):
    vector = PineconeVector(
        id="test1",
        values=[0.1] * 8,
        metadata=VectorMetadata(content="test content")
    )
    
    with patch('httpx.AsyncClient.post', return_value=httpx.Response(200, json=MOCK_UPSERT_RESPONSE)):
        response = await index.upsert([vector])
        assert response.upsertedCount == 1

@pytest.mark.asyncio
async def test_query_vectors(index):
    with patch('httpx.AsyncClient.post', return_value=httpx.Response(200, json=MOCK_QUERY_RESPONSE)):
        response = await index.query(
            vector=[0.1] * 8,
            top_k=1,
            include_metadata=True
        )
        assert len(response.matches) == 1
        assert response.matches[0].id == "test1"
        assert response.matches[0].score == pytest.approx(0.999999881)
        assert response.matches[0].metadata.content == "test content"

@pytest.mark.asyncio
async def test_fetch_vectors(index):
    with patch('httpx.AsyncClient.get', return_value=httpx.Response(200, json=MOCK_FETCH_RESPONSE)):
        response = await index.fetch(ids=["test1"])
        assert "test1" in response.vectors
        vector = response.vectors["test1"]
        assert vector.values == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        assert vector.metadata.content == "test content"

@pytest.mark.asyncio
async def test_delete_vectors(index):
    with patch('httpx.AsyncClient.post', return_value=httpx.Response(200, json=MOCK_DELETE_RESPONSE)):
        response = await index.delete(ids=["test1"])
        assert response == {}
