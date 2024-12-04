import pytest
from unittest.mock import patch
import httpx
from pinecone_async import PineconeClient
from pinecone_async.schema import Document, RerankParameters, RerankResponse, RerankResult, RerankUsage

# Mock responses
MOCK_RERANK_RESPONSE = {
    "data": [
        {
            "index": 0,
            "document": {
                "id": "1",
                "text": "Python is a programming language"
            },
            "score": 0.95
        },
        {
            "index": 1,
            "document": {
                "id": "2",
                "text": "JavaScript is a web language"
            },
            "score": 0.75
        }
    ],
    "usage": {
        "rerank_units": 2
    }
}

MOCK_DOCUMENTS = [
    {
        "id": "1",
        "text": "Python is a programming language"
    },
    {
        "id": "2",
        "text": "JavaScript is a web language"
    }
]

@pytest.mark.asyncio
async def test_rerank_basic_functionality():
    """Test basic reranking functionality with default parameters."""
    response = httpx.Response(200, json=MOCK_RERANK_RESPONSE)
    
    with patch('httpx.AsyncClient.post', return_value=response):
        async with PineconeClient(api_key="test-key") as client:
            result = await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS
            )
            
            assert isinstance(result, RerankResponse)
            assert len(result.data) == 2
            assert result.data[0].score == 0.95
            assert result.data[0].document.text == "Python is a programming language"
            assert result.usage.rerank_units == 2

@pytest.mark.asyncio
async def test_rerank_with_custom_model():
    """Test reranking with a specific model."""
    response = httpx.Response(200, json=MOCK_RERANK_RESPONSE)
    
    with patch('httpx.AsyncClient.post', return_value=response) as mock_post:
        async with PineconeClient(api_key="test-key") as client:
            await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS,
                model="bge-reranker-v2-m3"
            )
            
            # Verify the correct model was sent in the request
            called_json = mock_post.call_args.kwargs['json']
            assert called_json['model'] == "bge-reranker-v2-m3"

@pytest.mark.asyncio
async def test_rerank_with_invalid_model():
    """Test reranking with an invalid model."""
    async with PineconeClient(api_key="test-key") as client:
        with pytest.raises(ValueError) as exc_info:
            await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS,
                model="invalid-model"
            )
        assert "Invalid model" in str(exc_info.value)

@pytest.mark.asyncio
async def test_rerank_with_parameters():
    """Test reranking with custom parameters."""
    response = httpx.Response(200, json=MOCK_RERANK_RESPONSE)
    
    with patch('httpx.AsyncClient.post', return_value=response) as mock_post:
        async with PineconeClient(api_key="test-key") as client:
            await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS,
                parameters={"truncate": "START"},
                top_n=5
            )
            
            called_json = mock_post.call_args.kwargs['json']
            assert called_json['parameters']['truncate'] == "START"
            assert called_json['top_n'] == 5

@pytest.mark.asyncio
async def test_rerank_error_response():
    """Test handling of error responses from the rerank API."""
    error_response = httpx.Response(400, text="Bad Request")
    
    with patch('httpx.AsyncClient.post', return_value=error_response):
        async with PineconeClient(api_key="test-key") as client:
            with pytest.raises(Exception) as exc_info:
                await client.rerank(
                    query="What is Python?",
                    documents=MOCK_DOCUMENTS
                )
            assert "Failed to rerank" in str(exc_info.value)

@pytest.mark.asyncio
async def test_rerank_empty_documents():
    """Test reranking with empty documents list."""
    async with PineconeClient(api_key="test-key") as client:
        with pytest.raises(ValueError) as exc_info:
            await client.rerank(
                query="What is Python?",
                documents=[]
            )
        assert "documents cannot be empty" in str(exc_info.value)

@pytest.mark.asyncio
async def test_rerank_with_rank_fields():
    """Test reranking with custom rank fields."""
    response = httpx.Response(200, json=MOCK_RERANK_RESPONSE)
    
    with patch('httpx.AsyncClient.post', return_value=response) as mock_post:
        async with PineconeClient(api_key="test-key") as client:
            await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS,
                rank_fields=["text", "title"]
            )
            
            called_json = mock_post.call_args.kwargs['json']
            assert called_json['rank_fields'] == ["text", "title"]

@pytest.mark.asyncio
async def test_rerank_without_return_documents():
    """Test reranking without returning documents."""
    mock_response_without_docs = {
        "data": [
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.75}
        ],
        "usage": {"rerank_units": 2}
    }
    response = httpx.Response(200, json=mock_response_without_docs)
    
    with patch('httpx.AsyncClient.post', return_value=response):
        async with PineconeClient(api_key="test-key") as client:
            result = await client.rerank(
                query="What is Python?",
                documents=MOCK_DOCUMENTS,
                return_documents=False
            )
            
            assert isinstance(result, RerankResponse)
            assert result.data[0].document is None
            assert result.data[0].score == 0.95

@pytest.mark.asyncio
async def test_list_supported_models():
    """Test listing supported reranking models."""
    models = PineconeClient.list_supported_models()
    assert isinstance(models, list)
    assert "cohere-rerank-3.5" in models
    assert "bge-reranker-v2-m3" in models