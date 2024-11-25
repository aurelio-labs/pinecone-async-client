# tests/test_client.py
import pytest
from unittest.mock import AsyncMock, patch
from pinecone_async import PineconeClient
from pinecone_async.exceptions import IndexNotFoundError

@pytest.mark.asyncio
async def test_list_indexes():
   mock_session = AsyncMock()
   mock_session.__aenter__.return_value = mock_session
   mock_session.get.return_value.__aenter__.return_value.status = 200
   mock_session.get.return_value.__aenter__.return_value.json.return_value = [{"name": "test-index"}]
   
   with patch('aiohttp.ClientSession', return_value=mock_session):
       client = PineconeClient(api_key="fake-key")
       indexes = await client.list_indexes()
       assert isinstance(indexes, list)

@pytest.mark.asyncio
async def test_describe_index_not_found():
   mock_session = AsyncMock()
   mock_session.__aenter__.return_value = mock_session
   mock_session.get.return_value.__aenter__.return_value.status = 404
   mock_session.get.return_value.__aenter__.return_value.text.return_value = "Not Found"
   
   with patch('aiohttp.ClientSession', return_value=mock_session):
       client = PineconeClient(api_key="fake-key")
       with pytest.raises(IndexNotFoundError):
           await client.describe_index("non-existent-index")

@pytest.mark.asyncio
async def test_create_index():
   response_data = {"database": {}, "status": {}, "host": "test-host"}
   mock_session = AsyncMock()
   mock_session.__aenter__.return_value = mock_session
   mock_session.post.return_value.__aenter__.return_value.status = 201
   mock_session.post.return_value.__aenter__.return_value.json.return_value = response_data
   
   with patch('aiohttp.ClientSession', return_value=mock_session):
       client = PineconeClient(api_key="fake-key")