# tests/conftest.py
from pinecone_async.client import PineconeClient
import pytest
import aiohttp
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_session():
    with patch('aiohttp.ClientSession') as mock:
        session = AsyncMock()
        mock.return_value.__aenter__.return_value = session
        yield session

# tests/test_client.py
@pytest.mark.asyncio
async def test_list_indexes(mock_session):
    mock_session.get.return_value.__aenter__.return_value.json.return_value = [{"name": "test-index"}]
    
    client = PineconeClient(api_key="fake-key")
    indexes = await client.list_indexes()
    assert indexes == [{"name": "test-index"}]