import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_pinecone_response():
    """Create a mock response that properly handles async context management"""
    def _make_response(status=200, json_data=None, text_data=None):
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.json.return_value = json_data
        mock_response.text.return_value = text_data or ""
        return mock_response
    return _make_response

@pytest.fixture
def mock_pinecone_session():
    """Create a mock session with proper async context management"""
    async def _make_context_manager(response):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = response
        mock_cm.__aexit__.return_value = None
        return mock_cm
    
    session = AsyncMock()
    session._make_context_manager = _make_context_manager
    return session