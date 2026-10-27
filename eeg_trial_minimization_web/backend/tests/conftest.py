import pytest
import pytest_asyncio
from backend.core.database import init_database

@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    await init_database()
