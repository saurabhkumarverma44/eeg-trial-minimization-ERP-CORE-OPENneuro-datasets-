import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database_connected" in data
    assert "models_registered" in data

@pytest.mark.asyncio
async def test_system_info_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "python_version" in data
    assert "torch_version" in data
    assert "disk_free_gb" in data
    assert "paths" in data

@pytest.mark.asyncio
async def test_project_inventory_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/project/inventory")
    assert response.status_code == 200
    data = response.json()
    assert "project_root" in data
    assert "items" in data
    assert isinstance(data["items"], list)
