import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_models_list_and_builtin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    # Check built-in model is registered
    builtin = next((m for m in models if m.get("is_builtin")), None)
    assert builtin is not None
    assert "EEGNet" in builtin["architecture"]
    assert builtin["num_channels"] == 31

@pytest.mark.asyncio
async def test_builtin_model_deletion_protected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Attempt to delete built-in research model
        response = await ac.delete("/api/models/eegnet-ds002680-gonogo-4")
    # Must be forbidden (403)
    assert response.status_code == 403
    assert "Forbidden" in response.json()["detail"]
