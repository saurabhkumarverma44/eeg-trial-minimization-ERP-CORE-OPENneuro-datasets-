import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_single_epoch_real_inference():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/inference/single", json={
            "dataset_name": "DS002680_gonogo_4",
            "epoch_index": 0
        })
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in [0, 1]
    assert "probabilities" in data
    assert len(data["probabilities"]) == 2
    assert "inference_time_ms" in data
    assert data["channel_count"] == 31
    assert data["time_points"] == 1001

@pytest.mark.asyncio
async def test_channel_incompatibility_validation():
    """Verify Section 11 requirement: 8-channel input to 31-channel model must be strictly rejected."""
    transport = ASGITransport(app=app)
    # Craft custom 8-channel input
    eight_channel_data = [[0.0] * 1001 for _ in range(8)]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/inference/single", json={
            "model_id": "eegnet-ds002680-gonogo-4",
            "epoch_data": eight_channel_data
        })
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Electrode Channel Incompatibility Error" in detail
    assert "expects 31 channels, but received data contains 8 channels" in detail
