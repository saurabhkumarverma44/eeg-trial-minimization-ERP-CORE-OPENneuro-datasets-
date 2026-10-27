import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_streaming_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Check initial status
        stat_res = await ac.get("/api/streaming/status")
        assert stat_res.status_code == 200
        stat = stat_res.json()
        assert "is_streaming" in stat
        assert "channel_count" in stat

        # Configure adapter
        cfg_res = await ac.post("/api/streaming/configure", json={
            "adapter_type": "simulated",
            "dataset_id": "DS002680_gonogo_4",
            "channel_count": 31
        })
        assert cfg_res.status_code == 200

        # Start stream
        start_res = await ac.post("/api/streaming/start")
        assert start_res.status_code == 200
        assert start_res.json()["is_streaming"] is True

        # Stop stream
        stop_res = await ac.post("/api/streaming/stop")
        assert stop_res.status_code == 200
        assert stop_res.json()["is_streaming"] is False

@pytest.mark.asyncio
async def test_hardware_adapters_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/hardware/adapters")
    assert response.status_code == 200
    adapters = response.json()
    assert len(adapters) >= 4
    # Ensure physical cap and simulated adapters are listed
    ids = [a["id"] for a in adapters]
    assert "simulated" in ids
    assert "playback" in ids
    assert "physical" in ids
