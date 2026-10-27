import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_list_datasets():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert isinstance(datasets, list)
    assert len(datasets) >= 1
    # Check DS002680_gonogo_4 is discovered
    ds4 = next((d for d in datasets if "gonogo_4" in d["dataset_id"]), None)
    assert ds4 is not None
    assert ds4["num_channels"] == 31
    assert ds4["sampling_rate"] == 1000.0

@pytest.mark.asyncio
async def test_preprocessing_preview():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/preprocessing/preview", json={
            "dataset_id": "DS002680_gonogo_4",
            "channel_to_plot": "Fz",
            "config": {
                "l_freq": 0.1,
                "h_freq": 40.0,
                "notch_freq": 50.0,
                "tmin": -0.2,
                "tmax": 0.8,
                "baseline": [-0.2, 0.0]
            }
        })
    assert response.status_code == 200
    data = response.json()
    assert data["channel_name"] == "Fz"
    assert len(data["raw_signal"]) == 1001
    assert len(data["filtered_signal"]) == 1001
    assert data["status"] == "Success"
