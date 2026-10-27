import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_create_and_list_experiments():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create experiment with minimal trial count for fast test execution
        response = await ac.post("/api/experiments", json={
            "name": "Test_Minimisation_Fast",
            "dataset_name": "DS002680",
            "task_name": "gonogo_4",
            "model_type": "EEGNet",
            "trial_counts": [10],
            "epochs": 2,
            "num_folds": 2,
            "use_gan_augmentation": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert data["status"] == "running"
        exp_id = data["experiment_id"]

        # List experiments
        list_res = await ac.get("/api/experiments")
        assert list_res.status_code == 200
        exps = list_res.json()
        assert any(e["id"] == exp_id for e in exps)

@pytest.mark.asyncio
async def test_experiment_figure_and_csv_export():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        list_res = await ac.get("/api/experiments")
        assert list_res.status_code == 200
        exps = list_res.json()
        completed = [e for e in exps if e["status"] == "completed"]
        if completed:
            exp_id = completed[0]["id"]
            fig_res = await ac.get(f"/api/results/{exp_id}/figure")
            assert fig_res.status_code == 200
            assert fig_res.headers["content-type"] == "image/png"
            assert len(fig_res.content) > 1000

            csv_res = await ac.get(f"/api/results/{exp_id}/export/csv")
            assert csv_res.status_code == 200
            assert "text/csv" in csv_res.headers["content-type"]
            assert b"trial_fraction" in csv_res.content

