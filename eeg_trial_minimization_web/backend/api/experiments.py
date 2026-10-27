import io
import csv
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from backend.core.config import settings
from backend.models.schemas import ExperimentCreateRequest, ExperimentSummary
from backend.services.experiment_service import (
    create_and_launch_experiment,
    list_experiments,
    get_experiment_details,
    get_experiment_progress
)

router = APIRouter(prefix="/api", tags=["Experiments & Minimization"])

@router.post("/experiments", response_model=Dict[str, Any])
async def launch_experiment(req: ExperimentCreateRequest):
    exp_id = await create_and_launch_experiment(req)
    return {"experiment_id": exp_id, "status": "running", "message": "Experiment started in background."}

@router.get("/experiments", response_model=List[ExperimentSummary])
async def get_all_experiments():
    return await list_experiments()

@router.get("/experiments/{experiment_id}")
async def get_single_experiment(experiment_id: str):
    return await get_experiment_details(experiment_id)

@router.get("/experiments/{experiment_id}/progress")
async def get_single_experiment_progress(experiment_id: str):
    prog = get_experiment_progress(experiment_id)
    if not prog:
        details = await get_experiment_details(experiment_id)
        exp = details["experiment"]
        return {
            "status": exp["status"],
            "progress": exp["progress"],
            "current_step": exp["current_step"],
            "summary": exp["summary_metrics"],
            "results": details["results"]
        }
    return prog

@router.get("/results/{experiment_id}/figure")
async def get_experiment_figure(experiment_id: str):
    fig_path = settings.EEG_RESULTS_ROOT / experiment_id / "trial_minimization_curve.png"
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail="Figure not found for this experiment.")
    return FileResponse(str(fig_path), media_type="image/png")

@router.get("/results/{experiment_id}/export/csv")
async def export_experiment_csv(experiment_id: str):
    details = await get_experiment_details(experiment_id)
    results = details["results"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "trial_fraction", "n_train_trials", "test_acc", "balanced_acc",
        "precision", "recall", "f1_score", "roc_auc"
    ])
    for r in results:
        writer.writerow([
            r["trial_fraction"], r["n_train_trials"], r["test_acc"],
            r["balanced_acc"], r["precision"], r["recall"], r["f1_score"], r["roc_auc"]
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={experiment_id}_results.csv"}
    )
