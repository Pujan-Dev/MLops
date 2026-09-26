"""FastAPI serving layer for the agentic assistant.

Endpoints:
  GET  /health               -> status + available prompt versions
  GET  /prompts              -> list prompt versions + configs
  POST /query                -> run agent for one question (+ trace)
  POST /evaluate              -> run Evidently regression for one/all versions
  GET  /evaluations/{ver}    -> fetch saved eval JSON for v1/v2/v3

Run:
  uv run uvicorn src.app:app --reload --port 8000
  # or: uv run python -m src.app
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PROJECT_ROOT = SRC_DIR.parent

try:
    from src.agent import PROMPT_CONFIGS, load_prompt, run_agent  # noqa: E402
    from src.evaluate import REGRESSION_DATASET, evaluate_prompt_version  # noqa: E402
except ImportError:
    from agent import PROMPT_CONFIGS, load_prompt, run_agent  # noqa: E402
    from evaluate import REGRESSION_DATASET, evaluate_prompt_version  # noqa: E402

app = FastAPI(title="Agentic MLOps Assistant", version="0.1.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    prompt_version: str = Field("v3", description="v1, v2 or v3")
    verbose: bool = False


class QueryResponse(BaseModel):
    answer: str
    prompt_version: str
    iterations: int
    tool_calls: int
    tokens: int
    latency_s: float
    termination_reason: str
    trace: list[dict]


class EvaluateRequest(BaseModel):
    prompt_version: str = Field("all", description="v1, v2, v3 or all")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "prompts": sorted(PROMPT_CONFIGS.keys()),
            "n_regression_tests": len(REGRESSION_DATASET)}


@app.get("/prompts")
def prompts() -> dict:
    out = {}
    for ver, cfg in PROMPT_CONFIGS.items():
        try:
            text = load_prompt(ver)
        except FileNotFoundError:
            text = ""
        out[ver] = {"config": cfg, "prompt_preview": text[:500]}
    return out


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    ver = req.prompt_version.lower()
    if ver not in PROMPT_CONFIGS and ver.lstrip("v") not in ("1", "2", "3"):
        raise HTTPException(status_code=400, detail="prompt_version must be v1, v2 or v3")
    if not ver.startswith("v"):
        ver = f"v{ver}"
    try:
        res = run_agent(req.question, prompt_version=ver, verbose=req.verbose)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "answer": res["answer"],
        "prompt_version": res["prompt_version"],
        "iterations": res["iterations"],
        "tool_calls": res["tool_calls"],
        "tokens": res["tokens"],
        "latency_s": res["latency_s"],
        "termination_reason": res["termination_reason"],
        "trace": res["trace"],
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    ver = req.prompt_version.lower()
    versions = ["v1", "v2", "v3"] if ver == "all" else [ver if ver.startswith("v") else f"v{ver}"]
    for v in versions:
        if v not in PROMPT_CONFIGS:
            raise HTTPException(status_code=400, detail=f"unknown version {v}")
    summaries = [evaluate_prompt_version(v) for v in versions]
    return {"results": [
        {"prompt_version": s["prompt_version"], "pass_rate": s["pass_rate"],
         "n_passed": s["n_passed"], "n_tests": s["n_tests"],
         "correctness_avg": s["correctness_avg"],
         "preservation_avg": s["preservation_avg"]} for s in summaries
    ]}


@app.get("/evaluations/{version}")
def get_evaluation(version: str) -> dict:
    import json
    ver = version.lower() if version.startswith("v") else f"v{version}"
    path = PROJECT_ROOT / "evaluations" / f"eval_{ver}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no report for {ver}; POST /evaluate first")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
