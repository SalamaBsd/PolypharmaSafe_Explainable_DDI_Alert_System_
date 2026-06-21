"""
ClearDDI — serveur API
Sert l'API d'inférence et l'interface web statique.

Prototype académique d'aide à la décision — non destiné à un usage clinique réel.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference as inf

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR.parent / "frontend"

app = FastAPI(title="ClearDDI — Explainable DDI Alert System (prototype académique)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    drug_a: str
    drug_b: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model": inf.CONFIG.get("model_name"), "n_drugs": len(inf.NAME_TO_ID)}


@app.get("/api/samples")
def samples():
    return {"names": inf.list_sample_names()}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    """Suggestions de noms pour l'autocomplétion légère (facultatif, non utilisé
    pour valider — seulement pour aider l'utilisateur à taper)."""
    key = inf.normalize_name(q)
    matches = [n for n in inf._ALL_NAMES_SORTED if n.startswith(key)][:8]
    if len(matches) < 8:
        contains = [n for n in inf._ALL_NAMES_SORTED if key in n and n not in matches][:8 - len(matches)]
        matches += contains
    return {"matches": matches}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.drug_a.strip() or not req.drug_b.strip():
        return JSONResponse(status_code=400, content={"error": "empty_input"})

    result = inf.analyze_pair(req.drug_a, req.drug_b)

    if result.get("error") == "not_found":
        return JSONResponse(status_code=404, content=result)
    if result.get("error") == "invalid_smiles":
        return JSONResponse(status_code=422, content=result)

    return result


@app.get("/api/molecule_svg")
def molecule_svg_endpoint(name: str):
    drug = inf.find_drug(name)
    if drug is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "query": name})
    svg = inf.molecule_svg(drug["smiles"])
    return JSONResponse(content={"svg": svg, "smiles": drug["smiles"], "drugbank_id": drug["drugbank_id"]})


# ── Frontend statique ────────────────────────────────────────────────────
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
