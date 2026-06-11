from __future__ import annotations
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from src.ranking.inference.rank import rank_candidates

app = FastAPI(title="Content Ranking API", version="1.0")

class Context(BaseModel):
    device: str = "tv"
    hour: int = 20
    day_of_week: int = 2
    session_id: str = "s_online"

class RankRequest(BaseModel):
    user_id: str
    candidates: List[str] = Field(min_length=1)
    context: Context = Context()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rank")
def rank(req: RankRequest):
    # Load raw user/item info (demo). Production would come from online feature store.
    raw_dir = "data/raw"
    users_path = os.path.join(raw_dir, "users.csv")
    items_path = os.path.join(raw_dir, "items.csv")

    if not (os.path.exists(users_path) and os.path.exists(items_path)):
        raise HTTPException(status_code=400, detail="Missing data/raw/users.csv or data/raw/items.csv. Run offline generation first.")

    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)

    user = users[users["user_id"] == req.user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {req.user_id}")

    cand_items = items[items["item_id"].isin(req.candidates)]
    missing = set(req.candidates) - set(cand_items["item_id"].tolist())
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown item_ids: {sorted(list(missing))}")

    user_row = user.iloc[0].to_dict()
    item_rows = cand_items.to_dict(orient="records")

    ranked = rank_candidates(
        user_row=user_row,
        item_rows=item_rows,
        context=req.context.model_dump(),
        models_dir="artifacts/models",
    )
    return {
        "user_id": req.user_id,
        "ranked": ranked,
        "context": req.context.model_dump(),
    }
