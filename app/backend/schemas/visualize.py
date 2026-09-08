from __future__ import annotations

from pydantic import BaseModel


class VisualizeRequest(BaseModel):
    run_id: str
    model_config = {"protected_namespaces": ()}
