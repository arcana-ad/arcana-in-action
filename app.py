import os
import logging
from contextlib import asynccontextmanager
from enum import StrEnum

from arcana_codex import (
    AdUnitsFetchModel,
    AdUnitsIntegrateModel,
    ArcanaCodexClient,
)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import FileResponse
from transformers import pipeline


class SupportedModelPipes(StrEnum):
    SmolLLM2 = "smollm2"
    SmolVLM = "smolvlm"


smollm2_pipeline = pipeline(
    "text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct"
)
smolvlm_pipeline = pipeline(
    "image-text-to-text", model="HuggingFaceTB/SmolVLM-500M-Instruct"
)


class ChatRequest(BaseModel):
    model: SupportedModelPipes = SupportedModelPipes.SmolLLM2
    message: str


class ChatResponse(BaseModel):
    response: str


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Set API key in FastAPI app
    app.ARCANA_API_KEY = os.environ.get("ARCANA_API_KEY", "")

    logging.info("Application started")

    yield

    # Clear API key to avoid leaking it
    app.ARCANA_API_KEY = ""

    logging.info("Application stopped")


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    logger.info(f"Received message: {payload.message}")

    client = ArcanaCodexClient(request.app.ARCANA_API_KEY)
    fetch_payload = AdUnitsFetchModel(query=payload.message)
    ad_fetch_response = client.fetch_ad_units(fetch_payload)

    match payload.model:
        case SupportedModelPipes.SmolLLM2:
            ai_pipeline = smollm2_pipeline
        case SupportedModelPipes.SmolVLM:
            ai_pipeline = smolvlm_pipeline

    ai_response = (
        ai_pipeline(
            [{"role": "user", "content": f"{payload.message}"}],
            do_sample=False,
            max_new_tokens=512,
        )[0]
        .get("generated_text", [{}, {}])[1]
        .get("content", "")
    )

    integrate_payload = AdUnitsIntegrateModel(
        ad_unit_ids=[
            ad_unit.get("id") for ad_unit in ad_fetch_response.get("response_data", [])
        ],
        base_content=ai_response,
    )

    integration_result = client.integrate_ad_units(integrate_payload)
    integrated_content = integration_result.get("response_data", {}).get(
        "integrated_content"
    )

    return ChatResponse(response=integrated_content)


@app.get("/")
def frontend():
    return FileResponse("frontend.html")
