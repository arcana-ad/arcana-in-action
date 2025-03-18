import logging
import os
import time
from contextlib import asynccontextmanager
from enum import StrEnum

from arcana_codex import (
    AdUnitsFetchModel,
    AdUnitsIntegrateModel,
    ArcanaCodexClient,
)
from llama_cpp import Llama
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import FileResponse


class SupportedModelPipes(StrEnum):
    Gemma3 = "gemma3"
    QwenOpenR1 = "qwen-open-r1"
    SmolLLM2 = "smollm2"
    SmolLLM2Reasoning = "smollm2-reasoning"


smollm2_pipeline = Llama.from_pretrained(
    repo_id="tensorblock/SmolLM2-135M-Instruct-GGUF",
    filename="SmolLM2-135M-Instruct-Q8_0.gguf",
    verbose=False,
)

smollm2_reasoning_pipeline = Llama.from_pretrained(
    repo_id="tensorblock/Reasoning-SmolLM2-135M-GGUF",
    filename="Reasoning-SmolLM2-135M-Q8_0.gguf",
    verbose=False,
)

qwen_open_r1_pipeline = Llama.from_pretrained(
    repo_id="tensorblock/Qwen2.5-0.5B-Open-R1-Distill-GGUF",
    filename="Qwen2.5-0.5B-Open-R1-Distill-Q8_0.gguf",
    verbose=False,
)

gemma_3_pipeline = Llama.from_pretrained(
    repo_id="ggml-org/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q8_0.gguf",
    verbose=False,
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

    logger.info(f"Using {payload.model}")

    match payload.model:
        case SupportedModelPipes.Gemma3:
            ai_pipeline = gemma_3_pipeline
        case SupportedModelPipes.QwenOpenR1:
            ai_pipeline = qwen_open_r1_pipeline
        case SupportedModelPipes.SmolLLM2:
            ai_pipeline = smollm2_pipeline
        case SupportedModelPipes.SmolLLM2Reasoning:
            ai_pipeline = smollm2_reasoning_pipeline

    inference_start_time = time.perf_counter()
    ai_response = ai_pipeline.create_chat_completion(
        messages=[{"role": "user", "content": f"{payload.message}"}],
        max_tokens=512,
        seed=8,
    )["choices"][0]["message"]["content"].strip()
    inference_end_time = time.perf_counter()

    elapsed_time = inference_end_time - inference_start_time
    logger.info(f"Inference took: {elapsed_time:.4f} seconds")

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
