import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from arcana_codex import (
    AdUnitsFetchModel,
    AdUnitsIntegrateModel,
    ArcanaCodexClient,
)
from bson.objectid import ObjectId
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from llama_cpp import Llama
from pydantic import BaseModel, EmailStr
from pymongo.mongo_client import MongoClient
from starlette.responses import FileResponse

__version__ = "0.0.0"


class SupportedModelPipes(StrEnum):
    Gemma3 = "gemma3"
    QwenOpenR1 = "qwen-open-r1"
    SmolLLM2 = "smollm2"
    SmolLLM2Reasoning = "smollm2-reasoning"


class LogEvent(StrEnum):
    CHAT_INTERACTION = "chat_interaction"
    LOGIN = "login"


smollm2_pipeline = Llama.from_pretrained(
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
    filename="smollm2-360m-instruct-q8_0.gguf",
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


class LoginRequest(BaseModel):
    email_id: EmailStr
    access_key: str


class LoginResponse(BaseModel):
    verified_id: str


class User(BaseModel):
    _id: ObjectId
    email_id: EmailStr


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_authorization_header(
    request: Request, authorization: str | None = Header(None)
) -> User:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Bearer required.",
            )

        user = request.app.mongo_db["users"].find_one({"_id": ObjectId(token)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verified_id",
            )

        return User(**user)

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Set API key in FastAPI app
    app.ARCANA_API_KEY = os.environ.get("ARCANA_API_KEY", "")

    app.mongo_db = MongoClient(
        os.environ.get("MONGO_URI", "mongodb+srv://localhost:27017/")
    )["arcana_hf_demo"]

    logging.info("Application started")

    yield

    # Clear API key to avoid leaking it
    app.ARCANA_API_KEY = ""

    logging.info("Application stopped")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return JSONResponse({"health_check": "pass"})


@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request):
    user = request.app.mongo_db["users"].find_one(
        {"email_id": payload.email_id, "access_key": payload.access_key}
    )
    if user:
        request.app.mongo_db["logs"].insert_one(
            {
                "email_id": user["email_id"],
                "timestamp": datetime.now(UTC),
                "event": LogEvent.LOGIN,
            }
        )
        verified_id = user["_id"]
        return LoginResponse(verified_id=str(verified_id))
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    user: Annotated[User, Depends(verify_authorization_header)],
):
    logger.info(f"Received message: {payload.message}")

    client = ArcanaCodexClient(
        api_key=request.app.ARCANA_API_KEY, base_url="http://gateway-backend/api/public"
    )
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

    request.app.mongo_db["logs"].insert_one(
        {
            "email_id": user.email_id,
            "timestamp": datetime.now(UTC),
            "event": LogEvent.CHAT_INTERACTION,
        }
    )

    return ChatResponse(response=integrated_content)


@app.get("/")
async def read_index():
    return FileResponse("./static/index.html")


@app.get("/login")
async def read_login():
    return FileResponse("./static/login.html")
