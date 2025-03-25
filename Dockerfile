FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ cmake ninja-build git pkg-config libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PATH="/root/.local/bin/:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-group dev --no-group test -C cmake.args="-DCMAKE_BUILD_TYPE=Release;-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS;-DGGML_NATIVE=OFF"

FROM python:3.13-slim-bookworm AS runtime

ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y libgomp1 libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 arcana
USER arcana

COPY --chown=arcana --from=builder /.venv /.venv

COPY --chown=arcana ./app /app

ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/app:/.venv/lib/python3.13/dist-packages"

WORKDIR /app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--workers", "2", "--host", "0.0.0.0", "--port", "7860"]