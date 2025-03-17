FROM python:3.13-slim-bookworm

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=OFF"

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev g++ ninja-build git pkg-config libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user ./app.py app.py
COPY --chown=user ./frontend.html frontend.html
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
