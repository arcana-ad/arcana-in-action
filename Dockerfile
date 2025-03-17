FROM python:3.13-slim-bookworm

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user ./app.py app.py
COPY --chown=user ./frontend.html frontend.html
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
