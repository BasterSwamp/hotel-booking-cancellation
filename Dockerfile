FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-cache --no-install-package nvidia-nccl-cu12

COPY predict.py model.bin ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
