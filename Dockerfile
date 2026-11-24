
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY . .

CMD ["uv", "run", "--no-sync", "python", "-m", "src.mini_mlops.main"]
