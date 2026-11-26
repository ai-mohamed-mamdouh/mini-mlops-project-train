# syntax=docker/dockerfile:1

# Base stage
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


# Builder stage
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.9 \
    /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# Runtime stage
FROM base AS runtime

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 \
        --create-home app \
    && mkdir -p /app/artifacts \
    && chown app:app /app/artifacts

USER app

CMD ["python", "-m", "mini_mlops.main"]