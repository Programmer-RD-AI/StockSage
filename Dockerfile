# Use the same base image for both stages
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /src

# Copy dependency files first
COPY pyproject.toml uv.lock ./
COPY .env ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the rest of the application
COPY . .

# Install project deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Use the same base image for runtime
FROM python:3.13-slim AS runner

WORKDIR /src

COPY --from=builder --chown=1000:1000 /src /src

# Make sure the virtual environment is properly activated
ENV PYTHONPATH="/src" \
    PATH="/src/.venv/bin:$PATH" \
    VIRTUAL_ENV="/src/.venv"

# Run Streamlit instead of CLI
CMD ["streamlit", "run", "src/stocksage/user_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
