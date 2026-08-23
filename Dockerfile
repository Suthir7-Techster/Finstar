# --- Build stage -----------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

# --- Runtime stage ----------------------------------------------------------
FROM python:3.12-slim

# Non-root runtime user; artifacts live in a writable app-owned directory.
RUN groupadd --system feg && useradd --system --gid feg --create-home feg
WORKDIR /app
COPY --from=builder /dist/*.whl /tmp/
COPY configs ./configs
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl \
    && mkdir -p /home/feg/artifacts && chown -R feg:feg /home/feg /app

USER feg
ENV FEG_ARTIFACTS_DIR=/home/feg/artifacts

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

EXPOSE 8000
# The service reports 503 on /readyz until a model is registered; train one
# first (docker compose does this automatically):
#   docker compose run --rm trainer
CMD ["uvicorn", "feg_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
