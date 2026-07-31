# GenFragility graph image — minimal, self-contained.
#
# Build from the repo root so the COPY paths below can see results/ and data/:
#   docker build -f docker/Dockerfile -t genfrag-graph:latest .
#
# The image bundles:
#   /app/data/final.pkl             (the 100k DiGraph, ~125 MB)
#   /app/data/graph_qid_index.json  (node-name -> Wikidata QID, ~4 MB)
#
# Default entrypoint drops you into an IPython REPL with `api` and `G`
# pre-loaded. Override with `python /app/cli.py ...` for one-shot queries
# or `jupyter lab ...` for notebook use (see docker/README.md).

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python deps first so subsequent COPY of the (big) graph data
# does not invalidate the pip layer.
COPY docker/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Application code (small).
COPY docker/graph_api.py     /app/graph_api.py
COPY docker/interactive.py   /app/interactive.py
COPY docker/cli.py           /app/cli.py
COPY docker/report.py        /app/report.py
COPY docker/paper_claims.md  /app/paper_claims.md
COPY docker/demo_dataset.jsonl /app/demo_dataset.jsonl

# Data assets (large — keep at the end of the file so code edits don't
# re-trigger this layer).
RUN mkdir -p /app/data
COPY results/checkpoints/final.pkl       /app/data/final.pkl
COPY data/external_eval/graph_qid_index.json /app/data/graph_qid_index.json

# Jupyter, when the user opts into notebook mode.
EXPOSE 8888

# Default: IPython REPL with the graph pre-loaded.
ENTRYPOINT ["python", "/app/interactive.py"]
