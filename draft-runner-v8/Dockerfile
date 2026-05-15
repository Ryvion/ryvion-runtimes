FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir llama-cpp-python==0.3.8

RUN useradd -r -s /usr/sbin/nologin runner
COPY run.py /app/run.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -m py_compile /app/run.py || exit 1

STOPSIGNAL SIGTERM
USER runner
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
