FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    blender \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY run.py /usr/local/bin/blender-runner

RUN chmod +x /usr/local/bin/blender-runner \
    && useradd -r -s /usr/sbin/nologin runner

USER runner
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/blender-runner"]
