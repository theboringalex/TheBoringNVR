ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG BUILD_ARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Debian (glibc) instead of Alpine: onnxruntime / OpenVINO / insightface wheels do not exist for musl.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
      ffmpeg libgl1 libglib2.0-0 ca-certificates \
 && python3 -m venv /opt/venv \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-amd64.txt requirements-aarch64.txt /tmp/

RUN /opt/venv/bin/pip install -U pip wheel setuptools \
 && /opt/venv/bin/pip install "numpy<2" cython \
 && /opt/venv/bin/pip install -r /tmp/requirements.txt \
 && /opt/venv/bin/pip install -r /tmp/requirements-${BUILD_ARCH}.txt

COPY app /app/app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

WORKDIR /app
CMD ["/run.sh"]
