#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
export NVR_DATA_DIR=/data
bashio::log.info "Starting TheBoringNVR (ingress on :8099)"
cd /app || exit 1
# No --proxy-headers: the app must see the real peer address (172.30.32.2 = HA ingress) for its auth check.
exec /opt/venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8099 \
  --log-level "$(bashio::config 'log_level')"
