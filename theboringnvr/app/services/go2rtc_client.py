"""Thin go2rtc API client: snapshots, stream info, RTSP URL for the ring buffer.

Note: the go2rtc that ships *inside* Home Assistant Core listens on 127.0.0.1 only and is not reachable
from an add-on. Use the go2rtc add-on or a standalone go2rtc container and point `go2rtc_url` at it.
"""
from __future__ import annotations

import logging
from urllib.parse import quote, urlparse

import httpx

log = logging.getLogger("nvr.go2rtc")


class Go2RTC:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.http = httpx.AsyncClient(timeout=8.0)

    @property
    def base(self) -> str:
        return self.cfg.go2rtc_url.rstrip("/")

    def rtsp_url(self, src: str) -> str:
        host = urlparse(self.base).hostname or "localhost"
        return f"rtsp://{host}:{int(self.cfg.go2rtc_rtsp_port)}/{quote(src)}"

    async def streams(self) -> dict:
        r = await self.http.get(f"{self.base}/api/streams")
        r.raise_for_status()
        return r.json()

    async def snapshot(self, src: str) -> bytes:
        """Single JPEG frame from the (already running or on-demand) stream."""
        r = await self.http.get(f"{self.base}/api/frame.jpeg", params={"src": src})
        r.raise_for_status()
        if not r.content:
            raise RuntimeError(f"go2rtc returned an empty frame for '{src}'")
        return r.content

    async def test(self, src: str) -> dict:
        try:
            streams = await self.streams()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"go2rtc unreachable: {exc}"}
        if src not in streams:
            return {"ok": False, "error": f"stream '{src}' not found in go2rtc", "available": sorted(streams)}
        try:
            jpg = await self.snapshot(src)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"snapshot failed: {exc}"}
        return {"ok": True, "snapshot_bytes": len(jpg), "rtsp": self.rtsp_url(src)}

    def webrtc_info(self, src: str) -> dict:
        """What the card needs for native WebRTC playback (it talks to go2rtc directly)."""
        return {"api": f"{self.base}/api/webrtc?src={quote(src)}", "ws": f"{self.base}/api/ws?src={quote(src)}"}

    async def close(self) -> None:
        await self.http.aclose()
