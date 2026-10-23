"""Event listener for Reolink cameras.

The camera does the detection (Person / Vehicle / Pet / Motion). We only watch its AI state over the
Reolink HTTP API (GetAiState / GetMdState, ~1 request per second, tiny payload) and emit an event on the
rising edge. There is no continuous object detection on our side.

Alternative for cameras/firmwares without a usable state API: use the webhook mode
(POST /api/webhook/<camera_id>?type=person) from a camera push rule, Node-RED or an HA automation.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("nvr.reolink")

AI_KEYS = {"people": "person", "vehicle": "vehicle", "dog_cat": "animal"}
OnEvent = Callable[[str, str], Awaitable[None]]


class ReolinkListener:
    def __init__(self, cam_id: str, cam: dict, on_event: OnEvent, verify_tls: bool = False,
                 poll_interval: float = 1.0) -> None:
        self.cam_id = cam_id
        self.cam = cam
        self.on_event = on_event
        self.verify_tls = verify_tls
        self.poll_interval = poll_interval
        self.token: str | None = None
        self.token_expiry = 0.0
        self.active: dict[str, bool] = {}
        self.last_error: str | None = None
        self.connected = False
        self.wanted = set(cam.get("events") or ["person", "vehicle", "animal", "motion"])

    @property
    def base(self) -> str:
        scheme = "https" if self.cam.get("https") else "http"
        default = 443 if self.cam.get("https") else 80
        return f"{scheme}://{self.cam['host']}:{int(self.cam.get('port') or default)}"

    async def _login(self, http: httpx.AsyncClient) -> None:
        body = [{"cmd": "Login", "param": {"User": {
            "Version": "0", "userName": self.cam["username"], "password": self.cam.get("password", "")}}}]
        r = await http.post(f"{self.base}/api.cgi?cmd=Login", json=body)
        r.raise_for_status()
        d = r.json()[0]
        if d.get("code") != 0:
            raise RuntimeError(f"login failed: {d.get('error', d)}")
        tok = d["value"]["Token"]
        self.token = tok["name"]
        self.token_expiry = time.time() + int(tok.get("leaseTime", 3600)) - 60
        log.info("[%s] logged in", self.cam_id)

    async def _call(self, http: httpx.AsyncClient, cmd: str, param: dict) -> dict:
        body = [{"cmd": cmd, "action": 0, "param": param}]
        r = await http.post(f"{self.base}/api.cgi", params={"cmd": cmd, "token": self.token}, json=body)
        r.raise_for_status()
        d = r.json()[0]
        if d.get("code") != 0:
            raise RuntimeError(f"{cmd} failed: {d.get('error', d)}")
        return d.get("value", {})

    @staticmethod
    def _is_on(v) -> bool:
        if isinstance(v, dict):
            return v.get("alarm_state") == 1
        return v == 1

    async def _poll(self, http: httpx.AsyncClient) -> None:
        ch = int(self.cam.get("channel") or 0)
        now_active: dict[str, bool] = {}
        try:
            ai = await self._call(http, "GetAiState", {"channel": ch})
            for key, etype in AI_KEYS.items():
                if key in ai:
                    now_active[etype] = self._is_on(ai[key])
        except RuntimeError:
            pass   # camera without AI: fall back to motion only
        md = await self._call(http, "GetMdState", {"channel": ch})
        now_active["motion"] = md.get("state") == 1

        for etype, on in now_active.items():
            if on and not self.active.get(etype) and etype in self.wanted:
                log.debug("[%s] %s started", self.cam_id, etype)
                asyncio.create_task(self.on_event(self.cam_id, etype))
        self.active = now_active

    async def run(self) -> None:
        backoff = 1.0
        async with httpx.AsyncClient(verify=self.verify_tls, timeout=5.0) as http:
            while True:
                try:
                    if not self.token or time.time() > self.token_expiry:
                        await self._login(http)
                    await self._poll(http)
                    self.connected, self.last_error, backoff = True, None, 1.0
                    await asyncio.sleep(self.poll_interval)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.connected, self.last_error, self.token = False, str(exc), None
                    log.warning("[%s] %s (retry in %.0fs)", self.cam_id, exc, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    @classmethod
    async def test(cls, cam: dict, verify_tls: bool = False) -> dict:
        """Used by the GUI 'Test' button."""
        probe = cls("test", cam, on_event=lambda *_: asyncio.sleep(0), verify_tls=verify_tls)
        try:
            async with httpx.AsyncClient(verify=verify_tls, timeout=5.0) as http:
                await probe._login(http)
                dev = await probe._call(http, "GetDevInfo", {})
                try:
                    ai = await probe._call(http, "GetAiState", {"channel": int(cam.get("channel") or 0)})
                except RuntimeError:
                    ai = {}
            supported = [AI_KEYS[k] for k in AI_KEYS if isinstance(ai.get(k), dict) and ai[k].get("support")]
            info = dev.get("DevInfo", {})
            return {"ok": True, "model": info.get("model"), "firmware": info.get("firmVer"),
                    "ai_supported": supported}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
