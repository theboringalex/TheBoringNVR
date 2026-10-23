"""Settings (HA add-on options + GUI overrides) and the camera store.

Camera credentials live in /data/cameras.json (add-on private volume, mode 600) and are never
returned by the API.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("NVR_DATA_DIR", "/data"))
OPTIONS_FILE = DATA_DIR / "options.json"        # written by the HA supervisor
OVERRIDES_FILE = DATA_DIR / "settings.json"     # written by the GUI
CAMERAS_FILE = DATA_DIR / "cameras.json"
TOKEN_FILE = DATA_DIR / "api_token"

DEFAULTS: dict[str, Any] = {
    "storage_path": "/share/theboringnvr",
    "go2rtc_url": "http://homeassistant.local:1984",
    "go2rtc_rtsp_port": 8554,
    "api_token": "",
    "pre_roll_seconds": 3,
    "post_roll_seconds": 12,
    "frames_per_event": 4,
    "frame_interval_ms": 700,
    "match_threshold": 0.45,
    "auto_learn_threshold": 0.0,   # 0 = never add embeddings automatically
    "min_face_px": 60,
    "min_det_score": 0.6,
    "model_name": "buffalo_l",
    "inference_device": "CPU",
    "event_cooldown_seconds": 20,
    "retention_days": 14,
    "verify_tls": False,
    "log_level": "info",
    "buffer_path": "/tmp/theboringnvr-buffer",
    # which camera events are processed at all (a camera can override this list)
    "event_types": ["person", "vehicle", "animal", "motion"],
    # camera recordings (FTP etc.) are linked to events by timestamp
    "recording_timezone": "Europe/Berlin",       # timezone of the time stamps in the recording file names
    "recording_link_timeout_seconds": 900,       # how long to wait for the camera to upload the file
}

EVENT_TYPES = ("person", "vehicle", "animal", "motion")

# Keys that may be changed at runtime from the GUI (the rest needs the HA add-on options page).
EDITABLE = {
    "go2rtc_url", "go2rtc_rtsp_port", "pre_roll_seconds", "post_roll_seconds",
    "frames_per_event", "frame_interval_ms", "match_threshold", "auto_learn_threshold",
    "min_face_px", "min_det_score", "event_cooldown_seconds", "retention_days",
    "event_types", "recording_timezone", "recording_link_timeout_seconds",
}


class Settings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = dict(DEFAULTS)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in (OPTIONS_FILE, OVERRIDES_FILE):
            if f.exists():
                try:
                    self.values.update(json.loads(f.read_text()))
                except Exception:  # noqa: BLE001
                    pass
        if not self.values.get("api_token"):
            if TOKEN_FILE.exists():
                token = TOKEN_FILE.read_text().strip()
            else:
                token = secrets.token_urlsafe(24)
                TOKEN_FILE.write_text(token)
                os.chmod(TOKEN_FILE, 0o600)
            self.values["api_token"] = token

    def __getattr__(self, key: str) -> Any:
        try:
            return self.__dict__["values"][key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    @property
    def storage(self) -> Path:
        return Path(self.values["storage_path"])

    def public(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k != "api_token"}

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            overrides: dict[str, Any] = {}
            if OVERRIDES_FILE.exists():
                try:
                    overrides = json.loads(OVERRIDES_FILE.read_text())
                except Exception:  # noqa: BLE001
                    overrides = {}
            for k, v in patch.items():
                if k in EDITABLE:
                    overrides[k] = v
                    self.values[k] = v
            OVERRIDES_FILE.write_text(json.dumps(overrides, indent=2))
        return self.public()


_SLUG = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    return _SLUG.sub("_", name.lower()).strip("_") or "camera"


class CameraStore:
    """cameras.json: {id: {name, mode, host, port, https, username, password, channel,
    go2rtc_src, events (null = use the global event_types), recording_path, buffer}}"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cams: dict[str, dict[str, Any]] = {}
        if CAMERAS_FILE.exists():
            try:
                self.cams = json.loads(CAMERAS_FILE.read_text())
            except Exception:  # noqa: BLE001
                self.cams = {}

    def _save(self) -> None:
        CAMERAS_FILE.write_text(json.dumps(self.cams, indent=2))
        os.chmod(CAMERAS_FILE, 0o600)

    def get(self, cam_id: str) -> dict[str, Any] | None:
        return self.cams.get(cam_id)

    def upsert(self, cam: dict[str, Any], cam_id: str | None = None) -> str:
        with self._lock:
            cam_id = cam_id or slugify(cam["name"])
            old = self.cams.get(cam_id, {})
            if not cam.get("password"):          # empty password on edit = keep the old one
                cam["password"] = old.get("password", "")
            self.cams[cam_id] = {**old, **cam}
            self._save()
        return cam_id

    def delete(self, cam_id: str) -> None:
        with self._lock:
            self.cams.pop(cam_id, None)
            self._save()

    @staticmethod
    def public(cam_id: str, cam: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in cam.items() if k != "password"}
        out["id"] = cam_id
        out["has_password"] = bool(cam.get("password"))
        return out

    def all_public(self) -> list[dict[str, Any]]:
        return [self.public(i, c) for i, c in self.cams.items()]
