"""Links NVR events to the video files a camera records on its own (FTP upload, SMB/NFS share, ...).

Every camera has a `recording_path`: the directory (below /media or /share) where its recordings land.
After an event we look for the file whose time range contains the event timestamp:

  * start time  - taken from the file name (Reolink: Cam_00_20261023121530.mp4,
                  RecS01_20261023_121530_121645_0_xxx.mp4, 2026-10-23 12.15.30 ...), interpreted in
                  `recording_timezone`; if the name carries no time, `mtime - duration` is used
  * duration    - from the name when it contains an end time, otherwise ffprobe
The result is (file, offset) where offset is the position in seconds to start playback (minus pre-roll).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("nvr.recindex")

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"}
ALLOWED_ROOTS = ("/media", "/share")        # folders mapped into the add-on
MAX_SCAN = 50000                             # safety cap per directory walk

_PAT_SEP = re.compile(r"(20\d{2})[-_.](\d{2})[-_.](\d{2})[T _-](\d{2})[-_.:](\d{2})[-_.:](\d{2})")
_PAT_RANGE = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?:[_-](\d{2})(\d{2})(\d{2}))?(?!\d)")
_PAT_COMPACT = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)")


def valid_recording_path(path: str) -> tuple[bool, str]:
    """Used when saving a camera: must exist and live below a mapped folder."""
    if not path or not path.strip():
        return False, "recording_path is required (directory where the camera's recordings are stored)"
    p = Path(os.path.normpath(path.strip()))
    if not p.is_absolute():
        return False, "recording_path must be an absolute path, e.g. /media/reolink/front_door"
    try:
        real = p.resolve()
    except OSError as exc:
        return False, f"cannot resolve path: {exc}"
    if not any(real == Path(r) or Path(r) in real.parents for r in ALLOWED_ROOTS):
        return False, "recording_path must be below /media or /share (mount network storage via HA first)"
    if not real.is_dir():
        return False, f"directory does not exist: {p} (create it / check the FTP target)"
    return True, ""


def _epoch(tz, y, mo, d, h, mi, s) -> float | None:
    try:
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=tz).timestamp()
    except ValueError:
        return None


def parse_filename(name: str, tz) -> tuple[float | None, float | None]:
    """-> (start, end) epoch seconds found in the file name; either may be None."""
    m = _PAT_SEP.search(name)
    if m:
        return _epoch(tz, *m.groups()), None
    m = _PAT_RANGE.search(name)
    if m:
        g = m.groups()
        start = _epoch(tz, *g[:6])
        end = None
        if g[6] is not None and start is not None:
            end = _epoch(tz, g[0], g[1], g[2], g[6], g[7], g[8])
            if end is not None and end < start:
                end += 86400
        return start, end
    m = _PAT_COMPACT.search(name)
    if m:
        return _epoch(tz, *m.groups()), None
    return None, None


@dataclass
class Rec:
    path: Path
    start: float
    end: float


class RecordingIndex:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._cache: dict[str, tuple[int, int, Rec | None]] = {}

    def _tz(self):
        try:
            return ZoneInfo(str(self.cfg.recording_timezone))
        except Exception:  # noqa: BLE001
            log.warning("unknown recording_timezone %r - using UTC", self.cfg.recording_timezone)
            return timezone.utc

    @staticmethod
    async def _duration(path: Path) -> float | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(proc.communicate(), 15)
            return float(out.decode().strip())
        except Exception:  # noqa: BLE001
            return None

    async def _describe(self, p: Path, st: os.stat_result, tz) -> Rec | None:
        key = str(p)
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._cache.get(key)
        if hit and hit[:2] == sig:
            return hit[2]
        start, end = parse_filename(p.name, tz)
        if start is None or end is None:
            dur = await self._duration(p)
            if start is None:
                if dur is None:
                    rec = None
                    self._cache[key] = (*sig, rec)
                    return rec
                end = st.st_mtime
                start = end - dur
            else:
                end = start + dur if dur else st.st_mtime
        rec = Rec(p, start, end)
        self._cache[key] = (*sig, rec)
        if len(self._cache) > 20000:
            self._cache.clear()
        return rec

    @staticmethod
    def _walk(root: str, since: float) -> list[tuple[Path, os.stat_result]]:
        out: list[tuple[Path, os.stat_result]] = []
        stack, seen = [root], 0
        while stack and seen < MAX_SCAN:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        seen += 1
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(e.path)
                            elif os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                                st = e.stat()
                                if st.st_mtime >= since and st.st_size > 0:
                                    out.append((Path(e.path), st))
                        except OSError:
                            continue
            except OSError:
                continue
        return out

    async def find(self, root: str, ts: float, pre_roll: float = 0.0) -> tuple[Path, float] | None:
        """File containing timestamp `ts` -> (path, playback offset in seconds) or None (not there yet)."""
        tz = self._tz()
        files = await asyncio.to_thread(self._walk, root, ts - 5)
        best: Rec | None = None
        for p, st in files:
            rec = await self._describe(p, st, tz)
            if rec and rec.start - 2 <= ts <= rec.end + 2:
                if best is None or (rec.end - rec.start) < (best.end - best.start):
                    best = rec          # prefer the shortest (most specific) file
        if best is None:
            return None
        return best.path, max(0.0, ts - best.start - pre_roll)


def check_dir(path: str) -> dict:
    """Used by the camera 'Test' button."""
    ok, msg = valid_recording_path(path)
    if not ok:
        return {"ok": False, "error": msg}
    files = RecordingIndex._walk(os.path.normpath(path.strip()), 0)
    if not files:
        return {"ok": True, "files": 0, "warning": "directory is readable but contains no video files yet"}
    newest = max(files, key=lambda x: x[1].st_mtime)
    return {"ok": True, "files": len(files), "newest": newest[0].name,
            "newest_age_seconds": int(time.time() - newest[1].st_mtime)}
