"""Pre-roll ring buffer + event clips.

go2rtc cannot rewind a live stream, so to get a 2-4 s pre-buffer we run one `ffmpeg -c copy` per camera
that remuxes the go2rtc RTSP restream into 2 s MPEG-TS segments (no decoding -> negligible CPU). Only the
last ~pre+post+margin seconds are kept. When an event fires we concatenate the right segments into an MP4.

H.265 cameras: the clip is H.265 as well (we only remux). Safari plays it, most Chromium builds need
hardware HEVC support. Set the camera to H.264 or add a go2rtc `ffmpeg:...#video=h264` source if needed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

log = logging.getLogger("nvr.rec")
SEG = 2  # seconds


class Recorder:
    def __init__(self, cfg, go2rtc) -> None:
        self.cfg, self.go2rtc = cfg, go2rtc
        self.tasks: dict[str, asyncio.Task] = {}

    def _dir(self, cam_id: str) -> Path:
        d = Path(self.cfg.buffer_path) / cam_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def start(self, cam_id: str, src: str) -> None:
        await self.stop(cam_id)
        self.tasks[cam_id] = asyncio.create_task(self._supervise(cam_id, src))

    async def stop(self, cam_id: str) -> None:
        t = self.tasks.pop(cam_id, None)
        if t:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _supervise(self, cam_id: str, src: str) -> None:
        d = self._dir(cam_id)
        while True:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
                   "-i", self.go2rtc.rtsp_url(src), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                   "-f", "segment", "-segment_time", str(SEG), "-segment_format", "mpegts",
                   "-reset_timestamps", "1", "-strftime", "1", str(d / "%s.ts")]
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
                cleaner = asyncio.create_task(self._cleanup(d))
                _, err = await proc.communicate()
                cleaner.cancel()
                log.warning("[%s] ring buffer ffmpeg exited (%s) %s", cam_id, proc.returncode,
                            (err or b"").decode()[-200:])
            except asyncio.CancelledError:
                if proc and proc.returncode is None:
                    proc.kill()
                raise
            await asyncio.sleep(5)

    async def _cleanup(self, d: Path) -> None:
        keep = self.cfg.pre_roll_seconds + self.cfg.post_roll_seconds + 30
        while True:
            await asyncio.sleep(10)
            cutoff = time.time() - keep
            for f in d.glob("*.ts"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass

    async def make_clip(self, cam_id: str, t_event: float, out: Path) -> bool:
        pre, post = self.cfg.pre_roll_seconds, self.cfg.post_roll_seconds
        wait = (t_event + post + SEG + 1) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        d = self._dir(cam_id)
        segs = []
        for f in d.glob("*.ts"):
            try:
                start = int(f.stem)
            except ValueError:
                continue
            if t_event - pre - SEG <= start <= t_event + post:
                segs.append((start, f))
        segs.sort()
        if len(segs) < 2:
            log.warning("[%s] no buffered segments for event at %s", cam_id, t_event)
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        listfile = out.with_suffix(".txt")
        listfile.write_text("".join(f"file '{f}'\n" for _, f in segs))
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile), "-c", "copy", "-movflags", "+faststart", str(out),
            stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        listfile.unlink(missing_ok=True)
        if proc.returncode != 0:
            log.error("[%s] clip concat failed: %s", cam_id, err.decode()[-300:])
            return False
        return out.exists()
