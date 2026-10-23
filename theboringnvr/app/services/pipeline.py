"""Event pipeline + camera manager.

camera event -> DB row + snapshot (instant) -> [person: 3-5 frames -> face engine -> person/review]
             -> clip (pre-roll + post-roll) -> SSE update to the GUI / card
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from .reolink_listener import ReolinkListener

log = logging.getLogger("nvr.pipeline")


class Pipeline:
    def __init__(self, cfg, db, cams, go2rtc, engine, recorder) -> None:
        self.cfg, self.db, self.cams = cfg, db, cams
        self.go2rtc, self.engine, self.recorder = go2rtc, engine, recorder
        self.listeners: dict[str, ReolinkListener] = {}
        self.listener_tasks: dict[str, asyncio.Task] = {}
        self.subscribers: set[asyncio.Queue] = set()
        self._last: dict[tuple[str, str], float] = {}
        self._face_sem = asyncio.Semaphore(1)   # one inference job at a time

    # ---------- camera lifecycle ----------
    async def start_camera(self, cam_id: str) -> None:
        await self.stop_camera(cam_id)
        cam = self.cams.get(cam_id)
        if not cam:
            return
        src = cam.get("go2rtc_src") or cam_id
        if cam.get("buffer", True):
            await self.recorder.start(cam_id, src)
        if cam.get("mode", "reolink") == "reolink":
            lst = ReolinkListener(cam_id, cam, self.handle_event, verify_tls=bool(self.cfg.verify_tls))
            self.listeners[cam_id] = lst
            self.listener_tasks[cam_id] = asyncio.create_task(lst.run())

    async def stop_camera(self, cam_id: str) -> None:
        t = self.listener_tasks.pop(cam_id, None)
        self.listeners.pop(cam_id, None)
        if t:
            t.cancel()
        await self.recorder.stop(cam_id)

    async def start_all(self) -> None:
        for cam_id in list(self.cams.cams):
            await self.start_camera(cam_id)

    def status(self) -> dict:
        return {cid: {"connected": l.connected, "error": l.last_error} for cid, l in self.listeners.items()}

    # ---------- SSE ----------
    def publish(self, payload: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ---------- helpers ----------
    def event_dir(self, cam_id: str, ts: float) -> Path:
        return self.cfg.storage / "events" / cam_id / datetime.fromtimestamp(ts).strftime("%Y%m%d")

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.cfg.storage))

    def event_json(self, ev: dict) -> dict:
        person = None
        if ev.get("person_id"):
            p = self.db.one("SELECT id, name, avatar FROM persons WHERE id=?", (ev["person_id"],))
            if p:
                person = {"id": p["id"], "name": p["name"],
                          "avatar": f"/api/media/{p['avatar']}" if p["avatar"] else None}
        m = lambda v: f"/api/media/{v}" if v else None  # noqa: E731
        return {"id": ev["id"], "camera": ev["camera"], "type": ev["type"], "ts": ev["ts"],
                "person": person, "confidence": ev["confidence"], "status": ev["status"],
                "snapshot": m(ev["snapshot"]), "face": m(ev["face"]), "clip": m(ev["clip"])}

    def publish_event(self, event_id: int) -> None:
        ev = self.db.one("SELECT * FROM events WHERE id=?", (event_id,))
        if ev:
            self.publish(self.event_json(ev))

    # ---------- main entry ----------
    async def handle_event(self, cam_id: str, etype: str, ts: float | None = None) -> int | None:
        cam = self.cams.get(cam_id)
        if not cam:
            return None
        ts = ts or time.time()
        cd = self.cfg.event_cooldown_seconds
        if ts - self._last.get((cam_id, etype), 0) < cd:
            return None
        if etype == "motion" and any(ts - t < cd for (c, _), t in self._last.items() if c == cam_id):
            return None   # motion is noise when a classified event just happened
        self._last[(cam_id, etype)] = ts

        src = cam.get("go2rtc_src") or cam_id
        d = self.event_dir(cam_id, ts)
        d.mkdir(parents=True, exist_ok=True)
        event_id = self.db.x("INSERT INTO events(camera,type,ts,status) VALUES(?,?,?,'new')", (cam_id, etype, ts))
        try:
            jpg = await self.go2rtc.snapshot(src)
            snap = d / f"{event_id}_snap.jpg"
            snap.write_bytes(jpg)
            self.db.x("UPDATE events SET snapshot=? WHERE id=?", (self.rel(snap), event_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] snapshot failed: %s", cam_id, exc)
        self.publish_event(event_id)

        asyncio.create_task(self._clip(event_id, cam_id, ts, d))
        if etype == "person":
            asyncio.create_task(self._faces(event_id, cam_id, src, d))
        return event_id

    async def _clip(self, event_id: int, cam_id: str, ts: float, d: Path) -> None:
        out = d / f"{event_id}_clip.mp4"
        try:
            if await self.recorder.make_clip(cam_id, ts, out):
                self.db.x("UPDATE events SET clip=? WHERE id=?", (self.rel(out), event_id))
                self.publish_event(event_id)
        except Exception:  # noqa: BLE001
            log.exception("clip failed for event %s", event_id)

    async def _faces(self, event_id: int, cam_id: str, src: str, d: Path) -> None:
        async with self._face_sem:
            try:
                frames: list[bytes] = []
                for i in range(int(self.cfg.frames_per_event)):
                    try:
                        frames.append(await self.go2rtc.snapshot(src))
                    except Exception as exc:  # noqa: BLE001
                        log.debug("frame %d failed: %s", i, exc)
                    if i < self.cfg.frames_per_event - 1:
                        await asyncio.sleep(self.cfg.frame_interval_ms / 1000)
                if not frames:
                    return
                res = await asyncio.to_thread(self.engine.analyze_frames, frames)
            except Exception:  # noqa: BLE001
                log.exception("face analysis failed for event %s", event_id)
                return
        if res is None:
            self.db.x("UPDATE events SET status='no_face' WHERE id=?", (event_id,))
            self.publish_event(event_id)
            return

        face_path = d / f"{event_id}_face.png"
        face_path.write_bytes(res.crop_png)
        face_rel = self.rel(face_path)
        if res.person_id is not None:
            self.db.x("UPDATE events SET person_id=?, confidence=?, face=?, status='known' WHERE id=?",
                      (res.person_id, res.score, face_rel, event_id))
            al = float(self.cfg.auto_learn_threshold)
            if al > 0 and res.score >= al:
                self.db.add_embedding(res.person_id, res.embedding, source="auto")
        else:
            self.db.x("UPDATE events SET confidence=?, face=?, status='unknown' WHERE id=?",
                      (res.score, face_rel, event_id))
            self.db.x("""INSERT INTO review(event_id, face, vec, best_person_id, best_score, status, created)
                         VALUES(?,?,?,?,?, 'open', ?)""",
                      (event_id, face_rel, res.embedding.tobytes(), res.best_person_id, res.score, time.time()))
        self.publish_event(event_id)

    # ---------- retention ----------
    async def retention_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)
            days = int(self.cfg.retention_days)
            if days <= 0:
                continue
            cutoff = time.time() - days * 86400
            for ev in self.db.q("SELECT id, snapshot, face, clip FROM events WHERE ts < ?", (cutoff,)):
                for k in ("snapshot", "face", "clip"):
                    if ev[k]:
                        (self.cfg.storage / ev[k]).unlink(missing_ok=True)
                self.db.x("DELETE FROM events WHERE id=?", (ev["id"],))
            for d in (self.cfg.storage / "events").glob("*/*"):
                if d.is_dir() and not any(d.iterdir()):
                    shutil.rmtree(d, ignore_errors=True)


def sse_format(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
