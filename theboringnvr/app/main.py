"""TheBoringNVR - FastAPI app (HA Ingress GUI on :8099, optional token-protected API for the card)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import EVENT_TYPES, CameraStore, Settings
from .db import DB
from .services.face_engine import FaceEngine
from .services.go2rtc_client import Go2RTC
from .services.pipeline import Pipeline, sse_format
from .services.recorder import Recorder
from .services.recording_index import check_dir, valid_recording_path
from .services.reolink_listener import ReolinkListener

INGRESS_IP = "172.30.32.2"   # HA supervisor ingress proxy

cfg = Settings()
logging.basicConfig(level=getattr(logging, str(cfg.log_level).upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nvr")

cfg.storage.mkdir(parents=True, exist_ok=True)
db = DB(cfg.storage / "nvr.sqlite")
cams = CameraStore()
go2rtc = Go2RTC(cfg)
engine = FaceEngine(cfg, db)
recorder = Recorder(cfg, go2rtc)
pipe = Pipeline(cfg, db, cams, go2rtc, engine, recorder)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("API token for the Lovelace card: %s", cfg.api_token)   # also shown in the GUI settings
    await pipe.start_all()
    bg = [asyncio.create_task(pipe.retention_loop())]
    yield
    for t in bg:
        t.cancel()
    for cid in list(cams.cams):
        await pipe.stop_camera(cid)
    await go2rtc.close()


app = FastAPI(title="TheBoringNVR", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["X-API-Key", "Content-Type"], allow_credentials=False)


async def auth(request: Request, x_api_key: str | None = Header(None), token: str | None = Query(None)):
    if request.client and request.client.host == INGRESS_IP:   # HA already authenticated the user
        return
    supplied = x_api_key or token or ""
    if not secrets.compare_digest(supplied, cfg.api_token):
        raise HTTPException(401, "invalid or missing API token")


api = APIRouter(prefix="/api", dependencies=[Depends(auth)])


# ------------------------------------------------------------------ models
class CameraIn(BaseModel):
    name: str
    mode: str = "reolink"            # reolink | webhook
    host: str = ""
    port: int | None = None
    https: bool = False
    username: str = "admin"
    password: str = ""
    channel: int = 0
    go2rtc_src: str = ""
    events: list[str] | None = None      # None / empty = use the global event types from Settings
    recording_path: str = ""             # where the camera stores its own recordings (FTP target etc.)
    buffer: bool = True                  # internal pre-roll ring buffer (fallback clip)


class PersonIn(BaseModel):
    name: str


class AssignIn(BaseModel):
    person_id: int | None = None
    new_person_name: str | None = None


# ------------------------------------------------------------------ system
@api.get("/system")
async def system():
    return {"version": "0.2.0", "engine": engine.provider_in_use, "cameras": pipe.status(),
            "api_token": cfg.api_token, "storage": str(cfg.storage), "event_types": list(EVENT_TYPES)}


@api.get("/settings")
async def get_settings():
    return cfg.public()


@api.put("/settings")
async def put_settings(patch: dict):
    if "event_types" in patch:
        types = [t for t in (patch["event_types"] or []) if t in EVENT_TYPES]
        if not types:
            raise HTTPException(422, "select at least one event type")
        patch["event_types"] = types
    if "recording_timezone" in patch:
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(str(patch["recording_timezone"]))
        except Exception:  # noqa: BLE001
            raise HTTPException(422, "unknown timezone - use an IANA name like Europe/Berlin")
    return cfg.update(patch)


# ------------------------------------------------------------------ cameras
@api.get("/cameras")
async def list_cameras():
    out = cams.all_public()
    for c in out:
        c["status"] = pipe.status().get(c["id"])
    return out


def camera_data(c: CameraIn) -> dict:
    """Validate and normalise a camera before it is stored. A recording directory is mandatory so every
    event can be linked to the camera's own recording."""
    ok, msg = valid_recording_path(c.recording_path)
    if not ok:
        raise HTTPException(422, msg)
    data = c.model_dump()
    data["recording_path"] = os.path.normpath(c.recording_path.strip())
    if data["events"]:
        bad = set(data["events"]) - set(EVENT_TYPES)
        if bad:
            raise HTTPException(422, f"unknown event type(s): {', '.join(sorted(bad))}")
    else:
        data["events"] = None
    return data


@api.post("/cameras")
async def add_camera(c: CameraIn):
    cam_id = cams.upsert(camera_data(c))
    await pipe.start_camera(cam_id)
    return {"id": cam_id}


@api.put("/cameras/{cam_id}")
async def edit_camera(cam_id: str, c: CameraIn):
    if not cams.get(cam_id):
        raise HTTPException(404)
    cams.upsert(camera_data(c), cam_id)
    await pipe.start_camera(cam_id)
    return {"id": cam_id}


@api.delete("/cameras/{cam_id}")
async def del_camera(cam_id: str):
    await pipe.stop_camera(cam_id)
    cams.delete(cam_id)
    return {"ok": True}


@api.post("/cameras/test")
async def test_camera(c: CameraIn):
    res: dict = {"go2rtc": await go2rtc.test(c.go2rtc_src or c.name),
                 "recording": await asyncio.to_thread(check_dir, c.recording_path)}
    if c.mode == "reolink":
        data = c.model_dump()
        if not data["password"]:     # re-test an existing camera without retyping the password
            old = cams.get(c.name) or next((v for v in cams.cams.values() if v["name"] == c.name), None)
            data["password"] = (old or {}).get("password", "")
        res["camera"] = await ReolinkListener.test(data, bool(cfg.verify_tls))
    return res


@api.get("/cameras/{cam_id}/snapshot")
async def cam_snapshot(cam_id: str):
    cam = cams.get(cam_id)
    if not cam:
        raise HTTPException(404)
    jpg = await go2rtc.snapshot(cam.get("go2rtc_src") or cam_id)
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


@api.get("/cameras/{cam_id}/webrtc")
async def cam_webrtc(cam_id: str):
    cam = cams.get(cam_id)
    if not cam:
        raise HTTPException(404)
    return go2rtc.webrtc_info(cam.get("go2rtc_src") or cam_id)


@api.post("/webhook/{cam_id}")
async def webhook(cam_id: str, type: str = "person"):
    """For cameras without a state API: POST here when the camera/automation sees something."""
    if type not in EVENT_TYPES:
        raise HTTPException(400, "type must be person|vehicle|animal|motion")
    return {"event_id": await pipe.handle_event(cam_id, type)}


# ------------------------------------------------------------------ events
@api.get("/events")
async def events(camera: str | None = None, person: str | None = None, type: str | None = None,
                 since: float | None = None, before: float | None = None, limit: int = Query(100, le=500)):
    """person: person id, 'unknown' or 'any' (any recognised person)."""
    where, args = [], []
    if camera:
        where.append("camera=?"); args.append(camera)
    if type:
        where.append("type=?"); args.append(type)
    if since:
        where.append("ts>=?"); args.append(since)
    if before:
        where.append("ts<?"); args.append(before)
    if person == "unknown":
        where.append("type='person' AND person_id IS NULL")
    elif person == "any":
        where.append("person_id IS NOT NULL")
    elif person:
        where.append("person_id=?"); args.append(int(person))
    sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY ts DESC LIMIT ?"
    return [pipe.event_json(e) for e in db.q(sql, tuple(args) + (limit,))]


@api.get("/events/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    pipe.subscribers.add(q)

    async def gen():
        try:
            while not await request.is_disconnected():
                try:
                    yield sse_format(await asyncio.wait_for(q.get(), 20))
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            pipe.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.get("/events/{event_id}")
async def event(event_id: int):
    ev = db.one("SELECT * FROM events WHERE id=?", (event_id,))
    if not ev:
        raise HTTPException(404)
    return pipe.event_json(ev)


@api.get("/events/{event_id}/recording")
async def event_recording(event_id: int):
    """The camera's own recording that covers this event (play from `rec_offset` seconds)."""
    ev = db.one("SELECT camera, recording FROM events WHERE id=?", (event_id,))
    cam = cams.get(ev["camera"]) if ev else None
    if not ev or not ev["recording"] or not cam or not cam.get("recording_path"):
        raise HTTPException(404)
    root = Path(cam["recording_path"]).resolve()
    path = (root / ev["recording"]).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@api.post("/events/{event_id}/relink")
async def relink_event(event_id: int):
    """Search the recording directory again (e.g. after a late FTP upload)."""
    if not pipe.relink(event_id):
        raise HTTPException(404)
    return {"ok": True}


@api.delete("/events/{event_id}")
async def delete_event(event_id: int):
    ev = db.one("SELECT * FROM events WHERE id=?", (event_id,))
    if ev:
        for k in ("snapshot", "face", "clip"):
            if ev[k]:
                (cfg.storage / ev[k]).unlink(missing_ok=True)
        db.x("DELETE FROM events WHERE id=?", (event_id,))
    return {"ok": True}


# ------------------------------------------------------------------ persons
def person_json(p: dict) -> dict:
    n = db.one("SELECT COUNT(*) c FROM embeddings WHERE person_id=?", (p["id"],))["c"]
    e = db.one("SELECT COUNT(*) c FROM events WHERE person_id=?", (p["id"],))["c"]
    return {"id": p["id"], "name": p["name"], "embeddings": n, "events": e,
            "avatar": f"/api/media/{p['avatar']}" if p["avatar"] else None}


@api.get("/persons")
async def persons():
    return [person_json(p) for p in db.q("SELECT * FROM persons ORDER BY name")]


@api.post("/persons")
async def create_person(p: PersonIn):
    try:
        pid = db.x("INSERT INTO persons(name, created) VALUES(?,?)", (p.name.strip(), time.time()))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(409, f"person exists? {exc}")
    return person_json(db.one("SELECT * FROM persons WHERE id=?", (pid,)))


@api.patch("/persons/{pid}")
async def rename_person(pid: int, p: PersonIn):
    db.x("UPDATE persons SET name=? WHERE id=?", (p.name.strip(), pid))
    return person_json(db.one("SELECT * FROM persons WHERE id=?", (pid,)))


@api.delete("/persons/{pid}")
async def delete_person(pid: int):
    row = db.one("SELECT avatar FROM persons WHERE id=?", (pid,))
    if row and row["avatar"]:
        (cfg.storage / row["avatar"]).unlink(missing_ok=True)
    db.x("DELETE FROM persons WHERE id=?", (pid,))
    db.emb_version += 1
    return {"ok": True}


@api.post("/persons/{pid}/photos")
async def add_photo(pid: int, file: UploadFile = File(...)):
    """Enrol a face from an uploaded photo."""
    res = await asyncio.to_thread(engine.embed_photo, await file.read())
    if res is None:
        raise HTTPException(422, "no usable face found in the photo")
    emb, crop = res
    db.add_embedding(pid, emb, source="photo")
    p = db.one("SELECT avatar FROM persons WHERE id=?", (pid,))
    if p and not p["avatar"]:
        path = cfg.storage / "persons" / f"{pid}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(crop)
        db.x("UPDATE persons SET avatar=? WHERE id=?", (f"persons/{pid}.png", pid))
    return person_json(db.one("SELECT * FROM persons WHERE id=?", (pid,)))


@api.get("/persons/{pid}/events")
async def person_events(pid: int, limit: int = 50):
    return [pipe.event_json(e) for e in db.q(
        "SELECT * FROM events WHERE person_id=? ORDER BY ts DESC LIMIT ?", (pid, limit))]


# ------------------------------------------------------------------ review queue
@api.get("/review")
async def review():
    rows = db.q("SELECT * FROM review WHERE status='open' ORDER BY created DESC LIMIT 200")
    out = []
    for r in rows:
        hint = db.one("SELECT id, name FROM persons WHERE id=?", (r["best_person_id"],)) if r["best_person_id"] else None
        ev = db.one("SELECT * FROM events WHERE id=?", (r["event_id"],))
        out.append({"id": r["id"], "event_id": r["event_id"], "face": f"/api/media/{r['face']}",
                    "score": r["best_score"], "closest": hint, "created": r["created"],
                    "camera": ev["camera"] if ev else None, "ts": ev["ts"] if ev else None})
    return out


@api.post("/review/{rid}/assign")
async def review_assign(rid: int, a: AssignIn):
    import numpy as np
    r = db.one("SELECT * FROM review WHERE id=?", (rid,))
    if not r:
        raise HTTPException(404)
    pid = a.person_id
    if pid is None and a.new_person_name:
        pid = db.x("INSERT INTO persons(name, created) VALUES(?,?)", (a.new_person_name.strip(), time.time()))
    if pid is None:
        raise HTTPException(400, "person_id or new_person_name required")
    db.add_embedding(pid, np.frombuffer(r["vec"], dtype=np.float32), source="review")
    db.x("UPDATE events SET person_id=?, status='known' WHERE id=?", (pid, r["event_id"]))
    db.x("UPDATE review SET status='assigned' WHERE id=?", (rid,))
    p = db.one("SELECT avatar FROM persons WHERE id=?", (pid,))
    if p and not p["avatar"]:
        src = cfg.storage / r["face"]
        dst = cfg.storage / "persons" / f"{pid}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        db.x("UPDATE persons SET avatar=? WHERE id=?", (f"persons/{pid}.png", pid))
    pipe.publish_event(r["event_id"])
    return {"ok": True, "person_id": pid}


@api.post("/review/{rid}/ignore")
async def review_ignore(rid: int):
    db.x("UPDATE review SET status='ignored' WHERE id=?", (rid,))
    return {"ok": True}


# ------------------------------------------------------------------ media
@api.get("/media/{rel:path}")
async def media(rel: str):
    root = cfg.storage.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


app.include_router(api)

STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()
