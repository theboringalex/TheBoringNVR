<p align="center">
  <img src="assets/logo.svg" alt="BoringNVR" width="180">
</p>

# TheBoringNVR

Event-driven NVR with HomeKit-Secure-Video-style face recognition for Home Assistant.

- **Event-driven:** no continuous object detection. The camera's own AI (Reolink person / vehicle / pet / motion) triggers everything.
- **go2rtc:** snapshots, live view and clips come from an existing go2rtc instance.
- **Faces:** InsightFace (optional Intel OpenVINO), multi-frame voting over 3-5 snapshots, cosine-similarity matching, circular face crops.
- **People directory:** create people, upload photos, review queue for unknown faces (assign / ignore with one click).
- **Pre-roll clips:** a lightweight `ffmpeg -c copy` ring buffer (no decoding) provides 2-4 s of pre-roll.
- **GUI:** dark dashboard via Home Assistant Ingress (events, faces, cameras, settings).
- **Companion card:** [TheBoringCameraCardv2](https://github.com/theboringalex/TheBoringCameraCardv2).

> Status: 0.1.1. Early version - expect to iterate.

## Architecture

```
Reolink camera --(HTTP GetAiState, 1 Hz)--> ReolinkListener --event--> Pipeline
                                                                         |-- go2rtc /api/frame.jpeg  -> snapshot + 3-5 frames
go2rtc --(RTSP restream)--> ffmpeg -c copy ring buffer ---- clip --------|-- FaceEngine (InsightFace/OpenVINO) -> person / review
                                                                         '-- SQLite + files in storage_path
Lovelace card <-- REST + SSE (/api/*, token) ---- FastAPI ---- HA Ingress GUI
```

## Repository layout

```
TheBoringNVR/
|- repository.yaml
|- README.md
|- assets/logo.svg           logo source
'- theboringnvr/
   |- config.yaml  build.yaml  Dockerfile  run.sh  icon.png  logo.png
   |- requirements*.txt
   '- app/
      |- main.py            FastAPI app, REST + SSE API, auth
      |- config.py          options, runtime settings, camera store (credentials in /data, mode 600)
      |- db.py              SQLite (persons, embeddings, events, review)
      |- static/index.html  Ingress GUI
      '- services/
         |- reolink_listener.py   event listener (Reolink HTTP API)
         |- face_engine.py        InsightFace + voting + circular crops
         |- go2rtc_client.py      snapshots / stream test / RTSP URL
         |- recorder.py           ring buffer + clip assembly
         '- pipeline.py           event -> snapshot -> faces -> clip, retention
```

## Installation

1. In Home Assistant: **Settings -> Add-ons -> Add-on Store -> (...) -> Repositories** and add
   `https://github.com/theboringalex/TheBoringNVR`.
2. Install **TheBoringNVR** (the first build downloads/compiles ML dependencies and takes a while), then open the
   **Configuration** tab:
   - `go2rtc_url`: your go2rtc instance, e.g. `http://192.168.1.10:1984`. The go2rtc bundled inside HA Core listens on
     localhost only and cannot be reached from an add-on - use the go2rtc add-on or a standalone container.
   - `storage_path`: `/share/theboringnvr` (default) or `/media/theboringnvr`.
   - `inference_device`: `CPU`, or `GPU` / `AUTO:GPU,CPU` for Intel iGPU (uncomment `devices: - /dev/dri` in `config.yaml`).
3. Start the add-on, open **BoringNVR** in the sidebar.
4. **Cameras -> Add**: name, host, credentials and the go2rtc stream name; press **Test**.
5. **Faces**: add people and upload a few photos each, or let unknown faces arrive in the review queue and assign them.
6. Copy the API token from **Settings** (also printed in the add-on log) into the card config.

### Webhook mode

For cameras without a state API, set the camera mode to *Webhook only* and POST to
`/api/webhook/<camera_id>?type=person` (header `X-API-Key: <token>`) from an HA automation or the camera's push rule.

## API (token via `X-API-Key` header or `?token=`)

`GET /api/events`, `GET /api/events/stream` (SSE), `GET/POST/PATCH/DELETE /api/persons`, `POST /api/persons/{id}/photos`,
`GET /api/review`, `POST /api/review/{id}/assign|ignore`, `GET/POST/PUT/DELETE /api/cameras`, `POST /api/cameras/test`,
`GET/PUT /api/settings`, `GET /api/media/{path}`. Requests coming through HA Ingress are trusted automatically.

## Known limitations / TODO

- ONVIF event listener is not implemented yet (use webhook mode).
- One face (the best) per person event; multi-person events are not split.
- Clips are remuxed, not transcoded: H.265 cameras produce H.265 clips.
- Base image tags and OpenVINO provider options should be verified on the first build.
