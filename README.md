<p align="center">
  <img src="assets/logo.svg" alt="BoringNVR" width="180">
</p>

# TheBoringNVR

Event-driven NVR with HomeKit-Secure-Video-style face recognition for Home Assistant.

- **Event-driven:** no continuous object detection. The camera's own AI (Reolink person / vehicle / pet / motion) triggers everything.
- **Selectable events:** choose globally (Settings) which event types are processed - person, vehicle, animal, motion - and override the list per camera.
- **Linked to the camera's recording:** every camera has a recording directory (FTP target, network share, ...). Each event is automatically linked to the video file that covers its time stamp, including the playback offset.
- **go2rtc:** snapshots, live view and clips come from an existing go2rtc instance.
- **Faces:** InsightFace (optional Intel OpenVINO), multi-frame voting over 3-5 snapshots, cosine-similarity matching, circular face crops.
- **People directory:** create people, upload photos, review queue for unknown faces (assign / ignore with one click).
- **Pre-roll clips:** a lightweight `ffmpeg -c copy` ring buffer (no decoding) provides 2-4 s of pre-roll and serves as fallback when the camera recording is not available.
- **GUI:** dark dashboard via Home Assistant Ingress (events, faces, cameras, settings).
- **Companion card:** [TheBoringCameraCardv2](https://github.com/theboringalex/TheBoringCameraCardv2).

> Status: 0.2.0. Early version - expect to iterate.

## Architecture

```
Reolink camera --(HTTP GetAiState, 1 Hz)--> ReolinkListener --event--> Pipeline (event type filter)
                                                                         |-- go2rtc /api/frame.jpeg  -> snapshot + 3-5 frames
go2rtc --(RTSP restream)--> ffmpeg -c copy ring buffer ---- clip --------|-- FaceEngine (InsightFace/OpenVINO) -> person / review
Camera --(FTP / share)--> recording_path --- RecordingIndex ---- link ---|-- SQLite + files in storage_path
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
         |- recording_index.py    finds the camera recording that covers an event
         '- pipeline.py           event filter -> snapshot -> faces -> clip -> recording link, retention
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
4. **Settings**: tick the event types to react to (person, vehicle, animal, motion).
5. **Cameras -> Add**: name, host, credentials, the go2rtc stream name and the **recording directory**; press **Test**.
6. **Faces**: add people and upload a few photos each, or let unknown faces arrive in the review queue and assign them.
7. Copy the API token from **Settings** (also printed in the add-on log) into the card config.

## Event types

Settings -> *React to events* is the global default. In a camera's settings you can either keep *Use global setting*
or select a custom list for that camera. Events of a type that is not selected are dropped before anything is
stored, so they cost nothing (no snapshot, no clip, no recording link).

## Camera recordings (FTP etc.)

The NVR does not record the continuous stream itself. The camera (or any other recorder) writes its files into a
directory, and the NVR links events to them:

1. Make the target available to the add-on **below `/media` or `/share`**. Typical setups:
   - the camera uploads via FTP to a server on your NAS and the share is mounted in HA
     (**Settings -> System -> Storage -> Add network storage**, usage *Media* or *Share*), or
   - an FTP add-on in Home Assistant writes to `/media/reolink/<camera>`.
2. Enter this folder as **Recording directory** of the camera, for example `/media/reolink/front_door`.
   The directory must exist; sub folders (date folders) are searched recursively.
3. When an event arrives, the NVR waits for the camera to finish and upload the file (up to *Wait for recording
   upload*, default 900 s), finds the video that covers the event time stamp and stores the file and the playback offset.
   The event then shows a recording symbol; in the event dialog the camera recording is the primary source and the
   internal pre-roll clip is the fallback.

How the time stamp is determined:

| Source | Used when |
|---|---|
| time in the file name (`Cam_00_20261023121530.mp4`, `RecS01_20261023_121530_121645_0_x.mp4`, `2026-10-23 12.15.30.mp4`) | the name contains a date and time; it is interpreted in *Recording file time zone* (default `Europe/Berlin`) |
| file modification time minus duration (ffprobe) | the name has no time stamp |

If an event shows *no camera recording found*, check the recording directory, the time zone setting and whether the
camera actually recorded at that moment (the camera's own recording schedule decides that), then use
*Search recording again* in the event dialog. The NVR never deletes files in the recording directory - rotate them on the
camera / FTP side. *Retention* only affects snapshots, face crops and buffer clips owned by the NVR.

### Webhook mode

For cameras without a state API, set the camera mode to *Webhook only* and POST to
`/api/webhook/<camera_id>?type=person` (header `X-API-Key: <token>`) from an HA automation or the camera's push rule.
The event type filter applies here as well.

## API (token via `X-API-Key` header or `?token=`)

`GET /api/events`, `GET /api/events/stream` (SSE), `GET /api/events/{id}/recording` (camera recording, start at `rec_offset`),
`POST /api/events/{id}/relink`, `GET/POST/PATCH/DELETE /api/persons`, `POST /api/persons/{id}/photos`,
`GET /api/review`, `POST /api/review/{id}/assign|ignore`, `GET/POST/PUT/DELETE /api/cameras`, `POST /api/cameras/test`,
`GET/PUT /api/settings`, `GET /api/media/{path}`. Requests coming through HA Ingress are trusted automatically.

Event objects contain `rec_status` (`linked`, `pending`, `missing`, `none`), `recording` (URL) and `rec_offset` (seconds).

## Upgrading from 0.1.x

The database is migrated automatically. Existing cameras have no recording directory yet: open **Cameras -> Edit**,
enter the directory and save (saving is refused without it).

## Known limitations / TODO

- ONVIF event listener is not implemented yet (use webhook mode).
- One face (the best) per person event; multi-person events are not split.
- Clips are remuxed, not transcoded: H.265 cameras produce H.265 clips.
- A recording is found only after the camera has finished and uploaded the file; very long continuous files are matched by file name time.
- Base image tags and OpenVINO provider options should be verified on the first build.
