"""SQLite store: persons, face embeddings (float32 blobs), events and the review queue."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS persons(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, avatar TEXT, created REAL);
CREATE TABLE IF NOT EXISTS embeddings(
  id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  vec BLOB NOT NULL, source TEXT, created REAL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, camera TEXT NOT NULL, type TEXT NOT NULL, ts REAL NOT NULL,
  person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  confidence REAL, snapshot TEXT, face TEXT, clip TEXT, status TEXT DEFAULT 'new');
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_cam ON events(camera, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id, ts DESC);
CREATE TABLE IF NOT EXISTS review(
  id INTEGER PRIMARY KEY, event_id INTEGER REFERENCES events(id) ON DELETE CASCADE,
  face TEXT NOT NULL, vec BLOB NOT NULL, best_person_id INTEGER, best_score REAL,
  status TEXT DEFAULT 'open', created REAL);
"""


class DB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self.emb_version = 0   # bumped whenever embeddings change -> engine reloads its matrix

    def q(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid or 0

    # ---- embeddings ----
    def add_embedding(self, person_id: int, vec: np.ndarray, source: str = "manual", cap: int = 25) -> None:
        self.x("INSERT INTO embeddings(person_id, vec, source, created) VALUES(?,?,?,?)",
               (person_id, vec.astype(np.float32).tobytes(), source, time.time()))
        # keep the newest `cap` embeddings per person
        self.x("""DELETE FROM embeddings WHERE person_id=? AND id NOT IN
                  (SELECT id FROM embeddings WHERE person_id=? ORDER BY id DESC LIMIT ?)""",
               (person_id, person_id, cap))
        self.emb_version += 1

    def embedding_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.q("SELECT person_id, vec FROM embeddings")
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 512), dtype=np.float32)
        ids = np.array([r["person_id"] for r in rows], dtype=np.int64)
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return ids, mat
