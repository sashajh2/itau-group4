import sqlite3
from typing import List, Dict, Iterable, Optional


def insert_segment(db_path, segment_dict):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO segments (
                segment_id, source, video_id, video_path,
                start_time, duration, video_label, audio_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment_dict["segment_id"],
            segment_dict["source"],
            segment_dict["video_id"],
            segment_dict["video_path"],
            segment_dict["start_time"],
            segment_dict["duration"],
            segment_dict["video_label"],
            segment_dict["audio_label"],
            segment_dict.get("created_at")  # Can be None; DB will default it
        ))
        conn.commit()


def get_segments_by_video_id(db_path, video_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM segments WHERE video_id = ?
        """, (video_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


_EMBED_COLS = [
    "embedding_id",
    "segment_id",
    "mode",
    "noise",
    "model_name",
    "denoiser_name",
    "shard_path",
    "row_index",
    "vector_dim",
    "dtype",
    "embedding_type",
    "reducer_id",
    "contraster_id",
    "version",
    "created_at",
]

def _as_tuple(row: Dict) -> tuple:
    """Order dict fields to match _EMBED_COLS."""
    return tuple(row.get(k) for k in _EMBED_COLS)

def insert_embedding(db_path: str, embedding_dict: Dict, on_conflict: str = "ABORT") -> None:
    """
    Insert a single embedding row.
    on_conflict: one of "ABORT" (default), "IGNORE", "REPLACE"
    """
    if on_conflict.upper() not in {"ABORT", "IGNORE", "REPLACE"}:
        raise ValueError("on_conflict must be ABORT, IGNORE, or REPLACE")
    placeholders = ",".join(["?"] * len(_EMBED_COLS))
    sql = f"INSERT OR {on_conflict.upper()} INTO embeddings ({','.join(_EMBED_COLS)}) VALUES ({placeholders})"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(sql, _as_tuple(embedding_dict))
        conn.commit()

def insert_many_embeddings(
    db_path: str,
    rows: Iterable[Dict],
    on_conflict: str = "IGNORE",
) -> int:
    """
    Batch insert many embedding rows inside ONE transaction (fast).
    Returns the number of rows that changed the DB (approx via total_changes delta).

    on_conflict:
      - "IGNORE": skip duplicates (recommended with your UNIQUE index)
      - "REPLACE": overwrite on conflict
      - "ABORT": fail the whole batch on first conflict
    """
    on_conflict = on_conflict.upper()
    if on_conflict not in {"ABORT", "IGNORE", "REPLACE"}:
        raise ValueError("on_conflict must be ABORT, IGNORE, or REPLACE")

    tuples = [_as_tuple(r) for r in rows]
    if not tuples:
        return 0

    placeholders = ",".join(["?"] * len(_EMBED_COLS))
    sql = f"INSERT OR {on_conflict} INTO embeddings ({','.join(_EMBED_COLS)}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        before = conn.total_changes
        conn.executemany(sql, tuples)
        conn.commit()
        after = conn.total_changes
        return after - before


def get_embeddings_by_segment(db_path, segment_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM embeddings WHERE segment_id = ?
        """, (segment_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


def get_segments_by_created_at(db_path, created_at):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM segments
            WHERE created_at == ?
        """, (created_at,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
