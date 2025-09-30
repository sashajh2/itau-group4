import argparse
import io
import os
import sqlite3
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from utils.config_loader import load_config

try:
    from dropbox_utils.dropbox_utils import get_client as get_dbx
    HAS_DROPBOX = True
except Exception:
    HAS_DROPBOX = False


def connect_sqlite(sqlite_path: str) -> sqlite3.Connection:
    # Read-only connection when possible
    uri = f"file:{os.path.abspath(sqlite_path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def connect_postgres(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def iter_segments(conn: sqlite3.Connection, chunk_size: int) -> Iterable[List[tuple]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT segment_id, source, video_id, video_path, start_time, duration,
               video_label, audio_label, audio_model, video_model, created_at
        FROM segments
        """
    )
    while True:
        rows = cur.fetchmany(chunk_size)
        if not rows:
            break
        yield rows


def iter_embeddings(conn: sqlite3.Connection, chunk_size: int) -> Iterable[List[tuple]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT embedding_id, segment_id, mode, noise, model_name, denoiser_name,
               shard_path, row_index, vector_dim, dtype,
               embedding_type, reducer_id, contraster_id, version, created_at
        FROM embeddings
        ORDER BY shard_path, row_index
        """
    )
    while True:
        rows = cur.fetchmany(chunk_size)
        if not rows:
            break
        yield rows


def download_shard(shard_path: str) -> np.ndarray:
    if shard_path.startswith("/") and HAS_DROPBOX:
        # Dropbox path
        dbx = get_dbx()
        _, resp = dbx.files_download(shard_path)
        return np.load(io.BytesIO(resp.content))
    # Local file path
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    return np.load(shard_path, mmap_mode='r')


def route_table(mode: str, model_name: str, vector_dim: int) -> str:
    # Adjust here if you later add more spaces
    # Normalize defensively
    try:
        mode = (mode or "").lower()
        model_name = (model_name or "").lower()
        vector_dim = int(vector_dim)
    except Exception:
        pass
    key = (mode, model_name, vector_dim)
    if key == ("audio", "hubert", 768):
        return "embeddings_audio_hubert"
    if key == ("audio", "openl3", 512):
        return "embeddings_audio_openl3"
    if key == ("video", "senet", 2048):
        return "embeddings_video_senet"
    return ""


def flush(pg_cur, table: str, buffer: List[tuple]):
    if not buffer:
        return
    # Explicitly cast last parameter to vector so pg parses our string literal
    tpl = "(" + ",".join(["%s"] * 12) + ",%s::vector)"
    execute_values(
        pg_cur,
        f"""
        INSERT INTO {table}(
          embedding_id, segment_id, model_name, mode, noise, denoiser_name,
          dtype, embedding_type, reducer_id, contraster_id, version, created_at, embedding
        ) VALUES %s
        ON CONFLICT (embedding_id) DO NOTHING
        """,
        buffer,
        template=tpl,
        page_size=min(5000, len(buffer)),
    )
    buffer.clear()


def migrate(sqlite_path: str, use_direct_url: bool, batch_segments: int, batch_embeddings: int):
    config = load_config()
    dsn = (
        config["database"]["postgres"]["neon_direct_url"]
        if use_direct_url
        else config["database"]["postgres"]["neon_database_url"]
    )

    print(f"🔗 Connecting to SQLite: {sqlite_path}")
    sconn = connect_sqlite(sqlite_path)

    print("🔗 Connecting to Neon Postgres")
    pconn = connect_postgres(dsn)
    pcur = pconn.cursor()

    # 1) Segments
    total_segments = 0
    for seg_batch in iter_segments(sconn, batch_segments):
        execute_values(
            pcur,
            """
            INSERT INTO segments(
              segment_id, source, video_id, video_path, start_time, duration,
              video_label, audio_label, audio_model, video_model, created_at
            ) VALUES %s
            ON CONFLICT (segment_id) DO NOTHING
            """,
            seg_batch,
            page_size=len(seg_batch),
        )
        pconn.commit()
        total_segments += len(seg_batch)
        print(f"  ↳ segments inserted so far: {total_segments}")

    # 2) Embeddings: process by shard_path
    print("📥 Streaming embeddings in batches…")
    buffers: Dict[str, List[tuple]] = defaultdict(list)

    current_shard = None
    shard_matrix = None

    processed = 0
    for batch in iter_embeddings(sconn, batch_embeddings):
        # Group rows by shard_path in-batch to minimize reloads
        by_shard: Dict[str, List[tuple]] = defaultdict(list)
        for row in batch:
            by_shard[row[6]].append(row)  # index 6 = shard_path

        for shard_path, rows in by_shard.items():
            if shard_path != current_shard:
                # Switch shard: drop previous matrix and load new
                shard_matrix = download_shard(shard_path)
                current_shard = shard_path
                try:
                    print(
                        f"  • loaded shard: {shard_path} "
                        f"type={type(shard_matrix)}, dtype={getattr(shard_matrix,'dtype',None)}, "
                        f"shape={getattr(shard_matrix,'shape',None)}"
                    )
                except Exception:
                    pass

            for (eid, sid, mode, noise, mname, dname,
                 spath, idx, vdim, dtype, etype, rid, cid, ver, created) in rows:
                table = route_table(mode, mname, vdim)
                if not table:
                    # Unknown/unsupported space; skip
                    continue
                try:
                    idx_int = int(idx)
                    # bounds check
                    if idx_int < 0 or (hasattr(shard_matrix, 'shape') and idx_int >= shard_matrix.shape[0]):
                        raise IndexError(
                            f"row_index out of bounds: {idx_int} not in [0,{getattr(shard_matrix,'shape',('?',))[0]})"
                        )
                    vec = shard_matrix[idx_int]
                except Exception as e:
                    print(f"⚠️ Could not fetch vector idx={idx} ({type(idx)} ) from {spath}: {e}")
                    continue
                # Serialize vector to pgvector literal: "[v1,v2,...]"
                if isinstance(vec, np.ndarray):
                    vec_list = vec.tolist()
                else:
                    vec_list = list(vec)
                vec_literal = "[" + ",".join(str(float(x)) for x in vec_list) + "]"

                rec = (
                    eid, sid, (mname or "").lower(), (mode or "").lower(), noise, dname,
                    dtype, etype, rid, cid, ver, created, vec_literal,
                )
                buffers[table].append(rec)

        # Flush per table when big
        for table, buf in buffers.items():
            if len(buf) >= 1000:
                flush(pcur, table, buf)
                pconn.commit()
        processed += len(batch)
        if processed % 5000 == 0:
            print(f"  ↳ processed embedding rows: {processed}")

    # Final flush
    for table, buf in buffers.items():
        flush(pcur, table, buf)
    pconn.commit()
    print("✅ Migration complete")


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite embeddings/segments to Neon Postgres")
    parser.add_argument("--sqlite", type=str, default="./sqlite3_tables/combined.sqlite3", help="Path to source SQLite file")
    parser.add_argument("--direct", action="store_true", help="Use neon_direct_url from config for Postgres")
    parser.add_argument("--seg-batch", type=int, default=5000, help="Segments batch size")
    parser.add_argument("--emb-batch", type=int, default=5000, help="Embeddings batch size (rows, not vectors)")
    args = parser.parse_args()

    migrate(args.sqlite, args.direct, args.seg_batch, args.emb_batch)


if __name__ == "__main__":
    raise SystemExit(main())


