import argparse
from datetime import datetime, timezone
import numpy as np
import psycopg2

from utils.config_loader import load_config
from data.preprocessing.storage.neon_writer import NeonEmbeddingWriter


def insert_segments(pg_dsn: str, segment_ids: list[str]) -> None:
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for sid in segment_ids:
        rows.append(
            (
                sid,                 # segment_id
                "SMOKE",            # source
                sid,                 # video_id
                None,                # video_path
                0.0,                 # start_time
                0.25,                # duration
                0,                   # video_label
                0,                   # audio_label
                None,                # audio_model
                None,                # video_model
                now,                 # created_at
            )
        )

    args_str = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"] * len(rows))
    flat = [v for row in rows for v in row]
    cur.execute(
        f"""
        INSERT INTO segments(
          segment_id, source, video_id, video_path, start_time, duration,
          video_label, audio_label, audio_model, video_model, created_at
        ) VALUES {args_str}
        ON CONFLICT (segment_id) DO NOTHING
        """,
        flat,
    )
    conn.commit()
    cur.close(); conn.close()


def main():
    parser = argparse.ArgumentParser(description="Neon direct writer smoke test (10 segments)")
    parser.add_argument("--count", type=int, default=10, help="Number of segments to create")
    parser.add_argument("--version", type=str, default="smoke-test", help="Version tag for embeddings")
    args = parser.parse_args()

    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]

    # 1) Insert N segments
    segment_ids = [f"smoke_seg_{i:02d}" for i in range(args.count)]
    insert_segments(dsn, segment_ids)

    # 2) Insert embeddings via NeonEmbeddingWriter (3 models per segment)
    writer = NeonEmbeddingWriter(version=args.version, batch_size=4)

    for i, sid in enumerate(segment_ids):
        # Deterministic simple vectors
        h768 = np.full(768, float(i), dtype=np.float32)
        o512 = np.full(512, float(i) + 0.5, dtype=np.float32)
        s2048 = np.full(2048, float(i) + 1.0, dtype=np.float32)

        writer.add("hubert", "audio", "none", "none", sid, h768)
        writer.add("openl3", "audio", "none", "none", sid, o512)
        writer.add("senet", "video", "none", "none", sid, s2048)

    writer.flush_all()
    writer.close()

    print(f"✅ Smoke write complete: segments={len(segment_ids)}, embeddings={len(segment_ids)*3}, version={args.version}")


if __name__ == "__main__":
    raise SystemExit(main())


