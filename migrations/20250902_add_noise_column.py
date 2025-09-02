# migrations/20250902_add_noise_column.py
import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys = ON;")
    return con

def backup_db(db_path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(db_path.suffix + f".bak.{ts}")
    shutil.copy2(db_path, backup_path)
    print(f"[backup] -> {backup_path}")
    return backup_path

def column_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table});")
    return any(row[1] == col for row in cur.fetchall())  # row[1] = name

def run_migration(db_path: Path) -> None:
    """
    Adds `noise` column and normalizes `mode` values into (mode, noise).
    - audio_denoised  -> mode='audio', noise='denoised'
    - audio_noise     -> mode='audio', noise='noise'
    - audio           -> mode='audio', noise='none' (default)
    - video           -> mode='video', noise='none'
    """
    print(f"[migrate] opening {db_path}")
    backup_db(db_path)

    con = connect(db_path)
    cur = con.cursor()

    try:
        print("[migrate] BEGIN")
        cur.execute("BEGIN;")

        # 1) add noise column if missing
        if not column_exists(cur, "embeddings", "noise"):
            print("[migrate] ALTER TABLE add column noise")
            cur.execute("ALTER TABLE embeddings ADD COLUMN noise TEXT NOT NULL DEFAULT 'none';")
        else:
            print("[migrate] column `noise` already exists, skipping ALTER")

        # 2) normalize mode -> (mode, noise)
        print("[migrate] updating rows…")
        cur.execute("""
            UPDATE embeddings
            SET mode='audio', noise='denoised'
            WHERE mode='audio_denoised';
        """)
        print(f"    set audio_denoised -> (audio, denoised): {cur.rowcount}")

        cur.execute("""
            UPDATE embeddings
            SET mode='audio', noise='noisy'
            WHERE mode='audio_noise';
        """)
        print(f"    set audio_noise    -> (audio, noisy):     {cur.rowcount}")

        # ensure video/audio plain have an explicit noise
        cur.execute("""
            UPDATE embeddings
            SET noise='none'
            WHERE (mode='video' OR mode='audio') AND (noise IS NULL OR TRIM(noise)='');
        """)
        # index for common filters
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_mode_noise_model
            ON embeddings(mode, noise, model_name);
        """)

        con.commit()
        print("[migrate] COMMIT")
    except Exception as e:
        con.rollback()
        print("[migrate] ROLLBACK due to error:", e, file=sys.stderr)
        raise
    finally:
        con.close()

def fetch_all(cur: sqlite3.Cursor, q: str) -> List[Tuple]:
    cur.execute(q)
    return cur.fetchall()

def run_checks(db_path: Path) -> None:
    print(f"[check] opening {db_path}")
    con = connect(db_path)
    cur = con.cursor()

    # current distincts
    combos = fetch_all(cur, """
        SELECT mode, noise, model_name, COUNT(*) AS n
        FROM embeddings
        GROUP BY mode, noise, model_name
        ORDER BY mode, noise, model_name;
    """)
    print("\n[check] (mode, noise, model_name) counts:")
    for mode, noise, model, n in combos:
        print(f"  {mode:<6} | {noise:<9} | {model:<20} | {n}")

    # legacy reconstruction to compare with pre-migration mental model
    legacy = fetch_all(cur, """
        SELECT
          CASE
            WHEN mode='audio' AND noise='denoised' THEN 'audio_denoised'
            WHEN mode='audio' AND noise='noise'     THEN 'audio_noise'
            ELSE mode
          END AS legacy_mode,
          COUNT(*) AS n
        FROM embeddings
        GROUP BY legacy_mode
        ORDER BY legacy_mode;
    """)
    print("\n[check] counts by legacy_mode view:")
    for legacy_mode, n in legacy:
        print(f"  {legacy_mode:<15} | {n}")

    # show distinct noise values as a sanity check
    noises = fetch_all(cur, "SELECT DISTINCT noise FROM embeddings ORDER BY noise;")
    print("\n[check] distinct noise values:", [row[0] for row in noises])

    con.close()

def main():
    parser = argparse.ArgumentParser(description="Add `noise` column and normalize modes.")
    parser.add_argument("--db", required=True, help="Path to SQLite database file")
    parser.add_argument("--check-only", action="store_true", help="Only run checks; do not migrate")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    if not args.check_only:
        run_migration(db_path)
    run_checks(db_path)

if __name__ == "__main__":
    main()



### HOW TO RUN
# migrate + checks
# python3 migrations/20250902_add_noise_column.py --db path/to/your.db

# # just run the sanity checks
# python3 migrations/20250902_add_noise_column.py --db path/to/your.db --check-only
