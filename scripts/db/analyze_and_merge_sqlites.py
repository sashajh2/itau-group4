import argparse
import os
import sqlite3
import json
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    from dropbox_utils.dropbox_utils import get_client
    _HAS_DROPBOX = True
except Exception:
    _HAS_DROPBOX = False


def list_sqlite_files(input_paths: List[str]) -> List[str]:
    files: List[str] = []
    for path in input_paths:
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.endswith('.sqlite3'):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path) and path.endswith('.sqlite3'):
            files.append(path)
    return sorted(files)


def get_table_info(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return cur.fetchall()


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    info = get_table_info(conn, table)
    return [row[1] for row in info]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def embeddings_shard_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT shard_path, COUNT(*) FROM embeddings GROUP BY shard_path")
    return {row[0]: int(row[1]) for row in cur.fetchall()}


def fetch_meta_json_for_shard(shard_path: str) -> Dict:
    # Try local first
    meta_path_local = shard_path
    if meta_path_local.endswith('.npy'):
        meta_path_local = meta_path_local[:-4] + '.meta.json'
    if os.path.exists(meta_path_local):
        with open(meta_path_local, 'r') as f:
            return json.load(f)

    # Try Dropbox if available
    if _HAS_DROPBOX and shard_path.startswith('/'):
        try:
            dbx = get_client()
            meta_path_dbx = shard_path[:-4] + '.meta.json'
            _, resp = dbx.files_download(meta_path_dbx)
            return json.loads(resp.content.decode('utf-8'))
        except Exception:
            return {}
    return {}


def analyze_db(db_path: str, verify_shards: bool) -> Dict:
    out: Dict = {
        'path': db_path,
        'has_segments': False,
        'has_embeddings': False,
        'segments_cols': [],
        'embeddings_cols': [],
        'segments_count': 0,
        'embeddings_count': 0,
        'shard_counts': {},
        'shard_meta_counts': {},
        'shard_meta_mismatch': {},
    }

    with sqlite3.connect(db_path) as conn:
        out['has_segments'] = table_exists(conn, 'segments')
        out['has_embeddings'] = table_exists(conn, 'embeddings')

        if out['has_segments']:
            out['segments_cols'] = get_columns(conn, 'segments')
            out['segments_count'] = count_rows(conn, 'segments')

        if out['has_embeddings']:
            out['embeddings_cols'] = get_columns(conn, 'embeddings')
            out['embeddings_count'] = count_rows(conn, 'embeddings')
            out['shard_counts'] = embeddings_shard_counts(conn)

            if verify_shards:
                for shard_path, cnt in out['shard_counts'].items():
                    meta = fetch_meta_json_for_shard(shard_path)
                    meta_count = int(meta.get('count', -1)) if meta else -1
                    out['shard_meta_counts'][shard_path] = meta_count
                    if meta_count >= 0 and meta_count != cnt:
                        out['shard_meta_mismatch'][shard_path] = {
                            'db_count': cnt,
                            'meta_count': meta_count,
                        }
    return out


def ensure_target_schema(target: sqlite3.Connection, segments_cols: List[str], embeddings_cols: List[str]):
    cur = target.cursor()

    # Create segments
    if not table_exists(target, 'segments'):
        cols_def = ', '.join([f'{c} TEXT' for c in segments_cols])
        cur.execute(f'CREATE TABLE segments ({cols_def})')
    else:
        # Add missing columns as TEXT
        existing = set(get_columns(target, 'segments'))
        for c in segments_cols:
            if c not in existing:
                cur.execute(f'ALTER TABLE segments ADD COLUMN {c} TEXT')

    # Create embeddings
    if not table_exists(target, 'embeddings'):
        cols_def = ', '.join([f'{c} TEXT' for c in embeddings_cols])
        cur.execute(f'CREATE TABLE embeddings ({cols_def})')
    else:
        existing = set(get_columns(target, 'embeddings'))
        for c in embeddings_cols:
            if c not in existing:
                cur.execute(f'ALTER TABLE embeddings ADD COLUMN {c} TEXT')

    target.commit()


def merge_dbs(input_files: List[str], output_path: str):
    analyses = []
    for f in input_files:
        analyses.append(analyze_db(f, verify_shards=False))

    # Union schemas
    all_seg_cols: List[str] = []
    all_emb_cols: List[str] = []
    seen = set()
    for a in analyses:
        for c in a.get('segments_cols', []):
            if c not in seen:
                all_seg_cols.append(c); seen.add(c)
    seen = set()
    for a in analyses:
        for c in a.get('embeddings_cols', []):
            if c not in seen:
                all_emb_cols.append(c); seen.add(c)

    # Create target
    if os.path.exists(output_path):
        os.remove(output_path)
    with sqlite3.connect(output_path) as tgt:
        ensure_target_schema(tgt, all_seg_cols, all_emb_cols)

        for f in input_files:
            with sqlite3.connect(f) as src:
                cur_s = src.cursor()
                cur_t = tgt.cursor()

                # Segments
                if table_exists(src, 'segments'):
                    src_cols = get_columns(src, 'segments')
                    # Select with explicit order
                    select_cols = ', '.join(src_cols)
                    cur_s.execute(f'SELECT {select_cols} FROM segments')
                    rows = cur_s.fetchall()
                    # Map to target column order
                    col_to_idx = {c: i for i, c in enumerate(src_cols)}
                    for row in rows:
                        values = [row[col_to_idx[c]] if c in col_to_idx else None for c in all_seg_cols]
                        placeholders = ','.join(['?'] * len(values))
                        cur_t.execute(f'INSERT INTO segments ({", ".join(all_seg_cols)}) VALUES ({placeholders})', values)

                # Embeddings
                if table_exists(src, 'embeddings'):
                    src_cols = get_columns(src, 'embeddings')
                    select_cols = ', '.join(src_cols)
                    cur_s.execute(f'SELECT {select_cols} FROM embeddings')
                    rows = cur_s.fetchall()
                    col_to_idx = {c: i for i, c in enumerate(src_cols)}
                    for row in rows:
                        values = [row[col_to_idx[c]] if c in col_to_idx else None for c in all_emb_cols]
                        placeholders = ','.join(['?'] * len(values))
                        cur_t.execute(f'INSERT INTO embeddings ({", ".join(all_emb_cols)}) VALUES ({placeholders})', values)

                tgt.commit()


def main():
    parser = argparse.ArgumentParser(description='Analyze and merge multiple SQLite3 embedding stores')
    parser.add_argument('--inputs', nargs='+', required=True, help='Input .sqlite3 files or directories containing them')
    parser.add_argument('--output', type=str, default='./sqlite3_tables/combined.sqlite3', help='Output combined sqlite3 path')
    parser.add_argument('--verify-shards', action='store_true', help='Verify shard counts by reading meta.json (Dropbox or local)')
    args = parser.parse_args()

    files = list_sqlite_files(args.inputs)
    if not files:
        print('❌ No .sqlite3 files found in provided inputs')
        return 1

    print('🔎 Found SQLite files:')
    for f in files:
        print(f'  - {f}')

    print('\n📊 Analyzing each database...')
    analyses = []
    for f in files:
        a = analyze_db(f, verify_shards=args.verify_shards)
        analyses.append(a)
        print(f'\n— {f}')
        print(f'  segments table: {"present" if a["has_segments"] else "missing"} | columns: {a["segments_cols"]}')
        print(f'  embeddings table: {"present" if a["has_embeddings"] else "missing"} | columns: {a["embeddings_cols"]}')
        print(f'  segments rows: {a["segments_count"]}')
        print(f'  embeddings rows: {a["embeddings_count"]}')
        if a['shard_counts']:
            print('  shard_path counts:')
            for sp, cnt in a['shard_counts'].items():
                extra = ''
                if args.verify_shards:
                    meta_cnt = a['shard_meta_counts'].get(sp, -1)
                    extra = f' | meta.count={meta_cnt}'
                    if sp in a['shard_meta_mismatch']:
                        extra += '  ⚠️ MISMATCH'
                print(f'    - {sp}: {cnt}{extra}')

    print('\n🧩 Merging into:', args.output)
    merge_dbs(files, args.output)
    print('✅ Merge complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


