import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, List, Tuple
import numpy as np

from utils.config_loader import load_config


_SPACE_TO_TABLE = {
    ("audio", "hubert", 768): "embeddings_audio_hubert",
    ("audio", "openl3", 512): "embeddings_audio_openl3",
    ("video", "senet", 2048): "embeddings_video_senet",
}


def _route_table(mode: str, model_name: str, vector_dim: int) -> str:
    mode = (mode or "").lower()
    model_name = (model_name or "").lower()
    try:
        vector_dim = int(vector_dim)
    except Exception:
        pass
    return _SPACE_TO_TABLE.get((mode, model_name, vector_dim), "")


def _to_vector_literal(vec: np.ndarray) -> str:
    if isinstance(vec, np.ndarray):
        vec_list = vec.tolist()
    else:
        vec_list = list(vec)
    return "[" + ",".join(str(float(x)) for x in vec_list) + "]"


class NeonEmbeddingWriter:
    """Batches embedding inserts into Neon Postgres with pgvector.

    Usage:
      writer = NeonEmbeddingWriter(version="2025-09-12")
      writer.add(model, mode, noise, denoiser, segment_id, vector)
      writer.flush_all()
    """

    def __init__(self, version: str, batch_size: int = 1000):
        cfg = load_config()
        dsn = cfg["database"]["postgres"]["neon_database_url"]
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False
        self.cur = self.conn.cursor()
        self.version = version
        self.batch_size = batch_size
        # buffers keyed by target table
        self.buffers: Dict[str, List[tuple]] = {
            "embeddings_audio_hubert": [],
            "embeddings_audio_openl3": [],
            "embeddings_video_senet": [],
        }

    def add(self, model_name: str, mode: str, noise: str, denoiser_name: str, segment_id: str, emb) -> None:
        table = _route_table(mode, model_name, len(emb))
        if not table:
            return
        vec_literal = _to_vector_literal(emb)
        rec = (
            # embedding_id is generated on PG side if needed; we use (segment_id, config) UNIQUE
            # but to keep parity with previous schema, generate client-side ID by composite key if desired.
            # For now, set to segment_id+table+version to remain unique-ish; conflicts will be ignored.
            f"{segment_id}:{table}:{self.version}",
            segment_id,
            (model_name or "").lower(),
            (mode or "").lower(),
            noise or "none",
            denoiser_name or "none",
            "float32",
            "raw",
            None,
            None,
            self.version,
            None,
            vec_literal,
        )
        self.buffers[table].append(rec)
        if len(self.buffers[table]) >= self.batch_size:
            self._flush(table)

    def _flush(self, table: str):
        buf = self.buffers.get(table) or []
        if not buf:
            return
        tpl = "(" + ",".join(["%s"] * 12) + ",%s::vector)"
        execute_values(
            self.cur,
            f"""
            INSERT INTO {table}(
              embedding_id, segment_id, model_name, mode, noise, denoiser_name,
              dtype, embedding_type, reducer_id, contraster_id, version, created_at, embedding
            ) VALUES %s
            ON CONFLICT (embedding_id) DO NOTHING
            """,
            buf,
            template=tpl,
            page_size=len(buf),
        )
        self.conn.commit()
        buf.clear()

    def flush_all(self):
        for table in list(self.buffers.keys()):
            self._flush(table)

    def close(self):
        try:
            self.flush_all()
        finally:
            try:
                self.cur.close()
            except Exception:
                pass
            try:
                self.conn.close()
            except Exception:
                pass


