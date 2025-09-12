from collections import defaultdict
import os, json, uuid
import numpy as np
from datetime import datetime, timezone
from dropbox_utils.dropbox_utils import get_client
from db.embedding_store_utils import insert_many_embeddings  # implement batch insert

TARGET_SHARD_BYTES = 512 * 1024 * 1024  # 512 MB
DTYPE = "float32"

def approx_bytes(n_rows, dim, dtype):
    return n_rows * dim * (4 if dtype == "float32" else 2)

def partition_dir(root, source, model, mode, noise, denoiser_name, version):
    dn = "none" if denoiser_name in (None, "", "none") else denoiser_name
    return f"{root}/{source}/{mode}/{noise}/{model}/{dn}/v{version}/"

def atomic_upload(local_path, dropbox_path):
    dbx = get_client()
    tmp = dropbox_path + ".tmp"
    with open(local_path, "rb") as f:
        dbx.files_upload(f.read(), tmp, mode=dbx.files.WriteMode.overwrite)
    dbx.files_move_v2(tmp, dropbox_path, allow_shared_folder=True, autorename=False)

class ShardWriter:
    def __init__(self, dropbox_root, db_path, source, version, tmp_dir="/tmp/emb"):
        self.dropbox_root = dropbox_root
        self.db_path = db_path
        self.source = source
        self.version = version
        self.tmp_dir = tmp_dir
        self.buf = defaultdict(lambda: {"embs": [], "segs": [], "dim": None, "shard_idx": 0})

    def add(self, model, mode, noise, denoiser_name, segment_id, emb):
        key = (model, mode, noise, denoiser_name or None)
        st = self.buf[key]
        emb = np.asarray(emb, dtype=np.float32)
        if st["dim"] is None:
            st["dim"] = emb.shape[0]
        elif st["dim"] != emb.shape[0]:
            # skip / log mismatch
            return

        st["embs"].append(emb)
        st["segs"].append(segment_id)

        if approx_bytes(len(st["embs"]), st["dim"], DTYPE) >= TARGET_SHARD_BYTES:
            self._flush_key(key)

    def finalize(self):
        for key in list(self.buf.keys()):
            if self.buf[key]["embs"]:
                self._flush_key(key)

    def _flush_key(self, key):
        model, mode, noise, denoiser_name = key
        st = self.buf[key]
        embs = np.stack(st["embs"])
        segs = st["segs"]
        N, D = embs.shape

        # local write
        local_dir = os.path.join(self.tmp_dir, model, mode, noise, denoiser_name or "none", f"v{self.version}")
        os.makedirs(local_dir, exist_ok=True)
        shard_name = f"shard_{st['shard_idx']:03d}"
        npy_local = os.path.join(local_dir, shard_name + ".npy")
        meta_local = os.path.join(local_dir, shard_name + ".meta.json")
        np.save(npy_local, embs)
        with open(meta_local, "w") as f:
            json.dump({
                "model": model, "mode": mode, "noise": noise, "denoiser_name": denoiser_name,
                "version": self.version, "count": N, "dim": D, "dtype": DTYPE
            }, f, indent=2)

        # dropbox path
        d_dir = partition_dir(self.dropbox_root, self.source, model, mode, noise, denoiser_name, self.version)
        shard_dbx = d_dir + shard_name + ".npy"
        meta_dbx  = d_dir + shard_name + ".meta.json"

        # ensure folders exist (Dropbox will create on upload/move; optional pre-create)
        atomic_upload(npy_local,  shard_dbx)
        atomic_upload(meta_local, meta_dbx)

        # DB rows
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for i, sid in enumerate(segs):
            rows.append({
                "embedding_id": str(uuid.uuid4()),
                "segment_id": sid,
                "model_name": model,
                "mode": mode,
                "noise": noise,
                "denoiser_name": denoiser_name,
                "shard_path": shard_dbx,
                "row_index": i,
                "vector_dim": int(D),
                "dtype": DTYPE,
                "embedding_type": "raw",
                "reducer_id": None,
                "contraster_id": None,
                "version": str(self.version),
                "created_at": now
            })
        insert_many_embeddings(self.db_path, rows)

        # clear and advance
        st["embs"].clear(); st["segs"].clear(); st["shard_idx"] += 1
