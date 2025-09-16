from collections import defaultdict
import os, json, uuid
import numpy as np
from datetime import datetime, timezone
from dropbox_utils.dropbox_utils import get_client, upload_file
from .dropbox_storage import check_dropbox_file_exists
from db.embedding_store_utils import insert_many_embeddings  # implement batch insert

TARGET_SHARD_BYTES = 64 * 1024 * 1024  # 64 MB
DTYPE = "float32"

def approx_bytes(n_rows, dim, dtype):
    return n_rows * dim * (4 if dtype == "float32" else 2)

def partition_dir(root, source, model, mode, noise, denoiser_name, version):
    dn = "none" if denoiser_name in (None, "", "none") else denoiser_name
    return f"{root}/{source}/raw/{mode}/{noise}/{model}/{dn}/v{version}/"

class ShardWriter:
    def __init__(self, dropbox_root, db_path, source, version, run_id=None, tmp_dir="/tmp/emb", allow_overwrite=False):
        self.dropbox_root = dropbox_root
        self.db_path = db_path
        self.source = source
        self.version = version
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.tmp_dir = tmp_dir
        self.allow_overwrite = allow_overwrite
        self.buf = defaultdict(lambda: {"embs": [], "segs": [], "dim": None, "shard_idx": 0, "collision_checked": False})

    def add(self, model, mode, noise, denoiser_name, segment_id, emb):
        key = (model, mode, noise, denoiser_name or None)
        st = self.buf[key]
        emb = np.asarray(emb, dtype=np.float32)
        
        # No continuation logic; start fresh per run_id
        
        if st["dim"] is None:
            st["dim"] = emb.shape[0]
        elif st["dim"] != emb.shape[0]:
            # skip / log mismatch
            return

        st["embs"].append(emb)
        st["segs"].append(segment_id)

        if approx_bytes(len(st["embs"]), st["dim"], DTYPE) >= TARGET_SHARD_BYTES:
            self._flush_key(key)

    def _check_run_collision(self, model, mode, noise, denoiser_name, shard_idx):
        if shard_idx != 0 or self.allow_overwrite:
            return
        d_dir = partition_dir(self.dropbox_root, self.source, model, mode, noise, denoiser_name, self.version) + f"{self.run_id}/"
        shard_name = f"shard_{self.run_id}_{shard_idx:03d}.npy"
        shard_dbx = d_dir + shard_name
        try:
            if check_dropbox_file_exists(shard_dbx):
                raise RuntimeError(f"Run collision: {shard_dbx} already exists. Use a different run_id or enable allow_overwrite.")
        except Exception as e:
            # Propagate unexpected errors
            if not (isinstance(e, RuntimeError)):
                raise e
            else:
                raise e

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

        # local write (include run_id)
        local_dir = os.path.join(self.tmp_dir, model, mode, noise, denoiser_name or "none", f"v{self.version}", self.run_id)
        os.makedirs(local_dir, exist_ok=True)
        shard_name = f"shard_{self.run_id}_{st['shard_idx']:03d}"
        npy_local = os.path.join(local_dir, shard_name + ".npy")
        meta_local = os.path.join(local_dir, shard_name + ".meta.json")
        np.save(npy_local, embs)
        with open(meta_local, "w") as f:
            json.dump({
                "model": model, "mode": mode, "noise": noise, "denoiser_name": denoiser_name,
                "version": self.version, "count": N, "dim": D, "dtype": DTYPE
            }, f, indent=2)

        # dropbox path
        d_dir = partition_dir(self.dropbox_root, self.source, model, mode, noise, denoiser_name, self.version) + f"{self.run_id}/"
        shard_dbx = d_dir + shard_name + ".npy"
        meta_dbx  = d_dir + shard_name + ".meta.json"

        # collision check once per key before first upload
        if not st.get("collision_checked", False):
            self._check_run_collision(model, mode, noise, denoiser_name, st["shard_idx"])
            st["collision_checked"] = True

        # Upload files using the existing upload_file function with error handling
        try:
            upload_file(npy_local, shard_dbx, overwrite=True)
            upload_file(meta_local, meta_dbx, overwrite=True)
        except Exception as e:
            print(f"❌ Upload failed for shard {shard_name}: {e}")
            raise e

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
