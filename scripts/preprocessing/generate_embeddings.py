from retriever.embedders import AUDIO_EMBEDDERS, VIDEO_EMBEDDERS
from db.embedding_store_utils import get_segments_by_created_at, insert_embedding
from utils.config_loader import load_config
import numpy as np
import os
import uuid
from datetime import datetime
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array
import faiss
import dropbox

def embed_segments(segments):
    accumulator = {
        (e.model_name, e.mode): {"embeddings": [], "segment_ids": []}
        for e in AUDIO_EMBEDDERS + VIDEO_EMBEDDERS
    }

    for segment in segments:
        print("segment: ", segment)
        print("type: ", type(segment))
        segment_id = segment["segment_id"]
        filepath = segment["video_path"]
        start_time = segment["start_time"]
        duration = segment["duration"]

        try:
            video = VideoFileClip(filepath)
            print("Successfully loaded video: ", type(video))
            video = video.subclipped(start_time, start_time + duration)
            print("Successfully subclipped video: ", type(video))
            audio = video.audio
            sr = audio.fps
            audio_array = get_audio_array(audio, sr)

            for embedder in AUDIO_EMBEDDERS:
                try:
                    temp_audio = (
                        embedder.get_audio_noise(audio_array)
                        if embedder.mode == "audio noise"
                        else audio_array
                    )
                    emb = embedder.embed(temp_audio, sr)
                    key = (embedder.model_name, embedder.mode)
                    accumulator[key]["embeddings"].append(emb)
                    accumulator[key]["segment_ids"].append(segment_id)
                    print(f"✅ Audio embed success {segment_id} with {embedder.model_name}")
                except Exception as e:
                    print(f"❌ Audio embed fail {segment_id} with {embedder.model_name}: {e}")

            for embedder in VIDEO_EMBEDDERS:
                try:
                    temp_video = (
                        embedder.get_video_noise(video)
                        if embedder.mode == "video noise"
                        else video
                    )
                    emb = embedder.embed(temp_video)
                    key = (embedder.model_name, embedder.mode)
                    accumulator[key]["embeddings"].append(emb)
                    accumulator[key]["segment_ids"].append(segment_id)
                    print(f"✅ Video embed success {segment_id} with {embedder.model_name}")
                except Exception as e:
                    print(f"❌ Video embed fail {segment_id} with {embedder.model_name}: {e}")

        except Exception as e:
            print(f"❌ Segment load fail {segment_id}: {e}")

    return accumulator

def save_embeddings(accumulator, output_dir, created_at):
    os.makedirs(output_dir, exist_ok=True)

    for (model, mode), data in accumulator.items():
        if not data["embeddings"]:
            continue

        embs = np.stack(data["embeddings"])
        seg_ids = data["segment_ids"]

        base = f"{model}_{mode}_{created_at}"
        npy_path = os.path.join(output_dir, f"{base}.npy")
        csv_path = os.path.join(output_dir, f"{base}.csv")

        np.save(npy_path, embs)
        with open(csv_path, "w") as f:
            for sid in seg_ids:
                f.write(sid + "\n")

        print(f"✅ Saved: {npy_path} and {csv_path}")

def insert_embeddings_to_db(accumulator, db_path, created_at, output_dir):
    now = datetime.utcnow().isoformat()

    for (model, mode), data in accumulator.items():
        if not data["embeddings"]:
            continue

        base = f"{model}_{mode}_{created_at}"
        npy_path = os.path.join(output_dir, f"{base}.npy")

        for sid in data["segment_ids"]:
            embedding_dict = {
                "embedding_id": str(uuid.uuid4()),
                "segment_id": sid,
                "mode": mode,
                "model_name": model,
                "embedding_type": "raw",
                "reducer_id": None,
                "contraster_id": None,
                "embedding_path": npy_path,
                "created_at": now
            }
            insert_embedding(db_path, embedding_dict)

    print("✅ Inserted embeddings into DB.")

def create_faiss_index_and_upload(output_dir, dim=512, dropbox_path="/faiss_index/"):

    from utils.config_loader import load_config

    config = load_config()
    access_token = config["dropbox"]["access_token"]

    for fname in os.listdir(output_dir):
        if fname.endswith(".npy"):
            npy_path = os.path.join(output_dir, fname)
            embs = np.load(npy_path)

            index = faiss.IndexFlatL2(embs.shape[1] if dim is None else dim)
            index.add(embs)

            faiss_path = npy_path.replace(".npy", ".faiss")
            faiss.write_index(index, faiss_path)
            print(f"✅ FAISS index written: {faiss_path}")

            # Upload to Dropbox
            dbx = dropbox.Dropbox(access_token)
            with open(faiss_path, "rb") as f:
                dbx.files_upload(f.read(), dropbox_path + os.path.basename(faiss_path), mode=dropbox.files.WriteMode.overwrite)
                print(f"☁️ Uploaded to Dropbox: {dropbox_path + os.path.basename(faiss_path)}")


def main():
    config = load_config()
    db_path = config["database"]["embedding_db_path"]

    # Dates
    # # 2025-07-31T16:46:45.022260
    # # 2025-08-01T15:30:28.371559
    created_at = "2025-07-31T16:46:45.022260"

    print(f"Getting segments for {created_at}")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} segments")

    test_segment = segments[0]
    print(f"Embedding test segment {test_segment['segment_id']}")
    accumulator = embed_segments([test_segment])
    print(f"Embedding done")

    # Print keys and embedding shapes
    for (model, mode), data in accumulator.items():
        print(f"--- {model} | {mode} ---")
        print(f"Num embeddings: {len(data['embeddings'])}")
        if data["embeddings"]:
            print(f"Shape: {np.array(data['embeddings'][0]).shape}")
            print(f"Segment IDs: {data['segment_ids']}")
        else:
            print("⚠️ No embeddings produced.")
    # save_embeddings(accumulator, embedding_out_dir, created_at)
    
    # # Optional: insert to DB
    # insert_embeddings_to_db(
    #     accumulator, db_path, created_at, embedding_out_dir
    # )

    # # Optional: FAISS index + Dropbox
    # create_faiss_index_and_upload(embedding_out_dir)    


if __name__ == "__main__":
    main()
