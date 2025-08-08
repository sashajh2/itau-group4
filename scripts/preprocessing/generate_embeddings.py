from retriever.embedders import AUDIO_EMBEDDERS, VIDEO_EMBEDDERS, DENOISERS
from retriever.embedders.hubert_embedder import HubertEmbedder
from retriever.embedders.openl3_embedder import Openl3Embedder
from db.embedding_store_utils import get_segments_by_created_at, insert_embedding
from utils.config_loader import load_config
import numpy as np
import os
import uuid
from datetime import datetime, timezone
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array
import faiss
import dropbox

def embed_segments(segments):
    # Create accumulator with regular audio embedders
    accumulator = {
        (e.model_name, e.mode): {"embeddings": [], "segment_ids": []}
        for e in AUDIO_EMBEDDERS + VIDEO_EMBEDDERS
    }
    
    # Add entries for denoised/noise embedders that will be created dynamically
    for embedder in AUDIO_EMBEDDERS:
        if embedder.mode == "audio":
            for denoiser_name in DENOISERS.keys():
                # Add entries for both denoised and noise modes
                key_denoised = (f"{embedder.model_name}_{denoiser_name}", "audio_denoised")
                key_noise = (f"{embedder.model_name}_{denoiser_name}", "audio_noise")
                accumulator[key_denoised] = {"embeddings": [], "segment_ids": []}
                accumulator[key_noise] = {"embeddings": [], "segment_ids": []}

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

            # Process audio with denoisers to get noised and denoised versions
            denoised_audio = {}
            noise_audio = {}
            
            for denoiser_name, denoiser in DENOISERS.items():
                try:
                    # Use the new split_audio function for efficiency
                    if denoiser_name == "voicefixer":
                        denoised, noise = denoiser.split_audio(audio_array, sr=sr)
                    else:
                        denoised, noise = denoiser.split_audio(audio_array)
                    
                    denoised_audio[denoiser_name] = denoised
                    noise_audio[denoiser_name] = noise
                    print(f"✅ Denoising success with {denoiser_name}")
                except Exception as e:
                    print(f"❌ Denoising fail with {denoiser_name}: {e}")

            # Process regular audio embedders
            for embedder in AUDIO_EMBEDDERS:
                if embedder.mode == "audio":
                    try:
                        emb = embedder.embed(audio_array, sr)
                        key = (embedder.model_name, embedder.mode)
                        accumulator[key]["embeddings"].append(emb)
                        accumulator[key]["segment_ids"].append(segment_id)
                        print(f"✅ Audio embed success {segment_id} with {embedder.model_name}")
                    except Exception as e:
                        print(f"❌ Audio embed fail {segment_id} with {embedder.model_name}: {e}")

            # Process denoised audio embedders
            for denoiser_name in DENOISERS.keys():
                if denoiser_name in denoised_audio:
                    # Create embedders for denoised audio
                    hubert_denoised = HubertEmbedder(mode="audio_denoised")
                    openl3_denoised = Openl3Embedder(mode="audio_denoised")
                    
                    for embedder in [hubert_denoised, openl3_denoised]:
                        try:
                            emb = embedder.embed(denoised_audio[denoiser_name], sr)
                            key = (f"{embedder.model_name}_{denoiser_name}", embedder.mode)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                            print(f"✅ Denoised embed success {segment_id} with {embedder.model_name}_{denoiser_name}")
                        except Exception as e:
                            print(f"❌ Denoised embed fail {segment_id} with {embedder.model_name}_{denoiser_name}: {e}")

            # Process noise audio embedders
            for denoiser_name in DENOISERS.keys():
                if denoiser_name in noise_audio:
                    # Create embedders for noise audio
                    hubert_noise = HubertEmbedder(mode="audio_noise")
                    openl3_noise = Openl3Embedder(mode="audio_noise")
                    
                    for embedder in [hubert_noise, openl3_noise]:
                        try:
                            emb = embedder.embed(noise_audio[denoiser_name], sr)
                            key = (f"{embedder.model_name}_{denoiser_name}", embedder.mode)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                            print(f"✅ Noise embed success {segment_id} with {embedder.model_name}_{denoiser_name}")
                        except Exception as e:
                            print(f"❌ Noise embed fail {segment_id} with {embedder.model_name}_{denoiser_name}: {e}")

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

# def save_embeddings(accumulator, output_dir, created_at):
#     os.makedirs(output_dir, exist_ok=True)

#     for (model, mode), data in accumulator.items():
#         if not data["embeddings"]:
#             continue

#         embs = np.stack(data["embeddings"])
#         seg_ids = data["segment_ids"]

#         # Create appropriate filename based on model and mode
#         if mode == "audio_denoised":
#             # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
#             denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
#             base = f"{model}_{mode}_{denoiser_name}_{created_at}"
#         elif mode == "audio_noise":
#             # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
#             denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
#             base = f"{model}_{mode}_{denoiser_name}_{created_at}"
#         else:
#             base = f"{model}_{mode}_{created_at}"
            
#         npy_path = os.path.join(output_dir, f"{base}.npy")
#         csv_path = os.path.join(output_dir, f"{base}.csv")

#         np.save(npy_path, embs)
#         with open(csv_path, "w") as f:
#             for sid in seg_ids:
#                 f.write(sid + "\n")

#         print(f"✅ Saved: {npy_path} and {csv_path}")

# def insert_embeddings_to_db(accumulator, db_path, created_at, output_dir):
#     now = datetime.now(timezone.utc).isoformat()

#     for (model, mode), data in accumulator.items():
#         if not data["embeddings"]:
#             continue

#         # Create appropriate filename based on model and mode
#         if mode == "audio_denoised":
#             # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
#             denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
#             base = f"{model}_{mode}_{denoiser_name}_{created_at}"
#         elif mode == "audio_noise":
#             # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
#             denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
#             base = f"{model}_{mode}_{denoiser_name}_{created_at}"
#         else:
#             base = f"{model}_{mode}_{created_at}"
            
#         npy_path = os.path.join(output_dir, f"{base}.npy")

#         for sid in data["segment_ids"]:
#             embedding_dict = {
#                 "embedding_id": str(uuid.uuid4()),
#                 "segment_id": sid,
#                 "mode": mode,
#                 "model_name": model,
#                 "embedding_type": "raw",
#                 "reducer_id": None,
#                 "contraster_id": None,
#                 "embedding_path": npy_path,
#                 "created_at": now
#             }
#             insert_embedding(db_path, embedding_dict)

#     print("✅ Inserted embeddings into DB.")

# def create_faiss_index_and_upload(output_dir, dim=512, dropbox_path="/faiss_index/"):

#     from utils.config_loader import load_config

#     config = load_config()
#     access_token = config["dropbox"]["access_token"]

#     for fname in os.listdir(output_dir):
#         if fname.endswith(".npy"):
#             npy_path = os.path.join(output_dir, fname)
#             embs = np.load(npy_path)

#             index = faiss.IndexFlatL2(embs.shape[1] if dim is None else dim)
#             index.add(embs)

#             faiss_path = npy_path.replace(".npy", ".faiss")
#             faiss.write_index(index, faiss_path)
#             print(f"✅ FAISS index written: {faiss_path}")

#             # Upload to Dropbox
#             dbx = dropbox.Dropbox(access_token)
#             with open(faiss_path, "rb") as f:
#                 dbx.files_upload(f.read(), dropbox_path + os.path.basename(faiss_path), mode=dropbox.files.WriteMode.overwrite)
#                 print(f"☁️ Uploaded to Dropbox: {dropbox_path + os.path.basename(faiss_path)}")


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
