from db.embedding_store_utils import get_segments_by_created_at
from utils.config_loader import load_config
from ..generators.embedding_generator import embed_segments
from ..generators.embedding_saver import (
    save_embeddings_to_files,
    insert_embeddings_to_db,
)
from ..storage.dropbox_storage import create_faiss_index_and_upload
import argparse
import os

def generate_for_created_at(created_at: str, output_dir: str = "./embeddings/generated") -> tuple[int, int]:
    """
    Generate embeddings for all segments with the given created_at.

    Returns (num_segments, num_uploaded_indices)
    """
    config = load_config()
    db_path = config["database"]["embedding_db_path"]

    print(f"Getting segments for {created_at}")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} segments")
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return 0, 0

    print("🔄 Generating embeddings for all segments...")
    accumulator = embed_segments(segments)
    print("✅ Embedding generation complete")

    print("\n📊 Embedding Generation Summary:")
    for (model, mode), data in accumulator.items():
        if data["embeddings"]:
            print(f"  {model} | {mode}: {len(data['embeddings'])} embeddings")
        else:
            print(f"  {model} | {mode}: ⚠️ No embeddings produced")

    print("\n💾 Saving embeddings to files...")
    saved_files = save_embeddings_to_files(accumulator, output_dir, created_at)
    print(f"✅ Saved {len(saved_files)} embedding files")

    print("\n🗄️ Inserting embeddings into database...")
    insert_embeddings_to_db(accumulator, db_path, created_at, output_dir)

    print("\n🔍 Creating FAISS indices and uploading to Dropbox...")
    uploaded_files = create_faiss_index_and_upload(output_dir)
    print(f"✅ Uploaded {len(uploaded_files)} FAISS indices to Dropbox")

    return len(segments), len(uploaded_files)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for segments with a specific created_at")
    parser.add_argument("--created-at", type=str, required=True, help="ISO8601 created_at partition to process")
    parser.add_argument("--output-dir", type=str, default="./embeddings/generated", help="Directory to save embeddings")
    args = parser.parse_args()

    num_segments, num_uploaded = generate_for_created_at(args.created_at, args.output_dir)
    if num_segments > 0:
        print("\n🎉 Complete! Summary:")
        print(f"  - Processed {num_segments} segments")
        print(f"  - Uploaded {num_uploaded} FAISS indices to Dropbox")

if __name__ == "__main__":
    main()
