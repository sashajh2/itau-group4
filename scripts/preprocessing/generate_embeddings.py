from db.embedding_store_utils import get_segments_by_created_at
from utils.config_loader import load_config
from embedding_generator import embed_segments
from embedding_saver import save_embeddings_to_files, insert_embeddings_to_db
from dropbox_uploader import create_faiss_index_and_upload
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for segments with a specific created_at")
    parser.add_argument("--created-at", type=str, required=True, help="ISO8601 created_at partition to process")
    parser.add_argument("--output-dir", type=str, default="./embeddings/generated", help="Directory to save embeddings")
    args = parser.parse_args()

    config = load_config()
    db_path = config["database"]["embedding_db_path"]

    created_at = args.created_at
    output_dir = args.output_dir

    print(f"Getting segments for {created_at}")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} segments")
    
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return
    
    # Generate embeddings for all segments
    print("🔄 Generating embeddings for all segments...")
    accumulator = embed_segments(segments)
    print("✅ Embedding generation complete")
    
    # Print summary of generated embeddings
    print("\n📊 Embedding Generation Summary:")
    for (model, mode), data in accumulator.items():
        if data["embeddings"]:
            print(f"  {model} | {mode}: {len(data['embeddings'])} embeddings")
        else:
            print(f"  {model} | {mode}: ⚠️ No embeddings produced")
    
    # Save embeddings to files
    print("\n💾 Saving embeddings to files...")
    saved_files = save_embeddings_to_files(accumulator, output_dir, created_at)
    print(f"✅ Saved {len(saved_files)} embedding files")
    
    # Insert embeddings into database
    print("\n🗄️ Inserting embeddings into database...")
    insert_embeddings_to_db(accumulator, db_path, created_at, output_dir)
    
    # Create FAISS indices and upload to Dropbox
    print("\n🔍 Creating FAISS indices and uploading to Dropbox...")
    uploaded_files = create_faiss_index_and_upload(output_dir)
    print(f"✅ Uploaded {len(uploaded_files)} FAISS indices to Dropbox")
    
    # Print final summary
    print("\n🎉 Complete! Summary:")
    print(f"  - Processed {len(segments)} segments")
    print(f"  - Generated {len(saved_files)} embedding files")
    print(f"  - Uploaded {len(uploaded_files)} FAISS indices to Dropbox")
    print(f"  - Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
