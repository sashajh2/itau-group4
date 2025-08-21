import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone


def run(cmd: list[str]) -> None:
    cmd = [c for c in cmd if c != ""]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch process AVDeepfake parts 002-050")
    parser.add_argument("--start", type=int, default=2, help="Start part (inclusive), e.g., 2 for 002")
    parser.add_argument("--end", type=int, default=50, help="End part (inclusive), e.g., 50 for 050")
    parser.add_argument("--base-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Base local dir for downloads")
    parser.add_argument("--scan-delete-corrupted", action="store_true", help="Scan and delete corrupted files after extraction")
    args = parser.parse_args()

    for part in range(args.start, args.end + 1):
        part_str = f"{part:03d}"
        print(f"\n===== Processing part {part_str} =====")

        # Step 1: Download and extract this part
        run([
            "python3", "scripts/dataloaders/load_avdeepfake_zip.py",
            "--part", part_str,
            "--local-dir", args.base_dir,
            "--scan-delete-corrupted" if args.scan_delete_corrupted else ""
        ])

        # Where files are extracted
        extracted_part_dir = os.path.join(args.base_dir, "extracted", f"part_{part_str}")
        lrs3_root = os.path.join(extracted_part_dir, "train", "lrs3")

        # Step 2: Extract segments into DB with a shared created_at for this part
        created_at = datetime.now(timezone.utc).isoformat()
        run([
            "python3", "scripts/preprocessing/extract_segments.py",
            "--video-root", lrs3_root,
            "--created-at", created_at,
        ])

        # Step 3: Generate embeddings for this created_at; Dropbox appends if exists
        run([
            "python3", "scripts/preprocessing/generate_embeddings.py",
            "--created-at", created_at,
            "--output-dir", "./embeddings/generated",
        ])

        # Step 4: Clean up local zip and extracted directory for this part
        zip_file = os.path.join(args.base_dir, "train", f"train.zip.{part_str}")
        try:
            if os.path.exists(zip_file):
                os.remove(zip_file)
                print(f"🗑️ Deleted {zip_file}")
        except Exception as e:
            print(f"⚠️ Could not delete {zip_file}: {e}")

        try:
            if os.path.exists(extracted_part_dir):
                shutil.rmtree(extracted_part_dir)
                print(f"🗑️ Deleted extracted dir {extracted_part_dir}")
        except Exception as e:
            print(f"⚠️ Could not delete {extracted_part_dir}: {e}")


if __name__ == "__main__":
    main()