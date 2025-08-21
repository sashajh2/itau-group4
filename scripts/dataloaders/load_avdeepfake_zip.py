import os
import argparse
import subprocess
import json
from typing import Optional
from utils.config_loader import load_config


def _ensure_hf_login(token: str) -> None:
    # Login to Hugging Face (skip if already logged in)
    subprocess.run(["huggingface-cli", "login", "--token", token], check=False)


def _download_zip_part(local_dir: str, part_str: str, token: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    os.environ["HF_TOKEN"] = token
    _ensure_hf_login(token)

    zip_rel_path = f"train/train.zip.{part_str}"
    subprocess.run([
        "huggingface-cli", "download", "ControlNet/AV-Deepfake1M-PlusPlus",
        zip_rel_path, "--repo-type", "dataset", "--local-dir", local_dir
    ], check=True)
    return os.path.join(local_dir, zip_rel_path)


def _extract_zip(zip_file: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["7z", "x", zip_file, f"-o{out_dir}"], check=True)
    return out_dir


def _is_video_valid_ffprobe(video_path: str) -> bool:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "default=noprint_wrappers=1", video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        # ffprobe not installed; fallback to best-effort True
        return True


def _scan_and_remove_corrupted_files(extracted_root: str) -> int:
    removed = 0
    for root, _, files in os.walk(extracted_root):
        for file in files:
            if not file.endswith(".mp4"):
                continue
            mp4_path = os.path.join(root, file)
            json_path = mp4_path.replace(".mp4", ".json")

            # JSON must exist and be valid
            json_ok = True
            if not os.path.exists(json_path):
                json_ok = False
            else:
                try:
                    with open(json_path, "r") as f:
                        json.load(f)
                except Exception:
                    json_ok = False

            video_ok = _is_video_valid_ffprobe(mp4_path)

            if not (json_ok and video_ok):
                try:
                    if os.path.exists(mp4_path):
                        os.remove(mp4_path)
                finally:
                    if os.path.exists(json_path):
                        try:
                            os.remove(json_path)
                        except Exception:
                            pass
                removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="Download and extract a specific AV-Deepfake1M zip part")
    parser.add_argument("--part", type=str, required=True, help="Three-digit part number, e.g., 002")
    parser.add_argument("--local-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Base local directory")
    parser.add_argument("--scan-delete-corrupted", action="store_true", help="Scan extracted videos and delete corrupted pairs")
    args = parser.parse_args()

    config = load_config()
    token = config["huggingface"]["token"]

    # Normalize part to 3 digits
    part_str = str(args.part).zfill(3)

    # Download
    zip_path = _download_zip_part(args.local_dir, part_str, token)

    # Extract to unique part dir
    part_out_dir = os.path.join(args.local_dir, "extracted", f"part_{part_str}")
    _extract_zip(zip_path, part_out_dir)

    # Optional: scan and delete corrupted
    if args.scan_delete_corrupted:
        lrs3_root = os.path.join(part_out_dir, "train", "lrs3")
        if os.path.exists(lrs3_root):
            removed = _scan_and_remove_corrupted_files(lrs3_root)
            print(f"🧹 Removed {removed} corrupted video/json pairs in {lrs3_root}")
        else:
            print(f"⚠️ lrs3 root not found at {lrs3_root}")

    print(f"✅ Downloaded and extracted part {part_str}")
    print(f"ZIP_PATH={zip_path}")
    print(f"EXTRACTED_PART_DIR={part_out_dir}")


if __name__ == "__main__":
    main()

