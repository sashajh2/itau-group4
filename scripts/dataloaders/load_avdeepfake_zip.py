import os
import argparse
import subprocess
import json
import re
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


def _extract_zip(zip_file: str, out_dir: str) -> tuple[str, str]:
    """
    Extract zip using 7z and capture stdout/stderr.

    Returns (out_dir, combined_output)
    """
    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(
        ["7z", "x", zip_file, f"-o{out_dir}"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return out_dir, combined


# def _is_video_valid_ffprobe(video_path: str) -> bool:
#     try:
#         proc = subprocess.run(
#             [
#                 "ffprobe", "-v", "error", "-show_format", "-show_streams",
#                 "-of", "default=noprint_wrappers=1", video_path
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=False,
#         )
#         return proc.returncode == 0
#     except FileNotFoundError:
#         # ffprobe not installed; fallback to best-effort True
#         return True


# def _scan_and_remove_corrupted_files(extracted_root: str) -> int:
#     removed = 0
#     for root, _, files in os.walk(extracted_root):
#         for file in files:
#             if not file.endswith(".mp4"):
#                 continue
#             mp4_path = os.path.join(root, file)
#             json_path = mp4_path.replace(".mp4", ".json")

#             # JSON must exist and be valid
#             json_ok = True
#             if not os.path.exists(json_path):
#                 json_ok = False
#             else:
#                 try:
#                     with open(json_path, "r") as f:
#                         json.load(f)
#                 except Exception:
#                     json_ok = False

#             video_ok = _is_video_valid_ffprobe(mp4_path)

#             if not (json_ok and video_ok):
#                 try:
#                     if os.path.exists(mp4_path):
#                         os.remove(mp4_path)
#                 finally:
#                     if os.path.exists(json_path):
#                         try:
#                             os.remove(json_path)
#                         except Exception:
#                             pass
#                 removed += 1
#     return removed


def download_and_extract_part(part: str, local_dir: str = "./data/temp_video_extracted/AV1M", scan_delete_corrupted: bool = False) -> tuple[str, str, str]:
    """
    Download and extract a specific AV-Deepfake1M zip part.

    Returns (zip_path, extracted_part_dir)
    """
    config = load_config()
    token = config["huggingface"]["token"]

    part_str = str(part).zfill(3)
    zip_path = _download_zip_part(local_dir, part_str, token)
    part_out_dir = os.path.join(local_dir, "extracted", f"part_{part_str}")
    part_out_dir, extract_output = _extract_zip(zip_path, part_out_dir)

    # Save extraction log for inspection/parsing
    log_path = os.path.join(part_out_dir, "extraction_log.txt")
    try:
        with open(log_path, "w") as f:
            f.write(extract_output)
    except Exception:
        log_path = ""

    # Parse extraction output to find corrupted files and delete them with paired JSON
    # Looks for lines like: "ERROR: Data Error : train/.../file.mp4"
    corrupted_rel_paths: set[str] = set()
    for line in extract_output.splitlines():
        if "ERROR" in line and ".mp4" in line:
            m = re.search(r"ERROR: .*?:\s+(.+?\.mp4)\s*$", line)
            if m:
                corrupted_rel_paths.add(m.group(1).strip())

    removed_count = 0
    for rel_path in corrupted_rel_paths:
        abs_mp4 = os.path.join(part_out_dir, rel_path)
        abs_json = abs_mp4.replace(".mp4", ".json")
        try:
            if os.path.exists(abs_mp4):
                os.remove(abs_mp4)
                print(f"🧹 Removed corrupted video: {abs_mp4}")
        except Exception as e:
            print(f"⚠️ Failed to remove {abs_mp4}: {e}")
        try:
            if os.path.exists(abs_json):
                os.remove(abs_json)
                print(f"🧹 Removed paired JSON: {abs_json}")
        except Exception as e:
            print(f"⚠️ Failed to remove {abs_json}: {e}")
        removed_count += 1

    if removed_count > 0:
        print(f"✅ Cleaned {removed_count} corrupted entries reported by 7z")

    return zip_path, part_out_dir, log_path


def main():
    parser = argparse.ArgumentParser(description="Download and extract a specific AV-Deepfake1M zip part")
    parser.add_argument("--part", type=str, required=True, help="Three-digit part number, e.g., 002")
    parser.add_argument("--local-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Base local directory")
    parser.add_argument("--scan-delete-corrupted", action="store_true", help="Scan extracted videos and delete corrupted pairs")
    args = parser.parse_args()

    zip_path, part_out_dir, log_path = download_and_extract_part(
        part=args.part,
        local_dir=args.local_dir,
        scan_delete_corrupted=args.scan_delete_corrupted,
    )

    print(f"✅ Downloaded and extracted part {str(args.part).zfill(3)}")
    print(f"ZIP_PATH={zip_path}")
    print(f"EXTRACTED_PART_DIR={part_out_dir}")
    if log_path:
        print(f"EXTRACTION_LOG={log_path}")


if __name__ == "__main__":
    main()

