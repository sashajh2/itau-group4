import os
import argparse
import tarfile
from typing import Optional, Tuple
from pathlib import Path
from huggingface_hub import hf_hub_download
from utils.config_loader import load_config


def download_tar_part(part: int, local_dir: str = "./data/temp_video_extracted/ShareVeo3") -> str:
    """
    Download a specific ShareVeo3 tar part from Hugging Face.
    
    Args:
        part: Part number (1-50)
        local_dir: Local directory to save the tar file
        
    Returns:
        Path to the downloaded tar file
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Format filename as veo3_videos_1.tar, veo3_videos_2.tar, etc.
    filename = f"generated_videos_veo3_tar/veo3_videos_{part}.tar"
    
    try:
        print(f"Downloading {filename} from ShareVeo3 dataset...")
        tar_path = hf_hub_download(
            repo_id="WenhaoWang/ShareVeo3",
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir
        )
        print(f"✅ Successfully downloaded {filename}")
        return tar_path
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        raise


def extract_tar(tar_file: str, out_dir: str) -> Tuple[str, str]:
    """
    Extract tar file and capture stdout/stderr.
    
    Args:
        tar_file: Path to the tar file
        out_dir: Directory to extract to
        
    Returns:
        Tuple of (extracted_dir, extraction_log)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    extraction_log = []
    extraction_log.append(f"Extracting {tar_file} to {out_dir}")
    
    try:
        with tarfile.open(tar_file, 'r') as tar:
            # Get total number of members for progress tracking
            members = tar.getmembers()
            total_members = len(members)
            
            print(f"Extracting {total_members} files from {os.path.basename(tar_file)}")
            
            for i, member in enumerate(members):
                if i % 100 == 0:
                    print(f"Extracting file {i+1}/{total_members}")
                
                try:
                    tar.extract(member, out_dir)
                except Exception as e:
                    error_msg = f"Error extracting {member.name}: {e}"
                    extraction_log.append(error_msg)
            
            print(f"✅ Successfully extracted {tar_file}")
            extraction_log.append(f"Extraction completed successfully")
            
    except Exception as e:
        error_msg = f"Failed to extract {tar_file}: {e}"
        extraction_log.append(error_msg)
        raise
    
    return out_dir, "\n".join(extraction_log)


def download_and_extract_part(part: int, local_dir: str = "./data/temp_video_extracted/ShareVeo3") -> Tuple[str, str, str]:
    """
    Download and extract a specific ShareVeo3 tar part.
    
    Args:
        part: Part number (1-50)
        local_dir: Base local directory for downloads
        
    Returns:
        Tuple of (tar_path, extracted_part_dir, log_path)
    """
    # Download the tar file
    tar_path = download_tar_part(part, local_dir)
    
    # Create extraction directory
    part_out_dir = os.path.join(local_dir, "extracted", f"part_{part:02d}")
    
    # Extract the tar file
    part_out_dir, extract_output = extract_tar(tar_path, part_out_dir)
    
    # Save extraction log
    log_path = os.path.join(part_out_dir, "extraction_log.txt")
    try:
        with open(log_path, "w") as f:
            f.write(extract_output)
    except Exception as e:
        print(f"Could not save extraction log: {e}")
        log_path = ""
    
    return tar_path, part_out_dir, log_path


def cleanup_files(tar_path: str, extracted_dir: str) -> None:
    """
    Clean up downloaded tar file and extracted directory.
    
    Args:
        tar_path: Path to the tar file to delete
        extracted_dir: Path to the extracted directory to delete
    """
    # Remove tar file
    try:
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"🗑️ Deleted tar file: {tar_path}")
    except Exception as e:
        print(f"⚠️ Could not delete {tar_path}: {e}")
    
    # Remove extracted directory
    try:
        if os.path.exists(extracted_dir):
            import shutil
            shutil.rmtree(extracted_dir)
            print(f"🗑️ Deleted extracted directory: {extracted_dir}")
    except Exception as e:
        print(f"⚠️ Could not delete {extracted_dir}: {e}")


def main():
    """Main CLI function for testing the loader."""
    parser = argparse.ArgumentParser(description="Download and extract a specific ShareVeo3 tar part")
    parser.add_argument("--part", type=int, required=True, help="Part number (1-50)")
    parser.add_argument("--local-dir", type=str, default="./data/temp_video_extracted/ShareVeo3", help="Base local directory")
    parser.add_argument("--cleanup", action="store_true", help="Clean up files after extraction")
    # no log level needed
    
    args = parser.parse_args()
    
    # Validate part number
    if args.part < 1 or args.part > 50:
        print("Part number must be between 1 and 50")
        return 1
    
    try:
        # Download and extract
        tar_path, part_out_dir, log_path = download_and_extract_part(
            part=args.part,
            local_dir=args.local_dir
        )
        
        print(f"✅ Successfully processed part {args.part:02d}")
        print(f"TAR_PATH={tar_path}")
        print(f"EXTRACTED_PART_DIR={part_out_dir}")
        if log_path:
            print(f"EXTRACTION_LOG={log_path}")
        
        # Cleanup if requested
        if args.cleanup:
            cleanup_files(tar_path, part_out_dir)
            print("🧹 Cleanup completed")
        
        return 0
        
    except Exception as e:
        print(f"Failed to process part {args.part}: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
