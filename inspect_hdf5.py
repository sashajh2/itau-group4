"""
Utility script to inspect HDF5 file structure and verify it's compatible with the model.
"""

import h5py
import numpy as np
import sys


def inspect_hdf5(hdf5_path: str):
    """Inspect HDF5 file structure and print summary"""

    print(f"📂 Inspecting: {hdf5_path}\n")
    
    try:
        with h5py.File(hdf5_path, "r") as f:
            # Check metadata
            print("=" * 60)
            print("METADATA")
            print("=" * 60)
            if "metadata" in f:
                meta = f["metadata"]
                print(f"Total videos: {meta.attrs.get('total_videos', 'N/A')}")
                print(f"Total segments: {meta.attrs.get('total_segments', 'N/A')}")
                
                if "video_ids" in meta:
                    video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                                 for vid in meta["video_ids"][:]]
                    print(f"Number of video IDs: {len(video_ids)}")
                    print(f"First 5 video IDs: {video_ids[:5]}")
                
                if "embedding_types" in meta:
                    emb_types = [t.decode() if isinstance(t, bytes) else t 
                               for t in meta["embedding_types"][:]]
                    print(f"Embedding types: {emb_types}")
            else:
                print("❌ No metadata group found!")
                return
            
            # Check videos
            print("\n" + "=" * 60)
            print("VIDEOS")
            print("=" * 60)
            if "videos" not in f:
                print("❌ No videos group found!")
                return
            
            videos_grp = f["videos"]
            video_keys = list(videos_grp.keys())
            print(f"Number of videos: {len(video_keys)}")
            
            if len(video_keys) == 0:
                print("❌ No videos found!")
                return
            
            # Inspect first video in detail
            first_video_key = video_keys[0]
            print(f"\n📹 Inspecting first video: {first_video_key}")
            vid_grp = videos_grp[first_video_key]
            
            print(f"  Attributes:")
            for attr_name in vid_grp.attrs.keys():
                print(f"    {attr_name}: {vid_grp.attrs[attr_name]}")
            
            # Check embeddings
            print(f"\n  Embeddings:")
            if "embeddings" in vid_grp:
                emb_grp = vid_grp["embeddings"]
                for emb_type in ["openl3", "hubert", "senet"]:
                    if emb_type in emb_grp:
                        emb_data = emb_grp[emb_type]
                        print(f"    {emb_type}: shape {emb_data.shape}, dtype {emb_data.dtype}")
                    else:
                        print(f"    {emb_type}: ❌ NOT FOUND")
            else:
                print("    ❌ No embeddings group found!")
            
            # Check labels
            print(f"\n  Labels:")
            if "labels" in vid_grp:
                lbl_grp = vid_grp["labels"]
                for lbl_type in ["audio", "video"]:
                    if lbl_type in lbl_grp:
                        lbl_data = lbl_grp[lbl_type]
                        print(f"    {lbl_type}: shape {lbl_data.shape}, dtype {lbl_data.dtype}")
                        if len(lbl_data.shape) == 2:
                            print(f"      Label distribution: {np.bincount(lbl_data[0].astype(int))}")
                    else:
                        print(f"    {lbl_type}: ❌ NOT FOUND")
            else:
                print("    ❌ No labels group found!")
            
            # Verify compatibility
            print("\n" + "=" * 60)
            print("COMPATIBILITY CHECK")
            print("=" * 60)
            
            issues = []
            
            # Check required embedding types
            if "embeddings" in vid_grp:
                emb_grp = vid_grp["embeddings"]
                required_audio = ["openl3", "hubert"]
                required_video = ["senet"]
                
                has_audio = any(et in emb_grp for et in required_audio)
                has_video = any(et in emb_grp for et in required_video)
                
                if not has_audio:
                    issues.append("Missing audio embeddings (need openl3 or hubert)")
                if not has_video:
                    issues.append("Missing video embeddings (need senet)")
                
                # Check shapes
                if "openl3" in emb_grp:
                    shape = emb_grp["openl3"].shape
                    if len(shape) != 3:
                        issues.append(f"openl3 should be 3D [num_augs, num_segs, emb_dim], got {len(shape)}D")
                    elif shape[2] != 512:
                        issues.append(f"openl3 embedding dim should be 512, got {shape[2]}")
                
                if "senet" in emb_grp:
                    shape = emb_grp["senet"].shape
                    if len(shape) != 3:
                        issues.append(f"senet should be 3D [num_augs, num_segs, emb_dim], got {len(shape)}D")
                    elif shape[2] != 2048:
                        issues.append(f"senet embedding dim should be 2048, got {shape[2]}")
            
            # Check labels
            if "labels" in vid_grp:
                lbl_grp = vid_grp["labels"]
                if "audio" not in lbl_grp:
                    issues.append("Missing audio labels")
                else:
                    shape = lbl_grp["audio"].shape
                    if len(shape) != 2:
                        issues.append(f"audio labels should be 2D [num_augs, num_segs], got {len(shape)}D")
            
            if issues:
                print("❌ Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ File structure looks compatible!")
            
            # Estimate dataset size
            print("\n" + "=" * 60)
            print("DATASET SIZE ESTIMATE")
            print("=" * 60)
            total_samples = 0
            for video_key in video_keys[:10]:  # Sample first 10
                vid_grp = videos_grp[video_key]
                if "num_augmentations" in vid_grp.attrs:
                    total_samples += vid_grp.attrs["num_augmentations"]
            
            if len(video_keys) > 10:
                avg_augs_per_video = total_samples / 10
                estimated_total = int(avg_augs_per_video * len(video_keys))
                print(f"Estimated total samples: ~{estimated_total:,} (based on first 10 videos)")
            else:
                print(f"Total samples: {total_samples:,}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {hdf5_path}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python inspect_hdf5.py <path_to_hdf5_file>")
    #     print("\nExample:")
    #     print("  python inspect_hdf5.py ./data/deepfake_embeddings.h5")
    #     sys.exit(1)
    
    # hdf5_path = sys.argv[1]
    hdf5_path = "/Users/jerrysheng/Desktop/Lab/deepfake_embeddings_2.h5"
    inspect_hdf5(hdf5_path)

